"""Incremental agent stores, legacy imports and cursor stability (offline)."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_modules.organize.agent_runtime import AgentRuntime
from core_modules.organize.conversational_agent import ConversationalAgent


class AgentStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.agent = ConversationalAgent(self.root, prompt_history=4)
        self.history = self.agent.store.history('archive')
        self.history.legacy.parent.mkdir()

    def seed(self):
        self.artifact = self.root / 'report.json'
        self.artifact.write_text('{}')
        call = {'tool': 'write', 'project_id': 'default', 'ok': True, 'result': {},
                'outcome': {'artifacts': [{'kind': 'report_path', 'path': str(self.artifact)}]}}
        messages = [{'role': 'assistant', 'content': f'old {index}'} for index in range(1000)]
        self.history.legacy.write_text(json.dumps({'session_id': 'archive', 'memory': {},
            'created_at': 'old', 'updated_at': 'old', 'summary': 'remember', 'messages': messages,
            'projects': {'default': {'memory': {}, 'artifacts': {}, 'tasks': {'queued': {
                'tool': 'example', 'outcome': {'status': 'submitted'}, 'snapshot': {'task_id': 'queued'}}}}},
            'active_project_id': 'default', 'tool_calls': [call]}))

    def test_migrates_once_and_appends_only_new_rows(self):
        self.seed()
        original = self.history.legacy.read_bytes()
        self.assertEqual(len(self.agent.get_session('archive')['messages']), 1000)
        self.assertEqual(len(self.agent.store.load('archive', for_turn=True)['messages']), 44)
        with self.history.open(write=True) as connection:
            for table in ('messages', 'tool_calls', 'business_tasks'):
                for operation in ('UPDATE', 'DELETE', 'INSERT'):
                    condition = (' WHEN NEW.seq<=1000' if table == 'messages' else '') if operation == 'INSERT' else ''
                    connection.execute(f'CREATE TRIGGER protect_{table}_{operation} AFTER {operation} ON {table}'
                                       f"{condition} BEGIN SELECT RAISE(ABORT,'rewrote history'); END")
        result = self.agent.chat('帮助', 'archive')
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(result['message_count'], 1002)
        self.assertEqual(self.history.legacy.read_bytes(), original)
        self.assertEqual(self.history.page('messages', None, 1)['tasks']['queued']['tool'], 'example')
        self.assertEqual(len(self.agent.get_session('archive')['tool_calls']), 1)
        self.assertIn('old 997', self.agent.get_session('archive')['summary'])

    def test_stable_cursor_and_verified_write_beyond_prompt(self):
        self.seed()
        latest = self.history.page('messages', None, 2)
        self.agent.chat('帮助', 'archive')
        older = self.history.page('messages', latest['next_before'], 2)
        self.assertEqual([row['content'] for row in older['items'] + latest['items']],
                         [f'old {index}' for index in range(996, 1000)])
        self.assertIsNone(self.history.page('tool_calls', None, 1)['next_before'])
        self.agent.register_tool('write', lambda: {}, '写入')
        session = self.agent.store.load('archive', for_turn=True)
        session['projects']['default']['tasks'].clear()
        self.assertIn('历史已核验', self.agent._final_answer(session, '已经完成', 0))
        self.artifact.unlink()
        self.assertIn('尚无可核验', self.agent._final_answer(session, '已经完成', 0))

    def test_invalid_import_rolls_back_without_losing_source(self):
        self.history.legacy.write_text('{')
        with self.assertRaises(ValueError):
            self.history.page('messages', None, 10)
        self.assertEqual(self.history.legacy.read_text(), '{')
        self.seed()
        self.assertEqual(self.history.page('messages', None, 1)['items'][0]['content'], 'old 999')

    def test_failed_checkpoint_can_retry_without_partial_or_duplicate_history(self):
        session = self.agent.store.load('archive', for_turn=True)
        self.agent._merge_memory(session, {'project_id': 'test_token_project', 'settings': {'api_key': 'hidden'}})
        self.agent.store.append(session, 'user', 'request')
        session['tool_calls'].append({'tool': 'read'})
        with self.history.open(write=True) as connection:
            connection.execute("CREATE TRIGGER fail_call BEFORE INSERT ON tool_calls BEGIN SELECT RAISE(ABORT,'stop'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.agent.store.save(session)
        self.assertEqual(self.agent.get_session('archive')['messages'], [])
        with self.history.open(write=True) as connection:
            connection.execute('DROP TRIGGER fail_call')
        self.agent.store.save(session)
        self.agent.store.save(session)
        stored = self.agent.get_session('archive')
        self.assertEqual(len(stored['messages']), 1)
        self.assertEqual(len(stored['tool_calls']), 1)
        self.assertEqual(stored['projects']['test_token_project']['memory']['settings'], {})

    def test_runtime_import_reuses_old_key_without_updating_unrelated_rows(self):
        runtime = AgentRuntime(self.root)
        runtime.register_tool('count', lambda: {'auto_approved': True})
        fresh = runtime.run('count', task_id='existing', idempotency_key='original')
        other = self.root / 'migrated'
        other.mkdir()
        legacy = other / 'agent_tasks.json'
        legacy.write_text(json.dumps({'existing': fresh}))
        original = legacy.read_bytes()
        restarted = AgentRuntime(other)
        restarted.register_tool('count', lambda: {'auto_approved': True})
        self.assertTrue(restarted.run('count', idempotency_key='original')['reused'])
        with restarted._open(write=True) as connection:
            connection.execute("CREATE TRIGGER keep_old BEFORE UPDATE ON tasks WHEN OLD.task_id='existing'"
                               " BEGIN SELECT RAISE(ABORT,'rewrote old task'); END")
        restarted.run('count', task_id='new')
        self.assertEqual(restarted.recover_interrupted(), [])
        self.assertEqual(len(restarted.list_tasks('approved')), 2)
        self.assertEqual(legacy.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
