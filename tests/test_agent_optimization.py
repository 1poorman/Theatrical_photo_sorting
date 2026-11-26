"""Agent contracts, project continuity, persistent task integration and vision."""
import asyncio
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

from PIL import Image
from app.agent_services import PhotoTaskService
from app import organize_tasks as task_api
from core_modules.organize.agent_budget import AgentBudget
from core_modules.organize.agent_contracts import outcome, validate_arguments
from core_modules.organize.agent_runtime import AgentRuntime
from core_modules.organize.conversational_agent import ConversationalAgent, _json_text
from core_modules.organize.langchain_agent import CompatibleToolModel, build_photo_agent
from core_modules.organize.photo_agent_tools import export_review, prepare_review
from core_modules.organize.review_queue import ReviewQueue, load_scene_evidence
from core_modules.organize.task_queue import TaskQueue
from core_modules.organize.visual_evidence import FIELDS
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from test_core_upgrade_tasks import wait_for
from test_langchain_agent import ScriptedModel, action, final


class VisionClient:
    base_url = 'http://fake.invalid/v1'
    model = 'fake-vision'
    model_identity = 'fake-vision'
    max_retries = 0
    temperature = 0
    timeout = 60
    chat_template_kwargs = {}

    def __init__(self):
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        return {'content': json.dumps({field: ['visible fact'] for field in FIELDS}),
                'trace': {'model': self.model}}


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        self.image = self.photos / 'neutral.png'
        Image.new('RGB', (20, 30), 'red').save(self.image)
        self.agent_root = self.root / 'agent'
        self.started = threading.Event()
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    def agent(self, responses, **options):
        return build_photo_agent(str(self.agent_root), model=CompatibleToolModel(
            endpoints=[ScriptedModel(responses=responses)]), **options)

    def service(self, blocked=False):
        def factory(kind, params):
            handler = task_api.deferred_handler(kind, params)
            original = handler.process
            def process(item):
                if blocked:
                    self.started.set()
                    if not self.release.wait(5):
                        raise TimeoutError('test worker not released')
                return original(item)
            handler.process = process
            return handler
        queue = TaskQueue(self.root / 'tasks', {kind: lambda params, kind=kind: factory(kind, params)
                         for kind in ('knowledge', 'scene_reason')})
        self.addCleanup(queue.close, 10)
        self.addCleanup(self.release.set)
        manager_patch = patch.object(task_api, 'manager', return_value=queue)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        service = PhotoTaskService(self.agent_root, task_reader=queue.get, task_cancel=queue.cancel,
                                   task_resume=queue.resume, env_path=None)
        return service, queue

    def test_full_and_partial_business_failures_are_not_ok(self):
        self.assertEqual(outcome({'n_images': 2, 'n_failed': 2})['status'], 'failed')
        self.assertEqual(outcome({'total_input': 4, 'failed': 1})['status'], 'partial')
        agent = self.agent([action('failure_counts', {}), final('已完成整理')])
        agent.register_tool('failure_counts', lambda: {'total_input': 2, 'failed': 2}, 'failed items')
        result = agent.chat('整理照片')
        self.assertFalse(result['turn_tool_calls'][0]['ok'])
        self.assertIn('failed', result['answer'])
        self.assertNotIn('已完成整理', result['answer'])

    def test_unexecuted_completion_cannot_be_accepted(self):
        result = self.agent([final('已经完成整理并导出。')]).chat('整理照片')
        self.assertIn('尚无可核验', result['answer'])
        self.assertEqual(result['turn_tool_calls'], [])

    def test_cross_turn_write_is_reused_and_changed_source_invalidates_key(self):
        arguments = {'input_dir': str(self.photos), 'output_file': str(self.root / 'tags.jsonl')}
        command = action('scan_dir_and_tag', arguments)
        agent = self.agent([command, final(), command, final(), command, final()])
        first = agent.chat('生成清单', 'repeat')['turn_tool_calls'][0]
        second = agent.chat('再次生成', 'repeat')['turn_tool_calls'][0]
        self.assertTrue(second['reused'])
        self.assertEqual(first['task_id'], second['task_id'])
        Image.new('RGB', (21, 30), 'blue').save(self.image)
        third = agent.chat('源图更新后生成', 'repeat')['turn_tool_calls'][0]
        self.assertFalse(third['ok'])
        self.assertFalse(third.get('reused'))

    def test_runtime_duplicate_id_and_changed_request_are_protected(self):
        calls = []
        runtime = AgentRuntime(str(self.root / 'runtime'))
        runtime.register_tool('write', lambda value: calls.append(value) or {})
        runtime.run('write', {'value': 1}, task_id='stable')
        self.assertTrue(runtime.run('write', {'value': 1}, task_id='stable')['reused'])
        with self.assertRaises(ValueError):
            runtime.run('write', {'value': 2}, task_id='stable')
        self.assertEqual(calls, [1])
        runtime.run('write', {'value': 2}, task_id='other', idempotency_key='other-key')
        with self.assertRaises(ValueError):
            runtime.run('write', {'value': 2}, task_id='stable', idempotency_key='other-key')

    def test_concurrent_runtime_instances_share_atomic_idempotency(self):
        from concurrent.futures import ThreadPoolExecutor
        calls = []
        first = AgentRuntime(str(self.root / 'runtime'))
        second = AgentRuntime(str(self.root / 'runtime'))
        def write():
            calls.append('started')
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError('test tool not released')
            return {'output_file': str(self.root / 'verified.jsonl')}
        for runtime in (first, second):
            runtime.register_tool('write', write)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(first.run, 'write', idempotency_key='same-key')
            try:
                self.assertTrue(self.started.wait(3))
                reused = second.run('write', idempotency_key='same-key')
                self.assertTrue(reused['reused'])
                self.assertEqual(reused['state'], 'running')
            finally:
                self.release.set()
            completed = pending.result(5)
        self.assertEqual(reused['task_id'], completed['task_id'])
        self.assertEqual(calls, ['started'])
        self.assertEqual(first.get_task(completed['task_id'])['state'], 'review')

    def test_recovered_runtime_does_not_replay_interrupted_write(self):
        runtime = AgentRuntime(str(self.root / 'runtime'))
        runtime.register_tool('write', lambda: (_ for _ in ()).throw(SystemExit('simulated process exit')))
        with self.assertRaises(SystemExit):
            runtime.run('write', idempotency_key='interrupted')
        restarted = AgentRuntime(runtime.root)
        restarted.register_tool('write', Mock(side_effect=AssertionError('interrupted write replayed')))
        changed = restarted.recover_interrupted()
        self.assertEqual(len(changed), 1)
        self.assertEqual(restarted.recover_interrupted(), [])
        task = restarted.run('write', idempotency_key='interrupted')
        self.assertTrue(task['reused'])
        self.assertEqual(task['state'], 'interrupted')
        self.assertFalse(task['outcome']['ok'])

    def test_projects_survive_reload_without_old_output_or_lost_memory(self):
        agent = self.agent([final('项目 A'), final('项目 B')])
        first = agent.chat('A', 'projects', {'input_dir': str(self.photos), 'output_file': 'a.jsonl'})
        other = self.root / 'other'
        other.mkdir()
        second = agent.chat('B', 'projects', {'input_dir': str(other)})
        self.assertNotIn('output_file', second['memory'])
        resumed = self.agent([final('回到 A')]).chat('A', 'projects', {'input_dir': str(self.photos)})
        self.assertEqual(resumed['memory']['output_file'], 'a.jsonl')
        self.assertEqual(resumed['project_id'], first['project_id'])
        self.assertEqual(resumed['workflow']['memory'], resumed['memory'])

    def test_sensitive_nested_context_and_message_are_redacted(self):
        result = self.agent([final('hello')]).chat('api_key=synthetic-value', context={
            'settings': {'api_key': 'synthetic-nested', 'theme': 'dark'}})
        self.assertEqual(result['memory']['settings'], {'theme': 'dark'})
        self.assertNotIn(b'synthetic', Path(result['session_file']).read_bytes())

    def test_explicit_project_name_still_isolates_changed_input(self):
        agent = self.agent([final('A'), final('B'), final('A')])
        first = agent.chat('A', 'named', {'project_id': 'named', 'input_dir': str(self.photos), 'output_file': 'a.jsonl'})
        second = agent.chat('B', 'named', {'project_id': 'named', 'input_dir': str(self.root)})
        self.assertNotIn('output_file', second['memory'])
        self.assertNotEqual(first['project_id'], second['project_id'])
        third = agent.chat('A', 'named', {'project_id': 'named', 'input_dir': str(self.photos)})
        self.assertEqual(third['memory']['output_file'], 'a.jsonl')

    def test_legacy_schema_rejects_range_enum_and_extra_arguments(self):
        schema = {'type': 'object', 'properties': {'limit': {'type': 'integer', 'minimum': 1},
                  'status': {'type': 'string', 'enum': ['pending']}}, 'required': ['limit', 'status']}
        agent = ConversationalAgent(str(self.agent_root))
        spec = {'parameters': schema}
        for arguments in ({'limit': -1, 'status': 'pending'}, {'limit': 1, 'status': 'other'},
                          {'limit': 1, 'status': 'pending', 'extra': 1}):
            with self.assertRaises(ValueError):
                agent._validate_arguments(spec, arguments, {})
        self.assertEqual(validate_arguments(schema, {'limit': '2', 'status': 'pending'}, coerce=True)[0]['limit'], 2)

    def test_large_results_remain_valid_json_with_artifact_reference(self):
        rendered = _json_text({'rows': ['sample' * 3000], 'report_path': '/verified/report.json'})
        result = json.loads(rendered)
        self.assertEqual(result['report_path'], '/verified/report.json')
        self.assertTrue(result['truncated'])

    def test_usage_and_every_cascade_attempt_are_recorded(self):
        small = ScriptedModel(responses=[RuntimeError('down')])
        big = ScriptedModel(responses=[AIMessage(content=json.dumps({'action': 'final', 'content': 'hello'}),
            usage_metadata={'input_tokens': 12, 'output_tokens': 5, 'total_tokens': 17})])
        budget = AgentBudget(output_tokens=4096)
        response = CompatibleToolModel(endpoints=[small, big], budget=budget).invoke('hello')
        self.assertEqual(response.usage_metadata['total_tokens'], 17)
        self.assertEqual(budget.summary()['endpoint_calls'], 2)
        self.assertEqual(budget.summary()['reported_usage']['total_tokens'], 17)
        with self.assertRaisesRegex(RuntimeError, 'budget_exhausted'):
            budget.reserve()

    def test_invalid_json_usage_is_not_lost_during_cascade(self):
        small = ScriptedModel(responses=[AIMessage(content='invalid JSON', usage_metadata={
            'input_tokens': 10, 'output_tokens': 3, 'total_tokens': 13})])
        big = ScriptedModel(responses=[final()])
        budget = AgentBudget()
        CompatibleToolModel(endpoints=[small, big], budget=budget).invoke('hello')
        summary = budget.summary()
        self.assertEqual(summary['calls'][0]['status'], 'failed')
        self.assertEqual(summary['reported_usage']['total_tokens'], 13)

    def test_remaining_budget_reaches_actual_sdk_request(self):
        client = Mock()
        completions = client.chat.completions
        completions.with_raw_response.parse.return_value.parse.return_value = {
            'choices': [{'message': {'role': 'assistant', 'content': json.dumps({
                'action': 'final', 'content': 'hello'})}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        endpoint = ChatOpenAI(model='fake', api_key='synthetic', root_client=client, client=completions, timeout=60, max_retries=0)
        budget = AgentBudget(seconds=1, output_tokens=256)
        CompatibleToolModel(endpoints=[endpoint], budget=budget).invoke('hello')
        request = completions.with_raw_response.parse.call_args.kwargs
        self.assertEqual(request['response_format'], {'type': 'json_object'})
        self.assertGreater(request['timeout'], 0)
        self.assertLessEqual(request['timeout'], 1)
        self.assertEqual(request['max_completion_tokens'], 256)
        self.assertEqual(budget.summary()['reported_usage']['total_tokens'], 15)

    def test_task_submission_query_and_restart_preserve_knowledge_artifact(self):
        service, queue = self.service(blocked=True)
        info = self.root / 'plot.txt'
        info.write_text('第1幕：演员上场。', encoding='utf-8')
        output = self.root / 'knowledge'
        params = {'info_path': str(info), 'play_root': str(self.photos), 'output_dir': str(output)}
        agent = self.agent([action('build_play_knowledge', params), final('已完成建库')], services=service)
        result = agent.chat('构建知识库', 'knowledge', {'input_dir': str(self.photos)})
        self.assertTrue(self.started.wait(3))
        self.assertIn('尚未完成', result['answer'])
        task_id = result['turn_tool_calls'][0]['outcome']['business_task_id']
        self.release.set()
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        restarted = self.agent([action('get_task', {'task_id': task_id}), final('已完成建库')], services=service)
        resumed = restarted.chat('查询结果', 'knowledge')
        self.assertEqual(resumed['memory']['knowledge_dir'], str(output))
        self.assertEqual(resumed['workflow']['tasks'][task_id]['tool'], 'build_play_knowledge')
        self.assertTrue((output / 'knowledge_snapshot.json').exists())

    def test_foreign_task_id_is_not_executed(self):
        service, _ = self.service()
        result = self.agent([action('cancel_task', {'task_id': 'foreign'}), final()], services=service).chat('取消')
        call = result['turn_tool_calls'][0]
        self.assertFalse(call['ok'])
        self.assertNotIn('task_id', call)

    def test_agent_task_cancel_resume_and_repeat_submit_use_durable_state(self):
        service, queue = self.service(blocked=True)
        info = self.root / 'plot.txt'
        info.write_text('第1幕：人物上场。', encoding='utf-8')
        params = {'info_path': str(info), 'play_root': str(self.photos), 'output_dir': str(self.root / 'knowledge')}
        agent = self.agent([action('build_play_knowledge', params), final()], services=service)
        response = agent.chat('建库', 'control', {'input_dir': str(self.photos)})
        task_id = response['turn_tool_calls'][0]['outcome']['business_task_id']
        self.assertTrue(self.started.wait(3))
        cancelled = self.agent([action('cancel_task', {'task_id': task_id}), final()], services=service).chat('取消', 'control')
        self.assertEqual(cancelled['turn_tool_calls'][0]['result']['status'], 'cancel_requested')
        self.release.set()
        wait_for(queue, task_id, {'cancelled'})
        self.agent([action('resume_task', {'task_id': task_id}), final()], services=service).chat('恢复', 'control')
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        repeated = self.agent([action('build_play_knowledge', params), final()], services=service).chat('再次建库', 'control')
        call = repeated['turn_tool_calls'][0]
        self.assertTrue(call['reused'])
        self.assertEqual(call['outcome']['business_task_id'], task_id)
        self.assertEqual(call['outcome']['status'], 'succeeded')

    def test_chat_api_contracts_and_validation_match(self):
        import httpx
        from config import base
        from core_modules.organize import langchain_agent
        server = ModuleType('server')
        server.DEFAULT_OUTPUT_DIR = str(self.root / 'api')
        server.FACE_DATABASE_ROOT = str(self.root / 'face_database')
        server.face_recognition_model = None
        server.organize_models = {}
        server.KNOWN_FACE_THRESHOLD = base.KNOWN_FACE_THRESHOLD
        server.get_error = lambda code=-1, message='': {'code': code, 'message': message}
        server.initialize_models = lambda: True
        image_search = ModuleType('core_modules.image_search')
        for name in ('build_index', 'search_image', 'find_similar_image_groups_from_folder'):
            setattr(image_search, name, Mock(side_effect=AssertionError('unrelated image search invoked')))
        age_estimation = ModuleType('core_modules.face_recognition.age_estimation')
        age_estimation.FaceAgeEstimator = Mock(side_effect=AssertionError('unrelated age model invoked'))
        spec = importlib.util.spec_from_file_location('agent_api_test', ROOT / 'app' / 'main.py')
        main = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'server': server, 'agent_api_test': main,
                                     'core_modules.image_search': image_search,
                                     'core_modules.face_recognition.age_estimation': age_estimation}):
            spec.loader.exec_module(main)
            async def check_routes():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                             base_url='http://test') as client:
                    legacy = (await client.get('/api/organize/agent/tools')).json()['data']['tools']
                    advanced = (await client.get('/api/organize/agent/advanced/tools')).json()['data']['tools']
                    self.assertEqual(legacy, advanced)
                    output = self.root / 'api-tags.jsonl'
                    response = await client.post('/api/organize/agent/chat', data={
                        'message': '目录标注', 'use_llm': 'false', 'context_json': json.dumps({
                            'input_dir': str(self.photos), 'output_file': str(output)})})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(output.exists())
                    history = '/api/organize/agent/session/' + response.json()['data']['session_id'] + '/history'
                    page = (await client.get(history, params={'limit': 1})).json()['data']
                    self.assertEqual(page['items'][0]['role'], 'assistant')
                    previous = await client.get(history, params={'limit': 1, 'before': page['next_before']})
                    self.assertEqual(previous.json()['data']['items'][0]['role'], 'tool')
                    calls = (await client.get(history, params={'kind': 'tool_calls'})).json()['data']
                    self.assertEqual(calls['items'][0]['tool'], 'scan_dir_and_tag')
                    self.assertEqual((await client.get(history, params={'limit': 101})).status_code, 422)
                    self.assertEqual((await client.get(history, params={'kind': 'unknown'})).status_code, 400)
                    self.assertEqual((await client.get('/api/organize/agent/session/unknown/history')).status_code, 404)
                    model = CompatibleToolModel(endpoints=[ScriptedModel(responses=[
                        action('inspect_photos', {'input_dir': str(self.photos), 'limit': -1}), final()])])
                    with patch.object(langchain_agent, 'model_from_env', return_value=model):
                        invalid = await client.post('/api/organize/agent/advanced/chat', data={'message': '检查照片'})
                    self.assertEqual(invalid.status_code, 200, invalid.text)
                    self.assertFalse(invalid.json()['data']['turn_tool_calls'][0]['ok'])
                    rejected = await client.post('/api/organize/agent/advanced/chat', data={
                        'message': '检查', 'turn_timeout': 0})
                    self.assertEqual(rejected.status_code, 422)
            asyncio.run(check_routes())

    def test_fresh_output_guard_is_atomic_and_idempotency_precedes_it(self):
        queue = TaskQueue(self.root / 'plain_tasks', {'test': lambda params:
                          (lambda item: item, lambda rows: {})})
        self.addCleanup(queue.close, 10)
        target = self.root / 'output.json'
        task, _ = queue.submit('test', {}, [], 'stable', str(target), fresh_output=True)
        wait_for(queue, task['task_id'])
        target.write_text('keep')
        reused, created = queue.submit('test', {}, [], 'stable', str(target), fresh_output=True)
        self.assertFalse(created)
        self.assertEqual(reused['task_id'], task['task_id'])
        with self.assertRaises(ValueError):
            queue.submit('test', {}, [], 'different', str(target), fresh_output=True)
        self.assertEqual(target.read_text(), 'keep')

    def test_visual_observation_sends_real_pixels_and_uses_budgeted_cache(self):
        client = VisionClient()
        service = PhotoTaskService(self.agent_root, env_path=None, vision_client=client)
        first = service.observe(str(self.image))
        second = service.observe(str(self.image))
        self.assertEqual(len(client.messages), 1)
        self.assertTrue(client.messages[0][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(first['evidence_type'], 'visual_observation')
        self.assertTrue(second['trace']['cache_hit'])
        self.assertTrue(first['requires_human_review'])

    def test_export_requires_human_review_and_preserves_image_binding(self):
        evidence = self.root / 'candidate.jsonl'
        evidence.write_text(json.dumps({'image': 'neutral.png', 'label': '第一幕', 'scene_id': 'act-1'}))
        review = self.root / 'review'
        prepare_review(str(evidence), str(review))
        target = self.root / 'approved.jsonl'
        with self.assertRaises(ValueError):
            export_review(str(review), str(target))
        queue = ReviewQueue(str(review))
        queue.seed([{'image': str(self.image), 'label': '第一幕', 'scene_id': 'act-1'}])
        queue.decide('neutral.png', 'accept', reviewer='human-test')
        self.assertEqual(export_review(str(review), str(target))['n_approved'], 1)
        self.assertEqual(load_scene_evidence(str(target))['by_path'][str(self.image)], '第一幕')


if __name__ == '__main__':
    unittest.main()
