"""离线运行真实 LangChain graph，仅替换模型传输层。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from core_modules.organize.langchain_agent import CompatibleToolModel, build_photo_agent
from core_modules.organize.photo_agent_tools import prepare_review, scene_reason


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return 'scripted-test'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def action(tool, arguments):
    return AIMessage(content=json.dumps({'action': 'tool', 'tool': tool, 'arguments': arguments}))


def final(content='完成'):
    return AIMessage(content=json.dumps({'action': 'final', 'content': content}, ensure_ascii=False))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        (self.photos / '20140304戏剧场-京剧《天下归心》彩排.jpg').touch()

    def agent(self, responses, **kwargs):
        endpoint = ScriptedModel(responses=responses)
        model = CompatibleToolModel(endpoints=[endpoint])
        return build_photo_agent(str(self.root / 'agent'), model=model, **kwargs), endpoint

    def test_multistep_persistent_memory(self):
        output = str(self.root / 'tags.jsonl')
        agent, endpoint = self.agent([
            action('inspect_photos', {'input_dir': str(self.photos)}),
            action('scan_dir_and_tag', {'input_dir': str(self.photos), 'output_file': output}),
            final('已扫描并生成待审核清单'),
        ])
        result = agent.chat('检查然后生成清单', 'persistent', {'play': '天下归心'})
        self.assertFalse(result['fallback'])
        self.assertEqual(len(result['turn_tool_calls']), 2)
        self.assertTrue(all(call['ok'] for call in result['turn_tool_calls']))
        self.assertEqual(json.loads(Path(output).read_text())['scene'], 'unknown')
        self.assertIn('[工具结果', str(endpoint.seen[-1]))
        resumed, endpoint = self.agent([final('记得天下归心')])
        response = resumed.chat('还记得剧目和输出吗？', 'persistent')
        self.assertEqual(response['memory']['output_file'], output)
        self.assertIn('天下归心', str(endpoint.seen[0]))
        self.assertTrue((self.root / 'agent/tasks/agent_audit.jsonl').exists())

    def test_json_cascade(self):
        small = ScriptedModel(responses=[AIMessage(content='not JSON')])
        big = ScriptedModel(responses=[final('大模型接管')])
        result = CompatibleToolModel(endpoints=[small, big]).invoke('你好')
        self.assertEqual(result.content, '大模型接管')
        self.assertTrue(result.response_metadata['cascade'])

    def test_model_errors_keep_details(self):
        endpoint = ScriptedModel(responses=[RuntimeError('connection refused')])
        model = CompatibleToolModel(endpoints=[endpoint])
        with self.assertRaisesRegex(RuntimeError, 'connection refused'):
            model.invoke('你好')

    def test_native_calls(self):
        endpoint = ScriptedModel(responses=[
            AIMessage(content='', tool_calls=[{'name': 'inspect_photos',
                'args': {'input_dir': str(self.photos)}, 'id': 'native-1', 'type': 'tool_call'}]),
            AIMessage(content='目录有一张图片'),
        ])
        agent = build_photo_agent(str(self.root / 'agent'), model=CompatibleToolModel(
            endpoints=[endpoint], protocol='native'))
        self.assertTrue(agent.chat('检查目录')['turn_tool_calls'][0]['ok'])

    def test_schema_repair_and_no_overwrite(self):
        output = self.root / 'existing.jsonl'
        output.write_text('keep me')
        agent, _ = self.agent([
            action('inspect_photos', {'input_dir': str(self.photos), 'limit': 'bad'}),
            action('inspect_photos', {'input_dir': str(self.photos), 'limit': 1}),
            action('scan_dir_and_tag', {'input_dir': str(self.photos), 'output_file': str(output)}),
            final('已检查，但未覆盖已有文件'),
        ])
        calls = agent.chat('检查')['turn_tool_calls']
        self.assertEqual([call['ok'] for call in calls], [False, True, False])
        self.assertEqual(output.read_text(), 'keep me')

    def test_missing_parameter_does_not_execute(self):
        agent, _ = self.agent([action('inspect_photos', {}), final('请提供图片目录')])
        call = agent.chat('检查')['turn_tool_calls'][0]
        self.assertFalse(call['ok'])
        self.assertNotIn('task_id', call)

    def test_failure_after_write_does_not_replay(self):
        agent, _ = self.agent([
            action('scan_dir_and_tag', {'input_dir': str(self.photos),
                                       'output_file': str(self.root / 'tags.jsonl')}),
            RuntimeError('endpoint down'),
        ])
        result = agent.chat('生成清单')
        self.assertTrue(result['fallback'])
        self.assertEqual(len(result['turn_tool_calls']), 1)
        self.assertIn('已执行', result['answer'])

    def test_unknown_tool_and_budget(self):
        agent, _ = self.agent([action('shell', {'command': 'echo forbidden'})])
        self.assertEqual(agent.chat('测试')['turn_tool_calls'], [])
        agent, _ = self.agent([
            action('inspect_photos', {'input_dir': str(self.photos)}),
            action('inspect_photos', {'input_dir': str(self.photos), 'limit': 2}),
        ], max_steps=1)
        result = agent.chat('检查')
        self.assertEqual(len(result['turn_tool_calls']), 1)

    def test_repeat_success_is_blocked(self):
        command = action('inspect_photos', {'input_dir': str(self.photos)})
        agent, _ = self.agent([command, command, final()])
        calls = agent.chat('检查')['turn_tool_calls']
        self.assertEqual([call['ok'] for call in calls], [True, False])

    def test_review_cannot_auto_approve(self):
        evidence = self.root / 'evidence.jsonl'
        evidence.write_text(json.dumps({'image': 'test.jpg', 'label': 'scene',
                                       'review': {'status': 'approved', 'reviewer': 'model'}}))
        target = self.root / 'review'
        prepare_review(str(evidence), str(target))
        record = json.loads((target / 'review_queue.jsonl').read_text())
        self.assertEqual(record['review']['status'], 'pending')

    def test_scene_reason_and_review_chain(self):
        knowledge = self.root / 'knowledge'
        knowledge.mkdir()
        (knowledge / 'entities.json').write_text(json.dumps({'scenes': [], 'roles': []}))
        output = self.root / 'evidence.jsonl'
        result = scene_reason(str(self.photos), str(knowledge), str(output))
        self.assertEqual(result['n_images'], 1)
        self.assertFalse(result['visual'])
        self.assertEqual(prepare_review(str(output), str(self.root / 'review'))['total'], 1)

    def test_context_and_session_validation(self):
        agent, _ = self.agent([final()])
        with self.assertRaises(ValueError):
            agent.chat('你好', '../escape')
        with self.assertRaises(ValueError):
            agent.chat('你好', context=['invalid'])
        result = agent.chat('你好', context={'api_key': 'do-not-store', 'play': 'test'})
        self.assertNotIn('api_key', result['memory'])


if __name__ == '__main__':
    unittest.main()
