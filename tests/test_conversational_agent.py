# -*- coding: utf-8 -*-
"""连续对话智能体离线验收：持久记忆、自主工具调用和故障兜底。"""
import json
import os
import shutil
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from core_modules.organize.conversational_agent import ConversationalAgent


TMP = os.path.join(ROOT, 'outputs', 'tmp_conversational_agent')
shutil.rmtree(TMP, ignore_errors=True)


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        content = self.responses.pop(0)
        return {'content': content, 'trace': {'model': 'fake-small'}}


SCHEMA = {
    'type': 'object',
    'properties': {
        'knowledge_dir': {'type': 'string'},
        'question': {'type': 'string'},
    },
    'required': ['knowledge_dir'],
}


calls = []


def lookup(knowledge_dir, question=''):
    calls.append((knowledge_dir, question))
    return {'knowledge_dir': knowledge_dir, 'answer': '第1幕克段',
            'auto_approved': True}


# 1. 模型自主选择工具，工具结果回送模型后形成最终回答。
client = SequenceClient([
    json.dumps({'action': 'tool', 'tool': 'knowledge_lookup',
                'arguments': {'question': '有几幕'},
                'memory': {'play': '天下归心'}}, ensure_ascii=False),
    json.dumps({'action': 'final', 'content': '知识库显示第一幕为克段。'}, ensure_ascii=False),
])
agent = ConversationalAgent(TMP, client=client)
agent.register_tool('knowledge_lookup', lookup, '查询知识库', SCHEMA,
                    aliases=['知识库', '幕次'], side_effects='read')
result = agent.chat('查询幕次 knowledge_dir=/tmp/kb', session_id='session-a')
assert result['answer'] == '知识库显示第一幕为克段。', result
assert result['memory']['play'] == '天下归心'
assert result['memory']['knowledge_dir'] == '/tmp/kb'
assert calls == [('/tmp/kb', '有几幕')], calls
assert result['tool_calls'][0]['state'] == 'approved'
assert not result['fallback']
print('1. 模型自主工具调用与最终回答 OK')


# 2. 新建智能体实例模拟进程重启；同 session_id 仍加载记忆和历史。
client2 = SequenceClient([
    json.dumps({'action': 'final', 'content': '我记得剧目是天下归心。'}, ensure_ascii=False),
])
agent2 = ConversationalAgent(TMP, client=client2)
agent2.register_tool('knowledge_lookup', lookup, '查询知识库', SCHEMA,
                     aliases=['知识库', '幕次'], side_effects='read')
result2 = agent2.chat('还记得刚才的剧目吗？', session_id='session-a')
assert result2['memory']['play'] == '天下归心'
assert result2['message_count'] > result['message_count']
prompt = client2.messages[0][0]['content']
assert '天下归心' in prompt and '/tmp/kb' in prompt, prompt
stored = agent2.get_session('session-a')
assert len(stored['messages']) == result2['message_count']
print('2. 跨实例连续对话与持久记忆 OK')


# 3. 模型不可用：自然语言别名 + 已保存参数走确定性工具路由。
fallback = ConversationalAgent(TMP, client=None)
fallback.register_tool('knowledge_lookup', lookup, '查询知识库', SCHEMA,
                       aliases=['知识库', '幕次'], side_effects='read')
result3 = fallback.chat('请查询知识库幕次', session_id='session-a')
assert result3['fallback'] and '工具 knowledge_lookup 已执行' in result3['answer']
assert calls[-1][0] == '/tmp/kb'
print('3. 模型不可用时确定性路由兜底 OK')


# 4. 参数缺失时不猜测，返回明确的补充参数请求。
empty_agent = ConversationalAgent(TMP, client=None)
empty_agent.register_tool('knowledge_lookup', lookup, '查询知识库', SCHEMA,
                          aliases=['知识库'])
result4 = empty_agent.chat('查询知识库', session_id='missing-args')
assert 'knowledge_dir' in result4['answer'] and not result4['tool_calls']
print('4. 缺少工具参数时安全询问 OK')


# 5. 非法模型 JSON 时升级大模型；工具异常后模型可解释失败并收尾。
class BigClient(SequenceClient):
    pass


class CascadingInvalidClient(SequenceClient):
    def __init__(self, big):
        super().__init__(['not-json', 'not-json'])
        self.big = big
        self.small = type('Small', (), {'model': 'bad-small'})()
        self.cascade_enabled = True


big = BigClient([
    json.dumps({'action': 'tool', 'tool': 'boom', 'arguments': {}}, ensure_ascii=False),
    json.dumps({'action': 'final', 'content': '工具失败，已保留错误供人工检查。'}, ensure_ascii=False),
])
agent3 = ConversationalAgent(TMP, client=CascadingInvalidClient(big))
agent3.register_tool('boom', lambda: (_ for _ in ()).throw(RuntimeError('boom')),
                     '模拟故障工具')
result5 = agent3.chat('执行故障工具', session_id='failure-session')
assert 'failed' in result5['answer'] and 'boom' in result5['answer']
assert result5['tool_calls'][0]['state'] == 'failed'
assert 'boom' in result5['tool_calls'][0]['error']
assert result5['trace'][0]['cascade_reason'] == 'invalid_agent_json'
print('5. 非法 JSON 大模型兜底与工具失败收尾 OK')


# 6. 白名单外显式调用不会执行。
result6 = fallback.chat('/tool shell {"cmd":"rm -rf /"}', session_id='safe-session')
assert result6['fallback'] and '不在白名单' in result6['answer']
assert not any(c.get('tool') == 'shell' and c.get('ok') for c in result6['tool_calls'])
print('6. 白名单边界 OK')


# 7. 同一会话并发写入不会相互覆盖。
concurrent = ConversationalAgent(TMP, client=None)
barrier = threading.Barrier(3)
errors = []


def write_memory(index):
    try:
        barrier.wait()
        concurrent.chat(f'保存 value{index}=ok', session_id='concurrent-session')
    except Exception as exc:
        errors.append(exc)


threads = [threading.Thread(target=write_memory, args=(i,)) for i in range(2)]
for thread in threads:
    thread.start()
barrier.wait()
for thread in threads:
    thread.join()
concurrent_session = concurrent.get_session('concurrent-session')
assert not errors, errors
assert concurrent_session['memory']['value0'] == 'ok'
assert concurrent_session['memory']['value1'] == 'ok'
assert len([m for m in concurrent_session['messages'] if m['role'] == 'user']) == 2
print('7. 同一会话并发写入保护 OK')


# 8. 敏感字段不落长期记忆，参数按 schema 转为目标类型。
typed_calls = []


def typed_tool(limit=0, enabled=False):
    typed_calls.append((limit, enabled))
    return {'auto_approved': True, 'limit': limit, 'enabled': enabled}


typed = ConversationalAgent(TMP, client=None)
typed.register_tool(
    'typed', typed_tool, '类型转换测试',
    {'type': 'object', 'properties': {
        'limit': {'type': 'integer'}, 'enabled': {'type': 'boolean'}}},
)
result8 = typed.chat(
    '/tool typed {"limit":"3","enabled":"false"}',
    session_id='typed-session',
    context={'api_key': 'should-not-persist', 'password': 'hidden'},
)
assert typed_calls == [(3, False)], typed_calls
assert 'api_key' not in result8['memory'] and 'password' not in result8['memory']
print('8. 参数类型转换与敏感记忆过滤 OK')


# 9. 工具执行后模型故障时直接使用工具结果，不重复执行工具。
class FailsAfterTool(SequenceClient):
    def __init__(self):
        super().__init__([
            json.dumps({'action': 'tool', 'tool': 'once', 'arguments': {}},
                       ensure_ascii=False),
        ])

    def chat(self, messages, **kwargs):
        if self.responses:
            return super().chat(messages, **kwargs)
        raise RuntimeError('model disconnected')


once_calls = []
after_tool = ConversationalAgent(TMP, client=FailsAfterTool())
after_tool.register_tool(
    'once', lambda: once_calls.append('called') or {'value': 1}, '只应执行一次')
result9 = after_tool.chat('执行一次', session_id='after-tool-failure')
assert once_calls == ['called'], once_calls
assert result9['fallback'] and '工具 once 已执行' in result9['answer']
assert len(result9['tool_calls']) == 1
print('9. 工具后模型故障不重复执行 OK')


shutil.rmtree(TMP, ignore_errors=True)
print('\n✅ 连续对话智能体测试通过')
