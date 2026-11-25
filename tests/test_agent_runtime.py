# -*- coding: utf-8 -*-
"""M16 离线验收：受限智能体运行时（状态机、工具白名单、审计日志）。"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from core_modules.organize.agent_runtime import AgentRuntime, ToolNotAllowed

TMP = os.path.join(ROOT, 'outputs', 'tmp_agent_runtime')
shutil.rmtree(TMP, ignore_errors=True)

rt = AgentRuntime(TMP, allowed_tools=[])
rt.register_tool('scene_reason', lambda image: {'labels': {'x': '第1幕克段'}})
rt.register_tool('boom', lambda: (_ for _ in ()).throw(RuntimeError('boom')))
rt.register_tool('auto', lambda: {'auto_approved': True, 'n': 1})

# 1. 需人工复核的任务
task = rt.run('scene_reason', {'image': 'a.jpg'})
assert task['state'] == 'review', task
assert task['input_hash'].startswith('sha256:')
print('1. 推理任务进入 review OK:', task['task_id'])

# 2. 自动确认任务
task2 = rt.run('auto')
assert task2['state'] == 'approved'
print('2. auto_approved 直接 approved OK')

# 3. 未注册/白名单外工具被拒绝
try:
    rt.run('shell', {'cmd': 'rm -rf /'})
except ToolNotAllowed:
    pass
else:
    raise AssertionError('白名单外工具未被拒绝')
print('3. 白名单外工具拒绝 OK')

# 4. 失败任务也收尾为 failed 并记录错误
failed = rt.run('boom')
assert failed['state'] == 'failed' and 'boom' in failed['error']
print('4. 异常任务 failed 收尾 OK')

# 5. 人工仲裁
resolved = rt.resolve(task['task_id'], approve=True, reviewer='张三')
assert resolved['state'] == 'approved'
try:
    rt.resolve(task['task_id'])   # 已终态不可重复仲裁
except ValueError:
    pass
else:
    raise AssertionError('终态任务不应可重复仲裁')
print('5. 人工仲裁与状态保护 OK')

# 6. 状态查询与审计可回放
assert len(rt.list_tasks('failed')) == 1
events = [e['event'] for e in rt.audit_log() if e['task_id'] == task['task_id']]
assert events == ['queued', 'running', 'review', 'approved'], events
assert all('at' in e and 'task_id' in e for e in rt.audit_log())
print('6. 状态查询与审计日志 OK:', events)

shutil.rmtree(TMP, ignore_errors=True)
print('\n✅ M16 智能体运行时测试通过')
