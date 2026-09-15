# -*- coding: utf-8 -*-
"""agent_runtime - 受限智能体编排：任务状态机、工具白名单与可审计日志。

安全边界：
  - 只能调用显式注册且在白名单内的工具，禁止任意 shell/网络/文件删除。
  - 每个任务有 ``task_id``、输入哈希、状态机（queued/running/review/approved/failed）。
  - 所有状态变更写入追加式审计日志，可回放。
"""
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

STATES = ('queued', 'running', 'review', 'approved', 'failed')


def _now():
    return datetime.now(timezone.utc).isoformat()


class ToolNotAllowed(ValueError):
    pass


class AgentRuntime:
    def __init__(self, root, allowed_tools=None):
        self.root = root
        self.allowed = set(allowed_tools or [])
        self.tools = {}
        self.audit_path = os.path.join(root, 'agent_audit.jsonl')
        self.tasks_path = os.path.join(root, 'agent_tasks.json')

    # ---------- 工具注册 ----------

    def register_tool(self, name, fn, allowed=True):
        if not callable(fn):
            raise TypeError(f'工具必须可调用: {name}')
        self.tools[name] = fn
        if allowed:
            self.allowed.add(name)
        return name

    # ---------- 审计与状态 ----------

    def _audit(self, task_id, event, **detail):
        os.makedirs(self.root, exist_ok=True)
        with open(self.audit_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': _now(), 'task_id': task_id, 'event': event,
                                **detail}, ensure_ascii=False) + '\n')

    def _load_tasks(self):
        if not os.path.exists(self.tasks_path):
            return {}
        with open(self.tasks_path, encoding='utf-8') as f:
            return json.load(f)

    def _save_tasks(self, tasks):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.tasks_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.tasks_path)

    def get_task(self, task_id):
        return self._load_tasks().get(task_id)

    def list_tasks(self, state=None):
        rows = list(self._load_tasks().values())
        if state:
            rows = [r for r in rows if r['state'] == state]
        return sorted(rows, key=lambda r: r.get('created_at', ''))

    # ---------- 执行 ----------

    def run(self, tool, params=None, task_id=None):
        params = params or {}
        if tool not in self.allowed or tool not in self.tools:
            raise ToolNotAllowed(f'工具未在白名单内: {tool}')
        task_id = task_id or uuid.uuid4().hex[:12]
        input_hash = 'sha256:' + hashlib.sha256(
            json.dumps({'tool': tool, 'params': params},
                       ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]
        tasks = self._load_tasks()
        task = {'task_id': task_id, 'tool': tool, 'params': params,
                'input_hash': input_hash, 'state': 'queued',
                'created_at': _now(), 'updated_at': _now(), 'result': None,
                'error': None}
        tasks[task_id] = task
        self._save_tasks(tasks)
        self._audit(task_id, 'queued', tool=tool, input_hash=input_hash)

        task['state'] = 'running'
        task['updated_at'] = _now()
        self._save_tasks(tasks)
        self._audit(task_id, 'running', tool=tool)
        try:
            result = self.tools[tool](**params)
        except Exception as e:  # 失败也必须收尾
            task['state'] = 'failed'
            task['error'] = f'{type(e).__name__}: {e}'
            task['updated_at'] = _now()
            self._save_tasks(tasks)
            self._audit(task_id, 'failed', error=task['error'])
            return task
        task['result'] = result
        auto = result.get('auto_approved') if isinstance(result, dict) else False
        task['state'] = 'approved' if auto else 'review'
        task['updated_at'] = _now()
        self._save_tasks(tasks)
        self._audit(task_id, task['state'], tool=tool)
        return task

    def resolve(self, task_id, approve=True, reviewer='human', note=''):
        tasks = self._load_tasks()
        if task_id not in tasks:
            raise KeyError(f'任务不存在: {task_id}')
        task = tasks[task_id]
        if task['state'] not in ('review',):
            raise ValueError(f'任务状态不可仲裁: {task["state"]}')
        task['state'] = 'approved' if approve else 'failed'
        task['reviewer'] = reviewer
        task['note'] = note
        task['updated_at'] = _now()
        self._save_tasks(tasks)
        self._audit(task_id, task['state'], reviewer=reviewer, note=note)
        return task

    def audit_log(self):
        rows = []
        if os.path.exists(self.audit_path):
            with open(self.audit_path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        return rows
