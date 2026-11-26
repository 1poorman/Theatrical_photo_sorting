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
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from core_modules.organize.agent_contracts import outcome, redact
from core_modules.organize.agent_storage import encode, transaction

try:
    import fcntl
except ImportError:  # Windows 等无 fcntl 环境仍保留进程内锁
    fcntl = None

STATES = ('queued', 'running', 'waiting', 'review', 'approved', 'partial', 'failed', 'interrupted')
_TASK_LOCK = threading.RLock()


def _now():
    return datetime.now(timezone.utc).isoformat()


class ToolNotAllowed(ValueError):
    pass


class AgentRuntime:
    def __init__(self, root, allowed_tools=None):
        self.root = root
        self.allowed = set(allowed_tools or [])
        self.tools = {}
        self.contextual = set()
        self.audit_path = os.path.join(root, 'agent_audit.jsonl')
        self.tasks_path = os.path.join(root, 'agent_tasks.sqlite3')

    # ---------- 工具注册 ----------

    def register_tool(self, name, fn, allowed=True, contextual=False):
        if not callable(fn):
            raise TypeError(f'工具必须可调用: {name}')
        self.tools[name] = fn
        if contextual:
            self.contextual.add(name)
        if allowed:
            self.allowed.add(name)
        return name

    # ---------- 审计与状态 ----------

    @contextmanager
    def _file_lock(self):
        with _TASK_LOCK:
            os.makedirs(self.root, exist_ok=True)
            lock_path = os.path.join(self.root, 'agent_runtime.lock')
            with open(lock_path, 'a+', encoding='utf-8') as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _audit(self, task_id, event, **detail):
        with self._file_lock():
            with open(self.audit_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(redact({'at': _now(), 'task_id': task_id, 'event': event,
                                           **detail}), ensure_ascii=False) + '\n')

    def _initialize(self, connection):
        connection.execute('CREATE TABLE tasks (task_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE,'
                           ' state TEXT NOT NULL, data TEXT NOT NULL)')
        connection.execute('CREATE INDEX tasks_state ON tasks(state)')
        legacy = os.path.join(self.root, 'agent_tasks.json')
        if os.path.exists(legacy):
            with open(legacy, encoding='utf-8') as stream:
                for task in json.load(stream).values():
                    self._insert(connection, task)

    def _open(self, write=False):
        return transaction(self.tasks_path, 'tasks', self._initialize, write)

    @staticmethod
    def _insert(connection, task):
        connection.execute('INSERT INTO tasks VALUES (?,?,?,?)', (task['task_id'],
                           task.get('idempotency_key') or None, task['state'], encode(task)))

    @staticmethod
    def _update(connection, task):
        connection.execute('UPDATE tasks SET state=?,data=? WHERE task_id=?',
                           (task['state'], encode(task), task['task_id']))

    def _put_task(self, task):
        with self._open(write=True) as connection:
            self._update(connection, task)

    def get_task(self, task_id):
        with self._open() as connection:
            row = connection.execute('SELECT data FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def list_tasks(self, state=None):
        with self._open() as connection:
            query = 'SELECT data FROM tasks' + (' WHERE state=?' if state else '')
            rows = [json.loads(data) for data, in connection.execute(query, (state,) if state else ())]
        return sorted(rows, key=lambda r: r.get('created_at', ''))

    # ---------- 执行 ----------

    def run(self, tool, params=None, task_id=None, idempotency_key=None, context=None):
        params = params or {}
        if tool not in self.allowed or tool not in self.tools:
            raise ToolNotAllowed(f'工具未在白名单内: {tool}')
        task_id = task_id or uuid.uuid4().hex[:12]
        input_hash = 'sha256:' + hashlib.sha256(
            json.dumps({'tool': tool, 'params': params},
                       ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]
        task = {'task_id': task_id, 'tool': tool, 'params': params,
                'input_hash': input_hash, 'state': 'queued',
                'created_at': _now(), 'updated_at': _now(), 'result': None,
                'error': None}
        task['idempotency_key'] = idempotency_key
        with self._open(write=True) as connection:
            rows = connection.execute('SELECT data FROM tasks WHERE task_id=? OR idempotency_key=?',
                                      (task_id, idempotency_key or None)).fetchall()
            if len(rows) > 1:
                raise ValueError('任务 ID 与幂等键引用了不同的记录')
            if rows:
                existing = json.loads(rows[0][0])
                if existing['input_hash'] != input_hash:
                    raise ValueError('同一任务或幂等键对应不同参数')
                return dict(existing, reused=True)
            self._insert(connection, task)
        self._audit(task_id, 'queued', tool=tool, input_hash=input_hash)

        task['state'] = 'running'
        task['updated_at'] = _now()
        self._put_task(task)
        self._audit(task_id, 'running', tool=tool)
        try:
            arguments = dict(params)
            if tool in self.contextual:
                arguments['_execution_context'] = dict(context or {}, idempotency_key=idempotency_key)
            result = self.tools[tool](**arguments)
        except Exception as e:  # 失败也必须收尾
            task['state'] = 'failed'
            task['error'] = redact(f'{type(e).__name__}: {e}')
            task['outcome'] = outcome({'ok': False, 'error': task['error']})
            task['updated_at'] = _now()
            self._put_task(task)
            self._audit(task_id, 'failed', error=task['error'])
            return task
        task['result'] = redact(result)
        task['outcome'] = outcome(task['result'])
        auto = result.get('auto_approved') if isinstance(result, dict) else False
        execution = task['outcome']['status']
        task['state'] = {'submitted': 'waiting', 'failed': 'failed', 'partial': 'partial',
                         'interrupted': 'interrupted', 'cancelled': 'interrupted'}.get(
                             execution, 'approved' if auto else 'review')
        if execution == 'failed':
            task['error'] = task['outcome']['error'] or '业务处理失败，请查看失败计数与任务记录'
        task['updated_at'] = _now()
        self._put_task(task)
        self._audit(task_id, task['state'], tool=tool)
        return task

    def recover_interrupted(self):
        with self._open(write=True) as connection:
            rows = connection.execute("SELECT data FROM tasks WHERE state IN ('queued','running')").fetchall()
            changed = []
            for data, in rows:
                task = json.loads(data)
                task.update(state='interrupted', updated_at=_now(),
                            error='服务中断；请查询业务任务或核对产物后恢复')
                task['outcome'] = dict(outcome(task.get('result')), status='interrupted',
                                       ok=False, error=task['error'])
                self._update(connection, task)
                changed.append(task['task_id'])
        for task_id in changed:
            self._audit(task_id, 'interrupted')
        return changed

    def resolve(self, task_id, approve=True, reviewer='human', note=''):
        with self._open(write=True) as connection:
            row = connection.execute('SELECT data FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            if not row:
                raise KeyError(f'任务不存在: {task_id}')
            task = json.loads(row[0])
            if task['state'] not in ('review',):
                raise ValueError(f'任务状态不可仲裁: {task["state"]}')
            task['state'] = 'approved' if approve else 'failed'
            task['reviewer'] = reviewer
            task['note'] = note
            task['updated_at'] = _now()
            self._update(connection, task)
        self._audit(task_id, task['state'], reviewer=reviewer, note=note)
        return task

    def audit_log(self):
        rows = []
        with self._file_lock():
            if os.path.exists(self.audit_path):
                with open(self.audit_path, encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
        return rows
