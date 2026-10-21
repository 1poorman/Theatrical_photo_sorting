"""Bounded, single-worker tasks with durable item checkpoints.

The worker lease is process-wide through flock; a second server using the same
directory must not recover or execute the first server's tasks. Handlers should
compute side-effect-free items and publish outputs atomically in finish().
"""
from contextlib import closing, contextmanager
import fcntl
from functools import wraps
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid


BUSINESS_LOCK = threading.RLock()


def serialized(operation):
    """Legacy sync routes use the same model execution boundary as tasks."""
    @wraps(operation)
    def execute(*args, **kwargs):
        with BUSINESS_LOCK:
            return operation(*args, **kwargs)
    return execute


class TaskConflict(ValueError):
    pass


class TaskQueueFull(ValueError):
    pass


class TaskUnavailable(RuntimeError):
    pass


class TaskStopped(Exception):
    """Cooperative cancellation or service shutdown at a stage boundary."""


class TaskVersionConflict(ValueError):
    """A resumed task must not mix execution versions."""


class TaskContext:
    def __init__(self, queue, task_id):
        self.queue = queue
        self.task_id = task_id

    def check(self):
        if not self.queue._boundary(self.task_id):
            raise TaskStopped()

    def results(self):
        return self.queue.results(self.task_id)

    def result_for(self, image, stage):
        """Read one durable dependency without decoding all prior results."""
        with closing(self.queue._connect()) as connection:
            row = connection.execute("SELECT result FROM items WHERE task_id=? AND status='succeeded' "
                                     "AND json_extract(input, '$.image')=? AND json_extract(input, '$.stage')=?",
                                     (self.task_id, image, stage)).fetchone()
        if row is None:
            raise ValueError('task_dependency_missing: ' + stage)
        return json.loads(row[0])

    def checkpoint(self, key):
        with closing(self.queue._connect()) as connection:
            row = connection.execute('SELECT payload FROM checkpoints WHERE task_id=? AND name=?',
                                     (self.task_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def advance(self, key, payload, items):
        """Commit a stage plan and its next items together, exactly once."""
        with self.queue._write() as connection:
            if connection.execute('SELECT 1 FROM checkpoints WHERE task_id=? AND name=?',
                                  (self.task_id, key)).fetchone():
                return False
            position = connection.execute('SELECT COALESCE(MAX(position)+1, 0) FROM items WHERE task_id=?',
                                          (self.task_id,)).fetchone()[0]
            connection.execute('INSERT INTO checkpoints VALUES (?, ?, ?)', (self.task_id, key, encode(payload)))
            connection.executemany('INSERT INTO items(task_id, position, input) VALUES (?, ?, ?)',
                                   [(self.task_id, position + index, encode(item)) for index, item in enumerate(items)])
        return True


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class TaskQueue:
    def __init__(self, root, handlers, capacity=8):
        if capacity < 1:
            raise ValueError('task capacity must be positive')
        self.root = Path(root)
        self.handlers = handlers
        self.capacity = capacity
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'tasks.sqlite3'
        self._lease = (self.root / '.worker.lock').open('a+b')
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lease.close()
            raise TaskUnavailable('后台任务目录已有工作进程，请使用单个服务工作进程。') from exc
        try:
            with closing(self._connect()) as connection:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.executescript('''
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY, kind TEXT NOT NULL, request TEXT NOT NULL,
                        request_hash TEXT NOT NULL, idempotency_key TEXT UNIQUE,
                        output TEXT, status TEXT NOT NULL, created REAL NOT NULL,
                        updated REAL NOT NULL, result TEXT, error TEXT);
                    CREATE TABLE IF NOT EXISTS items (
                        task_id TEXT NOT NULL REFERENCES tasks(id), position INTEGER NOT NULL,
                        input TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', result TEXT,
                        PRIMARY KEY(task_id, position));
                    CREATE INDEX IF NOT EXISTS task_state ON tasks(status, created);
                    CREATE INDEX IF NOT EXISTS item_state ON items(task_id, status, position);
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        task_id TEXT NOT NULL REFERENCES tasks(id), name TEXT NOT NULL,
                        payload TEXT NOT NULL, PRIMARY KEY(task_id, name));
                    CREATE TABLE IF NOT EXISTS task_outputs (
                        task_id TEXT NOT NULL REFERENCES tasks(id), path TEXT NOT NULL,
                        PRIMARY KEY(task_id, path));
                    INSERT OR IGNORE INTO task_outputs
                        SELECT id, output FROM tasks WHERE output IS NOT NULL;
                ''')
                connection.execute("UPDATE tasks SET status='interrupted', updated=? "
                                   "WHERE status IN ('running','cancel_requested')", (time.time(),))
            self._thread = threading.Thread(target=self._run, name='organize-tasks', daemon=True)
            self._thread.start()
        except BaseException:
            self._lease.close()
            raise

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    @contextmanager
    def _write(self):
        with closing(self._connect()) as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def submit(self, kind, params, items, idempotency_key=None, output=None):
        if self._stop.is_set() or not self._thread.is_alive():
            raise TaskUnavailable('后台任务执行器不可用。')
        if kind not in self.handlers:
            raise ValueError('不支持的任务类型')
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 200:
            raise ValueError('幂等键长度必须为 1–200')
        items = list(items)
        request = encode({'params': params, 'items': items})
        digest = hashlib.sha256((kind + '\n' + request).encode()).hexdigest()
        outputs = list(dict.fromkeys(str(Path(path).resolve()) for path in
                       (output if isinstance(output, (list, tuple)) else [output]) if path))
        with self._write() as connection:
            if idempotency_key:
                old = connection.execute('SELECT id, request_hash FROM tasks WHERE idempotency_key=?',
                                         (idempotency_key,)).fetchone()
                if old:
                    if old['request_hash'] != digest:
                        raise TaskConflict('同一幂等键对应不同输入，请使用新的幂等键。')
                    return self.get(old['id']), False
            self._check_capacity(connection)
            self._check_output(connection, outputs)
            task_id = uuid.uuid4().hex
            now = time.time()
            connection.execute('INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)',
                               (task_id, kind, request, digest, idempotency_key, outputs[0] if outputs else None, 'queued', now, now))
            connection.executemany('INSERT INTO task_outputs VALUES (?, ?)', [(task_id, path) for path in outputs])
            connection.executemany('INSERT INTO items(task_id, position, input) VALUES (?, ?, ?)',
                                   [(task_id, i, encode(item)) for i, item in enumerate(items)])
        self._wake.set()
        return self.get(task_id), True

    def _check_capacity(self, connection):
        count = connection.execute("SELECT COUNT(*) FROM tasks WHERE status IN "
                                   "('queued','running','cancel_requested')").fetchone()[0]
        if count >= self.capacity:
            raise TaskQueueFull('后台队列已满，请稍后提交。')

    @staticmethod
    def _check_output(connection, output, task_id=None):
        if not output:
            return
        destinations = [Path(path).resolve() for path in (output if isinstance(output, (list, tuple)) else [output])]
        for row in connection.execute("SELECT tasks.id, task_outputs.path FROM tasks JOIN task_outputs "
                "ON tasks.id=task_outputs.task_id WHERE status IN ('queued','running','cancel_requested','interrupted')"):
            other = Path(row['path']).resolve()
            if row['id'] != task_id and any(destination == other or destination in other.parents or other in destination.parents
                                           for destination in destinations):
                raise TaskConflict('该输出位置已有未完成任务。')

    def get(self, task_id):
        with closing(self._connect()) as connection:
            connection.execute('BEGIN')
            row = connection.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row is None:
                raise KeyError('任务不存在')
            counts = dict(connection.execute("SELECT status, COUNT(*) FROM items WHERE task_id=? "
                "AND (COALESCE(json_extract(input, '$.stage'), '') <> 'fingerprint' OR status='failed') GROUP BY status",
                (task_id,)).fetchall())
            preparing = connection.execute("SELECT COUNT(*) FROM items WHERE task_id=? AND status='pending' "
                                           "AND json_extract(input, '$.stage')='fingerprint'", (task_id,)).fetchone()[0]
            contract = connection.execute("SELECT payload FROM checkpoints WHERE task_id=? AND name='execution_contract'",
                                          (task_id,)).fetchone()
        total = sum(counts.values())
        return {'task_id': row['id'], 'kind': row['kind'], 'status': row['status'],
                'created_at': row['created'], 'updated_at': row['updated'],
                'total': total, 'succeeded': counts.get('succeeded', 0), 'failed': counts.get('failed', 0),
                'pending': counts.get('pending', 0),
                'preparing': preparing,
                'execution_contract': json.loads(contract[0]) if contract else None,
                'progress': round(100 * (total - counts.get('pending', 0)) / total, 1) if total else 0,
                'result': json.loads(row['result']) if row['result'] else None, 'error': row['error']}

    def list(self, limit=50):
        if not 1 <= limit <= 100:
            raise ValueError('limit 必须为 1–100')
        with closing(self._connect()) as connection:
            ids = [row[0] for row in connection.execute('SELECT id FROM tasks ORDER BY created DESC LIMIT ?', (limit,))]
        return [self.get(task_id) for task_id in ids]

    def cancel(self, task_id):
        with self._write() as connection:
            row = connection.execute('SELECT status FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row is None:
                raise KeyError('任务不存在')
            if row['status'] in ('queued', 'interrupted'):
                state = 'cancelled'
            elif row['status'] in ('running', 'cancel_requested'):
                state = 'cancel_requested'
            else:
                state = row['status']
            connection.execute('UPDATE tasks SET status=?, updated=? WHERE id=?', (state, time.time(), task_id))
        self._wake.set()
        return self.get(task_id)

    def resume(self, task_id):
        if self._stop.is_set() or not self._thread.is_alive():
            raise TaskUnavailable('后台任务执行器不可用。')
        with self._write() as connection:
            row = connection.execute('SELECT status, output FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row is None:
                raise KeyError('任务不存在')
            if row['status'] not in ('interrupted', 'cancelled', 'failed'):
                raise TaskConflict('只能恢复已中断、已取消或执行失败的任务。')
            self._check_capacity(connection)
            outputs = [entry[0] for entry in connection.execute('SELECT path FROM task_outputs WHERE task_id=?', (task_id,))]
            self._check_output(connection, outputs, task_id)
            connection.execute("UPDATE tasks SET status='queued', error=NULL, updated=? WHERE id=?", (time.time(), task_id))
        self._wake.set()
        return self.get(task_id)

    def results(self, task_id):
        with closing(self._connect()) as connection:
            return [json.loads(row[0]) for row in connection.execute(
                'SELECT result FROM items WHERE task_id=? AND result IS NOT NULL ORDER BY position', (task_id,))]

    def _boundary(self, task_id):
        with self._write() as connection:
            state = connection.execute('SELECT status FROM tasks WHERE id=?', (task_id,)).fetchone()[0]
            if state == 'cancel_requested' or self._stop.is_set():
                connection.execute('UPDATE tasks SET status=?, updated=? WHERE id=?',
                                   ('cancelled' if state == 'cancel_requested' else 'interrupted', time.time(), task_id))
                return False
        return True

    def item_page(self, task_id, limit=50, offset=0):
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError('limit 必须为 1–200，offset 不可为负')
        with closing(self._connect()) as connection:
            if not connection.execute('SELECT 1 FROM tasks WHERE id=?', (task_id,)).fetchone():
                raise KeyError('任务不存在')
            return [{'position': row['position'], 'input': json.loads(row['input']),
                     'status': row['status'], 'result': json.loads(row['result']) if row['result'] else None}
                    for row in connection.execute('SELECT * FROM items WHERE task_id=? ORDER BY position LIMIT ? OFFSET ?',
                                                  (task_id, limit, offset))]

    def _execute(self, task_id, kind, request):
        try:
            with BUSINESS_LOCK:
                if not self._boundary(task_id):
                    return
                handler = self.handlers[kind](request['params'])
                context = TaskContext(self, task_id)
                def verify_version():
                    contract = getattr(handler, 'model_contract', None)
                    if contract is None:
                        return
                    # JSON normalization gives tuples/lists the same durable form.
                    contract_value = contract()
                    if contract_value is None:
                        return
                    current = json.loads(encode(contract_value))
                    saved = context.checkpoint('execution_contract')
                    if saved is None:
                        if any(row.get('stage') != 'fingerprint' for row in context.results()):
                            raise TaskVersionConflict('execution_version_unbound; 请创建新任务')
                        context.advance('execution_contract', current, [])
                    elif saved != current:
                        raise TaskVersionConflict('execution_version_changed; 请创建新任务')
                verify_version()
                staged = hasattr(handler, 'advance')
                if staged:
                    processor, finish = handler.process, handler.finish
                    handler.context = context
                else:
                    processor, finish = handler
                while self._boundary(task_id):
                    with closing(self._connect()) as connection:
                        item = connection.execute("SELECT position, input FROM items WHERE task_id=? "
                            "AND status='pending' ORDER BY position LIMIT 1", (task_id,)).fetchone()
                    if item is None:
                        verify_version()
                        if staged and handler.advance():
                            continue
                        # finish must publish atomically and tolerate repetition
                        # if the process dies before the task's final commit.
                        verify_version()
                        result = finish(self.results(task_id))
                        verify_version()
                        with self._write() as connection:
                            current = connection.execute('SELECT status FROM tasks WHERE id=?', (task_id,)).fetchone()[0]
                            errors = connection.execute("SELECT COUNT(*) FROM items WHERE task_id=? AND status='failed'", (task_id,)).fetchone()[0]
                            status = ('cancelled' if current == 'cancel_requested' else
                                      ('completed_with_errors' if errors else 'completed'))
                            connection.execute('UPDATE tasks SET status=?, result=?, updated=? WHERE id=?',
                                               (status, encode(result), time.time(), task_id))
                        return
                    value = json.loads(item['input'])
                    verify_version()
                    try:
                        result = processor(value)
                        payload = encode(result)
                        status = 'succeeded'
                    except Exception as exc:
                        if isinstance(exc, TaskStopped):
                            raise
                        payload = encode({'image': value.get('image'), 'stage': value.get('stage'),
                            'label': 'unknown', 'scene_id': 'unknown',
                            'decision': 'error', 'confidence': None,
                            'review': {'status': 'pending', 'reviewer': None, 'updated_at': None},
                            'error': {'type': type(exc).__name__, 'message': str(exc)}})
                        status = 'failed'
                    verify_version()
                    with self._write() as connection:
                        connection.execute('UPDATE items SET status=?, result=? WHERE task_id=? AND position=?',
                                           (status, payload, task_id, item['position']))
                        connection.execute('UPDATE tasks SET updated=? WHERE id=?', (time.time(), task_id))
        except TaskStopped:
            return
        except Exception as exc:
            with self._write() as connection:
                connection.execute("UPDATE tasks SET status='failed', error=?, updated=? WHERE id=?",
                                   (f'{type(exc).__name__}: {exc}', time.time(), task_id))

    def _run(self):
        try:
            while not self._stop.is_set():
                self._wake.clear()
                with self._write() as connection:
                    task = connection.execute("SELECT id, kind, request FROM tasks WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                    if task:
                        connection.execute("UPDATE tasks SET status='running', updated=? WHERE id=?", (time.time(), task['id']))
                if task:
                    self._execute(task['id'], task['kind'], json.loads(task['request']))
                else:
                    self._wake.wait(0.5)
        finally:
            self._lease.close()

    def close(self, timeout=1):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)
