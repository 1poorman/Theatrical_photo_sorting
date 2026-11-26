"""Transactional SQLite stores for agent history and execution state."""
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from core_modules.organize.agent_contracts import redact


def encode(value):
    return json.dumps(redact(value), ensure_ascii=False, default=str)


@contextmanager
def transaction(path, table, initialize, write=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as connection:
        def initialized():
            return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                      (table,)).fetchone()
        if not initialized():
            with connection:
                connection.execute('BEGIN IMMEDIATE')
                if not initialized():
                    initialize(connection)
        with connection:
            connection.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield connection


class TurnSession(dict):
    """Working window; checkpoint offsets advance only after commit."""

    def __init__(self, metadata, messages, message_count):
        super().__init__(metadata, messages=messages, tool_calls=[])
        self.older_messages = message_count - len(messages)
        self.saved_messages = len(messages)
        self.saved_calls = 0
        self.dirty_tasks = set()


class SessionHistory:
    def __init__(self, root, session_id):
        self.path = Path(root) / (session_id + '.sqlite3')
        self.legacy = Path(root) / (session_id + '.json')
        self.session_id = session_id

    @staticmethod
    def _append(connection, messages, calls):
        connection.executemany('INSERT INTO messages(data) VALUES (?)',
                               ((encode(row),) for row in messages))
        connection.executemany('INSERT INTO tool_calls(project_id,has_artifacts,data) VALUES (?,?,?)',
            ((row.get('project_id'), bool(row.get('ok') and (row.get('outcome') or {}).get('artifacts')),
              encode(row)) for row in calls))

    @staticmethod
    def _metadata(connection, session):
        metadata = redact({key: value for key, value in session.items() if key not in ('messages', 'tool_calls', 'projects')})
        if 'projects' in session:
            metadata['projects'] = {key: redact(dict(project, tasks={})) for key, project in session['projects'].items()}
        connection.execute('INSERT OR REPLACE INTO metadata VALUES (1,?)',
                           (json.dumps(metadata, ensure_ascii=False, default=str),))

    @staticmethod
    def _tasks(connection, session, changed):
        connection.executemany('INSERT OR REPLACE INTO business_tasks VALUES (?,?,?)',
            ((project, task, encode(session['projects'][project]['tasks'][task])) for project, task in changed))

    def _initialize(self, connection):
        connection.execute('CREATE TABLE metadata (id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
        connection.execute('CREATE TABLE messages (seq INTEGER PRIMARY KEY, data TEXT NOT NULL)')
        connection.execute('CREATE TABLE tool_calls (seq INTEGER PRIMARY KEY, project_id TEXT,'
                           ' has_artifacts INTEGER NOT NULL, data TEXT NOT NULL)')
        connection.execute('CREATE INDEX calls_artifacts ON tool_calls(project_id, seq) WHERE has_artifacts=1')
        connection.execute('CREATE TABLE business_tasks (project TEXT, task TEXT, data TEXT NOT NULL,'
                           ' PRIMARY KEY(project,task))')
        if self.legacy.exists():
            with self.legacy.open(encoding='utf-8') as stream:
                session = json.load(stream)
            if session['session_id'] != self.session_id:
                raise ValueError('历史会话 ID 与文件名不一致')
            self._append(connection, session['messages'], session['tool_calls'])
            self._metadata(connection, session)
            self._tasks(connection, session, ((key, task) for key, project in session.get('projects', {}).items()
                                              for task in project.get('tasks', {})))

    def open(self, write=False):
        return transaction(self.path, 'metadata', self._initialize, write)

    @staticmethod
    def _page(connection, kind, before, limit):
        if kind not in ('messages', 'tool_calls'):
            raise ValueError('kind 必须为 messages 或 tool_calls')
        where, params = ('WHERE seq < ?', (before,)) if before is not None else ('', ())
        rows = connection.execute(f'SELECT seq,data FROM {kind} {where} ORDER BY seq DESC LIMIT ?',
                                  (*params, limit + 1)).fetchall()
        page = rows[:limit][::-1]
        return {'items': [json.loads(row[1]) for row in page],
                'next_before': page[0][0] if len(rows) > limit else None}

    def read(self, default, window=None):
        if window is None and not self.path.exists() and not self.legacy.exists():
            return dict(default, messages=[], tool_calls=[])
        with self.open() as connection:
            row = connection.execute('SELECT data FROM metadata WHERE id=1').fetchone()
            metadata = json.loads(row[0]) if row else default
            for project, task, data in connection.execute('SELECT project,task,data FROM business_tasks'):
                metadata['projects'][project]['tasks'][task] = json.loads(data)
            if window is None:
                return dict(metadata, **{kind: [json.loads(item[0]) for item in connection.execute(
                    f'SELECT data FROM {kind} ORDER BY seq')] for kind in ('messages', 'tool_calls')})
            messages = self._page(connection, 'messages', None, window)['items']
            count = connection.execute('SELECT COALESCE(MAX(seq),0) FROM messages').fetchone()[0]
            return TurnSession(metadata, messages, count)

    def checkpoint(self, session):
        with self.open(write=True) as connection:
            self._append(connection, session['messages'][session.saved_messages:],
                         session['tool_calls'][session.saved_calls:])
            self._metadata(connection, session)
            self._tasks(connection, session, session.dirty_tasks)
        session.saved_messages = len(session['messages'])
        session.saved_calls = len(session['tool_calls'])
        session.dirty_tasks.clear()
        return str(self.path)

    def page(self, kind, before, limit):
        if not 1 <= limit <= 100 or (before is not None and not 1 <= before <= 2**63 - 1):
            raise ValueError('无效的历史分页范围')
        if not self.path.exists() and not self.legacy.exists():
            raise FileNotFoundError(self.session_id)
        with self.open() as connection:
            row = connection.execute('SELECT data FROM metadata WHERE id=1').fetchone()
            if not row:
                raise FileNotFoundError(self.session_id)
            page = dict(self._page(connection, kind, before, limit), session_id=self.session_id)
            if before is not None or kind != 'messages':
                return page
            metadata = json.loads(row[0])
            active = metadata.get('active_project_id')
            project = metadata.get('projects', {}).get(active, {})
            tasks = connection.execute("SELECT task,json_extract(data,'$.tool'),json_extract(data,'$.outcome')"
                                       ' FROM business_tasks WHERE project=?', (active,))
            return dict(page, tasks={key: {'tool': tool, 'outcome': json.loads(facts)} for key, tool, facts in tasks},
                        input_dir=project.get('memory', {}).get('input_dir'), artifacts=project.get('artifacts', {}))

    def previous_write(self, project_id, write_tools):
        with self.open() as connection:
            rows = connection.execute('SELECT data FROM tool_calls WHERE project_id=? AND has_artifacts=1 ORDER BY seq DESC',
                                      (project_id,))
            for data, in rows:
                call = json.loads(data)
                if (call.get('ok') and call['tool'] in write_tools and any(Path(item['path']).exists()
                        for item in call.get('outcome', {}).get('artifacts', []))):
                    return call
