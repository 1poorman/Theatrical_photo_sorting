"""SQLite authority for reviews, with recoverable legacy JSONL snapshots."""
from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile


class ReviewConflict(ValueError):
    """The record changed after a reviewer loaded it."""


def atomic_rows(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'review.sqlite3'

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.review.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    @staticmethod
    def encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def _initialize(self, connection):
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS records (
                image TEXT PRIMARY KEY, payload TEXT NOT NULL,
                status TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS records_status ON records(status, image);
            CREATE TABLE IF NOT EXISTS revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS bindings (image TEXT PRIMARY KEY, source_image TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS revisions_no_update BEFORE UPDATE ON revisions
                BEGIN SELECT RAISE(ABORT, 'revision history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS revisions_no_delete BEFORE DELETE ON revisions
                BEGIN SELECT RAISE(ABORT, 'revision history is immutable'); END;
        ''')
        if connection.execute("SELECT 1 FROM metadata WHERE key='imported'").fetchone():
            return
        # Parse and back up every old file before entering the import transaction.
        # Failed parsing/import leaves the old files untouched and is retryable.
        imported = {}
        for name in ('review_queue.jsonl', 'revisions.jsonl', 'image_bindings.jsonl'):
            path = self.root / name
            data = path.read_bytes() if path.exists() else b''
            imported[name] = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
            if data:
                backup = self.root / 'legacy_backup'
                backup.mkdir(exist_ok=True)
                target = backup / (name + '.' + hashlib.sha256(data).hexdigest())
                # Exclusive creation never overwrites a previous migration backup.
                try:
                    with target.open('xb') as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise ValueError('legacy_backup_content_mismatch')
        with connection:
            for row in imported['review_queue.jsonl']:
                connection.execute('INSERT OR REPLACE INTO records VALUES (?, ?, ?, 0)',
                                   (row['image'], self.encode(row), (row.get('review') or {}).get('status', 'pending')))
            for row in imported['revisions.jsonl']:
                connection.execute('INSERT INTO revisions(payload) VALUES (?)', (self.encode(row),))
            for row in imported['image_bindings.jsonl']:
                connection.execute('INSERT OR REPLACE INTO bindings VALUES (?, ?)', (row['image'], row['source_image']))
            connection.execute("INSERT INTO metadata VALUES ('imported','1')")
            connection.execute("INSERT INTO metadata VALUES ('dirty','0')")

    def _snapshots(self, connection):
        # SQLite is authoritative. If the process dies between file replacements,
        # dirty remains 1; the next reader regenerates all snapshots from one state.
        atomic_rows(self.root / 'review_queue.jsonl', (
            json.loads(row[0]) for row in connection.execute('SELECT payload FROM records ORDER BY image')))
        atomic_rows(self.root / 'revisions.jsonl', (
            json.loads(row[0]) for row in connection.execute('SELECT payload FROM revisions ORDER BY sequence')))
        bindings = [{'image': image, 'source_image': source} for image, source in
                    connection.execute('SELECT image, source_image FROM bindings ORDER BY image')]
        if bindings or (self.root / 'image_bindings.jsonl').exists():
            atomic_rows(self.root / 'image_bindings.jsonl', bindings)
        with connection:
            connection.execute("UPDATE metadata SET value='0' WHERE key='dirty'")

    @contextmanager
    def session(self, write=False):
        # Lock imports, mutations and compatibility snapshot recovery across
        # processes. Ordinary initialized readers only need SQLite snapshots.
        if not self.path.exists() or write:
            with self._lock():
                with closing(self._connect()) as connection:
                    self._initialize(connection)
                    if connection.execute("SELECT value FROM metadata WHERE key='dirty'").fetchone()[0] == '1':
                        self._snapshots(connection)
                    connection.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
                    try:
                        yield connection
                        if write:
                            connection.execute("UPDATE metadata SET value='1' WHERE key='dirty'")
                            connection.commit()
                            self._snapshots(connection)
                    except BaseException:
                        connection.rollback()
                        raise
            return
        connection = self._connect()
        try:
            try:
                state = dict(connection.execute('SELECT key, value FROM metadata'))
            except sqlite3.OperationalError:
                state = {}
            if state.get('imported') != '1' or state.get('dirty') == '1':
                connection.close()
                with self._lock():
                    connection = self._connect()
                    self._initialize(connection)
                    if connection.execute("SELECT value FROM metadata WHERE key='dirty'").fetchone()[0] == '1':
                        self._snapshots(connection)
            connection.execute('BEGIN')
            yield connection
        finally:
            connection.close()
