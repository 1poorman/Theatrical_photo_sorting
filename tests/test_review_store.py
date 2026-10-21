"""Transactional migration, contention, crashes, and indexed large-queue reads."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.review_queue import ReviewQueue, load_scene_evidence
from core_modules.organize.review_store import ReviewConflict


def crash_during_snapshot(root):
    queue = ReviewQueue(root)
    with patch('core_modules.organize.review_store.atomic_rows', side_effect=lambda *args: os._exit(7)):
        queue.decide('a.jpg', 'modify', label='committed', expected_revision=0)


def crash_during_transaction(root):
    queue = ReviewQueue(root)
    with queue.store.session(write=True) as connection:
        queue._decide(connection, 'a.jpg', 'modify', None, 'uncommitted', 'human', '', 0)
        os._exit(8)


def competing_writer(root, label, ready, go, result):
    queue = ReviewQueue(root)
    revision = queue.record('a.jpg')['review_revision']
    ready.put(label)
    go.wait(10)
    try:
        queue.decide('a.jpg', 'modify', label=label, expected_revision=revision)
        result.put('saved')
    except ReviewConflict:
        result.put('conflict')


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.queue = ReviewQueue(str(self.root))

    def seed(self, names=('a.jpg', 'b.jpg')):
        self.queue.seed([{'image': name, 'label': 'candidate', 'scene_id': 'act-01'} for name in names])

    def test_legacy_migration_preserves_payloads_history_and_exact_backups(self):
        record = {'image': 'a.jpg', 'label': 'human', 'review': {'status': 'approved'}}
        revision = {'image': 'a.jpg', 'before': {}, 'after': {'label': 'human'}}
        values = {'review_queue.jsonl': [record], 'revisions.jsonl': [revision],
                  'image_bindings.jsonl': [{'image': 'a.jpg', 'source_image': '/one/a.jpg'}]}
        for name, rows in values.items():
            data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
            (self.root / name).write_bytes(data)
        self.assertEqual(self.queue.records(), {'a.jpg': record})
        self.assertEqual(self.queue.revisions(), [revision])
        self.assertEqual(self.queue.record('a.jpg')['source_image'], '/one/a.jpg')
        for name in values:
            data = (self.root / name).read_bytes()
            backup = self.root / 'legacy_backup' / (name + '.' + hashlib.sha256(data).hexdigest())
            self.assertEqual(backup.read_bytes(), data)
        self.queue.seed([{'image': '/one/a.jpg', 'label': 'machine'}])
        self.assertEqual(self.queue.records(), {'a.jpg': record})
        self.assertEqual(self.queue.revisions(), [revision])
        exported, count = self.queue.export_final()
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(Path(exported).read_text()), {'/one/a.jpg': 'human'})
        self.assertEqual(load_scene_evidence(self.queue.queue_path)['by_path'], {'/one/a.jpg': 'human'})

    def test_invalid_migration_does_not_change_original_and_can_retry(self):
        path = self.root / 'review_queue.jsonl'
        path.write_bytes(b'not json\n')
        with self.assertRaises(ValueError):
            self.queue.records()
        self.assertEqual(path.read_bytes(), b'not json\n')
        path.write_text('{"image":"a.jpg","label":"candidate"}\n')
        self.assertEqual(set(self.queue.records()), {'a.jpg'})

    def test_same_revision_exactly_one_writer_succeeds(self):
        self.seed()
        def save(label):
            try:
                return ReviewQueue(str(self.root)).decide('a.jpg', 'modify', label=label, expected_revision=0)
            except ReviewConflict:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(save, ('first', 'second')))
        self.assertEqual(results.count('conflict'), 1)
        self.assertEqual(len(self.queue.revisions()), 1)
        self.assertEqual(self.queue.record('a.jpg')['review_revision'], 1)

    def test_different_records_both_persist(self):
        self.seed()
        def save(image):
            return ReviewQueue(str(self.root)).decide(image, 'modify', label=image, expected_revision=0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(save, ('a.jpg', 'b.jpg')))
        self.assertEqual(self.queue.final_labels(), {'a.jpg': 'a.jpg', 'b.jpg': 'b.jpg'})
        self.assertEqual(len(self.queue.revisions()), 2)

    def test_separate_processes_compare_same_revision_atomically(self):
        self.seed()
        context = multiprocessing.get_context('spawn')
        ready, result, go = context.Queue(), context.Queue(), context.Event()
        workers = [context.Process(target=competing_writer,
                   args=(str(self.root), label, ready, go, result)) for label in ('first', 'second')]
        for worker in workers:
            worker.start()
        for _ in workers:
            ready.get(timeout=10)
        go.set()
        outcomes = [result.get(timeout=10) for _ in workers]
        for worker in workers:
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted(outcomes), ['conflict', 'saved'])
        self.assertEqual(len(self.queue.revisions()), 1)

    def test_decision_and_revision_rollback_together_on_database_failure(self):
        self.seed()
        before = self.queue.records()
        with sqlite3.connect(self.queue.store.path) as connection:
            connection.execute("CREATE TRIGGER inject_failure BEFORE INSERT ON revisions BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.queue.decide('a.jpg', 'modify', label='never committed')
        self.assertEqual(self.queue.records(), before)
        self.assertEqual(self.queue.revisions(), [])

    def test_crash_before_commit_rolls_back_and_after_commit_recovers_snapshots(self):
        self.seed()
        context = multiprocessing.get_context('spawn')
        first = context.Process(target=crash_during_transaction, args=(str(self.root),))
        first.start(); first.join(15)
        self.assertEqual(first.exitcode, 8)
        self.assertEqual(self.queue.record('a.jpg')['label'], 'candidate')
        self.assertEqual(self.queue.revisions(), [])
        second = context.Process(target=crash_during_snapshot, args=(str(self.root),))
        second.start(); second.join(15)
        self.assertEqual(second.exitcode, 7)
        restarted = ReviewQueue(str(self.root))
        self.assertEqual(restarted.record('a.jpg')['label'], 'committed')
        self.assertEqual(len(restarted.revisions()), 1)
        rows = [json.loads(line) for line in (self.root / 'review_queue.jsonl').read_text().splitlines()]
        self.assertEqual({row['image']: row for row in rows}, restarted.records())
        revisions = [json.loads(line) for line in (self.root / 'revisions.jsonl').read_text().splitlines()]
        self.assertEqual(revisions, restarted.revisions())

    def test_history_is_immutable_at_database_layer(self):
        self.seed()
        self.queue.decide('a.jpg', 'accept')
        with sqlite3.connect(self.queue.store.path) as connection:
            for statement in ["UPDATE revisions SET payload='{}'", 'DELETE FROM revisions']:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement)

    def test_ten_thousand_records_page_without_loading_all_payloads(self):
        self.seed(tuple(f'photo-{index:05}.jpg' for index in range(10000)))
        with patch.object(self.queue, '_records', side_effect=AssertionError('full queue read')):
            pages = [self.queue.page(limit=200, offset=offset) for offset in range(0, 10000, 200)]
            images = [row['image'] for page in pages for row in page['records']]
            self.assertEqual(len(images), 10000)
            self.assertEqual(len(set(images)), 10000)
            self.assertEqual(images, sorted(images))
            self.assertEqual(pages[-1]['summary'], {'pending': 10000, 'approved': 0, 'rejected': 0})
            self.queue.decide('photo-09999.jpg', 'accept', expected_revision=0)
            self.assertEqual(self.queue.page('approved', limit=1)['records'][0]['image'], 'photo-09999.jpg')
            self.assertEqual(self.queue.record('photo-09999.jpg')['review_revision'], 1)


if __name__ == '__main__':
    unittest.main()
