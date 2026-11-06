"""Qdrant container integration: actual storage, controlled embeddings and failures.

Run with TRANS_QDRANT_URL set to a disposable/acceptance Qdrant service.
Only UUID-named test collections and temporary files are created and removed.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.vector_store import QdrantStore, VectorStoreError, normalized, point_id, match_filter
from core_modules import qdrant_images as images
from core_modules.face_recognition.qdrant_database import QdrantFaceDatabase, collection_name


class Embedder:
    model_name = 'fixture'
    feature_dim = 4
    model_signature = 'weights-v1'
    preprocessing_signature = 'decode-v1'

    def __init__(self):
        self.batch_sizes = []
        self.fail = False
        self.mutate = False

    def extract_features_batch(self, paths, sub_batch_size=32):
        self.batch_sizes.append(len(paths))
        if self.fail:
            return np.empty((0, 4), dtype=np.float32), []
        vectors = [np.frombuffer(hashlib.sha256(Path(p).read_bytes()).digest()[:4], dtype=np.uint8).astype(np.float32) + 1 for p in paths]
        if self.mutate and paths:
            Path(paths[0]).write_bytes(b'changed')
        return np.asarray(vectors), paths


class RecordingStore(QdrantStore):
    def __init__(self):
        super().__init__()
        self.created = []

    def create(self, collection, *args, **kwargs):
        result = super().create(collection, *args, **kwargs)
        self.created.append(collection)
        return result


class ValidationTests(unittest.TestCase):
    def test_invalid_vectors_and_collections_rejected(self):
        for vector in ([0, 0], [float('nan'), 1], [float('inf'), 1], [[1, 2]]):
            with self.assertRaises(ValueError):
                normalized(vector, 2)
        with self.assertRaises(ValueError):
            normalized([1, 2], 3)
        with self.assertRaises(ValueError):
            QdrantStore.path('../other')

    def test_connection_failure_is_explicit(self):
        with patch('core_modules.vector_store.urlopen', side_effect=URLError('unavailable')):
            with self.assertRaisesRegex(VectorStoreError, 'qdrant_unavailable'):
                QdrantStore().count('test')


@unittest.skipUnless(os.environ.get('TRANS_QDRANT_URL'), 'requires Qdrant service')
class QdrantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RecordingStore()
        self.embedder = Embedder()
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        for i in range(3):
            (self.photos / f'{i}.jpg').write_bytes(f'image-{i}'.encode())

    def tearDown(self):
        for collection in self.store.created:
            if self.store.info(collection, allow_missing=True):
                self.store.delete_collection(collection)

    def build(self, **kwargs):
        result = images.build(self.photos, self.root / 'index', 'fixture', embedder=self.embedder,
                              store=self.store, **kwargs)
        return Path(result[2]), images.load_manifest(result[2])

    def test_store_exact_filter_upsert_delete(self):
        collection = 'test_' + uuid.uuid4().hex
        self.store.create(collection, 4, ('project_id',))
        self.store.upsert(collection, [
            {'id': point_id('a'), 'vector': [1, 0, 0, 0], 'payload': {'project_id': 'A'}},
            {'id': point_id('b'), 'vector': [0, 1, 0, 0], 'payload': {'project_id': 'B'}}])
        hits = self.store.query(collection, [2, 0, 0, 0], exact=True)
        self.assertEqual(hits[0]['id'], point_id('a'))
        self.assertAlmostEqual(hits[0]['score'], 1, places=5)
        self.assertEqual(len(self.store.query(collection, [1, 1, 0, 0], filters=match_filter({'project_id': 'B'}))), 1)
        self.store.upsert(collection, [{'id': point_id('a'), 'vector': [0, 0, 1, 0], 'payload': {'project_id': 'A'}}])
        self.assertEqual(self.store.count(collection), 2)
        self.store.delete_points(collection, filters=match_filter({'project_id': 'A'}))
        self.assertEqual(self.store.count(collection), 1)

    def test_collection_dimension_mismatch_rejected(self):
        collection = 'test_' + uuid.uuid4().hex
        self.store.create(collection, 4)
        with self.assertRaisesRegex(ValueError, 'vector_space_mismatch'):
            self.store.ensure(collection, 512)

    def test_streaming_build_and_small_manifest(self):
        for i in range(70):
            (self.photos / f'extra-{i}.jpg').write_bytes(str(i).encode())
        path, manifest = self.build(metadata={'project_id': 'play'})
        self.assertEqual(self.store.count(manifest['collection']), 73)
        self.assertLessEqual(max(self.embedder.batch_sizes), 32)
        self.assertFalse((path / 'embeddings.npy').exists())
        self.assertFalse((path / 'vector_index.faiss').exists())
        self.assertLess((path / images.MANIFEST).stat().st_size, 4096)

    def test_failed_rebuild_preserves_published_collection(self):
        path, old = self.build()
        self.embedder.fail = True
        with self.assertRaisesRegex(ValueError, 'index_build_failed_images'):
            self.build()
        self.assertEqual(images.load_manifest(path), old)
        self.assertEqual(self.store.count(old['collection']), 3)
        self.assertIsNone(self.store.info(self.store.created[-1], allow_missing=True))

    def test_database_upload_does_not_block_query_inference(self):
        uploading, release = threading.Event(), threading.Event()
        original = self.store.upsert
        errors = []

        def delayed(*args, **kwargs):
            uploading.set()
            if not release.wait(10):
                raise RuntimeError('test_upload_timeout')
            return original(*args, **kwargs)

        def run():
            try:
                self.build()
            except Exception as exc:
                errors.append(exc)

        with patch.object(self.store, 'upsert', side_effect=delayed):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(uploading.wait(10))
                acquired = images.MODEL_LOCK.acquire(blocking=False)
                if acquired:
                    images.MODEL_LOCK.release()
                self.assertTrue(acquired, 'database IO must allow another query to embed')
            finally:
                release.set()
                thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_changed_source_does_not_publish(self):
        self.embedder.mutate = True
        with self.assertRaisesRegex(ValueError, 'image_changed'):
            self.build()
        self.assertFalse((self.root / 'index/fixture' / images.MANIFEST).exists())

    def test_upsert_is_idempotent_and_keeps_filter_metadata(self):
        path, manifest = self.build(metadata={'project_id': 'play'})
        photo = str(self.photos / '0.jpg')
        Path(photo).write_bytes(b'new photo')
        for _ in range(2):
            result = images.upsert_images(path, [photo], embedder=self.embedder, store=self.store)
            self.assertEqual(result['image_count'], 3)
        point = self.store.retrieve(manifest['collection'], [point_id(photo)])[0]
        self.assertEqual(point['payload']['project_id'], 'play')
        self.assertEqual(point['payload']['content_sha256'], images.fingerprint(photo))
        self.assertEqual(images.delete_images(path, [photo], self.store)['deleted'], 1)
        self.assertEqual(images.delete_images(path, [photo], self.store)['deleted'], 0)

    def test_model_change_refuses_incremental_write(self):
        path, manifest = self.build()
        self.embedder.model_signature = 'weights-v2'
        with self.assertRaisesRegex(ValueError, 'model_changed'):
            images.upsert_images(path, [str(self.photos / '0.jpg')], embedder=self.embedder, store=self.store)
        self.assertEqual(self.store.count(manifest['collection']), 3)

    def test_rebuild_publishes_new_collection_and_keeps_previous(self):
        path, old = self.build()
        path, new = self.build()
        self.assertNotEqual(old['collection'], new['collection'])
        self.assertEqual(new['previous_collection'], old['collection'])
        self.assertEqual(self.store.count(old['collection']), 3)

    def test_prune_keeps_published_and_previous_generations(self):
        from tools.prune_vector_generations import prune
        from datetime import datetime, timezone, timedelta
        path, first = self.build()
        self.build()
        path, current = self.build()
        old_time = (datetime.now(timezone.utc)-timedelta(days=2)).isoformat()
        for name in self.store.created:
            self.store.request('PATCH', self.store.path(name), {'metadata': {
                'owner_index': str(path.resolve()), 'created_at': old_time}})
        preview = prune(self.root, store=self.store)
        self.assertEqual([r['collection'] for r in preview['candidates']], [first['collection']])
        self.assertTrue(self.store.info(first['collection']))
        self.assertEqual(len(prune(self.root, apply=True, store=self.store)['candidates']), 1)
        self.assertIsNone(self.store.info(first['collection'], allow_missing=True))
        self.assertTrue(self.store.info(current['collection']))
        self.assertTrue(self.store.info(current['previous_collection']))

    def test_faces_exact_batch_threshold_and_idempotence(self):
        db = QdrantFaceDatabase('test-' + uuid.uuid4().hex, self.store)
        first, second = np.eye(2, 512, dtype=np.float32)
        db.add_face('a', '甲', first, 'a.jpg')
        db.add_face('a', '甲', first, 'a.jpg')
        db.add_face('b', '乙', second, 'b.jpg')
        self.assertEqual(self.store.count(db.index_name), 2)
        result = db.search_faces_batch([first, second], threshold=.55)
        self.assertEqual([row[0]['person_id'] for row in result], ['a', 'b'])
        self.assertAlmostEqual(result[0][0]['score'], 1, places=5)
        self.assertEqual(len(result[0]), 1)
        self.assertTrue(db.check_archive_exists('a'))
        self.assertFalse(db.check_archive_exists('a'))
        self.assertNotEqual(collection_name('model-1'), collection_name('model-2'))

    def test_faiss_migration_preserves_scores_and_source(self):
        import faiss
        from tools.migrate_vector_store import migrate_faiss
        source = self.root / 'legacy'
        source.mkdir()
        paths = images.image_paths(self.photos)
        vectors, _ = self.embedder.extract_features_batch(paths)
        vectors = np.asarray([normalized(v) for v in vectors], dtype=np.float32)
        index = faiss.IndexFlatIP(4)
        index.add(vectors)
        faiss.write_index(index, str(source / 'vector_index.faiss'))
        np.save(source / 'embeddings.npy', vectors)
        (source / 'image_paths.txt').write_text('\n'.join(paths) + '\n')
        (source / 'model_info.txt').write_text('model: fixture\ndimension: 4\n')
        before = images.fingerprint(source / 'vector_index.faiss')
        destination = self.root / 'migrated'
        report = migrate_faiss(source, destination, self.embedder, self.store)
        self.assertEqual(report['migrated'], 3)
        self.assertTrue(report['legacy_model_unverified'])
        hits = images.query_vector(images.load_manifest(destination), vectors[0], 3, exact=True, store=self.store)
        scores, order = index.search(vectors[:1], 3)
        self.assertEqual([h['image_path'] for h in hits], [paths[i] for i in order[0]])
        np.testing.assert_allclose([h['similarity'] for h in hits], scores[0], atol=1e-6)
        self.assertEqual(images.fingerprint(source / 'vector_index.faiss'), before)


if __name__ == '__main__':
    unittest.main()
