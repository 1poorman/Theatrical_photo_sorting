"""Durable clustering recovery with isolated images and gallery outputs."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT), str(ROOT / 'tests')]
from app.organize_tasks import prepare_face_cluster
from core_modules.organize.face_cluster import FaceClusterBuilder
from core_modules.organize.face_cluster_task import FaceClusterTask
from core_modules.organize.task_artifacts import fingerprint
from core_modules.organize.task_queue import TaskQueue
from test_core_upgrade_tasks import wait_for


class FaceClusterTasksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        for index in range(3):
            Image.new('RGB', (80, 60), (40 + index * 50, 0, 0)).save(self.photos / f'{index}.png')
        self.output = self.root / 'clusters'
        self.calls = []
        self.batches = []
        self.face = SimpleNamespace(x=0, y=0, r=59, b=59, width=60, height=60,
                                    landmark=np.zeros((5, 2)))
        def detect(img):
            self.calls.append(int(img[0, 0, 2]))
            return [self.face, self.face]
        def extract(faces):
            self.batches.append(len(faces))
            return np.tile(np.array([1., 0.], dtype=np.float32), (len(faces), 1))
        self.system = SimpleNamespace(
            detector=SimpleNamespace(model_signature='detector-v1', input_size=(640, 640), _detect_single=detect),
            extractor=SimpleNamespace(model_signature='extractor-v1', max_batch_size=1,
                                      align_face=lambda img, landmark: np.full((160, 160, 3), img[0, 0], dtype=np.uint8),
                                      extract_features=extract, l2_normalize=lambda values: np.asarray(values)))

    def submit(self, factory=None, output=None):
        params, items = prepare_face_cluster({'input_dir': str(self.photos), 'output_dir': str(output or self.output)})
        queue = TaskQueue(self.root / 'tasks', {'face_cluster': factory or (lambda params: FaceClusterTask(params, self.system))})
        self.addCleanup(queue.close, 10)
        task, _ = queue.submit('face_cluster', params, items)
        return queue, task['task_id']

    def test_legacy_grouping_previews_and_bounded_batches_match(self):
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        state = FaceClusterBuilder.load_state(self.output / 'cluster_state.json')
        legacy = FaceClusterBuilder(self.system).scan(str(self.photos), str(self.root / 'legacy'))
        for field in ('n_faces', 'n_images'):
            self.assertEqual(state[field], legacy[field])
        self.assertEqual([(c['id'], c['size'], c['cohesion'], c['members']) for c in state['clusters']],
                         [(c['id'], c['size'], c['cohesion'], c['members']) for c in legacy['clusters']])
        self.assertEqual(Path(state['clusters'][0]['preview']).read_bytes(), Path(legacy['clusters'][0]['preview']).read_bytes())
        self.assertTrue(all(batch <= 1 for batch in self.batches))

    def test_restart_after_cancel_skips_completed_detection(self):
        entered, release = threading.Event(), threading.Event()
        def factory(params):
            handler = FaceClusterTask(params, self.system)
            original = handler.process
            def process(item):
                row = original(item)
                if item['stage'] == 'faces':
                    entered.set()
                    release.wait(5)
                return row
            handler.process = process
            return handler
        queue, task_id = self.submit(factory)
        try:
            self.assertTrue(entered.wait(5))
            queue.cancel(task_id)
        finally:
            release.set()
        self.assertEqual(wait_for(queue, task_id)['status'], 'cancelled')
        queue.close(10)
        replacement = TaskQueue(self.root / 'tasks', {'face_cluster': lambda params: FaceClusterTask(params, self.system)})
        self.addCleanup(replacement.close, 10)
        replacement.resume(task_id)
        self.assertEqual(wait_for(replacement, task_id)['status'], 'completed')
        self.assertEqual(self.calls, [40, 90, 140])

    def test_bad_image_is_partial_failure_with_other_faces_preserved(self):
        (self.photos / 'bad.jpg').write_bytes(b'bad image')
        queue, task_id = self.submit()
        result = wait_for(queue, task_id)
        self.assertEqual(result['status'], 'completed_with_errors')
        self.assertEqual((result['result']['n_images'], result['result']['n_faces'], result['result']['failed']), (4, 6, 1))

    def test_export_failure_resume_reuses_preview_without_inference(self):
        from core_modules.organize import face_cluster_task
        original = face_cluster_task.atomic_json
        failed = []
        def publish(path, state):
            if not failed:
                failed.append(True)
                raise OSError('crash before state commit')
            return original(path, state)
        with patch.object(face_cluster_task, 'atomic_json', publish):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
            preview = self.output / 'previews/cluster-01.jpg'
            before = preview.stat().st_ino
            queue.resume(task_id)
            self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(preview.stat().st_ino, before)
        self.assertEqual(len(self.calls), 3)

    def test_new_scan_preserves_human_cluster_decisions(self):
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        state_path = self.output / 'cluster_state.json'
        FaceClusterBuilder(self.system).ignore(str(state_path), 'cluster-01')
        queue.close(10)
        second, second_id = self.submit()
        self.assertEqual(wait_for(second, second_id)['status'], 'completed')
        self.assertEqual(FaceClusterBuilder.load_state(state_path)['clusters'][0]['status'], 'ignored')

    def test_assign_saves_two_faces_from_same_image_and_reuses_retry(self):
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        builder = FaceClusterBuilder(self.system)
        gallery = self.root / 'gallery'
        state_path = str(self.output / 'cluster_state.json')
        result = builder.assign(state_path, 'cluster-01', '演员', str(gallery))
        self.assertEqual(result['saved'], 6)
        repeated = builder.assign(state_path, 'cluster-01', '演员', str(gallery))
        self.assertEqual((repeated['saved'], repeated['skipped']), (0, 6))
        self.assertEqual(len(list((gallery / '演员').glob('*.jpg'))), 6)

    def test_source_change_prevents_assignment_before_gallery_write(self):
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        (self.photos / '0.png').write_bytes(b'changed')
        gallery = self.root / 'gallery'
        with self.assertRaisesRegex(ValueError, 'source_content_changed'):
            FaceClusterBuilder(self.system).assign(str(self.output / 'cluster_state.json'), 'cluster-01', '演员', str(gallery))
        self.assertFalse(gallery.exists())

    def test_changed_model_prevents_resume_without_new_detection(self):
        from core_modules.organize import face_cluster_task
        with patch.object(face_cluster_task, 'atomic_json', side_effect=OSError('stop at publish')):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        self.system.extractor.model_signature = 'extractor-v2'
        queue.resume(task_id)
        self.assertIn('execution_version_changed', wait_for(queue, task_id)['error'])
        self.assertEqual(len(self.calls), 3)

    def test_existing_unrelated_state_and_preview_are_preserved(self):
        self.output.mkdir()
        state_path = self.output / 'cluster_state.json'
        original = json.dumps({'clusters': [], 'scan_id': 'unrelated'})
        state_path.write_text(original)
        queue, task_id = self.submit()
        self.assertIn('cluster_output_conflict', wait_for(queue, task_id)['error'])
        self.assertEqual(state_path.read_text(), original)

    def test_invalid_output_inside_originals_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_face_cluster({'input_dir': str(self.photos), 'output_dir': str(self.photos / 'clusters')})

    def test_cluster_plan_interruption_does_not_append_previews_twice(self):
        interrupted = []
        class Interrupted(FaceClusterTask):
            def advance(self):
                advanced = super().advance()
                if advanced and not interrupted:
                    interrupted.append(True)
                    raise OSError('interrupted after plan commit')
                return advanced
        queue, task_id = self.submit(lambda params: Interrupted(params, self.system))
        self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        queue.resume(task_id)
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(len(queue.item_page(task_id)), 4)
        self.assertEqual(len(self.calls), 3)

    def test_source_change_after_checkpoint_cannot_publish_stale_clustering(self):
        from core_modules.organize import face_cluster_task
        with patch.object(face_cluster_task, 'atomic_json', side_effect=OSError('stop at publish')):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        (self.photos / '0.png').write_bytes(b'changed source')
        queue.resume(task_id)
        self.assertIn('source_content_changed', wait_for(queue, task_id)['error'])
        self.assertEqual(len(self.calls), 3)
        self.assertFalse((self.output / 'cluster_state.json').exists())

    def test_assignment_output_symlink_cannot_write_into_originals(self):
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        gallery = self.root / 'gallery'
        gallery.mkdir()
        (gallery / '演员').symlink_to(self.photos, target_is_directory=True)
        before = {path.name: fingerprint(path) for path in self.photos.iterdir()}
        with self.assertRaisesRegex(ValueError, 'output_directory_symlink'):
            FaceClusterBuilder(self.system).assign(str(self.output / 'cluster_state.json'), 'cluster-01', '演员', str(gallery))
        self.assertEqual({path.name: fingerprint(path) for path in self.photos.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
