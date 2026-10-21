"""Annotated-gallery checkpoints, propagation and repeatable publication."""
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
from app.organize_tasks import prepare_face_database
from core_modules.organize.face_database_task import FaceDatabaseTask
from core_modules.organize.face_db_builder import FaceDBBuilder
from core_modules.organize.task_artifacts import fingerprint
from core_modules.organize.task_queue import TaskQueue
from test_core_upgrade_tasks import wait_for


class FaceDatabaseTasksTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.photos = self.root / '【整理完成】照片'
        self.photos.mkdir()
        self.gallery = self.root / 'gallery'
        self.report = self.root / 'report'
        self.calls = []
        self.single = self.photos / '01-20140304-戏剧场-京剧《甲》彩排-演员甲饰角色甲-摄影.jpg'
        self.multi = self.photos / '02-20140304-戏剧场-京剧《甲》彩排-演员甲饰角色甲、演员乙饰角色乙-摄影.jpg'
        self.unknown = self.photos / '03-20140304-戏剧场-京剧《甲》彩排-摄影.jpg'
        for path, color in ((self.single, 40), (self.multi, 140), (self.unknown, 90)):
            Image.new('RGB', (80, 60), (color, color, color)).save(path)
        def face(value, x):
            return SimpleNamespace(x=x, y=0, r=x+49, b=49, width=50, height=50,
                                   landmark=np.full((5, 2), value))
        def detect(img):
            value = int(img[0, 0, 0])
            self.calls.append(value)
            return [face(40, 0), face(220, 50)] if value > 100 else [face(40, 0)]
        def aligned(img, landmark):
            value = int(landmark[0, 0]) if landmark is not None else int(img[0, 0, 0])
            return np.full((160, 160, 3), value, dtype=np.uint8)
        def features(images):
            return np.array([[1., 0.] if image[0, 0, 0] < 100 else [0., 1.] for image in images], dtype=np.float32)
        self.system = SimpleNamespace(
            detector=SimpleNamespace(model_signature='det-v1', _detect_single=detect),
            extractor=SimpleNamespace(model_signature='arc-v1', align_face=aligned,
                                      extract_features=features, l2_normalize=lambda values: np.asarray(values)))

    def prepare(self, dry=False, gallery=None, maximum=8):
        return prepare_face_database({'organized_dirs': [str(self.photos)], 'db_root': str(gallery or self.gallery),
                                      'max_anchors': maximum, 'dry_run': dry, 'output_dir': str(self.report)})

    def submit(self, dry=False, factory=None, maximum=8):
        params, items = self.prepare(dry, maximum=maximum)
        queue = TaskQueue(self.root / 'tasks', {'face_database': factory or (lambda params: FaceDatabaseTask(params, self.system))})
        self.addCleanup(queue.close, 10)
        task, _ = queue.submit('face_database', params, items)
        return queue, task['task_id']

    def test_matches_legacy_single_and_propagated_anchor_bytes(self):
        legacy = self.root / 'legacy'
        baseline = FaceDBBuilder(self.system, str(legacy)).build_from_organized(str(self.photos))
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed', final)
        self.assertEqual(final['result']['added'], baseline['added'])
        self.assertEqual(final['result']['added'], 2)
        actual = {str(path.relative_to(self.gallery)): fingerprint(path) for path in self.gallery.rglob('*.jpg')}
        expected = {str(path.relative_to(legacy)): fingerprint(path) for path in legacy.rglob('*.jpg')}
        self.assertEqual(actual, expected)

    def test_dry_run_predicts_actual_propagation_without_gallery_writes(self):
        queue, task_id = self.submit(True)
        dry = wait_for(queue, task_id)
        self.assertEqual(dry['status'], 'completed', dry)
        self.assertEqual(dry['result']['added'], 2)
        self.assertFalse(self.gallery.exists())
        queue.close(10)
        queue, task_id = self.submit(False)
        self.assertEqual(wait_for(queue, task_id)['result']['added'], dry['result']['added'])

    def test_restart_skips_completed_candidate_extraction(self):
        reached, release = threading.Event(), threading.Event()
        def factory(params):
            handler = FaceDatabaseTask(params, self.system)
            original = handler.process
            def process(item):
                result = original(item)
                reached.set()
                release.wait(5)
                return result
            handler.process = process
            return handler
        queue, task_id = self.submit(factory=factory)
        try:
            self.assertTrue(reached.wait(5))
            queue.cancel(task_id)
        finally:
            release.set()
        self.assertEqual(wait_for(queue, task_id)['status'], 'cancelled')
        self.assertFalse(self.gallery.exists())
        queue.close(10)
        replacement = TaskQueue(self.root / 'tasks', {'face_database': lambda params: FaceDatabaseTask(params, self.system)})
        self.addCleanup(replacement.close, 10)
        replacement.resume(task_id)
        self.assertEqual(wait_for(replacement, task_id)['status'], 'completed')
        self.assertEqual(self.calls, [40, 140])

    def test_copy_failure_resume_reuses_own_anchors_without_inference(self):
        from core_modules.organize import face_database_task
        original = face_database_task.copy_verified
        calls, failed = [], []
        def copy(*args):
            if len(calls) == 1 and not failed:
                failed.append(True)
                raise OSError('stop before second anchor')
            calls.append(str(args[1]))
            return original(*args)
        with patch.object(face_database_task, 'copy_verified', copy):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
            completed = len(self.calls)
            queue.resume(task_id)
            final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed', final)
        self.assertEqual(final['result']['files_reused_this_attempt'], 1)
        self.assertEqual(len(self.calls), completed)

    def test_bad_photo_is_partial_error_and_other_anchors_publish(self):
        (self.photos / 'bad.jpg').write_bytes(b'not an image')
        # Use a labelled filename to exercise decode failure, not metadata skip.
        labelled = self.photos / self.single.name.replace('01-', '04-')
        labelled.write_bytes(b'not an image')
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed_with_errors', final)
        self.assertEqual((final['result']['added'], final['result']['failed']), (2, 1))

    def test_existing_conflicting_anchor_is_preserved(self):
        target = self.gallery / '演员甲（饰角色甲）' / (self.single.stem + '_face.jpg')
        target.parent.mkdir(parents=True)
        Image.new('RGB', (160, 160), 'red').save(target)
        digest = fingerprint(target)
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'failed')
        self.assertIn('anchor_output_conflict', final['error'])
        self.assertEqual(fingerprint(target), digest)

    def test_external_gallery_change_blocks_resume(self):
        from core_modules.organize import face_database_task
        with patch.object(face_database_task, 'copy_verified', side_effect=OSError('stop')):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        path = self.gallery / '外部演员' / 'anchor.jpg'
        path.parent.mkdir(parents=True)
        Image.new('RGB', (160, 160), 'white').save(path)
        queue.resume(task_id)
        self.assertIn('face_gallery_content_changed', wait_for(queue, task_id)['error'])

    def test_source_change_blocks_resumed_write(self):
        from core_modules.organize import face_database_task
        with patch.object(face_database_task, 'copy_verified', side_effect=OSError('stop')):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        self.single.write_bytes(b'changed')
        queue.resume(task_id)
        self.assertIn('source_content_changed', wait_for(queue, task_id)['error'])
        self.assertFalse(self.gallery.exists())

    def test_plan_interruption_is_idempotent(self):
        interrupted = []
        class Interrupted(FaceDatabaseTask):
            def advance(self):
                changed = super().advance()
                if changed and not interrupted:
                    interrupted.append(True)
                    raise OSError('after plan commit')
                return changed
        queue, task_id = self.submit(factory=lambda params: Interrupted(params, self.system))
        self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        queue.resume(task_id)
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(queue.item_page(task_id)), 3)

    def test_quota_is_applied_to_virtual_and_published_anchors(self):
        queue, task_id = self.submit(maximum=1)
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['result']['added'], 1)

    def test_model_change_rejects_resume_and_original_version_recovers(self):
        from core_modules.organize import face_database_task
        with patch.object(face_database_task, 'copy_verified', side_effect=OSError('stop')):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        completed = len(self.calls)
        self.system.extractor.model_signature = 'arc-v2'
        queue.resume(task_id)
        self.assertIn('execution_version_changed', wait_for(queue, task_id)['error'])
        self.assertFalse(self.gallery.exists())
        self.system.extractor.model_signature = 'arc-v1'
        queue.resume(task_id)
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(len(self.calls), completed)

    def test_report_failure_after_publication_resumes_without_models(self):
        from core_modules.organize import face_database_task
        with patch.object(face_database_task, 'atomic_json', side_effect=OSError('report unavailable')):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        completed = len(self.calls)
        queue.resume(task_id)
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed', final)
        self.assertEqual(final['result']['files_reused_this_attempt'], 2)
        self.assertEqual(len(self.calls), completed)
        self.assertEqual(json.loads((self.report / 'face_database_report.json').read_text())['added'], 2)

    def test_gallery_bmp_tiff_and_nested_images_have_complete_snapshot(self):
        person = self.gallery / '演员甲（饰角色甲）'
        person.mkdir(parents=True)
        for name in ('existing.bmp', 'existing.tiff'):
            Image.new('RGB', (160, 160), (40, 40, 40)).save(person / name)
        nested = person / 'archive' / 'retained.png'
        nested.parent.mkdir()
        Image.new('RGB', (160, 160), 'red').save(nested)
        params, items = self.prepare()
        self.assertEqual(len(params['gallery_paths']), 3)
        # Preserve legacy quota semantics: BMP/TIFF are matchable, JPG/PNG
        # count toward quota. Nested images are bound but not matchable.
        self.assertEqual(params['gallery_counts'].get(person.name, 0), 0)
        self.assertEqual(sum(item['stage'] == 'gallery' for item in items), 2)
        before = fingerprint(nested)
        queue, task_id = self.submit(True)
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed', final)
        self.assertEqual(fingerprint(nested), before)
        self.assertEqual(final['result']['added'], 2)

    def test_bad_initial_gallery_stops_before_any_writes(self):
        path = self.gallery / '演员甲（饰角色甲）' / 'bad.jpg'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'broken')
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'failed')
        self.assertIn('initial_gallery_feature_failed', final['error'])
        self.assertEqual(list(self.gallery.rglob('*.jpg')), [path])

    def test_symlink_person_cannot_write_into_source_photos(self):
        self.gallery.mkdir()
        (self.gallery / '演员甲（饰角色甲）').symlink_to(self.photos, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'face_gallery_symlink'):
            self.prepare()
        self.assertEqual(len(list(self.photos.iterdir())), 3)

    def test_report_child_symlink_cannot_write_into_source_photos(self):
        self.report.symlink_to(self.photos, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.photos / 'face_database_report.json').exists())

    def test_different_input_images_with_same_basename_reject_collision(self):
        other = self.root / '【整理完成】另一目录'
        other.mkdir()
        Image.new('RGB', (80, 60), (60, 60, 60)).save(other / self.single.name)
        params, items = prepare_face_database({'organized_dirs': [str(self.photos), str(other)],
                                               'db_root': str(self.gallery)})
        # Distinct aligned pixels force a target conflict despite identical label.
        original = self.system.extractor.align_face
        self.system.extractor.align_face = lambda img, landmark: original(img, None) if landmark is not None else img
        queue = TaskQueue(self.root / 'tasks', {'face_database': lambda p: FaceDatabaseTask(p, self.system)})
        self.addCleanup(queue.close, 10)
        task, _ = queue.submit('face_database', params, items)
        final = wait_for(queue, task['task_id'])
        self.assertEqual(final['status'], 'failed', final)
        self.assertIn('anchor_output_conflict', final['error'])
        self.assertFalse(self.gallery.exists())


if __name__ == '__main__':
    unittest.main()
