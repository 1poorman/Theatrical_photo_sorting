"""Staged organize/knowledge behavior, partial failures and publication recovery."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]
from app.organize_tasks import prepare_organize, prepare_knowledge, knowledge_handler
from core_modules.organize.organize_task import OrganizeTask
from core_modules.organize.smart_organizer import SmartOrganizer
from core_modules.organize.task_queue import TaskQueue, TaskStopped
from core_modules.organize.task_artifacts import fingerprint
from core_modules.organize.play_knowledge import build_knowledge, read_knowledge
from test_core_upgrade_tasks import wait_for


class OrganizeTasksTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        self.gallery = self.root / 'gallery'
        self.gallery.mkdir()
        self.output = self.root / 'output'
        for name, color in [('20140304-剧场-京剧《甲》-摄影乙.jpg', 'red'),
                            ('20140304-剧场-京剧《甲》-摄影乙(2).jpg', 'green'),
                            ('20140305-剧场-京剧《乙》-摄影乙.jpg', 'blue')]:
            Image.new('RGB', (40, 30), color).save(self.photos / name)
        self.calls = []
        self.encoder = SimpleNamespace(model_signature='organize-task-test', extract_features=self.encode)
        self.organizer = SmartOrganizer(None, str(self.gallery), embedder=self.encoder,
                                        feature_cache_dir=str(self.root / 'features'))
        self.organizer._recognize_image = self.recognize

    def recognize(self, img, threshold):
        self.calls.append('recognize')
        return {'演员（饰角色）': .9}, []

    def encode(self, image):
        self.calls.append(('encode', image))
        return np.array([1., 0.])

    def submit(self, handler=None):
        prepared, items = prepare_organize({'input_dir': str(self.photos), 'output_dir': str(self.output),
                                            'db_root': str(self.gallery), 'keep_per_bucket': 1, 'classify_role': False})
        queue = TaskQueue(self.root / 'tasks', {'organize': handler or (lambda params: OrganizeTask(params, self.organizer))})
        self.addCleanup(queue.close, 10)
        task, _ = queue.submit('organize', prepared, items)
        return queue, task['task_id']

    def test_matches_legacy_selection_names_identity_and_output_bytes(self):
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed')
        report = json.loads((self.output / 'organize_report.json').read_text())
        self.assertEqual((report['total_input'], report['kept'], report['discarded'], report['failed']), (3, 2, 1, 0))
        old = self.organizer.organize(str(self.photos), str(self.root / 'legacy'), keep_per_bucket=1, classify_role=False)
        self.assertEqual([(r['source'], r['organized_name'], r['recognized']) for r in report['records']],
                         [(r['source'], r['organized_name'], r['recognized']) for r in old['records']])
        for row in report['records']:
            self.assertEqual(fingerprint(self.photos / row['source']), fingerprint(self.output / 'organized' / row['organized_name']))

    def test_bad_quality_image_and_recognition_error_leave_other_results(self):
        (self.photos / 'bad.jpg').write_bytes(b'broken image')
        self.organizer._recognize_image = lambda img, threshold: (_ for _ in ()).throw(ValueError('model failed')) if img[0, 0, 0] > 100 else ({}, [])
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'completed_with_errors')
        report = json.loads((self.output / 'organize_report.json').read_text())
        self.assertEqual(report['failed'], 2)
        self.assertEqual(report['kept'], 1)
        self.assertEqual({r['stage'] for r in report['errors']}, {'quality', 'recognition'})
        self.assertEqual(len(list(self.output.joinpath('organized').iterdir())), 1)

    def test_cancel_after_recognition_keeps_selection_and_skips_completed_inference(self):
        started, release = threading.Event(), threading.Event()
        def recognize(img, threshold):
            self.calls.append('recognize')
            started.set()
            release.wait(5)
            return {}, []
        self.organizer._recognize_image = recognize
        queue, task_id = self.submit()
        try:
            self.assertTrue(started.wait(5))
            queue.cancel(task_id)
        finally:
            release.set()
        final = wait_for(queue, task_id, {'cancelled'})
        self.assertEqual(final['pending'], 1)
        self.assertEqual(self.calls.count('recognize'), 1)
        self.assertFalse(self.output.exists())
        queue.resume(task_id)
        wait_for(queue, task_id, {'completed'})
        self.assertEqual(self.calls.count('recognize'), 2)
        self.assertEqual(len(queue.item_page(task_id)), 5, 'stage plan must not be appended twice')

    def test_publication_failure_reuses_copies_and_never_repeats_inference(self):
        from core_modules.organize import organize_task
        original = organize_task.copy_verified
        copied, failures = [], []
        def copy(source, destination, digest):
            if len(copied) == 2 and not failures:
                failures.append(True)
                raise OSError('injected copy failure')
            result = original(source, destination, digest)
            if result:
                copied.append(str(destination))
            return result
        with patch.object(organize_task, 'copy_verified', copy):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
            count = len(self.calls)
            queue.resume(task_id)
            self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(len(self.calls), count)
        self.assertEqual(len(copied), len(set(copied)))
        self.assertEqual(len(copied), 4)
        report = json.loads((self.output / 'organize_report.json').read_text())
        self.assertEqual(report['files_reused_this_attempt'], 2)

    def test_unrelated_output_is_preserved_and_causes_task_failure(self):
        (self.output / 'best').mkdir(parents=True)
        first = sorted(self.photos.iterdir())[0]
        collision = self.output / 'best' / first.name
        collision.write_bytes(b'preserve this existing file')
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        self.assertEqual(collision.read_bytes(), b'preserve this existing file')

    def test_output_directory_link_cannot_copy_into_source(self):
        self.output.mkdir()
        (self.output / 'organized').symlink_to(self.photos, target_is_directory=True)
        originals = {path.name: fingerprint(path) for path in self.photos.iterdir()}
        queue, task_id = self.submit()
        self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        self.assertEqual({path.name: fingerprint(path) for path in self.photos.iterdir()}, originals)

    def test_stage_plan_is_committed_once_even_after_advance_interruption(self):
        crashes = []
        class InterruptedPlan(OrganizeTask):
            def advance(self):
                value = super().advance()
                if value and not crashes:
                    crashes.append(True)
                    raise OSError('after committed stage plan')
                return value
        queue, task_id = self.submit(lambda params: InterruptedPlan(params, self.organizer))
        self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        queue.resume(task_id)
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(len(queue.item_page(task_id)), 5)

    def test_knowledge_matches_legacy_and_resume_preserves_media_checkpoints(self):
        info = self.root / 'info.txt'
        info.write_text('京剧《甲》\n第1幕开始\n甲劝谏乙。')
        params, items = prepare_knowledge({'info_path': str(info), 'play_root': str(self.photos), 'output_dir': str(self.root / 'knowledge')})
        started, release = threading.Event(), threading.Event()
        processed = []
        def handler(params):
            value = knowledge_handler(params)
            original = value.process
            def process(item):
                processed.append(item['image'])
                started.set()
                release.wait(5)
                return original(item)
            value.process = process
            return value
        queue = TaskQueue(self.root / 'tasks', {'knowledge': handler})
        self.addCleanup(queue.close, 10)
        task, _ = queue.submit('knowledge', params, items)
        task_id = task['task_id']
        try:
            self.assertTrue(started.wait(5))
            queue.cancel(task_id)
        finally:
            release.set()
        wait_for(queue, task_id, {'cancelled'})
        queue.resume(task_id)
        self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(len(processed), len(set(processed)))
        expected = build_knowledge(str(info), str(self.photos))
        entities = json.loads((self.root / 'knowledge/entities.json').read_text())
        self.assertEqual(entities['scenes'], expected['scenes'])
        self.assertEqual(entities['roles'], expected['roles'])
        self.assertEqual([json.loads(row) for row in (self.root / 'knowledge/media.jsonl').read_text().splitlines()], expected['media'])

    def test_knowledge_failed_publication_keeps_previous_generation_until_resume(self):
        info = self.root / 'info.txt'
        info.write_text('京剧《甲》\n第1幕旧幕\n旧剧情。')
        output = self.root / 'knowledge'
        old = build_knowledge(str(info), str(self.photos), str(output))
        info.write_text('京剧《甲》\n第2幕新幕\n新剧情。')
        params, items = prepare_knowledge({'info_path': str(info), 'play_root': str(self.photos), 'output_dir': str(output)})
        from core_modules.organize import knowledge_task
        original = knowledge_task.atomic_json
        failures = []
        def publish(path, value):
            if Path(path).name == 'knowledge_snapshot.json' and not failures:
                failures.append(True)
                raise OSError('crash before publication commit')
            return original(path, value)
        with patch.object(knowledge_task, 'atomic_json', publish):
            queue = TaskQueue(self.root / 'tasks', {'knowledge': knowledge_handler})
            self.addCleanup(queue.close, 10)
            task, _ = queue.submit('knowledge', params, items)
            self.assertEqual(wait_for(queue, task['task_id'])['status'], 'failed')
            self.assertEqual(read_knowledge(output)['entities']['scenes'], old['scenes'])
            queue.resume(task['task_id'])
            self.assertEqual(wait_for(queue, task['task_id'])['status'], 'completed')
        self.assertEqual(read_knowledge(output)['entities']['scenes'][0]['label'], '第2幕新幕')

    def test_first_knowledge_publication_failure_hides_incomplete_legacy_files(self):
        info = self.root / 'info.txt'
        info.write_text('京剧《甲》\n第1幕开始\n剧情。')
        output = self.root / 'knowledge'
        params, items = prepare_knowledge({'info_path': str(info), 'play_root': str(self.photos), 'output_dir': str(output)})
        from core_modules.organize import knowledge_task
        original = knowledge_task.atomic_json
        failed = []
        def publish(path, value):
            if Path(path).name == 'knowledge_snapshot.json' and not failed:
                failed.append(True)
                raise OSError('incomplete first publication')
            return original(path, value)
        with patch.object(knowledge_task, 'atomic_json', publish):
            queue = TaskQueue(self.root / 'tasks', {'knowledge': knowledge_handler})
            self.addCleanup(queue.close, 10)
            task, _ = queue.submit('knowledge', params, items)
            self.assertEqual(wait_for(queue, task['task_id'])['status'], 'failed')
            self.assertTrue((output / 'entities.json').exists())
            with self.assertRaises(FileNotFoundError):
                read_knowledge(output)
            queue.resume(task['task_id'])
            self.assertEqual(wait_for(queue, task['task_id'])['status'], 'completed')
        self.assertEqual(read_knowledge(output)['entities']['scenes'][0]['label'], '第1幕开始')

    def test_background_manifest_hash_cancellation_and_resume_never_rehashes_completed(self):
        from app import organize_tasks
        prepared, manifest = prepare_organize({'input_dir': str(self.photos), 'output_dir': str(self.output),
                                               'db_root': str(self.gallery), 'keep_per_bucket': 1,
                                               'classify_role': False}, defer_hash=True)
        prepared['_manifest_items'] = manifest
        started, release = threading.Event(), threading.Event()
        hashed = []
        original_hash = organize_tasks.fingerprint
        def hashed_image(path, check=None):
            if check:
                hashed.append(path)
                if len(hashed) == 2:
                    started.set()
                    release.wait(5)
            return original_hash(path, check)
        with patch.object(organize_tasks, 'fingerprint', hashed_image), \
                patch.object(organize_tasks, 'organize_handler', lambda params: OrganizeTask(params, self.organizer)):
            queue = TaskQueue(self.root / 'tasks', {'organize': lambda params: organize_tasks.deferred_handler('organize', params)})
            self.addCleanup(queue.close, 10)
            task, _ = queue.submit('organize', prepared, [dict(item, stage='fingerprint') for item in manifest])
            task_id = task['task_id']
            try:
                self.assertTrue(started.wait(5))
                queue.cancel(task_id)
            finally:
                release.set()
            wait_for(queue, task_id, {'cancelled'})
            queue.resume(task_id)
            self.assertEqual(wait_for(queue, task_id)['status'], 'completed')
        self.assertEqual(hashed.count(manifest[0]['image']), 1)
        self.assertEqual(hashed.count(manifest[1]['image']), 2, 'cancelled partial hash is retried')
        self.assertEqual(self.calls.count('recognize'), 2)

    def test_restart_rejects_changed_model_and_original_version_resumes(self):
        reached, release = threading.Event(), threading.Event()
        def recognize(img, threshold):
            self.calls.append('recognize')
            reached.set()
            release.wait(5)
            return {}, []
        self.organizer._recognize_image = recognize
        queue, task_id = self.submit()
        try:
            self.assertTrue(reached.wait(5))
            queue.cancel(task_id)
        finally:
            release.set()
        wait_for(queue, task_id, {'cancelled'})
        before = len(self.calls)
        queue.close(10)
        self.encoder.model_signature = 'new-weights'
        replacement = TaskQueue(self.root / 'tasks', {'organize': lambda params: OrganizeTask(params, self.organizer)})
        self.addCleanup(replacement.close, 10)
        replacement.resume(task_id)
        rejected = wait_for(replacement, task_id)
        self.assertEqual(rejected['status'], 'failed')
        self.assertIn('execution_version_changed', rejected['error'])
        self.assertEqual(len(self.calls), before)
        self.assertFalse(self.output.exists())
        self.encoder.model_signature = 'organize-task-test'
        replacement.resume(task_id)
        self.assertEqual(wait_for(replacement, task_id)['status'], 'completed')
        self.assertEqual(self.calls.count('recognize'), 2)

    def test_unbound_legacy_checkpoints_cannot_silently_adopt_current_model(self):
        from core_modules.organize import organize_task
        def fail_copy(*args):
            raise OSError('stop before export')
        with patch.object(organize_task, 'copy_verified', fail_copy):
            queue, task_id = self.submit()
            self.assertEqual(wait_for(queue, task_id)['status'], 'failed')
        with queue._write() as connection:
            connection.execute("DELETE FROM checkpoints WHERE task_id=? AND name='execution_contract'", (task_id,))
        before = len(self.calls)
        queue.resume(task_id)
        final = wait_for(queue, task_id)
        self.assertIn('execution_version_unbound', final['error'])
        self.assertEqual(len(self.calls), before)
        self.assertFalse(self.output.exists())

    def test_legacy_callable_scene_handler_keeps_version_binding_on_resume(self):
        from core_modules.organize.task_versions import VersionedHandler
        entered, release = threading.Event(), threading.Event()
        config = {'model': 'remote-v1', 'preprocessing': 'preview-v1'}
        calls = []
        def handler(params):
            def process(item):
                calls.append(item['image'])
                entered.set()
                release.wait(5)
                return {'image': item['image'], 'label': 'scene'}
            return VersionedHandler(process, lambda rows: {'rows': rows}, lambda: dict(config))
        queue = TaskQueue(self.root / 'tasks', {'scene': handler})
        self.addCleanup(queue.close, 10)
        task, _ = queue.submit('scene', {}, [{'image': 'a'}, {'image': 'b'}])
        try:
            self.assertTrue(entered.wait(5))
            queue.cancel(task['task_id'])
        finally:
            release.set()
        wait_for(queue, task['task_id'], {'cancelled'})
        self.assertEqual(queue.get(task['task_id'])['execution_contract'], config)
        config['preprocessing'] = 'preview-v2'
        queue.resume(task['task_id'])
        self.assertIn('execution_version_changed', wait_for(queue, task['task_id'])['error'])
        self.assertEqual(calls, ['a'])
        config['preprocessing'] = 'preview-v1'
        queue.resume(task['task_id'])
        self.assertEqual(wait_for(queue, task['task_id'])['status'], 'completed')
        self.assertEqual(calls, ['a', 'b'])

    def test_gallery_cache_checks_content_and_encoder_even_when_mtime_restored(self):
        from core_modules.organize import face_db_builder
        anchor = self.gallery / 'actor' / 'anchor.jpg'
        anchor.parent.mkdir()
        anchor.write_bytes(b'old-bytes')
        before = anchor.stat()
        encoder = SimpleNamespace(model_signature='face-v1')
        system = SimpleNamespace(extractor=encoder)
        calls = []
        def library(builder):
            calls.append(anchor.read_bytes())
            return {'actor': [(np.array([len(calls), 0.]), str(anchor))]}
        with patch.object(face_db_builder.FaceDBBuilder, '_library_features', library):
            first = face_db_builder.get_cached_library_matrix(system, str(self.gallery))
            second = face_db_builder.get_cached_library_matrix(system, str(self.gallery))
            self.assertTrue(second[2])
            anchor.write_bytes(b'new-bytes')
            os.utime(anchor, ns=(before.st_atime_ns, before.st_mtime_ns))
            changed = face_db_builder.get_cached_library_matrix(system, str(self.gallery))
            self.assertFalse(changed[2])
            self.assertFalse(np.array_equal(first[0], changed[0]))
            encoder.model_signature = 'face-v2'
            self.assertFalse(face_db_builder.get_cached_library_matrix(system, str(self.gallery))[2])
        self.assertEqual(len(calls), 3)

    def test_client_contract_tracks_configuration_without_credentials(self):
        from core_modules.organize.llm_client import OpenAICompatClient
        from core_modules.organize.task_versions import client_identity
        client = OpenAICompatClient('https://user:password@example.invalid/api?token=secret',
                                   'private-api-key', 'served-v1')
        first = client_identity(client)
        saved = json.dumps(first)
        for secret in ('password', 'secret', 'private-api-key', 'user:'):
            self.assertNotIn(secret, saved)
        client.model = 'served-v2'
        self.assertNotEqual(client_identity(client), first)

    def test_version_change_during_inference_does_not_commit_mixed_item(self):
        def recognize(img, threshold):
            self.calls.append('recognize')
            self.encoder.model_signature = 'changed-mid-inference'
            return {}, []
        self.organizer._recognize_image = recognize
        queue, task_id = self.submit()
        final = wait_for(queue, task_id)
        self.assertEqual(final['status'], 'failed')
        self.assertIn('execution_version_changed', final['error'])
        rows = queue.item_page(task_id)
        self.assertTrue(all(row['status'] == 'pending' for row in rows if row['input']['stage'] == 'recognition'))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
