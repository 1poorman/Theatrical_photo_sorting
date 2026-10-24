"""Workspace/API integration with real files, no GPU or external model service."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import main
import server_ui
from app import workspace_api
from app import organize_tasks
from core_modules.organize.thumbnail_cache import ThumbnailCache


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='theatre-workspace-test-')
        self.root = Path(self.temp.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        task_root = patch.object(organize_tasks, 'TASK_ROOT', self.root / 'tasks')
        task_root.start()
        self.addCleanup(task_root.stop)
        self.addCleanup(organize_tasks.shutdown)
        cache = patch.object(main, 'FEATURE_CACHE_DIR', str(self.root / 'features'))
        cache.start()
        self.addCleanup(cache.stop)
        thumbnails = patch.object(workspace_api, 'thumbnail_cache', ThumbnailCache(self.root / 'thumbnails'))
        thumbnails.start()
        self.addCleanup(thumbnails.stop)
        self.client = TestClient(server_ui.app)
        self.names = [
            '01-20140304-戏剧场-京剧《天下归心》彩排 第1幕克段-陈少云饰颍考叔-苏岩.jpg',
            '02-20140304-戏剧场-京剧《天下归心》彩排 第2幕庆功-孟广禄饰郑庄公-苏岩.jpg',
            '03-20140304-戏剧场-京剧《天下归心》彩排-苏岩.jpg',
        ]
        for name, color in zip(self.names, ('#324233', '#744233', '#324277')):
            Image.new('RGB', (800, 400), color).save(self.photos / name)

    def tearDown(self):
        self.temp.cleanup()

    def post(self, endpoint, **data):
        response = self.client.post(endpoint, data=data)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['data']

    def knowledge(self):
        info = self.root / '剧目信息.txt'
        info.write_text('京剧《天下归心》\n第1幕克段\n颍考叔劝谏郑庄公。\n第2幕庆功\n郑庄公设宴庆功。', encoding='utf-8')
        knowledge = self.root / 'knowledge'
        self.post('/api/organize/play/ingest', info_path=info, play_root=self.root, output_dir=knowledge)
        return knowledge

    def test_routes_and_assets(self):
        for path in ['/ui', '/organize']:
            response = self.client.get(path)
            self.assertIn('data-view="review"', response.text)
            self.assertIn('/static/workspace.js', response.text)
            self.assertIn('href="/legacy"', response.text)
        self.assertEqual(self.client.get('/', follow_redirects=False).headers['location'], '/organize')
        self.assertEqual(self.client.get('/legacy', follow_redirects=False).headers['location'], '/legacy/ui')
        for path, alias, content in [('/legacy/ui', '/tools', server_ui.UI_HTML),
                                     ('/legacy/organize', '/organize/advanced', server_ui.LEGACY_ORGANIZE_HTML)]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, content)
            self.assertEqual(response.text, self.client.get(alias).text)
            self.assertIn('href="/legacy/ui"', response.text)
            self.assertIn('href="/legacy/organize"', response.text)
            self.assertIn('href="/organize">返回新版工作台', response.text)
        for path in ['/tools', '/organize/advanced', '/static/workspace.js', '/static/workspace.css']:
            self.assertEqual(self.client.get(path).status_code, 200)

    def wait_task(self, task_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = self.client.get('/api/organize/tasks/' + task_id).json()['data']
            if value['status'] in ('completed', 'completed_with_errors', 'failed', 'cancelled'):
                return value
            time.sleep(0.01)
        self.fail('task timeout')

    def test_durable_analysis_api_idempotency_partial_failure_and_checkpoint_pages(self):
        knowledge = self.knowledge()
        output = self.root / 'task-evidence.jsonl'
        body = {'kind': 'scene_reason', 'idempotency_key': 'api-test', 'params': {
            'input_dir': str(self.photos), 'knowledge_dir': str(knowledge), 'output_file': str(output)}}
        from core_modules.organize.scene_reasoner import SceneReasoner
        original = SceneReasoner.reason
        calls = []
        def reason(reasoner, image):
            calls.append(image)
            if image.endswith(self.names[1]):
                raise ValueError('test single image error')
            return original(reasoner, image)
        with patch.object(SceneReasoner, 'reason', reason):
            response = self.client.post('/api/organize/tasks', json=body)
            self.assertEqual(response.status_code, 202, response.text)
            task_id = response.json()['data']['task_id']
            final = self.wait_task(task_id)
            duplicate = self.client.post('/api/organize/tasks', json=body).json()['data']
            self.assertEqual(duplicate['task_id'], task_id)
            self.assertFalse(duplicate['created'])
        self.assertEqual(final['status'], 'completed_with_errors')
        self.assertEqual((final['succeeded'], final['failed']), (2, 1))
        self.assertEqual(len(calls), 3)
        self.assertEqual(final['result']['n_failed'], 1)
        rows = [json.loads(row) for row in output.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]['decision'], 'error')
        page = self.client.get(f'/api/organize/tasks/{task_id}/items?limit=2&offset=1').json()['data']['items']
        self.assertEqual([row['position'] for row in page], [1, 2])
        self.assertEqual(self.client.get(f'/api/organize/tasks/{task_id}/items?limit=201').status_code, 400)
        changed = dict(body, params=dict(body['params'], top_k=2))
        self.assertEqual(self.client.post('/api/organize/tasks', json=changed).status_code, 409)
        self.assertEqual(self.client.post(f'/api/organize/tasks/{task_id}/resume').status_code, 409)
        self.assertEqual(self.client.get('/api/organize/tasks/missing').status_code, 404)
        seeded = self.post('/api/organize/scene/review/seed', review_dir=self.root / 'review', evidence_file=output)
        self.assertEqual(seeded['total'], 3)

    def test_background_cancel_keeps_health_responsive_and_detects_changed_source_on_resume(self):
        knowledge = self.knowledge()
        body = {'kind': 'scene_reason', 'params': {'input_dir': str(self.photos), 'knowledge_dir': str(knowledge),
                                                'output_file': str(self.root / 'cancel.jsonl')}}
        from core_modules.organize.scene_reasoner import SceneReasoner
        original = SceneReasoner.reason
        started, release = threading.Event(), threading.Event()
        def reason(reasoner, image):
            started.set()
            if not release.wait(8):
                raise TimeoutError('test worker release')
            return original(reasoner, image)
        with patch.object(SceneReasoner, 'reason', reason):
            try:
                response = self.client.post('/api/organize/tasks', json=body)
                self.assertEqual(response.status_code, 202, response.text)
                task_id = response.json()['data']['task_id']
                self.assertTrue(started.wait(5))
                start = time.monotonic()
                self.assertEqual(self.client.get('/api/health').status_code, 200)
                self.assertLess(time.monotonic() - start, 0.5)
                cancelled = self.client.post(f'/api/organize/tasks/{task_id}/cancel').json()['data']
                self.assertEqual(cancelled['status'], 'cancel_requested')
            finally:
                release.set()
            final = self.wait_task(task_id)
        self.assertEqual((final['succeeded'], final['pending']), (1, 2))
        (self.photos / self.names[1]).write_bytes(b'changed source')
        resumed = self.client.post(f'/api/organize/tasks/{task_id}/resume')
        self.assertEqual(resumed.status_code, 202)
        final = self.wait_task(task_id)
        self.assertEqual((final['status'], final['succeeded'], final['failed']), ('completed_with_errors', 2, 1))
        rows = self.client.get(f'/api/organize/tasks/{task_id}/items').json()['data']['items']
        changed = next(row for row in rows if row['input'].get('stage') != 'fingerprint' and row['input']['image'].endswith(self.names[1]))
        self.assertEqual(changed['result']['error']['message'], 'source_content_changed')

    def test_knowledge_and_organize_background_apis(self):
        info = self.root / 'plot.txt'
        info.write_text('京剧《天下归心》\n第1幕克段\n颍考叔劝谏郑庄公。')
        response = self.client.post('/api/organize/tasks', json={'kind': 'knowledge', 'params': {
            'info_path': str(info), 'play_root': str(self.photos), 'output_dir': str(self.root / 'task-knowledge')}})
        self.assertEqual(response.status_code, 202, response.text)
        final = self.wait_task(response.json()['data']['task_id'])
        self.assertEqual(final['status'], 'completed')
        self.assertTrue((self.root / 'task-knowledge/entities.json').is_file())
        gallery = self.root / 'gallery'
        gallery.mkdir()
        from core_modules.organize.smart_organizer import SmartOrganizer
        with patch.object(main.srv, 'face_recognition_model', object()), \
                patch.object(main.srv, 'pose_model', None), \
                patch.object(main, '_get_scene_embedder', return_value=None), \
                patch.object(SmartOrganizer, '_recognize_image', return_value=({}, [])):
            response = self.client.post('/api/organize/tasks', json={'kind': 'organize', 'params': {
                'input_dir': str(self.photos), 'output_dir': str(self.root / 'task-organized'),
                'db_root': str(gallery), 'classify_role': False}})
            self.assertEqual(response.status_code, 202, response.text)
            final = self.wait_task(response.json()['data']['task_id'])
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['result']['total_input'], 3)
        self.assertEqual(final['result']['failed'], 0)
        self.assertTrue(Path(final['result']['report_path']).is_file())

    def test_scene_task_binds_filename_parser_for_restart(self):
        from core_modules.organize.task_artifacts import fingerprint
        knowledge = self.knowledge()
        params, _ = organize_tasks.prepare_scene({'input_dir': str(self.photos),
            'knowledge_dir': str(knowledge), 'output_file': str(self.root / 'evidence.jsonl')})
        handler = organize_tasks.scene_handler(params)
        parser = 'core_modules/organize/filename_parser.py'
        self.assertEqual(handler.model_contract()['code']['files'][parser], fingerprint(ROOT / parser))

    def test_organize_identity_registry_reaches_background_and_legacy_handlers(self):
        from core_modules.organize.smart_organizer import SmartOrganizer
        from test_identity_registry import registry_data
        gallery=self.root/'gallery'; gallery.mkdir()
        identity_file=self.root/'cast.json'
        identity_file.write_text(json.dumps(registry_data()))
        observed=[]
        def recognize(organizer,img,threshold):
            observed.append((organizer.identity_registry_hash,threshold,organizer.face_margin))
            return {'别名（饰旧角）':.9}, []
        with patch.object(main.srv,'face_recognition_model',object()), \
             patch.object(main.srv,'pose_model',None), \
             patch.object(main,'_get_scene_embedder',return_value=None), \
             patch.object(SmartOrganizer,'_recognize_image',recognize):
            response=self.client.post('/api/organize/tasks',json={'kind':'organize','params':{
                'input_dir':str(self.photos),'output_dir':str(self.root/'background-out'),
                'db_root':str(gallery),'classify_role':False,'identity_registry_file':str(identity_file),
                'face_threshold':.6,'face_margin':.12}})
            self.assertEqual(response.status_code,202,response.text)
            final=self.wait_task(response.json()['data']['task_id'])
            self.assertEqual(final['status'],'completed')
            report=json.loads(Path(final['result']['report_path']).read_text())
            self.assertTrue(all(r['identities'][0]['identity_id']=='actor-a' for r in report['records']))
            self.assertTrue(all(r['identities'][0]['role_name'] is None for r in report['records']))
            self.assertTrue(all('旧角' not in r['organized_name'] for r in report['records']))
            legacy=self.post('/api/organize/run',input_dir=self.photos,output_dir=self.root/'legacy-out',
                             db_root=gallery,classify_role=False,identity_registry_file=identity_file,
                             face_threshold=.6,face_margin=.12)
            self.assertEqual(legacy['failed'],0)
        self.assertTrue(observed)
        self.assertTrue(all(h and t==.6 and m==.12 for h,t,m in observed))

    def test_actor_album_api_uses_registry_filter_and_identity_margin(self):
        from types import SimpleNamespace
        from core_modules.organize.actor_view import ActorViewGenerator
        from core_modules.organize.face_db_builder import FaceDBBuilder
        from test_identity_registry import registry_data
        gallery = self.root / 'gallery'
        gallery.mkdir()
        registry = self.root / 'cast.json'
        registry.write_text(json.dumps(registry_data(), ensure_ascii=False))
        system = SimpleNamespace(detector=SimpleNamespace(_detect_single=lambda img: [object()]))
        library = {'别名（饰旧角）': [(np.array([.9, 0]), 'a')],
                   '同名（饰第三角）': [(np.array([.7, 0]), 'b')]}
        with patch.object(main.srv, 'face_recognition_model', system), \
             patch.object(ActorViewGenerator, '_library_features', return_value=library), \
             patch.object(FaceDBBuilder, '_face_feature', return_value=(np.array([1, 0]), None)):
            for margin, expected in ((.3, 0), (.1, len(self.names))):
                with self.subTest(margin=margin):
                    output = self.root / f'albums-{margin}'
                    data = self.post('/api/organize/actor_view', input_dir=self.photos,
                                     output_dir=output, db_root=gallery, identity_registry_file=registry,
                                     person='actor-a', threshold=.6, margin=margin, copy_images=False)
                    self.assertEqual(sum(data['persons'].values()), expected)
                    self.assertEqual(data['margin'], margin)
                    self.assertTrue(data['identity_registry_sha256'])
                    report = json.loads((output / 'actor_view_report.json').read_text())
                    for row in report['images']:
                        if expected:
                            self.assertEqual(row['identities'][0]['identity_id'], 'actor-a')
                            self.assertIsNone(row['identities'][0]['role_name'])
                        else:
                            self.assertEqual(row['face_decisions'][0]['reason'], 'ambiguous_identity')

    def test_durable_cluster_legacy_api_restart_and_human_actions(self):
        from types import SimpleNamespace
        face = SimpleNamespace(x=0, y=0, r=99, b=99, width=100, height=100, landmark=np.zeros((5, 2)))
        system = SimpleNamespace(
            detector=SimpleNamespace(model_signature='api-detector', _detect_single=lambda img: [face]),
            extractor=SimpleNamespace(model_signature='api-extractor',
                align_face=lambda img, landmark: np.full((160, 160, 3), img[0, 0], dtype=np.uint8),
                extract_features=lambda faces: np.tile(np.array([1., 0.]), (len(faces), 1)),
                l2_normalize=lambda values: np.asarray(values)))
        with patch.object(main.srv, 'face_recognition_model', system):
            response = self.client.post('/api/organize/face_cluster/scan', data={
                'input_dir': str(self.photos), 'output_dir': str(self.root / 'cluster-output')})
            self.assertEqual(response.status_code, 200, response.text)
            task_id = response.json()['data']['task_id']
            final = self.wait_task(task_id)
            self.assertEqual(final['status'], 'completed')
            self.assertNotIn(task_id, main.srv.progress_tracker)
            organize_tasks.shutdown()
            progress = self.client.get(f'/api/progress/{task_id}')
            self.assertEqual(progress.json()['status'], 'completed')
            result = self.client.get('/api/organize/face_cluster/result', params={'task_id': task_id}).json()['data']
            cluster_id = result['clusters'][0]['id']
            self.post('/api/organize/face_cluster/ignore', task_id=task_id, cluster_id=cluster_id)
            restored = self.client.get('/api/organize/face_cluster/result', params={'task_id': task_id}).json()['data']
            self.assertEqual(restored['clusters'][0]['status'], 'ignored')
            self.post('/api/organize/face_cluster/ignore', task_id=task_id, cluster_id=cluster_id, ignored='false')
            assigned = self.post('/api/organize/face_cluster/assign', task_id=task_id, cluster_id=cluster_id,
                                 person='演员（饰角色）', db_root=self.root / 'gallery')
            self.assertEqual(assigned['saved'], 3)
        self.assertEqual(self.client.get('/api/progress/missing-task').status_code, 404)

    def test_background_face_database_refresh_cancel_resume_and_legacy_dry_run(self):
        from core_modules.organize.face_database_task import FaceDatabaseTask
        from test_face_database_tasks import FaceDatabaseTasksTests
        fixture = FaceDatabaseTasksTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        entered, release = threading.Event(), threading.Event()
        original = FaceDatabaseTask.process
        calls = []
        def process(handler, item):
            result = original(handler, item)
            calls.append(item['image'])
            if len(calls) == 1:
                entered.set()
                release.wait(5)
            return result
        body = {'kind': 'face_database', 'params': {
            'organized_dirs': str(fixture.photos), 'db_root': str(fixture.gallery),
            'output_dir': str(fixture.report)}}
        with patch.object(main.srv, 'face_recognition_model', fixture.system), patch.object(FaceDatabaseTask, 'process', process):
            try:
                response = self.client.post('/api/organize/tasks', json=body)
                self.assertEqual(response.status_code, 202, response.text)
                task_id = response.json()['data']['task_id']
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.client.get('/api/health').status_code, 200)
                self.assertEqual(self.client.get('/api/organize/tasks/' + task_id).json()['data']['kind'], 'face_database')
                # A separate gallery must still conflict on the report root.
                other = dict(body, params=dict(body['params'], db_root=str(self.root / 'other-gallery')))
                self.assertEqual(self.client.post('/api/organize/tasks', json=other).status_code, 409)
                self.client.post(f'/api/organize/tasks/{task_id}/cancel')
            finally:
                release.set()
            self.assertEqual(self.wait_task(task_id)['status'], 'cancelled')
            self.assertFalse(fixture.gallery.exists())
            organize_tasks.shutdown()
            self.assertEqual(self.client.post(f'/api/organize/tasks/{task_id}/resume').status_code, 202)
            final = self.wait_task(task_id)
            self.assertEqual(final['status'], 'completed', final)
            self.assertEqual(final['result']['added'], 2)
            self.assertEqual(len(calls), 3)
            self.assertIsNotNone(final['execution_contract'])
            downloaded = self.client.get('/api/workspace/artifact', params={
                'output_dir': str(fixture.report), 'name': 'face_database_report.json'})
            self.assertEqual(downloaded.status_code, 200)
            self.assertEqual(downloaded.json()['added'], 2)
            legacy = self.post('/api/organize/build_face_database', organized_dirs=fixture.photos,
                               db_root=self.root / 'legacy-dry-gallery', dry_run='true')
            self.assertEqual(legacy['added'], 2)
            self.assertFalse((self.root / 'legacy-dry-gallery').exists())

    def test_legacy_cluster_result_waits_for_publication_even_at_full_item_progress(self):
        from core_modules.organize.face_cluster_task import FaceClusterTask
        entered, release = threading.Event(), threading.Event()
        original = FaceClusterTask.finish
        def finish(handler, rows):
            entered.set()
            release.wait(5)
            return original(handler, rows)
        with patch.object(main.srv, 'face_recognition_model', object()), \
                patch('core_modules.organize.face_cluster.FaceClusterBuilder.scan_image', return_value=[]), \
                patch.object(FaceClusterTask, 'finish', finish):
            try:
                response = self.client.post('/api/organize/tasks', json={'kind': 'face_cluster', 'params': {
                    'input_dir': str(self.photos), 'output_dir': str(self.root / 'waiting-cluster')}})
                task_id = response.json()['data']['task_id']
                self.assertTrue(entered.wait(5))
                result = self.client.get('/api/organize/face_cluster/result', params={'task_id': task_id}).json()['data']
                self.assertEqual(result['status'], 'running')
                self.assertLess(result['progress'], 100)
                self.assertNotIn('clusters', result)
            finally:
                release.set()
            self.assertEqual(self.wait_task(task_id)['status'], 'completed')

    def test_listing_matches_pipeline_scope_and_handles_metadata(self):
        child = self.photos / 'nested'
        child.mkdir()
        Image.new('RGB', (20, 20)).save(child / 'nested.jpg')
        (self.photos / '.env').write_text('private')
        outside = self.root / 'outside.jpg'
        Image.new('RGB', (20, 20)).save(outside)
        (self.photos / 'escape.jpg').symlink_to(outside)
        result = self.client.get('/api/workspace/photos', params={'input_dir': str(self.photos), 'limit': 2}).json()['data']
        self.assertEqual(result['total'], 3)
        self.assertTrue(result['truncated'])
        self.assertEqual(len(result['photos']), 2)
        self.assertEqual(result['photos'][0]['metadata']['people'], [['陈少云', '颍考叔']])
        self.assertEqual(self.client.get('/api/workspace/photos', params={'input_dir': '/not-a-directory'}).status_code, 400)

    def test_thumbnails_resize_and_reject_escape_and_non_images(self):
        params = {'input_dir': str(self.photos), 'name': self.names[0], 'size': 160}
        response = self.client.get('/api/workspace/thumbnail', params=params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(response.content)).size, (160, 80))
        for name in ['../outside.jpg', '.env', str(self.root / 'outside.jpg')]:
            response = self.client.get('/api/workspace/thumbnail', params={**params, 'name': name})
            self.assertEqual(response.status_code, 403)
        (self.photos / 'bad.jpg').write_text('not an image')
        self.assertEqual(self.client.get('/api/workspace/thumbnail', params={**params, 'name': 'bad.jpg'}).status_code, 422)

    def test_thumbnail_applies_exif_orientation(self):
        im = Image.new('RGB', (800, 400))
        exif = im.getexif()
        exif[274] = 6
        im.save(self.photos / 'rotated.jpg', exif=exif)
        response = self.client.get('/api/workspace/thumbnail', params={'input_dir': str(self.photos), 'name': 'rotated.jpg', 'size': 160})
        self.assertEqual(Image.open(io.BytesIO(response.content)).size, (80, 160))

    def test_thumbnail_cache_hits_revalidate_and_do_not_decode(self):
        params = {'input_dir': str(self.photos), 'name': self.names[0], 'size': 160}
        first = self.client.get('/api/workspace/thumbnail', params=params)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers['cache-control'], 'private, no-cache')
        with patch('core_modules.organize.thumbnail_cache.Image.open', side_effect=AssertionError('unexpected decode')):
            cached = self.client.get('/api/workspace/thumbnail', params=params)
            self.assertEqual(cached.content, first.content)
            conditional = self.client.get('/api/workspace/thumbnail', params=params,
                                          headers={'If-None-Match': first.headers['etag']})
            self.assertEqual(conditional.status_code, 304)
            self.assertEqual(conditional.content, b'')
        self.assertEqual(len(list((self.root / 'thumbnails').glob('*.thumb'))), 1)

    def test_thumbnail_content_rewrite_invalidates_etag_even_with_restored_mtime(self):
        path = self.photos / 'mutable.bmp'
        Image.new('RGB', (800, 400), 'red').save(path)
        params = {'input_dir': str(self.photos), 'name': path.name, 'size': 160}
        first = self.client.get('/api/workspace/thumbnail', params=params)
        stat = path.stat()
        Image.new('RGB', (800, 400), 'blue').save(path)
        import os
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(path.stat().st_size, stat.st_size)
        second = self.client.get('/api/workspace/thumbnail', params=params,
                                 headers={'If-None-Match': first.headers['etag']})
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first.headers['etag'], second.headers['etag'])
        self.assertNotEqual(first.content, second.content)
        larger = self.client.get('/api/workspace/thumbnail', params={**params, 'size': 320})
        self.assertNotEqual(larger.headers['etag'], second.headers['etag'])
        self.assertEqual(Image.open(io.BytesIO(larger.content)).size, (320, 160))

    def test_thumbnail_corruption_rebuilds_and_pruning_only_removes_cache_entries(self):
        cache = workspace_api.thumbnail_cache
        cache.max_entries = 1
        params = {'input_dir': str(self.photos), 'name': self.names[0], 'size': 160}
        first = self.client.get('/api/workspace/thumbnail', params=params)
        entry = next(cache.root.glob('*.thumb'))
        entry.write_bytes(b'broken')
        self.assertEqual(self.client.get('/api/workspace/thumbnail', params=params).content, first.content)
        unrelated = cache.root / 'keep.txt'
        unrelated.write_text('keep')
        outside = self.root / 'outside.thumb'
        outside.write_text('keep outside')
        (cache.root / ('f' * 64 + '.thumb')).symlink_to(outside)
        second = self.client.get('/api/workspace/thumbnail', params={**params, 'name': self.names[1]})
        self.assertEqual(second.status_code, 200)
        regular = [p for p in cache.root.glob('*.thumb') if not p.is_symlink()]
        self.assertEqual(len(regular), 1)
        self.assertEqual(unrelated.read_text(), 'keep')
        self.assertEqual(outside.read_text(), 'keep outside')
        cache.max_bytes = regular[0].stat().st_size
        self.client.get('/api/workspace/thumbnail', params={**params, 'name': self.names[2]})
        self.assertLessEqual(sum(p.stat().st_size for p in cache.root.glob('*.thumb') if not p.is_symlink()), cache.max_bytes)

    def test_thumbnail_changed_during_decode_is_not_cached(self):
        cache = workspace_api.thumbnail_cache
        path = self.photos / self.names[0]
        original = cache._render
        def mutate(source, size):
            jpeg = original(source, size)
            Image.new('RGB', (800, 400), 'red').save(path)
            return jpeg
        with patch.object(cache, '_render', side_effect=mutate):
            response = self.client.get('/api/workspace/thumbnail', params={
                'input_dir': str(self.photos), 'name': path.name, 'size': 160})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(list(cache.root.glob('*.thumb')), [])

    def test_thumbnail_copies_share_entries_and_concurrent_publication_stays_bounded(self):
        from concurrent.futures import ThreadPoolExecutor
        import shutil
        cache = workspace_api.thumbnail_cache
        original = self.photos / self.names[0]
        clone = self.photos / 'renamed.jpg'
        shutil.copyfile(original, clone)
        first = cache.preview(original, 160)
        with patch.object(cache, '_render', side_effect=AssertionError('copy should hit cache')):
            self.assertEqual(cache.preview(clone, 160), first)
        cache.max_entries = 2
        with ThreadPoolExecutor(max_workers=4) as pool:
            previews = list(pool.map(lambda size: cache.preview(original, size), (320, 480, 640, 960)))
        self.assertEqual(len({key for _, key in previews}), 4)
        for jpeg, _ in previews:
            with Image.open(io.BytesIO(jpeg)) as image:
                image.verify()
        self.assertLessEqual(len(list(cache.root.glob('*.thumb'))), 2)
        self.assertEqual([p.name for p in cache.root.iterdir() if not p.name.endswith('.thumb')], ['.publication.lock'])

    def test_artifact_download_restrictions(self):
        (self.root / 'scene_labels.json').write_text('{}')
        response = self.client.get('/api/workspace/artifact', params={'output_dir': str(self.root), 'name': 'scene_labels.json'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers['content-disposition'])
        for name in ['.env', '../scene_labels.json', 'entities.json', '../face_database_report.json']:
            self.assertEqual(self.client.get('/api/workspace/artifact', params={'output_dir': str(self.root), 'name': name}).status_code, 403)
        (self.photos / 'scene_labels.json').symlink_to(self.root / 'scene_labels.json')
        self.assertEqual(self.client.get('/api/workspace/artifact', params={'output_dir': str(self.photos), 'name': 'scene_labels.json'}).status_code, 403)

    def test_real_knowledge_reason_review_export_flow(self):
        knowledge = self.knowledge()
        data = self.client.get('/api/organize/play/knowledge', params={'output_dir': str(knowledge)}).json()['data']
        self.assertEqual(data['scene_options'][0], {'scene_id': 'act-01', 'label': '第1幕克段'})
        evidence = self.root / 'scene_evidence.jsonl'
        result = self.post('/api/organize/scene/reason', input_dir=self.photos, knowledge_dir=knowledge, output_file=evidence)
        self.assertEqual(result['n_images'], 3)
        rows = [json.loads(line) for line in evidence.read_text().splitlines()]
        self.assertTrue(all(Path(row['image']).is_file() for row in rows))
        self.assertTrue(all(row['retrieval_mode'] == 'lexical' for row in rows))
        self.assertTrue(any('劝谏' in item['quote'] for row in rows for item in row['evidence']))
        self.assertTrue(all('scene_id' in alt for row in rows for alt in row['alternatives']))
        review = self.root / 'review'
        self.post('/api/organize/scene/review/seed', review_dir=review, evidence_file=evidence)
        self.post('/api/organize/scene/review', review_dir=review, image=rows[0]['image'], action='accept')
        self.post('/api/organize/scene/review', review_dir=review, image=rows[1]['image'], action='modify', scene_id='act-01', label='第1幕克段')
        self.post('/api/organize/scene/review', review_dir=review, image=rows[2]['image'], action='reject')
        # Re-analysis must not overwrite a human correction.
        self.post('/api/organize/scene/review/seed', review_dir=review, evidence_file=evidence)
        result = self.post('/api/organize/scene/review/export', review_dir=review)
        self.assertEqual(result['n_labels'], 2)
        labels = json.loads(Path(result['path']).read_text())
        self.assertEqual(labels[rows[1]['image']], '第1幕克段')
        self.assertNotIn(rows[2]['image'], labels)

    def test_visual_encoder_receives_existing_absolute_image_paths(self):
        knowledge = self.knowledge()
        paths = []

        class Encoder:
            model_signature = 'workspace-test'

            def extract_features(self, path):
                paths.append(path)
                assert Path(path).is_absolute() and Path(path).is_file()
                return np.array([1., 0., 0.])

            def encode_text(self, text):
                return np.array([1., 0., 0.])

        with patch.object(main, '_get_scene_embedder', return_value=Encoder()):
            result = self.post('/api/organize/scene/reason', input_dir=self.photos, knowledge_dir=knowledge,
                               output_file=self.root / 'visual.jsonl', use_embedder='true')
        self.assertEqual(len(paths), 3)
        self.assertEqual(result['n_visual'], 3)

    def test_duplicate_content_reuses_vectors_and_keeps_all_evidence_rows(self):
        knowledge = self.knowledge()
        data = (self.photos / self.names[0]).read_bytes()
        for name in self.names[1:]:
            (self.photos / name).write_bytes(data)
        calls = {'image': 0, 'text': 0}

        class Encoder:
            model_signature = 'workspace-cache-test'

            def extract_features(self, path):
                calls['image'] += 1
                return np.array([1., 0., 0.])

            def encode_text(self, text):
                calls['text'] += 1
                return np.array([1., 0., 0.])

        output = self.root / 'visual.jsonl'
        with patch.object(main, '_get_scene_embedder', side_effect=[Encoder(), Encoder()]):
            for _ in range(2):
                result = self.post('/api/organize/scene/reason', input_dir=self.photos,
                                   knowledge_dir=knowledge, output_file=output, use_embedder='true')
                self.assertEqual(result['n_visual'], 3)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual({row['image'] for row in rows},
                                 {str(self.photos / name) for name in self.names})
        self.assertEqual(calls, {'image': 1, 'text': 2})

    def test_cluster_failure_is_terminal_and_queryable(self):
        task_id = 'workspace-test-failure'
        main.srv.progress_tracker[task_id] = {'progress': 1}
        try:
            with patch('core_modules.organize.face_cluster.FaceClusterBuilder.scan', side_effect=ValueError('test scan error')):
                main._run_face_cluster_scan(task_id, str(self.photos), str(self.root), .5, 1600)
            self.assertEqual(main.srv.progress_tracker[task_id]['status'], 'failed')
            self.assertEqual(self.client.get('/api/organize/face_cluster/result', params={'task_id': task_id}).status_code, 500)
        finally:
            main.srv.progress_tracker.pop(task_id, None)

    def test_reanalysis_preserves_legacy_filename_review(self):
        from core_modules.organize.review_queue import ReviewQueue
        queue = ReviewQueue(str(self.root / 'legacy-review'))
        queue.seed([{'image': self.names[0], 'scene_id': 'act-01', 'label': '第1幕克段'}])
        queue.decide(self.names[0], 'modify', label='第2幕庆功')
        queue.seed([{'image': str(self.photos / self.names[0]), 'scene_id': 'act-01', 'label': '第1幕克段'}])
        self.assertEqual(len(queue.records()), 1)
        self.assertEqual(queue.final_labels(), {self.names[0]: '第2幕庆功'})

    def test_review_metadata_and_bounded_details_preserve_paths_revisions_and_evidence(self):
        from core_modules.organize.review_queue import ReviewQueue
        queue = ReviewQueue(str(self.root / 'review'))
        paths = [str(self.photos / name) for name in self.names]
        queue.seed([{'image': image, 'label': 'candidate', 'evidence': [{'quote': 'long evidence'}]}
                    for image in paths])
        queue.decide(paths[0], 'modify', label='confirmed', expected_revision=0)
        index = self.client.get('/api/organize/scene/review/metadata',
                                params={'review_dir': queue.root}).json()['data']
        self.assertEqual(index['summary'], {'pending': 2, 'approved': 1, 'rejected': 0})
        self.assertEqual(len(index['records']), 3)
        self.assertTrue(all(row['_metadata_only'] and 'evidence' not in row for row in index['records']))
        confirmed = next(row for row in index['records'] if row['image'] == paths[0])
        self.assertEqual((confirmed['label'], confirmed['review_revision']), ('confirmed', 1))
        response = self.client.post('/api/organize/scene/review/records',
                                    json={'review_dir': queue.root, 'images': [paths[0], paths[0], 'missing']})
        self.assertEqual(response.status_code, 200, response.text)
        details = response.json()['data']['records']
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]['evidence'], [{'quote': 'long evidence'}])
        self.assertEqual(details[0]['review_revision'], 1)
        for images in ([paths[0]] * 201, [None], 'not a list'):
            self.assertEqual(self.client.post('/api/organize/scene/review/records',
                json={'review_dir': queue.root, 'images': images}).status_code, 400)
        # A bound legacy basename must never be returned for a different source.
        legacy = ReviewQueue(str(self.root / 'legacy-review'))
        legacy.seed([{'image': self.names[0], 'label': 'legacy'}])
        legacy.seed([{'image': paths[0], 'label': 'bound'}])
        self.assertEqual(legacy.records_for([paths[0]])['records'][0]['source_image'], paths[0])
        self.assertEqual(legacy.records_for([str(self.root / 'other' / self.names[0])])['records'], [])

    def test_review_revision_conflict_returns_409_and_single_record_refresh(self):
        from core_modules.organize.review_queue import ReviewQueue
        queue = ReviewQueue(str(self.root / 'review'))
        image = str(self.photos / self.names[0])
        queue.seed([{'image': image, 'label': 'candidate'}])
        params = {'review_dir': queue.root, 'image': image}
        loaded = self.client.get('/api/organize/scene/review', params=params).json()['data']['record']
        self.assertEqual(loaded['review_revision'], 0)
        data = {**params, 'action': 'modify', 'label': 'first', 'expected_revision': 0}
        first = self.client.post('/api/organize/scene/review', data=data)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()['data']['record']['review_revision'], 1)
        stale = self.client.post('/api/organize/scene/review', data={**data, 'label': 'stale'})
        self.assertEqual(stale.status_code, 409)
        latest = self.client.get('/api/organize/scene/review', params=params).json()['data']['record']
        self.assertEqual(latest['label'], 'first')
        self.assertEqual(latest['review_revision'], 1)
        self.assertEqual(len(queue.revisions()), 1)
        self.assertEqual(self.client.get('/api/organize/scene/review', params={**params, 'image': 'missing'}).status_code, 404)
        self.assertEqual(self.client.get('/api/organize/scene/review', params={'review_dir': queue.root, 'limit': -1}).status_code, 400)

    def test_deferred_review_saves_and_refreshes_without_full_snapshot_then_exports_latest(self):
        from core_modules.organize.review_queue import ReviewQueue
        queue = ReviewQueue(str(self.root / 'review'))
        image = str(self.photos / self.names[0])
        queue.seed([{'image': image, 'label': 'candidate'}])
        before = Path(queue.queue_path).read_bytes()
        data = {'review_dir': queue.root, 'image': image, 'action': 'modify',
                'label': 'confirmed', 'expected_revision': 0, 'defer_snapshot': 'true'}
        with patch('core_modules.organize.review_store.atomic_rows',
                   side_effect=AssertionError('full_queue_snapshot_in_single_record_request')):
            saved = self.client.post('/api/organize/scene/review', data=data)
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual(saved.json()['data']['record']['review_revision'], 1)
            self.assertEqual(self.client.post('/api/organize/scene/review', data=data).status_code, 409)
            latest = self.client.get('/api/organize/scene/review', params={
                'review_dir': queue.root, 'image': image, 'defer_snapshot': 'true'})
            self.assertEqual(latest.json()['data']['record']['label'], 'confirmed')
            index = self.client.get('/api/organize/scene/review/metadata',
                                    params={'review_dir': queue.root})
            self.assertEqual(index.json()['data']['summary']['approved'], 1)
            details = self.client.post('/api/organize/scene/review/records',
                json={'review_dir': queue.root, 'images': [image]})
            self.assertEqual(details.json()['data']['records'][0]['review_revision'], 1)
        self.assertEqual(Path(queue.queue_path).read_bytes(), before)
        result = self.post('/api/organize/scene/review/export', review_dir=queue.root)
        self.assertEqual(json.loads(Path(result['path']).read_text()), {image: 'confirmed'})
        self.assertEqual(json.loads(Path(queue.queue_path).read_text())['label'], 'confirmed')
        self.assertEqual(len(queue.revisions()), 1)


if __name__ == '__main__':
    unittest.main()
