"""Workspace/API integration with real files, no GPU or external model service."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import main
import server_ui


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='theatre-workspace-test-')
        self.root = Path(self.temp.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        self.client = TestClient(server_ui.app)
        self.names = [
            '01-20140304-戏剧场-京剧《天下归心》彩排 第1幕克段-陈少云饰颍考叔-苏岩.jpg',
            '02-20140304-戏剧场-京剧《天下归心》彩排 第2幕庆功-孟广禄饰郑庄公-苏岩.jpg',
            '03-20140304-戏剧场-京剧《天下归心》彩排-苏岩.jpg',
        ]
        for name in self.names:
            Image.new('RGB', (800, 400), '#324233').save(self.photos / name)

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

    def test_artifact_download_restrictions(self):
        (self.root / 'scene_labels.json').write_text('{}')
        response = self.client.get('/api/workspace/artifact', params={'output_dir': str(self.root), 'name': 'scene_labels.json'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers['content-disposition'])
        for name in ['.env', '../scene_labels.json', 'entities.json']:
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

        with patch.object(main, '_get_organize_models', return_value=(Encoder(), None)):
            result = self.post('/api/organize/scene/reason', input_dir=self.photos, knowledge_dir=knowledge,
                               output_file=self.root / 'visual.jsonl', use_embedder='true')
        self.assertEqual(len(paths), 3)
        self.assertEqual(result['n_visual'], 3)

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


if __name__ == '__main__':
    unittest.main()
