"""Age estimation contracts: crop mapping, invalid estimates and HTTP input isolation."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import face_age_api
from core_modules.face_recognition.age_estimation import age_fields, FaceAgeEstimator
from core_modules.face_recognition.scrfd.scrfd_det import FaceBBox


class FaceAgeTests(unittest.TestCase):
    def test_faces_remain_matched_after_filtering_and_sorting(self):
        detector = Mock()
        detector._detect_single.return_value = [
            FaceBBox(90, 20, 160, 100, 0.9),
            FaceBBox(0, 0, 2, 2, 0.9),
            FaceBBox(-5, 20, 65, 100, 0.8),
            FaceBBox(20, 20, 70, 80, 0.1),
        ]
        analyzer = Mock()
        analyzer.estimate_ages.return_value = np.array([31.25, 52.75])
        analyzer.metadata.return_value = {'name': 'SwinFace'}
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[:, :70] = [20, 30, 40]
        image[:, 90:] = [90, 100, 110]
        original = image.copy()
        report, annotated, crops = FaceAgeEstimator(detector=detector, analyzer=analyzer).estimate(image=image)
        self.assertEqual(report['face_count'], 2)
        self.assertEqual([face['estimated_age'] for face in report['faces']], [31.2, 52.8])
        self.assertEqual(report['faces'][0]['bbox'], [0, 20, 66, 101])
        self.assertEqual(report['bbox_coordinate_space'], 'analysis_image')
        self.assertEqual((report['image_width'], report['image_height']), (200, 120))
        np.testing.assert_array_equal(crops[0][50, 50], [20, 30, 40])
        np.testing.assert_array_equal(crops[1][50, 50], [90, 100, 110])
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(annotated, original))

    def test_empty_detection_does_not_load_age_weights(self):
        detector = Mock()
        detector._detect_single.return_value = []
        estimator = FaceAgeEstimator(checkpoint='/missing/checkpoint.pt', detector=detector)
        report, _, crops = estimator.estimate(image=np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(report['faces'], [])
        self.assertEqual(crops, [])
        self.assertIsNone(estimator._analyzer)

    def test_degenerate_landmarks_use_bbox_without_nan(self):
        detector = Mock()
        detector._detect_single.return_value = [FaceBBox(5, 5, 60, 60, 0.9, [(1, 1)] * 5)]
        analyzer = Mock()
        analyzer.estimate_ages.return_value = [42]
        report, _, _ = FaceAgeEstimator(detector=detector, analyzer=analyzer).estimate(
            image=np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(report['faces'][0]['alignment'], 'bbox_resize')

    def test_nonfinite_age_and_wrong_output_count_fail_explicitly(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                age_fields(value)
        self.assertIsNone(age_fields(-3)['estimated_age'])
        self.assertEqual(age_fields(135)['age_status'], 'out_of_range')
        analyzer = Mock()
        analyzer.estimate_ages.return_value = []
        with self.assertRaises(RuntimeError):
            FaceAgeEstimator(analyzer=analyzer).estimate_aligned([np.zeros((112, 112, 3), np.uint8)])


class AgeAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        Image.new('RGB', (80, 60), 'green').save(self.photos / '剧照.jpg')
        Image.new('RGB', (80, 60), 'blue').save(self.root / 'outside.jpg')
        app = FastAPI()
        app.include_router(face_age_api.router)
        self.client = TestClient(app)
        for key, value in [('TEMP_DIR', str(self.root / 'temp')), ('OUTPUT_DIR', str(self.root / 'outputs'))]:
            setting = patch.object(face_age_api, key, value)
            setting.start()
            self.addCleanup(setting.stop)

    def test_workflow_writes_unique_previews_and_preserves_source(self):
        original = (self.photos / '剧照.jpg').read_bytes()
        estimator = Mock()
        def estimate(**kwargs):
            self.assertEqual(Path(kwargs['image_path']), self.photos / '剧照.jpg')
            return ({'faces': [{'face_id': 1, **age_fields(44.3)}], 'face_count': 1, 'age_unit': 'years'},
                    np.zeros((60, 80, 3), np.uint8), [np.zeros((112, 112, 3), np.uint8)])
        estimator.estimate.side_effect = estimate
        with patch.object(face_age_api, 'age_estimator', return_value=estimator):
            first = self.client.post('/api/face/age', data={'input_dir': str(self.photos), 'name': '剧照.jpg'})
            second = self.client.post('/api/face/age', data={'input_dir': str(self.photos), 'name': '剧照.jpg'})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        data = first.json()['data']
        self.assertEqual(data['faces'][0]['estimated_age'], 44.3)
        self.assertTrue(Path(data['faces'][0]['face_image_path']).is_file())
        self.assertIsNotNone(cv2.imread(data['annotated_image_path']))
        self.assertNotEqual(data['annotated_image_path'], second.json()['data']['annotated_image_path'])
        self.assertEqual(original, (self.photos / '剧照.jpg').read_bytes())

    def test_rejects_traversal_symlink_missing_and_ambiguous_inputs(self):
        (self.photos / 'escape.jpg').symlink_to(self.root / 'outside.jpg')
        with patch.object(face_age_api, 'age_estimator') as model:
            for name in ('../outside.jpg', str(self.root / 'outside.jpg'), 'escape.jpg'):
                response = self.client.post('/api/face/age', data={'input_dir': str(self.photos), 'name': name})
                self.assertEqual(response.status_code, 403, response.text)
            self.assertEqual(self.client.post('/api/face/age', data={}).status_code, 400)
            self.assertEqual(self.client.post('/api/face/age', data={'input_dir': str(self.photos), 'name': 'missing.jpg'}).status_code, 404)
            self.assertEqual(self.client.post('/api/face/age', data={'input_dir': str(self.photos)},
                                             files={'image': ('x.jpg', b'bad')}).status_code, 400)
            model.assert_not_called()

    def test_bad_upload_is_422_and_temporary_files_are_cleaned(self):
        with patch.object(face_age_api, 'age_estimator', return_value=FaceAgeEstimator()):
            response = self.client.post('/api/face/age', files={'image': ('../../bad.jpg', b'not an image')})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(list((self.root / 'temp').iterdir()), [])

    def test_chinese_upload_name_and_model_failure_are_handled(self):
        image = io.BytesIO()
        Image.new('RGB', (80, 60)).save(image, format='PNG')
        for error in (FileNotFoundError('checkpoint missing'), RuntimeError('CUDA unavailable')):
            estimator = Mock()
            def estimate(**kwargs):
                self.assertIsNotNone(cv2.imread(kwargs['image_path']))
                raise error
            estimator.estimate.side_effect = estimate
            with patch.object(face_age_api, 'age_estimator', return_value=estimator):
                response = self.client.post('/api/face/age', files={'image': ('中文图.png', image.getvalue())})
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(list((self.root / 'temp').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
