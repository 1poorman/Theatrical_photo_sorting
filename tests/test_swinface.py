"""SwinFace preprocessing plus optional official-checkpoint numerical parity.

RUN_SWINFACE_REAL=1 enables eight real crops and checks the selected branches
against the unmodified upstream ModelBox.forward computation in the vendored code.
"""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.face_recognition.swinface import SwinFaceAnalyzer


class SwinFacePreprocessingTests(unittest.TestCase):
    def test_bgr_rgb_scale_and_empty_shape(self):
        image = np.full((80, 160, 3), [0, 127, 255], dtype=np.uint8)
        output = SwinFaceAnalyzer.preprocess([image])
        self.assertEqual(tuple(output.shape), (1, 3, 112, 112))
        np.testing.assert_allclose(output[0, :, 50, 50].numpy(), [1, -1 / 255, -1], atol=1e-7)
        self.assertEqual(tuple(SwinFaceAnalyzer.preprocess([]).shape), (0, 3, 112, 112))

    def test_gray_alpha_and_invalid_inputs(self):
        gray = np.full((120, 80), 255, np.uint8)
        np.testing.assert_array_equal(SwinFaceAnalyzer.preprocess([gray]).numpy(), np.ones((1, 3, 112, 112)))
        rgba = np.full((112, 112, 4), [0, 127, 255, 0], dtype=np.uint8)
        np.testing.assert_array_equal(SwinFaceAnalyzer.preprocess([rgba]), SwinFaceAnalyzer.preprocess([rgba[:, :, :3]]))
        for value in (None, np.empty((0, 0, 3), np.uint8), np.zeros((112, 112, 3), np.float32), np.zeros((10, 10, 2), np.uint8)):
            with self.assertRaises(ValueError):
                SwinFaceAnalyzer.preprocess([value])

    def test_missing_weights_and_empty_batches_are_explicit(self):
        with self.assertRaises(FileNotFoundError):
            SwinFaceAnalyzer('/missing/swinface/checkpoint.pt', device='cpu')
        analyzer = SwinFaceAnalyzer.__new__(SwinFaceAnalyzer)
        analyzer.batch_size = 3
        analyzer.model = Mock()
        self.assertEqual(analyzer.extract_features([]).shape, (0, 512))
        self.assertEqual(analyzer.estimate_ages([]).shape, (0,))
        analyzer.model.backbone.assert_not_called()
        with self.assertRaises(ValueError):
            analyzer.predict([], recognition=False, age=False)


@unittest.skipUnless(os.environ.get('RUN_SWINFACE_REAL') == '1', 'set RUN_SWINFACE_REAL=1 for local weights')
class SwinFaceOfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        # Keep this reference tied to all official heads even when the app's
        # default checkpoint is an exported recognition + age model.
        cls.analyzer = SwinFaceAnalyzer(
            ROOT / 'weights/swinface/checkpoint_step_79999_gpu_0.pt',
            device=os.environ.get('SWINFACE_TEST_DEVICE', 'cpu'), batch_size=3,
        )
        paths = sorted((ROOT / 'data/face_database').rglob('*.jpg'))
        if len(paths) < 8:
            raise RuntimeError('real parity requires at least eight local face crops')
        cls.paths = paths[::max(1, len(paths) // 8)][:8]
        cls.images = [cv2.imread(str(path)) for path in cls.paths]

    def test_official_forward_batch_boundaries_and_paired_outputs(self):
        analyzer = self.analyzer
        with torch.inference_mode():
            reference = analyzer.model(analyzer.preprocess(self.images).to(analyzer.device))
        expected_features = analyzer.l2_normalize(reference['Recognition'].cpu().numpy())
        expected_ages = reference['Age'].flatten().cpu().numpy()
        actual = analyzer.predict(self.images)
        np.testing.assert_allclose(actual['embeddings'], expected_features, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(actual['ages'], expected_ages, atol=0.05, rtol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(actual['embeddings'], axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(analyzer.extract_features(self.images), actual['embeddings'], atol=1e-6)
        np.testing.assert_allclose(analyzer.estimate_ages(self.images), actual['ages'], atol=1e-5)
        one = analyzer.predict(self.images[:1])
        np.testing.assert_allclose(one['embeddings'][0], actual['embeddings'][0], atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(one['ages'][0], actual['ages'][0], atol=0.05, rtol=1e-4)
        self.assertTrue(all(not module.training for module in analyzer.model.modules()))
        receipt = {
            'passed': True, 'faces': len(self.images), 'paths': [str(path) for path in self.paths],
            'reference': 'vendored upstream ModelBox.forward, all official heads',
            'model': analyzer.metadata(),
            'maximum_feature_absolute_error': float(np.max(np.abs(actual['embeddings'] - expected_features))),
            'maximum_age_absolute_error_years': float(np.max(np.abs(actual['ages'] - expected_ages))),
            'age_estimates': actual['ages'].tolist(),
        }
        if os.environ.get('SWINFACE_TEST_REPORT'):
            destination = Path(os.environ['SWINFACE_TEST_REPORT'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
