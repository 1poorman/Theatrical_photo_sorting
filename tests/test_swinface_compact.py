"""Compact checkpoint contract and optional real-weight compression checks.

RUN_SWINFACE_COMPACT_REAL=1 exports fresh artifacts into a temporary directory
and exercises reload against the original checkpoint. CUDA excludes CPU INT8.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_modules.face_recognition.swinface import SwinFaceAnalyzer
from core_modules.face_recognition.swinface.compact import (
    COMPACT_FORMAT, INT8_ENGINE, export_checkpoint, validate_compact_checkpoint,
)
from core_modules.face_recognition.swinface.inference import PREPROCESSING, UPSTREAM_COMMIT


def checkpoint_header(precision='fp32'):
    result = {'format': COMPACT_FORMAT, 'precision': precision, 'state_dict': {},
              'upstream_commit': UPSTREAM_COMMIT, 'preprocessing': PREPROCESSING}
    if precision == 'int8_dynamic':
        result['quantization'] = {'engine': INT8_ENGINE, 'scheme': 'per_channel_symmetric', 'modules': 'Linear'}
    return result


class CompactCheckpointTests(unittest.TestCase):
    def test_invalid_format_precision_and_preprocessing_fail_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unsupported.pt'
            for key, value in [('format', 'future-format'), ('precision', 'int4'),
                               ('preprocessing', 'bgr-unscaled'), ('upstream_commit', 'different'),
                               ('state_dict', None)]:
                with self.subTest(key=key):
                    checkpoint = checkpoint_header()
                    checkpoint[key] = value
                    torch.save(checkpoint, path)
                    with patch('core_modules.face_recognition.swinface.inference.RecognitionAgeModel') as builder:
                        with self.assertRaises(ValueError):
                            SwinFaceAnalyzer(path, device='cpu')
                        builder.assert_not_called()

    def test_int8_cuda_and_wrong_quantization_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'int8.pt'
            checkpoint = checkpoint_header('int8_dynamic')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'CPU only'):
                SwinFaceAnalyzer(path, device='cuda:0')
            checkpoint['quantization']['scheme'] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'quantization'):
                validate_compact_checkpoint(checkpoint)

    def test_export_cannot_clobber_source_or_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'output.pt'
            source.write_bytes(b'original source')
            output.write_bytes(b'previous output')
            for destination in (source, output):
                with self.assertRaises(FileExistsError):
                    export_checkpoint(source, destination)
            self.assertEqual(source.read_bytes(), b'original source')
            self.assertEqual(output.read_bytes(), b'previous output')
            with self.assertRaisesRegex(ValueError, 'precision'):
                export_checkpoint(source, Path(directory) / 'new.pt', 'int4')


@unittest.skipUnless(os.environ.get('RUN_SWINFACE_COMPACT_REAL') == '1', 'set RUN_SWINFACE_COMPACT_REAL=1 for local weights')
class RealCompactCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device = os.environ.get('SWINFACE_TEST_DEVICE', 'cpu')
        cls.source = ROOT / 'weights/swinface/checkpoint_step_79999_gpu_0.pt'
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.paths = sorted((ROOT / 'data/face_database').rglob('*.jpg'))
        if len(cls.paths) < 8:
            raise RuntimeError('eight real aligned faces are required')
        cls.paths = cls.paths[::max(1, len(cls.paths) // 8)][:8]
        cls.images = [cv2.imread(str(path)) for path in cls.paths]
        official = SwinFaceAnalyzer(cls.source, device=cls.device, batch_size=3)
        cls.reference = official.predict(cls.images)
        cls.source_sha = official.checkpoint_sha256
        del official
        cls.receipts = []

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('SWINFACE_COMPACT_TEST_REPORT'):
            destination = Path(os.environ['SWINFACE_COMPACT_TEST_REPORT'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({'device': cls.device, 'faces': len(cls.images),
                                               'variants': cls.receipts}, indent=2) + '\n')

    def check_export(self, precision, feature_tolerance, age_tolerance):
        path = Path(self.temp.name) / f'{precision}.pt'
        receipt = export_checkpoint(self.source, path, precision)
        model = SwinFaceAnalyzer(path, device=self.device, batch_size=3)
        actual = model.predict(self.images)
        np.testing.assert_allclose(actual['embeddings'], self.reference['embeddings'], atol=feature_tolerance, rtol=0)
        np.testing.assert_allclose(actual['ages'], self.reference['ages'], atol=age_tolerance, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(actual['embeddings'], axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(model.extract_features(self.images), actual['embeddings'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(model.estimate_ages(self.images), actual['ages'], atol=1e-5, rtol=0)
        self.assertEqual(actual['embeddings'].dtype, np.float32)
        self.assertEqual(actual['ages'].dtype, np.float32)
        self.assertEqual(model.source_sha256, self.source_sha)
        self.assertEqual(model.metadata()['parameters'], 32035547)
        self.assertLess(receipt['bytes'], self.source.stat().st_size * .15)
        self.assertTrue(all(not module.training for module in model.model.modules()))
        receipt.update({
            'passed': True, 'model': model.metadata(),
            'max_feature_absolute_difference': float(np.abs(actual['embeddings'] - self.reference['embeddings']).max()),
            'max_age_drift_years': float(np.abs(actual['ages'] - self.reference['ages']).max()),
        })
        self.receipts.append(receipt)
        return model, path

    def test_lossless_fp32_roundtrip(self):
        self.check_export('fp32', 2e-6, 2e-5)

    def test_fp16_roundtrip_and_runtime_dtype(self):
        model, _ = self.check_export('fp16', 0.002, 0.15)
        expected = torch.float16 if model.device.type == 'cuda' else torch.float32
        self.assertEqual(next(model.model.parameters()).dtype, expected)

    def test_dynamic_int8_roundtrip_cpu_auto_and_packed_weights(self):
        if self.device.startswith('cuda'):
            self.skipTest('dynamic INT8 kernels are CPU only')
        model, path = self.check_export('int8_dynamic', 0.1, 8.0)
        count = sum(isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in model.model.modules())
        self.assertGreater(count, 40)
        self.assertEqual(count, model.metadata()['quantized_linear_layers'])
        del model
        with patch('torch.cuda.is_available', return_value=True):
            reloaded = SwinFaceAnalyzer(path, device='auto', batch_size=3)
        self.assertEqual(reloaded.device.type, 'cpu')
        with self.assertRaisesRegex(ValueError, 'CPU only'):
            SwinFaceAnalyzer(path, device='cuda:0')


if __name__ == '__main__':
    unittest.main()
