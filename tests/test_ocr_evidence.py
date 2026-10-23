"""OCR schemas, selected-image scope, exact-content caching and failure isolation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from core_modules.organize.ocr_evidence import PaddleOCRProvider, validate_ocr


class OCRTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / 'text.jpg'
        Image.new('RGB', (100, 50), 'white').save(self.image)
        self.models = []
        for name in ('detector', 'recognizer'):
            model = self.root / name
            model.mkdir()
            (model / 'inference.pdiparams').write_bytes(b'local weights')
            self.models.append(model)
        self.observation = {'width': 100, 'height': 50, 'lines': [{'text': '可见文字', 'score': .9,
                            'polygon': [[1, 1], [90, 1], [90, 40], [1, 40]]}]}
        self.worker_calls = []

    def subprocess(self, command, **kwargs):
        if command[1] == '-c':
            return SimpleNamespace(stdout='{"paddleocr":"test-version"}', returncode=0)
        self.worker_calls.append(command)
        self.assertEqual(Path(command[command.index('--image') + 1]).name, 'image.png')
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'observation': self.observation, 'trace': {'test': True}}))
        return SimpleNamespace(stdout='', returncode=0)

    def provider(self):
        return PaddleOCRProvider(sys.executable, *self.models, self.root / 'cache')

    def test_strict_schema_rejects_wrong_types_and_out_of_bounds_evidence(self):
        self.assertTrue(validate_ocr(self.observation))
        self.assertTrue(validate_ocr({'width': 100, 'height': 50, 'lines': []}))
        for invalid in (dict(self.observation, width=True), dict(self.observation, unknown='guessed'),
                        dict(self.observation, lines=[{'text': 'x', 'score': float('nan'), 'polygon': []}]),
                        dict(self.observation, lines=[{'text': 'x', 'score': 1,
                             'polygon': [[-1, 0], [0, 0], [0, 0], [0, 0]]}])):
            self.assertFalse(validate_ocr(invalid))

    def test_unselected_images_never_run_and_reconstruction_reuses_verified_cache(self):
        with patch('core_modules.organize.ocr_evidence.subprocess.run', side_effect=self.subprocess):
            provider = self.provider()
            evidence, trace = provider.observe(self.image)
            self.assertIsNone(evidence)
            self.assertFalse(trace['called'])
            evidence, trace = provider.observe(self.image, requested=True)
            self.assertTrue(trace['called'])
            self.assertEqual(evidence['verified_by'], None)
            renamed = self.root / 'renamed.jpg'
            renamed.write_bytes(self.image.read_bytes())
            cached, trace = self.provider().observe(renamed, requested=True)
            self.assertTrue(trace['cache_hit'])
            self.assertEqual(cached['id'], evidence['id'])
            self.assertEqual(cached['ref'], 'renamed.jpg')
            self.assertEqual(len(self.worker_calls), 1)

    def test_changed_image_and_weights_invalidate_cache(self):
        with patch('core_modules.organize.ocr_evidence.subprocess.run', side_effect=self.subprocess):
            self.provider().observe(self.image, requested=True)
            Image.new('RGB', (100, 50), 'blue').save(self.image)
            self.provider().observe(self.image, requested=True)
            (self.models[0] / 'inference.pdiparams').write_bytes(b'new weights')
            self.provider().observe(self.image, requested=True)
            self.assertEqual(len(self.worker_calls), 3)

    def test_invalid_output_is_not_cached_and_source_change_is_rejected(self):
        with patch('core_modules.organize.ocr_evidence.subprocess.run', side_effect=self.subprocess):
            self.observation['width'] = False
            evidence, trace = self.provider().observe(self.image, requested=True)
            self.assertIsNone(evidence)
            self.assertEqual(trace['error'], 'invalid_ocr_result')
            self.assertEqual(list((self.root / 'cache').rglob('*.json')), [])
            self.observation['width'] = 100
            original = self.subprocess
            def changed(command, **kwargs):
                result = original(command, **kwargs)
                if command[1] != '-c':
                    Image.new('RGB', (100, 50), 'red').save(self.image)
                return result
            with patch('core_modules.organize.ocr_evidence.subprocess.run', side_effect=changed):
                evidence, trace = self.provider().observe(self.image, requested=True)
            self.assertIsNone(evidence)
            self.assertEqual(trace['error'], 'source_changed_during_ocr')

    def test_missing_weights_and_runtime_are_reported_before_requests(self):
        (self.models[1] / 'inference.pdiparams').unlink()
        with self.assertRaisesRegex(ValueError, 'ocr_local_weights_missing'):
            self.provider()

    def test_running_provider_rejects_changed_weights_before_cache_or_worker(self):
        with patch('core_modules.organize.ocr_evidence.subprocess.run', side_effect=self.subprocess):
            provider = self.provider()
            provider.observe(self.image, requested=True)
            (self.models[0] / 'inference.pdiparams').write_bytes(b'changed during task')
            evidence, trace = provider.observe(self.image, requested=True)
            self.assertIsNone(evidence)
            self.assertEqual(trace['error'], 'ocr_model_files_changed')
            self.assertFalse(trace['called'])
            self.assertEqual(len(self.worker_calls), 1)


if __name__ == '__main__':
    unittest.main()
