"""A failed optional GPU decoder must preserve valid neighboring inputs."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_dali_preprocessing import DaliImageEmbedder, ImageEmbedder


class DaliFallbackTests(unittest.TestCase):
    def test_failed_decoder_retries_paths_in_order_and_preserves_batch_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for i in range(11):
                path = root / f'{i}.jpg'
                Image.new('RGB', (12, 8), (i, 0, 0)).save(path)
                paths.append(str(path))
            broken = root / 'broken.jpg'
            broken.write_bytes(b'bad')
            mixed = paths[:4] + [str(broken)] + paths[4:]
            model = DaliImageEmbedder.__new__(DaliImageEmbedder)
            model._dali_lock = threading.RLock()
            model._dali_size = {'width': 8, 'height': 8}
            model.dali_statistics = {'decoded': 0, 'fallback': 0, 'decoder_failures': 0}
            batches, retried = [], []
            class BrokenPipeline:
                def feed_input(self, name, values):
                    batches.append(len(values))
                    raise RuntimeError('simulated decoder failure')
            model._pipeline = lambda: BrokenPipeline()
            def fallback(_, path):
                retried.append(path)
                return None if path == str(broken) else np.array([paths.index(path), 1.], dtype='float32')
            with patch.object(ImageEmbedder, 'extract_features', fallback):
                vectors, valid = model.extract_features_batch(mixed, sub_batch_size=3)
            self.assertEqual(valid, paths)
            np.testing.assert_array_equal(vectors[:, 0], np.arange(len(paths)))
            self.assertCountEqual(retried, mixed)
            self.assertTrue(all(size <= 3 for size in batches))
            self.assertEqual(model.dali_statistics, {'decoded': 0, 'fallback': 12, 'decoder_failures': 11})

    def test_upscaling_uses_reference_without_attempting_gpu_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'small.jpg'
            Image.new('RGB', (12, 8), 'red').save(path)
            model = DaliImageEmbedder.__new__(DaliImageEmbedder)
            model._dali_lock = threading.RLock()
            model._dali_size = {'width': 16, 'height': 16}
            model.dali_statistics = {'decoded': 0, 'fallback': 0, 'decoder_failures': 0}
            def unexpected():
                raise AssertionError('upscaling must use Pillow')
            model._pipeline = unexpected
            expected = np.array([1., 2.], dtype='float32')
            with patch.object(ImageEmbedder, 'extract_features', return_value=expected):
                result, valid = model.extract_features_batch([str(path)])
            self.assertEqual(valid, [str(path)])
            np.testing.assert_array_equal(result[0], expected)
            self.assertEqual(model.dali_statistics, {'decoded': 0, 'fallback': 1, 'decoder_failures': 0})


if __name__ == '__main__':
    unittest.main()
