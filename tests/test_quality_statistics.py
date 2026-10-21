"""Preserve corrected-baseline quality scores when reducing temporary memory."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.organize.burst_dedup import image_quality


def corrected_quality(image):
    """The original float64/NumPy reductions, kept as a numerical reference."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness, contrast = float(np.mean(gray)), float(gray.std())
    sharpness = min(1., np.log10(max(lap_var, 1.)) / 3.5)
    exposure = 1. if 40 <= brightness <= 200 else .7 if 25 <= brightness <= 225 else .3
    score = .5 * sharpness + .25 * exposure + .25 * min(1., contrast / 55.)
    return {'sharpness': round(float(lap_var), 1), 'brightness': round(brightness, 1),
            'contrast': round(contrast, 1), 'score': round(float(score), 4)}


class QualityStatisticsTests(unittest.TestCase):
    def test_uint8_scores_match_original_reductions(self):
        rng = np.random.default_rng(71)
        images = [rng.integers(0, 256, size=shape, dtype=np.uint8)
                  for shape in [(2, 3), (31, 27), (480, 640), (43, 87, 3)]]
        images += [np.full((25, 35), value, dtype=np.uint8)
                   for value in (0, 25, 39, 40, 200, 201, 225, 255)]
        images += [np.tile(np.arange(256, dtype=np.uint8), (64, 1)),
                   np.indices((63, 73)).sum(axis=0).astype(np.uint8) % 2 * 255,
                   rng.integers(249, 252, (61, 83), dtype=np.uint8)]
        images += [image[::2, ::2] for image in images]
        for index, image in enumerate(images):
            with self.subTest(index=index, shape=image.shape):
                self.assertEqual(image_quality(image), corrected_quality(image))

    def test_non_uint8_and_empty_inputs_keep_existing_behavior(self):
        image = np.linspace(0, 1, 117).reshape(9, 13)
        self.assertEqual(image_quality(image), corrected_quality(image))
        empty = {'sharpness': 0., 'brightness': 0., 'contrast': 0., 'score': 0.}
        self.assertEqual(image_quality(None), empty)
        self.assertEqual(image_quality(np.empty((0, 0), dtype=np.uint8)), empty)


if __name__ == '__main__':
    unittest.main()
