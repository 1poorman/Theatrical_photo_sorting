"""Contiguous change points, capture alignment, missing EXIF and repetitions."""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.scene_change_points import (
    ChangePointConfig, change_points, split_ordered_scenes, window_scores,
)


class ChangePointTests(unittest.TestCase):
    def test_return_to_earlier_visual_state_is_a_new_contiguous_segment(self):
        paths = [str(i) for i in range(36)]
        features = np.repeat(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32), 12, axis=0)
        result = split_ordered_scenes(paths, features, config=ChangePointConfig(window=3, penalty=.5))
        self.assertEqual(result['boundaries'], [12, 24])
        self.assertNotEqual(result['assignments']['0'], result['assignments']['24'])
        self.assertEqual(list(result['assignments']), paths)

    def test_capture_sort_keeps_path_and_embedding_alignment(self):
        ordered = [str(i) for i in range(24)]
        paths = list(reversed(ordered))
        base = datetime(2026, 1, 1)
        times = {path: base + timedelta(seconds=int(path)) for path in paths}
        values = [[1, 0] if int(path) < 12 else [0, 1] for path in paths]
        result = split_ordered_scenes(paths, values, times,
                                      ChangePointConfig(window=3, penalty=.5, sequence_order='capture'))
        self.assertEqual(result['ordered_paths'], ordered)
        self.assertEqual(result['boundaries'], [12])

    def test_missing_time_preserves_input_positions_and_disables_gap_evidence(self):
        paths = ['first', 'no-time', 'last']
        times = {'first': datetime(2026, 1, 1), 'last': datetime(2026, 1, 2)}
        result = split_ordered_scenes(paths, timestamps=times,
                                      config=ChangePointConfig(sequence_order='capture'))
        self.assertEqual(result['ordered_paths'], paths)
        self.assertEqual(result['time_gap_boundaries'], [])

    def test_untrusted_order_disables_time_and_plot_prior(self):
        paths = ['later', 'earlier']
        times = [datetime(2026, 1, 2), datetime(2026, 1, 1)]
        result = split_ordered_scenes(paths, timestamps=times,
                                      config=ChangePointConfig(sequence_order='capture', mode='performance'),
                                      order_trusted=False)
        self.assertEqual(result['ordered_paths'], paths)
        self.assertFalse(result['temporal_prior_enabled'])
        self.assertFalse(result['plot_order_prior_applied'])

    def test_gap_splits_without_features_and_bad_segment_does_not_break_others(self):
        paths = [str(i) for i in range(16)]
        base = datetime(2026, 1, 1)
        times = {path: base + timedelta(seconds=int(path) + (1000 if int(path) >= 8 else 0)) for path in paths}
        values = {path: [1, 0] if int(path) < 12 else [0, 1] for path in paths}
        values['0'] = None
        result = split_ordered_scenes(paths, values, times, ChangePointConfig(window=2, penalty=.5))
        self.assertEqual(result['boundaries'], [8, 12])
        self.assertEqual(result['diagnostics'][0]['subdivision'], 'unavailable_features')

    def test_minimum_segment_length_no_oversegmentation_for_uniform_sequence(self):
        points, _ = change_points(np.tile([1, 0], (100, 1)), ChangePointConfig(penalty=.01))
        self.assertEqual(points, [])
        random = np.random.default_rng(3).normal(size=(200, 16)).astype('float32')
        points, _ = change_points(random, ChangePointConfig(min_segment=8, penalty=.1))
        self.assertTrue(all(b - a >= 8 for a, b in zip([0] + points, points + [200])))

    def test_missing_feature_preserves_valid_runs_in_same_time_segment(self):
        paths = [str(i) for i in range(33)]
        values = {path: [1, 0] if int(path) < 17 else [0, 1] for path in paths}
        values['0'] = None
        result = split_ordered_scenes(paths, values, config=ChangePointConfig(window=3, penalty=.5))
        self.assertEqual(result['boundaries'], [17])
        self.assertEqual(set(result['assignments']), set(paths))

    def test_backward_clock_disables_time_gaps_without_reordering_images(self):
        paths = ['a', 'b', 'c']
        base = datetime(2026, 1, 1)
        result = split_ordered_scenes(paths, timestamps=[base, base - timedelta(days=1), base],
                                      order_trusted=True)
        self.assertEqual(result['ordered_paths'], paths)
        self.assertEqual(result['time_gap_boundaries'], [])
        self.assertFalse(result['temporal_prior_enabled'])

    def test_window_scores_match_direct_means(self):
        values = np.random.default_rng(8).normal(size=(30, 8)).astype('float32')
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        scores = window_scores(values, 3)
        for i, score in enumerate(scores, 1):
            a, b = values[max(0, i - 3):i].sum(axis=0), values[i:i + 3].sum(axis=0)
            expected = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
            self.assertAlmostEqual(float(score), float(expected), places=5)

    def test_invalid_inputs_and_config(self):
        for values in ([1, 2], [[0, 0]], [[float('nan'), 1]]):
            with self.assertRaises(ValueError):
                window_scores(values)
        with self.assertRaises(ValueError):
            split_ordered_scenes(['same', 'same'])
        with self.assertRaises(ValueError):
            split_ordered_scenes(['a', 'b'], [[1, 0]])
        with self.assertRaises(ValueError):
            ChangePointConfig(penalty=float('nan'))


if __name__ == '__main__':
    unittest.main()
