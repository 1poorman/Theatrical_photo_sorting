"""Temporal burst constraints, chain drift, dark scenes and fixed-quota protection."""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.temporal_burst import BurstConfig, select_equal_quota, temporal_groups


class TemporalBurstTests(unittest.TestCase):
    def rows(self):
        return [{'image': f'{i}.jpg', 'stream_id': 'camera', 'order_trusted': True,
                 'captured_at': datetime(2026, 1, 1) + timedelta(seconds=i), 'embedding': [1, 0],
                 'quality': {'sharpness': 1000, 'contrast': 40, 'brightness': 10}} for i in range(5)]

    def test_trusted_near_identical_neighbours_group(self):
        groups, audit = temporal_groups(self.rows())
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0][1]), 5)
        self.assertEqual(audit['input_count'], 5)

    def test_same_filename_people_do_not_merge_across_scene_or_camera(self):
        rows = self.rows()
        rows[0]['confirmed_scene'] = 'first'
        rows[1]['confirmed_scene'] = 'second'
        rows[2]['stream_id'] = 'different-camera'
        groups, audit = temporal_groups(rows)
        self.assertGreaterEqual(len(groups), 3)
        self.assertEqual(audit['scene_conflict_rejections'], 1)

    def test_missing_time_or_features_keep_singletons(self):
        rows = self.rows()
        for row in rows:
            row['captured_at'] = None
        groups, _ = temporal_groups(rows)
        self.assertEqual(len(groups), 5)
        rows = self.rows()
        for row in rows:
            row['embedding'] = None
        self.assertEqual(len(temporal_groups(rows)[0]), 5)

    def test_chain_drift_does_not_merge_different_first_and_last_view(self):
        import math
        rows = self.rows()
        for i, row in enumerate(rows):
            row['embedding'] = [math.cos(i * .18), math.sin(i * .18)]
        groups, _ = temporal_groups(rows, BurstConfig(min_similarity=.97))
        self.assertGreater(len(groups), 1)

    def test_long_pause_or_span_breaks_group(self):
        rows = self.rows()
        rows[-1]['captured_at'] += timedelta(seconds=100)
        groups, _ = temporal_groups(rows)
        self.assertEqual([len(members) for _, members in groups], [4, 1])

    def test_equal_quota_and_protected_unique_frame(self):
        rows = self.rows()
        rows[-1]['must_keep'] = True
        rows[-1]['quality'] = {'sharpness': 1, 'contrast': 0}
        result = select_equal_quota(rows, 2)
        self.assertEqual(result['selection_count'], 2)
        self.assertIn(rows[-1]['image'], result['selected'])
        with self.assertRaises(ValueError):
            select_equal_quota(rows, 0)

    def test_dark_stage_brightness_does_not_penalize_equal_subject_detail(self):
        rows = self.rows()[:2]
        rows[0]['quality']['brightness'] = 5
        rows[1]['quality']['brightness'] = 150
        result = select_equal_quota(rows, 1)
        self.assertEqual(result['selected'], ['0.jpg'])

    def test_invalid_config_duplicate_images(self):
        with self.assertRaises(ValueError):
            BurstConfig(min_similarity=2)
        rows = self.rows()
        with self.assertRaises(ValueError):
            temporal_groups(rows + rows[:1])


if __name__ == '__main__':
    unittest.main()
