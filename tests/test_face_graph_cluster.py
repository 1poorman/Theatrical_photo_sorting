"""Physical-face constraints, transitive merging, determinism and graph input guards."""
from pathlib import Path
import sys
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.face_graph_cluster import (
    FaceGraphConfig, cannot_link_pairs, graph_clusters, graph_labels,
)


class FaceGraphTests(unittest.TestCase):
    def test_identical_faces_in_same_photo_never_merge(self):
        values = np.tile([1, 0, 0], (6, 1)).astype('float32')
        constraints = [(0, 1), (2, 3), (4, 5)]
        labels, audit = graph_labels(values, constraints)
        self.assertTrue(all(labels[a] != labels[b] for a, b in constraints))
        self.assertEqual(audit['constraint_violations'], 0)
        self.assertGreater(audit['blocked_edges']['cannot_link'], 0)

    def test_transitive_constraint_is_checked_across_entire_components(self):
        values = np.array([[1, 0], [.99, .01], [.98, .02], [.97, .03]], dtype='float32')
        labels, _ = graph_labels(values, [(0, 3)])
        self.assertNotEqual(labels[0], labels[3])
        self.assertEqual(len(set(labels)), 2)

    def test_orthogonal_identity_groups_separate_and_result_is_deterministic(self):
        values = np.repeat(np.eye(4, dtype='float32'), 5, axis=0)
        a, _ = graph_labels(values)
        b, _ = graph_labels(values)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 4)
        for start in range(0, 20, 5):
            self.assertEqual(len(set(a[start:start + 5])), 1)

    def test_cluster_size_guard(self):
        labels, audit = graph_labels(np.tile([1, 0], (20, 1)), config=FaceGraphConfig(max_cluster_size=4))
        self.assertLessEqual(max(np.bincount(labels)), 4)
        self.assertGreater(audit['blocked_edges']['cluster_size'], 0)

    def test_same_face_index_duplicate_source_is_not_false_cannot_link(self):
        members = [{'image': 'a.jpg', 'img_path': '/a.jpg', 'face_index': i} for i in (0, 0, 1)]
        self.assertEqual(cannot_link_pairs(members), [(0, 2), (1, 2)])
        members[2]['cannot_link_exempt'] = True
        self.assertEqual(cannot_link_pairs(members), [])

    def test_graph_cluster_payload_keeps_every_face_and_singletons(self):
        members = [{'image': f'{i}.jpg', 'img_path': f'/{i}.jpg', 'face_index': 0,
                    'feat': [1, 0] if i < 3 else [0, 1]} for i in range(4)]
        clusters, audit = graph_clusters(members)
        self.assertEqual(sum(c['size'] for c in clusters), 4)
        self.assertEqual(sorted(c['size'] for c in clusters), [1, 3])
        self.assertTrue(all('feat' not in m for c in clusters for m in c['members']))
        self.assertTrue(all(c['suggest_manual'] for c in clusters))
        self.assertEqual(audit['constraint_violations'], 0)

    def test_invalid_features_config_and_constraints(self):
        for values in ([[0, 0]], [[float('nan'), 0]], [1, 2]):
            with self.assertRaises(ValueError):
                graph_labels(values)
        with self.assertRaises(ValueError):
            graph_labels([[1, 0], [0, 1]], [(0, 0)])
        with self.assertRaises(ValueError):
            graph_labels([[1, 0], [0, 1]], [(0, 2)])
        with self.assertRaises(ValueError):
            FaceGraphConfig(neighbors=True)


if __name__ == '__main__':
    unittest.main()
