"""Reviewed-face metrics never promote pending/unknown boxes to identity GT."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_face_identity_review import evaluate_labels


class FaceIdentityQualityTests(unittest.TestCase):
    def setUp(self):
        self.faces = [{'face_id': f'face-{i}', 'image_id': f'image-{i // 2}',
                      'source_sha256': str(i // 2) * 64, 'face_index': i % 2, 'bbox': [i, 0, i + 1, 1],
                      'decision': 'pending', 'identity_id': None, 'exception': False,
                      'reviewer': '', 'source_ref': '', 'notes': '', 'reviewed_at': ''} for i in range(4)]
        self.template = {'protocol_sha256': 'a' * 64, 'faces': self.faces,
                         'identities': [{'identity_id': 'A', 'actor_name': 'actor A'}, {'identity_id': 'B', 'actor_name': 'actor B'}]}
        self.members = [{key: row[key] for key in ('source_sha256', 'face_index', 'bbox')} for row in self.faces]

    def labelled(self, decisions):
        draft = deepcopy(self.template)
        for row, identity in zip(draft['faces'], decisions):
            row.update(decision='known' if identity in ('A', 'B') else identity, identity_id=identity if identity in ('A', 'B') else None,
                       reviewer='reviewer', source_ref='inspection', reviewed_at='2026-10-24T00:00:00Z')
        return draft

    def test_pending_boxes_have_no_B_cubed_FPIR_or_constraint_GT(self):
        r = evaluate_labels(self.template, self.template, self.members, [0, 0, 0, 0], [0, 1, 2, 3])
        self.assertEqual(r['box_label_counts']['pending'], 4)
        self.assertIsNone(r['partitions']['legacy']['known_subset_b_cubed']['f1'])
        self.assertEqual(r['partitions']['graph']['reviewed_cannot_link_pairs'], 0)
        self.assertIsNone(r['identity_matching']['unknown_FPIR'])

    def test_known_subset_metrics_preserve_detection_order_when_review_rows_differ(self):
        draft = self.labelled(['A', 'B', 'A', 'B'])
        draft['faces'].reverse()
        r = evaluate_labels(self.template, draft, self.members, [0, 0, 0, 0], [0, 1, 0, 1])
        self.assertEqual(r['partitions']['legacy']['known_subset_b_cubed']['precision'], .5)
        self.assertEqual(r['partitions']['graph']['known_subset_b_cubed']['f1'], 1)
        self.assertEqual(r['partitions']['legacy']['reviewed_constraint_violations'], 2)
        self.assertEqual(r['partitions']['graph']['reviewed_constraint_violations'], 0)

    def test_unknown_boxes_are_not_assumed_distinct_people_or_false_positives(self):
        draft = self.labelled(['unknown', 'unknown', 'not_evaluable', 'not_evaluable'])
        r = evaluate_labels(self.template, draft, self.members, [0, 0, 0, 0], [0, 1, 2, 3])
        self.assertEqual(r['exclusions']['unknown_faces_without_cross_photo_identity_GT'], 2)
        self.assertEqual(r['partitions']['graph']['known_subset_b_cubed']['count'], 0)
        self.assertIsNone(r['identity_matching']['unknown_FPIR'])

    def test_mirror_exemption_requires_explicit_review_and_removes_pair(self):
        draft = self.labelled(['A', 'A', 'B', 'not_evaluable'])
        r = evaluate_labels(self.template, draft, self.members, [0, 0, 1, 2], [0, 0, 1, 2])
        self.assertEqual(len(r['same_identity_physical_face_pairs_requiring_exception_review']), 1)
        draft['faces'][1]['exception'] = True
        r = evaluate_labels(self.template, draft, self.members, [0, 0, 1, 2], [0, 0, 1, 2])
        self.assertEqual(r['partitions']['graph']['reviewed_cannot_link_pairs'], 0)
        self.assertEqual(r['exclusions']['known_exempt_faces'], 1)

    def test_tampered_feature_source_or_bbox_binding_is_rejected(self):
        for field in ('bbox', 'source_sha256'):
            members = deepcopy(self.members)
            members[0][field] = [9, 0, 10, 1] if field == 'bbox' else 'f' * 64
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                evaluate_labels(self.template, self.template, members, [0, 1, 2, 3], [0, 1, 2, 3])

    def test_small_perfect_subset_does_not_satisfy_matching_or_final_quality(self):
        r = evaluate_labels(self.template, self.labelled(['A', 'B', 'A', 'B']), self.members, [0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(r['partitions']['graph']['known_subset_b_cubed']['f1'], 1)
        self.assertFalse(r['identity_matching']['known_faces_at_least_200'])
        self.assertFalse(r['identity_matching']['gallery_query_isolation_verified'])
        self.assertEqual(r['quality_gate'], 'insufficient_data')
        self.assertFalse(r['default_adoption'])


if __name__ == '__main__':
    unittest.main()
