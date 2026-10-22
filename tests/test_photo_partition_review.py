"""Photo review contracts: provenance, time hints, reuse and contamination."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.organize.task_artifacts import fingerprint
from tools.prepare_photo_partition_review import (
    POLICY, bind_labels, build_images, checked_checkpoint, exposure_from_manifest,
    features, original_capture, partition_review, prepare_image, safe_source,
    temporal_reviews)


class Exif(dict):
    def __init__(self, top=None, nested=None):
        super().__init__(top or {})
        self.nested = nested or {}

    def get_ifd(self, number):
        return self.nested


class PhotoPartitionReviewTests(unittest.TestCase):
    def image(self, name, capture, sequence=None, scene=None, stream='camera-1', base='broad-name'):
        return {'image_id': name, 'canonical_source': '剧《甲》/【原始】照片/' + name + '.jpg',
            'source_kind_hint': 'stage_photo', 'play_title': '剧《甲》',
            'camera_stream_hint_id': stream, 'filename_base_hint': base,
            'exif': {'capture_local': capture, 'camera_model': stream, 'camera_body_id': None},
            'photographer_hint': '摄影师', 'organized_sequence_hint': sequence,
            'imported_scene': scene, 'date_hint': '20260101',
            'imported_label_file_id': name if scene else None, 'exposure_refs': []}

    def test_original_capture_uses_nested_original_and_subseconds(self):
        result = original_capture(Exif({306: '2026:10:01 12:00:00', 272: 'Camera'},
            {36867: '2016:06:09 19:00:00', 37521: '007', 36881: '+08:00', 42033: 'serial-secret'}))
        self.assertEqual(result['capture_local'], '2016-06-09T19:00:00.007000')
        self.assertEqual(result['offset_raw'], '+08:00')
        self.assertNotIn('serial-secret', str(result))
        self.assertFalse(result['capture_verified'])

    def test_edit_time_and_digitized_time_never_become_capture(self):
        result = original_capture(Exif({306: '2026:01:01 12:00:00'}, {36868: '2016:06:09 19:00:00'}))
        self.assertIsNone(result['capture_local'])
        self.assertIsNone(original_capture(Exif(nested={36867: 'invalid'}))['capture_local'])

    def test_time_neighbors_are_separate_from_name_groups_and_cross_camera(self):
        images = [self.image('a', '2026-01-01T12:00:00.000000', 1, '第1场'),
                  self.image('b', '2026-01-01T12:00:01.000000', 2, '第1场'),
                  self.image('c', '2026-01-01T13:00:00.000000', 3, '第2场'),
                  self.image('d', '2026-01-01T12:00:01.500000', 4, '第1场', stream='camera-2')]
        result = temporal_reviews(images)
        self.assertEqual([r['image_ids'] for r in result['time_neighbors']], [['a', 'b']])
        self.assertEqual(len(result['filename_groups']), 1)
        self.assertEqual(result['filename_groups'][0]['capture_span_seconds'], 3600)
        self.assertTrue(result['filename_groups'][0]['multiple_imported_scenes'])
        self.assertIsNone(result['filename_groups'][0]['burst_group_id'])
        self.assertEqual(result['verified_bursts'], 0)

    def test_organized_sequence_does_not_override_capture_hints(self):
        result = temporal_reviews([self.image('later', '2026-01-01T12:05:00.000000', 1, '第1场'),
                                   self.image('earlier', '2026-01-01T12:00:00.000000', 2, '第1场')])
        self.assertEqual(result['streams'][0]['image_ids_in_capture_hint_order'], ['earlier', 'later'])
        self.assertEqual(len(result['organized_order_inversions']), 1)
        self.assertFalse(result['streams'][0]['chronology_verified'])

    def test_exposure_uses_image_identity_and_rejects_changed_hash(self):
        rows = [{'image_id': 'same', 'content_hash': 'hash', 'source': 'raw.jpg'},
                {'image_id': 'same', 'content_hash': 'hash', 'source': 'renamed.jpg'}]
        self.assertEqual(exposure_from_manifest({'images': [rows[0]]}, rows, 'history'), {'same'})
        with self.assertRaisesRegex(ValueError, 'inventory_mismatch'):
            exposure_from_manifest({'rows': [{**rows[0], 'content_hash': 'changed'}]}, rows, 'history')

    def test_label_binding_rejects_contradictory_exact_copies(self):
        rows = [{'file_id': 'a', 'image_id': 'same', 'content_hash': 'hash', 'source': 'a.jpg'},
                {'file_id': 'b', 'image_id': 'same', 'content_hash': 'hash', 'source': 'b.jpg'}]
        labels = [{**rows[0], 'play_id': 'play', 'scene_id': 'scene1'},
                  {**rows[1], 'play_id': 'play', 'scene_id': 'scene2'}]
        with self.assertRaisesRegex(ValueError, 'conflicting_labels'):
            bind_labels(rows, labels)
        labels[1]['scene_id'] = 'scene1'
        labels[1]['content_hash'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'label_inventory_mismatch'):
            bind_labels(rows, labels)

    def test_history_final_name_and_unexposed_images_remain_review_development(self):
        rows = [{'file_id': 'a', 'image_id': 'a', 'split': 'final_test_provisional'},
                {'file_id': 'b', 'image_id': 'b', 'split': 'validation'}]
        images = [self.image('a', None, scene='第1场'), self.image('b', None)]
        result = partition_review(rows, images, [], {'time_neighbors': []}, {'history': {'a'}})
        self.assertEqual(result['legacy_split_exposure']['final_test_provisional']['known_exposed_unique_images'], 1)
        self.assertEqual(set(result['new_assignments'].values()), {'development_review'})
        self.assertFalse(result['final_test_locked'])
        self.assertEqual(result['verified_performances'], 0)

    def test_source_paths_reject_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            outside = Path(directory) / 'outside.jpg'
            outside.write_bytes(b'test')
            (root / 'link.jpg').symlink_to(outside)
            for value in ('../outside.jpg', 'link.jpg', str(outside)):
                with self.assertRaises(ValueError):
                    safe_source(root, value)

    def test_resume_reuses_verified_checkpoint_and_rejects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            pixels = np.random.default_rng(4).integers(0, 256, (35, 47, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            image.save(source)
            digest = hashlib.sha256(b'RGB:47:35:')
            digest.update(image.tobytes())
            row = {'image_id': digest.hexdigest(), 'content_hash': fingerprint(source),
                   'width': 47, 'height': 35, 'dhash': features(image)['dhash']}
            output = root / 'output'
            output.mkdir()
            value, reused = prepare_image(row, source, output, 'protocol', False)
            self.assertFalse(reused)
            with patch('tools.prepare_photo_partition_review.Image.open', side_effect=AssertionError('decode repeated')):
                actual, reused = prepare_image(row, source, output, 'protocol', True)
                self.assertTrue(reused)
                self.assertEqual(actual, value)
            checkpoint = output / 'image_checkpoints' / (row['image_id'] + '.json')
            corrupt = json.loads(checkpoint.read_text())
            corrupt['features']['dhash'] = 'f' * 16
            checkpoint.write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError, 'changed_image_checkpoint'):
                checked_checkpoint(checkpoint, row['image_id'], row['content_hash'], 'protocol', output)


if __name__ == '__main__':
    unittest.main()
