"""Checkpoint face annotation binds sources and never derives unknown from names."""
from copy import deepcopy
import base64
import hashlib
from datetime import datetime, timezone
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
import sqlite3

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_face_identity_review import digest, prepare, validate_decisions, verify_existing
from core_modules.organize.task_artifacts import fingerprint


class FaceIdentityReviewTests(unittest.TestCase):
    def setUp(self):
        self.template = {'protocol_sha256': 'protocol', 'identities': [{'identity_id': 'actor-1'}],
                         'faces': [{'face_id': 'face-1', 'image_id': 'image-1', 'source_sha256': 'content',
                            'face_index': 0, 'bbox': [1, 2, 3, 4], 'decision': 'pending', 'identity_id': None,
                            'exception': False, 'reviewer': '', 'source_ref': '', 'notes': '', 'reviewed_at': ''}]}

    def completed(self, decision='known'):
        draft = deepcopy(self.template)
        draft['faces'][0].update(decision=decision, identity_id='actor-1' if decision == 'known' else None,
            reviewer='reviewer-a', source_ref='manual_crop_and_source_check',
            reviewed_at=datetime.now(timezone.utc).isoformat())
        return draft

    def test_pending_unchanged_does_not_create_labels(self):
        self.assertEqual(validate_decisions(self.template, deepcopy(self.template)), self.template['faces'])

    def test_explicit_known_unknown_and_unevaluable_are_distinct(self):
        for decision in ('known', 'unknown', 'not_evaluable'):
            with self.subTest(decision=decision):
                self.assertEqual(validate_decisions(self.template, self.completed(decision))[0]['decision'], decision)

    def test_known_identity_must_exist_unknown_cannot_carry_identity(self):
        draft = self.completed()
        draft['faces'][0]['identity_id'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'unknown_identity'):
            validate_decisions(self.template, draft)
        draft['faces'][0]['decision'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'requires_known'):
            validate_decisions(self.template, draft)

    def test_bindings_protocol_and_duplicate_ids_cannot_be_replaced(self):
        for field in ('face_id', 'image_id', 'source_sha256', 'bbox'):
            draft = self.completed()
            draft['faces'][0][field] = 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding_changed'):
                validate_decisions(self.template, draft)
        draft = self.completed(); draft['protocol_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'protocol'):
            validate_decisions(self.template, draft)
        draft = self.completed(); draft['faces'].append(deepcopy(draft['faces'][0]))
        template = deepcopy(self.template); template['faces'].append(dict(template['faces'][0], face_id='face-2'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_decisions(template, draft)

    def test_completed_annotations_require_reviewer_source_and_timezone(self):
        for field in ('reviewer', 'source_ref', 'reviewed_at'):
            draft = self.completed(); draft['faces'][0][field] = ''
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_decisions(self.template, draft)
        draft = self.completed(); draft['faces'][0]['reviewed_at'] = '2026-10-23T10:00:00'
        with self.assertRaisesRegex(ValueError, 'timezone'):
            validate_decisions(self.template, draft)

    def test_mirror_exception_does_not_assign_identity(self):
        draft = self.completed('not_evaluable'); draft['faces'][0]['exception'] = True
        row = validate_decisions(self.template, draft)[0]
        self.assertTrue(row['exception']); self.assertIsNone(row['identity_id'])
        draft['faces'][0]['exception'] = 1
        with self.assertRaisesRegex(ValueError, 'invalid_decision'):
            validate_decisions(self.template, draft)

    def test_review_export_cannot_promote_existing_images_to_final_test(self):
        for key in ('final_test_locked', 'human_decisions_applied'):
            draft = self.completed(); draft[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'cannot_promote'):
                validate_decisions(self.template, draft)

    def test_saved_artifacts_and_sources_reject_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'review'; root.mkdir()
            source = Path(directory) / 'source.png'; source.write_bytes(b'original')
            template = root / 'review_template.json'; template.write_text(json.dumps(self.template))
            (root / 'artifact_manifest.json').write_text(json.dumps({'sources': [{'path': str(source), 'sha256': fingerprint(source)}],
                'artifacts': {'review_template.json': fingerprint(template)}}))
            self.assertEqual(verify_existing(root), self.template)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'input_changed'): verify_existing(root)
            source.write_bytes(b'original'); template.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact_changed'): verify_existing(root)

    def test_binding_digest_deterministic(self):
        self.assertEqual(digest({'b': 2, 'a': 1}), digest({'a': 1, 'b': 2}))

    def test_prepare_reuses_crops_and_covers_faces_omitted_by_collage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / 'photos'; photos.mkdir()
            source = photos / 'source.png'; Image.new('RGB', (80, 60), 'red').save(source)
            photo_review = root / 'photo_review'; photo_review.mkdir()
            preview = photo_review / 'preview.jpg'; Image.new('RGB', (80, 60), 'red').save(preview)
            checksum = fingerprint(source)
            (photo_review / 'images.json').write_text(json.dumps([{'image_id': 'image-1', 'content_hash': checksum,
                'preview': 'preview.jpg', 'preview_sha256': fingerprint(preview), 'width': 80, 'height': 60, 'play_title': '剧目甲'}]))
            manifest = root / 'inputs.json'; manifest.write_text(json.dumps({'images': [
                {'path': str(source), 'source': str(source), 'sha256': checksum}]}))
            registry = root / 'registry.json'; registry.write_text(json.dumps({'schema_version': 1, 'plays': [], 'identities': []}))
            stream = io.BytesIO(); Image.new('RGB', (160, 160), 'green').save(stream, format='PNG')
            crop = stream.getvalue(); crop_hash = hashlib.sha256(crop).hexdigest()
            db = root / 'tasks.sqlite3'
            with sqlite3.connect(db) as c:
                c.executescript('CREATE TABLE tasks(id TEXT,kind TEXT,status TEXT); CREATE TABLE items(task_id TEXT,status TEXT,input TEXT,result TEXT,position INTEGER); CREATE TABLE checkpoints(task_id TEXT,name TEXT,payload TEXT);')
                c.execute('INSERT INTO tasks VALUES (?,?,?)', ('task', 'face_cluster', 'completed'))
                c.execute('INSERT INTO items VALUES (?,?,?,?,?)', ('task', 'succeeded', '{"stage":"faces"}',
                    json.dumps({'image': str(source), 'members': [{'face_index': index, 'bbox': [10, 10, 30, 35],
                                'img_path': str(source)} for index in (0, 1)]}), 0))
                c.execute('INSERT INTO items VALUES (?,?,?,?,?)', ('task', 'succeeded', '{"stage":"crops"}',
                    json.dumps({'image': str(source), 'crops': {'0': {'png': base64.b64encode(crop).decode(), 'sha256': crop_hash}}}), 1))
                for key, payload in [('input_fingerprints', {'hashes': {str(source): checksum}}), ('execution_contract', {})]:
                    c.execute('INSERT INTO checkpoints VALUES (?,?,?)', ('task', key, json.dumps(payload)))
            output = root / 'output'
            summary = prepare(db, 'task', manifest, photo_review, registry, output)
            self.assertEqual(summary['faces'], 2)
            self.assertEqual(summary['model_inferences'], 0)
            self.assertEqual(summary['crop_counts'], {'checkpoint_aligned': 1, 'recorded_bbox_unaligned': 1})
            template = verify_existing(output)
            self.assertEqual((output / template['faces'][0]['crop']).read_bytes(), crop)
            self.assertTrue(all(row['decision'] == 'pending' and row['identity_id'] is None for row in template['faces']))
            self.assertEqual(validate_decisions(template, deepcopy(template)), template['faces'])


if __name__ == '__main__':
    unittest.main()
