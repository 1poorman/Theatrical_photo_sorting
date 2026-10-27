"""Behavior regressions for correctness and the local multimodal adapters."""
import io
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.scene_split import split_scenes
from core_modules.organize.multimodal_evidence import MultimodalRetriever, retrieve_evidence
from core_modules.organize.smart_organizer import SmartOrganizer, apply_scene_evidence
from core_modules.organize.review_queue import ReviewQueue, load_scene_evidence
from core_modules.organize.scene_reasoner import SceneReasoner
from core_modules.organize.llm_client import image_content, vision_client_from_env, LLMError
from core_modules.organize.embedding_gemma import EmbeddingGemmaEmbedder


class CoreUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def knowledge(self, n=6):
        return {'scenes': [{'scene_id': f'act-{i:02d}', 'act': i, 'title': f'scene{i}',
                           'label': f'scene{i}', 'text': f'description{i}', 'source_ref': str(i)}
                          for i in range(1, n + 1)]}

    def test_exif_sort_and_segments_keep_original_vectors(self):
        paths = [f'p{i}' for i in reversed(range(16))]
        vectors = np.asarray([[int(p[1:]), 1] for p in paths])
        seen = []
        def cluster(segment, mapping, max_k):
            seen.append([int(mapping[p][0]) for p in segment])
            return [segment]
        def time(path):
            i = int(path[1:])
            return datetime(2020, 1, 1) + timedelta(seconds=i + (1000 if i >= 8 else 0))
        with patch('core_modules.organize.scene_split.read_exif_datetime', side_effect=time), \
             patch('core_modules.organize.scene_split._subcluster', side_effect=cluster):
            assignment = split_scenes(paths, embeddings=vectors)
        self.assertEqual(seen, [list(range(8)), list(range(8, 16))])
        self.assertEqual(assignment['p15'], 'scene-02')

    def test_partial_vectors_only_disable_affected_segment(self):
        paths = [f'p{i}' for i in range(16)]
        vectors = {p: [i % 2, 1] for i, p in enumerate(paths) if i != 0}
        time = lambda p: datetime(2020, 1, 1) + timedelta(seconds=int(p[1:]) + (1000 if int(p[1:]) >= 8 else 0))
        with patch('core_modules.organize.scene_split.read_exif_datetime', side_effect=time), \
             patch('sklearn.cluster.KMeans') as km, patch('sklearn.metrics.silhouette_score', return_value=.8):
            km.return_value.fit_predict.return_value = np.arange(8) % 2
            result = split_scenes(paths, embeddings=vectors)
        self.assertEqual(result['p0'], 'scene-01')
        self.assertTrue(result['p15'].startswith('scene-02-'))
        self.assertEqual(km.call_count, 2)

    def test_reject_misaligned_matrix(self):
        with self.assertRaises(ValueError):
            split_scenes(['a', 'b'], embeddings=np.ones((1, 2)))

    def test_malformed_vector_disables_only_its_segment(self):
        from core_modules.organize.scene_split import _subcluster
        paths = [f'p{i}' for i in range(8)]
        vectors = {p: [1, 0] for p in paths}
        vectors['p4'] = [1, 0, 0]
        self.assertEqual(_subcluster(paths, vectors, 3), [paths])

    def test_organizer_keeps_processing_after_one_feature_failure(self):
        photos = self.root / 'photos'
        photos.mkdir()
        for name, color in (('a.jpg', 'red'), ('b.jpg', 'green')):
            Image.new('RGB', (20, 20), color).save(photos / name)
        def encode(path):
            if Path(path).name == 'a.jpg':
                raise ValueError('bad feature')
            return np.array([1, 0])
        organizer = SmartOrganizer(None, 'unused', embedder=SimpleNamespace(extract_features=encode))
        dedup = {'total': 2, 'kept': 2, 'discarded': 0, 'n_groups': 1,
                 'groups': [{'members': [{'name': name, 'kept': True} for name in ('a.jpg', 'b.jpg')]}]}
        with patch('core_modules.organize.smart_organizer.BurstDeduplicator.dedup_folder', return_value=dedup), \
             patch.object(organizer, '_recognize_image', return_value=({}, [])):
            report = organizer.organize(str(photos), str(self.root / 'output'))
        self.assertEqual(len(report['records']), 2)
        self.assertTrue(all('organized_name' in row for row in report['records']))

    def test_visual_recalls_sixth_scene_with_top_one(self):
        path = self.root / 'neutral.jpg'
        path.touch()
        model = SimpleNamespace(model_signature='test', extract_features=lambda p: np.array([1, 0]),
                                encode_text=lambda t: np.array([1, 0] if 'scene6' in t else [0, 1]))
        row = retrieve_evidence(str(path), self.knowledge(), top_k=1,
                                embedder=MultimodalRetriever(model))
        self.assertEqual(row['candidates'][0]['scene_id'], 'act-06')
        self.assertEqual(row['candidates'][0]['visual_score'], 1)

    def test_partial_text_failure_keeps_lexical_candidate_and_reports_reason(self):
        path = self.root / 'neutral.jpg'
        path.touch()
        def text(value):
            if 'scene6' in value:
                raise ValueError('encoding failed')
            return np.array([0, 1])
        retriever = MultimodalRetriever(SimpleNamespace(
            extract_features=lambda p: np.array([1, 0]), encode_text=text))
        row = retrieve_evidence(str(path), self.knowledge(), top_k=6, embedder=retriever)
        self.assertEqual(len(row['candidates']), 6)
        sixth = next(c for c in row['candidates'] if c['scene_id'] == 'act-06')
        self.assertIsNone(sixth['visual_score'])
        self.assertEqual(sixth['retrieval_mode'], 'lexical')
        self.assertIn({'component': 'text_embedding', 'scene_id': 'act-06', 'reason': 'ValueError'},
                      row['retrieval']['errors'])

    def test_batch_prefill_cache_rerun_and_bad_image_isolation(self):
        paths = [str(self.root / name) for name in ('good.jpg', 'bad.jpg')]
        for path in paths:
            Path(path).write_bytes(Path(path).name.encode())
        calls = []
        class Model:
            model_signature = 'batch-test'
            def extract_features_batch(self, paths):
                calls.append('batch')
                raise ValueError('bad image in batch')
            def extract_features(self, path):
                calls.append(Path(path).name)
                if Path(path).name == 'bad.jpg':
                    raise ValueError('bad image')
                return np.array([1, 0])
            def encode_text_batch(self, texts):
                calls.append('text-batch')
                return np.tile([1, 0], (len(texts), 1))
            def encode_text(self, text):
                raise AssertionError('text cache should be populated')
        retriever = MultimodalRetriever(Model(), cache_dir=str(self.root))
        retriever.prepare(paths, self.knowledge())
        self.assertEqual(retrieve_evidence(paths[0], self.knowledge(), embedder=retriever)['retrieval']['mode'], 'visual')
        self.assertEqual(retrieve_evidence(paths[1], self.knowledge(), embedder=retriever)['retrieval']['mode'], 'lexical')
        retriever.save()
        calls.clear()
        fresh = MultimodalRetriever(Model(), cache_dir=str(self.root))
        fresh.prepare([paths[0]], self.knowledge())
        self.assertEqual(calls, [])
        self.assertEqual(fresh.calls, {'image': 0, 'text': 0})
        Path(paths[0]).write_bytes(b'changed')
        fresh.prepare([paths[0]], self.knowledge())
        self.assertIn('good.jpg', calls)

    def test_siglip_batch_failure_retries_paths_and_preserves_alignment(self):
        import torch
        from core_modules.image_search import ImageEmbedder
        paths = [str(self.root / name) for name in ('red.jpg', 'green.jpg')]
        for path, color in zip(paths, ('red', 'green')):
            Image.new('RGB', (20, 20), color).save(path)
        model = ImageEmbedder.__new__(ImageEmbedder)
        self.addCleanup(model.close)
        model.model_name = 'siglip2_base'
        model.device = 'cpu'
        model.processor = lambda images, **kwargs: {
            'pixel_values': torch.tensor(np.asarray(images).mean((0, 1))[:2]).reshape(1, 2, 1, 1).float()}
        attempts = []
        def fail_batch(pixel_values):
            attempts.append(len(pixel_values))
            if len(pixel_values) > 1:
                raise RuntimeError('forced batch failure')
            return pixel_values.mean((2, 3))
        model.model = SimpleNamespace(get_image_features=fail_batch)
        with patch.object(model, '_prepare_image', wraps=model._prepare_image) as decode, \
                patch.object(model, 'extract_features', side_effect=AssertionError('unexpected decode retry')) as single:
            vectors, valid = model.extract_features_batch(paths)
        self.assertCountEqual([call.args[0] for call in decode.call_args_list], paths)
        single.assert_not_called()
        self.assertEqual(attempts, [2, 1, 1])
        self.assertEqual(valid, paths)
        np.testing.assert_array_equal(vectors, np.eye(2))

    def recognize(self, vectors, anchors, persons):
        extractor = SimpleNamespace(align_face=lambda img, lm: np.zeros((160, 160, 3)),
                                    extract_features=lambda batch: np.asarray(vectors),
                                    l2_normalize=lambda value: value)
        system = SimpleNamespace(extractor=extractor, detector=SimpleNamespace(
            _detect_single=lambda img: [SimpleNamespace(landmark=[]) for _ in vectors]))
        organizer = SmartOrganizer(system, 'unused')
        organizer._lib_matrix = np.asarray(anchors)
        organizer._lib_persons = persons
        return organizer._recognize_image(np.zeros((160, 160, 3)), .55)[0]

    def test_single_face_confirms_only_best_actor(self):
        self.assertEqual(self.recognize([[1, 0]], [[.9, 0], [.7, 0]], ['A', 'B']), {'A': .9})

    def test_ambiguous_face_is_unrecognized(self):
        self.assertEqual(self.recognize([[1, 0]], [[.9, 0], [.88, 0]], ['A', 'B']), {})

    def test_same_actor_roles_do_not_cause_false_ambiguity(self):
        hits = self.recognize([[1, 0], [0, 1]], [[.9, .1], [.89, .1], [.1, .95]],
                              ['A（饰甲）', 'A（饰乙）', 'B（饰丙）'])
        self.assertEqual(hits, {'A（饰甲）': .9, 'B（饰丙）': .95})

    def test_pending_updates_and_all_human_decisions_stay_locked(self):
        queue = ReviewQueue(str(self.root))
        queue.seed([{'image': name, 'label': 'old', 'scene_id': 'act-01'}
                    for name in ('a', 'b', 'c', 'd')])
        for name, action in zip(('a', 'b', 'c'), ('accept', 'modify', 'reject')):
            queue.decide(name, action, label='human')
        before = queue.records()
        revisions = queue.revisions()
        stats = queue.seed([{'image': name, 'label': 'new', 'scene_id': 'act-02'}
                            for name in ('a', 'b', 'c', 'd')])
        self.assertEqual(stats['updated'], 1)
        self.assertEqual(queue.records()['d']['label'], 'new')
        for name in ('a', 'b', 'c'):
            self.assertEqual(queue.records()[name], before[name])
        self.assertEqual(queue.revisions(), revisions)
        self.assertEqual(queue.records()['d']['previous_candidate_version'], before['d']['candidate_version'])
        self.assertNotEqual(queue.records()['d']['candidate_version'], before['d']['candidate_version'])
        again = queue.seed([{'image': 'd', 'label': 'new', 'scene_id': 'act-02'}])
        self.assertEqual(again['updated'], 0)

    def test_legacy_approved_record_without_reviewer_stays_locked(self):
        row = {'image': 'a.jpg', 'label': 'human', 'review': {'status': 'approved'}}
        (self.root / 'review_queue.jsonl').write_text(json.dumps(row) + '\n')
        queue = ReviewQueue(str(self.root))
        queue.seed([{'image': 'a.jpg', 'label': 'machine'}])
        self.assertEqual(queue.records()['a.jpg'], row)

    def test_legacy_binding_survives_restart_and_export_uses_full_path(self):
        queue = ReviewQueue(str(self.root))
        queue.seed([{'image': 'a.jpg', 'label': 'candidate'}])
        queue.decide('a.jpg', 'modify', label='human')
        before = queue.records()
        queue.seed([{'image': '/one/a.jpg', 'label': 'new'}])
        self.assertEqual(queue.records(), before)
        self.assertEqual(queue.list_pending()[0]['source_image'], '/one/a.jpg')
        export, _ = queue.export_final()
        self.assertEqual(json.loads(Path(export).read_text()), {'/one/a.jpg': 'human'})
        restarted = ReviewQueue(str(self.root))
        with self.assertRaises(ValueError):
            restarted.seed([{'image': '/two/a.jpg', 'label': 'another project'}])
        self.assertEqual(restarted.records(), before)

    def test_scoped_labels_never_apply_to_another_directory(self):
        path = self.root / 'labels.json'
        path.write_text(json.dumps({'/one/a.jpg': 'A', '/two/a.jpg': 'B'}))
        evidence = load_scene_evidence(str(path))
        assigned, stats = apply_scene_evidence(
            {'/one/a.jpg': 'scene-01', '/two/a.jpg': 'scene-02', '/three/a.jpg': 'scene-03'}, evidence)
        self.assertEqual(assigned, {'/one/a.jpg': 'A', '/two/a.jpg': 'B', '/three/a.jpg': 'scene-03'})
        self.assertEqual(stats['applied'], 2)

    def test_unreviewed_jsonl_does_not_override_scenes(self):
        path = self.root / 'evidence.jsonl'
        path.write_text(json.dumps({'image': 'a.jpg', 'label': 'machine'}) + '\n')
        assigned, _ = apply_scene_evidence({'a.jpg': 'scene-01'}, load_scene_evidence(str(path)))
        self.assertEqual(assigned, {'a.jpg': 'scene-01'})

    def test_legacy_pending_updates_without_duplicate_and_rejects_ambiguity(self):
        queue = ReviewQueue(str(self.root))
        queue.seed([{'image': 'a.jpg', 'label': 'old'}])
        queue.seed([{'image': '/one/a.jpg', 'label': 'new'}])
        self.assertEqual(list(queue.records()), ['a.jpg'])
        self.assertEqual(queue.records()['a.jpg']['label'], 'new')
        with self.assertRaises(ValueError):
            queue.seed([{'image': '/one/a.jpg'}, {'image': '/two/a.jpg'}])

    def test_lexical_score_is_not_visual_evidence(self):
        reasoner = SceneReasoner(self.knowledge())
        conf, matched, _ = reasoner._score({'scene_id': 'act-01'}, {},
                                          {'score': 1, 'retrieval_mode': 'lexical'}, None)
        self.assertEqual(conf, 0)
        self.assertNotIn('visual', matched)
        conf, matched, _ = reasoner._score({'scene_id': 'act-01'}, {},
            {'score': 1, 'retrieval_mode': 'visual', 'visual_score': 0}, None)
        self.assertEqual(conf, 0)
        self.assertNotIn('visual', matched)

    def test_real_image_message_and_visual_observation_feed_closed_set(self):
        path = self.root / 'neutral.jpg'
        Image.new('RGB', (1800, 900), 'red').save(path)
        messages = []
        class Client:
            model = 'Qwen3.6-27B'
            model_identity = 'Qwen3.8-27B'
            def chat(self, prompt):
                messages.append(prompt)
                if isinstance(prompt[1]['content'], list):
                    content = json.dumps({k: ['red'] for k in (
                        'clothing', 'props', 'setting', 'actions', 'visible_text', 'uncertain')})
                else:
                    payload = json.loads(prompt[1]['content'])
                    ref = next(e['id'] for e in payload['evidences'] if e['type'] == 'observation')
                    content = json.dumps({'scene_id': 'act-01', 'evidence_refs': [ref]})
                return {'content': content, 'trace': {'model': self.model}}
        client = Client()
        row = SceneReasoner(self.knowledge(), client=client, vision_client=client).reason(str(path))
        self.assertEqual(row['scene_id'], 'act-01')
        self.assertEqual(row['review']['status'], 'pending')
        self.assertEqual(row['model_trace']['vision']['model_identity'], 'Qwen3.8-27B')
        self.assertTrue(messages[0][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,'))

    def test_vision_configuration_preserves_served_alias(self):
        path = self.root / 'test.env'
        path.write_text('url=http://localhost:9/v1\nbig_model_name=Qwen3.6-27B\nkey=test\n')
        client = vision_client_from_env(str(path))
        self.assertEqual(client.model, 'Qwen3.6-27B')
        self.assertEqual(client.model_identity, 'Qwen3.8-27B')

    def test_empty_model_response_is_a_protocol_failure(self):
        path = self.root / 'test.env'
        path.write_text('url=http://localhost:9/v1\nbig_model_name=Qwen3.6-27B\nkey=test\n')
        payloads = []
        def opener(request, timeout):
            payloads.append(json.loads(request.data))
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': None},
                                                        'finish_reason': 'length'}]}).encode())
        client = vision_client_from_env(str(path), opener=opener)
        client.max_retries = 0
        with self.assertRaises(LLMError):
            client.chat([{'role': 'user', 'content': 'structured result'}])
        self.assertEqual(payloads[0]['chat_template_kwargs'], {'enable_thinking': False})

    def test_direct_manual_label_keeps_custom_label_without_machine_score(self):
        row = SceneReasoner(self.knowledge(), top_k=1).reason('photo.jpg', manual_label='人工确认的场景')
        self.assertEqual(row['label'], '人工确认的场景')
        self.assertIsNone(row['confidence'])
        self.assertEqual(row['review']['status'], 'approved')
        queue = ReviewQueue(str(self.root))
        queue.seed([row])
        queue.seed([{'image': 'photo.jpg', 'label': 'new machine candidate'}])
        self.assertEqual(queue.records()['photo.jpg']['label'], '人工确认的场景')

    def test_visual_schema_failure_leaves_no_observation(self):
        from core_modules.organize.scene_reasoner import validate_choice
        self.assertFalse(validate_choice({'scene_id': 'act-01', 'evidence_refs': ''}, ['act-01'], [])[0])
        self.assertFalse(validate_choice({'scene_id': 'act-01', 'confidence': True}, ['act-01'], [])[0])
        path = self.root / 'photo.jpg'
        Image.new('RGB', (20, 20)).save(path)
        client = SimpleNamespace(model='vision', chat=lambda messages: {
            'content': '{"clothing": "wrong type"}', 'trace': {'model': 'vision'}})
        evidence, trace = SceneReasoner(self.knowledge(), vision_client=client)._observe(str(path))
        self.assertIsNone(evidence)
        self.assertEqual(trace['error'], 'invalid_visual_schema')

    def test_image_message_applies_exif_orientation_and_size_limit(self):
        path = self.root / 'rotated.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (40, 20), 'red').save(path, exif=exif)
        part = image_content(str(path), max_side=30)
        with Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',', 1)[1]))) as preview:
            self.assertEqual(preview.size, (15, 30))
            self.assertFalse(preview.getexif())

    def test_evaluation_loads_scene_entities_and_rejects_empty_knowledge(self):
        from tools.evaluate_core_upgrade import load_scenes
        scenes = self.knowledge()['scenes']
        (self.root / 'entities.json').write_text(json.dumps({'scenes': scenes}))
        (self.root / 'scene_candidates.jsonl').write_text(json.dumps({'image': 'a.jpg', 'candidates': []}))
        self.assertEqual(load_scenes(self.root), scenes)
        (self.root / 'entities.json').write_text(json.dumps({'scenes': []}))
        with self.assertRaisesRegex(ValueError, '有效幕次'):
            load_scenes(self.root)

    def test_inventory_keeps_duplicate_copies_in_one_split(self):
        from tools.build_core_upgrade_manifest import inventory
        for play in ('play-a', 'play-b', 'play-c'):
            folder = self.root / play
            folder.mkdir()
            array = np.random.default_rng(ord(play[-1])).integers(0, 256, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(array).save(folder / 'photo.png')
        (self.root / 'play-b/copy.png').write_bytes((self.root / 'play-a/photo.png').read_bytes())
        rows, summary = inventory(self.root)
        first = next(row for row in rows if row['source'] == 'play-a/photo.png')
        copy = next(row for row in rows if row['source'] == 'play-b/copy.png')
        self.assertEqual(first['image_id'], copy['image_id'])
        self.assertEqual(first['split'], copy['split'])
        self.assertEqual(summary['file_count'], 4)
        self.assertEqual(summary['unique_decoded_images'], 3)
        self.assertEqual(summary['verified_scene_labels'], 0)
        self.assertFalse(summary['final_test_locked'])

    def test_gemma_input_modalities_and_normalization(self):
        calls = []
        class Model:
            def encode(self, inputs, **kwargs):
                calls.append((inputs, kwargs))
                return np.ones((len(inputs), 768))
        (self.root / 'config.json').write_text(json.dumps(
            {'model_type': 'embedding_gemma2', 'vision_config': {'hidden_size': 768}}))
        model = EmbeddingGemmaEmbedder(model_path=str(self.root), model=Model())
        images = model.extract_features_batch(['a.jpg', 'b.jpg'])
        texts = model.encode_text_batch(['剧情'])
        self.assertAlmostEqual(float(np.linalg.norm(images[0])), 1, places=5)
        self.assertEqual(texts.shape, (1, 768))
        self.assertIn('image', calls[0][0][0])
        self.assertEqual(calls[0][1]['prompt'], '')
        self.assertEqual(calls[1][1]['prompt_name'], 'Document')


if __name__ == '__main__':
    unittest.main()
