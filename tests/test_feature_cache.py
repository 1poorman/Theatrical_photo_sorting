"""Cache invalidation, partial failures and analysis/organization interoperability."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.feature_cache import ContentHasher, FeatureService, VectorCache, checkpoint_signature
from core_modules.organize.multimodal_evidence import MultimodalRetriever
from core_modules.organize.smart_organizer import SmartOrganizer


class Encoder:
    model_signature = 'test-weights-v1'
    preprocessing_signature = 'test-preprocessing-v1'
    feature_dim = 4

    def __init__(self):
        self.batches, self.singles, self.text_batches = [], [], []
        self.metrics = {'image_forward_calls': 0, 'text_forward_calls': 0}

    def vector(self, path):
        data = Path(path).read_bytes()
        if data == b'bad':
            raise ValueError('bad image')
        return np.array([sum(data) % 251, len(data), data[0], 1], dtype=np.float32)

    def extract_features(self, path):
        self.singles.append(path)
        self.metrics['image_forward_calls'] += 1
        return self.vector(path)

    def extract_features_batch(self, paths):
        self.batches.append(list(paths))
        self.metrics['image_forward_calls'] += 1
        valid = [path for path in reversed(paths) if Path(path).read_bytes() != b'bad']
        return np.asarray([self.vector(path) for path in valid]), valid

    def encode_text(self, text):
        self.metrics['text_forward_calls'] += 1
        if text == 'bad':
            raise ValueError('bad document')
        return np.array([len(text), 1, 1, 1], dtype=np.float32)

    def encode_text_batch(self, texts):
        self.text_batches.append(list(texts))
        self.metrics['text_forward_calls'] += 1
        if 'bad' in texts:
            raise ValueError('bad document in batch')
        return np.asarray([[len(text), 1, 1, 1] for text in texts], dtype=np.float32)


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'

    def photo(self, name, data=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if data is not None else name.encode())
        return str(path)

    def service(self, model=None, **kwargs):
        return FeatureService(model or Encoder(), cache_dir=self.cache, **kwargs)

    def test_copy_rename_and_fresh_instance_reuse_image_and_text(self):
        path = self.photo('first.jpg')
        first = self.service()
        vector = first.images([path])[path]
        first.texts(['scene'])
        renamed = self.photo('other/renamed.jpg', Path(path).read_bytes())
        fresh = self.service()
        np.testing.assert_array_equal(fresh.images([renamed])[renamed], vector)
        fresh.texts(['scene'])
        self.assertEqual(fresh.calls, {'image': 0, 'text': 0})
        self.assertEqual(fresh.report()['model_metrics'], {'image_forward_calls': 0, 'text_forward_calls': 0})

    def test_rewrite_equal_size_and_restored_mtime_invalidates_in_same_service(self):
        path = self.photo('image.jpg', b'aaaa')
        service = self.service()
        before = service.images([path])[path].copy()
        stat = os.stat(path)
        Path(path).write_bytes(b'bbbb')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        after = service.images([path])[path]
        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(service.calls['image'], 2)

    def test_weight_and_preprocessing_namespaces_are_independent(self):
        path = self.photo('image.jpg')
        self.service().images([path])
        for field in ('model_signature', 'preprocessing_signature'):
            model = Encoder()
            setattr(model, field, 'changed')
            fresh = self.service(model)
            fresh.images([path])
            self.assertEqual(fresh.calls['image'], 1)

    def test_full_checkpoint_bytes_and_tokenizer_changes_invalidate(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'model.safetensors').write_bytes(b'abcd')
        (model / 'tokenizer.json').write_text('{}')
        hasher = ContentHasher()
        first = checkpoint_signature(model, hasher)
        path = model / 'model.safetensors'
        stat = path.stat()
        path.write_bytes(b'abce')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = checkpoint_signature(model, hasher)
        self.assertNotEqual(first, second)
        (model / 'tokenizer.json').write_text('{"changed": true}')
        self.assertNotEqual(second, checkpoint_signature(model, hasher))

    def test_unsigned_encoders_cannot_share_unknown_namespace(self):
        path = self.photo('image.jpg')
        first = SimpleNamespace(extract_features=lambda p: np.array([1, 0]))
        second = SimpleNamespace(extract_features=lambda p: np.array([0, 1]))
        self.service(first).images([path])
        np.testing.assert_array_equal(self.service(second).images([path])[path], [0, 1])

    def test_reversed_partial_batch_is_aligned_and_bounded(self):
        paths = [self.photo(f'{i}.jpg') for i in range(11)]
        duplicate = self.photo('copy.jpg', Path(paths[0]).read_bytes())
        bad = self.photo('bad.jpg', b'bad')
        model = Encoder()
        service = self.service(model, batch_size=3)
        vectors = service.images(paths + [duplicate, bad])
        self.assertEqual(set(vectors), set(paths + [duplicate]))
        for path in vectors:
            np.testing.assert_array_equal(vectors[path], model.vector(path))
        self.assertEqual(model.singles, [bad])
        self.assertEqual(len(model.batches), 4)
        self.assertTrue(all(len(batch) <= 3 for batch in model.batches))
        self.assertEqual(service.calls['image'], 11)
        calls = len(model.singles)
        service.images([bad], use_batch=False)
        self.assertEqual(len(model.singles), calls)

    def test_bad_batch_mapping_retries_paths_without_misalignment(self):
        paths = [self.photo('a.jpg'), self.photo('b.jpg')]
        invalid_results = [(np.ones((2, 4)), [paths[0], paths[0]]),
                           (np.ones((2, 4)), [paths[0], '/outside.jpg']),
                           np.ones((1, 4))]
        for invalid in invalid_results:
            model = Encoder()
            model.extract_features_batch = lambda paths: invalid
            service = FeatureService(model)
            result = service.images(paths)
            for path in paths:
                np.testing.assert_array_equal(result[path], model.vector(path))
            self.assertEqual(model.singles, paths)

    def test_invalid_batch_vectors_retry_only_affected_paths(self):
        paths = [self.photo(f'{i}.jpg') for i in range(4)]
        model = Encoder()
        model.extract_features_batch = lambda paths: ([model.vector(paths[0]), [0] * 4,
                                                       [np.nan] * 4, [1, 2]], paths)
        result = self.service(model).images(paths)
        self.assertEqual(len(result), 4)
        self.assertEqual(model.singles, paths[1:])

    def test_source_changed_during_encode_is_never_published(self):
        path = self.photo('image.jpg', b'aaaa')
        model = Encoder()
        def mutate(paths):
            value = model.vector(paths[0])
            Path(path).write_bytes(b'bbbb')
            return [value], paths
        model.extract_features_batch = mutate
        service = self.service(model)
        self.assertEqual(service.images([path]), {})
        self.assertEqual(list(self.cache.rglob('*.npz')), [])
        fresh = self.service()
        np.testing.assert_array_equal(fresh.images([path])[path], Encoder().vector(path))

    def test_sources_are_hashed_at_lookup_and_publication_including_cache_hits(self):
        path = self.photo('image.jpg')
        service = self.service()
        with patch.object(service.hasher, 'fingerprint', wraps=service.hasher.fingerprint) as fingerprint:
            first = service.images([path])[path].copy()
            self.assertEqual(fingerprint.call_count, 2)
            fingerprint.reset_mock()
            np.testing.assert_array_equal(service.images([path])[path], first)
            self.assertEqual(fingerprint.call_count, 2)

    def test_later_batch_mutation_does_not_publish_earlier_vector(self):
        first, second = self.photo('first.jpg', b'aaaa'), self.photo('second.jpg', b'cccc')
        model = Encoder()
        original_batch = model.extract_features_batch
        stat = os.stat(first)
        def mutate(paths):
            vectors = original_batch(paths)
            if paths == [second]:
                Path(first).write_bytes(b'bbbb')
                os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return vectors
        model.extract_features_batch = mutate
        service = self.service(model, batch_size=1)
        self.assertEqual(set(service.images([first, second])), {second})
        self.assertEqual(len(list(self.cache.rglob('*.npz'))), 1)

    def test_partial_text_failure_does_not_repeat_for_every_image(self):
        model = Encoder()
        service = self.service(model, batch_size=2)
        result = service.texts(['good', 'bad', 'other'])
        self.assertEqual(set(result), {'good', 'other'})
        self.assertEqual(service.text_errors, {'bad': 'ValueError'})
        calls = model.metrics['text_forward_calls']
        service.texts(['good', 'bad', 'other'], use_batch=False)
        self.assertEqual(model.metrics['text_forward_calls'], calls)
        fresh = self.service()
        fresh.texts(['good', 'other'])
        self.assertEqual(fresh.calls['text'], 0)

    def test_custom_text_encoder_cannot_reuse_base_texts_or_batch_method(self):
        self.service().texts(['scene'])
        custom = self.service(text_encoder=lambda text: np.array([7, 0, 0, 1]))
        result = custom.texts(['scene'])
        np.testing.assert_array_equal(result['scene'], [7, 0, 0, 1])
        self.assertEqual(custom.embedder.text_batches, [])

    def test_corrupted_vector_is_recomputed(self):
        path = self.photo('image.jpg')
        self.service().images([path])
        cache_path = next(self.cache.rglob('*.npz'))
        with np.load(cache_path) as data:
            metadata = data['metadata'].copy()
        np.savez(cache_path, vector=np.ones(4), metadata=metadata)
        fresh = self.service()
        np.testing.assert_array_equal(fresh.images([path])[path], Encoder().vector(path))
        self.assertEqual(fresh.calls['image'], 1)

    def test_concurrent_writers_preserve_all_entries(self):
        paths = [self.photo(f'{i}.jpg') for i in range(12)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda path: self.service().images([path]), paths))
        self.assertEqual(sum(map(len, results)), len(paths))
        fresh = self.service()
        self.assertEqual(len(fresh.images(paths)), len(paths))
        self.assertEqual(fresh.calls['image'], 0)

    def test_failed_replace_keeps_previous_entry_and_cleans_temporary(self):
        cache = VectorCache(self.cache, {'model': 'test'}, dimension=2)
        key = cache.key('image', 'content')
        cache.put(key, [1, 0])
        with patch('core_modules.organize.feature_cache.os.replace', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                cache.put(key, [0, 1])
        fresh = VectorCache(self.cache, {'model': 'test'}, dimension=2)
        np.testing.assert_array_equal(fresh.get(key), [1, 0])
        self.assertEqual(len(list(self.cache.rglob('*.npz'))), 1)

    def test_encoder_output_reuse_cannot_modify_cached_vectors(self):
        cache = VectorCache(self.cache, {'model': 'test'}, dimension=2)
        key = cache.key('image', 'content')
        buffer = np.array([1., 2.], dtype=np.float32)
        cached = cache.put(key, buffer)
        buffer[:] = [3., 4.]
        np.testing.assert_array_equal(cached, [1., 2.])
        np.testing.assert_array_equal(cache.get(key), [1., 2.])
        with self.assertRaises(ValueError):
            cached.setflags(write=True)

    def test_analysis_cache_is_reused_by_organizer_and_manual_labels_remain(self):
        folder = self.root / 'photos'
        folder.mkdir()
        for name, color in [('a.jpg', 'red'), ('b.jpg', 'green')]:
            Image.new('RGB', (20, 20), color).save(folder / name)
        paths = [str(path) for path in sorted(folder.iterdir())]
        analysis = MultimodalRetriever(Encoder(), cache_dir=self.cache)
        analysis.prepare(paths, {'scenes': [{'label': 'scene', 'text': 'text'}]})
        model = Encoder()
        organizer = SmartOrganizer(None, 'unused', embedder=model, feature_cache_dir=self.cache)
        labels = self.root / 'labels.json'
        labels.write_text(json.dumps({'scene-01': '人工幕名'}))
        dedup = {'total': 2, 'kept': 2, 'discarded': 0, 'n_groups': 1,
                 'groups': [{'members': [{'name': Path(path).name, 'kept': True} for path in paths]}]}
        with patch('core_modules.organize.smart_organizer.BurstDeduplicator.dedup_folder', return_value=dedup), \
                patch.object(organizer, '_recognize_image', return_value=({}, [])):
            report = organizer.organize(str(folder), str(self.root / 'organized'),
                                        scene_labels=str(labels))
        self.assertEqual(report['scene_features']['encoded_vectors'], {'image': 0, 'text': 0})
        self.assertEqual(model.batches + model.singles, [])
        self.assertTrue(all('人工幕名' in row['organized_name'] for row in report['records']))

    def test_loaded_weight_bytes_change_signature(self):
        import torch
        from core_modules.image_search import ImageEmbedder
        embedder = ImageEmbedder.__new__(ImageEmbedder)
        embedder.model_name = 'same-name'
        embedder.model = torch.nn.Linear(2, 2)
        first = embedder.model_signature
        with torch.no_grad():
            embedder.model.weight[0, 0] += 1
        self.assertNotEqual(first, embedder.model_signature)
        first = embedder.model_signature
        embedder.model.config = SimpleNamespace(to_dict=lambda: {'activation': 'gelu'})
        self.assertNotEqual(first, embedder.model_signature)

    def test_tokenizer_runtime_padding_does_not_invalidate_features(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from core_modules.image_search import ImageEmbedder

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'scene': 2}, unk_token='[UNK]')),
            unk_token='[UNK]', pad_token='[PAD]', model_max_length=8)
        embedder = ImageEmbedder.__new__(ImageEmbedder)
        embedder.preprocess = None
        embedder.processor = SimpleNamespace(tokenizer=tokenizer)
        first = embedder.preprocessing_signature
        tokenizer(['scene'], padding='max_length', truncation=True)
        self.assertEqual(first, embedder.preprocessing_signature)
        tokenizer.padding_side = 'left'
        self.assertNotEqual(first, embedder.preprocessing_signature)
        tokenizer.padding_side = 'right'
        tokenizer.add_tokens(['new-scene'])
        self.assertNotEqual(first, embedder.preprocessing_signature)

    def test_tokenizer_same_size_vocabulary_and_processing_changes_invalidate(self):
        from tokenizers import Tokenizer, normalizers
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from core_modules.image_search import ImageEmbedder

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'scene': 2}, unk_token='[UNK]')),
            unk_token='[UNK]', pad_token='[PAD]', model_max_length=8)
        embedder = ImageEmbedder.__new__(ImageEmbedder)
        embedder.preprocess = None
        embedder.processor = SimpleNamespace(tokenizer=tokenizer)
        first = embedder.preprocessing_signature
        tokenizer.backend_tokenizer.model = WordLevel(
            {'[UNK]': 0, '[PAD]': 1, 'different': 2}, unk_token='[UNK]')
        second = embedder.preprocessing_signature
        self.assertNotEqual(first, second)
        tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
        third = embedder.preprocessing_signature
        self.assertNotEqual(second, third)
        tokenizer.pad_token = '[UNK]'
        self.assertNotEqual(third, embedder.preprocessing_signature)

    def test_single_batch_decode_agree_on_exif_orientation(self):
        import torch
        from core_modules.image_search import ImageEmbedder
        path = self.root / 'rotated.jpg'
        image = Image.new('RGB', (12, 6), 'red')
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        embedder = ImageEmbedder.__new__(ImageEmbedder)
        embedder.model_name, embedder.device = 'siglip2_base', 'cpu'
        def process(images, **kwargs):
            array = np.asarray(images).copy()
            return {'pixel_values': torch.tensor(array).permute(2, 0, 1).unsqueeze(0).float()}
        embedder.processor = process
        embedder.model = SimpleNamespace(get_image_features=lambda pixel_values: pixel_values.mean((2, 3)))
        inputs, _, _ = embedder._prepare_image(str(path))
        self.assertEqual(tuple(inputs['pixel_values'].shape), (1, 3, 12, 6))
        single = embedder.extract_features(str(path))
        batch, valid = embedder.extract_features_batch([str(path)])
        np.testing.assert_allclose(single, batch[0], atol=1e-7)
        self.assertEqual(valid, [str(path)])
        self.assertEqual(embedder.metrics['image_forward_calls'], 2)
        embedder.close()

    def test_decoder_pool_is_bounded_reused_and_can_be_closed(self):
        import concurrent.futures
        import threading
        import torch
        from core_modules.image_search import ImageEmbedder
        model = ImageEmbedder.__new__(ImageEmbedder)
        model.device, model.model_name = 'cpu', 'siglip2_base'
        self.addCleanup(model.close)
        threads = set()
        def prepare(path):
            threads.add(threading.get_ident())
            return {'pixel_values': torch.ones((1, 3, 2, 2))}, 0., 0.
        model._prepare_image = prepare
        model.model = SimpleNamespace(get_image_features=lambda pixel_values: pixel_values.mean((2, 3)))
        factory = concurrent.futures.ThreadPoolExecutor
        with patch('core_modules.image_search.concurrent.futures.ThreadPoolExecutor', wraps=factory) as pools:
            for _ in range(6):
                vectors, paths = model.extract_features_batch(['a.jpg', 'b.jpg'], sub_batch_size=2)
                self.assertEqual(paths, ['a.jpg', 'b.jpg'])
                self.assertEqual(vectors.shape, (2, 3))
            self.assertEqual(pools.call_count, 1)
            self.assertLessEqual(len(threads), pools.call_args.kwargs['max_workers'])
            executor = model._decode_executor
            model.close()
            self.assertTrue(executor._shutdown)
            model.extract_features_batch(['a.jpg'])
            self.assertEqual(pools.call_count, 2)

    def test_face_batches_respect_extractor_limit(self):
        batches = []
        def extract(images):
            batches.append(len(images))
            return np.tile([1., 0.], (len(images), 1))
        system = SimpleNamespace(
            detector=SimpleNamespace(_detect_single=lambda img: [SimpleNamespace(landmark=[]) for _ in range(70)]),
            extractor=SimpleNamespace(max_batch_size=16, align_face=lambda img, lm: np.zeros((160, 160, 3)),
                                      extract_features=extract, l2_normalize=lambda x: x))
        organizer = SmartOrganizer(system, 'unused')
        organizer._lib_matrix, organizer._lib_persons = np.array([[1., 0.]]), ['actor']
        hits, _ = organizer._recognize_image(np.zeros((20, 20, 3)), .55)
        self.assertEqual(hits, {'actor': 1.})
        self.assertEqual(batches, [16, 16, 16, 16, 6])


if __name__ == '__main__':
    unittest.main()
