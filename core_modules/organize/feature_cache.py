"""Content-addressed image/text features shared by analysis and organization.

Only vectors are retained. Each disk entry is independently replaced, so writers
cannot discard another process's entries by saving an old whole-cache snapshot.
Unsigned third-party encoders get a private namespace, never a shared 'unknown'.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from zipfile import BadZipFile

import numpy as np


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def tokenizer_signature(backend):
    """Hash all tokenizer content without rebuilding its large vocabulary in Python.

    Tokenizers writes runtime padding/truncation before ``added_tokens``. Parse
    that small header only; retain every byte of the vocabulary and processing
    rules. Fall back to full JSON normalization if the serialization changes.
    """
    serialized = backend.to_str()
    header, separator, content = serialized.partition(',"added_tokens":')
    if separator:
        try:
            options = json.loads(header + '}')
        except ValueError:
            options = None
        if isinstance(options, dict) and set(options) <= {'version', 'padding', 'truncation'}:
            options.pop('padding', None)
            options.pop('truncation', None)
            digest = hashlib.sha256(signature(options).encode())
            digest.update(content.encode())
            return digest.hexdigest()
    options = json.loads(serialized)
    options.pop('padding', None)
    options.pop('truncation', None)
    return signature(options)


def file_version(path):
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ContentHasher:
    """Stream full content; stat metadata is only a concurrent-write guard.

    Even ctime may have insufficient resolution on some filesystems. Never use
    unchanged metadata as proof of content equality, including within a process.
    """

    def fingerprint(self, path):
        path = os.path.abspath(path)
        before = file_version(path)
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if file_version(path) != before:
            raise ValueError('source_changed_during_hash')
        return digest.hexdigest(), before


def checkpoint_signature(root, hasher):
    """Full bytes of weights, tokenizer and configuration, independent of location."""
    root = Path(root)
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   p.suffix in {'.safetensors', '.bin', '.json', '.model', '.txt'})
    return signature([(p.relative_to(root).as_posix(), hasher.fingerprint(p)[0])
                      for p in files])


def tensor_signature(model):
    """Hash the loaded state, including fallback/online weights and actual dtype.

    Call at encoder creation or after a parameter version change; never per image.
    Streaming one tensor at a time bounds the temporary CPU copy.
    """
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(value))
    return digest.hexdigest()


def valid_vector(value, dimension=None):
    try:
        vector = np.asarray(value, dtype='<f4')
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1 or not vector.size or (dimension and vector.size != dimension):
            return None
        if not np.isfinite(vector).all() or not np.isfinite(np.linalg.norm(vector)) \
                or np.linalg.norm(vector) <= 0:
            return None
        return np.ascontiguousarray(vector)
    except (TypeError, ValueError, OverflowError):
        return None


class VectorCache:
    def __init__(self, directory, identity, dimension=None, capacity=512):
        self.directory = Path(directory) / 'features-v2' if directory else None
        self.identity = identity
        self.dimension = dimension
        self.capacity = capacity
        self._memory = OrderedDict()

    def key(self, kind, content):
        return signature({'schema': 2, 'encoder': self.identity, 'kind': kind, 'content': content})

    def path(self, key):
        return self.directory / key[:2] / (key + '.npz')

    def _remember(self, key, vector):
        # Encoders may return a view of a reusable output buffer. Own immutable
        # bytes so a later inference cannot silently change a cached vector.
        vector = np.frombuffer(vector.tobytes(), dtype='<f4')
        self._memory[key] = vector
        self._memory.move_to_end(key)
        if len(self._memory) > self.capacity:
            self._memory.popitem(last=False)
        return vector

    def get(self, key):
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        if self.directory is None:
            return None
        try:
            with np.load(self.path(key), allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata'].item()))
                vector = valid_vector(archive['vector'], self.dimension)
            if vector is None or metadata != self._metadata(key, vector):
                return None
            return self._remember(key, vector)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            return None

    def _metadata(self, key, vector):
        return {'schema': 2, 'key': key, 'encoder': self.identity,
                'dimension': int(vector.size), 'dtype': '<f4',
                'vector_sha256': hashlib.sha256(vector.tobytes()).hexdigest()}

    def put(self, key, value):
        vector = valid_vector(value, self.dimension)
        if vector is None:
            raise ValueError('invalid_vector')
        if self.directory is not None:
            path = self.path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
                    temporary = stream.name
                    np.savez(stream, vector=vector,
                             metadata=json.dumps(self._metadata(key, vector), sort_keys=True))
                os.replace(temporary, path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return self._remember(key, vector)


class FeatureService:
    """Bounded batches with explicit path mapping and per-item failure recovery."""
    def __init__(self, embedder, cache_dir=None, batch_size=8,
                 model_signature=None, text_encoder=None):
        initialized = time.perf_counter()
        self.embedder = embedder
        self.batch_size = max(1, int(batch_size))
        self.model_signature = (model_signature or getattr(embedder, 'model_signature', None)
                                or 'private-' + uuid.uuid4().hex)
        self.identity = {'model': self.model_signature,
                         'preprocessing': getattr(embedder, 'preprocessing_signature', 'declared-by-model')}
        self.cache = VectorCache(cache_dir, self.identity, getattr(embedder, 'feature_dim', None))
        default_text = getattr(embedder, 'encode_text', None)
        self.text_encoder = text_encoder or default_text
        self.text_batch = getattr(embedder, 'encode_text_batch', None) if self.text_encoder == default_text else None
        # A caller-supplied text encoder must not reuse the base model's documents.
        self.text_namespace = ('' if self.text_encoder == default_text else
                               getattr(text_encoder, 'model_signature', None) or uuid.uuid4().hex)
        self.hasher = ContentHasher()
        self.calls = {'image': 0, 'text': 0}  # successful newly encoded vectors, legacy API
        self.encoder_calls = dict.fromkeys(('image_batch', 'image_single', 'text_batch', 'text_single'), 0)
        self.hits = {'image': 0, 'text': 0}
        self.timings = {'identity_seconds': time.perf_counter() - initialized,
                        'hash_seconds': 0., 'cache_seconds': 0., 'encoder_seconds': 0.}
        self.image_errors, self.text_errors = {}, {}
        self._failed = {}
        self._metrics_start = dict(getattr(embedder, 'metrics', {}))

    def clear_failures(self):
        self._failed.clear()
        self.image_errors.clear()
        self.text_errors.clear()

    def _cached(self, key, kind):
        start = time.perf_counter()
        vector = self.cache.get(key)
        self.timings['cache_seconds'] += time.perf_counter() - start
        if vector is not None:
            self.hits[kind] += 1
        return vector

    def _encode(self, kind, batch, values, prefetch_batch_size=None):
        method = (getattr(self.embedder, 'extract_features_batch', None) if batch else
                  getattr(self.embedder, 'extract_features', None)) if kind == 'image' else (
                      self.text_batch if batch else self.text_encoder)
        self.encoder_calls[kind + ('_batch' if batch else '_single')] += 1
        start = time.perf_counter()
        try:
            if prefetch_batch_size is not None:
                return self.embedder.extract_features_prefetched(values, batch_size=prefetch_batch_size)
            return method(values)
        finally:
            self.timings['encoder_seconds'] += time.perf_counter() - start

    def _store(self, key, kind, vector):
        vector = valid_vector(vector, self.cache.dimension)
        if vector is None:
            raise ValueError('invalid_vector')
        start = time.perf_counter()
        # Disk failures surface to callers; a cache cannot silently claim persistence.
        result = self.cache.put(key, vector)
        self.timings['cache_seconds'] += time.perf_counter() - start
        self.calls[kind] += 1
        return result

    def images(self, paths, batch_size=None, use_batch=True):
        result, pending = {}, {}
        encoded_vectors = {}
        aliases, versions = {}, {}
        for original in dict.fromkeys(map(os.fspath, paths)):
            path = os.path.abspath(original)
            start = time.perf_counter()
            try:
                content, version = self.hasher.fingerprint(path)
            except (OSError, ValueError) as exc:
                self.image_errors[original] = type(exc).__name__
                continue
            finally:
                self.timings['hash_seconds'] += time.perf_counter() - start
            key = self.cache.key('image', content)
            aliases.setdefault(key, []).append(original)
            versions[original] = (content, version)
            vector = self._cached(key, 'image')
            if vector is not None:
                result[key] = vector
            elif key not in self._failed:
                pending.setdefault(key, path)
        items = list(pending.items())
        size = max(1, int(batch_size or self.batch_size))
        prefetch = use_batch and callable(getattr(self.embedder, 'extract_features_prefetched', None))
        # Give capable encoders a next batch to prepare while the current batch
        # runs. Other encoders keep the original batch-size contract unchanged.
        window = size * (2 if prefetch else 1)
        for start in range(0, len(items), window):
            chunk = items[start:start + window]
            chunk_paths = [p for _, p in chunk]
            by_path = {}
            if use_batch and callable(getattr(self.embedder, 'extract_features_batch', None)):
                try:
                    encoded = self._encode('image', True, chunk_paths,
                                           prefetch_batch_size=size if prefetch else None)
                    vectors, valid = encoded if isinstance(encoded, tuple) else (encoded, chunk_paths)
                    valid = [os.path.abspath(p) for p in valid]
                    if len(vectors) != len(valid) or len(set(valid)) != len(valid) or not set(valid) <= set(chunk_paths):
                        raise ValueError('batch_path_mismatch')
                    by_path = dict(zip(valid, vectors))
                except Exception:
                    pass  # retry each path; never guess alignment for a malformed batch
            for key, path in chunk:
                try:
                    vector = valid_vector(by_path.get(path), self.cache.dimension)
                    if vector is None:
                        vector = self._encode('image', False, path)
                    vector = valid_vector(vector, self.cache.dimension)
                    if vector is None:
                        raise ValueError('invalid_vector')
                    # Retain only small owned vectors. Validate source bytes once
                    # at publication, after every batch has finished, rather than
                    # hashing here and hashing the same file again on return.
                    encoded_vectors[key] = vector.copy()
                except Exception as exc:
                    self._failed[key] = type(exc).__name__
        aligned = {}
        for key, originals in aliases.items():
            if key in encoded_vectors:
                original = originals[0]
                before_hash = time.perf_counter()
                try:
                    try:
                        current = self.hasher.fingerprint(original)
                    finally:
                        self.timings['hash_seconds'] += time.perf_counter() - before_hash
                    if current != versions[original]:
                        raise ValueError('source_changed_during_encoding')
                    result[key] = self._store(key, 'image', encoded_vectors[key])
                    aligned[original] = result[key]
                    self.image_errors.pop(original, None)
                except Exception as exc:
                    self._failed[key] = type(exc).__name__
                    self.image_errors[original] = type(exc).__name__
                originals = originals[1:]
            for original in originals:
                try:
                    before_hash = time.perf_counter()
                    current = self.hasher.fingerprint(original)
                    self.timings['hash_seconds'] += time.perf_counter() - before_hash
                    if current != versions[original]:
                        raise ValueError('source_changed_during_encoding')
                    if key in result:
                        aligned[original] = result[key]
                        self.image_errors.pop(original, None)
                    else:
                        self.image_errors[original] = self._failed.get(key, 'unavailable')
                except (OSError, ValueError) as exc:
                    self.image_errors[original] = type(exc).__name__
        return aligned

    def texts(self, texts, batch_size=None, use_batch=True):
        result, pending = {}, []
        for value in dict.fromkeys(texts):
            key = self.cache.key('text', signature([self.text_namespace, value]))
            vector = self._cached(key, 'text')
            if vector is not None:
                result[value] = vector
            elif key in self._failed:
                self.text_errors[value] = self._failed[key]
            elif self.text_encoder is not None:
                pending.append((key, value))
        size = max(1, int(batch_size or self.batch_size))
        for start in range(0, len(pending), size):
            chunk = pending[start:start + size]
            encoded = [None] * len(chunk)
            if use_batch and callable(self.text_batch):
                try:
                    vectors = self._encode('text', True, [value for _, value in chunk])
                    if len(vectors) != len(chunk):
                        raise ValueError('batch_text_count_mismatch')
                    encoded = vectors
                except Exception:
                    pass
            for (key, value), vector in zip(chunk, encoded):
                try:
                    if valid_vector(vector, self.cache.dimension) is None:
                        vector = self._encode('text', False, value)
                    result[value] = self._store(key, 'text', vector)
                    self.text_errors.pop(value, None)
                except Exception as exc:
                    self._failed[key] = self.text_errors[value] = type(exc).__name__
        return result

    def report(self):
        metrics = getattr(self.embedder, 'metrics', {})
        return {'encoder': self.identity, 'batch_size': self.batch_size,
                'encoded_vectors': dict(self.calls), 'cache_hits': dict(self.hits),
                'encoder_calls': dict(self.encoder_calls), 'timings': dict(self.timings),
                'model_metrics': {k: v - self._metrics_start.get(k, 0) for k, v in metrics.items()},
                'image_errors': dict(self.image_errors), 'text_errors': dict(self.text_errors)}

    def save(self):
        # Entries are persisted atomically as they are encoded; legacy callers may flush.
        return str(self.cache.directory) if self.cache.directory else None
