"""Bounded-memory image indexing and online search with a small local manifest."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import uuid

from config.base import QDRANT_PREFIX
from core_modules.vector_store import QdrantStore, normalized, point_id, match_filter

MANIFEST = 'vector_store.json'
FIELDS = ('project_id', 'play_id', 'scene_id')
MODEL_LOCK = threading.RLock()


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_manifest(directory, value):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.vector-store-', dir=directory)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / MANIFEST)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def index_lock(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / '.vector-store.lock', 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_manifest(directory):
    value = json.loads((Path(directory) / MANIFEST).read_text())
    if value.get('schema_version') != 1 or value.get('backend') != 'qdrant':
        raise ValueError('unsupported_vector_manifest')
    QdrantStore.path(value['collection'])
    if not isinstance(value.get('dimension'), int) or value['dimension'] < 1:
        raise ValueError('invalid_manifest_dimension')
    return value


@lru_cache(maxsize=2)
def get_embedder(model_name):
    from core_modules.image_search import ImageEmbedder
    return ImageEmbedder(model_name=model_name)


def space(embedder):
    return {'model_name': embedder.model_name, 'dimension': embedder.feature_dim,
            'model_signature': embedder.model_signature,
            'preprocessing_signature': embedder.preprocessing_signature}


def verify_space(manifest, embedder):
    actual = space(embedder)
    if any(manifest.get(key) != value for key, value in actual.items()):
        raise ValueError('image_vector_model_changed_rebuild_required')


def metadata_values(metadata):
    if metadata is None:
        return {}
    if not isinstance(metadata, dict) or set(metadata) - set(FIELDS):
        raise ValueError('metadata_supports_project_id_play_id_scene_id')
    if any(not isinstance(v, str) or not v for v in metadata.values()):
        raise ValueError('metadata_values_must_be_nonempty_strings')
    return dict(metadata)


def image_paths(source):
    source = Path(source)
    if not source.is_dir():
        raise ValueError('image_folder_not_found')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return sorted(str(p.resolve()) for p in source.rglob('*') if p.is_file() and p.suffix.lower() in extensions)


def upload(store, manifest, paths, embedder, metadata=None, batch_size=32):
    metadata = metadata_values(metadata)
    paths = list(dict.fromkeys(str(Path(path).resolve()) for path in paths))
    failed, written = [], 0
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        hashes, usable = {}, []
        for path in batch:
            try:
                hashes[path] = fingerprint(path)
                usable.append(path)
            except OSError:
                failed.append({'image_path': path, 'error': 'image_not_readable'})
        with MODEL_LOCK:
            vectors, valid = embedder.extract_features_batch(usable, sub_batch_size=batch_size)
        if len(vectors) != len(valid):
            raise ValueError('feature_path_count_mismatch')
        failed.extend({'image_path': path, 'error': 'feature_extraction_failed'} for path in usable if path not in valid)
        points = []
        existing = {row['id']: row.get('payload', {}) for row in
                    store.retrieve(manifest['collection'], [point_id(path) for path in valid])}
        for path, vector in zip(valid, vectors):
            if fingerprint(path) != hashes[path]:
                raise ValueError('image_changed_during_extraction')
            payload = {'image_path': path, 'content_sha256': hashes[path],
                       'model_name': manifest['model_name'],
                       **{key: value for key, value in existing.get(point_id(path), {}).items() if key in FIELDS},
                       **metadata}
            points.append({'id': point_id(path), 'vector': normalized(vector, manifest['dimension']), 'payload': payload})
        store.upsert(manifest['collection'], points)
        written += len(points)
    return {'upserted': written, 'failed': len(failed), 'errors': failed}


def build(image_folder, index_save_path, model_name, metadata=None, batch_size=32, embedder=None, store=None):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', model_name):
        raise ValueError('invalid_image_model_name')
    store = store or QdrantStore()
    directory = Path(index_save_path) / model_name
    paths = image_paths(image_folder)
    if not paths:
        raise ValueError('no_images_to_index')
    metadata_values(metadata)
    with index_lock(directory):
        with MODEL_LOCK:
            embedder = embedder or get_embedder(model_name)
        manifest = {'schema_version': 1, 'backend': 'qdrant', **space(embedder),
                    'collection': QDRANT_PREFIX + '_images_' + uuid.uuid4().hex,
                    'source_root': str(Path(image_folder).resolve()),
                    'created_at': datetime.now(timezone.utc).isoformat()}
        old = load_manifest(directory) if (directory / MANIFEST).exists() else None
        store.create(manifest['collection'], manifest['dimension'], FIELDS,
                     metadata={'owner_index': str(directory.resolve()), 'created_at': manifest['created_at']})
        try:
            result = upload(store, manifest, paths, embedder, metadata, batch_size)
            if result['failed']:
                raise ValueError(f"index_build_failed_images: {result['failed']}")
            manifest['image_count'] = store.count(manifest['collection'])
            if manifest['image_count'] != len(paths):
                raise ValueError('published_point_count_mismatch')
            if old:
                manifest['previous_collection'] = old['collection']
            atomic_manifest(directory, manifest)
        except Exception:
            store.delete_collection(manifest['collection'])
            raise
    return SimpleNamespace(image_count=manifest['image_count'], backend='qdrant'), embedder, str(directory)


def upsert_images(index_path, paths, metadata=None, embedder=None, store=None):
    if not paths or not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ValueError('image_paths_must_be_nonempty_list')
    store = store or QdrantStore()
    with index_lock(index_path):
        manifest = load_manifest(index_path)
        with MODEL_LOCK:
            embedder = embedder or get_embedder(manifest['model_name'])
        verify_space(manifest, embedder)
        result = upload(store, manifest, paths, embedder, metadata)
        manifest['image_count'] = store.count(manifest['collection'])
        atomic_manifest(index_path, manifest)
        return dict(result, image_count=manifest['image_count'])


def delete_images(index_path, paths, store=None):
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError('image_paths_must_be_nonempty_list')
    store = store or QdrantStore()
    with index_lock(index_path):
        manifest = load_manifest(index_path)
        before = store.count(manifest['collection'])
        store.delete_points(manifest['collection'], ids=[point_id(str(Path(p).resolve())) for p in paths])
        manifest['image_count'] = store.count(manifest['collection'])
        atomic_manifest(index_path, manifest)
        return {'deleted': before - manifest['image_count'], 'image_count': manifest['image_count']}


def query_vector(manifest, vector, top_k=5, filters=None, exact=True, store=None):
    store = store or QdrantStore()
    hits = store.query(manifest['collection'], normalized(vector, manifest['dimension']),
                       limit=top_k, filters=match_filter(metadata_values(filters)), exact=exact)
    return [{'rank': i + 1, 'image_path': hit['payload']['image_path'],
             'image_id': hit['id'], 'similarity': float(hit['score'])}
            for i, hit in enumerate(hits)]


def search(query_image_path, index_path, top_k=5, filters=None, exact=True):
    manifest = load_manifest(index_path)
    with MODEL_LOCK:
        embedder = get_embedder(manifest['model_name'])
        verify_space(manifest, embedder)
        vector = embedder.extract_features(query_image_path)
    if vector is None:
        raise ValueError('query_feature_extraction_failed')
    results = query_vector(manifest, vector.reshape(-1), top_k, filters, exact)
    from core_modules.image_search import create_results_image
    merged = create_results_image(query_image_path, results) if results else None
    return results, merged, manifest['model_name']


def status(index_path, store=None):
    store = store or QdrantStore()
    manifest = load_manifest(index_path)
    info = store.info(manifest['collection'])
    return {'backend': 'qdrant', 'collection': manifest['collection'],
            'model_name': manifest['model_name'], 'dimension': manifest['dimension'],
            'image_count': store.count(manifest['collection']), 'status': info['status'],
            'indexed_vectors_count': info.get('indexed_vectors_count', 0),
            'legacy_model_unverified': manifest.get('legacy_model_unverified', False)}
