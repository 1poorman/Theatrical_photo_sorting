"""Copy legacy vectors to Qdrant without modifying FAISS files or ES documents.

Old indexes lack model provenance. --acknowledge-legacy-model is required;
destination metadata records that limitation. Always use a new destination.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.base import QDRANT_PREFIX
from core_modules.qdrant_images import (MANIFEST, FIELDS, atomic_manifest, fingerprint,
    get_embedder, index_lock, space)
from core_modules.vector_store import QdrantStore, normalized, point_id


def migrate_faiss(source, destination, embedder, store=None):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('migration_requires_new_destination')
    store = store or QdrantStore()
    files = [source / name for name in ('vector_index.faiss', 'image_paths.txt', 'model_info.txt')]
    if (source / 'embeddings.npy').exists():
        files.append(source / 'embeddings.npy')
    before = {str(p): fingerprint(p) for p in files}
    model_info = dict(line.split(':', 1) for line in (source / 'model_info.txt').read_text().splitlines() if ':' in line)
    if model_info['model'].strip() != embedder.model_name or int(model_info['dimension']) != embedder.feature_dim:
        raise ValueError('legacy_index_model_mismatch')
    paths = (source / 'image_paths.txt').read_text().splitlines()
    if len(set(paths)) != len(paths):
        raise ValueError('legacy_index_has_duplicate_paths')
    if (source / 'embeddings.npy').exists():
        vectors = np.load(source / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        count, dimension = vectors.shape
        read_batch = lambda start, end: vectors[start:end]
    else:
        import faiss
        index = faiss.read_index(str(source / 'vector_index.faiss'))
        count, dimension = index.ntotal, index.d
        read_batch = lambda start, end: index.reconstruct_n(start, end - start)
    if count != len(paths) or dimension != embedder.feature_dim:
        raise ValueError('legacy_index_count_or_dimension_mismatch')
    with index_lock(destination):
        if (destination / MANIFEST).exists():
            raise ValueError('migration_destination_already_published')
        manifest = {'schema_version': 1, 'backend': 'qdrant', **space(embedder),
                    'collection': QDRANT_PREFIX + '_images_' + uuid.uuid4().hex,
                    'legacy_model_unverified': True, 'source_files_sha256': before}
        store.create(manifest['collection'], dimension, FIELDS,
                     metadata={'owner_index': str(destination.resolve()), 'created_at': datetime.now(timezone.utc).isoformat()})
        try:
            for start in range(0, count, 128):
                points = [{'id': point_id(str(Path(path).resolve())), 'vector': normalized(vector, dimension),
                           'payload': {'image_path': str(Path(path).resolve()), 'model_name': embedder.model_name,
                                       'legacy_model_unverified': True}}
                          for path, vector in zip(paths[start:start + 128], read_batch(start, min(start + 128, count)))]
                store.upsert(manifest['collection'], points)
            if before != {str(p): fingerprint(p) for p in files}:
                raise ValueError('legacy_index_changed_during_migration')
            manifest['image_count'] = store.count(manifest['collection'])
            if manifest['image_count'] != count:
                raise ValueError('migration_count_mismatch')
            atomic_manifest(destination, manifest)
        except Exception:
            store.delete_collection(manifest['collection'])
            raise
    return {'source': str(source), 'destination': str(destination), 'migrated': count,
            'source_unchanged': True, 'legacy_model_unverified': True, 'collection': manifest['collection']}


def migrate_elasticsearch(model_signature, store=None, source_index=None):
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import scan
    from config.base import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASSWORD, ELASTIC_FACE_INDEX
    from core_modules.face_recognition.qdrant_database import QdrantFaceDatabase
    target = QdrantFaceDatabase(model_signature, store=store)
    with Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD)) as es:
        index = source_index or ELASTIC_FACE_INDEX
        initial = es.count(index=index)['count']
        records = ({'person_id': hit['_source']['person_id'], 'person_name': hit['_source']['person_name'],
                    'image_path': hit['_source']['image_path'], 'embedding': hit['_source']['face_embedding'],
                    'metadata': dict(hit['_source'].get('metadata') or {}, legacy_model_unverified=True)}
                   for hit in scan(es, index=index, size=128, query={'query': {'match_all': {}}}))
        migrated = target.batch_add_faces(records)
        if es.count(index=index)['count'] != initial or migrated != initial:
            raise ValueError('source_es_count_changed_during_migration')
    return {'source_index': index, 'source_count': initial, 'processed': migrated,
            'destination_count': target.store.count(target.index_name), 'collection': target.index_name,
            'source_deleted': False, 'legacy_model_unverified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('faiss', 'elasticsearch'))
    parser.add_argument('--source')
    parser.add_argument('--destination')
    parser.add_argument('--model', default='siglip2_base')
    parser.add_argument('--acknowledge-legacy-model', action='store_true')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    if not args.acknowledge_legacy_model:
        parser.error('confirm the original vector model with --acknowledge-legacy-model')
    if args.kind == 'faiss':
        if not args.source or not args.destination:
            parser.error('--source and --destination are required for FAISS')
        result = migrate_faiss(args.source, args.destination, get_embedder(args.model))
    else:
        from config.base import ARCFACE_MODEL_PATH
        from core_modules.organize.feature_cache import ContentHasher
        result = migrate_elasticsearch(ContentHasher().fingerprint(ARCFACE_MODEL_PATH)[0], source_index=args.source)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
