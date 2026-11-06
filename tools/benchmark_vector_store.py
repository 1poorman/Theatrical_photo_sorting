"""Repeatable FAISS/Qdrant/ES comparison. Real embeddings and synthetic scale are separate.

--prepare uses the vision/GPU container. The benchmark itself uses precomputed
vectors, four CPU threads and isolated worker processes; no image/model loading
is hidden inside vector-query timing. FAISS reload timing explicitly includes IO.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.vector_store import QdrantStore, normalized, point_id, match_filter


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def prepare(output):
    from core_modules.image_search import ImageEmbedder
    from core_modules.qdrant_images import fingerprint
    from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
    from config.base import ARCFACE_MODEL_PATH
    from PIL import Image
    import cv2
    source = sorted(p for p in (ROOT / 'data/ncpa_test').rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    # Deterministic spread across directories rather than a single contiguous scene.
    selected = [source[i] for i in np.linspace(0, len(source) - 1, min(160, len(source)), dtype=int)]
    photos = output / 'photos'
    photos.mkdir(parents=True, exist_ok=True)
    rows, paths = [], []
    for i, path in enumerate(selected):
        target = photos / f'{i:04d}.jpg'
        with Image.open(path) as image:
            image = image.convert('RGB')
            image.thumbnail((1024, 1024))
            image.save(target, quality=95)
        paths.append(str(target))
        rows.append({'source': str(path), 'source_sha256': fingerprint(path), 'copy': str(target)})
    embedder = ImageEmbedder(model_name='siglip2_base')
    vectors, valid = embedder.extract_features_batch(paths, sub_batch_size=16)
    assert valid == paths
    vectors = np.asarray([normalized(v) for v in vectors], dtype=np.float32)
    split = max(1, len(vectors) - 32)
    np.save(output / 'real_images_gallery.npy', vectors[:split])
    np.save(output / 'real_images_queries.npy', vectors[split:])
    save(output / 'real_images_sources.json', {'images': rows, 'gallery_count': split,
         'model_signature': embedder.model_signature, 'preprocessing_signature': embedder.preprocessing_signature})
    extractor = ArcFaceFeatureExtractor(ARCFACE_MODEL_PATH, device='cuda')
    anchors = sorted(p for p in (ROOT / 'data/face_database').glob('*/*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:256]
    aligned = [extractor.align_face(cv2.imread(str(p)), None) for p in anchors]
    face_vectors = np.vstack([extractor.extract_features(aligned[i:i+32]) for i in range(0, len(aligned), 32)])
    face_vectors = np.asarray([normalized(v) for v in face_vectors], dtype=np.float32)
    split = max(1, len(anchors) - 20)
    np.save(output / 'real_faces_gallery.npy', face_vectors[:split])
    np.save(output / 'real_faces_queries.npy', face_vectors[split:])
    save(output / 'real_faces_sources.json', {'anchors': [{'source': str(p), 'sha256': fingerprint(p)} for p in anchors],
         'gallery_count': split, 'model_signature': extractor.model_signature})
    print(json.dumps({'images': len(vectors), 'image_gallery': len(vectors[:len(vectors)-32]),
                      'faces': len(face_vectors), 'face_gallery': split}), flush=True)


def es_client():
    from elasticsearch import Elasticsearch
    from config.base import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASSWORD
    return Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD))


def disk_usage(path):
    logical = allocated = 0
    if not Path(path).exists():
        return None
    for file in Path(path).rglob('*'):
        if file.is_file():
            stat = file.stat()
            logical += stat.st_size
            allocated += stat.st_blocks * 512
    return {'logical_bytes': logical, 'allocated_bytes': allocated}


def qdrant_memory(store):
    # Qdrant Prometheus exports process RSS on Linux; keep absence explicit.
    try:
        with urlopen(store.url + '/metrics', timeout=10) as response:
            text = response.read().decode()
        for line in text.splitlines():
            if line.startswith('process_resident_memory_bytes '):
                return int(float(line.split()[1]))
    except Exception:
        pass
    return None


def index_state(store, row):
    info = store.info(row['collection'])
    row.update(qdrant_indexed_vectors=info.get('indexed_vectors_count', 0),
               qdrant_status=info['status'], qdrant_segments=info['segments_count'],
               qdrant_config=info['config'],
               qdrant_collection_disk=disk_usage(Path('/benchmark/qdrant_storage/collections') / row['collection']))
    row['approximate_query_mode'] = ('flat_small_collection' if not row['qdrant_indexed_vectors']
                                     else 'hnsw_with_possible_unindexed_tail')


def configure_scale_index(store, collection):
    # Lower both thresholds for the explicit ANN experiment. Small real datasets
    # remain flat; production full_scan_threshold is unchanged.
    store.request('PATCH', store.path(collection), {
        'hnsw_config': {'full_scan_threshold': 1000},
        'optimizers_config': {'indexing_threshold': 1000}})
    started = time.monotonic()
    while True:
        info = store.info(collection)
        if info['status'] == 'green' and info.get('indexed_vectors_count', 0) >= info['points_count'] * .95:
            return time.monotonic() - started
        if time.monotonic() - started > 300:
            raise RuntimeError('scale_collection_not_indexed_before_benchmark')
        time.sleep(2)


def worker(args):
    import faiss
    faiss.omp_set_num_threads(4)
    directory = args.output / args.dataset
    queries = np.load(directory / 'queries.npy')
    k = min(10, len(np.load(directory / 'gallery.npy', mmap_mode='r')))
    backend = args.worker
    client = QdrantStore()
    es = es_client() if backend == 'elasticsearch' else None
    def rss():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
        raise RuntimeError('linux_rss_unavailable')
    rss_before = rss()
    peak = [rss_before]
    stop = threading.Event()
    def sample():
        while not stop.wait(.002):
            peak[0] = max(peak[0], rss())
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    index = faiss.read_index(str(directory / 'vector_index.faiss')) if backend == 'faiss_resident' else None

    def query(vector):
        if backend.startswith('faiss'):
            if backend == 'faiss_reload':
                current = faiss.read_index(str(directory / 'vector_index.faiss'))
                extra_vectors = np.load(directory / 'embeddings.npy')
                paths = (directory / 'image_paths.txt').read_text().splitlines()
                assert len(extra_vectors) == len(paths) == current.ntotal
            else:
                current = index
            scores, ids = current.search(vector[None], k)
            return ids[0].tolist(), scores[0].tolist()
        if backend == 'elasticsearch':
            response = es.search(index=args.collection, body={'size': k,
                'query': {'script_score': {'query': {'match_all': {}}, 'script': {
                    'source': "Math.max(0.0, cosineSimilarity(params.q, 'face_embedding'))",
                    'params': {'q': vector.tolist()}}}}, '_source': ['ordinal']})
            hits = response['hits']['hits']
            return [h['_source']['ordinal'] for h in hits], [h['_score'] for h in hits]
        hits = client.query(args.collection, vector, limit=k, exact=backend == 'qdrant_exact')
        return [h['payload']['ordinal'] for h in hits], [h['score'] for h in hits]

    started = time.perf_counter()
    query(queries[0])
    first_ms = (time.perf_counter() - started) * 1000
    for vector in queries[:3]:
        query(vector)
    times, ids, scores = [], [], []
    for i in range(args.requests):
        vector = queries[i % len(queries)]
        started = time.perf_counter()
        found, values = query(vector)
        times.append((time.perf_counter() - started) * 1000)
        ids.append(found)
        scores.append(values)
    if es:
        es.close()
    peak[0] = max(peak[0], rss())
    stop.set()
    sampler.join()
    result = {'backend': backend, 'requests': len(times), 'first_request_ms': first_ms,
              'p50_ms': float(np.percentile(times, 50)), 'p95_ms': float(np.percentile(times, 95)),
              'mean_ms': float(np.mean(times)), 'sequential_qps': 1000 / float(np.mean(times)),
              'client_peak_rss_bytes': peak[0], 'rss_method': 'proc_self_VmRSS_sample_2ms',
              'client_import_rss_bytes': rss_before, 'ids': ids, 'scores': scores}
    save(directory / (backend + '.json'), result)


def compare(output, sizes, requests):
    import faiss
    faiss.omp_set_num_threads(4)
    store = QdrantStore()
    datasets = [('real_images', 'real', 768), ('real_faces', 'real', 512)]
    for count in sizes:
        name = 'synthetic_images_' + str(count)
        rng = np.random.default_rng(20261102 + count)
        vectors = rng.standard_normal((count, 768), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = rng.standard_normal((32, 768), dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        np.save(output / (name + '_gallery.npy'), vectors)
        np.save(output / (name + '_queries.npy'), queries)
        datasets.append((name, 'synthetic', 768))
    results = []
    for name, source_kind, dimension in datasets:
        directory = output / name
        directory.mkdir(exist_ok=True)
        vectors = np.load(output / (name + '_gallery.npy'))
        queries = np.load(output / (name + '_queries.npy'))
        np.save(directory / 'gallery.npy', vectors)
        np.save(directory / 'queries.npy', queries)
        started = time.perf_counter()
        flat = faiss.IndexFlatIP(dimension)
        flat.add(vectors)
        faiss.write_index(flat, str(directory / 'vector_index.faiss'))
        np.save(directory / 'embeddings.npy', vectors)
        (directory / 'image_paths.txt').write_text(''.join(f'/benchmark/photos/{i}.jpg\n' for i in range(len(vectors))))
        faiss_write = time.perf_counter() - started
        exact_scores, exact_ids = flat.search(queries, min(10, len(vectors)))
        collection = 'benchmark_' + uuid.uuid4().hex
        store.create(collection, dimension, ('project_id',))
        if source_kind == 'synthetic':
            store.request('PATCH', store.path(collection), {'hnsw_config': {'full_scan_threshold': 1000}})
        started = time.perf_counter()
        for offset in range(0, len(vectors), 128):
            store.upsert(collection, [{'id': i, 'vector': v.tolist(), 'payload': {
                'ordinal': i, 'image_path': f'/benchmark/photos/{i}.jpg', 'project_id': str(i % 4)}}
                for i, v in enumerate(vectors[offset:offset+128], offset)])
        qdrant_write = time.perf_counter() - started
        assert store.count(collection) == len(vectors)
        indexing_started = time.perf_counter()
        info = store.info(collection)
        # Small real collections intentionally remain flat; report the actual count.
        while len(vectors) * dimension * 4 >= 10_000 * 1024 and (
                info['status'] != 'green' or info.get('indexed_vectors_count', 0) < len(vectors) * .95):
            if time.perf_counter() - indexing_started > 300:
                break
            time.sleep(2)
            info = store.info(collection)
        row = {'dataset': name, 'source_kind': source_kind, 'count': len(vectors), 'dimension': dimension,
               'query_vectors': len(queries), 'faiss_write_seconds': faiss_write,
               'qdrant_upload_seconds': qdrant_write, 'qdrant_index_wait_seconds': time.perf_counter()-indexing_started,
               'qdrant_indexed_vectors': info.get('indexed_vectors_count'), 'collection': collection,
               'qdrant_status': info['status'], 'measurements': []}
        es_index = None
        if name == 'real_faces':
            from elasticsearch.helpers import bulk
            es_index = 'benchmark_' + uuid.uuid4().hex
            with es_client() as es:
                es.indices.create(index=es_index, mappings={'properties': {
                    'face_embedding': {'type': 'dense_vector', 'dims': dimension, 'index': True, 'similarity': 'cosine'},
                    'ordinal': {'type': 'integer'}}}, settings={'number_of_shards': 2, 'number_of_replicas': 1})
                started = time.perf_counter()
                bulk(es, ({'_index': es_index, '_id': str(i), '_source': {
                    'face_embedding': v.tolist(), 'ordinal': i}} for i, v in enumerate(vectors)), refresh=True)
                row['es_upload_seconds'] = time.perf_counter()-started
                row['es_primary_store_bytes'] = es.indices.stats(index=es_index)['indices'][es_index]['primaries']['store']['size_in_bytes']
        backends = ['faiss_reload', 'faiss_resident', 'qdrant_exact', 'qdrant_ann']
        if es_index:
            backends.append('elasticsearch')
        for backend in backends:
            subprocess.run([sys.executable, __file__, '--output', str(output), '--worker', backend,
                '--dataset', name, '--collection', es_index if backend == 'elasticsearch' else collection,
                '--requests', str(requests)], check=True)
            measurement = json.loads((directory / (backend + '.json')).read_text())
            recalls, tie_recalls, errors = [], [], []
            for i, (ids, scores) in enumerate(zip(measurement.pop('ids'), measurement.pop('scores'))):
                q = i % len(queries)
                expected = exact_ids[q]
                actual_scores = vectors[ids] @ queries[q]
                if backend == 'elasticsearch':
                    actual_scores = np.maximum(actual_scores, 0)
                recalls.append(len(set(ids) & set(expected.tolist())) / len(expected))
                tie_recalls.append(float(np.mean(actual_scores >= max(0, exact_scores[q][-1]) - 1e-5)
                    if backend == 'elasticsearch' else np.mean(actual_scores >= exact_scores[q][-1] - 1e-5)))
                errors.append(float(np.max(np.abs(actual_scores - scores))))
            measurement.update(recall_at_10=float(np.mean(recalls)), tie_aware_recall_at_10=float(np.mean(tie_recalls)),
                               max_returned_score_error=max(errors))
            row['measurements'].append(measurement)
            print(json.dumps({'dataset': name, **measurement}), flush=True)
        row['qdrant_server_rss_bytes_after'] = qdrant_memory(store)
        row['qdrant_collection_disk'] = disk_usage(Path('/benchmark/qdrant_storage/collections') / collection)
        row['faiss_persistent_bytes'] = sum((directory / n).stat().st_size for n in ('vector_index.faiss','embeddings.npy','image_paths.txt'))
        filtered = store.query(collection, queries[0], limit=10, filters=match_filter({'project_id': '1'}))
        row['filter_correct'] = all(h['payload']['project_id'] == '1' for h in filtered) and bool(filtered)
        index_state(store, row)
        # Keep scale collections until size/resource inspection, then remove them explicitly.
        results.append(row)
        save(output / 'comparison.json', {'at': datetime.now(timezone.utc).isoformat(), 'cpu_threads': 4,
            'warmup_requests': 3, 'hnsw_ef': 128, 'hnsw_m': 16, 'datasets': results,
            'limitations': ['No OS page cache flush; first-request is not cold-disk latency.',
                'Sequential vector-only timings exclude feature extraction; no peak-concurrency claims.',
                'Synthetic vectors are random normalized Gaussian vectors, not accuracy ground truth.',
                'Qdrant collection disk includes WAL/preallocation; FAISS bytes include both legacy vector copies.']})
        if es_index:
            with es_client() as es:
                es.indices.delete(index=es_index)
    print('REPORT=' + str(output / 'comparison.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--remeasure', action='store_true', help='Reuse stored collections and rerun only query/memory measurements')
    parser.add_argument('--sizes', type=int, nargs='*', default=[10000, 100000])
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--worker', choices=['faiss_reload','faiss_resident','qdrant_exact','qdrant_ann','elasticsearch'])
    parser.add_argument('--dataset')
    parser.add_argument('--collection')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
    elif args.prepare:
        prepare(args.output)
    elif args.remeasure:
        remeasure(args.output, args.requests)
    else:
        compare(args.output, args.sizes, args.requests)


def remeasure(output, requests):
    """Reuse existing data to correct measurement instrumentation without rebuilding."""
    import faiss
    from elasticsearch.helpers import bulk
    faiss.omp_set_num_threads(4)
    report = json.loads((output / 'comparison.json').read_text())
    save(output / 'comparison_before_remeasure.json', report)
    store = QdrantStore()
    for row in report['datasets']:
        if row['source_kind'] == 'synthetic':
            row['final_index_config_wait_seconds'] = configure_scale_index(store, row['collection'])
    for row in report['datasets']:
        directory = output / row['dataset']
        vectors = np.load(directory / 'gallery.npy', mmap_mode='r')
        queries = np.load(directory / 'queries.npy')
        flat = faiss.IndexFlatIP(row['dimension'])
        flat.add(vectors)
        expected_scores, expected_ids = flat.search(queries, min(10,len(vectors)))
        es_index = None
        if row['dataset'] == 'real_faces':
            es_index = 'benchmark_' + uuid.uuid4().hex
            with es_client() as es:
                es.indices.create(index=es_index, mappings={'properties': {
                    'face_embedding': {'type':'dense_vector','dims':512,'index':True,'similarity':'cosine'},
                    'ordinal':{'type':'integer'}}},settings={'number_of_shards':2,'number_of_replicas':1})
                bulk(es, ({'_index':es_index,'_id':str(i),'_source':{'face_embedding':v.tolist(),'ordinal':i}}
                    for i,v in enumerate(vectors)),refresh=True)
        measurements = []
        for previous in row['measurements']:
            backend = previous['backend']
            subprocess.run([sys.executable,__file__,'--output',str(output),'--worker',backend,
                '--dataset',row['dataset'],'--collection',es_index if backend=='elasticsearch' else row['collection'],
                '--requests',str(requests)],check=True)
            measurement = json.loads((directory/(backend+'.json')).read_text())
            recall,ties,errors = [],[],[]
            for i,(ids,scores) in enumerate(zip(measurement.pop('ids'),measurement.pop('scores'))):
                q=i%len(queries)
                actual=vectors[ids]@queries[q]
                boundary=expected_scores[q][-1]
                if backend=='elasticsearch':
                    actual=np.maximum(actual,0);boundary=max(0,boundary)
                recall.append(len(set(ids)&set(expected_ids[q].tolist()))/len(expected_ids[q]))
                ties.append(float(np.mean(actual>=boundary-1e-5)))
                errors.append(float(np.max(np.abs(actual-scores))))
            measurement.update(recall_at_10=float(np.mean(recall)),tie_aware_recall_at_10=float(np.mean(ties)),max_returned_score_error=max(errors))
            measurements.append(measurement)
            print(json.dumps({'dataset':row['dataset'],**measurement}),flush=True)
        row['measurements']=measurements
        index_state(store, row)
        report['remeasured_at'] = datetime.now(timezone.utc).isoformat()
        report['measurement_note'] = 'All final client RSS sampled from /proc/self/status every 2ms; synthetic HNSW full_scan_threshold/indexing_threshold=1000 KB. Original upload timings predate final config.'
        save(output/'comparison.json',report)
        if es_index:
            with es_client() as es:es.indices.delete(index=es_index)


if __name__ == '__main__':
    main()
