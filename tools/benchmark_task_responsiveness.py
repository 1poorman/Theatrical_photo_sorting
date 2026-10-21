"""Measure HTTP responsiveness under a controlled slow task (no model quality).

Five clients send 300 requests over 60 seconds. The task's per-image operation
waits 250 ms to model blocking I/O; this is explicitly not a GPU acceptance run.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]


def worker(root, port, real=False):
    import uvicorn
    from fastapi import FastAPI
    from app import organize_tasks
    from core_modules.organize.review_queue import ReviewQueue
    queue = ReviewQueue(str(root / 'review'))
    organize_tasks.TASK_ROOT = root / 'tasks'
    original = organize_tasks.scene_handler
    def slow_handler(params):
        process, finish = original(params)
        def slow(item):
            time.sleep(0.25)
            return process(item)
        return slow, finish
    organize_tasks.scene_handler = slow_handler
    if real:
        import main
        from core_modules.image_search import ImageEmbedder
        main.FEATURE_CACHE_DIR = str(root / 'features')
        model = ImageEmbedder('siglip2_base', device='cuda:0')
        main.srv.organize_models['embedder'] = model
        # Explicitly bypass configuration backends for this SigLIP2 measurement.
        main._get_scene_embedder = lambda: model
        organize_tasks.scene_handler = original
        app = main.app
        (root / 'model.json').write_text(json.dumps({'model': 'siglip2_base', 'device': 'cuda:0',
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'face_model': 'not_used'}))
    else:
        app = FastAPI()
        app.include_router(organize_tasks.router)
        @app.get('/api/health')
        def health():
            return {'code': 200}
        @app.get('/api/organize/scene/review')
        def reviews(limit: int = 200, offset: int = 0):
            return {'data': queue.page(limit=limit, offset=offset)}
        @app.on_event('shutdown')
        def shutdown():
            organize_tasks.shutdown()
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning', access_log=False, lifespan='off' if real else 'on')


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


async def measure(root, port, real=False):
    import httpx
    base = f'http://127.0.0.1:{port}'
    timings = {'health': [], 'page': [], 'create': [], 'cancel': []}
    errors = []
    body = {'kind': 'scene_reason', 'idempotency_key': 'responsiveness', 'params': {
        'input_dir': str(root / 'photos'), 'knowledge_dir': str(root / 'knowledge'),
        'output_file': str(root / 'evidence.jsonl'), 'use_embedder': real}}
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        for _ in range(600):
            try:
                if (await client.get('/api/health')).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError('benchmark server did not start')
        for _ in range(5):
            start = time.perf_counter()
            response = await client.post('/api/organize/tasks', json=body)
            timings['create'].append(time.perf_counter() - start)
            response.raise_for_status()
            task_id = response.json()['data']['task_id']
        start_load = time.perf_counter()
        async def queries(client_index):
            async with httpx.AsyncClient(base_url=base, timeout=5) as connection:
                for index in range(60):
                    await asyncio.sleep(max(0, start_load + index + client_index * 0.2 - time.perf_counter()))
                    kind = 'health' if (index + client_index) % 2 == 0 else 'page'
                    path = '/api/health' if kind == 'health' else f'/api/organize/scene/review?limit=200&offset={(index % 50) * 200}&review_dir={root / "review"}'
                    started = time.perf_counter()
                    try:
                        response = await connection.get(path)
                        response.raise_for_status()
                        if kind == 'page':
                            data = response.json()['data']
                            assert len(data['records']) == 200 and sum(data['summary'].values()) == 10000
                        timings[kind].append(time.perf_counter() - started)
                    except Exception as exc:
                        errors.append(str(exc))
        await asyncio.gather(*(queries(i) for i in range(5)))
        await asyncio.sleep(max(0, start_load + 60 - time.perf_counter()))
        before = (await client.get('/api/organize/tasks/' + task_id)).json()['data']
        started = time.perf_counter()
        acknowledgement = (await client.post(f'/api/organize/tasks/{task_id}/cancel')).json()['data']
        timings['cancel'].append(time.perf_counter() - started)
        for _ in range(100):
            final = (await client.get('/api/organize/tasks/' + task_id)).json()['data']
            if final['status'] == 'cancelled':
                break
            await asyncio.sleep(0.02)
        cancellation_elapsed = time.perf_counter() - started
    measurements = {kind: {'count': len(values), 'median_seconds': percentile(values, .5),
                          'p95_seconds': percentile(values, .95), 'max_seconds': max(values)}
                    for kind, values in timings.items()}
    report = {'status': 'measured', 'clients': 5, 'duration_seconds': 60, 'requests_per_second': 5,
            'records': 10000, 'image_items': len(list((root / 'photos').iterdir())), 'measurements': measurements, 'errors': errors,
            'task_before_cancel': before, 'cancel_acknowledgement': acknowledgement['status'],
            'cancel_final_seconds': cancellation_elapsed, 'task_final': final,
            'checks': {'query_p95': max(measurements['health']['p95_seconds'], measurements['page']['p95_seconds']) <= .5,
                       'create_p95': measurements['create']['p95_seconds'] <= 1,
                       'cancel_ack': measurements['cancel']['max_seconds'] <= 2,
                       'task_active_during_load': before['status'] == 'running', 'no_errors': not errors},
            'limitations': ['Controlled 250 ms blocking operation per image; no real GPU or model endpoint.',
                            '1000 tiny generated JPEGs measure API scheduling, not realistic input hashing costs.',
                            'Five creation samples include four idempotent duplicates; not sustained unique task admission.',
                            'Isolated task/review routers; does not prove full production server latency.']}
    if real:
        report['limitations'] = ['Real SigLIP2 on isolated copies of original images; no face model or model quality evaluation.',
                                'Full main API with logging middleware; unrelated production model startup disabled.',
                                'Five creation samples include four idempotent duplicates.',
                                'Other GPU processes can affect timings; model is warmed before requests.']
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--real-manifest', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.worker:
        worker(root, args.port, bool(args.real_manifest))
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('output must be empty')
    root.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from core_modules.organize.review_queue import ReviewQueue
    (root / 'photos').mkdir()
    (root / 'knowledge').mkdir()
    if args.real_manifest:
        manifest = json.loads(args.real_manifest.read_text())
        from core_modules.organize.task_artifacts import fingerprint
        for index, row in enumerate(manifest['images']):
            source = Path(row['path'])
            destination = root / 'photos' / (f'{index:04d}' + source.suffix)
            shutil.copy2(source, destination)
            if fingerprint(destination) != row['content_hash']:
                raise ValueError('manifest source content changed')
        (root / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        from core_modules.organize.play_knowledge import build_knowledge
        play = ROOT / 'data/ncpa_test/京剧《天下归心》'
        build_knowledge(str(play / '剧目信息.txt'), str(play), str(root / 'knowledge'))
    else:
        (root / 'knowledge' / 'entities.json').write_text('{"scenes":[],"roles":[]}')
        for index in range(1000):
            Image.new('RGB', (16, 16), (index % 256, 0, 0)).save(root / 'photos' / f'{index:04d}.jpg')
    ReviewQueue(str(root / 'review')).seed([{'image': f'{index:05d}.jpg', 'label': 'candidate'} for index in range(10000)])
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with (root / 'server.log').open('w') as log:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(root), '--port', str(port)]
        if args.real_manifest:
            command.extend(['--real-manifest', str(args.real_manifest.resolve())])
        source_files = ('tools/benchmark_task_responsiveness.py', 'core_modules/organize/task_queue.py',
                        'app/organize_tasks.py', 'app/main.py', 'core_modules/organize/review_store.py',
                        'core_modules/organize/review_queue.py', 'core_modules/organize/task_artifacts.py',
                        'core_modules/organize/task_versions.py', 'core_modules/organize/scene_reasoner.py',
                        'core_modules/organize/multimodal_evidence.py', 'core_modules/organize/llm_client.py',
                        'core_modules/image_search.py', 'core_modules/organize/feature_cache.py')
        measured_sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            report = asyncio.run(measure(root, port, bool(args.real_manifest)))
            report['source_sha256'] = measured_sources
            report['source_unchanged_during_measurement'] = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                                                               for name, digest in measured_sources.items())
            (root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({'checks': report['checks'], 'measurements': report['measurements'], 'output': str(root)}, indent=2), flush=True)
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == '__main__':
    main()
