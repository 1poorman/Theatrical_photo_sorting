"""Real face-task HTTP responsiveness on isolated, content-verified copies."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]


def worker(root, port, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import main
    import uvicorn
    from app import organize_tasks
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
    from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
    from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
    organize_tasks.TASK_ROOT = root / 'tasks'
    main.srv.face_recognition_model = SimpleNamespace(
        detector=SCRFDDetector(SCRFD_MODEL_PATH, device='cuda'),
        extractor=ArcFaceFeatureExtractor(ARCFACE_MODEL_PATH, device='cuda'))
    # No other production models or official gallery/Elasticsearch startup.
    uvicorn.run(main.app, host='127.0.0.1', port=port, log_level='warning', access_log=False, lifespan='off')


async def ready(client):
    import httpx
    for _ in range(1200):
        try:
            if (await client.get('/api/health')).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(.1)
    raise TimeoutError('face benchmark server did not start')


async def measure(root, port, kind):
    import httpx
    from benchmark_task_responsiveness import percentile
    base = f'http://127.0.0.1:{port}'
    dirs = json.loads((root / 'input_manifest.json').read_text())['copied_directories']
    def body(index):
        params = {'input_dir': str(root / 'cluster_photos'), 'output_dir': str(root / f'cluster_{index}')}
        if kind == 'face_database':
            params = {'organized_dirs': dirs, 'db_root': str(root / f'gallery_{index}'),
                      'output_dir': str(root / f'report_{index}')}
        return {'kind': kind, 'idempotency_key': f'{kind}-{index}', 'params': params}
    timings = {'health': [], 'page': [], 'create': [], 'cancel': []}
    errors = []
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        await ready(client)
        started = time.perf_counter()
        response = await client.post('/api/organize/tasks', json=body(0))
        first_create_seconds = time.perf_counter() - started
        response.raise_for_status()
        task_id = response.json()['data']['task_id']
        duplicate = await client.post('/api/organize/tasks', json=body(0))
        duplicate.raise_for_status()
        duplicate_ok = duplicate.json()['data']['task_id'] == task_id and not duplicate.json()['data']['created']
        for index in range(1, 6):
            started = time.perf_counter()
            response = await client.post('/api/organize/tasks', json=body(index))
            timings['create'].append(time.perf_counter() - started)
            response.raise_for_status()
            extra_id = response.json()['data']['task_id']
            cancelled = await client.post(f'/api/organize/tasks/{extra_id}/cancel')
            cancelled.raise_for_status()
        initial = (await client.get('/api/organize/tasks/' + task_id)).json()['data']
        start_load = time.perf_counter()
        async def queries(client_index):
            async with httpx.AsyncClient(base_url=base, timeout=10) as connection:
                for index in range(60):
                    await asyncio.sleep(max(0, start_load + index + client_index * .2 - time.perf_counter()))
                    query = 'health' if (index + client_index) % 2 == 0 else 'page'
                    path = '/api/health' if query == 'health' else '/api/organize/scene/review'
                    params = None if query == 'health' else {'review_dir': str(root / 'review'), 'limit': 200, 'offset': index % 50 * 200}
                    started = time.perf_counter()
                    try:
                        response = await connection.get(path, params=params)
                        response.raise_for_status()
                        if query == 'page':
                            data = response.json()['data']
                            assert len(data['records']) == 200 and sum(data['summary'].values()) == 10000
                        timings[query].append(time.perf_counter() - started)
                    except Exception as exc:
                        errors.append(f'{query}: {type(exc).__name__}: {exc}')
        await asyncio.gather(*(queries(i) for i in range(5)))
        await asyncio.sleep(max(0, start_load + 60 - time.perf_counter()))
        before = (await client.get('/api/organize/tasks/' + task_id)).json()['data']
        started = time.perf_counter()
        response = await client.post(f'/api/organize/tasks/{task_id}/cancel')
        timings['cancel'].append(time.perf_counter() - started)
        response.raise_for_status()
        acknowledgement = response.json()['data']
        deadline = time.perf_counter() + 60
        while True:
            final = (await client.get('/api/organize/tasks/' + task_id)).json()['data']
            if final['status'] in ('cancelled', 'completed', 'completed_with_errors', 'failed'):
                break
            if time.perf_counter() > deadline:
                raise TimeoutError('task cancellation did not finish')
            await asyncio.sleep(.02)
        cancellation_elapsed = time.perf_counter() - started
    metrics = {query: {'count': len(values), 'median_seconds': percentile(values, .5),
                       'p95_seconds': percentile(values, .95), 'max_seconds': max(values)}
               for query, values in timings.items() if values}
    checks = {'query_p95': all(query in metrics and metrics[query]['p95_seconds'] <= .5 for query in ('health', 'page')),
              'create_p95': metrics['create']['p95_seconds'] <= 1,
              'cancel_ack': metrics['cancel']['max_seconds'] <= 2 and acknowledgement['status'] in ('cancel_requested', 'cancelled'),
              'task_active_during_load': initial['status'] == before['status'] == 'running',
              'cancelled': final['status'] == 'cancelled',
              'no_next_item_after_cancel': final['succeeded'] <= acknowledgement['succeeded'] + 1,
              'idempotent_duplicate': duplicate_ok,
              'no_query_errors': not errors and len(timings['health']) + len(timings['page']) == 300}
    return {'kind': kind, 'clients': 5, 'duration_seconds': 60, 'requests_per_second': 5,
            'records': 10000, 'measurements': metrics, 'errors': errors, 'checks': checks,
            'first_unique_create_seconds': first_create_seconds,
            'task_initial': initial, 'task_before_cancel': before, 'task_final': final,
            'cancel_acknowledgement': acknowledgement['status'], 'cancel_final_seconds': cancellation_elapsed,
            'limitations': ['Real SCRFD/ArcFace on verified copies; official gallery and original photos read only.',
                            'Full main API; unrelated production startup disabled; no model quality evaluation.',
                            'Five unique creation samples while first task runs; each additional task is immediately cancelled.',
                            'Models loaded before requests; first real inference is not prewarmed.',
                            'One run on a shared GPU; no controlled performance comparison.',
                            'If the task completes before 60 seconds, active-load/cancellation checks fail explicitly.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--kind', choices=('face_cluster', 'face_database'), required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.worker:
        worker(root, args.port, args.gpu)
        return
    from core_modules.organize.task_artifacts import atomic_json, fingerprint
    from core_modules.organize.review_queue import ReviewQueue
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['images']
    if len(rows) < 200 or len({row['content_hash'] for row in rows}) != len(rows):
        raise ValueError('at least 200 unique images required')
    if root.exists() and any(root.iterdir()):
        raise ValueError('output must be empty')
    if any(root == Path(row['path']).parent or root in Path(row['path']).resolve().parents
           or Path(row['path']).resolve().parent in root.parents for row in rows):
        raise ValueError('output must be outside source directories')
    root.mkdir(parents=True, exist_ok=True)
    copied, directories = [], {}
    for index, row in enumerate(rows):
        source = Path(row['path']).resolve()
        if fingerprint(source) != row['content_hash']:
            raise ValueError('manifest source content changed')
        if args.kind == 'face_cluster':
            directory = root / 'cluster_photos'
            destination = directory / (f'{index:04d}' + source.suffix)
        else:
            if str(source.parent) not in directories:
                directories[str(source.parent)] = root / 'photos' / f'directory_{len(directories):02d}' / source.parent.name
            directory = directories[str(source.parent)]
            destination = directory / source.name
        directory.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError('duplicate destination filename')
        shutil.copy2(source, destination)
        if fingerprint(destination) != row['content_hash']:
            raise ValueError('copied content changed')
        copied.append({'source': str(source), 'path': str(destination), 'sha256': row['content_hash']})
    atomic_json(root / 'input_manifest.json', {'images': copied, 'copied_directories': [str(path) for path in directories.values()]})
    ReviewQueue(str(root / 'review')).seed([{'image': f'{index:05d}.jpg', 'label': 'candidate'} for index in range(10000)])
    measured = ['tools/benchmark_face_task_http.py', 'tools/benchmark_task_responsiveness.py',
                'app/main.py', 'app/organize_tasks.py', 'core_modules/organize/task_queue.py',
                'core_modules/organize/task_artifacts.py', 'core_modules/organize/task_versions.py',
                'core_modules/organize/face_cluster_task.py', 'core_modules/organize/face_cluster.py',
                'core_modules/organize/face_database_task.py', 'core_modules/organize/face_db_builder.py',
                'core_modules/organize/review_queue.py', 'core_modules/organize/review_store.py']
    version = {name: fingerprint(ROOT / name) for name in measured}
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with (root / 'server.log').open('w') as log:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(root),
                   '--manifest', str(args.manifest.resolve()), '--kind', args.kind, '--gpu', str(args.gpu), '--port', str(port)]
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            report = asyncio.run(measure(root, port, args.kind))
            report.update(source_sha256=version, physical_gpu=args.gpu, image_items=len(rows),
                          original_directories=len({str(Path(row['path']).parent) for row in rows}))
            report['checks']['source_photos_and_copies_unchanged'] = all(
                fingerprint(row['source']) == fingerprint(row['path']) == row['sha256'] for row in copied)
            report['checks']['measured_code_unchanged'] = all(fingerprint(ROOT / name) == value for name, value in version.items())
            atomic_json(root / 'summary.json', report)
            print(json.dumps({'kind': args.kind, 'checks': report['checks'], 'measurements': report['measurements'],
                              'task_before_cancel': report['task_before_cancel']['status'],
                              'cancel_final_seconds': report['cancel_final_seconds'], 'output': str(root)}, indent=2), flush=True)
        finally:
            process.terminate()
            process.wait(timeout=20)


if __name__ == '__main__':
    main()
