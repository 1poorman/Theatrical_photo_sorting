"""Continue a real HTTP task through plan cancellation and publication recovery.

Reuses the isolated task/input directories of benchmark_face_task_http.py.
Instrumented handlers keep business sources and execution contracts unchanged.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT), str(ROOT / 'tools')]


def event(output, value):
    with (output / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(value, at=time.time()), ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import main
    import uvicorn
    from app import organize_tasks
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
    from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
    from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
    from core_modules.organize import face_cluster_task, face_database_task
    from core_modules.organize.task_artifacts import atomic_json
    organize_tasks.TASK_ROOT = args.resume_from / 'tasks'
    system = SimpleNamespace(detector=SCRFDDetector(SCRFD_MODEL_PATH, device='cuda'),
                             extractor=ArcFaceFeatureExtractor(ARCFACE_MODEL_PATH, device='cuda'))
    main.srv.face_recognition_model = system
    module = face_cluster_task if args.kind == 'face_cluster' else face_database_task
    task_class = module.FaceClusterTask if args.kind == 'face_cluster' else module.FaceDatabaseTask
    original_copy = module.copy_verified
    def copy(*values):
        started = time.perf_counter()
        new = original_copy(*values)
        event(args.output, {'phase': 'copy', 'image': str(values[1]), 'seconds': time.perf_counter() - started, 'new': new})
        marker = args.output / 'copy_failure_injected.json'
        if new and not marker.exists():
            atomic_json(marker, {'destination': str(values[1])})
            raise OSError('benchmark_interruption_after_first_copy')
        return new
    module.copy_verified = copy
    def factory(params):
        handler = task_class(params, system)
        original_process, original_advance, original_finish = handler.process, handler.advance, handler.finish
        def timed(operation, phase, **details):
            started = time.perf_counter()
            try:
                value = operation()
            except BaseException as exc:
                event(args.output, dict(details, phase=phase, seconds=time.perf_counter() - started, error=type(exc).__name__))
                raise
            event(args.output, dict(details, phase=phase, seconds=time.perf_counter() - started))
            return value
        def process(item):
            result = timed(lambda: original_process(item), item['stage'], image=item['image'],
                           cluster_id=item.get('cluster_id'))
            marker = args.output / 'crop_cancel_injected.json'
            if args.cancel_after_crop and item['stage'] == 'crops' and not marker.exists():
                atomic_json(marker, {'image': item['image']})
                organize_tasks.manager().cancel(args.task_id)
                event(args.output, {'phase': 'crop_cancel', 'image': item['image']})
            return result
        def advance():
            key = 'clusters' if args.kind == 'face_cluster' else 'anchor_plan'
            before = handler.context.checkpoint(key)
            changed = timed(original_advance, 'advance')
            marker = args.output / 'plan_cancel_injected.json'
            if before is None and handler.context.checkpoint(key) is not None and not marker.exists():
                atomic_json(marker, {'checkpoint': key})
                organize_tasks.manager().cancel(args.task_id)
                event(args.output, {'phase': 'plan_cancel'})
            return changed
        handler.process = process
        handler.advance = advance
        handler.finish = lambda rows: timed(lambda: original_finish(rows), 'finish')
        return handler
    setattr(organize_tasks, args.kind + '_handler', factory)
    uvicorn.run(main.app, host='127.0.0.1', port=args.port, access_log=False, log_level='warning', lifespan='off')


async def attempt(args, port, process):
    import httpx
    import psutil
    from benchmark_face_task_http import ready
    base = f'http://127.0.0.1:{port}'
    metrics = {'health': [], 'page': []}
    errors, rss, states = [], [], []
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        await ready(client)
        response = await client.post(f'/api/organize/tasks/{args.task_id}/resume')
        response.raise_for_status()
        started = time.perf_counter()
        while time.perf_counter() - started < 1200:
            status = (await client.get(f'/api/organize/tasks/{args.task_id}')).json()['data']
            states.append(status['status'])
            try:
                rss.append(psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if status['status'] in ('completed', 'completed_with_errors', 'failed', 'cancelled'):
                return {'status': status, 'seconds': time.perf_counter() - started, 'rss_peak_bytes': max(rss, default=0),
                        'query_seconds': metrics, 'errors': errors}
            batch_start = time.perf_counter()
            async def query(index):
                await asyncio.sleep(index * .2)
                name = 'health' if index % 2 == 0 else 'page'
                url = '/api/health' if name == 'health' else '/api/organize/scene/review'
                params = None if name == 'health' else {'review_dir': str(args.resume_from / 'review'), 'limit': 200, 'offset': 0}
                start = time.perf_counter()
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    if name == 'page':
                        data = response.json()['data']
                        assert len(data['records']) == 200 and sum(data['summary'].values()) == 10000
                    metrics[name].append(time.perf_counter() - start)
                except Exception as exc:
                    errors.append(type(exc).__name__ + ': ' + str(exc))
            await asyncio.gather(*(query(index) for index in range(5)))
            await asyncio.sleep(max(0, batch_start + 1 - time.perf_counter()))
    raise TimeoutError('face task completion timed out')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('face_cluster', 'face_database'), required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--cancel-after-crop', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--task-id')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    args.resume_from, args.output = args.resume_from.resolve(), args.output.resolve()
    if args.cancel_after_crop and args.kind != 'face_cluster':
        raise ValueError('crop cancellation requires face_cluster')
    if args.worker:
        worker(args)
        return
    from collections import Counter
    import sqlite3
    from core_modules.organize.task_artifacts import atomic_json, fingerprint
    from benchmark_task_responsiveness import percentile
    historical = json.loads((args.resume_from / 'summary.json').read_text())
    if historical['kind'] != args.kind or not all(historical['checks'].values()):
        raise ValueError('requires successful real HTTP report of the same task kind')
    for name, digest in historical['source_sha256'].items():
        if fingerprint(ROOT / name) != digest:
            raise ValueError('measured business source changed: ' + name)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty')
    manifest = json.loads((args.resume_from / 'input_manifest.json').read_text())['images']
    if any(args.output == Path(row['path']).parent or Path(row['path']).parent in args.output.parents
           or args.output in Path(row['path']).parents for row in manifest):
        raise ValueError('output must be outside input photos')
    with sqlite3.connect(args.resume_from / 'tasks/tasks.sqlite3') as connection:
        row = connection.execute('SELECT id, status FROM tasks ORDER BY created LIMIT 1').fetchone()
        args.task_id = row[0]
        if row[1] != 'cancelled':
            raise ValueError('expected cancelled HTTP task')
        completed = [(position, payload) for position, payload in connection.execute(
            "SELECT position, result FROM items WHERE task_id=? AND status='succeeded'", (args.task_id,))]
        original_inputs = [json.loads(payload) for position, payload in connection.execute(
            "SELECT position, input FROM items WHERE task_id=? AND status='succeeded'", (args.task_id,))]
    args.output.mkdir(parents=True, exist_ok=True)
    versions = dict(historical['source_sha256'], **{str(Path(__file__).resolve().relative_to(ROOT)): fingerprint(__file__)})
    attempts = []
    expected_states = ('cancelled', 'cancelled', 'failed', 'completed') if args.cancel_after_crop else ('cancelled', 'failed', 'completed')
    for index, expected in enumerate(expected_states):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--resume-from', str(args.resume_from),
                   '--output', str(args.output), '--kind', args.kind, '--task-id', args.task_id, '--port', str(port), '--gpu', str(args.gpu)]
        if args.cancel_after_crop:
            command.append('--cancel-after-crop')
        with (args.output / f'server_{index}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                result = asyncio.run(attempt(args, port, process))
                attempts.append(result)
                atomic_json(args.output / f'attempt_{index}.json', result)
                print(json.dumps({'attempt': index, 'status': result['status']['status'], 'seconds': result['seconds'],
                                  'succeeded': result['status']['succeeded'], 'pending': result['status']['pending'],
                                  'rss_peak_mib': result['rss_peak_bytes'] / 1024**2}), flush=True)
            finally:
                process.terminate()
                process.wait(timeout=20)
        if result['status']['status'] != expected:
            raise RuntimeError(f'expected {expected}: {result["status"]}')
    events = [json.loads(line) for line in (args.output / 'events.jsonl').read_text().splitlines()]
    phases = {}
    for phase in sorted({event['phase'] for event in events}):
        values = [event['seconds'] for event in events if event['phase'] == phase and 'seconds' in event]
        phases[phase] = {'calls': len(values), 'total_seconds': sum(values), 'max_seconds': max(values, default=0)}
    candidates = [event['image'] for event in events if event['phase'] in ('faces', 'anchors', 'gallery')]
    crops = [event['image'] for event in events if event['phase'] == 'crops']
    previews = [(event['image'], event.get('cluster_id')) for event in events if event['phase'] == 'preview']
    with sqlite3.connect(args.resume_from / 'tasks/tasks.sqlite3') as connection:
        rows = dict(connection.execute('SELECT position, result FROM items WHERE task_id=?', (args.task_id,)))
        results = [json.loads(payload) for payload in rows.values() if payload]
        plan_key = 'clusters' if args.kind == 'face_cluster' else 'anchor_plan'
        plan = json.loads(connection.execute('SELECT payload FROM checkpoints WHERE task_id=? AND name=?', (args.task_id, plan_key)).fetchone()[0])
    copies = [event for event in events if event['phase'] == 'copy']
    if args.kind == 'face_cluster':
        import base64
        import hashlib
        state = json.loads(Path(attempts[-1]['status']['result']['state_path']).read_text())
        planned = {row['cluster_id']: hashlib.sha256(base64.b64decode(row['jpeg'])).hexdigest()
                   for row in results if row.get('stage') == 'preview' and not row.get('error')}
        published = {cluster['id']: fingerprint(cluster['preview']) for cluster in state['clusters']}
        expected_crops = {member['img_path'] for cluster in plan for member in cluster['members'][:12]}
    else:
        planned = {anchor['relative']: anchor['sha256'] for anchor in plan['anchors']}
        gallery = Path(attempts[-1]['status']['result']['db_root'])
        published = {str(path.relative_to(gallery)): fingerprint(path) for path in gallery.rglob('*.jpg')}
        expected_crops = set()
    queries = {name: [value for attempt in attempts for value in attempt['query_seconds'][name]] for name in ('health', 'page')}
    metrics = {name: {'count': len(values), 'p95_seconds': percentile(values, .95), 'max_seconds': max(values)}
               for name, values in queries.items() if values}
    checks = {'checkpoint_payloads_unchanged': all(rows[position] == payload for position, payload in completed),
              'no_candidate_repeated': max(Counter(candidates).values(), default=0) <= 1,
              'no_previously_completed_candidate_reprocessed': not ({row['image'] for row in original_inputs
                  if row.get('stage') in ('faces', 'anchors', 'gallery')} & set(candidates)),
              'no_crop_repeated': max(Counter(crops).values(), default=0) <= 1,
              'all_selected_source_crops_completed': not args.cancel_after_crop or set(crops) == expected_crops,
              'no_preview_repeated': max(Counter(previews).values(), default=0) <= 1,
              'crop_cancelled_then_resumed': not args.cancel_after_crop or (args.output / 'crop_cancel_injected.json').exists(),
              'published_paths_and_bytes_match_plan': published == planned,
              'each_file_created_once': sum(event['new'] for event in copies) == len(published),
              'one_partial_file_reused': sum(not event['new'] for event in copies) == 1,
              'plan_cancelled_then_resumed': (args.output / 'plan_cancel_injected.json').exists(),
              'partial_copy_failed_then_resumed': (args.output / 'copy_failure_injected.json').exists(),
              'completed_without_item_failures': attempts[-1]['status']['failed'] == 0,
              'query_p95': all(name in metrics and metrics[name]['p95_seconds'] <= .5 for name in ('health', 'page')),
              'no_query_errors': not any(attempt['errors'] for attempt in attempts),
              'source_photos_and_copies_unchanged': all(fingerprint(row['source']) == fingerprint(row['path']) == row['sha256'] for row in manifest),
              'measured_code_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in versions.items())}
    report = {'kind': args.kind, 'resume_from': str(args.resume_from), 'task_id': args.task_id,
              'n_images': len(manifest), 'source_sha256': versions, 'attempts': attempts,
              'cancel_after_crop': args.cancel_after_crop,
              'phase_metrics': phases, 'query_metrics': metrics, 'checks': checks,
              'final_result': attempts[-1]['status']['result'], 'physical_gpu': args.gpu,
              'limitations': ['Continuation of earlier 200-image run; not a fresh cold baseline.',
                              'Handler wrappers cancel after the plan and fail after the first copied file.',
                              f'{len(attempts)} server processes verify saved plans, crops and partial outputs across restart.',
                              'Five scheduled queries/second until completion; short publication phases have sparse HTTP samples.',
                              'RSS sampled once/second; shared GPU; no identity GT or controlled performance comparison.']}
    atomic_json(args.output / 'summary.json', report)
    print(json.dumps({'checks': checks, 'phase_metrics': phases, 'query_metrics': metrics, 'result': report['final_result']}, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
