"""Measure persistent HTTP reason/review/export/organize costs on raw copies.

Five warmed service runs use empty independent feature caches. Scripted decisions
reuse existing photo labels only in disposable output; they are not new GT.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]
from core_modules.organize.task_artifacts import atomic_json, fingerprint, gallery_hashes
from tools.benchmark_feature_pipeline import Resources, gpu_environment


def worker(args):
    started = time.perf_counter()
    os.environ['TRANS_TASK_DIR'] = str(args.output / 'tasks')
    os.environ['TRANS_OUTPUT_DIR'] = str(args.output / 'runtime')
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.cuda.set_device(0)
    import main
    import uvicorn
    from core_modules.image_search import ImageEmbedder
    from core_modules.organize.smart_organizer import SmartOrganizer
    from core_modules.face_recognition.face_recognition import FaceRecognitionSystem
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
    from core_modules.organize.task_versions import model_identity
    inputs = json.loads((args.output / 'inputs.json').read_text())
    model = ImageEmbedder('siglip2_base', device='cuda:0', image_backend='processor')
    if not type(model.model).__name__.startswith('Siglip'):
        raise RuntimeError('refusing_fallback_model')
    system = FaceRecognitionSystem(SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH, device='cuda')
    main.srv.face_recognition_model = system
    main.srv.pose_model = None
    main.srv.organize_models['embedder'] = model
    photos = [row['copy'] for project in inputs['projects'] for row in project['images']]
    model.extract_features_batch(photos[:8], sub_batch_size=8)
    model.encode_text_batch(['warmup'])
    organizer = SmartOrganizer(system, str(args.output / 'gallery'))
    organizer._load_library()
    import cv2
    from core_modules.tools.image_io import resize_max_side
    organizer._recognize_image(resize_max_side(cv2.imread(photos[0]), 1920), .55)
    contract = {
        'embedding': model_identity(model), 'face': model_identity(system),
        'face_providers': system.extractor.session.get_providers(),
        'shot': None, 'classify_role': False, 'image_backend': 'processor',
        'torch_threads': torch.get_num_threads(), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}
    contract['warmed_setup_seconds'] = time.perf_counter() - started
    atomic_json(args.output / 'model_contract.json', contract)
    counts = {'image': 0, 'text': 0}
    handles = []
    for kind, module in [('image', model.model.vision_model), ('text', model.model.text_model)]:
        def count(module, values, kind=kind):
            counts[kind] += 1
        handles.append(module.register_forward_pre_hook(count))
    resources = Resources(0)
    resources.__enter__()

    @main.app.post('/benchmark/reset/{repeat}')
    def reset(repeat: int):
        main.FEATURE_CACHE_DIR = str(args.output / 'features' / str(repeat))
        if Path(main.FEATURE_CACHE_DIR).exists():
            raise ValueError('feature_cache_must_be_new')
        counts.update(image=0, text=0)
        resources.peak_rss = resources.peak_gpu = 0
        return {'gpu_before': gpu_environment(0)}

    @main.app.get('/benchmark/counters')
    def counters():
        return {'forwards': dict(counts), 'rss_peak_bytes': resources.peak_rss,
                'whole_device_peak_bytes': resources.peak_gpu}

    try:
        uvicorn.run(main.app, host='127.0.0.1', port=args.port, lifespan='off',
                    access_log=False, log_level='warning')
    finally:
        from app import organize_tasks
        organize_tasks.shutdown()
        resources.__exit__()
        for handle in handles:
            handle.remove()


async def measure(args, process):
    import httpx
    inputs = json.loads((args.output / 'inputs.json').read_text())
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{args.port}', timeout=60) as client:
        for _ in range(1800):
            if process.poll() is not None:
                raise RuntimeError('benchmark_server_exited_' + str(process.returncode))
            try:
                response = await client.get('/api/health')
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(.1)
        else:
            raise TimeoutError('benchmark_server_start_timeout')

        async def call(path, **kwargs):
            response = await client.post(path, **kwargs)
            response.raise_for_status()
            value = response.json()
            if value.get('code', 200) not in (200, 202):
                raise RuntimeError('benchmark_application_error')
            return value.get('data', value)

        async def task(kind, params, key):
            start = time.perf_counter()
            created = await call('/api/organize/tasks', json={
                'kind': kind, 'params': params, 'idempotency_key': key})
            admission = time.perf_counter() - start
            polls = 0
            deadline = start + 3600
            while time.perf_counter() < deadline:
                response = await client.get('/api/organize/tasks/' + created['task_id'])
                response.raise_for_status()
                row = response.json()['data']
                polls += 1
                if row['status'] in ('completed', 'completed_with_errors', 'failed', 'cancelled'):
                    if row['status'] != 'completed':
                        atomic_json(args.output / 'failed_task.json', row)
                        raise RuntimeError('benchmark_task_' + row['status'])
                    return row, {'seconds': time.perf_counter() - start,
                                 'admission_seconds': admission, 'polls': polls}
                await asyncio.sleep(.1)
            raise TimeoutError('benchmark_task_timeout')

        runs = []
        for repeat in range(args.repeats):
            reset = await call('/benchmark/reset/' + str(repeat))
            projects = []
            for number, project in enumerate(inputs['projects']):
                output = args.output / 'runs' / str(repeat) / str(number)
                output.mkdir(parents=True)
                params = {'input_dir': project['photo_root'], 'knowledge_dir': project['knowledge_root'],
                          'output_file': str(output / 'evidence.jsonl'), 'use_embedder': True, 'use_llm': False}
                start = time.perf_counter()
                reason, reason_time = await task('scene_reason', params, f'reason-{repeat}-{number}')
                before = (await client.get('/benchmark/counters')).json()['forwards']
                review = str(output / 'review')
                stage = time.perf_counter()
                await call('/api/organize/scene/review/seed', data={
                    'review_dir': review, 'evidence_file': params['output_file']})
                first, second = project['images'][:2]
                for image, action in [(first, 'modify'), (second, 'reject')]:
                    await call('/api/organize/scene/review', data={
                        'review_dir': review, 'image': image['copy'], 'action': action,
                        'scene_id': image['scene_id'], 'label': image['scene_label'],
                        'reviewer': 'scripted-cost-probe-not-new-GT', 'expected_revision': 0})
                review_seconds = time.perf_counter() - stage
                stage = time.perf_counter()
                exported = await call('/api/organize/scene/review/export', data={'review_dir': review})
                export_seconds = time.perf_counter() - stage
                organized, organize_time = await task('organize', {
                    'input_dir': project['photo_root'], 'output_dir': str(output / 'organized'),
                    'db_root': str(args.output / 'gallery'), 'keep_per_bucket': 2, 'gap_seconds': 300,
                    'classify_role': False, 'scene_evidence_file': str(output / 'review/review_queue.jsonl')},
                    f'organize-{repeat}-{number}')
                elapsed = time.perf_counter() - start
                after = (await client.get('/benchmark/counters')).json()['forwards']
                report = json.loads(Path(organized['result']['report_path']).read_text())
                truth = {row['copy']: row for row in project['images']}
                outputs = [{'source': row['image'], 'name': row['organized_name'],
                            'content_equal': fingerprint(output / 'organized/organized' / row['organized_name']) ==
                                             truth[row['image']]['sha256']} for row in report['records']]
                labels = json.loads(Path(exported['path']).read_text())
                record = {'play_id': project['play_id'], 'seconds': elapsed,
                    'reason': reason_time, 'review_seconds': review_seconds, 'export_seconds': export_seconds,
                    'organize': organize_time, 'organize_forwards': {key: after[key] - before[key] for key in before},
                    'kept': report['kept'], 'discarded': report['discarded'], 'failed': report['failed'],
                    'outputs': outputs, 'reason_task': reason['task_id'], 'organize_task': organized['task_id'],
                    'checks': {'reason_count': reason['result']['n_images'] == len(truth),
                        'real_visual_retrieval': reason['result']['n_visual'] == len(truth),
                        'zero_organize_forwards': before == after,
                        'only_approved_exported': labels == {first['copy']: first['scene_label']},
                        'conservation': report['total_input'] == report['kept'] + report['discarded'] + report['failed'],
                        'no_errors': report['failed'] == 0,
                        'output_count': report['copied'] == len(outputs),
                        'output_content': all(row['content_equal'] for row in outputs)}}
                projects.append(record)
                atomic_json(output / 'costs.json', record)
                print(json.dumps({'repeat': repeat + 1, 'project': number + 1,
                    'seconds': elapsed, 'checks': record['checks']}), flush=True)
            counters = (await client.get('/benchmark/counters')).json()
            run = {'repeat': repeat + 1, 'seconds': sum(row['seconds'] for row in projects),
                   'projects': projects, 'resources': counters, 'gpu_before': reset['gpu_before'],
                   'gpu_after': gpu_environment(0)}
            runs.append(run)
            atomic_json(args.output / 'partial_results.json', runs)
        return runs


@contextmanager
def monitor(root, pid):
    stop = threading.Event()
    def sample():
        with (root / 'hardware_samples.jsonl').open('w') as stream:
            while not stop.is_set():
                stream.write(json.dumps({'time': time.time(), 'server_pid': pid,
                    'cpu_load_average': os.getloadavg(), 'gpu': gpu_environment(0)}) + '\n')
                stream.flush()
                stop.wait(2)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, default=ROOT / 'data/face_database')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if min(args.repeats, args.torch_threads, args.limit) < 1:
        parser.error('positive_counts_required')
    if args.worker:
        worker(args)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('new_empty_output_required')
    old = json.loads(args.input_manifest.read_text())
    selected = [(project, row) for index in range(max(len(p['images']) for p in old['projects']))
                for project in old['projects'] for row in project['images'][index:index + 1]][:args.limit]
    if len(selected) != args.limit or len({row['sha256'] for _, row in selected}) != args.limit:
        raise ValueError('insufficient_unique_photos')
    args.output.mkdir(parents=True)
    projects = []
    for number, old_project in enumerate(old['projects']):
        rows = [row for project, row in selected if project is old_project]
        if not rows:
            continue
        if len(rows) < 2:
            raise ValueError('two_photos_per_project_required')
        directory = args.output / 'inputs' / str(number)
        photos = directory / 'photos'
        photos.mkdir(parents=True)
        knowledge = directory / 'knowledge'
        knowledge.mkdir()
        source_knowledge = Path(old_project['root']) / 'knowledge/entities.json'
        shutil.copy2(source_knowledge, knowledge / 'entities.json')
        images = []
        for row in rows:
            source = Path(row['source']).resolve()
            if '整理完成' in str(source) or fingerprint(source) != row['sha256']:
                raise ValueError('raw_input_scope_or_content_changed')
            target = photos / source.name
            if target.exists():
                raise ValueError('duplicate_photo_basename')
            shutil.copy2(source, target)
            images.append(dict(row, copy=str(target)))
        projects.append({'play_id': old_project['play_id'], 'images': images,
                         'photo_root': str(photos), 'knowledge_root': str(knowledge),
                         'knowledge_sha256': fingerprint(source_knowledge)})
    gallery = gallery_hashes(args.gallery)
    shutil.copytree(args.gallery, args.output / 'gallery')
    atomic_json(args.output / 'inputs.json', {'projects': projects, 'gallery_sha256': gallery,
                                            'source_manifest_sha256': fingerprint(args.input_manifest)})
    sources = [path for folder in ('core_modules', 'config', 'app') for path in (ROOT / folder).rglob('*.py')]
    sources.append(Path(__file__).resolve())
    sources.append(ROOT / 'tools/benchmark_feature_pipeline.py')
    hashes = {str(path.relative_to(ROOT)): fingerprint(path) for path in sources}
    for path, digest in hashes.items():
        snapshot = args.output / 'source_snapshot' / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, snapshot)
        if fingerprint(snapshot) != digest:
            raise ValueError('source_changed_during_snapshot')
    state = {'status': 'running', 'command': [sys.executable, *sys.argv], 'source_sha256': hashes,
             'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'started_at': datetime.now(timezone.utc).isoformat()}
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        args.port = listener.getsockname()[1]
    atomic_json(args.output / 'run_state.json', state)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--port', str(args.port)]
    with (args.output / 'server.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
        try:
            with monitor(args.output, process.pid):
                runs = asyncio.run(measure(args, process))
        except Exception as exc:
            state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            atomic_json(args.output / 'run_state.json', state)
            raise
        finally:
            process.terminate()
            process.wait(timeout=30)
    checks = {'all_workflow_checks': all(all(project['checks'].values()) for run in runs for project in run['projects']),
              'originals_and_copies_unchanged': all(fingerprint(row['source']) == fingerprint(row['copy']) == row['sha256']
                                                 for project in projects for row in project['images']),
              'gallery_unchanged': gallery_hashes(args.gallery) == gallery_hashes(args.output / 'gallery') == gallery,
              'sources_unchanged': all(fingerprint(ROOT / path) == sha for path, sha in hashes.items()),
              'output_decisions_repeat_equal': len({hashlib.sha256(json.dumps([
                  {'play': p['play_id'], 'kept': p['kept'], 'names': [o['name'] for o in p['outputs']]}
                  for p in run['projects']], sort_keys=True).encode()).hexdigest() for run in runs}) == 1}
    medians = {'workflow': statistics.median(run['seconds'] for run in runs)}
    for stage in ('reason', 'organize', 'review', 'export'):
        values = [sum(p[stage]['seconds'] if stage in ('reason', 'organize') else p[stage + '_seconds']
                      for p in run['projects']) for run in runs]
        medians[stage] = statistics.median(values)
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'input_count': args.limit, 'project_count': len(projects), 'repeats': args.repeats,
        'median_seconds': medians, 'runs': runs,
        'warmed_setup_seconds': json.loads((args.output / 'model_contract.json').read_text())['warmed_setup_seconds'],
        'formal_U2_U3_U8': 'not_passed',
        'limitations': ['Same warmed service, independent empty feature cache per run, actual persistent HTTP tasks.',
            'Existing confirmed photo labels reused by scripted probes; no new human GT or final test.',
            'SigLIP2 processor/default feature workers; real SCRFD/ArcFace; shot and role classifiers disabled.',
            'Shared CPU/GPU/storage; no baseline comparison or controlled speedup claim.',
            'End-to-end time includes HTTP, polling, scripted review and export; excludes setup, model warmup and integrity audits.',
            'Production lifespan startup and remote LLM disabled; deployment preparation and human review time not measured.']}
    atomic_json(args.output / 'summary.json', summary)
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output / 'run_state.json', state)
    print(json.dumps({'status': summary['status'], 'median_seconds': medians, 'checks': checks}), flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
