"""Real-model recovery/equivalence smoke; not an identity quality benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]


def wait(queue, task_id, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = queue.get(task_id)
        if task['status'] in ('completed', 'completed_with_errors', 'cancelled', 'failed'):
            return task
        time.sleep(.05)
    raise TimeoutError('cluster task timed out')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    from app.organize_tasks import prepare_face_cluster
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
    from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
    from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
    from core_modules.organize.face_cluster_task import FaceClusterTask
    from core_modules.organize.task_artifacts import atomic_json, fingerprint
    from core_modules.organize.task_queue import TaskQueue
    detector = SCRFDDetector(SCRFD_MODEL_PATH, device=args.device)
    extractor = ArcFaceFeatureExtractor(ARCFACE_MODEL_PATH, device=args.device)
    system = SimpleNamespace(detector=detector, extractor=extractor)
    params, items = prepare_face_cluster({'input_dir': str(args.input_dir.resolve()),
                                         'output_dir': str(output / 'resumed')})
    source_hashes = dict(params['input_hashes'])
    measured = ['tools/benchmark_face_cluster_recovery.py', 'app/organize_tasks.py',
                'core_modules/organize/face_cluster.py', 'core_modules/organize/face_cluster_task.py',
                'core_modules/organize/task_queue.py', 'core_modules/organize/task_artifacts.py',
                'core_modules/organize/task_versions.py']
    source_version = {name: fingerprint(ROOT / name) for name in measured}
    reached, release = threading.Event(), threading.Event()
    detections = []
    def factory(params):
        handler = FaceClusterTask(params, system)
        original = handler.process
        def process(item):
            result = original(item)
            if item['stage'] == 'faces':
                detections.append(item['image'])
                if len(detections) == 1:
                    reached.set()
                    release.wait(60)
            return result
        handler.process = process
        return handler
    started = time.perf_counter()
    queue = TaskQueue(output / 'tasks', {'face_cluster': factory})
    try:
        task, _ = queue.submit('face_cluster', params, items)
        task_id = task['task_id']
        try:
            if not reached.wait(60):
                raise TimeoutError('first image failed to reach checkpoint')
            queue.cancel(task_id)
        finally:
            release.set()
        cancelled = wait(queue, task_id)
        if cancelled['status'] != 'cancelled':
            raise RuntimeError(cancelled)
    finally:
        queue.close(60)
    queue = TaskQueue(output / 'tasks', {'face_cluster': factory})
    try:
        queue.resume(task_id)
        final = wait(queue, task_id)
    finally:
        queue.close(60)
    task_seconds = time.perf_counter() - started
    if final['status'] != 'completed':
        raise RuntimeError(final)
    state = json.loads(Path(final['result']['state_path']).read_text())
    # Compare with the pre-upgrade per-face implementation from the pushed commit.
    legacy_source = subprocess.check_output(['git', 'show', 'f7cb66a:core_modules/organize/face_cluster.py'], text=True)
    namespace = {'__name__': 'cluster_recovery_legacy', '__file__': str(ROOT / 'core_modules/organize/face_cluster.py')}
    exec(compile(legacy_source, namespace['__file__'], 'exec'), namespace)
    started = time.perf_counter()
    baseline = namespace['FaceClusterBuilder'](system).scan(str(args.input_dir.resolve()), str(output / 'baseline'))
    baseline_seconds = time.perf_counter() - started
    def members(state):
        return sorted(sorted((member['image'], tuple(member['bbox']), tuple(tuple(row) for row in member['landmark']))
                             for member in cluster['members']) for cluster in state['clusters'])
    checks = {'sources_unchanged': all(fingerprint(path) == digest for path, digest in source_hashes.items()),
              'code_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in source_version.items()),
              'no_detection_repeated': len(detections) == len(set(detections)) == len(items),
              'face_count_equal': state['n_faces'] == baseline['n_faces'],
              'cluster_members_equal': members(state) == members(baseline),
              'no_failures': state['failed'] == 0}
    report = {'checks': checks, 'n_images': len(items), 'n_faces': state['n_faces'],
              'n_clusters': len(state['clusters']), 'cancelled_succeeded': cancelled['succeeded'],
              'task_seconds_including_recovery': task_seconds, 'baseline_seconds': baseline_seconds,
              'source_sha256': source_version, 'device': args.device, 'physical_gpu': args.gpu,
              'model_contract': state['execution_contract'], 'baseline_commit': 'f7cb66a',
              'limitations': ['One directory, one run; no quality GT or performance acceptance.',
                              'Task executor restarted after cancellation; this is not a process-kill experiment.',
                              'Competing GPU load and sequential measurements can affect elapsed time.']}
    atomic_json(output / 'summary.json', report)
    print(json.dumps({key: report[key] for key in ('checks', 'n_images', 'n_faces', 'n_clusters',
                     'task_seconds_including_recovery', 'baseline_seconds')}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
