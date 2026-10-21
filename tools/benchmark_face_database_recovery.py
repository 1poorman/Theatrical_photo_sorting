"""Isolated real-model annotated-gallery equivalence/recovery smoke."""
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


def wait(queue, task_id, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = queue.get(task_id)
        if task['status'] in ('completed', 'completed_with_errors', 'cancelled', 'failed'):
            return task
        time.sleep(.05)
    raise TimeoutError('face database task timed out')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    source, output = args.input_dir.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('output must be outside input photos')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    from unittest.mock import patch
    from app import organize_tasks
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
    from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
    from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
    from core_modules.organize.face_database_task import FaceDatabaseTask
    from core_modules.organize.task_artifacts import atomic_json, fingerprint, face_database_images
    from core_modules.organize.task_queue import TaskQueue
    system = SimpleNamespace(detector=SCRFDDetector(SCRFD_MODEL_PATH, device=args.device),
                             extractor=ArcFaceFeatureExtractor(ARCFACE_MODEL_PATH, device=args.device))
    base = {'organized_dirs': [str(source)], 'db_root': str(output / 'resumed'),
            'output_dir': str(output / 'report')}
    params, manifest = organize_tasks.prepare_face_database(base, defer_hash=True)
    source_hashes = {item['image']: fingerprint(item['image']) for item in manifest}
    params['_manifest_items'] = manifest
    items = [dict(item, stage='fingerprint') for item in manifest]
    measured = ['tools/benchmark_face_database_recovery.py', 'app/organize_tasks.py',
                'core_modules/organize/face_database_task.py', 'core_modules/organize/face_db_builder.py',
                'core_modules/organize/filename_parser.py', 'core_modules/organize/task_queue.py',
                'core_modules/organize/task_artifacts.py', 'core_modules/organize/task_versions.py']
    version = {name: fingerprint(ROOT / name) for name in measured}
    reached, release = threading.Event(), threading.Event()
    processed = []
    def factory(task_params):
        handler = FaceDatabaseTask(task_params, system)
        original = handler.process
        def process(item):
            result = original(item)
            processed.append(item['image'])
            if len(processed) == 1:
                reached.set()
                release.wait(60)
            return result
        handler.process = process
        return handler
    handlers = {'face_database': lambda p: organize_tasks.deferred_handler('face_database', p)}
    started = time.perf_counter()
    with patch.object(organize_tasks, 'face_database_handler', factory):
        queue = TaskQueue(output / 'tasks', handlers)
        try:
            task, _ = queue.submit('face_database', params, items, output=[base['db_root'], base['output_dir']])
            task_id = task['task_id']
            try:
                if not reached.wait(60):
                    raise TimeoutError('first candidate checkpoint not reached')
                queue.cancel(task_id)
            finally:
                release.set()
            cancelled = wait(queue, task_id)
            if cancelled['status'] != 'cancelled':
                raise RuntimeError(cancelled)
        finally:
            queue.close(60)
        queue = TaskQueue(output / 'tasks', handlers)
        try:
            queue.resume(task_id)
            final = wait(queue, task_id)
            saved_rows = queue.results(task_id)
        finally:
            queue.close(60)
    task_seconds = time.perf_counter() - started
    if final['status'] != 'completed':
        raise RuntimeError(final)
    # Run the pushed pre-upgrade implementation in a separate empty gallery.
    legacy_source = subprocess.check_output(['git', 'show', 'f7cb66a:core_modules/organize/face_db_builder.py'], text=True)
    namespace = {'__name__': 'face_database_recovery_legacy', '__file__': str(ROOT / 'core_modules/organize/face_db_builder.py')}
    exec(compile(legacy_source, namespace['__file__'], 'exec'), namespace)
    started = time.perf_counter()
    baseline = namespace['FaceDBBuilder'](system, str(output / 'baseline')).build_from_organized([str(source)])
    baseline_seconds = time.perf_counter() - started
    # Reuse candidate checkpoints for an independent virtual planning pass.
    class DryContext:
        def __init__(self):
            self.plans = {}
        def check(self):
            pass
        def results(self):
            return [row for row in saved_rows if row.get('stage') != 'fingerprint']
        def checkpoint(self, name):
            return self.plans.get(name)
        def advance(self, name, payload, items):
            self.plans[name] = payload
            return True
    dry = FaceDatabaseTask(dict(params, db_root=str(output / 'dry'), dry_run=True,
                                input_hashes=source_hashes, output_dir=None), system)
    dry.context = DryContext()
    dry.advance()
    prediction = dry.finish(dry.context.results())
    def hashes(root):
        return {str(path.relative_to(root)): fingerprint(path) for path in face_database_images(root)}
    actual, expected = hashes(output / 'resumed'), hashes(output / 'baseline')
    checks = {'sources_unchanged': all(fingerprint(path) == digest for path, digest in source_hashes.items()),
              'code_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in version.items()),
              'no_candidate_repeated': len(processed) == len(set(processed)) == len(manifest),
              'anchor_paths_and_bytes_equal': actual == expected,
              'added_count_equal': final['result']['added'] == baseline['added'] == len(actual),
              'dry_prediction_equal': prediction['added'] == len(actual) and prediction['added_persons'] == final['result']['added_persons'],
              'dry_gallery_absent': not (output / 'dry').exists(),
              'no_failures': final['failed'] == 0}
    report = {'checks': checks, 'n_images': len(manifest), 'n_anchors': len(actual),
              'n_persons': len(final['result']['added_persons']), 'cancelled_succeeded': cancelled['succeeded'],
              'task_seconds_including_recovery': task_seconds, 'baseline_seconds': baseline_seconds,
              'source_sha256': version, 'model_contract': final['execution_contract'],
              'device': args.device, 'physical_gpu': args.gpu, 'baseline_commit': 'f7cb66a',
              'limitations': ['One directory and one run; no identity GT or performance acceptance.',
                              'Executor restart after cancellation, not a process-kill experiment.',
                              'Sequential runs and shared GPU load affect timings.',
                              'Dry planning reuses verified candidate checkpoints; no extra model inference.']}
    atomic_json(output / 'summary.json', report)
    print(json.dumps({key: report[key] for key in ('checks', 'n_images', 'n_anchors', 'n_persons',
                     'task_seconds_including_recovery', 'baseline_seconds')}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
