"""Paired persistent preview replay with real 200-image detection checkpoints.

No face detection/embedding inference: replay the same frames/landmarks from
an earlier run and compare 961ffb7 preview processing with current crop stages.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import psutil
    import sqlite3
    from core_modules.face_recognition.arcface.arcface_onnx import norm_crop
    from core_modules.organize.face_cluster import FaceClusterBuilder, CLUSTER_COLLAGE_MAX
    from core_modules.organize.face_cluster_task import FaceClusterTask
    from core_modules.organize.task_artifacts import atomic_json, fingerprint
    from core_modules.organize.task_queue import TaskQueue
    args.output = args.output.resolve()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty')
    with sqlite3.connect(args.task_root / 'tasks.sqlite3') as connection:
        row = connection.execute("SELECT id,request FROM tasks WHERE kind='face_cluster' ORDER BY created LIMIT 1").fetchone()
        original_params = json.loads(row[1])['params']
        contract = json.loads(connection.execute("SELECT payload FROM checkpoints WHERE task_id=? AND name='execution_contract'", (row[0],)).fetchone()[0])
        hashes = json.loads(connection.execute("SELECT payload FROM checkpoints WHERE task_id=? AND name='input_fingerprints'", (row[0],)).fetchone()[0])['hashes']
        clusters = json.loads(connection.execute("SELECT payload FROM checkpoints WHERE task_id=? AND name='clusters'", (row[0],)).fetchone()[0])
        faces = [json.loads(row[0]) for row in connection.execute("SELECT result FROM items WHERE task_id=? AND json_extract(input,'$.stage')='faces' AND status='succeeded' ORDER BY position", (row[0],))]
    if len(hashes) < 200 or len(faces) != len(hashes):
        raise ValueError('requires complete 200-image real detection checkpoints')
    if any(Path(path).parent == args.output or Path(path).parent in args.output.parents or args.output in Path(path).parents for path in hashes):
        raise ValueError('output must be outside input photos')
    if any(fingerprint(path) != digest for path, digest in hashes.items()):
        raise ValueError('source content changed')
    args.output.mkdir(parents=True, exist_ok=True)
    legacy = subprocess.check_output(['git', 'show', '961ffb7:core_modules/organize/face_cluster_task.py'], text=True)
    namespace = {'__name__': 'cluster_preview_baseline', '__file__': str(ROOT / 'core_modules/organize/face_cluster_task.py')}
    exec(compile(legacy, namespace['__file__'], 'exec'), namespace)
    classes = {'baseline': namespace['FaceClusterTask'], 'optimized': FaceClusterTask}
    system = SimpleNamespace(detector=SimpleNamespace(model_signature='not-used'),
        extractor=SimpleNamespace(model_signature='alignment-only', align_face=lambda img, landmark: norm_crop(img, np.asarray(landmark, dtype=np.float32), image_size=112)))
    # These are exactly the loaded ArcFace alignment rules; model inference is absent.
    measured = ['tools/benchmark_cluster_previews.py', 'core_modules/organize/face_cluster_task.py',
                'core_modules/organize/face_cluster.py', 'core_modules/organize/task_queue.py',
                'core_modules/organize/task_artifacts.py', 'core_modules/organize/task_versions.py',
                'core_modules/face_recognition/arcface/arcface_onnx.py']
    versions = {name: fingerprint(ROOT / name) for name in measured}
    samples = []
    expected = None
    baseline_contract = contract
    for repeat in range(args.repeats):
        order = ('baseline', 'optimized') if repeat % 2 == 0 else ('optimized', 'baseline')
        for variant in order:
            output = args.output / f'{repeat:02d}_{variant}'
            params = dict(original_params, input_hashes=hashes, output_dir=str(output / 'previews'))
            phases, decoded = {}, []
            original_read = FaceClusterBuilder.read_image
            def read(path):
                decoded.append(path)
                return original_read(path)
            FaceClusterBuilder.read_image = staticmethod(read)
            class Replay:
                def __init__(self, params):
                    self.delegate = classes[variant](params, system)
                    self.context = None
                def model_contract(self):
                    return self.delegate.model_contract()
                def process(self, item):
                    self.delegate.context = self.context
                    start = time.perf_counter()
                    value = faces[item['index']] if item['stage'] == 'seed' else self.delegate.process(item)
                    phases[item['stage']] = phases.get(item['stage'], 0) + time.perf_counter() - start
                    return value
                def advance(self):
                    self.delegate.context = self.context
                    if self.context.checkpoint('seed') is None:
                        return self.context.advance('seed', True, [{'image': row['image'], 'stage': 'seed', 'index': index} for index, row in enumerate(faces)])
                    if self.context.checkpoint('clusters') is None:
                        plan = json.loads(json.dumps(clusters))
                        if variant == 'baseline':
                            items = [{'image': c['members'][0]['img_path'], 'stage': 'preview', 'cluster_id': c['id']} for c in plan]
                        else:
                            by_image = {}
                            for cluster in plan:
                                for member in cluster['members'][:CLUSTER_COLLAGE_MAX]:
                                    by_image.setdefault(member['img_path'], {})[member['face_index']] = member
                            items = [{'image': path, 'stage': 'crops', 'members': list(members.values())} for path, members in by_image.items()]
                        return self.context.advance('clusters', plan, items)
                    return self.delegate.advance()
                def finish(self, rows):
                    self.delegate.context = self.context
                    start = time.perf_counter()
                    result = self.delegate.finish(rows)
                    phases['finish'] = time.perf_counter() - start
                    return result
            rss, stop = [], threading.Event()
            def memory():
                process = psutil.Process()
                while not stop.wait(.02):
                    rss.append(process.memory_info().rss)
            monitor = threading.Thread(target=memory, daemon=True)
            monitor.start()
            started = time.perf_counter()
            queue = TaskQueue(output / 'tasks', {'face_cluster': Replay})
            try:
                task, _ = queue.submit('face_cluster', params, [])
                deadline = time.perf_counter() + 1200
                while True:
                    final = queue.get(task['task_id'])
                    if final['status'] in ('completed', 'completed_with_errors', 'failed', 'cancelled'):
                        break
                    if time.perf_counter() > deadline:
                        raise TimeoutError('preview replay timed out')
                    time.sleep(.05)
                elapsed = time.perf_counter() - started
                if final['status'] != 'completed':
                    raise RuntimeError(final)
            finally:
                queue.close(10)
                stop.set()
                monitor.join(1)
                FaceClusterBuilder.read_image = staticmethod(original_read)
            state = json.loads(Path(final['result']['state_path']).read_text())
            artifacts = {cluster['id']: fingerprint(cluster['preview']) for cluster in state['clusters']}
            if expected is None:
                expected = artifacts
            checks = {'preview_paths_and_bytes_equal': artifacts == expected,
                      'source_content_unchanged': all(fingerprint(path) == digest for path, digest in hashes.items()),
                      'optimized_decodes_each_source_once': variant == 'baseline' or max(Counter(decoded).values(), default=0) == 1}
            sample = {'repeat': repeat, 'variant': variant, 'elapsed_seconds': elapsed, 'phase_seconds': phases,
                      'decodes': len(decoded), 'rss_peak_bytes': max(rss, default=0), 'checks': checks,
                      'task_database_bytes': (output / 'tasks/tasks.sqlite3').stat().st_size}
            samples.append(sample)
            atomic_json(args.output / 'samples.json', samples)
            print(json.dumps(sample), flush=True)
            if not all(checks.values()):
                raise RuntimeError('preview replay check failed')
    from statistics import median
    medians = {variant: median(row['elapsed_seconds'] for row in samples if row['variant'] == variant) for variant in classes}
    stages = {variant: median(sum(row['phase_seconds'].get(stage, 0) for stage in ('crops', 'preview')) for row in samples if row['variant'] == variant) for variant in classes}
    report = {'n_images': len(hashes), 'n_faces': sum(len(row['members']) for row in faces), 'n_clusters': len(clusters),
              'preview_members': sum(min(len(c['members']), CLUSTER_COLLAGE_MAX) for c in clusters),
              'repeats': args.repeats, 'samples': samples, 'median_elapsed_seconds': medians,
              'median_preview_stage_seconds': stages, 'elapsed_speedup': medians['baseline'] / medians['optimized'],
              'stage_speedup': stages['baseline'] / stages['optimized'], 'source_sha256': versions,
              'original_detection_contract': baseline_contract, 'baseline_commit': '961ffb7',
              'checks': {'all_sample_checks': all(all(row['checks'].values()) for row in samples),
                         'measured_code_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in versions.items())},
              'limitations': ['Real saved detection results; exact alignment and JPEG bytes; no detection or embedding timings.',
                              'Alternating order in one process; filesystem cache/allocator state can affect results.',
                              'Includes durable checkpoint and publication costs; this is preview-stage performance only.',
                              'Peak RSS includes imported model/runtime libraries, sampled every 20 ms; no identity GT or U2 acceptance.']}
    atomic_json(args.output / 'summary.json', report)
    print(json.dumps({key: report[key] for key in ('checks', 'median_elapsed_seconds', 'median_preview_stage_seconds', 'elapsed_speedup', 'stage_speedup')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
