"""Isolated CPU scale checks for experimental sequence and face grouping.

Synthetic precomputed features measure algorithm time and process peak RSS only.
They cannot establish scene-boundary or identity quality on theatre photographs.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint


def worker(args):
    import numpy as np
    from importlib.metadata import version
    from core_modules.organize.face_graph_cluster import FaceGraphConfig, graph_labels
    from core_modules.organize.scene_change_points import ChangePointConfig, change_points
    if args.worker == 'legacy':
        import inspect
        from sklearn.cluster import AgglomerativeClustering
    elif args.worker == 'graph':
        import faiss  # Import costs are outside the measured label computation for both backends.

    rng = np.random.default_rng(args.seed)
    dimension = 768 if args.worker == 'window' else 512
    group_size = 40 if args.worker == 'window' else 10
    centers = rng.normal(size=((args.samples + group_size - 1) // group_size, dimension)).astype('float32')
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    values = np.repeat(centers, group_size, axis=0)[:args.samples]
    values += .006 * rng.normal(size=values.shape).astype('float32')
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    started = time.perf_counter()
    if args.worker == 'window':
        config = ChangePointConfig()
        points, scores = change_points(values, config)
        details = {'segment_count': len(points) + 1, 'config': asdict(config),
                   'finite_scores': bool(np.isfinite(scores).all()),
                   'minimum_segment_length': min(b - a for a, b in zip([0] + points, points + [len(values)])),
                   'complexity': 'scores O(n*d); worst-case peak insertion O(n^2); memory O(n*d)'}
    elif args.worker == 'graph':
        config = FaceGraphConfig()
        # Pair adjacent synthetic identities, emulating separate detections in a
        # photograph without assuming those identities are actual human labels.
        cannot = [(i, i + group_size) for i in range(0, len(values) - group_size, 2 * group_size)]
        labels, details = graph_labels(values, cannot, config, threads=4)
        details['config'] = asdict(config)
        details['output_count'] = len(labels)
    else:
        options = {'n_clusters': None, 'distance_threshold': .40, 'linkage': 'average'}
        options['metric' if 'metric' in inspect.signature(AgglomerativeClustering).parameters else 'affinity'] = 'cosine'
        labels = AgglomerativeClustering(**options).fit_predict(values)
        details = {'output_count': len(labels), 'n_clusters': len(set(labels)), 'config': options,
                   'sklearn_version': version('scikit-learn'), 'complexity': 'quadratic pairwise distances and hierarchy'}
    elapsed = time.perf_counter() - started
    result = {'method': args.worker, 'seconds': elapsed,
              'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'samples': len(values), 'dimension': dimension, 'feature_sha256': digest,
              'generator': {'seed': args.seed, 'group_size': group_size, 'noise_sigma': .006},
              'numpy_version': np.__version__, 'details': details}
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--worker', choices=['window', 'graph', 'legacy'])
    args = parser.parse_args()
    if not 2 <= args.samples <= 10000:
        raise ValueError('samples_must_be_between_2_and_10000')
    if args.worker:
        worker(args)
        return
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'core_modules/organize/face_graph_cluster.py',
               ROOT / 'core_modules/organize/scene_change_points.py']
    state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
             'command': [sys.executable, *sys.argv], 'runs': [],
             'source_sha256': {str(p.relative_to(ROOT)): fingerprint(p) for p in sources}}
    atomic_json(root / 'run_state.json', state)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    results = {}
    for method in ('window', 'graph', 'legacy'):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', method,
                   '--output', str(root / (method + '.json')), '--samples', str(args.samples), '--seed', str(args.seed)]
        with (root / (method + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            run = {'method': method, 'pid': process.pid, 'command': command, 'status': 'running'}
            state['runs'].append(run)
            atomic_json(root / 'run_state.json', state)
            try:
                exit_code = process.wait(timeout=180)
                run.update(status='completed' if exit_code == 0 else 'failed', exit_code=exit_code)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                run.update(status='timeout', exit_code=process.returncode)
        if run['status'] == 'completed':
            results[method] = json.loads((root / (method + '.json')).read_text())
        atomic_json(root / 'run_state.json', state)
        print(json.dumps({'method': method, 'status': run['status'],
                          'seconds': results.get(method, {}).get('seconds')}), flush=True)
    same_features = bool('graph' in results and 'legacy' in results and
                         results['graph']['feature_sha256'] == results['legacy']['feature_sha256'])
    speedup = results['legacy']['seconds'] / results['graph']['seconds'] if same_features else None
    limit = 2 * 1024 ** 3
    checks = {'all_workers_completed': len(results) == 3,
              'n_at_least_10000': args.samples >= 10000,
              'window_within_30_seconds': results.get('window', {}).get('seconds', float('inf')) <= 30,
              'window_rss_within_2_gib': results.get('window', {}).get('process_peak_rss_bytes', limit + 1) <= limit,
              'graph_within_60_seconds': results.get('graph', {}).get('seconds', float('inf')) <= 60,
              'graph_rss_within_2_gib': results.get('graph', {}).get('process_peak_rss_bytes', limit + 1) <= limit,
              'same_face_features': same_features,
              'graph_at_least_twice_as_fast': speedup is not None and speedup >= 2,
              'zero_cannot_link_violations': results.get('graph', {}).get('details', {}).get('constraint_violations') == 0,
              'source_unchanged': all(fingerprint(ROOT / p) == digest for p, digest in state['source_sha256'].items())}
    summary = {'status': 'passed' if all(checks.values()) else 'not_passed', 'target_checks': checks,
               'synthetic_precomputed_features': True, 'quality_gate': 'insufficient_data',
               'graph_vs_legacy_speedup': speedup, 'results': results,
               'hardware': {'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
                            'cpu_model': next((line.split(':', 1)[1].strip() for line in
                                Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), None),
                            'thread_limit': 4},
               'limitations': ['One run per method on a shared CPU; not an end-to-end throughput comparison.',
                              'Feature generation precedes timing. Peak RSS covers the entire isolated worker process.',
                              'Synthetic features and cannot-link pairs cannot establish identity or boundary quality.',
                              'Legacy hierarchy and guarded graph use different merging policies; quality requires real GT.',
                              'DINOv3, AdaFace, TOPIQ and MUSIQ are not measured by this tool.']}
    atomic_json(root / 'summary.json', summary)
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(root / 'run_state.json', state)
    report = ['# Experimental grouping CPU scale check', '',
              f"Samples: {args.samples}; scale targets: {summary['status']}; formal quality: insufficient_data.", '',
              '| Method | Dimension | Seconds | Peak process MiB |', '| --- | ---: | ---: | ---: |']
    for name, row in results.items():
        report.append(f"| {name} | {row['dimension']} | {row['seconds']:.3f} | {row['process_peak_rss_bytes'] / 1024 ** 2:.1f} |")
    report += ['', f'Graph/legacy speedup: {speedup:.3f}x.' if speedup else 'No comparable speedup available.', '',
               *summary['limitations'], '']
    (root / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'target_checks': checks}), flush=True)


if __name__ == '__main__':
    main()
