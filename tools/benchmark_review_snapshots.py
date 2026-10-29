"""Measure same-version single-record writes in a synthetic large review queue.

Includes the full explicit export, so deferred snapshot costs remain visible.
It does not measure GPU, HTTP, end-to-end speed or any model quality.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.review_queue import ReviewQueue
from core_modules.organize.review_store import atomic_rows
from core_modules.organize.task_artifacts import atomic_json, fingerprint


@contextmanager
def count_snapshots():
    calls = []
    def write(path, rows):
        calls.append(path.name)
        return atomic_rows(path, rows)
    with patch('core_modules.organize.review_store.atomic_rows', side_effect=write):
        yield calls


def measure_exports(root, rows, repeats):
    """Alternate eager/deferred exports after real writes to isolated queues."""
    queues = {}
    results = {'eager': [], 'deferred': []}
    for mode in results:
        queue = ReviewQueue(str(root / ('export_' + mode)), snapshot_on_write=mode == 'eager')
        queue.seed(rows)
        queues[mode] = queue
    for repeat in range(repeats):
        for mode in (('eager', 'deferred') if repeat % 2 == 0 else ('deferred', 'eager')):
            queue = queues[mode]
            start = time.perf_counter()
            queue.decide(rows[0]['image'], 'modify', label='export-probe',
                         expected_revision=repeat, reviewer='synthetic-benchmark')
            decide_seconds = time.perf_counter() - start
            cpu_start = time.process_time()
            with count_snapshots() as calls:
                start = time.perf_counter()
                path, count = queue.export_final()
                seconds = time.perf_counter() - start
            cpu_seconds = time.process_time() - cpu_start
            directory = Path(queue.root)
            snapshot = {row['image']: row for row in map(json.loads,
                (directory / 'review_queue.jsonl').read_text().splitlines())}
            history = list(map(json.loads, (directory / 'revisions.jsonl').read_text().splitlines()))
            labels = json.loads(Path(path).read_text())
            results[mode].append({'repeat': repeat + 1, 'seconds': seconds,
                'cpu_seconds': cpu_seconds, 'decide_seconds': decide_seconds,
                'decide_plus_export_seconds': decide_seconds + seconds,
                'snapshot_replacements': calls, 'export_count': count,
                'file_bytes': {name: (directory / name).stat().st_size
                               for name in ('review_queue.jsonl', 'revisions.jsonl', 'scene_labels.json')},
                'labels_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'checks': {'snapshot_matches_SQLite': snapshot == queue.records(),
                           'history_matches_SQLite': history == queue.revisions(),
                           'all_records_present': len(snapshot) == len(rows),
                           'only_approved_exported': count == 1 and labels == {rows[0]['image']: 'export-probe'},
                           'history_count': len(history) == repeat + 1,
                           'expected_snapshot_writes': sorted(calls) ==
                               (['review_queue.jsonl', 'revisions.jsonl'] if mode == 'deferred' else [])}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--export-repeats', type=int, default=5)
    args = parser.parse_args()
    if args.records < 2 or not 1 <= args.repeats <= min(args.records, 100) or args.export_repeats < 1:
        raise ValueError('invalid_scale_or_repeats')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    root.mkdir(parents=True, exist_ok=True)
    code = {name: fingerprint(ROOT / name) for name in ('tools/benchmark_review_snapshots.py',
            'core_modules/organize/review_store.py', 'core_modules/organize/review_queue.py')}
    atomic_json(root / 'run_state.json', {'status': 'running', 'command': [sys.executable, *sys.argv],
                                        'started_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': code})
    rows = [{'image': f'photo-{i:05d}.jpg', 'label': 'candidate',
             'evidence': [{'quote': 'synthetic evidence ' * 12}]} for i in range(args.records)]
    results = {}
    for mode in ('eager', 'deferred'):
        directory = root / mode
        ReviewQueue(str(directory)).seed(rows)
        queue = ReviewQueue(str(directory), snapshot_on_write=mode == 'eager')
        original = (directory / 'review_queue.jsonl').read_bytes()
        timings = []
        with count_snapshots() as calls:
            for i in range(args.repeats):
                started = time.perf_counter()
                queue.decide(rows[i]['image'], 'modify', label=f'human-{i}', expected_revision=0)
                queue.summary()
                timings.append(time.perf_counter() - started)
            snapshot_count = len(calls)
            # Fresh queue objects simulate reads after service recreation.
            restarted = ReviewQueue(str(directory), snapshot_on_write=mode == 'eager')
            latest = [restarted.record(rows[i]['image']) for i in range(args.repeats)]
            reads_wrote_snapshots = len(calls) - snapshot_count
        before_export_unchanged = (directory / 'review_queue.jsonl').read_bytes() == original
        final, count = queue.export_final()
        snapshot = {row['image']: row for row in
                    (json.loads(line) for line in (directory / 'review_queue.jsonl').read_text().splitlines())}
        results[mode] = {'write_and_summary_seconds': timings, 'median_seconds': statistics.median(timings),
            'snapshot_file_replacements': snapshot_count, 'restarted_read_snapshot_replacements': reads_wrote_snapshots,
            'snapshot_unchanged_until_export': before_export_unchanged,
            'latest_records_correct': all(row['label'] == f'human-{i}' and row['review_revision'] == 1
                                          for i, row in enumerate(latest)),
            'export_count': count, 'final_labels': json.loads(Path(final).read_text()),
            'snapshots_match_authority': snapshot == queue.records(), 'revisions': len(queue.revisions())}
    exports = measure_exports(root, rows, args.export_repeats)
    checks = {'both_modes_preserve_records': all(row['latest_records_correct'] for row in results.values()),
              'no_deferred_snapshot_replacement': results['deferred']['snapshot_file_replacements'] == 0,
              'deferred_restarted_reads_no_snapshot': results['deferred']['restarted_read_snapshot_replacements'] == 0,
              'eager_compatibility_snapshots_written': results['eager']['snapshot_file_replacements'] >= 2 * args.repeats,
              'deferred_snapshot_unchanged_until_export': results['deferred']['snapshot_unchanged_until_export'],
              'exports_equal': results['eager']['final_labels'] == results['deferred']['final_labels'],
              'snapshots_match_SQLite': all(row['snapshots_match_authority'] for row in results.values()),
              'counts_and_history_match': all(row['export_count'] == row['revisions'] == args.repeats for row in results.values()),
              'source_unchanged': all(fingerprint(ROOT / name) == sha for name, sha in code.items()),
              'full_export_checks': all(all(row['checks'].values()) for runs in exports.values() for row in runs),
              'full_export_labels_equal': len({row['labels_sha256'] for runs in exports.values() for row in runs}) == 1}
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'records': args.records, 'repeats': args.repeats, 'results': results, 'source_sha256': code,
               'explicit_export': {'repeats': args.export_repeats, 'runs': exports,
                   'median_seconds': {mode: statistics.median(row['seconds'] for row in runs)
                                      for mode, runs in exports.items()},
                   'median_decide_plus_export_seconds': {
                       mode: statistics.median(row['decide_plus_export_seconds'] for row in runs)
                       for mode, runs in exports.items()}},
               'environment': {'cpu_count': os.cpu_count(), 'cpu_load_average': os.getloadavg()},
               'limitations': ['Synthetic review payloads; same-version local function calls including summary.',
                              'Sequential modes on shared CPU/storage; timings are observations, not a controlled end-to-end gain.',
                              'No GPU inference, HTTP concurrency or quality ground truth.']}
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'run_state.json', {'status': summary['status'], 'completed_at': datetime.now(timezone.utc).isoformat()})
    (root / 'report.md').write_text('\n'.join(['# Review snapshot scale check', '',
        f"Records: {args.records}; decisions per mode: {args.repeats}.",
        f"Eager median: {results['eager']['median_seconds']:.6f}s; deferred median: {results['deferred']['median_seconds']:.6f}s.",
        f"Deferred snapshot replacements per decision: 0; exported state matches SQLite: {checks['snapshots_match_SQLite']}.",
        f"Full export medians: {summary['explicit_export']['median_seconds']}.",
        f"Decision + export medians: {summary['explicit_export']['median_decide_plus_export_seconds']}.",
        '', *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'checks': checks, 'medians_seconds': {
        mode: row['median_seconds'] for mode, row in results.items()}}))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
