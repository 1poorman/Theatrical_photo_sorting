"""Measure same-version single-record writes in a synthetic large review queue.

This CPU/storage experiment is independent of the paused U2 model pipeline.
It does not measure GPU, HTTP, end-to-end speed or any model quality.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.records < 2 or not 1 <= args.repeats <= min(args.records, 100):
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
    checks = {'both_modes_preserve_records': all(row['latest_records_correct'] for row in results.values()),
              'no_deferred_snapshot_replacement': results['deferred']['snapshot_file_replacements'] == 0,
              'deferred_restarted_reads_no_snapshot': results['deferred']['restarted_read_snapshot_replacements'] == 0,
              'eager_compatibility_snapshots_written': results['eager']['snapshot_file_replacements'] >= 2 * args.repeats,
              'deferred_snapshot_unchanged_until_export': results['deferred']['snapshot_unchanged_until_export'],
              'exports_equal': results['eager']['final_labels'] == results['deferred']['final_labels'],
              'snapshots_match_SQLite': all(row['snapshots_match_authority'] for row in results.values()),
              'counts_and_history_match': all(row['export_count'] == row['revisions'] == args.repeats for row in results.values()),
              'source_unchanged': all(fingerprint(ROOT / name) == sha for name, sha in code.items())}
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'records': args.records, 'repeats': args.repeats, 'results': results, 'source_sha256': code,
               'U2': 'paused; no feature/decode/DALI work',
               'limitations': ['Synthetic review payloads; same-version local function calls including summary.',
                              'Sequential modes on shared CPU/storage; timings are observations, not a controlled end-to-end gain.',
                              'No GPU inference, HTTP concurrency or quality ground truth.']}
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'run_state.json', {'status': summary['status'], 'completed_at': datetime.now(timezone.utc).isoformat()})
    (root / 'report.md').write_text('\n'.join(['# Review snapshot scale check', '',
        f"Records: {args.records}; decisions per mode: {args.repeats}.",
        f"Eager median: {results['eager']['median_seconds']:.6f}s; deferred median: {results['deferred']['median_seconds']:.6f}s.",
        f"Deferred snapshot replacements per decision: 0; exported state matches SQLite: {checks['snapshots_match_SQLite']}.",
        '', *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'checks': checks, 'medians_seconds': {
        mode: row['median_seconds'] for mode, row in results.items()}}))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
