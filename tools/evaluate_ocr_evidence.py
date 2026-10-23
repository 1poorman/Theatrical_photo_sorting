"""Evaluate explicitly selected text-bearing images using local PaddleOCR weights.

The manifest contains images with path and content_hash. Reports audit the actual
worker, source integrity and cache reuse; they do not infer OCR accuracy without
transcriptions supplied by reviewers.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_modules.organize.ocr_evidence import PaddleOCRProvider, validate_ocr
from core_modules.organize.task_artifacts import atomic_json, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--recognizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    rows = json.loads(args.manifest.read_text(encoding='utf-8'))['images']
    if not rows or len(rows) > 64 or len({row['content_hash'] for row in rows}) != len(rows):
        raise ValueError('expected_1_to_64_unique_selected_images')
    for row in rows:
        if fingerprint(row['path']) != row['content_hash']:
            raise ValueError('manifest_source_changed')
    root.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'command': [sys.executable, *sys.argv],
             'started_at': datetime.now(timezone.utc).isoformat(),
             'manifest_sha256': fingerprint(args.manifest), 'completed': 0}
    atomic_json(root / 'run_state.json', state)
    kwargs = dict(python=args.python, detector_dir=args.detector, recognizer_dir=args.recognizer,
                  cache_dir=root / 'cache')
    provider = PaddleOCRProvider(**kwargs)
    results = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        try:
            evidence, trace = provider.observe(row['path'], requested=True)
            observation = json.loads(evidence['quote']) if evidence else None
            result = {'image_id': row.get('image_id', row['content_hash']), 'source': row['path'],
                      'sha256': row['content_hash'], 'observation': observation, 'trace': trace,
                      'elapsed_seconds': time.perf_counter() - started,
                      'source_unchanged': fingerprint(row['path']) == row['content_hash']}
        except Exception as exc:
            result = {'source': row['path'], 'sha256': row['content_hash'],
                      'error': type(exc).__name__, 'elapsed_seconds': time.perf_counter() - started}
        results.append(result)
        atomic_json(root / f'image_{index:03d}.json', result)
        state['completed'] = index + 1
        atomic_json(root / 'run_state.json', state)
        print(json.dumps({'completed': index + 1, 'total': len(rows),
                          'lines': len((result.get('observation') or {}).get('lines', [])),
                          'error': result.get('error') or result.get('trace', {}).get('error')}), flush=True)
    # Reconstruct the provider: this exercises the JSON identity round-trip that
    # an in-memory second lookup would otherwise miss.
    cached_provider = PaddleOCRProvider(**kwargs)
    for row, result in zip(rows, results):
        if result.get('observation') is None:
            continue
        evidence, trace = cached_provider.observe(row['path'], requested=True)
        result['cache_reconstruction'] = {'cache_hit': trace.get('cache_hit', False),
                                          'called': trace.get('called'),
                                          'value_equal': bool(evidence) and json.loads(evidence['quote']) == result['observation']}
    checks = {'all_sources_unchanged': all(row.get('source_unchanged') for row in results),
              'all_schemas_valid': all(validate_ocr(row.get('observation')) for row in results),
              'actual_worker_for_each_input': all(row.get('trace', {}).get('called') for row in results),
              'reconstructed_cache_no_worker': all(row.get('cache_reconstruction', {}).get('cache_hit') and
                   row['cache_reconstruction']['called'] is False and row['cache_reconstruction']['value_equal'] for row in results),
              'manifest_unchanged': fingerprint(args.manifest) == state['manifest_sha256']}
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'input_count': len(rows), 'with_text': sum(bool((row.get('observation') or {}).get('lines')) for row in results),
               'identity': provider.identity, 'results': results,
               'quality_gate': 'insufficient_data', 'verified_transcriptions': 0,
               'limitations': ['Selected visible-text development samples; no held-out accuracy claim.',
                              'OCR strings are machine observations and cannot confirm actor identities.',
                              'CPU worker time includes model loading; no throughput comparison.']}
    atomic_json(root / 'summary.json', summary)
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(root / 'run_state.json', state)
    report = ['# Local OCR evidence check', '', f"Inputs: {len(rows)}; status: {summary['status']}.", '',
              'The second provider must reuse each valid observation without launching a worker.', '',
              '| Image | Recognized lines | Cache reused |', '| --- | ---: | --- |']
    for index, row in enumerate(results):
        report.append(f"| {index:03d} | {len((row.get('observation') or {}).get('lines', []))} | {row.get('cache_reconstruction', {}).get('cache_hit', False)} |")
    report += ['', 'OCR accuracy remains unverified: no independent reference transcriptions were supplied.', '']
    (root / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'checks': checks}), flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
