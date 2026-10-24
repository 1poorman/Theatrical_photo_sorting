"""Collect direct observations for every neutral image in a frozen manifest.

This explicit on-demand development run makes one observation request per image
and no scene-choice request. It prepares fact review without changing the
default automatic budget, production labels, or final-test status.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.llm_client import vision_client_from_env
from core_modules.organize.scene_reasoner import SceneReasoner
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from core_modules.organize.visual_evidence import VisualEvidenceController, VisualPolicy, validate_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Neutral U5 input manifest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    source_sha = fingerprint(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    rows = manifest.get('rows')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256 or not 1 <= args.timeout <= 120:
        raise ValueError('expected_1_to_256_neutral_images_and_valid_timeout')
    columns = ('image_id', 'play_id', 'neutral_path', 'neutral_sha256')
    if any(set(row) != set(columns) for row in rows):
        raise ValueError('neutral_manifest_must_not_contain_query_labels')
    if len({row['image_id'] for row in rows}) != len(rows) or len({row['neutral_sha256'] for row in rows}) != len(rows):
        raise ValueError('duplicate_image_or_content')
    for row in rows:
        if fingerprint(row['neutral_path']) != row['neutral_sha256']:
            raise ValueError('neutral_source_changed')
    policy = VisualPolicy(mode='on_demand', max_images=len(rows),
                          max_output_tokens_per_request=args.max_output_tokens,
                          max_batch_output_tokens=2 * len(rows) * args.max_output_tokens)
    client = vision_client_from_env()
    if client is None:
        raise ValueError('vision_client_unavailable')
    client.timeout = args.timeout
    root.mkdir(parents=True, exist_ok=True)
    (root / 'observations').mkdir()
    code = {name: fingerprint(ROOT / name) for name in ('tools/collect_visual_observations.py',
            'core_modules/organize/scene_reasoner.py', 'core_modules/organize/llm_client.py',
            'core_modules/organize/visual_evidence.py', 'config/base.py')}
    input_data = {'rows': rows, 'selection': manifest.get('selection'),
                  'source_manifest': str(args.manifest.resolve()), 'source_manifest_sha256': source_sha}
    atomic_json(root / 'input_manifest.json', input_data)
    state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
             'command': [sys.executable, *sys.argv], 'policy': asdict(policy),
             'source_sha256': code, 'completed': 0, 'input_count': len(rows)}
    atomic_json(root / 'run_state.json', state)
    attempts = []
    opener = client._opener
    def audited_open(request, timeout):
        payload = json.loads(request.data)
        messages = payload['messages']
        image_parts = [part for message in messages if isinstance(message.get('content'), list)
                       for part in message['content'] if part.get('type') == 'image_url']
        if len(image_parts) != 1:
            raise ValueError('one_actual_image_required')
        entry = {'request_body_sha256': hashlib.sha256(request.data).hexdigest(),
                 'max_tokens': payload['max_tokens'], 'image_inputs': 1}
        attempts.append(entry)
        atomic_json(root / 'http_attempts.json', attempts)
        try:
            return opener(request, timeout=timeout)
        except Exception as exc:
            entry.update(error_type=type(exc).__name__,
                         reason_type=type(getattr(exc, 'reason', None)).__name__,
                         http_status=getattr(exc, 'code', None))
            atomic_json(root / 'http_attempts.json', attempts)
            raise
    client._opener = audited_open
    kwargs = dict(client=client, cache_dir=root / 'cache', budget_path=root / 'budget.sqlite3',
                  scope=fingerprint(root / 'input_manifest.json'), count=len(rows), policy=policy)
    controller = VisualEvidenceController(**kwargs)
    reasoner = SceneReasoner({'scenes': []}, vision_client=client)
    results = []
    interrupted = False
    started = time.perf_counter()
    for index, row in enumerate(rows):
        image_started = time.perf_counter()
        evidence, trace = controller.observe(row['neutral_path'], ['explicit_fact_review_collection'], reasoner._observe)
        result = {**row, 'observation': json.loads(evidence['quote']) if evidence else None,
                  'trace': trace, 'elapsed_seconds': time.perf_counter() - image_started}
        results.append(result)
        atomic_json(root / 'observations' / f'{index:03d}.json', result)
        state.update(completed=len(results), http_requests=len(attempts))
        atomic_json(root / 'run_state.json', state)
        print(json.dumps({'completed': len(results), 'total': len(rows), 'http_requests': len(attempts),
                          'valid_observation': validate_observation(result['observation'])}), flush=True)
        if attempts and attempts[-1].get('error_type'):
            interrupted = True
            break
    elapsed = time.perf_counter() - started
    first_requests = len(attempts)
    replay_equal = []
    if not interrupted:
        replay = VisualEvidenceController(**kwargs)
        for row, result in zip(rows, results):
            evidence, _ = replay.observe(row['neutral_path'], ['explicit_fact_review_collection'], reasoner._observe)
            replay_equal.append(bool(evidence) and json.loads(evidence['quote']) == result['observation'])
    budget = controller.budget.summary()
    checks = {'all_images_processed': len(results) == len(rows),
              'all_images_have_valid_observations': len(results) == len(rows) and all(
                  validate_observation(row['observation']) for row in results),
              'one_actual_observation_request_per_image': first_requests == len(rows) and all(
                  row['image_inputs'] == 1 for row in attempts[:first_requests]),
              'actual_calls_equal_reservations': first_requests == budget['reserved_requests'],
              'output_reservations_within_budget': budget['reserved_max_output_tokens'] <= policy.max_batch_output_tokens,
              'replay_has_no_requests': not interrupted and len(attempts) == first_requests,
              'replay_equal': len(replay_equal) == len(rows) and all(replay_equal),
              'neutral_sources_unchanged': all(fingerprint(row['neutral_path']) == row['neutral_sha256'] for row in rows),
              'input_manifest_unchanged': fingerprint(args.manifest) == source_sha,
              'source_unchanged': all(fingerprint(ROOT / name) == sha for name, sha in code.items())}
    for row in rows[len(results):]:
        results.append({**row, 'observation': None, 'trace': {'skipped': 'transport_interruption'},
                        'elapsed_seconds': None})
    atomic_json(root / 'results.json', results)
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'input_count': len(rows), 'observed_images': sum(validate_observation(row['observation']) for row in results),
               'http_requests': first_requests, 'replay_extra_requests': len(attempts) - first_requests,
               'elapsed_seconds': elapsed, 'policy': asdict(policy), 'budget': budget,
               'model_identity': controller.identity, 'source_sha256': code,
               'artifact_sha256': {name: fingerprint(root / name) for name in
                   ('input_manifest.json', 'results.json', 'http_attempts.json')},
               'quality_gate': 'insufficient_data', 'fact_reviewers': 0, 'human_verified_images': 0,
               'final_test_locked': False, 'default_adoption': False,
               'limitations': ['Explicit on-demand observation collection; default automatic policy is unchanged.',
                              'Previously used neutral development images; no scene-choice calls or retrieval encoding.',
                              'Remote weights remain unverified; replay proves current-process reuse only.',
                              'No two-reviewer fact ground truth, final test, Macro-F1 or end-to-end performance claim.']}
    atomic_json(root / 'summary.json', summary)
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(root / 'run_state.json', state)
    (root / 'report.md').write_text('\n'.join(['# Direct visual observations for fact review', '',
        f"Valid observations: {summary['observed_images']}/{len(rows)}; actual requests: {first_requests}.",
        f"Collection wall time: {elapsed:.3f}s; replay extra requests: {summary['replay_extra_requests']}.",
        'Quality remains insufficient_data; no human labels are created.', '', *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'checks': checks, 'observed_images': summary['observed_images']}), flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
