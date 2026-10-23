"""Audit budgeted real vision requests with frozen U4 scores and neutral images.

This is a development integration experiment. It reuses verified precomputed
retrieval scores, never sends query labels to the model, and reports cache/budget
behavior separately from unmeasured fact accuracy and final-test quality.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint


class PreparedScoreRetriever:
    """Replay recorded model scores; this adapter performs no inference."""
    def __init__(self, manifest, scores, model_signature):
        self.indices = {row['neutral_path']: i for i, row in enumerate(manifest['rows'])}
        self.columns = {scene['scene_id']: i for i, scene in enumerate(manifest['scenes'])}
        self.scores = scores
        self.model_signature = {'origin': 'frozen_precomputed_u4_mean_scores', 'model': model_signature}
        self.diagnostics = {}

    def rank(self, image_path, knowledge, top_k=5):
        row = self.scores[self.indices[image_path]]
        ranked = []
        for scene in knowledge['scenes']:
            score = float(row[self.columns[scene['scene_id']]])
            score = min(1., max(0., score))
            ranked.append((.8 * score, dict(scene, _retrieval_scores={'visual_score': score, 'lexical_score': 0.})))
        ranked.sort(key=lambda item: -item[0])
        return '', ranked[:top_k], 'visual'


def main():
    import numpy as np
    from core_modules.organize.llm_client import vision_client_from_env
    from core_modules.organize.scene_reasoner import SceneReasoner, validate_choice
    from core_modules.organize.visual_evidence import VisualEvidenceController, VisualPolicy, validate_observation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True, help='Completed U4 backend directory')
    parser.add_argument('--samples-per-play', type=int, default=25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples_per_play <= 100:
        raise ValueError('invalid_sample_count')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    study = args.study.resolve()
    study_summary = json.loads((study / 'summary.json').read_text())
    frozen = study_summary['artifact_sha256']
    for name in ('input_manifest.json', 'candidate_scores.npz'):
        if fingerprint(study / name) != frozen[name]:
            raise ValueError('frozen_u4_artifact_changed:' + name)
    manifest = json.loads((study / 'input_manifest.json').read_text())
    with np.load(study / 'candidate_scores.npz', allow_pickle=False) as arrays:
        scores = arrays['mean']
    if scores.shape != (len(manifest['rows']), len(manifest['scenes'])) or not np.isfinite(scores).all():
        raise ValueError('precomputed_score_alignment')
    plays = sorted({row['play_id'] for row in manifest['rows']})
    groups = {play: sorted((row for row in manifest['rows'] if row['play_id'] == play),
                           key=lambda row: row['image_id'])[:args.samples_per_play] for play in plays}
    if any(len(rows) != args.samples_per_play for rows in groups.values()):
        raise ValueError('not_enough_distinct_queries_per_play')
    selected = [groups[play][i] for i in range(args.samples_per_play) for play in plays]
    for row in selected:
        if fingerprint(row['neutral_path']) != row['neutral_sha256']:
            raise ValueError('neutral_image_changed')
    root.mkdir(parents=True, exist_ok=True)
    policy = VisualPolicy()
    sources = ['tools/evaluate_visual_budget.py', 'core_modules/organize/scene_reasoner.py',
               'core_modules/organize/visual_evidence.py', 'core_modules/organize/llm_client.py',
               'core_modules/organize/multimodal_evidence.py']
    state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
             'command': [sys.executable, *sys.argv], 'policy': asdict(policy),
             'source_sha256': {name: fingerprint(ROOT / name) for name in sources},
             'study_summary_sha256': fingerprint(study / 'summary.json'), 'completed': 0}
    atomic_json(root / 'run_state.json', state)
    # No query scene/actor labels enter the prepared inference input.
    inputs = [{key: row[key] for key in ('image_id', 'play_id', 'neutral_path', 'neutral_sha256')} for row in selected]
    atomic_json(root / 'input_manifest.json', {'selection': 'round_robin_plays_then_image_id; no_label_selection',
                                              'rows': inputs, 'precomputed_source': str(study),
                                              'precomputed_hashes': {name: frozen[name] for name in ('input_manifest.json', 'candidate_scores.npz')}})
    client = vision_client_from_env()
    if client is None:
        atomic_json(root / 'run_state.json', dict(state, status='failed', error='vision_client_unavailable'))
        raise RuntimeError('vision_client_unavailable')
    client.timeout = 45
    attempts = []
    opener = client._opener
    def audited_open(request, timeout):
        payload = json.loads(request.data)
        entry = {'max_tokens': payload['max_tokens'],
                 'request_body_sha256': hashlib.sha256(request.data).hexdigest(),
                 'image_inputs': sum(part.get('type') == 'image_url' for msg in payload['messages']
                       if isinstance(msg.get('content'), list) for part in msg['content'])}
        attempts.append(entry)
        atomic_json(root / 'http_attempts.json', attempts)
        try:
            return opener(request, timeout=timeout)
        except Exception as exc:
            entry.update(error_type=type(exc).__name__, reason_type=type(getattr(exc, 'reason', None)).__name__,
                         http_status=getattr(exc, 'code', None))
            atomic_json(root / 'http_attempts.json', attempts)
            raise
    client._opener = audited_open
    retriever = PreparedScoreRetriever(manifest, scores, study_summary['model_signature'])
    knowledge = {play: {'scenes': [{key: scene[key] for key in ('scene_id', 'label', 'text', 'source_ref') if key in scene}
                                 for scene in manifest['scenes'] if scene['play_id'] == play]} for play in plays}
    kwargs = dict(client=client, cache_dir=root / 'cache', budget_path=root / 'budget.sqlite3',
                  scope=fingerprint(root / 'input_manifest.json'), count=len(selected), policy=policy)
    controller = VisualEvidenceController(**kwargs)
    results, interrupted = [], False
    started = time.perf_counter()
    for row in inputs:
        reasoner = SceneReasoner(knowledge[row['play_id']], client=client, vision_client=client,
                                 visual_controller=controller, embedder=retriever, use_rule_shortcut=False)
        result = reasoner.reason(row['neutral_path'])
        results.append(result)
        atomic_json(root / f"result_{len(results) - 1:03d}.json", result)
        state['completed'] = len(results)
        atomic_json(root / 'run_state.json', state)
        print(json.dumps({'completed': len(results), 'total': len(inputs), 'http_requests': len(attempts)}), flush=True)
        # Stop on a transport failure so a blocked endpoint cannot consume the
        # entire experiment's budget. Unprocessed queries remain in the report.
        if attempts and attempts[-1].get('error_type'):
            interrupted = True
            break
    elapsed = time.perf_counter() - started
    initial_attempts = len(attempts)
    replay_equal = []
    if not interrupted:
        replay = VisualEvidenceController(**kwargs)
        for row, first in zip(inputs, results):
            result = SceneReasoner(knowledge[row['play_id']], client=client, vision_client=client,
                        visual_controller=replay, embedder=retriever, use_rule_shortcut=False).reason(row['neutral_path'])
            replay_equal.append(all(result.get(key) == first.get(key) for key in ('scene_id', 'label', 'decision', 'evidence')))
    observations, choices, unsafe_choices = [], [], 0
    for row, result in zip(inputs, results):
        observation = next((json.loads(e['quote']) for e in result['evidence'] if e['type'] == 'observation'), None)
        if observation is not None:
            observations.append(observation)
        choice = result['model_trace'].get('choice')
        if choice:
            valid, _ = validate_choice(choice, [s['scene_id'] for s in knowledge[row['play_id']]['scenes']] + ['unknown'],
                                        [e['id'] for e in result['evidence']])
            choices.append(valid)
            unsafe_choices += int(not valid)
    observation_calls = sum(entry['image_inputs'] > 0 for entry in attempts[:initial_attempts])
    budget = controller.budget.summary()
    schema_ratio = len(observations) / observation_calls if observation_calls else None
    checks = {'all_inputs_processed': len(results) == len(inputs), 'actual_image_requests': observation_calls > 0,
              'observation_schema_at_least_99_percent': schema_ratio is not None and schema_ratio >= .99,
              'all_observations_valid': all(validate_observation(value) for value in observations),
              'no_illegal_choices': unsafe_choices == 0, 'at_least_one_valid_choice': any(choices),
              'automatic_http_ratio_within_30_percent': initial_attempts / len(inputs) <= .30,
              'actual_calls_equal_reserved': initial_attempts == budget['reserved_requests'],
              'output_reservations_within_budget': budget['reserved_max_output_tokens'] <= policy.max_batch_output_tokens,
              'replay_no_http': not interrupted and len(attempts) == initial_attempts,
              'replay_results_equal': len(replay_equal) == len(inputs) and all(replay_equal),
              'all_results_pending': all(row['review']['status'] == 'pending' for row in results),
              'neutral_images_unchanged': all(fingerprint(row['neutral_path']) == row['neutral_sha256'] for row in inputs),
              'source_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in state['source_sha256'].items())}
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'input_count': len(inputs), 'processed': len(results), 'unprocessed': len(inputs) - len(results),
               'plays': dict(Counter(row['play_id'] for row in inputs)), 'observed_images': len(observations),
               'observation_calls': observation_calls, 'valid_schema_ratio': schema_ratio,
               'accepted_model_choices': len(choices), 'unsafe_choices': unsafe_choices,
               'http_requests': initial_attempts, 'replay_extra_requests': len(attempts) - initial_attempts,
               'budget': budget, 'elapsed_seconds': elapsed, 'model_identity': controller.identity,
               'quality_gate': 'insufficient_data', 'fact_reviewers': 0, 'verified_fact_images': 0,
               'final_test_locked': False,
               'limitations': ['Development images and previously frozen expanded-pool mean SigLIP2 scores.',
                              'Image bytes sent to the configured real endpoint; no query labels or actor lists sent.',
                              'Timing excludes retrieval encoding and remote model loading; not a U4/U5 speed comparison.',
                              'Unverified remote weights: cache replay is restricted to the current client process.',
                              'No dual-reviewer visual fact labels or independent final test; no fact-accuracy/Macro-F1 claim.']}
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'fact_review_template.json', {'human_decisions_applied': 0, 'rows': [
        {'image_id': row['image_id'], 'image': row['neutral_path'],
         'observation': next((json.loads(e['quote']) for e in result['evidence'] if e['type'] == 'observation'), None),
         'reviewer_a': None, 'reviewer_b': None, 'field_decisions': None} for row, result in zip(inputs, results)]})
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(root / 'run_state.json', state)
    (root / 'report.md').write_text('\n'.join(['# Budgeted real vision check', '',
        f"Processed {len(results)}/{len(inputs)} images; observed {len(observations)}; HTTP requests {initial_attempts}.",
        f"Functional checks: {summary['status']}; formal quality: insufficient_data.", '', *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'observed_images': len(observations),
                      'http_requests': initial_attempts, 'checks': checks}), flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
