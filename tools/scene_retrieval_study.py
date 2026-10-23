"""Versioned retrieval study contracts and classification metrics."""
from collections import Counter
import json
from pathlib import Path

from core_modules.organize.identity_registry import IdentityRegistry
from core_modules.organize.task_artifacts import fingerprint
from tools.scene_text_blocks import POLICY, build_scene_blocks, retrieval_metrics
from tools.sourced_scene_candidates import validate_sourced_manifest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': fingerprint(path)}


def query_fields(row):
    return {k: v for k, v in row.items() if k not in {'neutral_path', 'neutral_sha256'}}


def classification_metrics(rows, truth_classes):
    result = retrieval_metrics(rows)
    labels = list(truth_classes)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError('invalid_truth_classes')
    expected = Counter(row['expected'] for row in rows)
    predicted = Counter(row['ranked_scenes'][0] for row in rows)
    if set(expected) - set(labels):
        raise ValueError('truth_class_missing')
    true_positive = Counter(row['expected'] for row in rows if row['expected'] == row['ranked_scenes'][0])
    # Zero-support classes stay in the declared class set; its size is reported.
    result['macro_f1'] = sum(2 * true_positive[label] / (expected[label] + predicted[label])
        if expected[label] + predicted[label] else 0.0 for label in labels) / len(labels)
    result['macro_f1_class_count'] = len(labels)
    result['macro_f1_class_policy'] = 'predeclared_confirmed_truth_scene_ids'
    return result


def subset_metrics(rows, candidates, query_subsets):
    output = {}
    label_by_id = {c['scene_id']: c['label'] for c in candidates}
    for name, ids in query_subsets.items():
        selected = [row for row in rows if row['file_id'] in set(ids)]
        if not selected:
            continue
        truth = sorted({row['expected'] for row in selected})
        plays = sorted({row['play_id'] for row in selected})
        events = [row for row in selected if label_by_id[row['expected']].split('：')[0].split(':')[0].strip() in {'序曲','序','尾声','谢幕'}]
        output[name] = {'metrics': classification_metrics(selected, truth),
            'per_play': {play: classification_metrics([r for r in selected if r['play_id'] == play],
                sorted({r['expected'] for r in selected if r['play_id'] == play})) for play in plays},
            'non_plot_events': classification_metrics(events, sorted({r['expected'] for r in events})) if events else None}
    return output


def validate_inputs(baseline, same, expanded, historical, labels, registry):
    ids = [row['file_id'] for row in baseline['rows']]
    if not ids or len(ids) != len(set(ids)) or len({row['image_id'] for row in baseline['rows']}) != len(ids):
        raise ValueError('queries_must_be_content_unique')
    if baseline['rows'] != same['rows'] or same['rows'] != expanded['rows'] or baseline['excluded'] != same['excluded'] or same['excluded'] != expanded['excluded']:
        raise ValueError('study_query_sets_differ')
    if len(baseline['scenes']) != len(same['scenes']) or [(c['play_id'],c['scene_id'],c['label']) for c in baseline['scenes']] != [(c['play_id'],c['scene_id'],c['label']) for c in same['scenes']]:
        raise ValueError('study_same_pool_vocabulary_changed')
    if expanded['scenes'][:len(same['scenes'])] != same['scenes']:
        raise ValueError('study_expanded_pool_does_not_preserve_same_pool')
    if len({c['scene_id'] for c in expanded['scenes']}) != len(expanded['scenes']):
        raise ValueError('study_duplicate_scene_id')
    label_rows = {row['file_id']: row for row in labels}
    if any(query_fields(row) != label_rows.get(row['file_id']) for row in baseline['rows']):
        raise ValueError('study_query_labels_changed')
    old_ids = [row['file_id'] for row in historical['rows']]
    if ids[:len(old_ids)] != old_ids or any(query_fields(a) != query_fields(b) or a['neutral_sha256'] != b['neutral_sha256']
            for a,b in zip(baseline['rows'], historical['rows'])):
        raise ValueError('study_historical_queries_changed')
    if any(row['scene_id'] not in {c['scene_id'] for c in expanded['scenes'] if c['play_id'] == row['play_id']} for row in baseline['rows']):
        raise ValueError('study_query_truth_outside_play_pool')
    for manifest in (baseline, same, expanded):
        if manifest['test_locked'] is not False:
            raise ValueError('study_requires_development_inputs')
        validate_sourced_manifest(manifest, registry)
    return {'historical': old_ids, 'added': ids[len(old_ids):], 'all': ids}


def validate_protocol(path, rebuild_blocks=False):
    protocol = read(path)
    if protocol.get('schema_version') != 2 or protocol.get('test_locked') is not False or protocol.get('block_policy') != POLICY:
        raise ValueError('invalid_retrieval_study_protocol')
    if any(not Path(p).is_file() or fingerprint(p) != sha for p,sha in protocol['bound_files_sha256'].items()):
        raise ValueError('retrieval_study_bound_file_changed')
    registry = IdentityRegistry.load(protocol['registry']['path'])
    inputs = {name: read(row['path']) for name,row in protocol['inputs'].items()}
    subsets = validate_inputs(inputs['baseline'], inputs['samepool'], inputs['expanded'], inputs['historical'],
        read(protocol['labels']['path']), registry)
    if subsets != protocol['query_subsets']:
        raise ValueError('study_query_subsets_changed')
    blocks = read(protocol['blocks']['path'])
    if [c['scene_id'] for c in blocks['candidates']] != [c['scene_id'] for c in inputs['expanded']['scenes']]:
        raise ValueError('study_block_candidate_order_changed')
    if rebuild_blocks:
        from tools.prepare_scene_text_block_study import tokenizer_for
        _, count = tokenizer_for(protocol['tokenizer']['model_path'])
        if blocks != build_scene_blocks(inputs['expanded']['scenes'], registry.plays, count):
            raise ValueError('study_blocks_do_not_rebuild')
    return protocol, inputs, blocks
