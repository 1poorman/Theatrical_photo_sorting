"""Evaluate reviewed boxes against frozen real face partitions, without inference.

Only known, evaluable, non-exempt faces have clustering GT. Unknown annotations
do not identify unknown people across photos, and never become singleton GT.
Identity recall/FPIR require a separate, verified and isolated gallery/query run.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from tools.prepare_face_identity_review import digest, validate_decisions, verify_existing
from tools.review_quality_metrics import b_cubed


def face_key(face):
    return face['source_sha256'], face['face_index'], tuple(face['bbox'])


def evaluate_labels(template, draft, members, legacy_labels, graph_labels):
    faces = validate_decisions(template, draft)
    if len(members) != len(legacy_labels) or len(members) != len(graph_labels):
        raise ValueError('face_partition_count_mismatch')
    positions = {face_key(member): i for i, member in enumerate(members)}
    if len(positions) != len(members) or set(positions) != {face_key(face) for face in faces}:
        raise ValueError('review_boxes_and_detection_features_mismatch')
    counts = Counter(face['decision'] for face in faces)
    known = [face for face in faces if face['decision'] == 'known' and not face['exception']]
    identities = {row['identity_id']: row for row in template['identities']}
    actor_names = {identities[face['identity_id']]['actor_name'] for face in known}
    groups = defaultdict(list)
    for face in faces:
        if face['decision'] in ('known', 'unknown') and not face['exception']:
            groups[face['source_sha256']].append(face)
    pairs, annotation_conflicts = [], []
    for group in groups.values():
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                pairs.append((positions[face_key(left)], positions[face_key(right)]))
                if left['decision'] == right['decision'] == 'known' and left['identity_id'] == right['identity_id']:
                    annotation_conflicts.append([left['face_id'], right['face_id']])
    partitions = {}
    for name, labels in (('legacy', legacy_labels), ('graph', graph_labels)):
        indices = [positions[face_key(face)] for face in known]
        partitions[name] = {'known_subset_b_cubed': b_cubed([face['identity_id'] for face in known],
                                                         [labels[index] for index in indices]),
                            'reviewed_cannot_link_pairs': len(pairs),
                            'reviewed_constraint_violations': sum(labels[a] == labels[b] for a, b in pairs),
                            'cluster_count_all_faces': len(set(labels))}
    return {'status': 'validated', 'quality_gate': 'insufficient_data', 'default_adoption': False,
            'face_count': len(faces), 'box_label_counts': {key: counts[key] for key in
                ('pending', 'known', 'unknown', 'not_evaluable')},
            'known_evaluable_non_exempt_faces': len(known), 'distinct_known_identity_ids': len({face['identity_id'] for face in known}),
            'distinct_actor_names': len(actor_names), 'exempt_faces': sum(face['exception'] for face in faces),
            'same_identity_physical_face_pairs_requiring_exception_review': annotation_conflicts,
            'partitions': partitions,
            'identity_matching': {'status': 'insufficient_data', 'known_recall': None, 'unknown_FPIR': None,
                'known_faces_at_least_200': len(known) >= 200, 'unknown_faces_at_least_300': counts['unknown'] >= 300,
                'actors_at_least_10': len(actor_names) >= 10, 'gallery_query_isolation_verified': False,
                'matching_predictions_provided': False},
            'exclusions': {'unreviewed_faces': counts['pending'], 'not_evaluable_faces': counts['not_evaluable'],
                          'unknown_faces_without_cross_photo_identity_GT': counts['unknown'],
                          'known_exempt_faces': sum(face['decision'] == 'known' and face['exception'] for face in faces)},
            'limitations': ['B-cubed is restricted to the explicitly reviewed known-face subset, not all detections.',
                'Unknown means outside the selected gallery identities; different unknown faces are not presumed different people.',
                'Reviewed same-image non-exempt detections supply physical-face cannot-link assumptions; conflicting labels need review.',
                'No verified isolated gallery/query predictions, unknown-identity FPIR, calibration or independent final test.',
                'Machine partitions and feature vectors are reused; no faces, actor identities or defaults are assigned.']}


def load_partitions(template, task_db, study):
    import numpy as np
    task_id = template['binding']['task_id']
    with sqlite3.connect(task_db.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.execute('BEGIN')
        if connection.execute('SELECT kind,status FROM tasks WHERE id=?', (task_id,)).fetchone() != ('face_cluster', 'completed'):
            raise ValueError('completed_named_face_task_required')
        detections = [json.loads(raw) for (raw,) in connection.execute(
            "SELECT result FROM items WHERE task_id=? AND status='succeeded' "
            "AND json_extract(input,'$.stage')='faces' ORDER BY position", (task_id,))]
        checkpoints = {name: json.loads(raw) for name, raw in connection.execute(
            'SELECT name,payload FROM checkpoints WHERE task_id=?', (task_id,))}
    if digest(detections) != template['binding']['detection_digest']:
        raise ValueError('review_detection_checkpoint_changed')
    if checkpoints['execution_contract'] != template['binding']['execution_contract']:
        raise ValueError('review_model_contract_changed')
    summary = json.loads((study / 'summary.json').read_text())
    if summary.get('status') != 'passed':
        raise ValueError('completed_partition_study_required')
    if json.loads((study / 'model_contract.json').read_text()) != checkpoints['execution_contract']:
        raise ValueError('partition_model_contract_mismatch')
    arrays = {}
    for name in ('input_vectors.npy', 'legacy_labels.npy', 'graph_labels.npy'):
        if fingerprint(study / name) != summary['artifact_sha256'][name]:
            raise ValueError('partition_artifact_changed:' + name)
        arrays[name] = np.load(study / name, allow_pickle=False)
    hashes = checkpoints['input_fingerprints']['hashes']
    members, features, physical_keys = [], [], []
    for row in detections:
        for member in row['members']:
            members.append({'source_sha256': hashes[row['image']], 'face_index': member['face_index'], 'bbox': member['bbox']})
            features.append(member['feat'])
            physical_keys.append((member.get('img_path', member['image']), member['face_index']))
    expected = np.asarray(features, dtype=np.float32)
    if not np.array_equal(expected, arrays['input_vectors.npy']) or not np.isfinite(expected).all():
        raise ValueError('partition_feature_order_or_values_changed')
    legacy = {(member.get('img_path', member['image']), member['face_index']): index
              for index, cluster in enumerate(checkpoints['clusters']) for member in cluster['members']}
    graph_clusters = json.loads((study / 'clusters.json').read_text())
    graph = {(member.get('img_path', member['image']), member['face_index']): index
             for index, cluster in enumerate(graph_clusters) for member in cluster['members']}
    if len(legacy) != len(members) or len(graph) != len(members):
        raise ValueError('partition_members_not_unique_or_complete')
    for name, mapping in (('legacy', legacy), ('graph', graph)):
        labels = arrays[name + '_labels.npy']
        if labels.shape != (len(members),) or labels.dtype.kind not in ('i', 'u') or set(mapping) != set(physical_keys):
            raise ValueError('invalid_partition_labels')
        if not np.array_equal(labels, np.asarray([mapping[key] for key in physical_keys])):
            raise ValueError('partition_member_alignment_changed')
    return members, arrays['legacy_labels.npy'].tolist(), arrays['graph_labels.npy'].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--task-db', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('new_empty_output_required')
    for source in (args.review, args.study):
        if output == source.resolve() or source.resolve() in output.parents:
            raise ValueError('evaluation_output_inside_frozen_input')
    template = verify_existing(args.review)
    decisions_path = args.decisions or args.review / 'review_template.json'
    draft = json.loads(decisions_path.read_text())
    inputs = [decisions_path, args.review / 'artifact_manifest.json', args.study / 'summary.json',
              args.study / 'clusters.json', args.study / 'model_contract.json']
    input_sha = {str(path.resolve()): fingerprint(path) for path in inputs}
    members, legacy, graph = load_partitions(template, args.task_db, args.study)
    result = evaluate_labels(template, draft, members, legacy, graph)
    if any(fingerprint(path) != sha for path, sha in input_sha.items()):
        raise ValueError('evaluation_input_changed_during_read')
    result.update(protocol_sha256=template['protocol_sha256'], task_id=template['binding']['task_id'],
                  input_sha256=input_sha, source_sha256={name: fingerprint(ROOT / name) for name in
                    ('tools/evaluate_face_identity_review.py', 'tools/review_quality_metrics.py', 'tools/prepare_face_identity_review.py')})
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'summary.json', result)
    (output / 'report.md').write_text('\n'.join(['# Reviewed face partition evaluation', '',
        f"Faces: {result['face_count']}; box labels: {result['box_label_counts']}.",
        f"Known evaluable non-exempt faces: {result['known_evaluable_non_exempt_faces']}.",
        'Formal identity quality remains insufficient_data; missing metrics are null.', '', *result['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'quality_gate': result['quality_gate'],
                      'faces': result['face_count'], 'box_label_counts': result['box_label_counts']}))


if __name__ == '__main__':
    main()
