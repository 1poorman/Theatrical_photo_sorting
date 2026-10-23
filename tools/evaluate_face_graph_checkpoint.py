"""Compare experimental face partitions on an existing real detection checkpoint.

This read-only study does not repeat detection, assign actor names or modify a
gallery. Constraint counts measure supplied assumptions, not verified identity GT.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint


def main():
    import numpy as np
    from core_modules.organize.face_graph_cluster import FaceGraphConfig, cannot_link_pairs, graph_clusters
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-db', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    with sqlite3.connect('file:' + str(args.task_db.resolve()) + '?mode=ro', uri=True) as connection:
        connection.execute('BEGIN')
        task = connection.execute('SELECT kind,request FROM tasks WHERE id=?', (args.task_id,)).fetchone()
        if not task or task[0] != 'face_cluster':
            raise ValueError('face_cluster_checkpoint_required')
        params = json.loads(task[1])['params']
        rows = connection.execute("SELECT result FROM items WHERE task_id=? AND status='succeeded' "
                                  "AND json_extract(input,'$.stage')='faces' ORDER BY position", (args.task_id,)).fetchall()
        pending = connection.execute("SELECT COUNT(*) FROM items WHERE task_id=? AND status!='succeeded' "
                                     "AND json_extract(input,'$.stage')='faces'", (args.task_id,)).fetchone()[0]
        checkpoints = {name: json.loads(payload) for name, payload in connection.execute(
            'SELECT name,payload FROM checkpoints WHERE task_id=?', (args.task_id,))}
    if pending or not rows or 'clusters' not in checkpoints or 'execution_contract' not in checkpoints:
        raise ValueError('complete_detection_and_cluster_plan_required')
    hashes = checkpoints.get('input_fingerprints', {}).get('hashes', params['input_hashes'])
    members = []
    for (raw,) in rows:
        row = json.loads(raw)
        for member in row['members']:
            members.append(dict(member, source_image_sha256=hashes[row['image']]))
    def key(member):
        return member.get('img_path', member['image']), member['face_index']
    if len({key(row) for row in members}) != len(members):
        raise ValueError('duplicate_physical_face_key')
    legacy = {key(member): index for index, cluster in enumerate(checkpoints['clusters']) for member in cluster['members']}
    if set(legacy) != {key(row) for row in members}:
        raise ValueError('legacy_plan_does_not_cover_detection_faces')
    root.mkdir(parents=True, exist_ok=True)
    config = FaceGraphConfig()
    code = {name: fingerprint(ROOT / name) for name in
            ('tools/evaluate_face_graph_checkpoint.py', 'core_modules/organize/face_graph_cluster.py')}
    state = {'status': 'running', 'command': [sys.executable, *sys.argv], 'task_id': args.task_id,
             'source_sha256': code, 'config': asdict(config), 'input_images': len(rows), 'faces': len(members)}
    atomic_json(root / 'run_state.json', state)
    started = time.perf_counter()
    clusters, audit = graph_clusters(members, config)
    seconds = time.perf_counter() - started
    graph = {key(member): index for index, cluster in enumerate(clusters) for member in cluster['members']}
    pairs = cannot_link_pairs(members)
    legacy_violations = sum(legacy[key(members[a])] == legacy[key(members[b])] for a, b in pairs)
    graph_violations = sum(graph[key(members[a])] == graph[key(members[b])] for a, b in pairs)
    artifacts = {'input_vectors.npy': np.asarray([row['feat'] for row in members], dtype=np.float32),
                 'legacy_labels.npy': np.asarray([legacy[key(row)] for row in members], dtype=np.int32),
                 'graph_labels.npy': np.asarray([graph[key(row)] for row in members], dtype=np.int32)}
    for name, value in artifacts.items():
        np.save(root / name, value, allow_pickle=False)
    atomic_json(root / 'clusters.json', clusters)
    atomic_json(root / 'model_contract.json', checkpoints['execution_contract'])
    summary = {'status': 'passed' if graph_violations == 0 and len(graph) == len(members) else 'failed',
               'faces': len(members), 'input_images': len(rows), 'dimension': artifacts['input_vectors.npy'].shape[1],
               'legacy_cluster_count': len(checkpoints['clusters']), 'graph_cluster_count': len(clusters),
               'supplied_cannot_link_pairs': len(pairs), 'legacy_constraint_violations': legacy_violations,
               'graph_constraint_violations': graph_violations, 'graph_seconds': seconds, 'graph_audit': audit,
               'legacy_distance_threshold': params['dist_threshold'], 'graph_config': asdict(config),
               'graph_cluster_size_histogram': dict(Counter(row['size'] for row in clusters)),
               'source_sha256': code, 'artifact_sha256': {name: fingerprint(root / name) for name in artifacts},
               'detection_rows_sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest(),
               'quality_gate': 'insufficient_data', 'verified_face_identities': 0, 'verified_mirror_exceptions': 0,
               'default_adoption': False,
               'limitations': ['Real SCRFD/ArcFace features are reused from the named immutable detection rows.',
                              'Distinct detections in one photograph are treated as cannot-link; mirror/duplicate exceptions are unverified.',
                              'Partition and constraint statistics are not B-cubed identity accuracy or unknown-identity FPIR.',
                              'Legacy and graph have different merging policies; this report does not compare their runtime.',
                              'No actor names, gallery anchors or production defaults are changed.']}
    atomic_json(root / 'summary.json', summary)
    state['status'] = summary['status']
    atomic_json(root / 'run_state.json', state)
    (root / 'report.md').write_text('\n'.join(['# Graph clustering of recorded real faces', '',
        f"{len(members)} faces in {len(rows)} images; legacy clusters {len(checkpoints['clusters'])}, graph clusters {len(clusters)}.",
        f"Supplied cannot-link pairs: {len(pairs)}; violations: legacy {legacy_violations}, graph {graph_violations}.",
        'Formal identity quality remains insufficient_data; legacy remains the default.', '', *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('status', 'faces', 'input_images', 'legacy_cluster_count',
        'graph_cluster_count', 'supplied_cannot_link_pairs', 'legacy_constraint_violations', 'graph_constraint_violations')}), flush=True)


if __name__ == '__main__':
    main()
