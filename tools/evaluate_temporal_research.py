"""Explore temporal grouping on frozen theatre vectors and unverified EXIF hints.

No inferred capture order, boundary or selection preference is promoted to a
human label. All inputs are read-only; the output is a separate research report.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint


def main():
    import cv2
    import numpy as np
    from core_modules.organize.burst_dedup import image_quality
    from core_modules.organize.scene_change_points import ChangePointConfig, split_ordered_scenes
    from core_modules.organize.scene_split import _subcluster
    from core_modules.organize.temporal_burst import BurstConfig, temporal_groups, select_equal_quota

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('new_empty_output_required')
    frozen = json.loads((args.study / 'summary.json').read_text())['artifact_sha256']
    for name in ('input_manifest.json', 'image_vectors.npy'):
        if fingerprint(args.study / name) != frozen[name]:
            raise ValueError('changed_study_artifact:' + name)
    artifact_hashes = json.loads((args.review / 'artifact_manifest.json').read_text())['sha256']
    if fingerprint(args.review / 'images.json') != artifact_hashes['images.json']:
        raise ValueError('changed_review_metadata')
    manifest = json.loads((args.study / 'input_manifest.json').read_text())
    photos = {row['content_hash']: row for row in json.loads((args.review / 'images.json').read_text())}
    vectors = np.load(args.study / 'image_vectors.npy', allow_pickle=False)
    if vectors.ndim != 2 or len(vectors) != len(manifest['rows']) or not np.isfinite(vectors).all():
        raise ValueError('invalid_precomputed_vectors')
    root.mkdir(parents=True, exist_ok=True)
    config, burst_config = ChangePointConfig(), BurstConfig()
    sources = [Path(__file__), ROOT / 'core_modules/organize/scene_change_points.py',
               ROOT / 'core_modules/organize/scene_split.py', ROOT / 'core_modules/organize/temporal_burst.py',
               ROOT / 'core_modules/organize/burst_dedup.py']
    state = {'status': 'running', 'command': [sys.executable, *sys.argv],
             'started_at': datetime.now(timezone.utc).isoformat(),
             'source_sha256': {str(p.relative_to(ROOT)): fingerprint(p) for p in sources},
             'input_sha256': {'images.json': artifact_hashes['images.json'],
                              **{name: frozen[name] for name in ('input_manifest.json', 'image_vectors.npy')}}}
    atomic_json(root / 'run_state.json', state)
    rows, truths, source_files = [], {}, {}
    for query, vector in zip(manifest['rows'], vectors):
        photo = photos[query['content_hash']]
        source = args.source_root / photo['canonical_source']
        if fingerprint(source) != query['content_hash']:
            raise ValueError('original_source_changed')
        source_files[str(source)] = query['content_hash']
        preview = args.review / photo['preview']
        if fingerprint(preview) != photo['preview_sha256']:
            raise ValueError('review_preview_changed')
        pixels = cv2.imdecode(np.frombuffer(preview.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError('invalid_review_preview')
        capture = photo.get('exif', {}).get('capture_local')
        image_id = query['image_id']
        truths[image_id] = query['scene_id']  # Used only after grouping, for descriptive cross-scene counts.
        rows.append({'image': image_id, 'play_id': query['play_id'],
                     'stream_id': photo.get('camera_stream_hint_id'), 'order_trusted': False,
                     'captured_at': datetime.fromisoformat(capture) if capture else None,
                     'embedding': vector, 'quality': image_quality(pixels),
                     'preview_shape': list(pixels.shape[:2])})
    row_index = {row['image']: row for row in rows}
    streams = defaultdict(list)
    for row in rows:
        streams[(row['play_id'], row['stream_id'])].append(row)
    sequence_results = []
    window_seconds = legacy_seconds = 0.
    for (play, stream), members in sorted(streams.items(), key=lambda item: str(item[0])):
        members.sort(key=lambda row: (row['captured_at'] is None, row['captured_at'] or datetime.max, row['image']))
        paths = [row['image'] for row in members]
        values = {row['image']: row['embedding'] for row in members}
        times = {row['image']: row['captured_at'] for row in members}
        started = time.perf_counter()
        window = split_ordered_scenes(paths, values, times, config, order_trusted=bool(stream))
        window_seconds += time.perf_counter() - started
        # Compare subdivision only, on the same predeclared time-gap blocks.
        gaps = window['time_gap_boundaries']
        labels = {}
        started = time.perf_counter()
        for block, (start, end) in enumerate(zip([0] + gaps, gaps + [len(paths)])):
            chunk = paths[start:end]
            clusters = _subcluster(chunk, values, 3) if len(chunk) >= 8 else [chunk]
            for label, images in enumerate(clusters):
                labels.update({image: f'{block}:{label}' for image in images})
        legacy_seconds += time.perf_counter() - started
        ordered_labels = [labels[path] for path in paths]
        legacy_boundaries = [i for i in range(1, len(paths)) if ordered_labels[i] != ordered_labels[i - 1]]
        sequence_results.append({'play_id': play, 'stream_hint': stream, 'image_ids': paths,
                                 'chronology_verified': False, 'performance_id': None,
                                 'window': window, 'legacy_kmeans_assignments': labels,
                                 'legacy_label_transitions': legacy_boundaries,
                                 'legacy_revisited_cluster_runs': len(legacy_boundaries) + 1 - len(set(labels.values()))})
    # Production behavior with current unverified chronology stays conservative.
    conservative, conservative_audit = temporal_groups(rows, burst_config)
    # A separate exploratory arm accepts raw hints only inside this report.
    experimental_rows = [dict(row, order_trusted=bool(row['stream_id'])) for row in rows]
    started = time.perf_counter()
    groups, audit = temporal_groups(experimental_rows, burst_config)
    grouping_seconds = time.perf_counter() - started
    selections, mixed = [], []
    for group_id, images in groups:
        members = [row_index[image] for image in images]
        quota = min(2, len(members))
        selected = select_equal_quota(members, quota, burst_config)
        baseline = sorted(members, key=lambda row: (-row['quality']['score'], row['image']))[:quota]
        scenes = sorted({truths[image] for image in images})
        if len(scenes) > 1:
            mixed.append(group_id)
        selections.append({'group_id': group_id, 'images': images, 'quota': quota,
                           'experimental_selected': selected['selected'],
                           'baseline_selected': [row['image'] for row in baseline],
                           'observed_confirmed_scene_ids_after_grouping': scenes,
                           'human_preference': None, 'must_keep_annotations': None})
    output_images = [image for _, images in groups for image in images]
    checks = {'all_queries_joined_by_exact_source_hash': len(rows) == len(manifest['rows']),
              'grouping_preserves_each_image_once': len(output_images) == len(rows) and set(output_images) == set(row_index),
              'untrusted_production_groups_are_singletons': all(len(images) == 1 for _, images in conservative),
              'equal_keep_quotas': all(len(row['experimental_selected']) == len(row['baseline_selected']) == row['quota'] for row in selections),
              'window_segments_contiguous': all(len(set(item['window']['assignments'].values())) == len(item['window']['boundaries']) + 1 for item in sequence_results),
              'originals_unchanged': all(fingerprint(path) == digest for path, digest in source_files.items()),
              'source_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in state['source_sha256'].items())}
    sizes = Counter(len(images) for _, images in groups)
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'input_count': len(rows), 'dimension': vectors.shape[1], 'stream_hint_count': len(streams),
               'scene_config': asdict(config), 'burst_config': asdict(burst_config),
               'window_algorithm_seconds': window_seconds, 'legacy_subdivision_seconds': legacy_seconds,
               'grouping_seconds': grouping_seconds, 'group_size_histogram': dict(sizes),
               'exploratory_group_count': len(groups), 'groups_with_at_least_3_images': sum(size >= 3 for _, images in groups for size in [len(images)]),
               'groups_spanning_confirmed_scenes': mixed, 'conservative_audit': conservative_audit,
               'exploratory_group_audit': audit, 'sequence_results': sequence_results, 'selections': selections,
               'quality_gate': 'insufficient_data', 'verified_boundaries': 0, 'verified_preferences': 0,
               'default_adoption': False,
               'limitations': ['All 262 queries remain development data; camera/performance/clock order is unverified.',
                              'Window and KMeans subdivision reuse identical SigLIP2 vectors and time-gap blocks.',
                              'Exploratory grouping accepts raw time hints; the conservative arm disables that assumption.',
                              'Confirmed scene labels only describe groups after computation; they are not grouping inputs.',
                              'Selection uses the same cached review previews and quota, without subject-region or IQA models.',
                              'No preference/boundary GT: no hit-rate, unique-frame retention, boundary-F1 or quality improvement claim.',
                              'Reported CPU times are descriptive and exclude original feature encoding.']}
    atomic_json(root / 'summary.json', summary)
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(root / 'run_state.json', state)
    report = ['# Temporal research on stored theatre features', '',
              f"{len(rows)} unique images, {len(streams)} unverified camera streams, {len(groups)} exploratory groups.",
              f"Group-size histogram: {dict(sorted(sizes.items()))}.",
              f"Groups spanning confirmed scene labels: {len(mixed)} (descriptive only).", '',
              f"Artifact checks: {summary['status']}; quality: insufficient_data; no default adoption.", '',
              *summary['limitations'], '']
    (root / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'input_count': len(rows), 'stream_hint_count': len(streams),
                      'group_size_histogram': dict(sizes), 'mixed_scene_groups': len(mixed), 'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
