"""Prepare offline photo/partition review evidence without promoting hints to GT.

Originals, gallery, imported labels and historical splits are read-only. Saved
per-image checkpoints support interruption recovery. All new assignments remain
development review data; an inspected photo is never made a fresh final test.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageOps, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.filename_parser import parse_filename, parse_dirname
from core_modules.organize.task_artifacts import atomic_json, fingerprint

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SOURCE_FILES = ('tools/prepare_photo_partition_review.py',
                'core_modules/organize/filename_parser.py',
                'core_modules/organize/task_artifacts.py')
POLICY = {'name': 'photo_partition_review_v1', 'near_dhash_max': 8,
          'near_phash_max': 8, 'near_aspect_delta_max': .04,
          'near_color_distance_max': .18, 'time_neighbor_max_seconds': 2.,
          'capture_gap_review_min_seconds': 60., 'final_test_locked': False,
          'all_assignments': 'development_review',
          'camera_clocks_synchronized': False, 'human_decisions_applied': False}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable(prefix, value):
    return prefix + '-' + hashlib.sha256(value.encode()).hexdigest()[:24]


def safe_source(root, relative):
    candidate = root / relative
    if Path(relative).is_absolute() or any(p.is_symlink() for p in [candidate, *candidate.parents]):
        raise ValueError('invalid_or_symlink_source: ' + str(relative))
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError('source_outside_root_or_missing: ' + str(relative))
    return resolved


def image_files(root):
    return sorted(safe_source(root, path.relative_to(root).as_posix()) for path in root.rglob('*')
                  if path.is_file() and path.suffix.lower() in EXTENSIONS)


def inventory_paths_match(root, rows):
    return {p.relative_to(root).as_posix() for p in image_files(root)} == {r['source'] for r in rows}


def load_inventory(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({r['file_id'] for r in rows}) != len(rows) or len({r['source'] for r in rows}) != len(rows):
        raise ValueError('empty_or_duplicate_inventory')
    if any(r['status'] != 'ok' for r in rows):
        raise ValueError('inventory_contains_decode_failures')
    return rows


def bind_labels(rows, labels):
    by_file = {r['file_id']: r for r in rows}
    by_image = {}
    for label in labels:
        row = by_file.get(label['file_id'])
        if row is None or any(row[k] != label[k] for k in ('source', 'image_id', 'content_hash')):
            raise ValueError('label_inventory_mismatch')
        previous = by_image.get(label['image_id'])
        if previous and (previous['play_id'], previous['scene_id']) != (label['play_id'], label['scene_id']):
            raise ValueError('conflicting_labels_for_exact_image')
        by_image[label['image_id']] = label
    return by_image


def exposure_from_manifest(manifest, rows, reference):
    """IDs propagate exposure to every exact copy; unsupported inputs fail closed."""
    records = manifest.get('rows', manifest.get('images')) if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ValueError('unsupported_exposure_manifest: ' + reference)
    by_id = {r['image_id']: r for r in rows}
    seen = set()
    for record in records:
        row = by_id.get(record.get('image_id'))
        if row is None or record.get('content_hash') != row['content_hash']:
            raise ValueError('exposure_manifest_inventory_mismatch: ' + reference)
        seen.add(row['image_id'])
    return seen


def text_tag(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value).strip(' \x00') or None


def original_capture(exif):
    """DateTime (306) is an edit time, never a substitute for capture time."""
    try:
        nested = exif.get_ifd(34665)
    except (KeyError, TypeError, ValueError, SyntaxError):
        nested = {}
    tag = lambda number: text_tag(nested.get(number, exif.get(number)))
    original = tag(36867)
    subsecond = tag(37521)
    offset = tag(36881)
    capture = None
    if original:
        try:
            value = datetime.strptime(original, '%Y:%m:%d %H:%M:%S')
            if subsecond and subsecond.isdigit():
                value += timedelta(microseconds=int((subsecond + '000000')[:6]))
            capture = value.isoformat(timespec='microseconds')
        except ValueError:
            pass
    serial = tag(42033)
    return {'original_raw': original, 'subsecond_raw': subsecond, 'offset_raw': offset,
            'capture_local': capture, 'edit_time_raw': text_tag(exif.get(306)),
            'camera_make': text_tag(exif.get(271)), 'camera_model': text_tag(exif.get(272)),
            'camera_body_id': stable('body', serial) if serial else None,
            'timezone_verified': False, 'capture_verified': False}


def features(image):
    gray = image.convert('L')
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    dhash = sum(int(bit) << i for i, bit in enumerate((small[:, 1:] > small[:, :-1]).reshape(-1)))
    grid = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    positions = np.arange(32, dtype=np.float64)
    frequencies = np.arange(8, dtype=np.float64)
    basis = np.cos(np.pi * (2 * positions[None, :] + 1) * frequencies[:, None] / 64)
    basis[0] *= 1 / np.sqrt(2)
    low = (basis @ grid @ basis.T).reshape(-1)[1:]
    phash = sum(int(bit) << i for i, bit in enumerate(low > np.median(low)))
    color = np.asarray(image.resize((8, 8), Image.Resampling.BICUBIC), dtype=np.float64).mean(axis=(0, 1)) / 255
    return {'dhash': f'{dhash:016x}', 'phash_ac63': f'{phash:016x}',
            'mean_rgb': color.tolist(), 'gray_std': float(grid.std()),
            'aspect': image.width / image.height}


def checked_checkpoint(path, image_id, content_hash, protocol_sha, output):
    value = json.loads(path.read_text())
    checksum = value.pop('payload_sha256', None)
    if checksum != digest_json(value) or value['image_id'] != image_id or value['content_hash'] != content_hash \
            or value['protocol_sha256'] != protocol_sha:
        raise ValueError('changed_image_checkpoint')
    preview = safe_source(output, value['preview'])
    if fingerprint(preview) != value['preview_sha256']:
        raise ValueError('changed_review_preview')
    return value


def prepare_image(row, source, output, protocol_sha, resume):
    checkpoint = output / 'image_checkpoints' / (row['image_id'] + '.json')
    if checkpoint.exists():
        if not resume:
            raise ValueError('unexpected_existing_checkpoint')
        return checked_checkpoint(checkpoint, row['image_id'], row['content_hash'], protocol_sha, output), True
    with Image.open(source) as original:
        metadata = original_capture(original.getexif())
        image = ImageOps.exif_transpose(original).convert('RGB')
        pixels = hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode())
        pixels.update(image.tobytes())
        if pixels.hexdigest() != row['image_id'] or (image.width, image.height) != (row['width'], row['height']):
            raise ValueError('decoded_inventory_image_changed')
        visual = features(image)
        if visual['dhash'] != row['dhash']:
            raise ValueError('historical_dhash_not_reproduced')
        image.thumbnail((640, 480))
        image.info.clear()
        relative = 'previews/' + row['image_id'][:24] + '.jpg'
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.pending.jpg')
        image.save(temporary, quality=85)
        temporary.replace(destination)
    if fingerprint(source) != row['content_hash']:
        raise ValueError('source_changed_during_decode')
    value = {'image_id': row['image_id'], 'content_hash': row['content_hash'],
             'protocol_sha256': protocol_sha, 'width': row['width'], 'height': row['height'],
             'exif': metadata, 'features': visual, 'preview': relative,
             'preview_sha256': fingerprint(destination)}
    atomic_json(checkpoint, {**value, 'payload_sha256': digest_json(value)})
    return value, False


def canonical_row(members):
    return min(members, key=lambda r: ('原始' not in r['source'], r['source']))


def build_images(rows, labels_by_image, checkpoints, exposure):
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    images = []
    for image_id, members in sorted(groups.items()):
        row = canonical_row(members)
        label = labels_by_image.get(image_id)
        parsed = parse_filename(Path(row['source']).name, done_style='整理完成' in row['source'])
        directory = parse_dirname(Path(row['source']).parent.name)
        exif = checkpoints[image_id]['exif']
        camera = exif['camera_body_id'] or exif['camera_model'] or 'camera_unknown'
        stream_key = str(Path(row['source']).parent) + ':' + camera + ':' + (exif['offset_raw'] or 'offset_unknown')
        imported_scene = label.get('scene_label') if label else None
        date_hint = row.get('performance_hint')
        candidate_performance = stable('performance-hint', row['play_id'] + ':' + (date_hint or 'date_unknown'))
        images.append({**checkpoints[image_id], 'play_title': row['play_id'],
            'play_id': label['play_id'] if label else None, 'canonical_source': row['source'],
            'source_file_ids': [r['file_id'] for r in members], 'sources': [r['source'] for r in members],
            'source_kind_hint': row['source_kind'], 'source_kind_verified': False,
            'legacy_splits': sorted({r['split'] for r in members}),
            'legacy_duplicate_groups': sorted({r['duplicate_group_id'] for r in members}),
            'imported_label_file_id': label['file_id'] if label else None,
            'imported_scene': imported_scene, 'imported_scene_id': label['scene_id'] if label else None,
            'annotation_source': label['annotation_source'] if label else None,
            'people_list_exhaustive': False, 'face_boxes_verified': False,
            'date_hint': date_hint, 'date_hint_matches_capture':
                exif['capture_local'][:10].replace('-', '') == date_hint if exif['capture_local'] and date_hint else None,
            'performance_hint_id': candidate_performance, 'performance_id': None,
            'performance_verified': False, 'camera_stream_hint_id': stable('stream-hint', stream_key),
            'photographer_hint': parsed.get('photographer') or directory.get('photographer'),
            'event_hint': directory.get('event') or parsed.get('event'),
            'filename_base_hint': parsed.get('base'), 'filename_serial_hint': parsed.get('burst_seq'),
            'organized_sequence_hint': label.get('sequence_hint') if label else None,
            'possible_unimported_scene_hint': bool(label and not imported_scene and re.search(r'第\d+[幕场]|尾声|谢幕|序曲', Path(label['source']).stem)),
            'exposure_refs': sorted(reference for reference, ids in exposure.items() if image_id in ids),
            'proposed_usage': 'development_review', 'final_test_eligible': False,
            'burst_group_id': None, 'chronology_verified': False})
    return images


def near_pairs(images, policy=POLICY):
    pairs = []
    for i, first in enumerate(images):
        for second in images[:i]:
            if first['play_title'] != second['play_title']:
                continue
            a, b = first['features'], second['features']
            aspect = abs(a['aspect'] - b['aspect'])
            color = float(np.linalg.norm(np.asarray(a['mean_rgb']) - b['mean_rgb']))
            dh = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
            ph = (int(a['phash_ac63'], 16) ^ int(b['phash_ac63'], 16)).bit_count()
            legacy = bool(set(first['legacy_duplicate_groups']) & set(second['legacy_duplicate_groups']))
            heuristic = aspect <= policy['near_aspect_delta_max'] and color <= policy['near_color_distance_max'] \
                and (dh <= policy['near_dhash_max'] or ph <= policy['near_phash_max'])
            if not legacy and not heuristic:
                continue
            ids = sorted([first['image_id'], second['image_id']])
            pairs.append({'review_id': stable('near-review', ':'.join(ids)), 'image_ids': ids,
                          'dhash_hamming': dh, 'phash_ac63_hamming': ph,
                          'aspect_delta': aspect, 'mean_rgb_distance': color,
                          'legacy_heuristic_group': legacy, 'relaxed_heuristic_match': heuristic,
                          'low_information_image': min(a['gray_std'], b['gray_std']) < 5,
                          'scene_labels_differ': bool(first['imported_scene'] and second['imported_scene']
                                                     and first['imported_scene'] != second['imported_scene']),
                          'status': 'pending_human_review', 'verified': False})
    return sorted(pairs, key=lambda r: (not r['legacy_heuristic_group'], min(r['dhash_hamming'], r['phash_ac63_hamming']), r['review_id']))


def temporal_reviews(images, policy=POLICY):
    by_stream, by_base = defaultdict(list), defaultdict(list)
    for image in images:
        if image['source_kind_hint'] != 'stage_photo' or '原始' not in image['canonical_source']:
            continue
        by_stream[image['camera_stream_hint_id']].append(image)
        if image['filename_base_hint']:
            by_base[(image['camera_stream_hint_id'], image['filename_base_hint'])].append(image)
    streams, neighbors, boundaries, inversions = [], [], [], []
    for stream_id, members in sorted(by_stream.items()):
        timed = sorted((r for r in members if r['exif']['capture_local']),
                       key=lambda r: (r['exif']['capture_local'], r['image_id']))
        streams.append({'stream_hint_id': stream_id, 'play_title': members[0]['play_title'],
                        'camera_model': members[0]['exif']['camera_model'],
                        'camera_body_known': members[0]['exif']['camera_body_id'] is not None,
                        'source_directory': str(Path(members[0]['canonical_source']).parent),
                        'photographer_hint': members[0]['photographer_hint'],
                        'image_ids_in_capture_hint_order': [r['image_id'] for r in timed],
                        'missing_capture_image_ids': [r['image_id'] for r in members if not r['exif']['capture_local']],
                        'capture_first': timed[0]['exif']['capture_local'] if timed else None,
                        'capture_last': timed[-1]['exif']['capture_local'] if timed else None,
                        'chronology_verified': False, 'performance_id': None})
        for first, second in zip(timed, timed[1:]):
            gap = (datetime.fromisoformat(second['exif']['capture_local']) - datetime.fromisoformat(first['exif']['capture_local'])).total_seconds()
            transition = bool(first['imported_scene'] and second['imported_scene'] and first['imported_scene'] != second['imported_scene'])
            edge = {'review_id': stable('time-review', stream_id + ':' + first['image_id'] + ':' + second['image_id']),
                    'stream_hint_id': stream_id, 'image_ids': [first['image_id'], second['image_id']],
                    'capture_gap_seconds': gap, 'imported_scene_change_hint': transition,
                    'status': 'pending_human_review', 'verified': False}
            if gap <= policy['time_neighbor_max_seconds']:
                neighbors.append(edge)
            if gap >= policy['capture_gap_review_min_seconds'] or transition:
                boundaries.append({**edge, 'reason_hints': (["capture_gap"] if gap >= policy['capture_gap_review_min_seconds'] else []) + (["imported_scene_change"] if transition else [])})
        curated = sorted((r for r in timed if r['organized_sequence_hint'] is not None),
                         key=lambda r: (r['organized_sequence_hint'], r['image_id']))
        for first, second in zip(curated, curated[1:]):
            if second['exif']['capture_local'] < first['exif']['capture_local']:
                inversions.append({'stream_hint_id': stream_id, 'image_ids': [first['image_id'], second['image_id']],
                                   'organized_sequence_hints': [first['organized_sequence_hint'], second['organized_sequence_hint']],
                                   'capture_local_hints': [first['exif']['capture_local'], second['exif']['capture_local']],
                                   'status': 'pending_clock_or_order_review'})
    filename_groups = []
    for (stream_id, base), members in sorted(by_base.items()):
        if len(members) < 2:
            continue
        times = sorted(r['exif']['capture_local'] for r in members if r['exif']['capture_local'])
        span = (datetime.fromisoformat(times[-1]) - datetime.fromisoformat(times[0])).total_seconds() if times else None
        labels = sorted({r['imported_scene'] for r in members if r['imported_scene']})
        filename_groups.append({'review_id': stable('filename-review', stream_id + ':' + base),
            'stream_hint_id': stream_id, 'filename_base_hint': base,
            'image_ids': sorted(r['image_id'] for r in members), 'capture_span_seconds': span,
            'imported_scene_labels': labels, 'multiple_imported_scenes': len(labels) > 1,
            'status': 'filename_group_only_burst_unverified', 'burst_group_id': None})
    return {'streams': streams, 'time_neighbors': neighbors, 'boundary_hints': boundaries,
            'organized_order_inversions': inversions, 'filename_groups': filename_groups,
            'thresholds': policy, 'verified_bursts': 0, 'verified_boundaries': 0}


def gallery_review(gallery_root, rows, registry, images, output):
    by_stem, by_directory = defaultdict(list), defaultdict(list)
    for row in rows:
        by_stem[Path(row['source']).stem].append(row)
    for identity in registry['identities']:
        for directory in identity['gallery_dirs']:
            by_directory[directory].append(identity['identity_id'])
    image_lookup = {r['image_id']: r for r in images}
    entries = []
    paths = image_files(gallery_root)
    for path in paths:
        relative = path.relative_to(gallery_root).as_posix()
        safe_source(gallery_root, relative)
        digest = fingerprint(path)
        match = re.fullmatch(r'(.*)_face_(\d+)', path.stem)
        candidates = by_stem.get(match.group(1), []) if match else []
        source = candidates[0] if len(candidates) == 1 else None
        image = image_lookup[source['image_id']] if source else None
        preview = 'gallery_previews/' + stable('crop', relative) + '.jpg'
        with Image.open(path) as crop:
            crop = ImageOps.exif_transpose(crop).convert('RGB')
            width, height = crop.size
            crop.thumbnail((160, 160))
            crop.info.clear()
            destination = output / preview
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.save(destination, quality=85)
        entries.append({'review_id': stable('gallery-review', relative), 'gallery_source': relative,
                        'content_hash': digest, 'width': width, 'height': height,
                        'preview': preview, 'preview_sha256': fingerprint(destination),
                        'registry_identity_ids': sorted(by_directory[path.parent.name]),
                        'face_index_hint': int(match.group(2)) if match else None,
                        'filename_source_match_count': len(candidates),
                        'parent_source_file_id_hint': source['file_id'] if source else None,
                        'parent_image_id_hint': source['image_id'] if source else None,
                        'parent_exposure_refs': image['exposure_refs'] if image else [],
                        'provenance_verified': False, 'crop_box_verified': False,
                        'status': 'filename_provenance_hint_only'})
    return entries


def partition_review(rows, images, gallery, temporal, exposure):
    by_play = defaultdict(list)
    gallery_ids = {r['parent_image_id_hint'] for r in gallery if r['parent_image_id_hint']}
    related_ids = set(gallery_ids)
    for pair in temporal['time_neighbors']:
        if set(pair['image_ids']) & gallery_ids:
            related_ids.update(pair['image_ids'])
    for image in images:
        image['gallery_parent_filename_hint'] = image['image_id'] in gallery_ids
        image['gallery_time_neighbor_hint'] = image['image_id'] in related_ids - gallery_ids
        by_play[image['play_title']].append(image)
    performance = []
    for title, members in sorted(by_play.items()):
        stage = [r for r in members if r['source_kind_hint'] == 'stage_photo']
        hints = sorted({r['date_hint'] for r in stage if r['date_hint']})
        performance.append({'play_title': title, 'date_hints': hints, 'verified_performance_ids': [],
            'stage_unique_images': len(stage), 'reference_unique_images': len(members) - len(stage),
            'scene_labeled_images': sum(bool(r['imported_scene']) for r in members),
            'stage_images_without_imported_labels': sum(not r['imported_label_file_id'] for r in stage),
            'observed_capture_dates': sorted({r['exif']['capture_local'][:10] for r in stage if r['exif']['capture_local']}),
            'independent_performance_available': 'not_verified', 'proposed_usage': 'development_review'})
    legacy = defaultdict(list)
    for row in rows:
        legacy[row['split']].append(row)
    return {'schema_version': 1, 'status': 'review_prepared_final_test_not_locked',
            'final_test_locked': False, 'verified_performances': 0,
            'performance_review': performance,
            'legacy_split_exposure': {key: {'files': len(value),
                'unique_images': len({r['image_id'] for r in value}),
                'known_exposed_unique_images': len({r['image_id'] for r in value
                    if any(r['image_id'] in ids for ids in exposure.values())})}
                for key, value in sorted(legacy.items())},
            'exposure_by_manifest': {ref: len(ids) for ref, ids in sorted(exposure.items())},
            'gallery_source_hint_unique_images': len(gallery_ids),
            'gallery_source_and_time_neighbor_hint_unique_images': len(related_ids),
            'new_assignments': {r['file_id']: 'development_review' for r in rows},
            'new_assignments_policy': 'all_existing_local_images_development_review_v1',
            'limitations': ['Exact copies share exposure; supplied manifests are a lower bound on prior use.',
                'Filename date, camera stream, source category and crop provenance require human verification.',
                'Camera model can represent several bodies if serial is absent; clocks are not synchronized.',
                'Short time gaps and similar whole-image hashes do not establish bursts or safe gallery isolation.',
                'Legacy final_test_provisional is preserved as history; exposed samples cannot be a fresh final test.',
                'No face boxes, unknown identities, burst preferences or shooting boundaries were annotated.']}


def review_template(images, pairs, temporal, gallery, partitions):
    items = []
    for performance in partitions['performance_review']:
        items.append({'review_id': stable('performance-review', performance['play_title']),
            'kind': 'performance', 'image_ids': [], 'evidence': performance,
            'allowed_decisions': ['pending', 'same_performance', 'multiple_performances', 'insufficient_evidence']})
    for pair in pairs:
        items.append({'review_id': pair['review_id'], 'kind': 'near_duplicate',
            'image_ids': pair['image_ids'], 'evidence': pair,
            'allowed_decisions': ['pending', 'near_duplicate', 'burst_related', 'different_images', 'insufficient_evidence']})
    for kind, source, choices in (
        ('time_neighbor', temporal['time_neighbors'], ['same_burst', 'different_bursts', 'clock_uncertain']),
        ('boundary', temporal['boundary_hints'], ['scene_boundary', 'shooting_pause', 'neither', 'clock_uncertain']),
        ('filename_group', temporal['filename_groups'], ['one_burst', 'multiple_bursts', 'insufficient_evidence']),
        ('order', temporal['organized_order_inversions'], ['curated_order', 'clock_uncertain', 'insufficient_evidence'])):
        for entry in source:
            review_id = entry.get('review_id') or stable('order-review', ':'.join(entry['image_ids']))
            items.append({'review_id': review_id, 'kind': kind, 'image_ids': entry['image_ids'],
                          'evidence': entry, 'allowed_decisions': ['pending', *choices]})
    for image in images:
        kind = 'unlabeled_stage' if image['source_kind_hint'] == 'stage_photo' and not image['imported_scene'] else \
               'reference_material' if image['source_kind_hint'] == 'reference_material' else None
        if kind:
            items.append({'review_id': stable('label-review', image['image_id']), 'kind': kind,
                'image_ids': [image['image_id']], 'evidence': {
                    'source_kind_hint': image['source_kind_hint'], 'possible_unimported_scene_hint': image['possible_unimported_scene_hint'],
                    'exposure_refs': image['exposure_refs'], 'original_category_verified': False},
                'allowed_decisions': ['pending', 'stage_photo', 'reference_material', 'insufficient_evidence']})
    by_parent = defaultdict(list)
    for crop in gallery:
        by_parent[crop['parent_image_id_hint'] or crop['review_id']].append(crop)
    for parent, crops in sorted(by_parent.items()):
        parent_id = crops[0]['parent_image_id_hint']
        items.append({'review_id': stable('gallery-parent-review', parent), 'kind': 'gallery_provenance',
            'image_ids': [parent_id] if parent_id else [], 'gallery_review_ids': [r['review_id'] for r in crops],
            'evidence': {'parent_source_is_filename_hint_only': True, 'crop_count': len(crops),
                         'parent_exposure_refs': crops[0]['parent_exposure_refs'], 'crop_boxes_unverified': True},
            'allowed_decisions': ['pending', 'parent_confirmed', 'parent_incorrect', 'insufficient_evidence']})
    for item in items:
        item.update(decision='pending', notes='', reviewer='', source_ref='', scene_label=None,
                    performance_id=None, burst_group_id=None, reviewed_at=None)
    return {'schema_version': 1, 'status': 'pending_human_review', 'human_decisions_applied': False,
            'final_test_locked': False, 'items': items}


KIND_NAMES = {'performance': '演出场次', 'near_duplicate': '近重复', 'time_neighbor': '拍摄时间相邻',
              'boundary': '边界线索', 'filename_group': '同名照片组', 'order': '整理顺序',
              'unlabeled_stage': '未导入幕次', 'reference_material': '资料图', 'gallery_provenance': '图库来源'}
DECISION_NAMES = {'pending': '待核查', 'same_performance': '同一场次', 'multiple_performances': '多个场次',
                 'insufficient_evidence': '证据不足', 'near_duplicate': '近重复', 'burst_related': '连拍相关',
                 'different_images': '不同画面', 'same_burst': '同一连拍', 'different_bursts': '不同连拍',
                 'clock_uncertain': '相机时间待核验', 'scene_boundary': '剧情切幕', 'shooting_pause': '拍摄停顿',
                 'neither': '均非', 'one_burst': '一个连拍组', 'multiple_bursts': '多个连拍组',
                 'curated_order': '人工整理顺序', 'stage_photo': '舞台照', 'reference_material': '资料图',
                 'parent_confirmed': '源图已核对', 'parent_incorrect': '源图对应错误'}
CSS = """body{margin:24px auto;max-width:1200px;padding:0 18px;font:16px/1.6 system-ui;color:#17212b;background:#f6f5f2}h1{font-size:28px}a{color:#185fa0}nav,.toolbar{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}.card,details{background:white;border:1px solid #ddd;border-radius:8px;padding:16px;margin:14px 0}.photos{display:flex;flex-wrap:wrap;gap:14px}figure{margin:0;max-width:340px}figure img{max-width:100%;max-height:250px;object-fit:contain;background:#151515}figcaption{font-size:13px;overflow-wrap:anywhere}.crop img{max-width:120px;max-height:120px}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}select,input,textarea,button{font:inherit;padding:6px;border:1px solid #aaa;border-radius:4px}textarea{width:90%;min-height:52px}.muted{color:#56636e}.decision{margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ddd;padding:7px;text-align:left}.hidden{display:none}.scroll{overflow-x:auto}"""


def page(title, body, script=''):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
        + '<title>' + html.escape(title) + '</title><style>' + CSS + '</style><body><h1>' + html.escape(title) \
        + '</h1><nav><a href="index.html">复核清单</a><a href="timeline.html">相机时间序列</a><a href="report.md">报告</a></nav>' \
        + body + script + '</body></html>'


def figure(image):
    capture = image['exif']['capture_local'] or '无原始拍摄时间'
    label = image['imported_scene'] or '未导入幕次'
    return '<figure><a href="' + image['preview'] + '"><img loading="lazy" src="' + image['preview'] \
        + '"></a><figcaption>' + html.escape(image['play_title'] + ' · ' + label + ' · ' + capture) \
        + '<br>' + html.escape(image['canonical_source']) + '</figcaption></figure>'


def render_pages(output, images, gallery, temporal, template, summary, protocol_sha):
    by_id = {r['image_id']: r for r in images}
    crops = {r['review_id']: r for r in gallery}
    cards = []
    for item in template['items']:
        photos = ''.join(figure(by_id[image_id]) for image_id in item['image_ids'])
        for crop_id in item.get('gallery_review_ids', []):
            crop = crops[crop_id]
            photos += '<figure class="crop"><img loading="lazy" src="' + crop['preview'] \
                + '"><figcaption>' + html.escape(crop['gallery_source']) + '</figcaption></figure>'
        options = ''.join('<option value="' + value + '">' + DECISION_NAMES[value] + '</option>' for value in item['allowed_decisions'])
        search = ' '.join([KIND_NAMES[item['kind']], json.dumps(item['evidence'], ensure_ascii=False),
                           *[by_id[i]['canonical_source'] for i in item['image_ids']]])
        cards.append('<article class="card" data-kind="' + item['kind'] + '" data-review="' + item['review_id'] \
            + '" data-search="' + html.escape(search, quote=True) + '"><b>' + KIND_NAMES[item['kind']] \
            + '</b> <small>' + item['review_id'] + '</small><div class="photos">' + photos \
            + '</div><details><summary>查看线索</summary><pre>' + html.escape(json.dumps(item['evidence'], ensure_ascii=False, indent=2)) \
            + '</pre></details><div class="decision"><select data-field="decision">' + options \
            + '</select> <input data-field="scene_label" placeholder="核对后的幕次（可选）"> '
            + '<input data-field="performance_id" placeholder="场次ID（可选）"> '
            + '<input data-field="burst_group_id" placeholder="连拍组ID（可选）"></div>'
            + '<textarea data-field="notes" placeholder="核查说明、分歧与必须保留的画面"></textarea>'
            + '<input data-field="source_ref" placeholder="确认依据或引用"></article>')
    toolbar = '<p>所有项目待人工核查。图库来源由裁剪文件名匹配，尚未核对裁剪框；相机时间未校准。此页面不会改写原图、标注或图库。</p>' \
        + '<p>草稿保存在当前浏览器，可导出JSON交接。导出文件仍为复核结果，不会自动应用标签或锁定测试集。</p>' \
        + '<div class="toolbar"><select id="kind"><option value="">全部类型</option>' \
        + ''.join('<option value="' + key + '">' + name + '</option>' for key, name in KIND_NAMES.items()) \
        + '</select><input id="search" placeholder="筛选剧目、幕次、线索"><label><input id="pending" type="checkbox">只看待核查</label>' \
        + '<input id="reviewer" placeholder="核查人"><button id="export">导出核查JSON</button><label>恢复JSON <input id="import" type="file" accept="application/json"></label></div><p id="message" role="status"></p>'
    data = json.dumps({**template, 'protocol_sha256': protocol_sha}, ensure_ascii=False).replace('<', '\\u003c')
    script = '<script id="review-data" type="application/json">' + data + '</script><script>' + r"""
const original=JSON.parse(document.getElementById('review-data').textContent), key='photo-partition-review:'+original.protocol_sha256;
let draft=structuredClone(original); const message=document.getElementById('message');
function validate(value){if(value.protocol_sha256!==original.protocol_sha256||!Array.isArray(value.items)||value.items.length!==original.items.length)throw Error('复核版本或项目数不匹配');const indexed=new Map(value.items.map(r=>[r.review_id,r]));if(indexed.size!==original.items.length)throw Error('复核项目ID重复');for(const item of original.items){const row=indexed.get(item.review_id);if(!row||row.kind!==item.kind||JSON.stringify(row.image_ids)!==JSON.stringify(item.image_ids)||!item.allowed_decisions.includes(row.decision))throw Error('复核项目或决定不匹配');}return value;}
try{const saved=localStorage.getItem(key);if(saved)draft=validate(JSON.parse(saved));}catch(error){message.textContent='无法恢复浏览器草稿：'+error.message;}
function persist(){draft.final_test_locked=false;draft.human_decisions_applied=false;draft.status='human_review_draft_not_applied';try{localStorage.setItem(key,JSON.stringify(draft));}catch(error){message.textContent='浏览器草稿未保存，请导出JSON：'+error.message;}}
function apply(){const indexed=new Map(draft.items.map(r=>[r.review_id,r]));for(const card of document.querySelectorAll('.card')){const row=indexed.get(card.dataset.review);for(const control of card.querySelectorAll('[data-field]'))control.value=row[control.dataset.field]??'';}document.getElementById('reviewer').value=draft.reviewer??'';filter();}
function filter(){const kind=document.getElementById('kind').value, query=document.getElementById('search').value.trim().toLowerCase(), pending=document.getElementById('pending').checked;let shown=0;for(const card of document.querySelectorAll('.card')){const show=(!kind||card.dataset.kind===kind)&&(!query||card.dataset.search.toLowerCase().includes(query))&&(!pending||card.querySelector('[data-field="decision"]').value==='pending');card.classList.toggle('hidden',!show);shown+=show;}message.textContent='显示 '+shown+' / '+original.items.length+' 项；尚未自动应用人工结果';}
for(const card of document.querySelectorAll('.card'))card.addEventListener('change',event=>{const field=event.target.dataset.field;if(!field)return;const row=draft.items.find(r=>r.review_id===card.dataset.review);row[field]=event.target.value||null;row.reviewer=document.getElementById('reviewer').value;row.reviewed_at=new Date().toISOString();persist();filter();});
for(const id of ['kind','search','pending'])document.getElementById(id).addEventListener('input',filter);
document.getElementById('reviewer').addEventListener('change',event=>{draft.reviewer=event.target.value;persist();});
document.getElementById('export').addEventListener('click',()=>{persist();const url=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='photo_partition_review_decisions.json';a.click();URL.revokeObjectURL(url);});
document.getElementById('import').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{const imported=validate(JSON.parse(await file.text()));const indexed=new Map(imported.items.map(r=>[r.review_id,r]));draft=structuredClone(original);draft.reviewer=imported.reviewer??'';for(const row of draft.items){const incoming=indexed.get(row.review_id);for(const field of ['decision','notes','reviewer','source_ref','scene_label','performance_id','burst_group_id','reviewed_at'])row[field]=incoming[field]??null;}persist();apply();}catch(error){message.textContent='恢复失败：'+error.message;}});
apply();
""" + '</script>'
    (output / 'index.html').write_text(page('剧照划分与来源复核', toolbar + ''.join(cards), script), encoding='utf-8')
    sections = ['<p>序列按原图EXIF DateTimeOriginal及亚秒排列。仅在同一目录/相机机身线索内相邻比较；缺机身序列号时相机型号不能证明同一机身，跨相机时间未校准。整理序号不作为时间顺序。</p>']
    for stream in temporal['streams']:
        figures = ''.join(figure(by_id[i]) for i in stream['image_ids_in_capture_hint_order'] + stream['missing_capture_image_ids'])
        sections.append('<details><summary>' + html.escape(stream['play_title'] + ' · ' + (stream['camera_model'] or '相机未知') \
            + ' · ' + str(len(stream['image_ids_in_capture_hint_order'])) + '张有时间 / ' + str(len(stream['missing_capture_image_ids'])) + '张缺时间') \
            + '</summary><p>' + html.escape(stream['source_directory']) + '</p><div class="photos">' + figures + '</div></details>')
    (output / 'timeline.html').write_text(page('相机拍摄时间线索', ''.join(sections)), encoding='utf-8')


def make_summary(rows, images, pairs, temporal, gallery, partitions, template, exposure):
    stage = [r for r in images if r['source_kind_hint'] == 'stage_photo']
    u4_refs = [ref for ref in exposure if '/u4_' in ref or Path(ref).parent.name.startswith('u4_')]
    u4_ids = set().union(*(exposure[ref] for ref in u4_refs)) if u4_refs else set()
    parent_ids = {r['parent_image_id_hint'] for r in gallery if r['parent_image_id_hint']}
    return {'schema_version': 1, 'status': 'offline_review_prepared_with_open_annotation_gaps',
        'file_count': len(rows), 'unique_decoded_images': len(images),
        'stage_unique_images_hint': len(stage), 'reference_unique_images_hint': len(images) - len(stage),
        'exact_copy_groups': sum(len(r['sources']) > 1 for r in images),
        'imported_labeled_images': sum(bool(r['imported_label_file_id']) for r in images),
        'scene_labeled_images': sum(bool(r['imported_scene']) for r in images),
        'stage_images_without_imported_labels': sum(not r['imported_label_file_id'] for r in stage),
        'imported_images_without_scene': sum(bool(r['imported_label_file_id']) and not r['imported_scene'] for r in images),
        'possible_unimported_scene_hints': sum(r['possible_unimported_scene_hint'] for r in images),
        'stage_images_with_DateTimeOriginal': sum(bool(r['exif']['capture_local']) for r in stage),
        'stage_images_missing_DateTimeOriginal': sum(not r['exif']['capture_local'] for r in stage),
        'filename_capture_date_mismatches': sum(r['date_hint_matches_capture'] is False for r in images),
        'known_exposed_unique_images_lower_bound': sum(bool(r['exposure_refs']) for r in images),
        'unlabeled_stage_known_exposed_images': sum(not r['imported_label_file_id'] and bool(r['exposure_refs']) for r in stage),
        'near_duplicate_review_pairs': len(pairs), 'legacy_near_review_pairs': sum(r['legacy_heuristic_group'] for r in pairs),
        'camera_stream_hints': len(temporal['streams']), 'time_neighbor_review_pairs': len(temporal['time_neighbors']),
        'boundary_review_pairs': len(temporal['boundary_hints']),
        'organized_order_inversion_hints': len(temporal['organized_order_inversions']),
        'filename_review_groups': len(temporal['filename_groups']),
        'filename_groups_with_multiple_imported_scenes': sum(r['multiple_imported_scenes'] for r in temporal['filename_groups']),
        'gallery_crops': len(gallery), 'gallery_unique_filename_parent_images': len(parent_ids),
        'gallery_crops_with_unique_filename_parent': sum(r['filename_source_match_count'] == 1 for r in gallery),
        'gallery_filename_parent_u4_overlap_crops': sum(r['parent_image_id_hint'] in u4_ids for r in gallery),
        'gallery_filename_parent_u4_overlap_images': len(parent_ids & u4_ids),
        'review_item_counts': dict(sorted(Counter(r['kind'] for r in template['items']).items())),
        'performance_review': partitions['performance_review'],
        'legacy_split_exposure': partitions['legacy_split_exposure'],
        'final_test_locked': False, 'new_final_test_images': 0, 'verified_performances': 0,
        'verified_near_duplicates': 0, 'verified_bursts': 0, 'verified_boundaries': 0,
        'verified_gallery_crop_origins': 0, 'verified_face_boxes': 0,
        'model_runs_started': 0, 'human_decisions_applied': False,
        'limitations': partitions['limitations']}


def render_report(summary, protocol_sha):
    s = summary
    lines = ['# 剧照划分、拍摄时间与图库来源复核准备', '',
        f'协议SHA256：`{protocol_sha}`。状态：`{s["status"]}`。', '',
        '本轮生成离线复核材料，全部新使用标记为development review。原图、图库、已导入标注、历史划分与冻结U4产物保持不变；final test未锁定。', '',
        f'实际清单{s["file_count"]}个文件 / {s["unique_decoded_images"]}张唯一解码图像，{s["exact_copy_groups"]}组原图/整理副本。目录规则提示舞台照{s["stage_unique_images_hint"]}、资料图{s["reference_unique_images_hint"]}，分类待核验。', '',
        f'已导入{s["imported_labeled_images"]}张，{s["scene_labeled_images"]}张有幕次，{s["imported_images_without_scene"]}张导入后无幕次，{s["possible_unimported_scene_hints"]}张仍有未采用的文件名幕次线索。另有{s["stage_images_without_imported_labels"]}张舞台照未导入，其中{s["unlabeled_stage_known_exposed_images"]}张已用于所提供的历史实验；不得当作未使用的final test。', '',
        '| 剧目 | 舞台图线索 | 资料图线索 | 幕次标注 | 未导入舞台照 | 日期线索 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for p in s['performance_review']:
        lines.append(f'| {p["play_title"]} | {p["stage_unique_images"]} | {p["reference_unique_images"]} | {p["scene_labeled_images"]} | {p["stage_images_without_imported_labels"]} | {", ".join(p["date_hints"])} |')
    lines += ['', '每剧目前只有一个日期线索，独立演出场次核验数仍为0。照片来源和EXIF日期能支持采集日期线索，无法证明正式/彩排边界、机身时钟一致或演出版逐幕等同。', '',
        f'舞台照{s["stage_images_with_DateTimeOriginal"]}张可读取原始拍摄时间、{s["stage_images_missing_DateTimeOriginal"]}张缺失；与文件名日期不一致{s["filename_capture_date_mismatches"]}张。DateTime编辑时间未用于代替拍摄时间。按目录及相机机身线索分为{s["camera_stream_hints"]}条待核验流，跨相机不合并时间顺序。', '',
        f'同流相邻≤2秒形成{s["time_neighbor_review_pairs"]}个时间核查对；≥60秒停顿或已导入幕次变化形成{s["boundary_review_pairs"]}个边界线索。整理序号在同流EXIF顺序中发现{s["organized_order_inversion_hints"]}处逆序线索，不证明真实回演。文件名分组{s["filename_review_groups"]}组，其中{s["filename_groups_with_multiple_imported_scenes"]}组跨已导入幕次，因此不能把同名组直接视为连拍。', '',
        f'近重复筛查{s["near_duplicate_review_pairs"]}对，包含旧启发式{s["legacy_near_review_pairs"]}对。同剧目/宽高比差≤0.04/平均RGB距离≤0.18，dHash≤8或63位去DC pHash≤8；均未人工核验，可能漏检或误报。旧dHash逐图复现，但不因旧分组或新阈值提升为GT。', '',
        f'图库{s["gallery_crops"]}张裁剪中，{s["gallery_crops_with_unique_filename_parent"]}张名称唯一对应{s["gallery_unique_filename_parent_images"]}张源图；{s["gallery_filename_parent_u4_overlap_crops"]}张对应{s["gallery_filename_parent_u4_overlap_images"]}张U4开发集query。对应关系仅为文件名来源线索，裁剪框/身份和是否同一连拍仍待人工验证。现有图库不能据此宣称与当前query隔离，亦不改变仅做图文检索的历史U4结果。', '',
        '| 历史集合 | 文件 | 唯一图像 | 已知实验使用图像（下界） |', '| --- | ---: | ---: | ---: |']
    for key, value in s['legacy_split_exposure'].items():
        lines.append(f'| {key} | {value["files"]} | {value["unique_images"]} | {value["known_exposed_unique_images"]} |')
    lines += ['', '所提供实验清单仅覆盖已知使用的下界；本轮复核也查看现有图片，所有新分配均为development review。保留历史final_test_provisional名称用于溯源，不沿用为未使用的final test。', '',
        '复核材料：', '',
        '- [可保存草稿/导出与恢复JSON的复核页面](index.html)；[按相机分开的拍摄时间序列](timeline.html)。',
        '- [唯一图片与副本/标注/使用记录](images.json)；[近重复对](near_duplicate_reviews.json)；[时间/边界/文件名线索](temporal_reviews.json)。',
        '- [图库来源线索](gallery_review.json)；[保守使用划分](partition_review.json)；[待人工核查模板](review_template.json)。',
        '- [输入协议](protocol.json)、[汇总](summary.json)、[完整性检查](integrity_checks.json)、[输出摘要清单](artifact_manifest.json)、[运行状态](run_state.json)。', '',
        '后续先核验场次/机身时钟/裁剪来源，补框级身份、未知人脸和连拍偏好。新增马可第3/4幕及尾声正例需另采集并确认；现有马可原图均有已导入对应图，资料图保持未标注。此准备包不修改任何人工决定或模型默认。', '']
    return '\n'.join(lines)


def verify_existing(output, inputs, rows, gallery_root, protocol, protocol_sha):
    if json.loads((output / 'protocol.json').read_text()) != protocol:
        raise ValueError('review_protocol_changed')
    for item in inputs:
        if fingerprint(item['path']) != item['sha256']:
            raise ValueError('review_input_changed')
    artifacts = json.loads((output / 'artifact_manifest.json').read_text())
    for relative, digest in artifacts['sha256'].items():
        if fingerprint(safe_source(output, relative)) != digest:
            raise ValueError('review_artifact_changed: ' + relative)
    for row in rows:
        if fingerprint(safe_source(Path(protocol['source_root']), row['source'])) != row['content_hash']:
            raise ValueError('review_source_changed')
    if not inventory_paths_match(Path(protocol['source_root']), rows):
        raise ValueError('review_source_inventory_members_changed')
    images = json.loads((output / 'images.json').read_text())
    for image in images:
        checked_checkpoint(output / 'image_checkpoints' / (image['image_id'] + '.json'),
                           image['image_id'], image['content_hash'], protocol_sha, output)
    gallery = json.loads((output / 'gallery_review.json').read_text())
    actual = {p.relative_to(gallery_root).as_posix(): fingerprint(p) for p in image_files(gallery_root)}
    if actual != {r['gallery_source']: r['content_hash'] for r in gallery}:
        raise ValueError('review_gallery_changed')
    checks = json.loads((output / 'integrity_checks.json').read_text())['checks']
    if not all(checks.values()):
        raise ValueError('saved_review_checks_failed')
    return {'status': 'read_only_review_verification_passed', 'checks': len(checks),
            'artifacts': len(artifacts['sha256']), 'source_files': len(rows), 'protocol_sha256': protocol_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/ncpa_test')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--legacy-splits', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, default=ROOT / 'data/face_database')
    parser.add_argument('--exposure-manifest', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    output, source, gallery_root = args.output.resolve(), args.root.resolve(), args.gallery.resolve()
    for protected in (source, gallery_root):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('review_output_must_be_outside_originals_and_gallery')
    if any(p.is_symlink() for p in [args.output, *args.output.parents]):
        raise ValueError('review_output_symlink')
    rows = load_inventory(args.inventory)
    labels = json.loads(args.labels.read_text())
    labels_by_image = bind_labels(rows, labels)
    registry = json.loads(args.registry.read_text())
    legacy_splits = json.loads(args.legacy_splits.read_text())
    if legacy_splits.get('manifest_sha256') != fingerprint(args.inventory) \
            or legacy_splits.get('assignments') != {r['file_id']: r['split'] for r in rows}:
        raise ValueError('legacy_split_inventory_mismatch')
    exposure = {str(path.resolve()): exposure_from_manifest(json.loads(path.read_text()), rows, str(path))
                for path in args.exposure_manifest}
    input_paths = [args.inventory, args.legacy_splits, args.labels, args.registry, *args.exposure_manifest]
    inputs = [{'path': str(path.resolve()), 'sha256': fingerprint(path)} for path in input_paths]
    protocol = {'schema_version': 1, 'policy': POLICY, 'source_root': str(source),
                'gallery_root': str(gallery_root), 'inputs': inputs,
                'gallery_sha256': {p.relative_to(gallery_root).as_posix(): fingerprint(p) for p in image_files(gallery_root)},
                'source_sha256': {name: fingerprint(ROOT / name) for name in SOURCE_FILES},
                'python_version': sys.version, 'pillow_version': PIL_VERSION, 'numpy_version': np.__version__}
    protocol_sha = digest_json(protocol)
    if args.verify_existing:
        print(json.dumps(verify_existing(output, inputs, rows, gallery_root, protocol, protocol_sha), ensure_ascii=False, indent=2))
        return
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('review_output_must_be_empty_or_resume')
    if args.resume and not (output / 'protocol.json').exists():
        raise ValueError('resume_protocol_missing')
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('resume_protocol_changed')
        if (output / 'summary.json').exists():
            raise ValueError('completed_review_use_verify_existing')
    else:
        atomic_json(output / 'protocol.json', protocol)
    state = {'schema_version': 1, 'status': 'validating_sources', 'protocol_sha256': protocol_sha,
             'final_test_locked': False, 'gpu_required': False, 'active_process': os.getpid(),
             'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'output_directory': str(output), 'completed_images': 0, 'resumed_images': 0}
    atomic_json(output / 'run_state.json', state)
    try:
        if not inventory_paths_match(source, rows):
            raise ValueError('inventory_source_members_changed')
        for row in rows:
            if fingerprint(safe_source(source, row['source'])) != row['content_hash']:
                raise ValueError('inventory_source_changed: ' + row['source'])
        canonical = {}
        for row in rows:
            canonical.setdefault(row['image_id'], []).append(row)
        checkpoints, resumed = {}, 0
        state['status'] = 'preparing_resumable_image_evidence'
        for index, (image_id, members) in enumerate(sorted(canonical.items())):
            row = canonical_row(members)
            value, reused = prepare_image(row, safe_source(source, row['source']), output, protocol_sha, args.resume)
            checkpoints[image_id] = value
            resumed += reused
            state.update(completed_images=index + 1, resumed_images=resumed)
            atomic_json(output / 'run_state.json', state)
            if (index + 1) % 25 == 0:
                print(json.dumps({'phase': state['status'], 'completed': index + 1, 'total': len(canonical), 'resumed': resumed}), flush=True)
        images = build_images(rows, labels_by_image, checkpoints, exposure)
        pairs = near_pairs(images)
        temporal = temporal_reviews(images)
        gallery = gallery_review(gallery_root, rows, registry, images, output)
        partitions = partition_review(rows, images, gallery, temporal, exposure)
        template = review_template(images, pairs, temporal, gallery, partitions)
        template['protocol_sha256'] = protocol_sha
        summary = make_summary(rows, images, pairs, temporal, gallery, partitions, template, exposure)
        outputs = {'images.json': images, 'near_duplicate_reviews.json': pairs,
                   'temporal_reviews.json': temporal, 'gallery_review.json': gallery,
                   'partition_review.json': partitions, 'review_template.json': template}
        for name, value in outputs.items():
            atomic_json(output / name, value)
        render_pages(output, images, gallery, temporal, template, summary, protocol_sha)
        (output / 'report.md').write_text(render_report(summary, protocol_sha), encoding='utf-8')
        actual_gallery = {p.relative_to(gallery_root).as_posix(): fingerprint(p) for p in image_files(gallery_root)}
        checks = {'every_inventory_file_present_unchanged': all(fingerprint(safe_source(source, r['source'])) == r['content_hash'] for r in rows),
            'inventory_members_unchanged': inventory_paths_match(source, rows),
            'every_input_unchanged': all(fingerprint(item['path']) == item['sha256'] for item in inputs),
            'measured_tool_sources_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in protocol['source_sha256'].items()),
            'every_unique_pixel_hash_recomputed': len(checkpoints) == len({r['image_id'] for r in rows}),
            'exact_copies_share_one_image_record': sorted(i for r in images for i in r['source_file_ids']) == sorted(r['file_id'] for r in rows),
            'old_dhash_reproduced': all(r['dhash'] == checkpoints[r['image_id']]['features']['dhash'] for r in rows),
            'labels_not_promoted_from_hints': all(r['imported_scene'] == (labels_by_image[r['image_id']]['scene_label'] if r['image_id'] in labels_by_image else None) for r in images),
            'all_file_assignments_remain_development_review': set(partitions['new_assignments'].values()) == {'development_review'},
            'no_final_test_eligible_images': not any(r['final_test_eligible'] for r in images),
            'no_performance_or_burst_verified': not any(r['performance_verified'] or r['chronology_verified'] or r['burst_group_id'] for r in images),
            'time_neighbors_same_camera_stream_hint': all(len({next(r['camera_stream_hint_id'] for r in images if r['image_id'] == i) for i in pair['image_ids']}) == 1 for pair in temporal['time_neighbors']),
            'all_near_pairs_pending': all(not r['verified'] for r in pairs),
            'gallery_sources_unchanged': actual_gallery == {r['gallery_source']: r['content_hash'] for r in gallery},
            'gallery_bound_protocol_unchanged': actual_gallery == protocol['gallery_sha256'],
            'gallery_crop_origin_and_box_unverified': not any(r['provenance_verified'] or r['crop_box_verified'] for r in gallery),
            'all_review_ids_unique': len({r['review_id'] for r in template['items']}) == len(template['items']),
            'all_reviews_pending': all(r['decision'] == 'pending' for r in template['items']),
            'no_human_decisions_applied': not summary['human_decisions_applied'],
            'final_test_not_locked': not summary['final_test_locked'] and summary['new_final_test_images'] == 0}
        atomic_json(output / 'integrity_checks.json', {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks})
        if not all(checks.values()):
            raise ValueError('partition_review_integrity_failed')
        summary['protocol_sha256'] = protocol_sha
        summary['integrity_checks_passed'] = len(checks)
        atomic_json(output / 'summary.json', summary)
        artifacts = {p.relative_to(output).as_posix(): fingerprint(p) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name not in ('artifact_manifest.json', 'run_state.json')}
        atomic_json(output / 'artifact_manifest.json', {'protocol_sha256': protocol_sha, 'sha256': artifacts})
        state.update(status=summary['status'], completed_images=len(images), resumed_images=resumed,
                     active_process=None, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     integrity_checks_passed=len(checks), human_decisions_applied=False,
                     next_actions=['Human review of performance, camera clocks, near pairs, burst/boundary and gallery crop origins.',
                                   'Supply verified face boxes and unknown queries for identity calibration.',
                                   'Acquire independent photos and additional Marco act3/4/epilogue positives.'])
        atomic_json(output / 'run_state.json', state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as exc:
        state.update(status='interrupted_or_failed_keep_checkpoints', active_process=None,
                     interrupted_at_utc=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__, error=str(exc),
                     next_action='Correct external condition then use identical inputs/source and --resume; preserve failed state.')
        atomic_json(output / 'run_state.json', state)
        raise


if __name__ == '__main__':
    main()
