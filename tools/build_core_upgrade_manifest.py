"""Inventory local photos without modifying originals or inventing verified GT.

Splits are provisional and conservative: a whole play and every detected
duplicate group stay together. Human verification is still required to freeze
a final test, performance IDs, burst groups, identities and scene boundaries.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.scene_reasoner import _parse_local
from tools.evaluate_core_upgrade import write_json


class Groups:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, value):
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def join(self, first, second):
        first, second = self.find(first), self.find(second)
        self.parent[max(first, second)] = min(first, second)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root, near_threshold=4):
    root = Path(root)
    rows, visual = [], []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions):
        relative = path.relative_to(root)
        play = relative.parts[0] if len(relative.parts) > 1 else 'unknown'
        parsed = _parse_local(str(path))
        row = {'file_id': hashlib.sha256(relative.as_posix().encode()).hexdigest()[:24],
               'source': relative.as_posix(), 'play_id': play,
               'performance_id': None, 'performance_hint': parsed.get('date'),
               'source_kind': 'reference_material' if any(
                   marker in relative.as_posix() for marker in ('剧组信息', '剧目信息')) else 'stage_photo',
               'source_kind_verified': False, 'content_hash': fingerprint(path),
               'provisional_scene': parsed.get('scene'),
               'ground_truth_source': 'filename_unverified', 'verified_by': None,
               'burst_group_id': None, 'status': 'ok'}
        feature = None
        try:
            with Image.open(path) as original:
                row['exif_orientation'] = original.getexif().get(274, 1)
                image = ImageOps.exif_transpose(original).convert('RGB')
                row['width'], row['height'] = image.size
                pixels = hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode())
                pixels.update(image.tobytes())
                row['image_id'] = pixels.hexdigest()
                thumb = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
                bits = (thumb[:, 1:] > thumb[:, :-1]).reshape(-1)
                dhash = sum(int(bit) << index for index, bit in enumerate(bits))
                color = np.asarray(image.resize((8, 8)), dtype='float32').mean(axis=(0, 1)) / 255.
                feature = (dhash, color, image.width / image.height)
                row['dhash'] = f'{dhash:016x}'
        except (OSError, ValueError) as exc:
            row.update(status='decode_failed', error_type=type(exc).__name__, image_id=row['content_hash'])
        rows.append(row)
        visual.append(feature)
    if not rows:
        raise ValueError('图片清单为空')

    duplicate = Groups(len(rows))
    first_by_id = {}
    for index, row in enumerate(rows):
        if row['image_id'] in first_by_id:
            duplicate.join(index, first_by_id[row['image_id']])
        else:
            first_by_id[row['image_id']] = index
    for index, first in enumerate(visual):
        if first is None:
            continue
        for other in range(index):
            second = visual[other]
            if second is not None and abs(first[2] - second[2]) < .03 \
                    and (first[0] ^ second[0]).bit_count() <= near_threshold \
                    and float(np.linalg.norm(first[1] - second[1])) <= .08:
                duplicate.join(index, other)

    leakage = Groups(len(rows))
    first_by_play = {}
    for index, row in enumerate(rows):
        row['duplicate_group_id'] = 'duplicate-' + rows[duplicate.find(index)]['file_id']
        leakage.join(index, duplicate.find(index))
        if row['play_id'] in first_by_play:
            leakage.join(index, first_by_play[row['play_id']])
        else:
            first_by_play[row['play_id']] = index
    group_ids = sorted({leakage.find(i) for i in range(len(rows))},
                       key=lambda i: hashlib.sha256(rows[i]['play_id'].encode()).hexdigest())
    split_by_group = {value: 'development' for value in group_ids}
    if len(group_ids) >= 3:
        split_by_group[group_ids[-1]] = 'final_test_provisional'
        split_by_group[group_ids[-2]] = 'validation'
    for index, row in enumerate(rows):
        group = leakage.find(index)
        row['leakage_group_id'] = 'leakage-' + rows[group]['file_id']
        row['split'] = split_by_group[group]
    by_duplicate = defaultdict(set)
    for row in rows:
        by_duplicate[row['duplicate_group_id']].add(row['split'])
    assert all(len(parts) == 1 for parts in by_duplicate.values())
    summary = {'status': 'insufficient_data', 'file_count': len(rows),
               'unique_file_hashes': len({r['content_hash'] for r in rows}),
               'unique_decoded_images': len({r['image_id'] for r in rows if r['status'] == 'ok'}),
               'heuristic_duplicate_groups': len(by_duplicate),
               'play_file_counts': dict(Counter(r['play_id'] for r in rows)),
               'stage_unique_images': len({r['image_id'] for r in rows
                                          if r['status'] == 'ok' and r['source_kind'] == 'stage_photo'}),
               'reference_unique_images': len({r['image_id'] for r in rows
                                              if r['status'] == 'ok' and r['source_kind'] == 'reference_material'}),
               'decode_failures': sum(r['status'] != 'ok' for r in rows),
               'verified_scene_labels': 0, 'verified_face_boxes': 0,
               'verified_bursts': 0, 'verified_boundaries': 0,
               'provisional_scene_files': sum(bool(r['provisional_scene']) for r in rows),
               'detected_duplicate_groups_crossing_splits': 0,
               'split_file_counts': dict(Counter(r['split'] for r in rows)),
               'split_policy': 'whole_play_plus_detected_duplicates_v1', 'final_test_locked': False,
               'duplicate_method': {'exact': 'EXIF-oriented RGB pixels and dimensions',
                                    'near': '64-bit dHash with aspect and mean-color checks; unverified',
                                    'hamming_threshold': near_threshold},
               'limitations': ['Weak scene labels and material/performance hints need human review.',
                               'Undetected near duplicates are possible; review groups before locking splits.',
                               'Filename actor lists are not face-box ground truth.',
                               'No independent annotations or endpoint performance acceptance yet.']}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT / 'data/ncpa_test'))
    parser.add_argument('--output', default='outputs/acceptance/core_upgrade/first_pass')
    args = parser.parse_args()
    rows, summary = inventory(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.jsonl'
    # JSONL is staged before replacing to avoid leaving a partial inventory.
    temporary = output / 'manifest.pending.jsonl'
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(manifest)
    summary['manifest_sha256'] = fingerprint(manifest)
    write_json(output / 'annotation_status.json', summary)
    write_json(output / 'splits.json', {
        'status': 'provisional', 'final_test_locked': False,
        'manifest_sha256': summary['manifest_sha256'], 'policy': summary['split_policy'],
        'assignments': {row['file_id']: row['split'] for row in rows}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
