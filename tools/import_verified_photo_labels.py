"""Import user-confirmed labels from organized filenames, without editing photos.

Scope is image-level scene and listed actor/role labels. Lists are not face-box
ground truth, and sequence numbers are not proof of shooting chronology.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.filename_parser import parse_filename
from core_modules.organize.identity_registry import IdentityRegistry, legacy_actor, stable_id
from core_modules.organize.task_artifacts import atomic_json, fingerprint


def confirmed_scene_from_filename(path, parsed):
    """Read a distinct scene segment only from the confirmed organized name.

    Some curated names put the scene after a hyphen rather than after the
    event's space. People, photographer and original-file hints are excluded.
    Multiple explicit labels are rejected instead of silently selecting one.
    """
    explicit = [parsed['scene']] if parsed['scene'] else []
    segments = Path(path).stem.split('-')
    after_play = False
    for segment in segments:
        segment = segment.strip()
        if '《' in segment and '》' in segment:
            after_play = True
            continue
        if not after_play:
            continue
        if re.fullmatch(r'(?:第\d+[幕场](?:\d+场)?[\u4e00-\u9fff：:·\s]{0,24}|序[曲幕]?|尾声|谢幕)(?:[：:][\u4e00-\u9fff·\s]{1,24})?', segment) \
                and '饰' not in segment and '摄影' not in segment:
            explicit.append(segment)
    unique = set(explicit)
    if len(unique) > 1:
        raise ValueError('conflicting_explicit_scene_segments: ' + str(path))
    return explicit[0] if explicit else None


def import_labels(root, manifest_file, gallery_root, confirmation_source, role_aliases=None):
    root = Path(root).resolve()
    inventory = [json.loads(line) for line in Path(manifest_file).read_text().splitlines() if line.strip()]
    indexed = {row['source']: row for row in inventory}
    role_aliases = role_aliases or []
    aliases = {(row['play'], row['actor'], row['alias']): row for row in role_aliases}
    if len(aliases) != len(role_aliases) or any(not row.get('source_ref') for row in role_aliases):
        raise ValueError('invalid_confirmed_role_aliases')
    rows, people, plays = [], {}, {}
    for directory in sorted(root.glob('*/【整理完成】*')):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'):
                continue
            relative = path.relative_to(root).as_posix()
            source = indexed.get(relative)
            if source is None or fingerprint(path) != source['content_hash']:
                raise ValueError('missing_or_changed_inventory_source: ' + relative)
            parsed = parse_filename(path.name, done_style=True)
            play = parsed['play']
            if not play:
                raise ValueError('missing_play_label: ' + relative)
            play_id = stable_id('play', play)
            plays[play_id] = {'play_id': play_id, 'title': play, 'aliases': [directory.parent.name]}
            actor_roles = []
            for actor, role in parsed['people']:
                original_role = role
                alias = aliases.get((play, actor, role))
                if alias:
                    role = alias['canonical']
                # Name alone never proves that same-named actors in different plays
                # are the same person. Gallery bindings require an unambiguous name.
                identity_id = stable_id('actor', play_id + ':' + actor)
                role_id = stable_id('role', play_id + ':' + role)
                identity = people.setdefault(identity_id, {'identity_id': identity_id, 'actor_name': actor,
                    'verified': True, 'gallery_dirs': [], 'roles': [], 'identity_scope': play_id})
                role_row = {'play_id': play_id, 'role_id': role_id, 'role_name': role,
                            'verified': True, 'source_ref': confirmation_source + ':' + relative}
                if alias:
                    role_row['alias_source'] = alias['source_ref']
                if not any(r['role_id'] == role_id for r in identity['roles']):
                    identity['roles'].append(role_row)
                actor_roles.append({'identity_id': identity_id, 'actor_name': actor,
                                    'role_id': role_id, 'role_name': role, 'original_role_name': original_role,
                                    'alias_source': alias['source_ref'] if alias else None})
            scene = confirmed_scene_from_filename(path.name, parsed)
            rows.append({'file_id': source['file_id'], 'image_id': source['image_id'],
                'source': relative, 'content_hash': source['content_hash'],
                'duplicate_group_id': source['duplicate_group_id'], 'play_id': play_id,
                'performance_hint': parsed['date'], 'scene_label': scene,
                'scene_id': stable_id('scene', play_id + ':' + scene) if scene else None,
                'actor_roles': actor_roles, 'sequence_hint': parsed['seq'],
                'annotation_source': confirmation_source, 'verified_fields': ['scene_label', 'actor_roles'],
                'people_list_exhaustive': False, 'face_boxes': None, 'burst_group_id': None,
                'chronology_verified': False,
                'status': 'ok' if source['status'] == 'ok' else 'decode_failed'})
    if not rows:
        raise ValueError('no_organized_labeled_photos')
    by_name = defaultdict(list)
    for identity in people.values():
        by_name[identity['actor_name']].append(identity)
    unresolved_gallery = []
    for directory in sorted(Path(gallery_root).iterdir()):
        if not directory.is_dir():
            continue
        matches = by_name.get(legacy_actor(directory.name), [])
        if len(matches) == 1:
            matches[0]['gallery_dirs'].append(directory.name)
        else:
            unresolved_gallery.append({'directory': directory.name, 'reason': 'missing_or_ambiguous_actor'})
    registry = {'schema_version': 1, 'plays': list(plays.values()), 'identities': list(people.values()),
                'source': confirmation_source, 'identity_policy': 'play_scoped_names_no_cross_play_name_merge',
                'confirmed_role_aliases': role_aliases}
    IdentityRegistry(registry)
    conflicts = [{'identity_id': p['identity_id'], 'actor_name': p['actor_name'],
                  'role_names': [r['role_name'] for r in p['roles']],
                  'handling': 'retain_all_labels_no_role_inference_until_disambiguated'}
                 for p in people.values() if len(p['roles']) > 1]
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    contradictions = [{'image_id': image_id, 'sources': [r['source'] for r in group],
                       'scene_labels': sorted({r['scene_label'] or '' for r in group})}
                      for image_id, group in groups.items()
                      if len({r['scene_label'] for r in group}) > 1]
    summary = {'status': 'labels_imported_final_test_not_locked', 'file_count': len(rows),
        'import_policy': 'confirmed_organized_inline_or_separate_scene_segments_v2',
        'unique_decoded_images': len(groups), 'play_counts': dict(Counter(r['play_id'] for r in rows)),
        'scene_labeled_files': sum(r['scene_id'] is not None for r in rows),
        'files_with_listed_people': sum(bool(r['actor_roles']) for r in rows),
        'actor_role_mentions': sum(len(r['actor_roles']) for r in rows),
        'play_scoped_identities': len(people), 'gallery_dirs_bound': sum(len(p['gallery_dirs']) for p in people.values()),
        'unresolved_gallery_dirs': unresolved_gallery, 'role_label_conflicts': conflicts,
        'duplicate_scene_conflicts': contradictions,
        'verified_face_boxes': 0, 'verified_unknown_queries': 0, 'verified_burst_preference_groups': 0,
        'verified_chronological_boundaries': 0, 'final_test_locked': False,
        'inventory_sha256': fingerprint(manifest_file), 'confirmation_source': confirmation_source,
        'confirmed_role_aliases': role_aliases,
        'limitations': ['User confirmed scene and listed actor/role labels in organized filenames.',
            'Listed people need not exhaust all visible faces; missing names are not unknown-identity GT.',
            'No face-box correspondence, burst preferences or verified shooting chronology supplied.',
            'Role spelling differences and multiple roles remain explicit; no silent alias correction.',
            'Existing whole-play/duplicate grouping retained; held-out split still requires review.']}
    return rows, registry, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/ncpa_test')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, default=ROOT / 'data/face_database')
    parser.add_argument('--confirmation-source', required=True)
    parser.add_argument('--role-aliases', type=Path, help='Explicit user-confirmed play/actor role aliases JSON list')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.root.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('output_must_be_outside_source')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output_must_be_empty')
    aliases = json.loads(args.role_aliases.read_text()) if args.role_aliases else None
    rows, registry, summary = import_labels(source, args.inventory, args.gallery, args.confirmation_source, aliases)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'verified_photo_labels.json', rows)
    atomic_json(output / 'identity_registry.json', registry)
    summary['labels_sha256'] = fingerprint(output / 'verified_photo_labels.json')
    summary['registry_sha256'] = fingerprint(output / 'identity_registry.json')
    atomic_json(output / 'annotation_status.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
