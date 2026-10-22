"""Freeze a text-block protocol from already verified sourced scene inputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.identity_registry import IdentityRegistry
from core_modules.organize.play_knowledge import parse_plot_text
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from tools.scene_text_blocks import BLOCK_VERSION, POLICY, build_scene_blocks
from tools.sourced_scene_candidates import validate_sourced_manifest


MEASURED_FILES = (
    'tools/scene_text_blocks.py', 'tools/prepare_scene_text_block_study.py',
    'tools/evaluate_scene_text_block_study.py', 'tools/sourced_scene_candidates.py',
    'core_modules/organize/identity_registry.py', 'core_modules/organize/play_knowledge.py',
    'core_modules/organize/task_artifacts.py', 'core_modules/image_search.py',
    'core_modules/organize/embedding_gemma.py', 'core_modules/organize/feature_cache.py', 'config/base.py',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_plot_bindings(scenes, titles, source_root):
    bindings = {}
    for scene in scenes:
        if scene['text_source'] != 'local_plot_and_scene_label':
            continue
        relative, anchor = scene['source_ref'].rsplit('#', 1)
        path = (source_root / relative).resolve()
        if source_root not in path.parents:
            raise ValueError('local_plot_path_outside_source_root')
        source_ref = str(path.relative_to(source_root))
        matches = [row for row in parse_plot_text(path.read_text(encoding='utf-8'), source_ref)[1]
                   if row['source_ref'] == source_ref + '#' + anchor]
        if len(matches) != 1 or titles[scene['play_id']] + ' ' + scene['label'] + ' ' + matches[0]['text'] != scene['text']:
            raise ValueError('local_plot_document_differs_from_frozen_manifest')
        bindings[str(path)] = fingerprint(path)
    return bindings


def tokenizer_for(path):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = processor.tokenizer
    count = lambda text: len(tokenizer(text, truncation=False)['input_ids'])
    return processor, count


def rebuild_blocks(protocol, registry):
    manifest = read(protocol['inputs']['expanded']['path'])
    validate_sourced_manifest(manifest, registry)
    processor, count = tokenizer_for(protocol['tokenizer']['model_path'])
    blocks = build_scene_blocks(manifest['scenes'], registry.plays, count, budget=64)
    for start in range(0, len(blocks['blocks']), 8):
        group = blocks['blocks'][start:start + 8]
        actual = processor(text=[row['text'] for row in group], padding='max_length', truncation=True, return_tensors='pt')
        if actual['input_ids'].shape[1] != 64:
            raise ValueError('unexpected_siglip_processor_window')
        if any(count(row['text']) > 64 for row in group):
            raise ValueError('short_block_would_truncate')
    return blocks


def protocol_files_unchanged(protocol):
    files = {**protocol['measured_sources_sha256'], **protocol['local_plot_files_sha256'],
             **protocol['tokenizer']['files_sha256']}
    for row in protocol['inputs'].values():
        files[row['path']] = row['sha256']
    files[protocol['labels']['path']] = protocol['labels']['sha256']
    files[protocol['registry']['path']] = protocol['registry']['sha256']
    files[protocol['blocks']['path']] = protocol['blocks']['sha256']
    return all(Path(path).is_file() and fingerprint(path) == value for path, value in files.items())


def validate_protocol(protocol_path):
    protocol = read(protocol_path)
    if protocol['schema_version'] != 1 or protocol['policy'] != POLICY or protocol['test_locked'] is not False:
        raise ValueError('block_study_protocol_changed_or_invalid')
    if not protocol_files_unchanged(protocol):
        raise ValueError('block_study_bound_files_changed')
    registry = IdentityRegistry.load(protocol['registry']['path'])
    same = read(protocol['inputs']['samepool']['path'])
    expanded = read(protocol['inputs']['expanded']['path'])
    validate_sourced_manifest(same, registry)
    validate_sourced_manifest(expanded, registry)
    if same['rows'] != expanded['rows'] or same['excluded'] != expanded['excluded'] or expanded['scenes'][:25] != same['scenes']:
        raise ValueError('block_study_pool_inputs_incompatible')
    blocks = read(protocol['blocks']['path'])
    if blocks != rebuild_blocks(protocol, registry):
        raise ValueError('block_study_blocks_do_not_rebuild')
    if local_plot_bindings(expanded['scenes'], registry.plays, Path(protocol['source_root'])) != protocol['local_plot_files_sha256']:
        raise ValueError('block_study_local_plot_changed')
    return protocol, same, expanded, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samepool', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--siglip-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('block_protocol_output_must_be_empty')
    source = args.source_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('block_protocol_output_must_be_outside_originals')
    inputs = {name: {'path': str(path.resolve()), 'sha256': fingerprint(path)}
              for name, path in (('samepool', args.samepool), ('expanded', args.expanded))}
    same, expanded = read(args.samepool), read(args.expanded)
    registry = IdentityRegistry.load(args.registry)
    validate_sourced_manifest(same, registry)
    validate_sourced_manifest(expanded, registry)
    if same['rows'] != expanded['rows'] or same['excluded'] != expanded['excluded'] or expanded['scenes'][:25] != same['scenes']:
        raise ValueError('block_study_pool_inputs_incompatible')
    if len(same['scenes']) != 25 or len(expanded['scenes']) != 28 or len(same['rows']) != 260:
        raise ValueError('unexpected_block_study_sizes')
    if same['labels_sha256'] != fingerprint(args.labels) or same['registry_sha256'] != fingerprint(args.registry):
        raise ValueError('block_study_annotations_changed')
    tokenizer_path = args.siglip_model.resolve()
    local_files = local_plot_bindings(expanded['scenes'], registry.plays, source)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'schema_version': 1, 'version': BLOCK_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(), 'test_locked': False,
        'policy': POLICY, 'source_root': str(source), 'inputs': inputs,
        'labels': {'path': str(args.labels.resolve()), 'sha256': fingerprint(args.labels)},
        'registry': {'path': str(args.registry.resolve()), 'sha256': fingerprint(args.registry)},
        'local_plot_files_sha256': local_files,
        'tokenizer': {'model_path': str(tokenizer_path),
                      'files_sha256': {str(path): fingerprint(path) for path in sorted(tokenizer_path.glob('*.json'))}},
        'measured_sources_sha256': {str(ROOT / name): fingerprint(ROOT / name) for name in MEASURED_FILES},
        'variants': [
            {'name': 'long_samepool25', 'representation': 'long_v1', 'aggregation': 'single', 'candidate_count': 25},
            {'name': 'long_expanded28', 'representation': 'long_v1', 'aggregation': 'single', 'candidate_count': 28},
            {'name': 'blocks_mean_samepool25', 'representation': 'blocks', 'aggregation': 'mean', 'candidate_count': 25},
            {'name': 'blocks_mean_expanded28', 'representation': 'blocks', 'aggregation': 'mean', 'candidate_count': 28},
            {'name': 'blocks_max_samepool25', 'representation': 'blocks', 'aggregation': 'max', 'candidate_count': 25},
            {'name': 'blocks_max_expanded28', 'representation': 'blocks', 'aggregation': 'max', 'candidate_count': 28},
        ],
        'notes': ['Same v1 source evidence and vocabulary; no new source paragraphs enter this text representation comparison.',
                  'All source quotations are covered. Existing five local paragraphs are bound and split as frozen prepared text.',
                  'Mean is primary, max is a predeclared secondary with block-count bias; no best-variant default selection.',
                  'Independent performance and final-test split not verified; U2 remains paused.'],
    }
    blocks = rebuild_blocks(protocol, registry)
    blocks_path = output / 'blocks.json'
    atomic_json(blocks_path, blocks)
    protocol['blocks'] = {'path': str(blocks_path), 'sha256': fingerprint(blocks_path)}
    protocol['counts'] = {'queries': len(same['rows']), 'candidates': len(expanded['scenes']),
                          'blocks': len(blocks['blocks']),
                          'blocks_by_play': dict(Counter(row['play_id'] for row in blocks['blocks'])),
                          'blocks_by_scene': {row['scene_id']: len(row['block_ids']) for row in blocks['candidates']},
                          'max_actual_tokens': max(row['token_count'] for row in blocks['blocks']),
                          'original_truth_scene_ids': [row['scene_id'] for row in same['scenes']],
                          'added_scene_ids': [row['scene_id'] for row in expanded['scenes'][25:]]}
    atomic_json(output / 'protocol.json', protocol)
    validate_protocol(output / 'protocol.json')
    print(json.dumps({'status': 'protocol_frozen', 'protocol': str(output / 'protocol.json'),
                      'sha256': fingerprint(output / 'protocol.json'), 'counts': protocol['counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
