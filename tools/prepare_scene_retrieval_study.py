"""Prepare a dynamic query study, preserving historical vectors when supplied."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.identity_registry import IdentityRegistry
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from tools.prepare_scene_text_block_study import local_plot_bindings, tokenizer_for
from tools.scene_text_blocks import POLICY, build_scene_blocks
from tools.scene_retrieval_study import binding, read, validate_inputs, validate_protocol

MEASURED_FILES = (
    'tools/scene_retrieval_study.py', 'tools/prepare_scene_retrieval_study.py', 'tools/evaluate_scene_retrieval_study.py',
    'tools/scene_text_blocks.py', 'tools/sourced_scene_candidates.py', 'tools/prepare_scene_text_block_study.py',
    'core_modules/organize/identity_registry.py', 'core_modules/organize/play_knowledge.py',
    'core_modules/organize/task_artifacts.py', 'core_modules/image_search.py',
    'core_modules/organize/embedding_gemma.py', 'core_modules/organize/feature_cache.py', 'config/base.py',
)


def prepare(args):
    output, source = args.output.resolve(), args.source_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('study_protocol_output_must_be_empty')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('study_output_overlaps_originals')
    inputs = {key: binding(getattr(args,key)) for key in ('baseline','samepool','expanded','historical')}
    manifests = {key: read(row['path']) for key,row in inputs.items()}
    registry = IdentityRegistry.load(args.registry)
    subsets = validate_inputs(manifests['baseline'],manifests['samepool'],manifests['expanded'],manifests['historical'],read(args.labels),registry)
    for manifest in (manifests['baseline'],manifests['samepool'],manifests['expanded']):
        if manifest['labels_sha256'] != fingerprint(args.labels) or manifest['registry_sha256'] != fingerprint(args.registry):
            raise ValueError('study_annotations_changed')
    _, count = tokenizer_for(args.siglip_model)
    blocks = build_scene_blocks(manifests['expanded']['scenes'], registry.plays, count)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'blocks.json', blocks)
    bindings = {str(ROOT/name): fingerprint(ROOT/name) for name in MEASURED_FILES}
    bindings.update(local_plot_bindings(manifests['expanded']['scenes'],registry.plays,source))
    for path in sorted(args.siglip_model.resolve().glob('*.json')):
        bindings[str(path)] = fingerprint(path)
    for row in inputs.values():
        bindings[row['path']] = row['sha256']
    for path in (args.labels,args.registry,output/'blocks.json'):
        row = binding(path)
        bindings[row['path']] = row['sha256']
    for row in manifests['expanded']['rows']:
        bindings[str((source / row['source']).resolve())] = row['content_hash']
        bindings[row['neutral_path']] = row['neutral_sha256']
    reused = {}
    for backend, folder in (('siglip2_base',args.reuse_siglip2),('embeddinggemma2',args.reuse_gemma)):
        if folder is None:
            continue
        folder = folder.resolve()
        summary = read(folder/'summary.json')
        manifest = read(folder/'input_manifest.json')
        if summary['backend'] != backend or not all(summary['checks'].values()):
            raise ValueError('reuse_summary_incomplete_or_wrong_backend')
        if [row['file_id'] for row in manifest['rows']] != subsets['historical'] or any(
            a != b for a,b in zip(manifest['scenes'],manifests['expanded']['scenes'])) or len(manifest['scenes']) != len(manifests['expanded']['scenes']):
            raise ValueError('reuse_input_not_same_historical_study')
        for name, sha in summary['artifact_sha256'].items():
            path = folder / name
            if fingerprint(path) != sha:
                raise ValueError('reuse_artifact_changed')
            bindings[str(path)] = sha
        bindings[str(folder/'summary.json')] = fingerprint(folder/'summary.json')
        reused[backend] = {'directory':str(folder), 'model_signature':summary['model_signature'],
            'preprocessing_signature':summary['preprocessing_signature'], 'historical_query_ids':subsets['historical']}
    same_count, expanded_count = len(manifests['samepool']['scenes']),len(manifests['expanded']['scenes'])
    variants = [{'name':'baseline_samepool','representation':'baseline','aggregation':'single','candidate_count':same_count}]
    variants += [{'name':name+'_'+pool,'representation':name,'aggregation':aggregation,'candidate_count':size}
        for name,aggregation in (('long','single'),('blocks_mean','mean'),('blocks_max','max'))
        for pool,size in (('samepool',same_count),('expanded',expanded_count))]
    protocol = {'schema_version':2,'version':'scene_retrieval_study/v2', 'created_at':datetime.now(timezone.utc).isoformat(),
        'test_locked':False,'query_assignment':'development','source_root':str(source), 'block_policy':POLICY,
        'inputs':inputs,'labels':binding(args.labels),'registry':binding(args.registry),'blocks':binding(output/'blocks.json'),
        'tokenizer':{'model_path':str(args.siglip_model.resolve())},'query_subsets':subsets,
        'reused_vectors':reused,'variants':variants,'bound_files_sha256':bindings,
        'counts':{'queries':len(subsets['all']),'historical_queries':len(subsets['historical']),'added_queries':len(subsets['added']),
                  'samepool_candidates':same_count,'expanded_candidates':expanded_count,'blocks':len(blocks['blocks'])},
        'notes':['Historical image/document vectors and scores are retained exactly when reuse is enabled.',
                 'Only missing baseline label documents and added queries require encoding in this version.',
                 'All, historical and added subsets are reported separately; added queries are reviewed development data.',
                 'Mean is primary; max is secondary. No candidate text selection or default adoption follows observed scores.',
                 'Timing excludes inherited work and is not a speed benchmark; U2 remains paused.']}
    atomic_json(output/'protocol.json',protocol)
    validate_protocol(output/'protocol.json',rebuild_blocks=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','samepool','expanded','historical','labels','registry','source-root','siglip-model','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reuse-siglip2',type=Path)
    parser.add_argument('--reuse-gemma',type=Path)
    args = parser.parse_args()
    protocol = prepare(args)
    import json
    print(json.dumps({'status':'protocol_frozen','protocol_sha256':fingerprint(args.output/'protocol.json'),'counts':protocol['counts']},indent=2))


if __name__ == '__main__':
    main()
