"""Evaluate versioned retrieval inputs with immutable historical vector reuse."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from tools.scene_retrieval_study import read, subset_metrics, validate_protocol
from tools.scene_text_blocks import normalize_vectors, ranked_rows, score_candidates, text_hash


def check_vectors(values, count, dimension=None):
    values = np.asarray(values)
    if values.ndim != 2 or len(values) != count or not np.isfinite(values).all() or values.dtype != np.float32:
        raise ValueError('invalid_saved_vectors')
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError('embedding_dimension_changed')
    if not np.allclose(np.linalg.norm(values,axis=1),1.0,atol=2e-5):
        raise ValueError('saved_vectors_not_normalized')
    return values


def document_plan(inputs, blocks, inherited=None):
    documents = list(inherited or [])
    index = {row['text']:i for i,row in enumerate(documents)}
    if len(index) != len(documents) or any(text_hash(row['text']) != row['text_sha256'] for row in documents):
        raise ValueError('inherited_document_index_invalid')
    def add(text):
        if text not in index:
            index[text] = len(documents)
            documents.append({'text':text,'text_sha256':text_hash(text)})
        return index[text]
    long = [[add(c['text'])] for c in inputs['expanded']['scenes']]
    block_by_id = {row['block_id']:row for row in blocks['blocks']}
    grouped = [[add(block_by_id[b]['text']) for b in c['block_ids']] for c in blocks['candidates']]
    baseline = [[add(c['text'])] for c in inputs['baseline']['scenes']]
    return documents, {'long_indices':long,'block_indices':grouped,'baseline_indices':baseline}


def evaluate(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('retrieval_study_output_must_be_empty')
    protocol_sha = fingerprint(args.protocol)
    protocol, inputs, blocks = validate_protocol(args.protocol,rebuild_blocks=True)
    source = Path(protocol['source_root'])
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('retrieval_study_output_overlaps_originals')
    output.mkdir(parents=True,exist_ok=True)
    state = {'status':'preparing_vectors','backend':args.backend,'protocol_sha256':protocol_sha,
        'started_at':datetime.now(timezone.utc).isoformat(),'document_encoding':{},'image_encoding':{}}
    def save_state():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(output/'run_state.json',state)
    save_state()
    manifest = inputs['expanded']
    atomic_json(output/'input_manifest.json',manifest)
    reuse = protocol['reused_vectors'].get(args.backend)
    inherited_index, old_images, old_documents, old_scores, old_summary = None, None, None, None, None
    if reuse:
        folder = Path(reuse['directory'])
        old_summary = read(folder/'summary.json')
        inherited_index = read(folder/'document_index.json')
        old_documents = check_vectors(np.load(folder/'document_vectors.npy',allow_pickle=False),len(inherited_index['documents']))
        old_images = check_vectors(np.load(folder/'image_vectors.npy',allow_pickle=False),len(protocol['query_subsets']['historical']),old_documents.shape[1])
        old_scores = np.load(folder/'document_scores.npy',allow_pickle=False)
        if old_scores.shape != (len(old_images),len(old_documents)) or not np.isfinite(old_scores).all():
            raise ValueError('inherited_score_table_invalid')
    documents, indices = document_plan(inputs,blocks,inherited_index['documents'] if inherited_index else None)
    if inherited_index and (indices['long_indices'] != inherited_index['long_indices'] or indices['block_indices'] != inherited_index['block_indices']):
        raise ValueError('inherited_document_groups_changed')
    old_count = len(old_documents) if old_documents is not None else 0
    atomic_json(output/'document_index.json',{'documents':documents,**indices,
        'inherited_document_count':old_count,'encoding_policy':protocol['block_policy']['document_encoding']})
    new_queries = manifest['rows'][len(old_images):] if old_images is not None else manifest['rows']
    if len(documents) > old_count or new_queries:
        state['status'] = 'loading_backend'
        save_state()
        if args.backend == 'siglip2_base':
            from core_modules.image_search import ImageEmbedder
            model = ImageEmbedder('siglip2_base',device=args.device)
            if model.processor is None or getattr(getattr(model.model,'config',None),'model_type',None) not in {'siglip','siglip2'}:
                raise ValueError('siglip_weights_not_loaded')
        else:
            from core_modules.organize.embedding_gemma import EmbeddingGemmaEmbedder
            model = EmbeddingGemmaEmbedder(device=args.device)
        # The Gemma backend loads lazily; ensure identity is available first.
        if hasattr(model,'_ensure_model'):
            model._ensure_model()
        elif hasattr(model,'_load'):
            model._load()
    else:
        model = None
    started = time.perf_counter()
    values = []
    state['status'] = 'encoding_missing_documents'
    save_state()
    for i,row in enumerate(documents[old_count:],1):
        value = normalize_vectors(model.encode_text_batch([row['text']],batch_size=1),1)
        if old_documents is not None and value.shape[1] != old_documents.shape[1]:
            raise ValueError('new_document_dimension_differs')
        values.append(value[0])
        state['document_encoding'] = {'completed':i,'total':len(documents)-old_count,'reused':old_count,'batch_size':1}
        save_state()
    if model is not None and reuse and (model.model_signature != reuse['model_signature'] or model.preprocessing_signature != reuse['preprocessing_signature']):
        raise ValueError('historical_backend_identity_changed')
    document_vectors = np.concatenate([old_documents,np.stack(values)],axis=0) if old_documents is not None and values else (old_documents.copy() if old_documents is not None else np.stack(values))
    check_vectors(document_vectors,len(documents))
    np.save(output/'document_vectors.npy',document_vectors,allow_pickle=False)
    state['status'] = 'encoding_added_queries'
    save_state()
    groups = defaultdict(list)
    for row in new_queries:
        groups[row['play_id']].append(row)
    by_id = {}
    for play,queries in groups.items():
        paths = [row['neutral_path'] for row in queries]
        if args.backend == 'siglip2_base':
            values,valid = model.extract_features_batch(paths,sub_batch_size=8)
            if paths != valid:
                raise ValueError('failed_query_cannot_be_removed')
        else:
            values = model.extract_features_batch(paths,batch_size=8)
        values = normalize_vectors(values,len(paths))
        for row,value in zip(queries,values):
            by_id[row['file_id']] = value
        state['image_encoding'][play] = {'encoded':len(queries),'calls':1}
        save_state()
    new_images = np.stack([by_id[row['file_id']] for row in new_queries]) if new_queries else None
    image_vectors = np.concatenate([old_images,new_images],axis=0) if old_images is not None and new_images is not None else (old_images.copy() if old_images is not None else new_images)
    check_vectors(image_vectors,len(manifest['rows']),document_vectors.shape[1])
    np.save(output/'image_vectors.npy',image_vectors,allow_pickle=False)
    raw_scores = image_vectors @ document_vectors.T
    inherited_score_drift = 0.0
    if old_scores is not None:
        inherited_score_drift = float(np.abs(raw_scores[:len(old_images),:old_count]-old_scores).max())
        # Original scores remain byte-exact even if BLAS block shape changes rounding.
        raw_scores[:len(old_images),:old_count] = old_scores
    np.save(output/'document_scores.npy',raw_scores,allow_pickle=False)
    tables = {'single':score_candidates(raw_scores,indices['long_indices'],'mean'),
        'mean':score_candidates(raw_scores,indices['block_indices'],'mean'),
        'max':score_candidates(raw_scores,indices['block_indices'],'max'),
        'baseline':score_candidates(raw_scores,indices['baseline_indices'],'mean')}
    np.savez(output/'candidate_scores.npz',**tables)
    variants = {}
    for item in protocol['variants']:
        count = item['candidate_count']
        table = tables['baseline'] if item['representation'] == 'baseline' else tables[item['aggregation']]
        candidates = inputs['baseline']['scenes'] if item['representation']=='baseline' else manifest['scenes'][:count]
        rows = ranked_rows(manifest['rows'],candidates,table[:,:count])
        variants[item['name']] = {'protocol_variant':item,
            'subsets':subset_metrics(rows,candidates,protocol['query_subsets']),'rows':rows}
    all_ids = {c['scene_id'] for c in inputs['samepool']['scenes']}
    pool_same = []
    historical_same = []
    for representation in ('long','blocks_mean','blocks_max'):
        left,right = variants[representation+'_samepool']['rows'],variants[representation+'_expanded']['rows']
        for a,b in zip(left,right):
            filtered = [(s,v) for s,v in zip(b['ranked_scenes'],b['scores']) if s in all_ids]
            pool_same.append(a['ranked_scenes']==[s for s,_ in filtered] and a['scores']==[v for _,v in filtered])
        if old_summary:
            for pool,old_pool in (('samepool','samepool25'),('expanded','expanded28')):
                old_rows = old_summary['variants'][representation+'_'+old_pool]['rows']
                new_rows = variants[representation+'_'+pool]['rows'][:len(old_rows)]
                historical_same.append(new_rows == old_rows)
    checks = {
        'protocol_unchanged':fingerprint(args.protocol)==protocol_sha,
        'all_bound_inputs_unchanged':all(fingerprint(path)==sha for path,sha in protocol['bound_files_sha256'].items()),
        'all_query_subsets_reported':all(set(row['subsets'])=={name for name,ids in protocol['query_subsets'].items() if ids} for row in variants.values()),
        'all_queries_in_every_variant':all(len(row['rows'])==len(protocol['query_subsets']['all']) for row in variants.values()),
        'no_failed_query_silently_excluded':len(by_id)==len(new_queries),
        'historical_image_vectors_retained':old_images is None or np.array_equal(image_vectors[:len(old_images)],old_images),
        'historical_document_vectors_retained':old_documents is None or np.array_equal(document_vectors[:old_count],old_documents),
        'historical_document_scores_retained':old_scores is None or np.array_equal(raw_scores[:len(old_images),:old_count],old_scores),
        'historical_six_variant_rankings_retained':not historical_same or all(historical_same),
        'same_pool_scores_unchanged_when_new_candidates_removed':all(pool_same),
        'new_backend_matches_inherited_weights_and_preprocessing':model is None or reuse is None or (model.model_signature==reuse['model_signature'] and model.preprocessing_signature==reuse['preprocessing_signature']),
        'blocks_within_declared_window':all(row['token_count']<=64 for row in blocks['blocks']),
        'development_and_no_default_change':protocol['test_locked'] is False,
    }
    artifacts = ('document_vectors.npy','image_vectors.npy','document_scores.npy','candidate_scores.npz','document_index.json','input_manifest.json')
    summary = {'status':'development_retrieval_study_complete' if all(checks.values()) else 'integrity_failed',
        'backend':args.backend,'protocol_sha256':protocol_sha,'query_subsets':protocol['query_subsets'],
        'model_signature':model.model_signature if model else reuse['model_signature'],
        'preprocessing_signature':model.preprocessing_signature if model else reuse['preprocessing_signature'],
        'counts':{**protocol['counts'],'documents':len(documents),'reused_image_vectors':len(old_images) if old_images is not None else 0,
            'new_image_vectors':len(new_queries),'reused_document_vectors':old_count,'new_document_vectors':len(documents)-old_count},
        'variants':variants,'checks':checks,'source_sha256':protocol['bound_files_sha256'],
        'artifact_sha256':{name:fingerprint(output/name) for name in artifacts},
        'inherited_score_reconstruction_max_abs_error':inherited_score_drift,
        'new_encoding_elapsed_seconds':time.perf_counter()-started,
        'limitations':['Development data; independent performance, face boxes and final test remain unverified.',
            'Historical vectors/scores are reused exactly; time includes only new work and cannot support a speed claim.',
            'Source-version equivalence remains unresolved. Mean is primary, maximum is secondary.',
            'Macro-F1 uses declared truth scene IDs; scene_macro_recall_at_1 is separately named.',
            'No production default adoption; U2 remains paused.']}
    atomic_json(output/'summary.json',summary)
    state.update(status=summary['status'],summary_sha256=fingerprint(output/'summary.json'),checks=checks)
    save_state()
    print(json.dumps({'backend':args.backend,'status':summary['status'],'counts':summary['counts'],'checks':checks,
        'metrics':{name:row['subsets']['all']['metrics'] for name,row in variants.items()}},ensure_ascii=False,indent=2))
    return all(checks.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--backend',choices=('siglip2_base','embeddinggemma2'),required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not evaluate(args):
        raise SystemExit(1)


if __name__=='__main__':
    main()
