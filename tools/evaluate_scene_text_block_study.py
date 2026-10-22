"""Encode once per backend and evaluate frozen long/block texts in two pools."""
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
from tools.prepare_scene_text_block_study import protocol_files_unchanged, validate_protocol
from tools.scene_text_blocks import (normalize_vectors, ranked_rows, retrieval_metrics,
                                    score_candidates, text_hash)
from tools.sourced_scene_candidates import candidate_files_unchanged


def state_save(path, state):
    state['recorded_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, state)


def images_unchanged(manifest, source_root):
    return all(fingerprint(source_root / row['source']) == row['content_hash'] and
               fingerprint(row['neutral_path']) == row['neutral_sha256'] for row in manifest['rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--backend', choices=('siglip2_base', 'embeddinggemma2'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('block_study_run_output_must_be_empty')
    protocol_hash = fingerprint(args.protocol)
    protocol, same, expanded, blocks = validate_protocol(args.protocol)
    source_root = Path(protocol['source_root'])
    if source_root == output or source_root in output.parents or output in source_root.parents:
        raise ValueError('block_study_output_must_be_outside_originals')
    if not images_unchanged(expanded, source_root):
        raise ValueError('block_study_images_changed_before_encoding')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'input_manifest.json', expanded)
    state_path = output / 'run_state.json'
    state = {'status': 'loading_backend', 'backend': args.backend,
             'protocol': str(args.protocol.resolve()), 'protocol_sha256': protocol_hash,
             'started_at': datetime.now(timezone.utc).isoformat(), 'image_encoding': {}, 'document_encoding': {}}
    state_save(state_path, state)
    if args.backend == 'siglip2_base':
        from core_modules.image_search import ImageEmbedder
        model = ImageEmbedder('siglip2_base', device=args.device)
        if model.processor is None or getattr(getattr(model.model, 'config', None), 'model_type', None) not in ('siglip', 'siglip2'):
            raise ValueError('siglip_weights_not_loaded')
    else:
        from core_modules.organize.embedding_gemma import EmbeddingGemmaEmbedder
        model = EmbeddingGemmaEmbedder(device=args.device)
    started = time.perf_counter()
    documents, document_index = [], {}

    def add_document(text):
        if text not in document_index:
            document_index[text] = len(documents)
            documents.append({'text': text, 'text_sha256': text_hash(text)})
        return document_index[text]

    long_indices = [[add_document(scene['text'])] for scene in expanded['scenes']]
    block_by_id = {row['block_id']: row for row in blocks['blocks']}
    block_indices = [[add_document(block_by_id[block_id]['text']) for block_id in scene['block_ids']]
                     for scene in blocks['candidates']]
    atomic_json(output / 'document_index.json', {'documents': documents, 'long_indices': long_indices,
                                               'block_indices': block_indices,
                                               'encoding_policy': protocol['policy']['document_encoding']})
    state['status'] = 'encoding_unique_documents'
    state_save(state_path, state)
    vectors = []
    for index, row in enumerate(documents):
        vector = normalize_vectors(model.encode_text_batch([row['text']], batch_size=1), 1)
        vectors.append(vector[0])
        if index % 8 == 0 or index + 1 == len(documents):
            state['document_encoding'] = {'completed': index + 1, 'total': len(documents), 'batch_size': 1}
            state_save(state_path, state)
    document_vectors = normalize_vectors(np.stack(vectors), len(documents))
    np.save(output / 'document_vectors.npy', document_vectors, allow_pickle=False)
    grouped = defaultdict(list)
    for row in expanded['rows']:
        grouped[row['play_id']].append(row)
    image_index, image_vectors = {}, []
    state['status'] = 'encoding_images_once'
    state_save(state_path, state)
    for play_id, queries in grouped.items():
        paths = [row['neutral_path'] for row in queries]
        if args.backend == 'siglip2_base':
            values, valid = model.extract_features_batch(paths, sub_batch_size=8)
            if valid != paths:
                raise ValueError('image_encoding_cannot_remove_failed_queries')
        else:
            values = model.extract_features_batch(paths, batch_size=8)
        values = normalize_vectors(values, len(paths))
        for query, vector in zip(queries, values):
            image_index[query['file_id']] = vector
        state['image_encoding'][play_id] = {'completed': len(queries), 'calls': 1}
        state_save(state_path, state)
    image_vectors = normalize_vectors(np.stack([image_index[row['file_id']] for row in expanded['rows']]), len(expanded['rows']))
    np.save(output / 'image_vectors.npy', image_vectors, allow_pickle=False)
    raw_scores = image_vectors @ document_vectors.T
    np.save(output / 'document_scores.npy', raw_scores, allow_pickle=False)
    score_tables = {'single': score_candidates(raw_scores, long_indices, 'mean'),
                    'mean': score_candidates(raw_scores, block_indices, 'mean'),
                    'max': score_candidates(raw_scores, block_indices, 'max')}
    variants = {}
    for item in protocol['variants']:
        count, aggregation = item['candidate_count'], item['aggregation']
        candidate_rows = expanded['scenes'][:count]
        rows = ranked_rows(expanded['rows'], candidate_rows, score_tables[aggregation][:, :count])
        variants[item['name']] = {'protocol_variant': item, 'metrics': retrieval_metrics(rows),
                                  'per_play': {play_id: retrieval_metrics([row for row in rows if row['play_id'] == play_id]) for play_id in grouped},
                                  'rows': rows}
    np.savez(output / 'candidate_scores.npz', **score_tables)
    original_ids = {row['scene_id'] for row in same['scenes']}
    pool_identity = {}
    for representation in ('long', 'blocks_mean', 'blocks_max'):
        left, right = variants[representation + '_samepool25']['rows'], variants[representation + '_expanded28']['rows']
        unchanged = True
        for a, b in zip(left, right):
            pairs = [(scene, score) for scene, score in zip(b['ranked_scenes'], b['scores']) if scene in original_ids]
            unchanged = unchanged and a['ranked_scenes'] == [scene for scene, _ in pairs] and a['scores'] == [score for _, score in pairs]
        pool_identity[representation] = unchanged
    checks = {
        'protocol_hash_unchanged': fingerprint(args.protocol) == protocol_hash,
        'protocol_bound_files_unchanged': protocol_files_unchanged(protocol),
        'source_snapshots_unchanged': candidate_files_unchanged(expanded),
        'original_and_neutral_images_unchanged': images_unchanged(expanded, source_root),
        'all_260_queries_in_every_variant': all(len(row['rows']) == len(expanded['rows']) == 260 for row in variants.values()),
        'all_short_blocks_within_frozen_token_budget': all(row['token_count'] <= 64 for row in blocks['blocks']),
        'all_image_vectors_encoded_once': len(image_index) == len(expanded['rows']) and all(row['calls'] == 1 for row in state['image_encoding'].values()),
        'one_document_vector_per_unique_text': len(document_index) == len(document_vectors) == len(documents),
        'samepool_rankings_and_scores_identical_when_extra_ids_removed': all(pool_identity.values()),
        'label_only_documents_have_one_shared_vector': all(
            long_indices[index] == block_indices[index] for index, scene in enumerate(expanded['scenes']) if scene['text_source'] == 'scene_label_only'),
        'all_predeclared_variants_reported': set(variants) == {row['name'] for row in protocol['variants']},
        'final_test_still_unlocked': protocol['test_locked'] is False and expanded['test_locked'] is False,
    }
    arrays = ['document_vectors.npy', 'image_vectors.npy', 'document_scores.npy', 'candidate_scores.npz', 'document_index.json', 'input_manifest.json']
    report = {'status': 'development_text_block_study_complete', 'backend': args.backend,
              'protocol_sha256': protocol_hash, 'policy': protocol['policy'],
              'model_signature': model.model_signature, 'preprocessing_signature': model.preprocessing_signature,
              'source_sha256': protocol['measured_sources_sha256'], 'variants': variants,
              'checks': checks, 'pool_score_identity': pool_identity,
              'artifact_sha256': {name: fingerprint(output / name) for name in arrays},
              'query_order': [row['file_id'] for row in expanded['rows']],
              'encoding_elapsed_seconds': time.perf_counter() - started,
              'counts': {'image_vectors': len(image_vectors), 'document_vectors': len(document_vectors),
                         'blocks': len(blocks['blocks']), 'long_documents': len(expanded['scenes'])},
              'limitations': ['Development data; final test not locked; production version equivalence still unverified.',
                             'No new source paragraphs; same selected v1 quotations, split only by fixed token/boundary rules.',
                             'Mean is primary and max is a predeclared secondary with block-count bias; no default model change.',
                             'Document batch size 1 differs from v1; long control is repeated within this fixed protocol.',
                             'Shared GPU, differing load boundaries; no speed or U2 claim.']}
    atomic_json(output / 'summary.json', report)
    state['status'] = 'completed' if all(checks.values()) else 'integrity_failed'
    state['summary_sha256'] = fingerprint(output / 'summary.json')
    state['checks'] = checks
    state_save(state_path, state)
    print(json.dumps({'backend': args.backend, 'checks': checks,
                      'variants': {name: row['metrics'] for name, row in variants.items()}}, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
