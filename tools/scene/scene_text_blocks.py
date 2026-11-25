"""Exact quotation blocks and candidate scoring for offline retrieval studies."""
from collections import Counter
import hashlib
import json
import re

import numpy as np

BLOCK_VERSION = 'scene_text_blocks/v1'
POLICY = {
    'version': BLOCK_VERSION,
    'max_tokens_including_prefix_and_special_tokens': 64,
    'prefix': 'play_title + space + confirmed_scene_label + space',
    'quotation_policy': 'all existing evidence, in source order; no query-based sentence selection',
    'split_policy': 'longest fitting prefix; prefer last fitting sentence punctuation, then clause punctuation, then Unicode character boundary',
    'normalization': 'collapse quotation whitespace only; preserve exact quote and offsets separately',
    'primary_aggregation': 'arithmetic_mean_of_block_cosines',
    'secondary_aggregation': 'maximum_block_cosine',
    'secondary_limitation': 'more blocks provide more opportunities for a high maximum',
    'document_encoding': 'one unique document per call, batch_size=1; deduplicate exact text',
    'pool_comparison': '25 and 28 candidate rankings derived from the same saved image/document vectors and score table',
    'no_hyperparameter_selection_from_results': True,
}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def quotation_document(prefix, quote):
    return prefix + re.sub(r'\s+', ' ', quote).strip()


def split_exact_quote(prefix, quote, count_tokens, budget=64):
    """Cover every source character exactly once; never decode truncated tokens."""
    if not quote or not quote.strip():
        raise ValueError('empty_block_quote')
    if count_tokens(prefix) >= budget:
        raise ValueError('block_prefix_exhausts_token_budget')
    output, start = [], 0
    while start < len(quote):
        fitting = [end for end in range(start + 1, len(quote) + 1)
                   if count_tokens(quotation_document(prefix, quote[start:end])) <= budget]
        if not fitting:
            raise ValueError('source_character_exceeds_token_budget')
        end = fitting[-1]
        if end != len(quote):
            sentences = [position for position in fitting if re.search(r'[。！？!?；;](?:[”’」』"])?$', quote[start:position])]
            clauses = [position for position in fitting if quote[position - 1] in '，,、：:\n']
            if sentences:
                end = sentences[-1]
            elif clauses:
                end = clauses[-1]
        value = quote[start:end]
        output.append({'start': start, 'end': end, 'quote': value,
                       'text': quotation_document(prefix, value),
                       'token_count': count_tokens(quotation_document(prefix, value))})
        start = end
    if ''.join(row['quote'] for row in output) != quote:
        raise ValueError('block_quote_coverage_failed')
    return output


def build_scene_blocks(scenes, titles, count_tokens, budget=64):
    """Read scene documents only; query captions and actor lists are not inputs.

    Web quotations retain their original extracted-text spans. Existing local
    plot documents use exact spans in the frozen prepared scene text; the
    preparation tool separately binds the original local plot file and parser.
    """
    blocks, candidates = [], []
    for scene in scenes:
        title, label = titles[scene['play_id']], scene['label']
        prefix = title + ' ' + label + ' '
        references = []
        if scene['text_source'] == 'scene_label_only':
            length = count_tokens(scene['text'])
            if length > budget:
                raise ValueError('label_only_document_exceeds_budget')
            value = {'play_id': scene['play_id'], 'scene_id': scene['scene_id'], 'label': label,
                     'text': scene['text'], 'token_count': length,
                     'evidence': None, 'source_ref': scene['source_ref']}
            value['block_id'] = 'block-' + text_hash(scene['scene_id'] + ':label')[:24]
            blocks.append(value)
            references.append(value['block_id'])
        else:
            if scene['text_source'] == 'sourced_plot_excerpt_and_scene_label':
                evidence = scene['evidence']
                if scene['text'] != prefix + ' '.join(re.sub(r'\s+', ' ', item['quote']).strip() for item in evidence):
                    raise ValueError('sourced_scene_text_differs_from_evidence')
                kind = 'source_extracted_text'
            elif scene['text_source'] == 'local_plot_and_scene_label':
                if not scene['text'].startswith(prefix):
                    raise ValueError('local_scene_prefix_mismatch')
                evidence = [{'source_id': scene['source_ref'], 'start': len(prefix),
                             'end': len(scene['text']), 'quote': scene['text'][len(prefix):]}]
                kind = 'frozen_prepared_scene_text'
            else:
                raise ValueError('unsupported_scene_text_source')
            for evidence_index, item in enumerate(evidence):
                if item['end'] - item['start'] != len(item['quote']):
                    raise ValueError('quote_span_length_mismatch')
                for part in split_exact_quote(prefix, item['quote'], count_tokens, budget):
                    reference = {'basis': kind, 'source_id': item['source_id'],
                                 'start': item['start'] + part['start'], 'end': item['start'] + part['end'],
                                 'quote': part['quote'], 'evidence_index': evidence_index,
                                 'prepared_scene_text_sha256': text_hash(scene['text'])}
                    identity = json.dumps([scene['scene_id'], evidence_index, reference['start'], reference['end']], ensure_ascii=False)
                    block_id = 'block-' + text_hash(identity)[:24]
                    value = {'block_id': block_id, 'play_id': scene['play_id'], 'scene_id': scene['scene_id'],
                             'label': label, 'text': part['text'], 'token_count': part['token_count'],
                             'evidence': reference, 'source_ref': scene['source_ref']}
                    blocks.append(value)
                    references.append(block_id)
        candidates.append({'play_id': scene['play_id'], 'scene_id': scene['scene_id'], 'label': label,
                           'block_ids': references, 'text_source': scene['text_source']})
    if len({row['block_id'] for row in blocks}) != len(blocks):
        raise ValueError('duplicate_block_id')
    return {'policy': {**POLICY, 'max_tokens_including_prefix_and_special_tokens': budget},
            'blocks': blocks, 'candidates': candidates}


def normalize_vectors(values, count):
    values = np.asarray(values, dtype='float32')
    if values.ndim != 2 or len(values) != count or not np.isfinite(values).all():
        raise ValueError('invalid_embedding_shape_or_value')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('zero_embedding')
    return values / norms


def score_candidates(document_scores, candidate_indices, aggregation):
    values = np.asarray(document_scores, dtype='float32')
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('invalid_document_scores')
    if aggregation not in ('mean', 'max'):
        raise ValueError('unknown_block_aggregation')
    if any(not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or i >= values.shape[1] for i in indices)
           for indices in candidate_indices):
        raise ValueError('invalid_candidate_document_indices')
    columns = []
    for indices in candidate_indices:
        block_scores = values[:, indices]
        columns.append(block_scores.mean(axis=1) if aggregation == 'mean' else block_scores.max(axis=1))
    return np.stack(columns, axis=1)


def ranked_rows(queries, candidates, scores):
    scores = np.asarray(scores, dtype='float32')
    if scores.shape != (len(queries), len(candidates)) or not np.isfinite(scores).all():
        raise ValueError('invalid_candidate_scores')
    if len({row['scene_id'] for row in candidates}) != len(candidates):
        raise ValueError('duplicate_ranked_candidate')
    result = []
    for row, values in zip(queries, scores):
        if row['scene_id'] not in {c['scene_id'] for c in candidates if c['play_id'] == row['play_id']}:
            raise ValueError('query_truth_missing_from_pool')
        pool = [i for i, scene in enumerate(candidates) if scene['play_id'] == row['play_id']]
        order = np.argsort(-values[pool], kind='stable')
        result.append({'file_id': row['file_id'], 'play_id': row['play_id'], 'expected': row['scene_id'],
                       'ranked_scenes': [candidates[pool[int(i)]]['scene_id'] for i in order],
                       'scores': [float(values[pool[int(i)]]) for i in order]})
    return result


def retrieval_metrics(rows):
    if not rows:
        raise ValueError('empty_retrieval_rows')
    scenes = Counter(row['expected'] for row in rows)
    correct = {k: sum(row['expected'] in row['ranked_scenes'][:k] for row in rows) for k in (1, 3)}
    return {'count': len(rows), 'correct_at_1': correct[1], 'correct_at_3': correct[3],
            'recall_at_1': correct[1] / len(rows), 'recall_at_3': correct[3] / len(rows),
            'scene_macro_recall_at_1': sum(sum(row['expected'] == scene and row['ranked_scenes'][0] == scene for row in rows) / count
                                         for scene, count in scenes.items()) / len(scenes)}
