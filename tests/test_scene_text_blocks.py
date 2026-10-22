"""Quotation coverage, token budgets and fixed-score pool comparisons."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.scene_text_blocks import (build_scene_blocks, normalize_vectors, ranked_rows,
                                    retrieval_metrics, score_candidates, split_exact_quote)


class SceneBlockTests(unittest.TestCase):
    def count(self, text):
        return len(text) + 1

    def scene(self):
        prefix = '剧《甲》 第1场 '
        quote = '宫殿中欢歌。公主迎接贵客，随后二人分别。'
        return {'play_id': 'play-a', 'scene_id': 'scene-a', 'label': '第1场',
                'text': prefix + quote, 'text_source': 'sourced_plot_excerpt_and_scene_label',
                'source_ref': 'https://example.test#text-30-51',
                'evidence': [{'source_id': 'official', 'start': 30, 'end': 30 + len(quote), 'quote': quote}]}

    def test_exact_coverage_and_source_offsets(self):
        scene = self.scene()
        result = build_scene_blocks([scene], {'play-a': '剧《甲》'}, self.count, budget=19)
        blocks = result['blocks']
        self.assertGreater(len(blocks), 1)
        self.assertEqual(''.join(b['evidence']['quote'] for b in blocks), scene['evidence'][0]['quote'])
        source = 'x' * 30 + scene['evidence'][0]['quote']
        for block in blocks:
            evidence = block['evidence']
            self.assertEqual(source[evidence['start']:evidence['end']], evidence['quote'])
            self.assertLessEqual(block['token_count'], 19)
            self.assertEqual(block['evidence']['basis'], 'source_extracted_text')

    def test_sentence_and_clause_boundaries_keep_all_characters(self):
        rows = split_exact_quote('剧 ', '甲乙。丙丁，戊己庚辛壬。', self.count, budget=11)
        self.assertEqual(rows[0]['quote'], '甲乙。')
        self.assertEqual(''.join(row['quote'] for row in rows), '甲乙。丙丁，戊己庚辛壬。')
        rows = split_exact_quote('剧 ', '甲乙，丙丁戊己庚辛壬', self.count, budget=9)
        self.assertEqual(rows[0]['quote'], '甲乙，')

    def test_long_sentence_falls_back_without_decoding_or_losing_tail(self):
        value = '甲乙丙丁戊己庚辛壬癸' * 3
        rows = split_exact_quote('剧 ', value, self.count, budget=10)
        self.assertEqual(''.join(row['quote'] for row in rows), value)
        self.assertTrue(all(self.count(row['text']) <= 10 for row in rows))
        self.assertEqual(rows[-1]['end'], len(value))

    def test_prefix_budget_and_empty_quote_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prefix_exhausts'):
            split_exact_quote('长前缀', '剧情', self.count, budget=4)
        with self.assertRaisesRegex(ValueError, 'empty_block_quote'):
            split_exact_quote('剧 ', '', self.count)

    def test_mapping_text_change_rejected_and_query_metadata_unused(self):
        scene = self.scene()
        scene['caption'] = 'PRIVATE_QUERY'
        scene['actors'] = ['PRIVATE_ACTOR']
        result = build_scene_blocks([scene], {'play-a': '剧《甲》'}, self.count)
        self.assertNotIn('PRIVATE_QUERY', str(result))
        self.assertNotIn('PRIVATE_ACTOR', str(result))
        scene['text'] += '凭空追加'
        with self.assertRaisesRegex(ValueError, 'differs_from_evidence'):
            build_scene_blocks([scene], {'play-a': '剧《甲》'}, self.count)

    def test_label_only_and_local_spans_use_explicit_different_basis(self):
        local = {**self.scene(), 'text_source': 'local_plot_and_scene_label'}
        label = {'play_id': 'play-a', 'scene_id': 'curtain', 'label': '谢幕', 'text': '剧《甲》 谢幕 ',
                 'text_source': 'scene_label_only', 'source_ref': 'vocabulary'}
        result = build_scene_blocks([local, label], {'play-a': '剧《甲》'}, self.count, budget=19)
        for block in result['blocks'][:-1]:
            e = block['evidence']
            self.assertEqual(e['basis'], 'frozen_prepared_scene_text')
            self.assertEqual(local['text'][e['start']:e['end']], e['quote'])
        self.assertIsNone(result['blocks'][-1]['evidence'])
        self.assertEqual(result['blocks'][-1]['text'], label['text'])

    def test_expansion_does_not_change_original_blocks(self):
        scene = self.scene()
        new = {**deepcopy(scene), 'scene_id': 'scene-b', 'label': '第2场'}
        new['text'] = new['text'].replace('第1场', '第2场')
        first = build_scene_blocks([scene], {'play-a': '剧《甲》'}, self.count, budget=19)
        expanded = build_scene_blocks([scene, new], {'play-a': '剧《甲》'}, self.count, budget=19)
        self.assertEqual(first['blocks'], expanded['blocks'][:len(first['blocks'])])
        self.assertEqual(first['candidates'], expanded['candidates'][:1])

    def test_mean_and_max_report_different_predefined_aggregations(self):
        scores = np.array([[.9, .1, .6]], dtype='float32')
        mean = score_candidates(scores, [[0, 1], [2]], 'mean')
        maximum = score_candidates(scores, [[0, 1], [2]], 'max')
        np.testing.assert_allclose(mean, [[.5, .6]])
        np.testing.assert_allclose(maximum, [[.9, .6]])
        with self.assertRaisesRegex(ValueError, 'invalid_candidate_document_indices'):
            score_candidates(scores, [[], [2]], 'max')
        with self.assertRaisesRegex(ValueError, 'invalid_candidate_document_indices'):
            score_candidates(scores, [[0, 0]], 'mean')

    def test_fixed_score_pool_restriction_and_play_isolation(self):
        queries = [{'file_id': 'q', 'play_id': 'p', 'scene_id': 'a'}]
        candidates = [{'scene_id': 'a', 'play_id': 'p'}, {'scene_id': 'b', 'play_id': 'p'},
                      {'scene_id': 'extra', 'play_id': 'p'}, {'scene_id': 'other_play', 'play_id': 'z'}]
        scores = np.array([[.4, .4, .6, .99]], dtype='float32')
        expanded = ranked_rows(queries, candidates, scores)
        original = ranked_rows(queries, candidates[:2], scores[:, :2])
        self.assertEqual(expanded[0]['ranked_scenes'], ['extra', 'a', 'b'])
        self.assertEqual(original[0]['ranked_scenes'], ['a', 'b'])
        self.assertEqual([scene for scene in expanded[0]['ranked_scenes'] if scene != 'extra'], original[0]['ranked_scenes'])
        self.assertEqual(expanded[0]['scores'][1:], original[0]['scores'])
        self.assertEqual(retrieval_metrics(original)['correct_at_1'], 1)
        self.assertEqual(retrieval_metrics(expanded)['correct_at_1'], 0)

    def test_invalid_embeddings_and_scores_cannot_hide_queries(self):
        with self.assertRaisesRegex(ValueError, 'zero_embedding'):
            normalize_vectors([[0, 0]], 1)
        with self.assertRaisesRegex(ValueError, 'shape_or_value'):
            normalize_vectors([[float('nan'), 1]], 1)
        with self.assertRaisesRegex(ValueError, 'missing_from_pool'):
            ranked_rows([{'file_id': 'q', 'play_id': 'p', 'scene_id': 'missing'}],
                        [{'scene_id': 'a', 'play_id': 'p'}], [[.9]])


if __name__ == '__main__':
    unittest.main()
