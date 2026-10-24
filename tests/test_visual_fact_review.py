"""Review integrity, independent reviewers, disagreements and OCR denominators."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from tools.prepare_visual_fact_review import make_row, prepare, validate_draft, evaluate_pair, verify_existing
from tools.review_quality_metrics import b_cubed, character_errors


def templates(rows):
    return {slot: {'schema_version': 1, 'protocol_sha256': 'f' * 64, 'slot': slot,
                   'reviewer': '', 'rows': deepcopy(rows), 'final_test_locked': False,
                   'human_decisions_applied': False} for slot in ('A', 'B')}


def completed(template, reviewer, decision='supported'):
    draft = deepcopy(template)
    draft['reviewer'] = reviewer
    for row in draft['rows']:
        row['review'].update(status='reviewed', source_ref='independent inspection',
                             reviewed_at='2026-10-24T00:00:00+00:00')
        for fact in row['review']['facts'].values():
            fact['decision'] = decision
            if decision == 'error':
                fact['correction'] = 'corrected statement'
        if row['kind'] == 'ocr':
            row['review']['reference_text'] = '舞台文字'
    return draft


class VisualFactReviewTests(unittest.TestCase):
    def setUp(self):
        self.observation = {'clothing': ['red coat'], 'props': [], 'setting': [],
                            'actions': [], 'visible_text': [], 'uncertain': ['actor identity']}
        self.visual = make_row('visual', 'image-1', '/tmp/source.jpg', 'a' * 64, self.observation)
        self.ocr = make_row('ocr', 'image-2', '/tmp/source2.jpg', 'b' * 64, {'lines': [{'text': '舞台文宁'}]})

    def test_pending_templates_do_not_create_quality_or_OCR_accuracy(self):
        t = templates([self.visual, self.ocr])
        result = evaluate_pair(t['A'], t['A'], t['B'], t['B'])
        self.assertEqual(result['paired_evaluable_images'], 0)
        self.assertEqual(result['distinct_active_reviewers'], 0)
        self.assertIsNone(result['consensus_error_ratio'])
        self.assertIsNone(result['ocr']['cer'])
        self.assertEqual(result['quality_gate'], 'insufficient_data')

    def test_sources_claims_protocol_and_test_status_are_immutable(self):
        t = templates([self.visual])['A']
        for mutation in ('source', 'statement', 'protocol', 'lock', 'duplicate_id'):
            with self.subTest(mutation=mutation):
                d = completed(t, 'reviewer A')
                if mutation == 'source':
                    d['rows'][0]['source_sha256'] = 'c' * 64
                elif mutation == 'statement':
                    d['rows'][0]['facts'][0]['statement'] = 'different coat'
                elif mutation == 'protocol':
                    d['protocol_sha256'] = 'c' * 64
                elif mutation == 'lock':
                    d['final_test_locked'] = True
                else:
                    d['rows'].append(deepcopy(d['rows'][0]))
                with self.assertRaises(ValueError):
                    validate_draft(t, d)

    def test_two_distinct_reviewers_are_required(self):
        t = templates([self.visual])
        with self.assertRaisesRegex(ValueError, 'distinct_reviewers'):
            evaluate_pair(t['A'], completed(t['A'], 'Rater'), t['B'], completed(t['B'], ' rater '))

    def test_pair_requires_same_protocol_sources_claims_and_distinct_slots(self):
        for mutation in ('protocol', 'source', 'claim', 'prediction', 'slot'):
            with self.subTest(mutation=mutation):
                t = templates([self.visual, self.ocr])
                if mutation == 'protocol':
                    t['B']['protocol_sha256'] = 'e' * 64
                elif mutation == 'source':
                    t['B']['rows'][0]['source_sha256'] = 'e' * 64
                elif mutation == 'claim':
                    t['B']['rows'][0]['facts'][0]['statement'] = 'blue coat'
                elif mutation == 'prediction':
                    t['B']['rows'][1]['predicted_text'] = 'other prediction'
                else:
                    t['B']['slot'] = 'A'
                with self.assertRaisesRegex(ValueError, 'pair_binding'):
                    evaluate_pair(t['A'], completed(t['A'], 'A'), t['B'], completed(t['B'], 'B'))

    def test_duplicate_image_bytes_cannot_inflate_fact_sample_or_OCR_denominator(self):
        for kind in ('visual', 'ocr'):
            with self.subTest(kind=kind):
                observation = self.observation if kind == 'visual' else {'lines': [{'text': '舞台'}]}
                rows = [make_row(kind, str(i), '/tmp/source.jpg', 'a' * 64, observation) for i in range(100)]
                t = templates(rows)
                with self.assertRaisesRegex(ValueError, 'duplicate_image_content'):
                    evaluate_pair(t['A'], completed(t['A'], 'A'), t['B'], completed(t['B'], 'B'))

    def test_completed_review_requires_all_facts_explanation_reviewer_source_and_timezone(self):
        t = templates([self.visual])['A']
        cases = []
        pending = completed(t, 'A'); next(iter(pending['rows'][0]['review']['facts'].values()))['decision'] = 'pending'; cases.append(pending)
        no_source = completed(t, 'A'); no_source['rows'][0]['review']['source_ref'] = ''; cases.append(no_source)
        no_time = completed(t, 'A'); no_time['rows'][0]['review']['reviewed_at'] = '2026-10-24'; cases.append(no_time)
        no_name = completed(t, ''); cases.append(no_name)
        error = completed(t, 'A', 'error'); next(iter(error['rows'][0]['review']['facts'].values()))['correction'] = ''; cases.append(error)
        for draft in cases:
            with self.assertRaises(ValueError):
                validate_draft(t, draft)

    def test_disagreement_is_not_turned_into_a_supported_fact(self):
        t = templates([self.visual])
        result = evaluate_pair(t['A'], completed(t['A'], 'A'), t['B'], completed(t['B'], 'B', 'error'))
        self.assertEqual(result['fact_counts']['disagreement'], 1)
        self.assertEqual(len(result['disagreements']), 1)
        self.assertIsNone(result['consensus_error_ratio'])

    def test_duplicate_error_stays_in_denominator_and_uncertain_is_not_a_fact(self):
        t = templates([self.visual])
        a, b = completed(t['A'], 'A', 'error'), completed(t['B'], 'B', 'error')
        for draft in (a, b):
            next(iter(draft['rows'][0]['review']['facts'].values()))['duplicate'] = True
        result = evaluate_pair(t['A'], a, t['B'], b)
        self.assertEqual(result['fact_counts'], {'error': 1})
        self.assertEqual(result['consensus_error_ratio'], 1)
        self.assertEqual(result['duplicate_flags'], {'A': 1, 'B': 1})

    def test_not_evaluable_remains_excluded_and_reported(self):
        t = templates([self.visual])
        result = evaluate_pair(t['A'], completed(t['A'], 'A', 'not_evaluable'),
                               t['B'], completed(t['B'], 'B', 'not_evaluable'))
        self.assertEqual(result['fact_counts']['not_evaluable'], 1)
        self.assertEqual(result['paired_evaluable_images'], 0)
        self.assertIsNone(result['consensus_error_ratio'])

    def test_agreed_OCR_transcriptions_and_disagreements_have_separate_denominators(self):
        t = templates([self.ocr])
        a, b = completed(t['A'], 'A'), completed(t['B'], 'B')
        result = evaluate_pair(t['A'], a, t['B'], b)
        self.assertEqual(result['ocr']['reference_characters'], 4)
        self.assertEqual(result['ocr']['edits'], 1)
        self.assertEqual(result['ocr']['cer'], .25)
        b['rows'][0]['review']['reference_text'] = '不同参考'
        result = evaluate_pair(t['A'], a, t['B'], b)
        self.assertEqual(result['ocr']['agreed_transcriptions'], 0)
        self.assertIsNone(result['ocr']['cer'])
        self.assertEqual(result['ocr']['rows'][0]['status'], 'disagreement')

    def test_empty_OCR_reference_requires_explicit_flag(self):
        t = templates([self.ocr])['A']
        d = completed(t, 'A'); d['rows'][0]['review']['reference_text'] = ''
        with self.assertRaises(ValueError):
            validate_draft(t, d)
        d['rows'][0]['review']['no_visible_text'] = True
        self.assertEqual(len(validate_draft(t, d)), 1)

    def test_development_point_check_does_not_promote_final_test(self):
        rows = [make_row('visual', str(i), '/tmp/source.jpg', f'{i:064x}', self.observation) for i in range(100)]
        t = templates(rows)
        result = evaluate_pair(t['A'], completed(t['A'], 'A'), t['B'], completed(t['B'], 'B'))
        self.assertEqual(result['development_fact_check'], 'passed')
        self.assertEqual(result['quality_gate'], 'insufficient_data')
        self.assertFalse(result['final_test_locked'])

    def test_preparation_preserves_images_and_detects_preview_or_source_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); images = root / 'sources'; images.mkdir()
            image = images / 'neutral.png'; Image.new('RGB', (160, 90), 'red').save(image)
            checksum = fingerprint(image)
            study = root / 'study'; study.mkdir()
            atomic_json(study / 'results.json', [{'image_id': 'id', 'neutral_path': str(image),
                       'neutral_sha256': checksum, 'observation': self.observation}])
            atomic_json(study / 'input_manifest.json', {'rows': []})
            atomic_json(study / 'summary.json', {'status': 'passed', 'artifact_sha256': {
                name: fingerprint(study / name) for name in ('results.json', 'input_manifest.json')}})
            output = root / 'review'; summary = prepare(study, None, output)
            self.assertEqual(summary['human_reviewed_images'], 0)
            self.assertEqual(fingerprint(image), checksum)
            self.assertEqual(len(verify_existing(output)['A']['rows']), 1)
            preview = next((output / 'images').iterdir()); preview.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'artifact_changed'):
                verify_existing(output)
            with self.assertRaisesRegex(ValueError, 'inside_existing_study'):
                prepare(study, None, study / 'nested')


class ReviewMetricTests(unittest.TestCase):
    def test_b_cubed_penalizes_both_merging_and_fragmentation(self):
        identities = ['A', 'A', 'B', 'B']
        perfect = b_cubed(identities, [0, 0, 1, 1]); self.assertEqual(perfect['f1'], 1)
        merged = b_cubed(identities, [0, 0, 0, 0]); self.assertEqual(merged['precision'], .5); self.assertEqual(merged['recall'], 1)
        split = b_cubed(identities, [0, 1, 2, 3]); self.assertEqual(split['precision'], 1); self.assertEqual(split['recall'], .5)
        self.assertEqual(merged['mixed_clusters'], 1); self.assertEqual(split['fragmented_identities'], 2)
        self.assertIsNone(b_cubed([], [])['f1'])
        with self.assertRaises(ValueError):
            b_cubed(['A'], [])

    def test_character_errors_do_not_clip_insertions_or_make_empty_accuracy(self):
        self.assertEqual(character_errors('一', '一二三')['cer'], 2)
        self.assertIsNone(character_errors('', '文字')['cer'])
        self.assertEqual(character_errors('é\r\n舞台', 'e\u0301\n舞台')['edits'], 0)


if __name__ == '__main__':
    unittest.main()
