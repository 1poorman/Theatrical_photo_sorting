"""Source provenance, scene mapping, query isolation and paired-input integrity."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from core_modules.organize.identity_registry import IdentityRegistry, stable_id
from sourced_scene_candidates import (apply_candidate_bundle, candidate_files_unchanged,
                                     digest, extract_html_text, validate_sourced_manifest)


class SourcedCandidateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / 'page.html'
        self.snapshot.write_text('<html><script>QUERY_SECRET</script>'
                                 '<h2>第一幕第一景：宫殿</h2><p>大厅中举行宴会。</p>'
                                 '<h2>第一幕第二景：庭院</h2><p>庭院中分别。</p></html>', encoding='utf-8')
        self.text = extract_html_text(self.snapshot.read_bytes())
        self.registry = IdentityRegistry({'schema_version': 1,
                                         'plays': [{'play_id': 'play-a', 'title': '歌剧《甲》'}],
                                         'identities': []})
        self.base = {'labels_sha256': 'labels', 'registry_sha256': 'registry', 'test_locked': False,
                     'rows': [{'file_id': 'photo-a', 'play_id': 'play-a', 'scene_id': 'scene-a',
                               'scene_label': '第1幕1场', 'caption': 'QUERY_SECRET',
                               'people': [{'actor_name': 'PRIVATE_ACTOR'}]}],
                     'excluded': [{'file_id': 'photo-b'}],
                     'scenes': [{'play_id': 'play-a', 'scene_id': 'scene-a', 'label': '第1幕1场',
                                 'text': '歌剧《甲》 第1幕1场', 'text_source': 'scene_label_only'},
                                {'play_id': 'play-a', 'scene_id': 'scene-b', 'label': '第1幕2场',
                                 'text': '歌剧《甲》 第1幕2场', 'text_source': 'scene_label_only'}]}
        self.base_path = self.root / 'base.json'
        self.base_path.write_text(json.dumps(self.base, ensure_ascii=False))
        self.bundle_path = self.root / 'bundle.json'
        section = self.text[:self.text.index('第一幕第二景')].rstrip()
        evidence = self.span('大厅中举行宴会。')
        self.bundle = {'schema_version': 1, 'version': 'fixture-v1',
                       'base_manifest_sha256': digest(self.base_path), 'pool_mode': 'same_pool',
                       'sources': [{'source_id': 'source-a', 'url': 'https://www.chncpa.org/plot',
                                    'snapshot': 'page.html', 'snapshot_sha256': digest(self.snapshot),
                                    'kind': 'official_plot', 'encoding': 'utf-8'}],
                       'replacements': [{'play_id': 'play-a', 'scene_id': 'scene-a', 'label': '第1幕1场',
                                         'mapping': {**self.span(section), 'method': 'explicit_source_numbering',
                                                     'source_heading': '第一幕第一景'},
                                         'evidence': [evidence]}], 'additions': []}

    def span(self, quote):
        start = self.text.index(quote)
        return {'source_id': 'source-a', 'start': start, 'end': start + len(quote), 'quote': quote}

    def run_bundle(self, bundle=None):
        self.bundle_path.write_text(json.dumps(bundle or self.bundle, ensure_ascii=False))
        return apply_candidate_bundle(self.base, self.base_path, self.bundle_path, self.registry)

    def test_exact_excerpt_preserves_queries_pool_and_unmapped_candidates(self):
        result = self.run_bundle()
        self.assertEqual(result['rows'], self.base['rows'])
        self.assertEqual(result['excluded'], self.base['excluded'])
        self.assertFalse(result['test_locked'])
        self.assertEqual(result['scenes'][1], self.base['scenes'][1])
        self.assertEqual(result['scenes'][0]['text'], '歌剧《甲》 第1幕1场 大厅中举行宴会。')
        self.assertNotIn('QUERY_SECRET', result['scenes'][0]['text'])
        self.assertNotIn('PRIVATE_ACTOR', result['scenes'][0]['text'])
        self.assertTrue(candidate_files_unchanged(result))
        validate_sourced_manifest(result, self.registry)

    def test_script_comment_and_image_attributes_are_excluded(self):
        body = b'<p>plot</p><script>secret</script><!--hidden--><img alt="caption"/>' \
               b'<script/><p>ending</p>'
        self.assertEqual(extract_html_text(body), 'plot\nending\n')

    def test_query_field_injection_and_nonexplicit_mapping_rejected(self):
        bundle = deepcopy(self.bundle)
        bundle['replacements'][0]['caption'] = 'QUERY_SECRET'
        with self.assertRaisesRegex(ValueError, 'unexpected_candidate_document_fields'):
            self.run_bundle(bundle)
        bundle = deepcopy(self.bundle)
        bundle['replacements'][0]['mapping']['method'] = 'guessed_from_photo'
        with self.assertRaisesRegex(ValueError, 'candidate_mapping_is_not_explicit'):
            self.run_bundle(bundle)

    def test_wrong_scene_number_title_and_unknown_id_rejected(self):
        for update, reason in [({'scene_id': 'unknown'}, 'candidate_vocabulary_changed'),
                               ({'label': '第2幕1场'}, 'candidate_vocabulary_changed')]:
            with self.subTest(update=update):
                bundle = deepcopy(self.bundle)
                bundle['replacements'][0].update(update)
                with self.assertRaisesRegex(ValueError, reason):
                    self.run_bundle(bundle)
        bundle = deepcopy(self.bundle)
        bundle['replacements'][0]['mapping'] = {**self.span('第一幕第二景：庭院'),
            'method': 'explicit_source_numbering', 'source_heading': '第一幕第二景'}
        with self.assertRaisesRegex(ValueError, 'candidate_scene_number_or_title_mismatch'):
            self.run_bundle(bundle)

    def test_invented_quotes_script_text_and_outside_sections_rejected(self):
        bundle = deepcopy(self.bundle)
        bundle['replacements'][0]['evidence'][0]['quote'] = '发明的剧情。'
        with self.assertRaisesRegex(ValueError, 'candidate_quote_does_not_match_snapshot'):
            self.run_bundle(bundle)
        bundle = deepcopy(self.bundle)
        bundle['replacements'][0]['evidence'] = [self.span('庭院中分别。')]
        with self.assertRaisesRegex(ValueError, 'candidate_quote_outside_mapped_section'):
            self.run_bundle(bundle)
        self.assertNotIn('QUERY_SECRET', self.text)

    def test_same_size_same_mtime_source_rewrite_rejected(self):
        import os
        result = self.run_bundle()
        before = self.snapshot.stat()
        body = self.snapshot.read_bytes().replace('宴'.encode(), '戏'.encode())
        self.snapshot.write_bytes(body)
        os.utime(self.snapshot, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertFalse(candidate_files_unchanged(result))
        with self.assertRaisesRegex(ValueError, 'candidate_source_snapshot_changed'):
            self.run_bundle()

    def test_base_rewrite_and_snapshot_path_escape_rejected(self):
        self.base_path.write_text(json.dumps({**self.base, 'test_locked': True}))
        with self.assertRaisesRegex(ValueError, 'candidate_base_manifest_changed'):
            self.run_bundle()
        self.base_path.write_text(json.dumps(self.base, ensure_ascii=False))
        bundle = deepcopy(self.bundle)
        bundle['sources'][0]['snapshot'] = '../other.html'
        with self.assertRaisesRegex(ValueError, 'source_snapshot_must_be_inside_bundle'):
            self.run_bundle(bundle)

    def test_duplicate_rows_and_numbering_only_text_rejected(self):
        bundle = deepcopy(self.bundle)
        bundle['replacements'].append(deepcopy(bundle['replacements'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate_candidate_replacement'):
            self.run_bundle(bundle)
        bundle = deepcopy(self.bundle)
        bundle['sources'][0]['kind'] = 'numbering_reference'
        with self.assertRaisesRegex(ValueError, 'numbering_reference_is_not_plot_text'):
            self.run_bundle(bundle)

    def test_expansion_keeps_original_pool_and_is_explicit(self):
        bundle = deepcopy(self.bundle)
        label = '第1幕3场'
        # A third scene with an explicit source heading is an unobserved distractor.
        self.snapshot.write_text(self.snapshot.read_text() + '<h2>第一幕第三景：码头</h2><p>船在码头。</p>')
        self.text = extract_html_text(self.snapshot.read_bytes())
        bundle['sources'][0]['snapshot_sha256'] = digest(self.snapshot)
        section = self.text[self.text.index('第一幕第三景'):].rstrip()
        bundle['additions'] = [{'play_id': 'play-a', 'scene_id': stable_id('scene', 'play-a:' + label),
                                'label': label, 'evidence': [self.span('船在码头。')],
                                'mapping': {**self.span(section), 'method': 'explicit_source_numbering',
                                            'source_heading': '第一幕第三景'}}]
        with self.assertRaisesRegex(ValueError, 'same_pool_cannot_add_scenes'):
            self.run_bundle(bundle)
        bundle['pool_mode'] = 'expanded_pool'
        result = self.run_bundle(bundle)
        self.assertEqual(len(result['scenes']), 3)
        self.assertEqual(result['scenes'][1], self.base['scenes'][1])
        self.assertEqual(result['rows'], self.base['rows'])
        self.assertEqual(len(result['candidate_texts']['added_scene_ids']), 1)

    def test_prepared_query_or_text_rewrite_cannot_be_used_for_paired_run(self):
        result = self.run_bundle()
        for area in ('rows', 'scenes'):
            with self.subTest(area=area):
                changed = deepcopy(result)
                changed[area][0]['text' if area == 'scenes' else 'scene_id'] = 'rewritten'
                with self.assertRaisesRegex(ValueError, 'prepared_candidate_manifest_changed'):
                    validate_sourced_manifest(changed, self.registry)
        self.bundle_path.write_text(self.bundle_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'prepared_candidate_sources_changed'):
            validate_sourced_manifest(result, self.registry)


if __name__ == '__main__':
    unittest.main()
