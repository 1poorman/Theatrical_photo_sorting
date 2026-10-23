"""Vision budgets, crash reservations, schema/cache integrity and real image audit."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from core_modules.organize.llm_client import LLMError, OpenAICompatClient, image_content
from core_modules.organize.scene_reasoner import SceneReasoner, validate_choice
from core_modules.organize.visual_evidence import (
    FIELDS, VisualBudget, VisualEvidenceController, VisualPolicy,
    call_reasons, evidence_identity, validate_observation,
)


class VisualEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = []
        for index in range(10):
            path = self.root / f'{index}.png'
            Image.new('RGB', (10, 20), (index * 10, 0, 0)).save(path)
            self.paths.append(path)
        self.client = SimpleNamespace(base_url='http://private:9/v1', model='served-alias',
                                      model_identity='declared-model', temperature=0,
                                      chat_template_kwargs={}, verified_weights_sha256='a' * 64)
        self.calls = []

    def observer(self, path, tokens):
        self.calls.append(('observation', path, tokens))
        return {'quote': json.dumps({field: ['visible'] for field in FIELDS})}, {'model': 'served-alias'}

    def asker(self, messages, candidates, evidence, tokens):
        self.calls.append(('choice', messages, tokens))
        return {'scene_id': candidates[0], 'evidence_refs': [evidence[0]]}, {'model': 'served-alias'}

    def controller(self, policy=None, scope='task', client=None):
        return VisualEvidenceController(client or self.client, self.root / 'cache', self.root / 'budget.sqlite3',
                                        scope, 10, policy or VisualPolicy())

    def test_default_counts_both_observation_and_choice_within_thirty_percent(self):
        controller = self.controller()
        for path in self.paths:
            evidence, _ = controller.observe(path, ['ambiguous'], self.observer)
            if evidence:
                controller.choice(path, [{'role': 'user', 'content': str(path)}],
                                  ['scene'], [evidence['id']], self.asker, validate_choice)
        summary = controller.budget.summary()
        self.assertEqual(len(self.calls), 2)
        self.assertLessEqual(summary['automatic_request_ratio'], .30)
        self.assertLessEqual(summary['automatic_image_fraction'], .30)
        self.assertEqual(summary['request_limit'], 3)

    def test_budget_reservations_survive_recreated_controller_and_failed_request(self):
        controller = self.controller()
        def failing(path, tokens):
            self.calls.append(('failed', path, tokens))
            raise LLMError('network unavailable')
        evidence, trace = controller.observe(self.paths[0], ['ambiguous'], failing)
        self.assertIsNone(evidence)
        self.assertEqual(trace['error'], 'LLMError')
        recreated = self.controller()
        evidence, trace = recreated.observe(self.paths[0], ['ambiguous'], self.observer)
        self.assertIsNone(evidence)
        self.assertEqual(trace['skipped'], 'request_already_reserved')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(recreated.budget.summary()['requests_by_status'], {'failed': 1})

    def test_output_token_reservation_is_a_hard_batch_bound(self):
        policy = VisualPolicy(mode='on_demand', max_output_tokens_per_request=100,
                              max_batch_output_tokens=150)
        controller = self.controller(policy)
        evidence, _ = controller.observe(self.paths[0], ['ambiguous'], self.observer)
        self.assertIsNotNone(evidence)
        choice, trace = controller.choice(self.paths[0], [], ['scene'], [evidence['id']], self.asker, validate_choice)
        self.assertIsNone(choice)
        self.assertEqual(trace['skipped'], 'batch_output_token_budget_exhausted')
        self.assertEqual(controller.budget.summary()['reserved_max_output_tokens'], 100)

    def test_concurrent_request_reservations_never_exceed_budget(self):
        budget = VisualBudget(self.root / 'race.sqlite3', 'race', 100, VisualPolicy())
        def request(index):
            if not budget.allocate(str(index), minimum_requests=1):
                return False
            return budget.reserve(str(index), 'observation')[0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(request, range(100)))
        self.assertEqual(sum(results), 30)
        self.assertEqual(budget.summary()['reserved_requests'], 30)

    def test_verified_cache_reuses_identical_bytes_after_rename_without_http(self):
        first = self.controller(VisualPolicy(mode='on_demand'))
        evidence, _ = first.observe(self.paths[0], ['ambiguous'], self.observer)
        copy = self.root / 'renamed.png'
        copy.write_bytes(self.paths[0].read_bytes())
        second = self.controller(VisualPolicy(mode='on_demand'), scope='different-task')
        cached, trace = second.observe(copy, ['ambiguous'], self.observer)
        self.assertTrue(trace['cache_hit'])
        self.assertEqual(evidence['id'], cached['id'])
        self.assertEqual(cached['ref'], 'renamed.png')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second.budget.summary()['reserved_requests'], 0)

    def test_changed_image_or_weights_invalidates_cache(self):
        policy = VisualPolicy(mode='on_demand')
        self.controller(policy).observe(self.paths[0], ['ambiguous'], self.observer)
        Image.new('RGB', (10, 20), 'blue').save(self.paths[0])
        self.controller(policy, scope='second').observe(self.paths[0], ['ambiguous'], self.observer)
        self.client.verified_weights_sha256 = 'b' * 64
        self.controller(policy, scope='third').observe(self.paths[0], ['ambiguous'], self.observer)
        self.assertEqual(len(self.calls), 3)

    def test_unverified_weights_do_not_share_cache_across_process_namespace(self):
        self.client.verified_weights_sha256 = None
        policy = VisualPolicy(mode='on_demand')
        first = self.controller(policy)
        first.observe(self.paths[0], ['ambiguous'], self.observer)
        with patch('core_modules.organize.visual_evidence._PROCESS', 'another-process'):
            second = self.controller(policy, scope='different-task')
            second.observe(self.paths[0], ['ambiguous'], self.observer)
        self.assertNotEqual(first.identity, second.identity)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(first.identity['remote_weight_verification'], 'unverified')
        self.assertIn('private_process', evidence_identity(self.client, 1280))

    def test_schema_rejects_identity_fields_and_cache_corruption(self):
        observation = {field: [] for field in FIELDS}
        self.assertTrue(validate_observation(observation))
        self.assertFalse(validate_observation(dict(observation, actor_name='guessed')))
        self.assertFalse(validate_observation(dict(observation, actions=['x' * 801])))
        controller = self.controller(VisualPolicy(mode='on_demand'))
        _, trace = controller.observe(self.paths[0], ['ambiguous'], self.observer)
        controller.cache.path(trace['cache_key']).write_text('{invalid')
        recreated = self.controller(VisualPolicy(mode='on_demand'), scope='other')
        _, trace = recreated.observe(self.paths[0], ['ambiguous'], self.observer)
        self.assertFalse(trace['cache_hit'])
        self.assertEqual(len(self.calls), 2)

    def test_choice_cache_binds_candidates_and_rejects_outside_references(self):
        controller = self.controller(VisualPolicy(mode='on_demand'))
        evidence, _ = controller.observe(self.paths[0], ['ambiguous'], self.observer)
        messages = [{'role': 'user', 'content': 'fixed candidates'}]
        good, _ = controller.choice(self.paths[0], messages, ['scene'], [evidence['id']], self.asker, validate_choice)
        self.assertIsNotNone(good)
        good, trace = controller.choice(self.paths[0], messages, ['scene'], [evidence['id']], self.asker, validate_choice)
        self.assertTrue(trace['cache_hit'])
        bad, trace = controller.choice(self.paths[0], messages, ['other'], [evidence['id']], self.asker, validate_choice)
        self.assertIsNone(bad)
        self.assertEqual(trace['skipped'], 'request_already_reserved')
        self.assertEqual(len(self.calls), 2)

    def test_no_call_for_clear_visual_candidates_and_manual_labels(self):
        candidates = [{'scene_id': 'a', 'visual_score': .8, 'retrieval_mode': 'visual', 'evidence': [{}]},
                      {'scene_id': 'b', 'visual_score': .2, 'retrieval_mode': 'visual', 'evidence': [{}]}]
        self.assertEqual(call_reasons(candidates), [])
        controller = self.controller(VisualPolicy(mode='on_demand'))
        knowledge = {'scenes': [{'scene_id': 'a', 'label': 'A', 'text': 'A'},
                                {'scene_id': 'b', 'label': 'B', 'text': 'B'}]}
        row = SceneReasoner(knowledge, vision_client=self.client,
                            visual_controller=controller).reason(str(self.paths[0]), manual_label='a')
        self.assertEqual(row['decision'], 'human')
        self.assertIsNone(row['confidence'])
        self.assertEqual(controller.budget.summary()['reserved_requests'], 0)

    def test_bad_policy_and_prompt_budget_are_rejected(self):
        for value in ({'unknown': True}, {'mode': 'invalid'}, {'automatic_fraction': float('nan')},
                      {'max_images': True}, {'candidate_margin': -1}):
            with self.assertRaises(ValueError):
                VisualPolicy.from_dict(value)
        controller = self.controller(VisualPolicy(mode='on_demand', max_prompt_characters=10))
        evidence, _ = controller.observe(self.paths[0], ['ambiguous'], self.observer)
        choice, trace = controller.choice(self.paths[0], [{'content': 'a long prompt'}],
                                          ['scene'], [evidence['id']], self.asker, validate_choice)
        self.assertIsNone(choice)
        self.assertEqual(trace['skipped'], 'prompt_character_budget_exhausted')

    def test_http_trace_audits_image_bytes_tokens_and_masks_endpoint(self):
        requests = []
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': '{"valid":true}'}, 'finish_reason': 'stop'}],
                                          'usage': {'prompt_tokens': 90, 'completion_tokens': 7, 'total_tokens': 97}}).encode())
        client = OpenAICompatClient('http://username:secret@host:9/v1?token=private', 'secret', 'alias',
                                    opener=opener, max_retries=0)
        result = client.chat([{'role': 'user', 'content': [image_content(self.paths[0])]}], max_tokens=99)
        self.assertEqual(result['trace']['max_output_tokens'], 99)
        self.assertEqual(result['trace']['usage']['total_tokens'], 97)
        self.assertEqual(len(result['trace']['image_inputs']), 1)
        self.assertNotIn('secret', json.dumps(result['trace']))
        self.assertNotIn('private', json.dumps(result['trace']))
        self.assertEqual(requests[0]['max_tokens'], 99)

    def test_budgeted_client_cannot_hide_http_retries(self):
        self.client.max_retries = 2
        with self.assertRaisesRegex(ValueError, 'zero_retries'):
            self.controller()

    def test_unverified_task_client_contract_cannot_resume_after_process_restart(self):
        from core_modules.organize.task_versions import client_identity
        client = OpenAICompatClient('http://service/v1', 'private-key', 'alias', max_retries=0)
        first = client_identity(client)
        with patch('core_modules.organize.task_versions._PROCESS', 'restarted'):
            self.assertNotEqual(first, client_identity(client))
        client.verified_weights_sha256 = 'invalid'
        self.assertFalse(client_identity(client)['remote_weights_verified'])
        client.verified_weights_sha256 = 'a' * 64
        first = client_identity(client)
        with patch('core_modules.organize.task_versions._PROCESS', 'restarted'):
            self.assertEqual(first, client_identity(client))
        self.assertNotIn('private-key', json.dumps(first))


if __name__ == '__main__':
    unittest.main()
