"""Conditional vision calls, durable budgets and content-bound observations.

Unverified remote weights get a process-private cache namespace. A served model
alias alone cannot authorize evidence reuse after a server/model restart.
"""
from contextlib import closing
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
import uuid

from core_modules.organize.feature_cache import signature
from core_modules.organize.task_artifacts import atomic_json, fingerprint

FIELDS = ('clothing', 'props', 'setting', 'actions', 'visible_text', 'uncertain')
OBSERVATION_VERSION = 'visual-observation-v2'
OBSERVATION_SYSTEM = ('请观察剧照，只描述可见事实，不猜演员姓名或幕次。'
                      '只输出 JSON，字段 clothing、props、setting、actions、visible_text、uncertain 均为字符串列表。')
OBSERVATION_REQUEST = '描述画面可见内容及不确定项。'
_PROCESS = uuid.uuid4().hex


def validate_observation(value):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return False
    return all(isinstance(value[field], list) and len(value[field]) <= 32 and
               all(isinstance(item, str) and len(item) <= 800 for item in value[field])
               for field in FIELDS)


@dataclass(frozen=True)
class VisualPolicy:
    mode: str = 'automatic'
    automatic_fraction: float = 0.30
    max_images: int = 64
    max_output_tokens_per_request: int = 1024
    max_batch_output_tokens: int = 32768
    candidate_margin: float = 0.03
    max_prompt_characters: int = 24000

    def __post_init__(self):
        if self.mode not in ('automatic', 'on_demand'):
            raise ValueError('invalid_visual_mode')
        for name in ('automatic_fraction', 'candidate_margin'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid_visual_policy:' + name)
        for name, upper in (('max_images', 10000), ('max_output_tokens_per_request', 8192),
                            ('max_batch_output_tokens', 1000000), ('max_prompt_characters', 200000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError('invalid_visual_policy:' + name)

    @classmethod
    def from_dict(cls, value=None):
        if value is not None and (not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__)):
            raise ValueError('invalid_visual_policy_fields')
        return cls(**(value or {}))

    def image_limit(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('invalid_visual_image_count')
        return min(self.max_images, count if self.mode == 'on_demand' else math.floor(count * self.automatic_fraction))

    def request_limit(self, count):
        return (2 * self.image_limit(count) if self.mode == 'on_demand' else
                min(2 * self.max_images, math.floor(count * self.automatic_fraction)))


def call_reasons(candidates, parsed=None, scenes=None, margin=0.03):
    """Use retrieval evidence, never filename order or a model confidence score."""
    reasons = []
    visual = [(row['scene_id'], row.get('visual_score')) for row in candidates
              if row.get('retrieval_mode') == 'visual' and
              type(row.get('visual_score')) in (float, int) and math.isfinite(row['visual_score'])]
    visual.sort(key=lambda row: -row[1])
    if len(visual) < 2:
        reasons.append('insufficient_visual_candidates')
    elif visual[0][1] - visual[1][1] <= margin:
        reasons.append('small_visual_candidate_margin')
    if not candidates or not any(row.get('evidence') for row in candidates):
        reasons.append('insufficient_reference_evidence')
    if visual and parsed and scenes:
        people = {name for pair in parsed.get('people', []) for name in pair if name}
        supported = set()
        for scene in scenes:
            roles = (scene.get('entities') or {}).get('roles', [])
            names = {value for role in roles for value in (role.get('actor'), role.get('role')) if value}
            if people.intersection(names):
                supported.add(scene['scene_id'])
        if supported and visual[0][0] not in supported:
            reasons.append('person_visual_conflict')
    return reasons


def evidence_identity(client, max_side):
    verified = getattr(client, 'verified_weights_sha256', None)
    if not isinstance(verified, str) or len(verified) != 64 or any(c not in '0123456789abcdef' for c in verified):
        verified = None
    identity = {'endpoint_sha256': signature(getattr(client, 'base_url', 'unsigned:' + str(id(client)))),
                'served_model': getattr(client, 'model', 'unknown'),
                'declared_model_identity': getattr(client, 'model_identity', None),
                'verified_weights_sha256': verified,
                'remote_weight_verification': 'verified' if verified else 'unverified',
                'prompt_version': OBSERVATION_VERSION, 'max_side': max_side, 'preview_format': 'jpeg-q90-exif-rgb',
                'observation_prompt_sha256': signature([OBSERVATION_SYSTEM, OBSERVATION_REQUEST]),
                'temperature': getattr(client, 'temperature', 0),
                'thinking': getattr(client, 'chat_template_kwargs', {})}
    if not verified:
        identity['private_process'] = _PROCESS
    from importlib.metadata import version
    identity['pillow_version'] = version('Pillow')
    identity['source_sha256'] = {name: fingerprint(Path(__file__).with_name(name)) for name in
                                 ('visual_evidence.py', 'scene_reasoner.py', 'llm_client.py')}
    return identity


class EvidenceCache:
    def __init__(self, directory, identity):
        self.directory = Path(directory) / 'visual-evidence-v1' if directory else None
        # Match the durable JSON form (not tuples before serialization), and
        # detach the signature from mutable caller-owned dictionaries.
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))

    def key(self, image_sha256, kind, prompt=None):
        return signature({'schema': 1, 'identity': self.identity, 'image_sha256': image_sha256,
                          'kind': kind, 'prompt': prompt})

    def path(self, key):
        return self.directory / key[:2] / (key + '.json')

    def get(self, key, validator):
        if self.directory is None:
            return None
        try:
            row = json.loads(self.path(key).read_text(encoding='utf-8'))
            if (row.get('schema') != 1 or row.get('key') != key or row.get('identity') != self.identity or
                    signature(row['value']) != row.get('value_sha256') or not isinstance(row.get('trace'), dict) or
                    not validator(row['value'])):
                return None
            return row
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key, value, trace, validator):
        if not validator(value):
            raise ValueError('invalid_evidence_cannot_be_cached')
        if self.directory is None:
            return
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, {'schema': 1, 'key': key, 'identity': self.identity,
                           'value': value, 'value_sha256': signature(value), 'trace': trace})


class VisualBudget:
    """Reserve calls before HTTP; a crashed call still consumes its reservation."""
    def __init__(self, path, scope, count, policy):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scope, self.count, self.policy = scope, count, policy
        self.limit = policy.image_limit(count)
        with closing(self._connect()) as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS image_allocations (
                    scope TEXT NOT NULL, image TEXT NOT NULL, PRIMARY KEY(scope,image));
                CREATE TABLE IF NOT EXISTS requests (
                    scope TEXT NOT NULL, image TEXT NOT NULL, kind TEXT NOT NULL,
                    output_tokens INTEGER NOT NULL, status TEXT NOT NULL,
                    trace TEXT, PRIMARY KEY(scope,image,kind));
            ''')

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    def allocate(self, image, minimum_requests=2):
        with closing(self._connect()) as connection:
            connection.execute('BEGIN IMMEDIATE')
            if connection.execute('SELECT 1 FROM image_allocations WHERE scope=? AND image=?',
                                  (self.scope, image)).fetchone():
                connection.commit()
                return True
            count = connection.execute('SELECT COUNT(*) FROM image_allocations WHERE scope=?', (self.scope,)).fetchone()[0]
            requests = connection.execute('SELECT COUNT(*) FROM requests WHERE scope=?', (self.scope,)).fetchone()[0]
            if count >= self.limit or requests + minimum_requests > self.policy.request_limit(self.count):
                connection.rollback()
                return False
            connection.execute('INSERT INTO image_allocations VALUES (?,?)', (self.scope, image))
            connection.commit()
            return True

    def reserve(self, image, kind):
        if kind not in ('observation', 'choice'):
            raise ValueError('invalid_visual_request_kind')
        with closing(self._connect()) as connection:
            connection.execute('BEGIN IMMEDIATE')
            if not connection.execute('SELECT 1 FROM image_allocations WHERE scope=? AND image=?',
                                      (self.scope, image)).fetchone():
                connection.rollback()
                return False, 'image_not_allocated'
            if connection.execute('SELECT 1 FROM requests WHERE scope=? AND image=? AND kind=?',
                                  (self.scope, image, kind)).fetchone():
                connection.rollback()
                return False, 'request_already_reserved'
            used = connection.execute('SELECT COALESCE(SUM(output_tokens),0) FROM requests WHERE scope=?', (self.scope,)).fetchone()[0]
            requests = connection.execute('SELECT COUNT(*) FROM requests WHERE scope=?', (self.scope,)).fetchone()[0]
            if requests >= self.policy.request_limit(self.count):
                connection.rollback()
                return False, 'batch_request_budget_exhausted'
            tokens = self.policy.max_output_tokens_per_request
            if used + tokens > self.policy.max_batch_output_tokens:
                connection.rollback()
                return False, 'batch_output_token_budget_exhausted'
            connection.execute('INSERT INTO requests VALUES (?,?,?,?,?,NULL)',
                               (self.scope, image, kind, tokens, 'reserved'))
            connection.commit()
        return True, 'reserved'

    def finish(self, image, kind, trace, valid):
        with closing(self._connect()) as connection:
            connection.execute('UPDATE requests SET status=?,trace=? WHERE scope=? AND image=? AND kind=?',
                               ('valid' if valid else 'failed', json.dumps(trace, ensure_ascii=False, allow_nan=False),
                                self.scope, image, kind))

    def summary(self):
        with closing(self._connect()) as connection:
            images = connection.execute('SELECT COUNT(*) FROM image_allocations WHERE scope=?', (self.scope,)).fetchone()[0]
            rows = connection.execute('SELECT status,COUNT(*),SUM(output_tokens) FROM requests WHERE scope=? GROUP BY status', (self.scope,)).fetchall()
        return {'policy': asdict(self.policy), 'input_images': self.count, 'image_limit': self.limit,
                'allocated_images': images, 'automatic_image_fraction': images / self.count if self.count else 0,
                'reserved_requests': sum(row[1] for row in rows),
                'request_limit': self.policy.request_limit(self.count),
                'automatic_request_ratio': sum(row[1] for row in rows) / self.count if self.count else 0,
                'reserved_max_output_tokens': sum(row[2] for row in rows),
                'requests_by_status': {row[0]: row[1] for row in rows},
                'token_budget_kind': 'hard_maximum_generated_output_tokens; input_usage_reported_separately'}


class VisualEvidenceController:
    def __init__(self, client, cache_dir, budget_path, scope, count, policy=None, max_side=1280):
        if getattr(client, 'max_retries', 0) != 0 or hasattr(client, 'small'):
            raise ValueError('budgeted_visual_client_requires_zero_retries_and_no_cascade')
        self.policy = policy if isinstance(policy, VisualPolicy) else VisualPolicy.from_dict(policy)
        self.identity = evidence_identity(client, max_side)
        self.identity['max_output_tokens_per_request'] = self.policy.max_output_tokens_per_request
        self.cache = EvidenceCache(cache_dir, self.identity)
        # Include policy and stable request configuration; transient cache identity
        # is deliberately excluded so request reservations survive process restart.
        identity = {key: value for key, value in self.identity.items() if key != 'private_process'}
        self.budget = VisualBudget(budget_path, signature([scope, count, asdict(self.policy), identity]), count, self.policy)

    def observe(self, image_path, reasons, observer):
        image_sha = fingerprint(image_path)
        key = self.cache.key(image_sha, 'observation')
        cached = self.cache.get(key, validate_observation)
        trace = {'source_image_sha256': image_sha, 'cache_key': key,
                 'identity': self.identity, 'call_reasons': reasons, 'called': False}
        if cached:
            trace.update(cache_hit=True, original_trace=cached['trace'])
            return self._evidence(image_path, image_sha, cached['value']), trace
        if not reasons and self.policy.mode == 'automatic':
            return None, dict(trace, skipped='sufficient_candidate_margin', cache_hit=False)
        if not self.budget.allocate(image_sha, minimum_requests=2):
            return None, dict(trace, skipped='image_budget_exhausted', cache_hit=False)
        reserved, reason = self.budget.reserve(image_sha, 'observation')
        if not reserved:
            return None, dict(trace, skipped=reason, cache_hit=False)
        from core_modules.organize.llm_client import LLMError
        try:
            observation, result_trace = observer(image_path, self.policy.max_output_tokens_per_request)
        except (LLMError, OSError, ValueError, TypeError, KeyError) as exc:
            observation, result_trace = None, {'error': type(exc).__name__}
        trace.update(result_trace, called=True, cache_hit=False)
        try:
            value = json.loads(observation['quote']) if observation else None
        except (ValueError, KeyError, TypeError):
            value = None
        valid = validate_observation(value) and fingerprint(image_path) == image_sha
        self.budget.finish(image_sha, 'observation', trace, valid)
        if not valid:
            return None, dict(trace, error=trace.get('error', 'invalid_visual_schema'))
        self.cache.put(key, value, trace, validate_observation)
        return self._evidence(image_path, image_sha, value), trace

    @staticmethod
    def _evidence(path, image_sha, value):
        return {'id': 'observation:' + image_sha[:24], 'type': 'observation',
                'quote': json.dumps(value, ensure_ascii=False), 'ref': Path(path).name,
                'source_image_sha256': image_sha, 'provenance': 'direct_image_observation'}

    def choice(self, image_path, messages, candidate_ids, evidence_ids, asker, validator):
        image_sha = fingerprint(image_path)
        prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        key = self.cache.key(image_sha, 'choice', prompt)
        valid = lambda value: validator(value, candidate_ids, evidence_ids)[0]
        cached = self.cache.get(key, valid)
        if cached:
            return cached['value'], {'called': False, 'cache_hit': True, 'cache_key': key,
                                     'original_trace': cached['trace']}
        if len(prompt) > self.policy.max_prompt_characters:
            return None, {'called': False, 'skipped': 'prompt_character_budget_exhausted'}
        if not self.budget.allocate(image_sha, minimum_requests=1):
            return None, {'called': False, 'skipped': 'image_budget_exhausted'}
        reserved, reason = self.budget.reserve(image_sha, 'choice')
        if not reserved:
            return None, {'called': False, 'skipped': reason}
        from core_modules.organize.llm_client import LLMError
        try:
            choice, trace = asker(messages, candidate_ids, evidence_ids, self.policy.max_output_tokens_per_request)
        except (LLMError, OSError, ValueError, TypeError, KeyError) as exc:
            choice, trace = None, {'error': type(exc).__name__}
        trace = dict(trace, called=True, cache_hit=False, cache_key=key)
        accepted = valid(choice) and fingerprint(image_path) == image_sha
        self.budget.finish(image_sha, 'choice', trace, accepted)
        if accepted:
            self.cache.put(key, choice, trace, valid)
        return choice if accepted else None, trace
