# -*- coding: utf-8 -*-
"""scene_reasoner - 候选召回 + 本地模型闭集判定 + 防幻觉校验。

设计原则：
  - 语言模型只能从候选 ``scene_id`` 或 ``unknown`` 中选择，不得创造幕次。
  - 置信度由可解释的证据权重累加，无有效证据时不超过 ``MAX_NO_EVIDENCE``。
  - 小模型输出非法 JSON/越界引用时升级大模型；仍失败则降级为 ``unknown`` 转人工。
"""
import json
import os
import re
import math

from core_modules.organize.filename_parser import parse_filename
from core_modules.organize.multimodal_evidence import retrieve_evidence

PROMPT_VERSION = 'scene-v1'
DEFAULT_WEIGHTS = {'rule': 0.35, 'person': 0.20, 'time': 0.15,
                   'visual': 0.20, 'language': 0.10}
HIGH_CONFIDENCE = 0.75
MAX_NO_EVIDENCE = 0.49


def _parse_json(content):
    """从模型输出中容错提取 JSON 对象。"""
    if not isinstance(content, str) or not content:
        return None
    text = content.strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_choice(choice, candidate_ids, evidence_ids):
    """校验模型输出：场景闭集、引用存在、置信度范围。"""
    if not isinstance(choice, dict):
        return False, 'not_a_dict'
    scene_id = choice.get('scene_id')
    if scene_id not in candidate_ids and scene_id != 'unknown':
        return False, f'scene_id 越界: {scene_id}'
    refs = choice.get('evidence_refs', [])
    if not isinstance(refs, list):
        return False, 'evidence_refs 非列表'
    for ref in refs:
        if ref not in evidence_ids:
            return False, f'引用不存在: {ref}'
    conf = choice.get('confidence')
    if conf is not None:
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
            return False, f'置信度非法: {conf}'
    return True, ''


def _parse_local(image_path):
    """兼容原始与整理完成两类命名，尽量解析出场景段。"""
    name = os.path.basename(image_path)
    done = '整理完成' in image_path or bool(re.match(r'^\d{1,3}-', name))
    return parse_filename(name, done_style=done)


def _filename_scene_match(parsed, scene):
    text = parsed.get('scene') or ''
    if not text:
        return False
    return any(term and term in text for term in (scene.get('title'), scene.get('label')))


def _person_match(parsed, scene):
    entities = scene.get('entities') or {}
    known = set()
    for item in entities.get('roles', []):
        known.add(item.get('actor'))
        known.add(item.get('role'))
    for actor, role in parsed.get('people', []):
        if actor in known or role in known:
            return True
    return False


def _build_prompt(image_path, parsed, candidates, evidences):
    candidate_lines = []
    for cand in candidates:
        candidate_lines.append({
            'scene_id': cand['scene_id'], 'label': cand['label'],
            'score': cand['score'],
            'evidence_ids': [
                e['id'] for e in evidences
                if e.get('source_ref') in {
                    item.get('source_ref') for item in cand.get('evidence', [])
                }
            ],
        })
    evidence_lines = [{'id': e['id'], 'type': e['type'],
                       'quote': e.get('quote', '')} for e in evidences]
    system = (
        '你是戏曲/话剧剧照整理助手。只能从给定候选 scene_id 中选择，'
        '或返回 "unknown"。禁止创造候选之外的幕次。'
        '只输出 JSON：{"scene_id": "...", "confidence": 0-1, '
        '"reason": "...", "evidence_refs": ["证据id", ...]}。'
        'evidence_refs 必须来自输入证据 id。')
    user = json.dumps({
        'image': os.path.basename(image_path),
        'filename_parsed': {'scene': parsed.get('scene'), 'event': parsed.get('event'),
                            'people': parsed.get('people', [])},
        'candidates': candidate_lines,
        'evidences': evidence_lines,
    }, ensure_ascii=False)
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': user}]


class SceneReasoner:
    def __init__(self, knowledge, client=None, weights=None, embedder=None,
                 top_k=5, use_rule_shortcut=True, vision_client=None, visual_controller=None,
                 boundary_check=None):
        self.knowledge = knowledge
        self.client = client
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.embedder = embedder
        self.top_k = top_k
        self.use_rule_shortcut = use_rule_shortcut
        self.vision_client = vision_client
        self.visual_controller = visual_controller
        self.boundary_check = boundary_check
        self.ocr_provider = None
        self.ocr_images = set()

    # ---------- 证据与候选 ----------

    def _collect(self, image_path):
        parsed = _parse_local(image_path)
        retrieved = retrieve_evidence(image_path, self.knowledge,
                                      top_k=self.top_k, embedder=self.embedder)
        candidates = retrieved['candidates']
        evidences = []
        if parsed.get('scene') or parsed.get('people'):
            quote = '、'.join([parsed.get('scene') or ''] +
                              [f'{a}饰{r}' for a, r in parsed.get('people', [])]).strip('、')
            evidences.append({'id': f'filename:{os.path.basename(image_path)}',
                              'type': 'filename', 'quote': quote,
                              'ref': os.path.basename(image_path)})
        for cand in candidates:
            scene = self._scene(cand['scene_id'])
            src = scene.get('source_ref')
            if not src:
                continue
            ev_id = f'knowledge:{src}'
            if ev_id not in [e['id'] for e in evidences]:
                evidences.append({'id': ev_id, 'type': 'knowledge',
                                  'source_ref': src, 'quote': scene.get('text') or scene.get('label'),
                                  'score': cand.get('score')})
        return parsed, candidates, evidences

    def _rule_matches(self, parsed, candidates):
        return [c for c in candidates
                if _filename_scene_match(parsed, self._scene(c['scene_id']))]

    def _scene(self, scene_id):
        for scene in self.knowledge.get('scenes', []):
            if scene['scene_id'] == scene_id:
                return scene
        return {}

    # ---------- 打分 ----------

    def _score(self, scene, parsed, candidate, llm_pick):
        matched, detail = {}, []
        if _filename_scene_match(parsed, scene):
            matched['rule'] = 1.0
            detail.append({'type': 'rule', 'quote': scene.get('label')})
        if _person_match(parsed, scene):
            matched['person'] = 1.0
            detail.append({'type': 'person', 'quote': '演员/角色匹配'})
        visual = candidate.get('visual_score')
        if candidate.get('retrieval_mode') == 'visual' and isinstance(visual, (int, float)) \
                and math.isfinite(visual) and visual > 0:
            matched['visual'] = min(1.0, visual)
        if llm_pick and llm_pick == scene['scene_id']:
            matched['language'] = 1.0
        conf = sum(self.weights[k] * v for k, v in matched.items())
        return min(0.99, round(conf, 4)), matched, detail

    # ---------- 模型调用 ----------

    def _observe(self, image_path, max_tokens=None):
        from core_modules.organize.llm_client import image_content, LLMError
        try:
            content = image_content(image_path)
            if self.boundary_check:
                self.boundary_check()
            from core_modules.organize.visual_evidence import OBSERVATION_SYSTEM, OBSERVATION_REQUEST, OBSERVATION_VERSION
            messages = [
                {'role': 'system', 'content': OBSERVATION_SYSTEM},
                {'role': 'user', 'content': [{'type': 'text', 'text': OBSERVATION_REQUEST}, content]},
            ]
            result = (self.vision_client.chat(messages, max_tokens=max_tokens)
                      if max_tokens is not None else self.vision_client.chat(messages))
            observation = _parse_json(result['content'])
            from core_modules.organize.visual_evidence import FIELDS, validate_observation
            fields = FIELDS
            if not validate_observation(observation):
                return None, {'error': 'invalid_visual_schema', **result.get('trace', {})}
            trace = dict(result.get('trace', {}))
            trace['model_identity'] = getattr(self.vision_client, 'model_identity',
                                              getattr(self.vision_client, 'model', 'unknown'))
            trace['model_identity_source'] = getattr(self.vision_client, 'model_identity_source', 'configuration')
            trace['prompt_version'] = OBSERVATION_VERSION
            return {'id': 'observation:' + os.path.basename(image_path), 'type': 'observation',
                    'quote': json.dumps({k: observation[k] for k in fields}, ensure_ascii=False),
                    'ref': os.path.basename(image_path)}, trace
        except (LLMError, OSError, ValueError, KeyError, TypeError) as exc:
            return None, {'error': type(exc).__name__}

    def _ask(self, messages, candidate_ids, evidence_ids, max_tokens=None):
        if self.client is None:
            return None, {}
        if self.boundary_check:
            self.boundary_check()
        result = (self.client.chat(messages, max_tokens=max_tokens)
                  if max_tokens is not None else self.client.chat(messages))
        choice = _parse_json(result['content'])
        ok, err = validate_choice(choice, candidate_ids, evidence_ids)
        if ok:
            return choice, result['trace']
        big = getattr(self.client, 'big', None)
        if big is not None and getattr(self.client, 'cascade_enabled', True) and max_tokens is None:
            result = big.chat(messages)
            choice = _parse_json(result['content'])
            ok, err = validate_choice(choice, candidate_ids, evidence_ids)
            if ok:
                result['trace']['cascade_upgraded_from'] = getattr(
                    getattr(self.client, 'small', None), 'model', 'small')
                return choice, result['trace']
        return None, {'rejected': err, **result.get('trace', {})}

    # ---------- 主流程 ----------

    def reason(self, image_path, manual_label=None):
        parsed, candidates, evidences = self._collect(image_path)
        candidate_ids = [c['scene_id'] for c in candidates] + ['unknown']
        evidence_ids = [e['id'] for e in evidences]
        rule_matches = self._rule_matches(parsed, candidates)

        decision = 'rule'
        model_trace = {'prompt_version': PROMPT_VERSION, 'weights': self.weights}
        if self.embedder is not None:
            model_trace['retrieval_model_signature'] = getattr(self.embedder, 'model_signature', 'unknown')
            model_trace['retrieval_errors'] = getattr(self.embedder, 'diagnostics', {}).get(image_path, [])
        llm_pick = None
        if self.ocr_provider is not None and image_path in self.ocr_images and not manual_label:
            if self.boundary_check:
                self.boundary_check()
            ocr, trace = self.ocr_provider.observe(image_path, requested=True)
            model_trace['ocr'] = trace
            if ocr:
                evidences.append(ocr)
                evidence_ids.append(ocr['id'])

        if manual_label:
            decision = 'human'
            chosen = next((c for c in candidates if c['label'] == manual_label
                           or c['scene_id'] == manual_label), None)
            chosen_id = chosen['scene_id'] if chosen else 'unknown'
        elif self.use_rule_shortcut and len(rule_matches) == 1:
            chosen_id = rule_matches[0]['scene_id']
        else:
            chosen_id = 'unknown'
            observation = None
            if self.vision_client is not None:
                if self.visual_controller is not None:
                    from core_modules.organize.visual_evidence import call_reasons
                    reasons = call_reasons(candidates, parsed, self.knowledge.get('scenes'),
                                           self.visual_controller.policy.candidate_margin)
                    observation, trace = self.visual_controller.observe(image_path, reasons, self._observe)
                else:
                    observation, trace = self._observe(image_path)
                model_trace['vision'] = trace
                if observation:
                    evidences.append(observation)
                    evidence_ids.append(observation['id'])
            if self.client is not None and (self.visual_controller is None or observation):
                decision = 'model'
                messages = _build_prompt(image_path, parsed, candidates, evidences)
                if self.visual_controller is not None:
                    choice, trace = self.visual_controller.choice(image_path, messages, candidate_ids,
                                                                 evidence_ids, self._ask, validate_choice)
                    model_trace['visual_choice'] = trace
                else:
                    choice, trace = self._ask(messages, candidate_ids, evidence_ids)
                model_trace.update(trace or {})
                if choice:
                    model_trace['choice'] = choice
                if choice and choice.get('scene_id') not in (None, 'unknown'):
                    llm_pick = choice['scene_id']
                    chosen_id = llm_pick
                    decision = 'model'
                elif choice and choice.get('scene_id') == 'unknown':
                    decision = 'model'
            elif self.visual_controller is not None and self.vision_client is None:
                model_trace['vision'] = {'called': False, 'skipped': 'vision_client_unavailable'}

        # 逐候选打分
        scored = []
        for cand in candidates:
            scene = self._scene(cand['scene_id'])
            conf, matched, detail = self._score(scene, parsed, cand, llm_pick)
            scored.append({'candidate': cand, 'scene': scene, 'confidence': conf,
                           'matched': matched, 'detail': detail})
        scored.sort(key=lambda x: -x['confidence'])

        if chosen_id == 'unknown' or not scored:
            confidence = min(MAX_NO_EVIDENCE, scored[0]['confidence'] if scored else 0.0)
            label, scene_id = 'unknown', 'unknown'
            matched = []
            alternatives = [{'scene_id': s['candidate']['scene_id'], 'label': s['candidate']['label'],
                             'confidence': s['confidence']} for s in scored[:3]]
        else:
            picked = next((s for s in scored if s['candidate']['scene_id'] == chosen_id),
                          scored[0])
            confidence = picked['confidence']
            if not picked['matched']:
                confidence = min(MAX_NO_EVIDENCE, confidence)
            label = picked['candidate']['label']
            scene_id = chosen_id
            matched = list(picked['matched'].keys())
            alternatives = [{'scene_id': s['candidate']['scene_id'], 'label': s['candidate']['label'],
                             'confidence': s['confidence']}
                            for s in scored if s['candidate']['scene_id'] != chosen_id][:3]

        status = 'approved' if decision == 'human' else 'pending'
        machine_confidence = confidence
        if decision == 'human':
            selected = next((scene for scene in self.knowledge.get('scenes', [])
                             if manual_label in (scene['scene_id'], scene.get('label'))), None)
            label = selected['label'] if selected else manual_label
            scene_id = selected['scene_id'] if selected else manual_label
            confidence = None
        record = {
            'image': image_path,
            'scene_id': scene_id,
            'label': label,
            'confidence': confidence,
            'score_kind': 'evidence',
            'decision': decision,
            'evidence_types': sorted(set(matched)),
            'retrieval_mode': candidates[0].get('retrieval_mode', 'lexical') if candidates else 'lexical',
            'evidence': evidences,
            'alternatives': alternatives,
            'model_trace': model_trace,
            'review': {'status': status, 'reviewer': None, 'updated_at': None},
        }
        if decision == 'human':
            record['machine_confidence'] = machine_confidence
            record['review']['reviewer'] = 'human'
        return record


def reason_image(image_path, knowledge, client=None, embedder=None, **kwargs):
    return SceneReasoner(knowledge, client=client, embedder=embedder,
                         **kwargs).reason(image_path)


def reason_batch(image_paths, knowledge, client=None, output_path=None, **kwargs):
    image_paths = list(image_paths)
    reasoner = SceneReasoner(knowledge, client=client, **kwargs)
    if reasoner.embedder is not None and hasattr(reasoner.embedder, 'prepare'):
        reasoner.embedder.prepare(image_paths, knowledge)
    rows = [reasoner.reason(p) for p in image_paths]
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows
