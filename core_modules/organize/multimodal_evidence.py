# -*- coding: utf-8 -*-
"""剧照到剧目知识库的候选证据召回。

默认实现是可离线、可审计的词法召回；传入 ``MultimodalRetriever`` 后可追加
SigLIP2 图文向量余弦相似度，并对图片/文本向量做磁盘缓存，重跑不重复推理。
该层只生成候选，不替代后续 scene_reasoner 的闭集判定。
"""
import json
import os
import re

from core_modules.organize.filename_parser import parse_filename
from config.base import FEATURE_BATCH_SIZE

try:
    import numpy as np
except Exception:  # numpy 缺失时退化为纯词法
    np = None


def _terms(text):
    return set(re.findall(r'[\u4e00-\u9fffA-Za-z·]{2,12}', text or ''))


def _lexical_score(query, text):
    q, t = _terms(query), _terms(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def _scene_candidates(image_path, knowledge, top_k=5):
    name = os.path.basename(image_path)
    parsed = parse_filename(name)
    query = ' '.join([name, parsed.get('event') or '',
                      ' '.join(a + r for a, r in parsed.get('people', []))])
    ranked = []
    for scene in knowledge.get('scenes', []):
        score = _lexical_score(query, scene.get('label', '') + ' ' + scene.get('text', ''))
        if parsed.get('scene') and scene.get('title') and scene['title'] in parsed['scene']:
            score += 1.0
        ranked.append((score, scene))
    ranked.sort(key=lambda x: (-x[0], x[1].get('act', 0)))
    return query, ranked if top_k is None else ranked[:top_k]


def _cosine(a, b):
    if np is None:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MultimodalRetriever:
    """图文向量召回：图片经 ``embedder.extract_features``，文本经 ``text_encoder``。

    图文向量使用与整理流程相同的内容缓存。缺图或缺少文本编码器时降级
    为词法召回；旧的路径缓存不满足签名契约，不迁入新缓存。
    """

    def __init__(self, embedder=None, text_encoder=None, cache_dir=None,
                 model_signature=None, feature_service=None):
        self.embedder = embedder
        if text_encoder is None and embedder is not None and hasattr(embedder, 'encode_text'):
            text_encoder = embedder.encode_text
        self.text_encoder = text_encoder
        self.cache_dir = cache_dir
        self.diagnostics = {}
        self.features = feature_service
        if self.features is None and np is not None:
            from core_modules.organize.feature_cache import FeatureService
            self.features = FeatureService(embedder, cache_dir=cache_dir,
                                           model_signature=model_signature, text_encoder=text_encoder)
        self.model_signature = self.features.model_signature if self.features else model_signature
        self.calls = self.features.calls if self.features else {'image': 0, 'text': 0}

    def save(self):
        return self.features.save() if self.features else None

    def prepare(self, image_paths, knowledge, batch_size=FEATURE_BATCH_SIZE):
        """Bounded batch prefill; batch failures retry each item independently."""
        if self.features is None:
            return
        self.features.clear_failures()
        self.features.images(image_paths, batch_size=batch_size)
        texts = list(dict.fromkeys(s.get('label', '') + ' ' + s.get('text', '')
                                   for s in knowledge.get('scenes', [])))
        self.features.texts(texts, batch_size=batch_size)

    def image_vector(self, image_path):
        return (self.features.images([image_path], use_batch=False).get(os.fspath(image_path))
                if self.features is not None else None)

    def text_vector(self, text):
        return (self.features.texts([text], use_batch=False).get(text)
                if self.features is not None else None)

    # ---------- 召回 ----------

    def rank(self, image_path, knowledge, top_k=5):
        query, lexical = _scene_candidates(image_path, knowledge, None)
        errors = []
        try:
            image_vec = self.image_vector(image_path)
        except Exception as exc:
            image_vec = None
            errors.append({'component': 'image_embedding', 'reason': type(exc).__name__})
        if image_vec is None:
            if not errors:
                reason = self.features.image_errors.get(os.fspath(image_path)) if self.features else None
                errors.append({'component': 'image_embedding', 'reason': reason or 'unavailable'})
            self.diagnostics[image_path] = errors
            return query, lexical[:top_k], 'lexical'
        merged = []
        available = 0
        for score, scene in lexical:
            error = None
            try:
                vec = self.text_vector(scene.get('label', '') + ' ' + scene.get('text', ''))
            except Exception as exc:
                vec = None
                error = type(exc).__name__
            lexical_score = min(1.0, max(0.0, score))
            visual_score = None
            if vec is not None and vec.shape == image_vec.shape and np.isfinite(vec).all():
                visual_score = min(1.0, max(0.0, _cosine(image_vec, vec)))
                available += 1
            else:
                reason = self.features.text_errors.get(scene.get('label', '') + ' ' + scene.get('text', '')) if self.features else None
                errors.append({'component': 'text_embedding', 'scene_id': scene['scene_id'],
                               'reason': error or reason or ('dimension_mismatch' if vec is not None else 'unavailable')})
            details = {'visual_score': visual_score, 'lexical_score': lexical_score}
            # Missing text features retain their lexical candidate. A single
            # failed encoding must not remove a scene from the candidate pool.
            fused = .8 * (visual_score or 0.0) + .2 * lexical_score
            merged.append((round(fused, 4), dict(scene, _retrieval_scores=details)))
        self.diagnostics[image_path] = errors
        if not available:
            return query, lexical[:top_k], 'lexical'
        merged.sort(key=lambda x: (-x[0], x[1].get('act', 0)))
        return query, merged[:top_k], 'visual'


def retrieve_evidence(image_path, knowledge, top_k=5, embedder=None):
    """返回单张图片的场景候选和引用证据。

    ``embedder`` 可为 ``MultimodalRetriever``；否则使用词法召回。
    """
    mode = 'lexical'
    if embedder is not None and hasattr(embedder, 'rank'):
        query, ranked, mode = embedder.rank(image_path, knowledge, top_k)
    else:
        query, ranked = _scene_candidates(image_path, knowledge, top_k)
    candidates = []
    for score, scene in ranked:
        evidence = []
        if score > 0:
            evidence.append({'type': 'knowledge', 'source_ref': scene.get('source_ref'),
                             'quote': scene['label'], 'score': round(float(score), 4)})
        candidates.append({'scene_id': scene['scene_id'], 'label': scene['label'],
                           'retrieval_mode': 'visual' if scene.get('_retrieval_scores', {}).get('visual_score') is not None else 'lexical',
                           'lexical_score': scene.get('_retrieval_scores', {}).get(
                               'lexical_score', round(float(min(1.0, score)), 4)),
                           'visual_score': scene.get('_retrieval_scores', {}).get('visual_score'),
                           'score': round(float(min(1.0, score)), 4),
                           'evidence': evidence})
    return {'image': image_path, 'query': query, 'candidates': candidates,
            'retrieval': {'lexical': True, 'visual': mode == 'visual', 'mode': mode,
                          'errors': getattr(embedder, 'diagnostics', {}).get(image_path, [])}}


def build_scene_candidates(image_paths, knowledge, output_path=None, top_k=5,
                           embedder=None, cache_dir=None):
    """批量生成候选；若传入缓存目录，重跑命中向量缓存不重复推理。"""
    image_paths = list(image_paths)
    if embedder is not None and hasattr(embedder, 'prepare'):
        embedder.prepare(image_paths, knowledge)
    rows = [retrieve_evidence(p, knowledge, top_k=top_k, embedder=embedder)
            for p in image_paths]
    if embedder is not None and hasattr(embedder, 'save'):
        embedder.save()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows
