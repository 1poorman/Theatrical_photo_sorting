# -*- coding: utf-8 -*-
"""剧照到剧目知识库的候选证据召回。

默认实现是可离线、可审计的词法召回；传入 ``MultimodalRetriever`` 后可追加
SigLIP2 图文向量余弦相似度，并对图片/文本向量做磁盘缓存，重跑不重复推理。
该层只生成候选，不替代后续 scene_reasoner 的闭集判定。
"""
import hashlib
import json
import os
import re

from core_modules.organize.filename_parser import parse_filename

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
        if parsed.get('scene') and scene.get('title') in (parsed.get('scene') or ''):
            score += 1.0
        ranked.append((score, scene))
    ranked.sort(key=lambda x: (-x[0], x[1].get('act', 0)))
    return query, ranked[:top_k]


def _cosine(a, b):
    if np is None:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MultimodalRetriever:
    """图文向量召回：图片经 ``embedder.extract_features``，文本经 ``text_encoder``。

    向量按内容键缓存到 ``cache_dir/embeddings.npz``；命中缓存不调用模型，满足
    “重跑不重复推理”。缺图或缺少文本编码器时自动降级为词法召回。
    """

    def __init__(self, embedder=None, text_encoder=None, cache_dir=None,
                 model_signature=None):
        self.embedder = embedder
        if text_encoder is None and embedder is not None and hasattr(embedder, 'encode_text'):
            text_encoder = embedder.encode_text
        self.text_encoder = text_encoder
        self.cache_dir = cache_dir
        self.model_signature = (model_signature
                                or getattr(embedder, 'model_signature', None)
                                or 'unknown')
        self._cache = {}
        self.calls = {'image': 0, 'text': 0}
        self._load_cache()

    # ---------- 缓存 ----------

    def _cache_path(self):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir,
                            f'embeddings_{self.model_signature}.npz')

    def _load_cache(self):
        path = self._cache_path()
        if not path or not os.path.exists(path) or np is None:
            return
        with np.load(path) as data:
            self._cache = {k: data[k] for k in data.files}

    def save(self):
        path = self._cache_path()
        if not path or np is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(path, **self._cache)
        return path

    def _image_key(self, path):
        path = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return f'image:{path}:{int(mtime)}'

    def _text_key(self, text):
        return 'text:' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]

    # ---------- 向量 ----------

    def image_vector(self, image_path):
        if self.embedder is None or not os.path.exists(image_path):
            return None
        key = self._image_key(image_path)
        if key in self._cache:
            return self._cache[key]
        vector = self.embedder.extract_features(image_path)
        if vector is None:
            return None
        vector = np.asarray(vector, dtype='float32').reshape(-1)
        self.calls['image'] += 1
        self._cache[key] = vector
        return vector

    def text_vector(self, text):
        if self.text_encoder is None:
            return None
        key = self._text_key(text)
        if key in self._cache:
            return self._cache[key]
        vector = self.text_encoder(text)
        if vector is None:
            return None
        vector = np.asarray(vector, dtype='float32').reshape(-1)
        self.calls['text'] += 1
        self._cache[key] = vector
        return vector

    # ---------- 召回 ----------

    def rank(self, image_path, knowledge, top_k=5):
        query, lexical = _scene_candidates(image_path, knowledge, top_k)
        image_vec = self.image_vector(image_path)
        if image_vec is None:
            return query, lexical, 'lexical'
        scene_vecs = []
        for score, scene in lexical:
            vec = self.text_vector(scene.get('label', '') + ' ' + scene.get('text', ''))
            if vec is not None:
                scene_vecs.append((_cosine(image_vec, vec), scene))
        if not scene_vecs:
            return query, lexical, 'lexical'
        scene_vecs.sort(key=lambda x: (-x[0], x[1].get('act', 0)))
        # 词法作为弱先验叠加，避免纯向量漏掉明确幕次信息
        merged = []
        lex = {s['scene_id']: sc for sc, s in lexical}
        for cos, scene in scene_vecs:
            visual = min(1.0, max(0.0, cos)) * 0.8
            lexical_prior = min(1.0, lex.get(scene['scene_id'], 0.0)) * 0.2
            merged.append((round(visual + lexical_prior, 4), scene))
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
            evidence.append({'type': 'knowledge', 'source_ref': scene['source_ref'],
                             'quote': scene['label'], 'score': round(float(score), 4)})
        candidates.append({'scene_id': scene['scene_id'], 'label': scene['label'],
                           'score': round(float(min(1.0, score)), 4),
                           'evidence': evidence})
    return {'image': image_path, 'query': query, 'candidates': candidates,
            'retrieval': {'lexical': True, 'visual': mode == 'visual', 'mode': mode}}


def build_scene_candidates(image_paths, knowledge, output_path=None, top_k=5,
                           embedder=None, cache_dir=None):
    """批量生成候选；若传入缓存目录，重跑命中向量缓存不重复推理。"""
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
