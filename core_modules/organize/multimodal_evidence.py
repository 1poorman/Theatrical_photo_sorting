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
import tempfile

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
        self.diagnostics = {}
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
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.npz', delete=False) as f:
            tmp = f.name
        try:
            np.savez(tmp, **self._cache)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path

    def _image_key(self, path):
        path = os.path.abspath(path)
        try:
            stat = os.stat(path)
            version = f'{stat.st_mtime_ns}:{stat.st_size}'
        except OSError:
            version = 'missing'
        return f'image:{path}:{version}'

    def _text_key(self, text):
        return 'text:' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]

    # ---------- 向量 ----------

    def prepare(self, image_paths, knowledge, batch_size=8):
        """Bounded batch prefill; batch failures retry each item independently."""
        if np is None:
            return
        image_paths = list(dict.fromkeys(os.path.abspath(p) for p in image_paths))
        missing = [p for p in image_paths if os.path.isfile(p) and self._image_key(p) not in self._cache]
        batch = getattr(self.embedder, 'extract_features_batch', None)
        if batch:
            for start in range(0, len(missing), batch_size):
                paths = missing[start:start + batch_size]
                try:
                    result = batch(paths)
                    vectors, valid = result if isinstance(result, tuple) else (result, paths)
                    if len(vectors) != len(valid) or not set(valid).issubset(paths):
                        raise ValueError('批量特征路径不匹配')
                    for path, vector in zip(valid, vectors):
                        vector = np.asarray(vector, dtype='float32').reshape(-1)
                        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                            continue
                        self._cache[self._image_key(path)] = vector
                        self.calls['image'] += 1
                except Exception:
                    pass
                for path in paths:
                    if self._image_key(path) not in self._cache:
                        try:
                            self.image_vector(path)
                        except Exception:
                            continue
        texts = list(dict.fromkeys(s.get('label', '') + ' ' + s.get('text', '')
                                   for s in knowledge.get('scenes', [])))
        missing_texts = [t for t in texts if self._text_key(t) not in self._cache]
        batch_text = getattr(self.embedder, 'encode_text_batch', None)
        if batch_text:
            for start in range(0, len(missing_texts), batch_size):
                chunk = missing_texts[start:start + batch_size]
                try:
                    vectors = batch_text(chunk)
                    if len(vectors) != len(chunk):
                        raise ValueError('批量文本特征数量不匹配')
                    for value, vector in zip(chunk, vectors):
                        vector = np.asarray(vector, dtype='float32').reshape(-1)
                        if np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                            self._cache[self._text_key(value)] = vector
                            self.calls['text'] += 1
                except Exception:
                    continue

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
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            return None
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
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            return None
        self.calls['text'] += 1
        self._cache[key] = vector
        return vector

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
                errors.append({'component': 'image_embedding', 'reason': 'unavailable'})
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
                errors.append({'component': 'text_embedding', 'scene_id': scene['scene_id'],
                               'reason': error or ('dimension_mismatch' if vec is not None else 'unavailable')})
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
