# -*- coding: utf-8 -*-
"""M13 离线验收：图文向量召回、缓存命中不重复推理、缺图降级。"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from core_modules.organize.play_knowledge import build_knowledge
from core_modules.organize.multimodal_evidence import (
    MultimodalRetriever, build_scene_candidates, retrieve_evidence)

TMP = os.path.join(ROOT, 'outputs', 'tmp_multimodal_evidence')
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)

ACTS = ['克段', '庆功', '私访', '天问', '见母']


class FakeEmbedder:
    """把含幕次关键词的文本/文件内容映射为 5 维 one-hot 向量。"""
    model_signature = 'fake-v1'

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        vec = np.zeros(5, dtype='float32')
        for i, act in enumerate(ACTS):
            if act in text:
                vec[i] = 1.0
        return vec

    def extract_features(self, path):
        self.calls += 1
        with open(path, encoding='utf-8') as stream:
            return self._vec(stream.read())

    def encode_text(self, text):
        return self._vec(text)


tx_root = os.path.join(ROOT, 'data', 'ncpa_test', '京剧《天下归心》')
tx = build_knowledge(os.path.join(tx_root, '剧目信息.txt'), tx_root)

# 构造 5 个内容不同的编码器替身输入，验证内容缓存和召回的行为契约。
samples = []
for i, act in enumerate(ACTS, start=1):
    name = (f'0{i}-20140304-戏剧场-京剧《天下归心》彩排 第{i}幕{act}'
            f'-孟广禄饰郑庄公-苏岩.jpg')
    path = os.path.join(TMP, name)
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(act)
    samples.append((path, f'act-{i:02d}'))

# 1. 图文向量召回 top-1 全中（召回率 100% ≥ 90%）
embedder = FakeEmbedder()
retriever = MultimodalRetriever(embedder=embedder, cache_dir=TMP)
hits = 0
for path, expected in samples:
    row = retrieve_evidence(path, tx, top_k=5, embedder=retriever)
    assert row['retrieval']['mode'] == 'visual', row['retrieval']
    assert row['candidates'][0]['scene_id'] == expected, (path, row['candidates'][0])
    hits += 1
assert hits / len(samples) >= 0.9
print(f'1. 图文向量 top-5 召回 {hits}/{len(samples)} = 100% OK')

# 2. 首次推理计数并落盘缓存
assert embedder.calls == len(samples)
first = dict(retriever.calls)
assert first == {'image': 5, 'text': 5}, first
cache_path = retriever.save()
assert cache_path and os.path.exists(cache_path)
print('2. 首次推理并写缓存 OK:', first)

# 3. 新实例命中缓存：重跑不重复推理
embedder2 = FakeEmbedder()
retriever2 = MultimodalRetriever(embedder=embedder2, cache_dir=TMP)
for path, expected in samples:
    row = retrieve_evidence(path, tx, top_k=5, embedder=retriever2)
    assert row['candidates'][0]['scene_id'] == expected
assert embedder2.calls == 0, embedder2.calls
assert retriever2.calls == {'image': 0, 'text': 0}, retriever2.calls
print('3. 缓存命中，0 次重复推理 OK')

# 4. 缺图降级为词法，不崩溃
row = retrieve_evidence(os.path.join(TMP, 'not-exist.jpg'), tx, top_k=5,
                        embedder=retriever2)
assert row['retrieval']['mode'] == 'lexical'
print('4. 缺图降级词法 OK')

# 5. 无文本编码器时降级
class ImageOnly:
    model_signature = 'img-only'

    def extract_features(self, path):
        return np.ones(5, dtype='float32')


row = retrieve_evidence(samples[0][0], tx, top_k=5,
                        embedder=MultimodalRetriever(embedder=ImageOnly()))
assert row['retrieval']['mode'] == 'lexical'
print('5. 缺文本编码器降级 OK')

# 6. build_scene_candidates 落盘
out = os.path.join(TMP, 'scene_candidates.jsonl')
rows = build_scene_candidates([p for p, _ in samples], tx, output_path=out,
                              embedder=MultimodalRetriever(embedder=FakeEmbedder(),
                                                           cache_dir=TMP))
assert len(rows) == 5 and os.path.exists(out)
with open(out, encoding='utf-8') as f:
    assert sum(1 for _ in f) == 5
print('6. 候选落盘 OK')

shutil.rmtree(TMP, ignore_errors=True)
print('\n✅ M13 图文向量召回与缓存测试通过')
