"""Local EmbeddingGemma 2 adapter; imports optional dependencies only on load."""
import json
from pathlib import Path
import time
import uuid
from importlib.metadata import version

import numpy as np

from config.base import EMBEDDINGGEMMA_DEVICE, EMBEDDINGGEMMA_MODEL_PATH
from core_modules.organize.feature_cache import ContentHasher, checkpoint_signature, signature


class EmbeddingGemmaEmbedder:
    """Text/image embeddings share a normalized 768-dimensional space.

    Uses the checkpoint's documented SentenceTransformer interface. Images have
    no task prefix; documents have the Document prefix. FP16 is never used.
    """

    def __init__(self, model_path=EMBEDDINGGEMMA_MODEL_PATH,
                 device=EMBEDDINGGEMMA_DEVICE, model=None):
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self._model = model
        self.feature_dim = 768
        self.metrics = {}
        self._hasher = ContentHasher()
        root = Path(self.model_path)
        config = json.loads((root / 'config.json').read_text())
        if config.get('model_type') != 'embedding_gemma2' or not config.get('vision_config'):
            raise ValueError('需要包含视觉编码器的 EmbeddingGemma 2 权重')
        self._checkpoint = checkpoint_signature(root, self._hasher)
        self._injected = uuid.uuid4().hex if model is not None else None

    def _dtype(self):
        import torch
        if self.device.startswith('cuda'):
            with torch.cuda.device(self.device):
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
        return torch.float32

    @property
    def model_signature(self):
        return signature({'model': 'embeddinggemma2', 'checkpoint_sha256': self._checkpoint,
                          'dtype': str(self._dtype()) if self._injected is None else 'injected',
                          'injected_instance': self._injected})

    @property
    def preprocessing_signature(self):
        return signature({'adapter': 'embedding-gemma-v2', 'checkpoint_sha256': self._checkpoint,
                          'text_prompt': 'Document', 'image_prompt': '', 'dimension': 768,
                          'normalize': True, 'audio_config': None,
                          'versions': {name: version(name) for name in
                                       ('torch', 'transformers', 'sentence-transformers', 'Pillow')}})

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.model_path, local_files_only=True)
            from sentence_transformers import SentenceTransformer
        except (ImportError, ValueError) as exc:
            raise RuntimeError('EmbeddingGemma 2 需要支持 embedding_gemma2 的 Transformers '
                               '及 SentenceTransformers；请使用 requirements_embeddinggemma.txt') from exc
        if checkpoint_signature(self.model_path, self._hasher) != self._checkpoint:
            raise ValueError('EmbeddingGemma 权重或处理器已变化，请重新创建编码器')
        self._model = SentenceTransformer(
            self.model_path, device=self.device, local_files_only=True,
            model_kwargs={'torch_dtype': self._dtype()}, config_kwargs={'audio_config': None})
        if checkpoint_signature(self.model_path, self._hasher) != self._checkpoint:
            self._model = None
            raise ValueError('EmbeddingGemma 权重在加载期间变化')
        return self._model

    def _encode(self, inputs, batch_size=8, kind='image', **kwargs):
        if not inputs:
            return np.empty((0, 768), dtype='float32')
        model = self._load()
        hooks, starts = [], []
        def synchronize():
            if self._injected is None and self.device.startswith('cuda'):
                import torch
                torch.cuda.synchronize(self.device)
        def before_forward(*args):
            synchronize()
            starts.append(time.perf_counter())
            key = kind + '_forward_calls'
            self.metrics[key] = self.metrics.get(key, 0) + 1
        def after_forward(*args):
            synchronize()
            key = kind + '_forward_seconds'
            self.metrics[key] = self.metrics.get(key, 0) + time.perf_counter() - starts.pop()
        if self._injected is None:
            encoder = model[0].auto_model
            hooks = [encoder.register_forward_pre_hook(before_forward),
                     encoder.register_forward_hook(after_forward, always_call=True)]
        started = time.perf_counter()
        try:
            vectors = model.encode(inputs, batch_size=batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
            key = kind + '_encoder_seconds'
            self.metrics[key] = self.metrics.get(key, 0) + time.perf_counter() - started
        vectors = np.asarray(vectors, dtype='float32').reshape(len(inputs), -1)
        if vectors.shape[1] != 768 or not np.isfinite(vectors).all():
            raise ValueError('EmbeddingGemma 2 输出维度错误或包含非有限数值')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError('EmbeddingGemma 2 输出零向量')
        return vectors / norms

    def extract_features(self, path):
        return self.extract_features_batch([path])[0]

    def extract_features_batch(self, paths, batch_size=8):
        return self._encode([{'image': str(Path(p).resolve())} for p in paths],
                            batch_size=batch_size, prompt='')

    def encode_text(self, text):
        return self.encode_text_batch([text])[0]

    def encode_text_batch(self, texts, batch_size=8):
        return self._encode(list(texts), batch_size=batch_size, kind='text', prompt_name='Document')
