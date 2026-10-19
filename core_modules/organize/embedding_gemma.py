"""Local EmbeddingGemma 2 adapter; imports optional dependencies only on load."""
import hashlib
import json
from pathlib import Path

import numpy as np

from config.base import EMBEDDINGGEMMA_DEVICE, EMBEDDINGGEMMA_MODEL_PATH


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
        root = Path(self.model_path)
        config = json.loads((root / 'config.json').read_text())
        if config.get('model_type') != 'embedding_gemma2' or not config.get('vision_config'):
            raise ValueError('需要包含视觉编码器的 EmbeddingGemma 2 权重')
        digest = hashlib.sha256()
        for name in ('config.json', 'processor_config.json', 'config_sentence_transformers.json'):
            path = root / name
            if path.exists():
                digest.update(path.read_bytes())
        for path in sorted(root.glob('*.safetensors')):
            stat = path.stat()
            digest.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
        digest.update(b'document-prefix:Document;image-prefix:none;dim:768;adapter:v1')
        self.model_signature = 'embeddinggemma2-' + digest.hexdigest()[:20]

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
        dtype = torch.float32
        if self.device.startswith('cuda'):
            with torch.cuda.device(self.device):
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
        self._model = SentenceTransformer(
            self.model_path, device=self.device, local_files_only=True,
            model_kwargs={'torch_dtype': dtype}, config_kwargs={'audio_config': None})
        return self._model

    def _encode(self, inputs, batch_size=8, **kwargs):
        if not inputs:
            return np.empty((0, 768), dtype='float32')
        vectors = self._load().encode(
            inputs, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, **kwargs)
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
        return self._encode(list(texts), batch_size=batch_size, prompt_name='Document')
