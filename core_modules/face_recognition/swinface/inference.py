"""Offline inference for the official SwinFace checkpoint.

Inputs are aligned BGR crops. Recognition returns L2-normalized 512-D features;
age is the unscaled regression output in years, never a probability. These
features belong to a different space from ArcFace, despite equal dimensions.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from .compact import (
    RecognitionAgeModel, dynamic_int8, tensor_bytes, validate_compact_checkpoint,
)

from ._vendor.swin import SwinTransformer
from ._vendor.subnets import (
    FeatureAttentionModule, ModelBox, OutputModule, TaskSpecificSubnets,
)


UPSTREAM_COMMIT = 'ca36196814b54964e663e0991efd7d163c0069d3'
PREPROCESSING = 'bgr-to-rgb-resize112-linear-div255-sub0.5-div0.5-v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_model():
    """Exactly the configuration in upstream inference.py:SwinFaceCfg."""
    return ModelBox(
        backbone=SwinTransformer(num_classes=512),
        fam=FeatureAttentionModule(
            in_chans=2112, kernel_size=3, conv_shared=False, conv_mode='split',
            channel_attention='CBAM', spatial_attention=None, pooling='max',
            la_num_list=[2] * 11),
        tss=TaskSpecificSubnets(), om=OutputModule(), feature='all',
    )


class SwinFaceAnalyzer:
    input_size = (112, 112)
    embedding_size = 512

    def __init__(self, model_path=None, device='auto', batch_size=8):
        if model_path is None:
            from config.base import SWINFACE_MODEL_PATH
            model_path = SWINFACE_MODEL_PATH
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'SwinFace checkpoint not found: {path}')
        if batch_size < 1:
            raise ValueError('SwinFace batch_size must be positive')
        self.batch_size = batch_size
        self.model_file = str(path)
        self.checkpoint_sha256 = file_sha256(path)
        # weights_only prevents arbitrary pickle execution; mmap avoids copying
        # the optimizer and classifier tensors that inference does not use.
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict):
            raise ValueError('SwinFace checkpoint must be a state dictionary')
        self.checkpoint_format = checkpoint.get('format', 'official-training')
        self.precision = checkpoint.get('precision', 'fp32') if 'format' in checkpoint else 'fp32'
        compact = 'format' in checkpoint
        if compact:
            validate_compact_checkpoint(checkpoint)
        if device == 'auto':
            device = 'cuda:0' if self.precision != 'int8_dynamic' and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('SwinFace device must be auto, cpu or cuda[:N]')
        if self.precision == 'int8_dynamic' and self.device.type != 'cpu':
            raise ValueError('SwinFace dynamic INT8 supports CPU only; use auto/cpu or FP16 weights for CUDA')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('SwinFace CUDA was requested but is unavailable')
        self.inference_precision = 'fp32' if self.device.type == 'cpu' else 'official_mixed'
        if compact:
            self.model = RecognitionAgeModel()
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            self.recognition_parameter_count = sum(p.numel() for p in self.model.backbone.parameters())
            if self.precision == 'int8_dynamic':
                self.model = dynamic_int8(self.model, checkpoint['quantization']['engine'])
                self.inference_precision = 'int8_dynamic_linear_fp32_other'
            elif self.precision == 'fp16' and self.device.type == 'cuda':
                self.model.half()
                self.inference_precision = 'fp16_mixed'
            # CPU FP16 storage is expanded to FP32, avoiding slow/unsupported
            # CPU half kernels. CUDA retains half parameters with AMP attention.
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.age_modules = (self.model.age_attention, self.model.age_subnet, self.model.age_head)
            self.checkpoint_step = checkpoint.get('checkpoint_step')
        else:
            self.model = build_model()
            for component in ('backbone', 'fam', 'tss', 'om'):
                key = f'state_dict_{component}'
                if key not in checkpoint:
                    raise ValueError(f'SwinFace checkpoint is missing {key}')
                getattr(self.model, component).load_state_dict(checkpoint[key], strict=True)
            self.age_modules = (self.model.fam.nets[1], self.model.tss.nets[1], self.model.om.output_fcs[1])
            self.checkpoint_step = checkpoint.get('global_step')
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            self.recognition_parameter_count = sum(p.numel() for p in self.model.backbone.parameters())
        self.source_sha256 = checkpoint.get('source_sha256', self.checkpoint_sha256)
        self.quantized_linear_layers = checkpoint.get('quantized_linear_layers', 0)
        self.runtime_state_tensor_bytes = tensor_bytes(self.model.state_dict())
        del checkpoint
        if file_sha256(path) != self.checkpoint_sha256:
            raise ValueError('SwinFace weights changed during loading')
        self.model.to(self.device).eval()
        self.model_signature = f'swinface:{self.checkpoint_sha256}:{PREPROCESSING}:l2:{self.inference_precision}'

    @staticmethod
    def preprocess(face_images):
        """Match the official RGB / 255 / 0.5 preprocessing operation order."""
        images = []
        for image in face_images:
            if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
                raise ValueError('SwinFace expects nonempty uint8 face crops')
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('SwinFace expects grayscale, BGR or BGRA face crops')
            image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image.transpose(2, 0, 1))
        if not images:
            return torch.empty((0, 3, 112, 112), dtype=torch.float32)
        return torch.from_numpy(np.stack(images)).float().div_(255).sub_(0.5).div_(0.5)

    @staticmethod
    def l2_normalize(features):
        features = np.asarray(features, dtype=np.float32)
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    @torch.inference_mode()
    def predict(self, face_images, *, recognition=True, age=True):
        """Run a shared backbone once per batch, evaluating requested heads only.

        The official age path is fam.nets[1] -> tss.nets[1] ->
        om.output_fcs[1]. It is independent of the other 41 analysis outputs.
        """
        if not recognition and not age:
            raise ValueError('Select recognition and/or age')
        face_images = list(face_images)
        embeddings, ages = [], []
        for offset in range(0, len(face_images), self.batch_size):
            batch = self.preprocess(face_images[offset:offset + self.batch_size]).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.inference_precision == 'fp16_mixed'):
                local, global_, embedding = self.model.backbone(batch)
                if age:
                    features = torch.cat((local, global_), dim=1)
                    for module in self.age_modules:
                        features = module(features)
            if recognition:
                value = embedding.float().cpu().numpy()
                if not np.isfinite(value).all() or np.any(np.linalg.norm(value, axis=1) < 1e-12):
                    raise RuntimeError('SwinFace returned invalid recognition features')
                embeddings.append(self.l2_normalize(value))
            if age:
                value = features.flatten().float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise RuntimeError('SwinFace returned a non-finite age estimate')
                ages.append(value)
        result = {}
        if recognition:
            result['embeddings'] = (np.concatenate(embeddings) if embeddings
                                    else np.empty((0, self.embedding_size), dtype=np.float32))
        if age:
            result['ages'] = np.concatenate(ages) if ages else np.empty(0, dtype=np.float32)
        return result

    def extract_features(self, face_images):
        return self.predict(face_images, recognition=True, age=False)['embeddings']

    def estimate_ages(self, face_images):
        return self.predict(face_images, recognition=False, age=True)['ages']

    def metadata(self):
        return {
            'name': 'SwinFace', 'checkpoint_sha256': self.checkpoint_sha256,
            'checkpoint_step': self.checkpoint_step, 'model_signature': self.model_signature,
            'upstream_commit': UPSTREAM_COMMIT, 'preprocessing': PREPROCESSING,
            'device': str(self.device), 'batch_size': self.batch_size,
            'parameters': self.parameter_count,
            'recognition_parameters': self.recognition_parameter_count,
            'checkpoint_format': self.checkpoint_format, 'storage_precision': self.precision,
            'inference_precision': self.inference_precision,
            'checkpoint_bytes': Path(self.model_file).stat().st_size,
            'source_sha256': self.source_sha256,
            'runtime_state_tensor_bytes': self.runtime_state_tensor_bytes,
            'quantized_linear_layers': self.quantized_linear_layers,
        }
