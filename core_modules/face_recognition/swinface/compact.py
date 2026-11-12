"""Versioned, inference-only SwinFace identity + age checkpoints.

The official training checkpoint remains the source of truth. Other attribute
branches and the unused backbone.head are omitted; no trained layer on the
recognition or age path is pruned. INT8 uses PyTorch's CPU dynamic Linear kernels.
"""
from collections import OrderedDict
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic

from ._vendor.swin import SwinTransformer
from ._vendor.subnets import FeatureAttentionNet, TaskSpecificSubnet


COMPACT_FORMAT = 'swinface-recognition-age-v1'
PRECISIONS = ('fp32', 'fp16', 'int8_dynamic')
INT8_ENGINE = 'fbgemm'


class RecognitionAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinTransformer(num_classes=512)
        # Upstream forward uses backbone.feature; backbone.head is never called.
        self.backbone.head = nn.Identity()
        self.age_attention = FeatureAttentionNet(
            in_chans=2112, kernel_size=3, conv_shared=False, conv_mode='split',
            channel_attention='CBAM', spatial_attention=None, pooling='max',
            la_branch_num=2,
        )
        self.age_subnet = TaskSpecificSubnet()
        self.age_head = nn.Linear(512, 1)

    def forward(self, images):
        local, global_, embedding = self.backbone(images)
        features = self.age_attention(torch.cat((local, global_), dim=1))
        age = self.age_head(self.age_subnet(features))
        return {'Recognition': embedding, 'Age': age}


def dynamic_int8(model, engine=INT8_ENGINE):
    """Quantize Linear weights per output channel; activations stay float at IO.

    Convolutions, normalization, softmax and attention matrix products stay FP32.
    Packed weights are serialized as a state_dict, never a pickled model object.
    """
    if engine != INT8_ENGINE or engine not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f'SwinFace INT8 requires the {INT8_ENGINE} CPU quantization backend')
    torch.backends.quantized.engine = engine
    return quantize_dynamic(
        model.cpu().eval(), {nn.Linear: per_channel_dynamic_qconfig},
        dtype=torch.qint8, inplace=True,
    )


def tensor_bytes(value):
    """Logical tensor bytes, including nested packed parameters and buffers."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def from_official_checkpoint(checkpoint):
    model = RecognitionAgeModel()
    parts = (
        (model.backbone, 'backbone', ''),
        (model.age_attention, 'fam', 'nets.1.'),
        (model.age_subnet, 'tss', 'nets.1.'),
        (model.age_head, 'om', 'output_fcs.1.'),
    )
    for module, component, prefix in parts:
        key = f'state_dict_{component}'
        if key not in checkpoint:
            raise ValueError(f'SwinFace official checkpoint is missing {key}')
        selected = OrderedDict(
            (name[len(prefix):], tensor) for name, tensor in checkpoint[key].items()
            if name.startswith(prefix) and not (component == 'backbone' and name.startswith('head.'))
        )
        module.load_state_dict(selected, strict=True)
    return model.eval()


def validate_compact_checkpoint(checkpoint):
    # Import here to keep the model builders independent of the public wrapper.
    from .inference import PREPROCESSING, UPSTREAM_COMMIT

    if checkpoint.get('format') != COMPACT_FORMAT:
        raise ValueError('Unsupported SwinFace compact checkpoint format')
    if checkpoint.get('precision') not in PRECISIONS:
        raise ValueError('Unsupported SwinFace compact checkpoint precision')
    if (checkpoint.get('upstream_commit') != UPSTREAM_COMMIT
            or checkpoint.get('preprocessing') != PREPROCESSING):
        raise ValueError('SwinFace compact checkpoint architecture/preprocessing mismatch')
    if not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('SwinFace compact checkpoint is missing state_dict')
    if checkpoint['precision'] == 'int8_dynamic':
        if checkpoint.get('quantization') != {
            'engine': INT8_ENGINE, 'scheme': 'per_channel_symmetric', 'modules': 'Linear',
        }:
            raise ValueError('Unsupported SwinFace INT8 quantization configuration')


def export_checkpoint(source, destination, precision='fp16'):
    """Export atomically without replacing any existing file, including source."""
    from .inference import PREPROCESSING, UPSTREAM_COMMIT, file_sha256

    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}')
    if source == destination or destination.exists():
        raise FileExistsError(f'Refusing to overwrite SwinFace weights: {destination}')
    source_sha = file_sha256(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or 'format' in checkpoint:
        raise ValueError('Export requires the original official SwinFace training checkpoint')
    model = from_official_checkpoint(checkpoint)
    payload = {
        'format': COMPACT_FORMAT, 'precision': precision,
        'upstream_commit': UPSTREAM_COMMIT, 'preprocessing': PREPROCESSING,
        'source_sha256': source_sha, 'checkpoint_step': checkpoint.get('global_step'),
        'tasks': ['recognition', 'age'], 'torch_version': str(torch.__version__),
        'parameters': sum(p.numel() for p in model.parameters()),
        'recognition_parameters': sum(p.numel() for p in model.backbone.parameters()),
    }
    original_parts = {key: tensor_bytes(value) for key, value in checkpoint.items()}
    del checkpoint
    if precision == 'fp16':
        model.half()
    elif precision == 'int8_dynamic':
        payload['quantization'] = {
            'engine': INT8_ENGINE, 'scheme': 'per_channel_symmetric', 'modules': 'Linear',
        }
        payload['quantized_linear_layers'] = sum(isinstance(m, nn.Linear) for m in model.modules())
        model = dynamic_int8(model)
    payload['state_dict'] = model.state_dict()
    if file_sha256(source) != source_sha:
        raise ValueError('SwinFace source weights changed during export')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.swinface-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        # Atomic no-clobber publication; a concurrent export cannot replace it.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        **{key: value for key, value in payload.items() if key != 'state_dict'},
        'source_path': str(source), 'source_bytes': source.stat().st_size,
        'source_component_tensor_bytes': original_parts,
        'path': str(destination), 'sha256': file_sha256(destination),
        'bytes': destination.stat().st_size,
        'state_tensor_bytes': tensor_bytes(payload['state_dict']),
        'size_reduction_percent': 100 * (1 - destination.stat().st_size / source.stat().st_size),
    }
