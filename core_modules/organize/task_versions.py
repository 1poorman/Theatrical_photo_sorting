"""Execution identities for durable tasks; never include client credentials."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import uuid

from core_modules.organize.feature_cache import signature, tensor_signature
from core_modules.organize.task_artifacts import fingerprint


_PROCESS = uuid.uuid4().hex


def model_identity(model):
    if model is None:
        return None
    identity = {'class': type(model).__module__ + '.' + type(model).__qualname__}
    declared = getattr(model, 'model_signature', None)
    if declared is not None:
        identity['weights'] = declared
        identity['preprocessing'] = getattr(model, 'preprocessing_signature', None)
    elif hasattr(model, 'detector') and hasattr(model, 'extractor'):
        identity['detector'] = model_identity(model.detector)
        identity['extractor'] = model_identity(model.extractor)
    elif hasattr(model, 'yolo_pose_model'):
        network = model.yolo_pose_model.model
        versions = tuple((name, tensor.data_ptr(), tensor._version, str(tensor.dtype))
                         for name, tensor in network.state_dict(keep_vars=True).items())
        if versions != getattr(model, '_task_weight_versions', None):
            model._task_weight_digest = tensor_signature(network)
            model._task_weight_versions = versions
        identity['weights'] = model._task_weight_digest
    elif hasattr(model, 'embedder'):
        identity['embedder'] = model_identity(model.embedder)
        templates = getattr(model, 'text_feats', None)
        if templates is not None:
            identity['templates'] = signature([templates[0], templates[1].tolist()])
    else:
        # An unsigned adapter cannot safely reuse checkpoints in another process.
        identity['private'] = _PROCESS + ':' + str(id(model))
    session = getattr(model, 'session', None)
    if session is not None:
        identity['providers'] = session.get_providers()
        identity['provider_options'] = session.get_provider_options()
    for field in ('input_size', 'input_mean', 'input_std', 'det_thresh', 'nms_thresh'):
        if hasattr(model, field):
            identity[field] = getattr(model, field)
    return identity


def client_identity(client):
    if client is None:
        return None
    if hasattr(client, 'small'):
        return {'small': client_identity(client.small), 'big': client_identity(client.big),
                'cascade_enabled': client.cascade_enabled}
    # Endpoint changes matter, but URLs can contain credentials or query tokens.
    verified = getattr(client, 'verified_weights_sha256', None)
    if not isinstance(verified, str) or len(verified) != 64 or any(c not in '0123456789abcdef' for c in verified):
        verified = None
    identity = {'endpoint_sha256': signature(client.base_url), 'model': client.model,
            'declared_identity': getattr(client, 'model_identity', None),
            'identity_source': getattr(client, 'model_identity_source', 'served_alias'),
            'verified_weights_sha256': verified,
            'remote_weights_verified': verified is not None,
            'temperature': client.temperature, 'max_tokens': client.max_tokens,
            'chat_template_kwargs': client.chat_template_kwargs,
            'max_retries': client.max_retries, 'timeout': client.timeout}
    if verified is None:
        # A stable alias cannot prove the remote weights survived a restart.
        identity['private_process'] = _PROCESS
    return identity


def execution_code(files):
    root = Path(__file__).resolve().parents[2]
    files = sorted(set(files) | {'core_modules/organize/task_versions.py',
                                'core_modules/organize/task_queue.py', 'config/base.py'})
    versions = {}
    for name in ('numpy', 'opencv-python', 'Pillow', 'torch', 'transformers',
                 'onnxruntime', 'onnxruntime-gpu', 'ultralytics'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {'files': {name: fingerprint(root / name) for name in files}, 'packages': versions}


class VersionedHandler(tuple):
    """Keep the legacy two-callable interface while exposing an identity."""
    def __new__(cls, process, finish, contract):
        value = super().__new__(cls, (process, finish))
        value.model_contract = contract
        return value
