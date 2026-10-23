# -*- coding: utf-8 -*-
"""本地大模型 OpenAI 兼容客户端与级联。

从 `.env` 运行时读取端点，密钥不写入代码、报告或 trace。
默认使用标准库 urllib，避免额外依赖；返回内容与可审计 trace 分离。
"""
import hashlib
import json
import os
import re
import time
import base64
import io
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')


def load_env(path=DEFAULT_ENV_PATH):
    """解析 `.env`，同时支持 ``key=value`` 与 ``key: value`` 两种写法。"""
    cfg = {}
    if not path or not os.path.exists(path):
        return cfg
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r'^([A-Za-z_][\w]*)\s*[:=]\s*(.*)$', line)
            if not m:
                continue
            cfg[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return cfg


def mask_secret(value):
    if not value:
        return ''
    return '***' if len(value) <= 6 else value[:4] + '***'


def _hash(text):
    return 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


class LLMError(RuntimeError):
    """端点不可用、超时或响应非法时抛出，供级联与降级判断。"""


class OpenAICompatClient:
    """最小 OpenAI ``/chat/completions`` 客户端，支持超时、重试和 trace。"""

    def __init__(self, base_url, api_key, model, timeout=60, max_retries=2,
                 temperature=0.0, max_tokens=1024, opener=urlopen,
                 chat_template_kwargs=None):
        if not base_url or not model:
            raise ValueError('base_url 与 model 不能为空')
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or ''
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._opener = opener
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def chat(self, messages, temperature=None, max_tokens=None):
        """返回 ``{'content': str, 'trace': dict}``；失败抛 ``LLMError``。"""
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature if temperature is None else temperature,
            'max_tokens': self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.chat_template_kwargs:
            payload['chat_template_kwargs'] = self.chat_template_kwargs
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        prompt_text = json.dumps(messages, ensure_ascii=False)
        url = f'{self.base_url}/chat/completions'
        last_err = None
        total_started = time.monotonic()
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                req = Request(url, data=body, method='POST', headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                })
                with self._opener(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode('utf-8', errors='replace')
                data = json.loads(raw)
                content = data['choices'][0]['message']['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('模型响应正文为空或类型错误')
                trace = {
                    'model': self.model,
                    'endpoint_sha256': hashlib.sha256(self.base_url.encode()).hexdigest(),
                    'prompt_hash': _hash(prompt_text),
                    'response_hash': _hash(content),
                    'temperature': payload['temperature'],
                    'attempts': attempt + 1,
                    'elapsed_ms': int((time.time() - started) * 1000),
                    'total_elapsed_ms': int((time.monotonic() - total_started) * 1000),
                    'max_output_tokens': payload['max_tokens'],
                }
                usage = data.get('usage')
                if isinstance(usage, dict):
                    trace['usage'] = {key: value for key, value in usage.items()
                                      if key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                      and type(value) is int and value >= 0}
                trace['finish_reason'] = data['choices'][0].get('finish_reason')
                images = []
                for message in messages:
                    if not isinstance(message.get('content'), list):
                        continue
                    for part in message['content']:
                        if part.get('type') != 'image_url':
                            continue
                        image_url = part.get('image_url', {}).get('url', '')
                        if image_url.startswith('data:image/') and ';base64,' in image_url:
                            raw_image = base64.b64decode(image_url.split(';base64,', 1)[1], validate=True)
                            images.append({'sha256': hashlib.sha256(raw_image).hexdigest(),
                                           'bytes': len(raw_image)})
                trace['image_inputs'] = images
                return {'content': content, 'trace': trace}
            except (HTTPError, URLError, TimeoutError, OSError, ValueError,
                    KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
        raise LLMError(f'模型端点调用失败: {type(last_err).__name__}: {last_err}')


class CascadeClient:
    """小模型优先，仅在小模型失败时升级到大模型；成功后不依赖模型置信度。"""

    def __init__(self, small, big=None, cascade_enabled=True):
        self.small = small
        self.big = big
        self.cascade_enabled = cascade_enabled

    def chat(self, messages, **kwargs):
        if self.small is None:
            if self.big is None:
                raise LLMError('未配置任何模型端点')
            return self.big.chat(messages, **kwargs)
        try:
            return self.small.chat(messages, **kwargs)
        except LLMError:
            if not self.cascade_enabled or self.big is None:
                raise
            result = self.big.chat(messages, **kwargs)
            result['trace']['cascade_upgraded_from'] = self.small.model
            return result


def image_content(image_path, max_side=None):
    """Encode an EXIF-corrected RGB preview for an OpenAI vision message."""
    from PIL import Image, ImageOps
    from config.base import VISION_IMAGE_MAX_SIDE
    with Image.open(image_path) as original:
        preview = ImageOps.exif_transpose(original).convert('RGB')
        preview.thumbnail((max_side or VISION_IMAGE_MAX_SIDE,) * 2)
        buffer = io.BytesIO()
        preview.save(buffer, format='JPEG', quality=90)
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}}


def vision_client_from_env(env_path=DEFAULT_ENV_PATH, opener=urlopen):
    """Use the deployed big vision model, preserving its served model alias."""
    from config.base import VISION_MODEL_IDENTITY, VISION_ENABLE_THINKING
    cfg = load_env(env_path)
    url = cfg.get('url') or cfg.get('big_diag_url')
    model = cfg.get('big_model_name')
    if not url or not model:
        return None
    client = OpenAICompatClient(url, cfg.get('key') or cfg.get('big_diag_key'),
                                model, max_tokens=1024, max_retries=0, opener=opener,
                                chat_template_kwargs={'enable_thinking': VISION_ENABLE_THINKING})
    client.model_identity = VISION_MODEL_IDENTITY
    client.model_identity_source = 'configuration'
    return client


def client_from_env(env_path=DEFAULT_ENV_PATH, cascade_enabled=None, opener=urlopen):
    """根据 `.env` 构造级联客户端；缺少配置时返回 ``None``。"""
    cfg = load_env(env_path)
    if cfg.get('small_diag_url') and cfg.get('small_diag_model'):
        small_url = cfg['small_diag_url']
        small_key = cfg.get('small_diag_key')
        small_model = cfg['small_diag_model']
    else:
        small_url = cfg.get('base_url')
        small_key = cfg.get('api_key')
        small_model = cfg.get('small_model_name')
    big_url = cfg.get('url') or cfg.get('big_diag_url')
    big_key = cfg.get('key') or cfg.get('big_diag_key')
    big_model = cfg.get('big_model_name')
    small = (OpenAICompatClient(small_url, small_key, small_model, opener=opener)
             if small_url and small_model else None)
    big = (OpenAICompatClient(big_url, big_key, big_model, opener=opener)
           if big_url and big_model else None)
    if small is None and big is None:
        return None
    if cascade_enabled is None:
        cascade_enabled = str(cfg.get('cascade_enabled', 'true')).lower() in ('1', 'true', 'yes')
    return CascadeClient(small, big, cascade_enabled=cascade_enabled)
