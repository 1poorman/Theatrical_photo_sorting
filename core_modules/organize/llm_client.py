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
                 temperature=0.0, max_tokens=1024, opener=urlopen):
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

    def chat(self, messages, temperature=None, max_tokens=None):
        """返回 ``{'content': str, 'trace': dict}``；失败抛 ``LLMError``。"""
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature if temperature is None else temperature,
            'max_tokens': self.max_tokens if max_tokens is None else max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        prompt_text = json.dumps(messages, ensure_ascii=False)
        url = f'{self.base_url}/chat/completions'
        last_err = None
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
                trace = {
                    'model': self.model,
                    'endpoint': self.base_url,
                    'prompt_hash': _hash(prompt_text),
                    'response_hash': _hash(content),
                    'temperature': payload['temperature'],
                    'attempts': attempt + 1,
                    'elapsed_ms': int((time.time() - started) * 1000),
                }
                return {'content': content, 'trace': trace}
            except (HTTPError, URLError, TimeoutError, OSError, ValueError,
                    KeyError, IndexError, json.JSONDecodeError) as e:
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
