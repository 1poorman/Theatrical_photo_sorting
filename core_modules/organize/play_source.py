# -*- coding: utf-8 -*-
"""剧目信息来源管理：URL 解析、受限抓取、robots 合规、重试与离线快照。

该模块只依赖 Python 标准库，确保在无网络、无额外爬虫依赖的环境中仍能使用
``剧目信息.txt`` 和本地 fixture 建立知识库；网络不可达时可从已保存快照离线运行。
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

URL_RE = re.compile(r'https?://[^\s<>"\u3001，。]+')
DEFAULT_ALLOWED_HOSTS = {'www.chncpa.org', 'chncpa.org'}
USER_AGENT = 'TheatricalPhotoSorting/1.0'


def extract_urls(info_path):
    """从剧目信息文本读取 URL，保持出现顺序并去重。"""
    with open(info_path, encoding='utf-8') as f:
        text = f.read()
    out = []
    for url in URL_RE.findall(text):
        url = url.rstrip('。.,，')
        if url not in out:
            out.append(url)
    return out


def validate_url(url, allowed_hosts=None):
    """只允许 http(s) 且默认限制到 NCPA 域名。"""
    parsed = urlparse(url)
    hosts = set(allowed_hosts or DEFAULT_ALLOWED_HOSTS)
    host = (parsed.hostname or '').lower().rstrip('.')
    if parsed.scheme not in ('http', 'https') or not host:
        raise ValueError(f'unsupported source URL: {url}')
    if host not in hosts and not any(host.endswith('.' + h) for h in hosts):
        raise ValueError(f'URL host is not allowed: {host}')
    return url


def source_manifest(info_path, urls=None, allowed_hosts=None):
    """生成不含敏感信息的来源清单。"""
    urls = urls if urls is not None else extract_urls(info_path)
    checked = [validate_url(u, allowed_hosts) for u in urls]
    return {
        'info_path': os.path.abspath(info_path),
        'urls': checked,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'parser': 'play_source/v1',
    }


class CrawlPolicy:
    """抓取策略：域名白名单、限速、大小/页数上限与 robots 合规。"""

    def __init__(self, allowed_hosts=None, user_agent=USER_AGENT, timeout=20,
                 max_bytes=20 * 1024 * 1024, delay_seconds=1.0, max_retries=2,
                 max_pages=50, respect_robots=True):
        self.allowed_hosts = set(allowed_hosts or DEFAULT_ALLOWED_HOSTS)
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.max_pages = max_pages
        self.respect_robots = respect_robots

    def as_dict(self):
        return {'allowed_hosts': sorted(self.allowed_hosts),
                'user_agent': self.user_agent, 'timeout': self.timeout,
                'max_bytes': self.max_bytes, 'delay_seconds': self.delay_seconds,
                'max_retries': self.max_retries, 'max_pages': self.max_pages,
                'respect_robots': self.respect_robots}


def robots_allows(url, robots_text, user_agent=USER_AGENT):
    """给定 robots.txt 文本判断 URL 是否允许抓取；无 robots 文本时放行。"""
    if robots_text is None:
        return True
    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    return parser.can_fetch(user_agent, url)


def fetch_robots(url, opener=urlopen, timeout=20, user_agent=USER_AGENT):
    """抓取站点 robots.txt；失败返回 None（按允许处理并记录）。"""
    parsed = urlparse(url)
    robots_url = f'{parsed.scheme}://{parsed.netloc}/robots.txt'
    req = Request(robots_url, headers={'User-Agent': user_agent})
    try:
        with opener(req, timeout=timeout) as response:
            return response.read(512 * 1024).decode('utf-8', errors='replace')
    except Exception:
        return None


def fetch_url(url, output_dir, allowed_hosts=None, timeout=20, max_bytes=20 * 1024 * 1024,
              delay_seconds=1.0, opener=urlopen, max_retries=0,
              user_agent=USER_AGENT):
    """抓取单个 URL 并保存快照，失败按 ``max_retries`` 退避重试。

    返回 manifest 条目。网络错误不会被吞掉，便于上层记录后切换到离线来源。
    """
    validate_url(url, allowed_hosts)
    last_err = None
    for attempt in range(max_retries + 1):
        if delay_seconds:
            time.sleep(delay_seconds)
        try:
            req = Request(url, headers={'User-Agent': user_agent})
            with opener(req, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise ValueError(f'source exceeds max_bytes: {url}')
                status = getattr(response, 'status', 200)
                content_type = response.headers.get('Content-Type', '')
            digest = hashlib.sha256(body).hexdigest()
            os.makedirs(output_dir, exist_ok=True)
            snapshot = os.path.join(output_dir, f'{digest[:16]}.html')
            with open(snapshot, 'wb') as f:
                f.write(body)
            return {
                'url': url,
                'status': status,
                'content_type': content_type,
                'bytes': len(body),
                'sha256': digest,
                'snapshot': snapshot,
                'attempts': attempt + 1,
                'fetched_at': datetime.now(timezone.utc).isoformat(),
            }
        except HTTPError as e:
            if 400 <= e.code < 500:      # 客户端错误不重试
                raise
            last_err = e
        except (URLError, TimeoutError, OSError) as e:
            last_err = e
        if attempt < max_retries:
            time.sleep(min(2 ** attempt, 4))
    raise last_err


def crawl(urls, output_dir, policy=None, opener=urlopen):
    """按策略抓取多个 URL，返回 manifest（含成功页、跳过与错误）。"""
    policy = policy or CrawlPolicy()
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'policy': policy.as_dict(), 'pages': [], 'skipped': [], 'errors': []}
    seen, robot_cache = set(), {}
    for url in urls:
        if len(manifest['pages']) >= policy.max_pages:
            manifest['skipped'].append({'url': url, 'reason': 'max_pages'})
            continue
        try:
            validate_url(url, policy.allowed_hosts)
        except ValueError as e:
            manifest['errors'].append({'url': url, 'error': str(e)})
            continue
        if url in seen:
            manifest['skipped'].append({'url': url, 'reason': 'duplicate'})
            continue
        seen.add(url)
        host = urlparse(url).netloc
        if policy.respect_robots:
            if host not in robot_cache:
                robot_cache[host] = fetch_robots(url, opener=opener,
                                                 timeout=policy.timeout,
                                                 user_agent=policy.user_agent)
            if not robots_allows(url, robot_cache[host], policy.user_agent):
                manifest['skipped'].append({'url': url, 'reason': 'robots'})
                continue
        try:
            entry = fetch_url(url, output_dir, allowed_hosts=policy.allowed_hosts,
                              timeout=policy.timeout, max_bytes=policy.max_bytes,
                              delay_seconds=policy.delay_seconds, opener=opener,
                              max_retries=policy.max_retries,
                              user_agent=policy.user_agent)
            manifest['pages'].append(entry)
        except Exception as e:
            manifest['errors'].append({'url': url, 'error': f'{type(e).__name__}: {e}'})
    os.makedirs(output_dir, exist_ok=True)
    write_manifest(os.path.join(output_dir, 'source_manifest.json'), manifest)
    return manifest


def load_snapshot(snapshot_path, encoding='utf-8'):
    """读取离线快照，同时返回内容哈希用于稳定性校验。"""
    with open(snapshot_path, 'rb') as f:
        body = f.read()
    text = body.decode(encoding, errors='replace')
    return {'path': os.path.abspath(snapshot_path), 'text': text,
            'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_local_sources(info_path, play_root):
    """返回本地剧目信息文本和可作为媒体来源的图片文件。

    ``剧组信息``、``剧目信息/剧照``、``剧目信息/主演`` 等目录均被保留，
    后续知识抽取模块按目录语义区分用途。
    """
    with open(info_path, encoding='utf-8') as f:
        text = f.read()
    image_paths = []
    for root, _, files in os.walk(play_root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                image_paths.append(os.path.join(root, name))
    return {'info_path': os.path.abspath(info_path), 'text': text,
            'image_paths': sorted(image_paths)}
