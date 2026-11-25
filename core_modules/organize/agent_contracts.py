"""Shared agent validation, observable outcomes and bounded prompt data."""
import hashlib
import json
from pathlib import Path
import re

CONTRACT_VERSION = 'photo-agent-v2'
SENSITIVE_KEY_RE = re.compile(
    r'(?:^|_)(?:api_?key|key|token|password|passwd|secret|authorization)(?:$|_)', re.I)
ASSIGNMENT_RE = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*'
    r'(?P<value>"[^"]*"|\'[^\']*\'|[^\s,，;；]+)')
COMPLETION_RE = re.compile(
    r'(?:已经|已)(?:成功)?(?:完成|执行|整理|导出|生成|创建|复制|扫描|分析|建库)'
    r'|\b(?:completed|exported|generated|organized|created|saved)\b', re.I)
ARTIFACT_FIELDS = ('output_file', 'output_dir', 'knowledge_dir', 'report_path',
                   'review_dir', 'db_root', 'scene_evidence_file')
PATH_FIELDS = set(ARTIFACT_FIELDS) | {
    'input_dir', 'info_path', 'play_root', 'evidence_file', 'image_path',
    'identity_registry_file', 'organized_dirs'}


def redact(value):
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()
                if not SENSITIVE_KEY_RE.search(str(key)) or key in ('token_usage', 'idempotency_key', 'cache_key')}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return json.dumps(redact(parsed), ensure_ascii=False)
        return ASSIGNMENT_RE.sub(
            lambda match: match['key'] + '=[REDACTED]'
            if SENSITIVE_KEY_RE.search(match['key']) else match[0], value)
    return value


def bounded_json(value, limit=12000):
    original = json.dumps(value, ensure_ascii=False, default=str)
    if len(original) <= limit:
        return original

    def compact(item, width):
        if isinstance(item, dict):
            return {key: compact(child, width) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [compact(child, width) for child in item[:width]]
        if isinstance(item, str):
            return item if len(item) <= width * 100 else item[:width * 100] + '…'
        return item

    for width in (10, 3, 1):
        reduced = compact(value, width)
        if isinstance(reduced, dict):
            reduced['truncated'] = True
        encoded = json.dumps(reduced, ensure_ascii=False, default=str)
        if len(encoded) <= limit:
            return encoded
    priority = ('ok', 'status', 'state', 'tool', 'task_id', 'business_task_id',
                'error', 'outcome', 'result') + ARTIFACT_FIELDS
    reduced = {key: compact(value[key], 1) for key in priority if isinstance(value, dict) and key in value}
    encoded = json.dumps(dict(reduced, truncated=True), ensure_ascii=False, default=str)
    if len(encoded) <= limit:
        return encoded
    return json.dumps({'truncated': True, 'message': '完整结果保存在任务或会话记录中'}, ensure_ascii=False)


def validate_arguments(schema, arguments, memory=None, coerce=False):
    from jsonschema import Draft202012Validator
    if not isinstance(arguments, dict):
        raise ValueError('工具参数必须为 JSON 对象')
    properties = schema.get('properties', {})
    merged = {key: item for key, item in (memory or {}).items()
              if key in schema.get('required', []) and key in properties}
    merged.update(arguments)
    if coerce:
        for key, item in list(merged.items()):
            expected = properties.get(key, {}).get('type')
            if isinstance(item, str):
                try:
                    if expected == 'integer':
                        merged[key] = int(item)
                    elif expected == 'number':
                        merged[key] = float(item)
                    elif expected == 'boolean' and item.lower() in ('true', '1', 'yes', 'on', 'false', '0', 'no', 'off'):
                        merged[key] = item.lower() in ('true', '1', 'yes', 'on')
                except ValueError:
                    pass
    missing = [key for key in schema.get('required', []) if merged.get(key) in (None, '')]
    if missing:
        return merged, missing
    errors = sorted(Draft202012Validator(dict(schema, additionalProperties=False)).iter_errors(merged),
                    key=lambda error: str(list(error.path)))
    if errors:
        raise ValueError('参数校验失败：' + '; '.join(error.message for error in errors))
    return merged, []


def outcome(result):
    data = result if isinstance(result, dict) else {}
    business = data.get('business_task_id') or (data.get('task_id') if data.get('kind') else None)
    status = data.get('status') if business else None
    payload = data.get('result') if business else data
    payload = payload if isinstance(payload, dict) else {}
    failed = max((value for value in (data.get('failed'), payload.get('failed'), payload.get('n_failed'))
                  if type(value) is int and value >= 0), default=0)
    total = next((payload[key] for key in ('total_input', 'n_images', 'total')
                  if type(payload.get(key)) is int), data.get('total'))
    if status in ('queued', 'running', 'cancel_requested'):
        execution = 'submitted'
    elif status in ('interrupted', 'cancelled'):
        execution = status
    elif status == 'failed' or data.get('ok') is False or (failed and type(total) is int and failed >= total):
        execution = 'failed'
    elif failed or status == 'completed_with_errors':
        execution = 'partial'
    else:
        execution = 'succeeded'
    artifacts = [{'kind': key, 'path': payload[key]} for key in ARTIFACT_FIELDS
                 if isinstance(payload.get(key), str) and payload[key]]
    review = bool(data.get('requires_human_review') or payload.get('requires_human_review')
                  or data.get('kind') == 'scene_reason')
    return {'status': execution, 'ok': execution in ('submitted', 'succeeded'),
            'failed': failed, 'total': total, 'business_task_id': business,
            'artifacts': artifacts, 'requires_human_review': review,
            'evidence_type': payload.get('evidence_type') or data.get('evidence_type'),
            'error': data.get('error') or payload.get('error')}


def request_key(project_id, spec, arguments):
    versions = {}
    for key, item in arguments.items():
        if key not in PATH_FIELDS:
            continue
        paths = item if isinstance(item, list) else [item]
        for value in paths:
            if not isinstance(value, str):
                continue
            path = Path(value).expanduser().resolve()
            if not path.exists() or key in ('output_file', 'output_dir', 'report_path', 'review_dir') or (
                    key == 'db_root' and spec['name'] == 'build_face_database'):
                continue
            sources = sorted(path.rglob('*') if key == 'db_root' else path.iterdir()) if path.is_dir() else [path]
            versions[str(path)] = [(str(source), source.stat().st_size,
                                    source.stat().st_mtime_ns, source.stat().st_ctime_ns)
                                   for source in sources if source.is_file()]
    value = [CONTRACT_VERSION, project_id, spec['name'], spec['parameters'], arguments, versions]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
