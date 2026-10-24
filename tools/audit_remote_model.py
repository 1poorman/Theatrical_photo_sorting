"""Read model metadata without inference or exposing endpoint credentials.

A served alias, model path or revision is not a verified weight manifest. This
diagnostic records that distinction and never enables persistent model caches.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.llm_client import vision_client_from_env
from core_modules.organize.task_artifacts import atomic_json, fingerprint


def probe(client, route, url):
    result = {'route': route}
    try:
        request = Request(url, headers={'Authorization': 'Bearer ' + client.api_key})
        with urlopen(request, timeout=15) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            result['http_status'] = response.status
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('metadata_response_too_large')
        value = json.loads(raw)
        result['response_sha256'] = hashlib.sha256(raw).hexdigest()
        if not isinstance(value, dict):
            raise ValueError('metadata_object_required')
        result['top_level_fields'] = sorted(str(key)[:120] for key in value)
        models = value.get('data', [])
        if isinstance(models, list):
            result['model_ids'] = [row['id'][:200] for row in models if isinstance(row, dict)
                                   and isinstance(row.get('id'), str)]
        fields = ('model_type', 'architectures', 'revision', 'model_revision',
                  'weights_sha256', 'model_weights_sha256', 'is_generation', 'max_model_len')
        result['declared_metadata'] = {key: value[key] for key in fields
                                      if key in value and isinstance(value[key], (str, bool, int, list))}
        for key in ('model_path', 'root'):
            if isinstance(value.get(key), str):
                result[key + '_sha256'] = hashlib.sha256(value[key].encode()).hexdigest()
    except HTTPError as exc:
        result.update(http_status=exc.code, error_type=type(exc).__name__)
    except (URLError, OSError, ValueError, TypeError) as exc:
        # Exception text can contain private URLs; retain structured types only.
        result.update(error_type=type(exc).__name__,
                      reason_type=type(getattr(exc, 'reason', None)).__name__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('new_empty_output_required')
    client = vision_client_from_env()
    if client is None:
        raise ValueError('vision_endpoint_not_configured')
    output.mkdir(parents=True, exist_ok=True)
    origin = urlsplit(client.base_url)
    root_url = urlunsplit((origin.scheme, origin.netloc, '', '', ''))
    identity = {'endpoint_sha256': hashlib.sha256(client.base_url.encode()).hexdigest(),
                'served_alias': client.model, 'declared_identity': client.model_identity,
                'identity_source': client.model_identity_source}
    atomic_json(output / 'run_state.json', {'status': 'probing', 'identity': identity,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, *sys.argv]})
    results = []
    for route, url in (('/models', client.base_url + '/models'),
                       ('/get_model_info', root_url + '/get_model_info')):
        results.append(probe(client, route, url))
        atomic_json(output / 'probes.json', results)
    reachable = any(isinstance(row.get('http_status'), int) for row in results)
    aliases = [alias for row in results for alias in row.get('model_ids', [])]
    summary = {'status': 'metadata_audit_completed' if reachable else 'transport_failed',
               'identity': identity, 'probes': results, 'configured_alias_listed': client.model in aliases,
               'remote_weights_verified': False, 'verified_weights_sha256': None,
               'persistent_cache_enabled_by_audit': False, 'inference_requests': 0,
               'source_sha256': {name: fingerprint(ROOT / name) for name in
                   ('tools/audit_remote_model.py', 'core_modules/organize/llm_client.py')},
               'limitations': ['Server metadata is a declaration, not a hash of the served tensors.',
                              'No server weight files or authenticated immutable weight manifest were available.',
                              'Unverified caches and task recovery keep their process-private identity.']}
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'run_state.json', {'status': summary['status'], 'identity': identity,
                                          'completed_at': datetime.now(timezone.utc).isoformat()})
    (output / 'report.md').write_text('\n'.join([
        '# Remote model metadata audit', '',
        'Configured alias listed: ' + str(summary['configured_alias_listed']) + '.',
        'Inference requests: 0; verified remote weights: false.', '', *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'configured_alias_listed': summary['configured_alias_listed'],
                      'remote_weights_verified': False, 'probes': results}, ensure_ascii=False))
    raise SystemExit(0 if reachable else 1)


if __name__ == '__main__':
    main()
