"""Persistent scene analysis API; business functions are shared by old routes."""
import hashlib
import json
import os
from pathlib import Path
import threading

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from core_modules.organize.review_store import atomic_rows
from core_modules.organize.task_queue import TaskQueue, TaskConflict, TaskQueueFull, TaskUnavailable, BUSINESS_LOCK
from core_modules.tools.image_io import list_images

router = APIRouter(prefix='/api/organize/tasks')
TASK_ROOT = Path(os.environ.get('TRANS_TASK_DIR', 'outputs/organize_tasks'))
_manager = None
_manager_lock = threading.Lock()


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_scene(params):
    allowed = {'input_dir', 'knowledge_dir', 'output_file', 'top_k', 'use_llm', 'use_embedder'}
    if set(params) - allowed:
        raise ValueError('分析参数含未知字段')
    for field in ('input_dir', 'knowledge_dir'):
        if not isinstance(params.get(field), str) or not os.path.isdir(params[field]):
            raise ValueError(f'{field} 必须为现有目录')
    top_k = params.get('top_k', 5)
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k 必须为 1–100 的整数')
    for field in ('use_llm', 'use_embedder'):
        if type(params.get(field, False)) is not bool:
            raise ValueError(f'{field} 必须为布尔值')
    if not isinstance(params.get('output_file', ''), str):
        raise ValueError('output_file 必须为路径字符串')
    with open(Path(params['knowledge_dir']) / 'entities.json', encoding='utf-8') as stream:
        entities = json.load(stream)
    knowledge = {key: entities.get(key, {} if key == 'role_aliases' else [])
                 for key in ('scenes', 'roles', 'role_aliases')}
    images = [str(Path(params['input_dir'], name).resolve()) for name in list_images(params['input_dir'])]
    if not images:
        raise ValueError('输入目录中没有图片')
    output = str(Path(params.get('output_file') or Path(params['input_dir']) / 'scene_evidence.jsonl').resolve())
    if output in images or not output.endswith('.jsonl'):
        raise ValueError('证据输出必须为独立的 .jsonl 文件')
    # Persist the exact knowledge snapshot and content identity used by this task.
    items = [{'image': image, 'sha256': fingerprint(image)} for image in images]
    prepared = dict(params, knowledge=knowledge, output_file=output, top_k=top_k,
                    input_hashes={item['image']: item['sha256'] for item in items})
    return prepared, items


def scene_handler(params):
    from core_modules.organize.scene_reasoner import SceneReasoner
    client = vision_client = embedder = None
    if params.get('use_llm'):
        from core_modules.organize.llm_client import client_from_env, vision_client_from_env
        vision_client = vision_client_from_env()
        client = vision_client or client_from_env()
    if params.get('use_embedder'):
        # main may be imported as top-level by the legacy entry point.
        import main
        from core_modules.organize.multimodal_evidence import MultimodalRetriever
        embedder = MultimodalRetriever(embedder=main._get_scene_embedder(), cache_dir=main.FEATURE_CACHE_DIR)
    reasoner = SceneReasoner(params['knowledge'], client=client, embedder=embedder,
                             vision_client=vision_client, top_k=params.get('top_k', 5))

    def process(item):
        path = item['image']
        if fingerprint(path) != item['sha256']:
            raise ValueError('source_content_changed')
        row = reasoner.reason(path)
        if fingerprint(path) != item['sha256']:
            raise ValueError('source_content_changed_during_analysis')
        return row

    def finish(rows):
        for row in rows:
            if not row.get('error') and fingerprint(row['image']) != params['input_hashes'][row['image']]:
                raise ValueError('completed_source_content_changed; 请创建新的分析任务。')
        path = Path(params['output_file'])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_rows(path, rows)
        if embedder is not None:
            embedder.save()
        return {'n_images': len(rows), 'output_file': str(path),
                'n_unknown': sum(row['label'] == 'unknown' for row in rows),
                'n_failed': sum(bool(row.get('error')) for row in rows),
                'labels': [row['label'] for row in rows],
                'n_visual': sum(row.get('retrieval_mode') == 'visual' for row in rows),
                'llm': client is not None, 'visual': embedder is not None}

    return process, finish


def execute_scene(params):
    """Same scene executor for legacy synchronous API and agent tools."""
    prepared, items = prepare_scene(params)
    with BUSINESS_LOCK:
        process, finish = scene_handler(prepared)
        rows = []
        for item in items:
            try:
                rows.append(process(item))
            except Exception as exc:
                rows.append({'image': item['image'], 'label': 'unknown', 'scene_id': 'unknown',
                             'decision': 'error', 'confidence': None,
                             'review': {'status': 'pending', 'reviewer': None, 'updated_at': None},
                             'error': {'type': type(exc).__name__, 'message': str(exc)}})
        return finish(rows)


def manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TaskQueue(TASK_ROOT, {'scene_reason': scene_handler},
                                 capacity=int(os.environ.get('TRANS_TASK_CAPACITY', '8')))
        return _manager


def shutdown():
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.close()
            _manager = None


def response(operation, status=200):
    try:
        return JSONResponse(status_code=status, content={'code': 200, 'data': operation()})
    except KeyError as exc:
        return JSONResponse(status_code=404, content={'message': str(exc)})
    except TaskConflict as exc:
        return JSONResponse(status_code=409, content={'message': str(exc)})
    except (TaskQueueFull, TaskUnavailable) as exc:
        return JSONResponse(status_code=503, content={'message': str(exc)})
    except (ValueError, OSError) as exc:
        return JSONResponse(status_code=400, content={'message': str(exc)})


@router.post('')
def create_task(body: dict = Body(...)):
    def create():
        if set(body) - {'kind', 'params', 'idempotency_key'} or body.get('kind') != 'scene_reason':
            raise ValueError('当前后台任务支持 kind=scene_reason')
        params = body.get('params')
        if not isinstance(params, dict):
            raise ValueError('params 必须为对象')
        key = body.get('idempotency_key')
        if key is not None and not isinstance(key, str):
            raise ValueError('idempotency_key 必须为字符串')
        prepared, items = prepare_scene(params)
        task, created = manager().submit('scene_reason', prepared, items, key, prepared['output_file'])
        return dict(task, created=created)
    return response(create, 202)


@router.get('')
def list_tasks(limit: int = 50):
    return response(lambda: {'tasks': manager().list(limit)})


@router.get('/{task_id}')
def get_task(task_id: str):
    return response(lambda: manager().get(task_id))


@router.post('/{task_id}/cancel')
def cancel_task(task_id: str):
    return response(lambda: manager().cancel(task_id))


@router.get('/{task_id}/items')
def task_items(task_id: str, limit: int = 50, offset: int = 0):
    return response(lambda: {'items': manager().item_page(task_id, limit, offset)})


@router.post('/{task_id}/resume')
def resume_task(task_id: str):
    return response(lambda: manager().resume(task_id), 202)
