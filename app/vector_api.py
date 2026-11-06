"""Incremental vector operations for an explicitly selected image index."""
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from core_modules import qdrant_images
from core_modules.vector_store import VectorStoreError

router = APIRouter(prefix='/api/embedding')


def response(function):
    try:
        return {'code': 200, 'data': function()}
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return JSONResponse(status_code=400, content={'message': str(exc)})
    except VectorStoreError as exc:
        return JSONResponse(status_code=503, content={'message': str(exc)})


@router.get('/index')
def index_status(index_path: str):
    return response(lambda: qdrant_images.status(index_path))


def checked_payload(payload, allowed):
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError('invalid_vector_request_fields')
    if not isinstance(payload.get('index_path'), str) or not payload['index_path']:
        raise ValueError('index_path_required')
    return payload


@router.post('/upsert')
def upsert(payload: dict = Body(...)):
    def run():
        checked_payload(payload, {'index_path', 'image_paths', 'metadata'})
        return qdrant_images.upsert_images(payload['index_path'], payload.get('image_paths'), payload.get('metadata'))
    return response(run)


@router.post('/delete')
def delete(payload: dict = Body(...)):
    def run():
        checked_payload(payload, {'index_path', 'image_paths'})
        return qdrant_images.delete_images(payload['index_path'], payload.get('image_paths'))
    return response(run)
