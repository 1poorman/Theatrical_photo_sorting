"""Read-only photo browsing for the organize workspace; no model initialization."""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

from core_modules.organize.filename_parser import parse_filename
from core_modules.organize.thumbnail_cache import ThumbnailCache

router = APIRouter(prefix='/api/workspace', tags=['整理工作台'])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
thumbnail_cache = ThumbnailCache(
    os.environ.get('TRANS_THUMBNAIL_CACHE_DIR', str(PROJECT_ROOT / 'outputs/thumbnail_cache')),
    max_bytes=int(os.environ.get('TRANS_THUMBNAIL_CACHE_MAX_BYTES', 256 * 1024 * 1024)),
    max_entries=int(os.environ.get('TRANS_THUMBNAIL_CACHE_MAX_ENTRIES', 4096)))


def directory(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise HTTPException(400, '照片目录不存在，请填写服务端可访问的目录。')
    return path


@router.get('/photos')
def photos(input_dir: str, limit: int = Query(10000, ge=1, le=10000)):
    """List direct children, matching the existing organize/reason input scope."""
    root = directory(input_dir)
    try:
        paths = sorted(p for p in root.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
                       and p.resolve().is_relative_to(root))
        items = []
        for path in paths[:limit]:
            meta = parse_filename(path.name)
            items.append({'name': path.name, 'path': str(path), 'size': path.stat().st_size,
                          'metadata': {k: meta.get(k) for k in
                                       ('play', 'date', 'venue', 'people', 'scene', 'photographer')}})
        return {'code': 200, 'data': {'input_dir': str(root), 'photos': items,
                                     'total': len(paths), 'truncated': len(paths) > limit}}
    except PermissionError:
        raise HTTPException(403, '没有读取这个目录的权限。')


@router.get('/thumbnail')
def thumbnail(request: Request, input_dir: str, name: str, size: int = Query(640, ge=160, le=1920)):
    root = directory(input_dir)
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(403, '仅可预览当前照片目录中的图片。')
    if not path.is_file():
        raise HTTPException(404, '照片已移动或不存在。')
    try:
        jpeg, key = thumbnail_cache.preview(path, size)
        headers = {'Cache-Control': 'private, no-cache', 'ETag': '"' + key + '"'}
        tags = request.headers.get('if-none-match', '').split(',')
        if any(tag.strip().removeprefix('W/') in (headers['ETag'], '*') for tag in tags):
            return Response(status_code=304, headers=headers)
        return Response(jpeg, media_type='image/jpeg', headers=headers)
    except ValueError:
        raise HTTPException(409, '照片正在更新，请稍后重试。')
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(422, '无法解码这张图片。')


@router.get('/artifact')
def artifact(output_dir: str, name: str):
    """Download the known workspace reports, including projects outside this repo."""
    if name not in {'scene_labels.json', 'organize_report.json', 'actor_view_report.json', 'face_database_report.json'}:
        raise HTTPException(403, '仅支持下载工作台报告与已确认标签。')
    root = directory(output_dir)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(403, '文件不在指定结果目录中。')
    if not path.is_file():
        raise HTTPException(404, '结果文件尚未生成。')
    return FileResponse(path, media_type='application/json', filename=name)
