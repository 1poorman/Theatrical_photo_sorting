"""Read-only photo browsing for the organize workspace; no model initialization."""
from io import BytesIO
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from core_modules.organize.filename_parser import parse_filename

router = APIRouter(prefix='/api/workspace', tags=['整理工作台'])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


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
            meta = parse_filename(path.name, done_style=bool(re.match(r'^\d{1,3}-\d{8}', path.name)))
            items.append({'name': path.name, 'path': str(path), 'size': path.stat().st_size,
                          'metadata': {k: meta.get(k) for k in
                                       ('play', 'date', 'venue', 'people', 'scene', 'photographer')}})
        return {'code': 200, 'data': {'input_dir': str(root), 'photos': items,
                                     'total': len(paths), 'truncated': len(paths) > limit}}
    except PermissionError:
        raise HTTPException(403, '没有读取这个目录的权限。')


@router.get('/thumbnail')
def thumbnail(input_dir: str, name: str, size: int = Query(640, ge=160, le=1920)):
    root = directory(input_dir)
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(403, '仅可预览当前照片目录中的图片。')
    if not path.is_file():
        raise HTTPException(404, '照片已移动或不存在。')
    try:
        with Image.open(path) as source:
            source.draft('RGB', (size, size))
            picture = ImageOps.exif_transpose(source)
            picture.thumbnail((size, size))
            buffer = BytesIO()
            picture.convert('RGB').save(buffer, format='JPEG', quality=85)
        return Response(buffer.getvalue(), media_type='image/jpeg',
                        headers={'Cache-Control': 'private, max-age=300'})
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(422, '无法解码这张图片。')


@router.get('/artifact')
def artifact(output_dir: str, name: str):
    """Download the known workspace reports, including projects outside this repo."""
    if name not in {'scene_labels.json', 'organize_report.json', 'actor_view_report.json'}:
        raise HTTPException(403, '仅支持下载整理报告与已确认标签。')
    root = directory(output_dir)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(403, '文件不在指定结果目录中。')
    if not path.is_file():
        raise HTTPException(404, '结果文件尚未生成。')
    return FileResponse(path, media_type='application/json', filename=name)
