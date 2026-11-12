"""On-demand age estimation for uploads and workspace photos."""
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import uuid

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional

from app.workspace_api import directory, IMAGE_EXTENSIONS
from config.base import OUTPUT_DIR, TEMP_DIR
from core_modules.face_recognition.age_estimation import FaceAgeEstimator
from core_modules.organize.task_queue import serialized


router = APIRouter(prefix='/api/face', tags=['人脸年龄估计'])


@lru_cache(maxsize=1)
def age_estimator():
    return FaceAgeEstimator()


@serialized
def estimate_and_save(image_path):
    report, annotated, crops = age_estimator().estimate(image_path=str(image_path))
    destination = Path(OUTPUT_DIR) / 'face_age_results' / uuid.uuid4().hex
    destination.mkdir(parents=True)
    annotated_path = destination / 'annotated.jpg'
    if not cv2.imwrite(str(annotated_path), annotated):
        raise OSError('无法保存年龄估计标注图。')
    report['annotated_image_path'] = str(annotated_path)
    for face, crop in zip(report['faces'], crops):
        crop_path = destination / f"face_{face['face_id']}.jpg"
        if not cv2.imwrite(str(crop_path), crop):
            raise OSError('无法保存人脸预览。')
        face['face_image_path'] = str(crop_path)
    return report


@router.post('/age')
def estimate_face_ages(image: Optional[UploadFile] = File(None),
                       input_dir: Optional[str] = Form(None), name: Optional[str] = Form(None)):
    """Upload an image, or supply a workspace input_dir and image name.

    Detects all eligible faces without identity lookup. Estimates are in years;
    detection_confidence describes face detection, not age accuracy.
    """
    if image is not None and (input_dir is not None or name is not None):
        raise HTTPException(400, '上传图片与工作台路径只能选择一种。')
    temporary = None
    try:
        if image is not None:
            Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(prefix='face-age-', dir=TEMP_DIR)
            image_path = Path(temporary.name) / 'input_image'
            with image_path.open('wb') as stream:
                shutil.copyfileobj(image.file, stream)
        else:
            if not input_dir or not name:
                raise HTTPException(400, '请上传图片，或提供当前照片目录与文件名。')
            root = directory(input_dir)
            image_path = (root / name).resolve()
            if not image_path.is_relative_to(root) or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise HTTPException(403, '仅可分析当前照片目录中的图片。')
            if not image_path.is_file():
                raise HTTPException(404, '照片已移动或不存在。')
        return {'code': 200, 'data': estimate_and_save(image_path)}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(503, f'年龄估计模型文件不可用：{exc}') from exc
    except (ImportError, RuntimeError) as exc:
        raise HTTPException(503, f'年龄估计模型暂不可用：{exc}') from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if temporary is not None:
            temporary.cleanup()
