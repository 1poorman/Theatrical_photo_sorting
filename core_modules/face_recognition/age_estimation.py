"""SCRFD detection and SwinFace age estimates, independent of an identity database."""
import threading

import cv2
import numpy as np

from config.base import (
    FACE_DETECT_MIN_SCORE, FACE_IMAGE_MAX_SIDE, FACE_MIN_SIZE,
    SCRFD_MODEL_PATH, SWINFACE_BATCH_SIZE, SWINFACE_DEVICE, SWINFACE_MODEL_PATH,
)
from core_modules.face_recognition.arcface.arcface_onnx import norm_crop
from core_modules.tools.image_io import imread_reduced, resize_max_side


AGE_NOTE = '年龄为模型对照片中外观的估计，妆造、姿态与画质会影响结果。'


def age_fields(value):
    """Keep raw regression output for audit; do not silently clamp invalid ages."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('SwinFace returned a non-finite age estimate')
    valid = 0 <= value <= 120
    return {'estimated_age': round(value, 1) if valid else None,
            'age_model_output': value,
            'age_status': 'estimated' if valid else 'out_of_range'}


class FaceAgeEstimator:
    def __init__(self, checkpoint=SWINFACE_MODEL_PATH, device=SWINFACE_DEVICE,
                 batch_size=SWINFACE_BATCH_SIZE, detector=None, analyzer=None):
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self._detector = detector
        self._analyzer = analyzer
        self._lock = threading.RLock()

    @property
    def analyzer(self):
        with self._lock:
            if self._analyzer is None:
                from core_modules.face_recognition.swinface import SwinFaceAnalyzer
                self._analyzer = SwinFaceAnalyzer(self.checkpoint, self.device, self.batch_size)
            return self._analyzer

    @property
    def detector(self):
        with self._lock:
            if self._detector is None:
                from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
                # A small CPU detector also makes the age-only endpoint usable
                # without ONNX CUDA/TensorRT libraries or a loaded face database.
                self._detector = SCRFDDetector(SCRFD_MODEL_PATH, device='cpu',
                                               det_thresh=FACE_DETECT_MIN_SCORE)
            return self._detector

    def estimate_aligned(self, crops):
        if not crops:
            return []
        with self._lock:
            values = self.analyzer.estimate_ages(crops)
        if len(values) != len(crops):
            raise RuntimeError('SwinFace age outputs do not match input faces')
        return [age_fields(value) for value in values]

    def estimate(self, image=None, image_path=None):
        """Return JSON-ready metadata, an annotated image, and aligned crops.

        Bounding boxes are in the returned analysis image's coordinates, whose
        longest side is at most FACE_IMAGE_MAX_SIDE. Input images are not modified.
        """
        if image is None:
            image = imread_reduced(str(image_path), target_max=FACE_IMAGE_MAX_SIDE)
        if image is None or image.size == 0:
            raise ValueError('无法解码图片，请选择有效的图片文件。')
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Age estimation expects a uint8 BGR image')
        image = resize_max_side(image, FACE_IMAGE_MAX_SIDE)
        height, width = image.shape[:2]
        with self._lock:
            detected = self.detector._detect_single(image)
        faces, crops = [], []
        for face in sorted(detected, key=lambda face: (face.x, face.y)):
            if (not np.isfinite([face.x, face.y, face.r, face.b, face.score]).all()
                    or face.score < FACE_DETECT_MIN_SCORE
                    or min(face.width, face.height) < FACE_MIN_SIZE):
                continue
            x1, y1 = max(0, int(face.x)), max(0, int(face.y))
            x2, y2 = min(width, int(face.r) + 1), min(height, int(face.b) + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = np.asarray(face.landmark, dtype=np.float32)
            valid_landmarks = (landmarks.shape == (5, 2) and np.isfinite(landmarks).all()
                               and np.linalg.matrix_rank(landmarks - landmarks.mean(axis=0)) == 2)
            if valid_landmarks:
                crop = norm_crop(image, landmarks, image_size=112)
                alignment = 'five_point'
            else:
                crop = cv2.resize(image[y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_LINEAR)
                alignment = 'bbox_resize'
            crops.append(crop)
            faces.append({'face_id': len(faces) + 1, 'bbox': [x1, y1, x2, y2],
                          'detection_confidence': float(face.score), 'alignment': alignment,
                          'small_face': min(x2 - x1, y2 - y1) < 40})
        estimates = self.estimate_aligned(crops)
        annotated = image.copy()
        for face, estimate in zip(faces, estimates):
            face.update(estimate)
            x1, y1, x2, y2 = face['bbox']
            age = estimate['estimated_age']
            label = f"#{face['face_id']} ~{age:.0f}y" if age is not None else f"#{face['face_id']} unavailable"
            cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (90, 210, 80), 2)
            cv2.putText(annotated, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (90, 210, 80), 2, cv2.LINE_AA)
        return {
            'faces': faces, 'face_count': len(faces), 'image_width': width, 'image_height': height,
            'bbox_coordinate_space': 'analysis_image', 'age_unit': 'years', 'note': AGE_NOTE,
            'model': self.analyzer.metadata() if faces else {'name': 'SwinFace', 'loaded': self._analyzer is not None},
        }, annotated, crops
