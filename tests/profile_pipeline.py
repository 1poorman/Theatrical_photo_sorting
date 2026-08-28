# -*- coding: utf-8 -*-
"""端到端各环节耗时剖析，定位瓶颈"""
import os, sys, time
BASE = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(BASE)

import cv2
import numpy as np
from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
from core_modules.face_recognition.utils.assess import assess_face_quality_simple

ROOT = BASE
IMG = os.path.join(ROOT, 'data/sample_images/4.jpg')


def bench(name, fn, n=20):
    fn()  # warmup
    t0 = time.time()
    for _ in range(n):
        fn()
    print(f'{name:<32} {(time.time() - t0) / n * 1000:8.2f} ms')


if __name__ == '__main__':
    det = SCRFDDetector(os.path.join(ROOT, 'weights/scrfd/scrfd_10g_bnkps.onnx'),
                        device='tensorrt', det_thresh=0.5)
    ext = ArcFaceFeatureExtractor(os.path.join(ROOT, 'weights/arcface/Glint100.onnx'),
                                  device='tensorrt')
    print('--- 分环节耗时 ---')
    bench('cv2.imread (5030x3357)', lambda: cv2.imread(IMG))

    image = cv2.imread(IMG)
    h, w = image.shape[:2]
    scale = 1920 / max(h, w)
    img1920 = cv2.resize(image, (int(w * scale), int(h * scale)))
    bench('resize -> 1920', lambda: cv2.resize(image, (img1920.shape[1], img1920.shape[0])))

    bench('SCRFD.detect 全流程', lambda: det.detect(img1920))

    # detect 内部环节
    def _pre():
        s = 640 / max(img1920.shape[:2])
        small = cv2.resize(img1920, (int(img1920.shape[1] * s), int(img1920.shape[0] * s)))
        return cv2.dnn.blobFromImage(small, 1 / 128., (640, 640), (127.5,) * 3, swapRB=True)
    bench('  预处理 resize+blob', _pre)
    blob = _pre()
    bench('  session.run (TRT)', lambda: det.session.run(det.output_names, {det.input_name: blob}))

    # forward 解码：需要 640x640 pad 图
    s = 640 / max(img1920.shape[:2])
    small = cv2.resize(img1920, (int(img1920.shape[1] * s), int(img1920.shape[0] * s)))
    det_img = np.zeros((640, 640, 3), dtype=np.uint8)
    det_img[:small.shape[0], :small.shape[1], :] = small
    bench('  解码+NMS', lambda: det.forward(det_img, 0.5))

    faces = det._detect_single(img1920)
    face = faces[0]
    aligned = ext.align_face(img1920, face.landmark)
    bench('ArcFace 单脸 (TRT)', lambda: ext.extract_features([aligned]))
    bench('质量评估 simple', lambda: assess_face_quality_simple(img1920[100:400, 100:400]))
    bench('align_face (warpAffine)', lambda: ext.align_face(img1920, face.landmark))
