# -*- coding: utf-8 -*-
"""
TensorRT 加速基准测试
对比 cpu / cuda / tensorrt 三种设备下 SCRFD 检测 + ArcFace 特征提取的耗时，
并验证各设备结果的一致性（检测框 IoU、特征余弦相似度）。

用法（face_scrfd_arcface 环境）：
    python tests/test_trt_speed.py [image_path]
"""
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'face_recognition'))

import cv2
import numpy as np

SCRFD_MODEL = './weights/scrfd/scrfd_10g_bnkps.onnx'
ARCFACE_MODEL = './weights/arcface/Glint100.onnx'
WARMUP = 3
REPEAT = 20


def bench_detector(device, img):
    from scrfd.scrfd_det import SCRFDDetector
    det = SCRFDDetector(SCRFD_MODEL, device=device, det_thresh=0.3)
    for _ in range(WARMUP):
        boxes, kpss = det.detect(img)
    t0 = time.time()
    for _ in range(REPEAT):
        boxes, kpss = det.detect(img)
    dt = (time.time() - t0) / REPEAT * 1000
    print(f"[SCRFD    ] device={device:<9} avg={dt:8.2f} ms  faces={boxes.shape[0]}")
    return boxes, kpss


def bench_extractor(device, face):
    from arcface.arcface_onnx import ArcFaceFeatureExtractor
    ext = ArcFaceFeatureExtractor(ARCFACE_MODEL, device=device)
    for _ in range(WARMUP):
        feat = ext.extract_features([face])
    t0 = time.time()
    for _ in range(REPEAT):
        feat = ext.extract_features([face])
    dt = (time.time() - t0) / REPEAT * 1000
    print(f"[ArcFace  ] device={device:<9} avg={dt:8.2f} ms  dim={feat.shape[1]}")
    return feat


def cos_sim(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_images/4.jpg'
    img = cv2.imread(img_path)
    assert img is not None, f'cannot read {img_path}'
    print(f'image: {img_path} {img.shape[1]}x{img.shape[0]}\n')

    results = {}
    for device in ('cpu', 'cuda', 'tensorrt'):
        try:
            results[device] = {}
            results[device]['det'] = bench_detector(device, img)
        except Exception as e:
            print(f"[SCRFD    ] device={device:<9} FAILED: {type(e).__name__}: {str(e)[:120]}")
            results[device] = None
            continue
        # 用 cuda 检出的最大脸做 ArcFace 基准输入
        try:
            boxes, kpss = results[device]['det']
            if boxes.shape[0] > 0:
                from arcface.arcface_onnx import norm_crop
                lmk = kpss[0]
                face = norm_crop(img, lmk, 112)
                results[device]['feat'] = bench_extractor(device, face)
            else:
                print(f"[ArcFace  ] device={device:<9} skipped (no face)")
        except Exception as e:
            print(f"[ArcFace  ] device={device:<9} FAILED: {type(e).__name__}: {str(e)[:120]}")

    # 一致性校验（以 cuda 为基准）
    base = results.get('cuda')
    if base and 'det' in base and results.get('tensorrt'):
        print('\n--- 一致性校验 (tensorrt vs cuda) ---')
        b_base, _ = base['det']
        b_trt, _ = results['tensorrt']['det']
        if b_base.shape[0] == b_trt.shape[0] and b_base.shape[0] > 0:
            ious = [iou(b, b_trt).max() for b in b_base]
            print(f'检测框数量一致: {b_base.shape[0]}, 平均 IoU: {np.mean(ious):.4f}')
        if 'feat' in base and 'feat' in results['tensorrt']:
            sim = cos_sim(base['feat'], results['tensorrt']['feat'])
            print(f'特征余弦相似度 (FP16 vs FP32): {sim:.4f}')
            print('（FP16 量化通常导致 0.001~0.01 的微小差异，属正常）')
