# -*- coding: utf-8 -*-
"""验证批量优化：ArcFace 批量推理 + 多线程预解码流水线"""
import os, sys, time
BASE = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(BASE)

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('frs', os.path.join(BASE, 'core_modules/face_recognition/face_recognition.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
FaceRecognitionSystem = mod.FaceRecognitionSystem

if __name__ == '__main__':
    r = FaceRecognitionSystem(
        os.path.join(BASE, 'weights/scrfd/scrfd_10g_bnkps.onnx'),
        os.path.join(BASE, 'weights/arcface/Glint100.onnx'),
        device='tensorrt')

    p = os.path.join(BASE, 'data/sample_images/4.jpg')
    results, _ = r.recognize_face(p, known_threshold=0.55)

    # 收集多张对齐人脸，验证批量提取
    faces = [res['face_image'] for res in results]
    print(f'\n批量提取验证: {len(faces)} 张脸')
    t0 = time.time()
    for _ in range(20):
        feats_batch = r.extractor.extract_features(faces)
    t_batch = (time.time() - t0) / 20 * 1000
    t0 = time.time()
    for _ in range(20):
        feats_single = np.vstack([r.extractor.extract_features([f]) for f in faces])
    t_single = (time.time() - t0) / 20 * 1000
    print(f'批量(b={len(faces)}): {t_batch:.2f} ms, 逐张x{len(faces)}: {t_single:.2f} ms')
    sim = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                         for a, b in zip(feats_batch, feats_single)]))
    print(f'批量 vs 逐张 特征余弦相似度: {sim:.6f}')
