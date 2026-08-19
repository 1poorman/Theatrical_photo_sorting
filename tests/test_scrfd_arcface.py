# -*- coding: utf-8 -*-
"""
SCRFD + ArcFace 端到端测试脚本
1. 从 data/face_database 构建人脸库（本地向量，不依赖 ES）
2. 对 data/sample_images 中的大图检测 + 识别
3. 统计同人/异人相似度分布
"""
import os
import sys
import time
import glob
import numpy as np
import cv2
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from face_recognition.scrfd.scrfd_det import SCRFDDetector
from face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor

SCRFD_PATH = os.path.join(PROJECT_ROOT, 'weights/scrfd/scrfd_10g_bnkps.onnx')
ARCFACE_PATH = os.path.join(PROJECT_ROOT, 'weights/arcface/Glint100.onnx')
FACE_INDEX_DIR = os.path.join(PROJECT_ROOT, 'data/face_database')
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'data/sample_images')


def build_local_database(detector, extractor, face_index_dir):
    """从 data/face_database 构建人脸库（每人的图片求平均特征）
    Returns:
        dict: person_name -> embedding (512,)
    """
    database = {}
    person_dirs = [d for d in os.listdir(face_index_dir)
                   if os.path.isdir(os.path.join(face_index_dir, d))]
    for person in tqdm(person_dirs, desc="Building database"):
        pdir = os.path.join(face_index_dir, person)
        images = [f for f in os.listdir(pdir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        feats = []
        for img_file in images:
            img = cv2.imread(os.path.join(pdir, img_file))
            if img is None:
                continue
            # 160x160 裁剪人脸 -> 居中裁剪对齐到 112x112
            aligned = extractor.align_face(img, None)
            feat = extractor.l2_normalize(extractor.extract_features([aligned]))[0]
            feats.append(feat)
        if feats:
            # 平均特征（人脸的类中心）
            database[person] = extractor.l2_normalize([np.mean(feats, axis=0)])[0]
    return database


def recognize_image(detector, extractor, database, image_path, threshold=0.4):
    """检测 + 识别单张图"""
    image = cv2.imread(image_path)
    if image is None:
        return [], None

    bboxes, kpss = detector.detect(image)
    results = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        score = bbox[4]
        kps = kpss[i] if kpss is not None else None
        aligned = extractor.align_face(image, kps)
        feat = extractor.l2_normalize(extractor.extract_features([aligned]))[0]

        # 与库中所有人比对
        best_name = None
        best_sim = -1
        sims = {}
        for name, db_feat in database.items():
            sim = float(np.dot(feat, db_feat))
            sims[name] = sim
            if sim > best_sim:
                best_sim = sim
                best_name = name

        results.append({
            'bbox': bbox[:4].astype(int),
            'score': float(score),
            'best_name': best_name,
            'best_sim': best_sim,
            'all_sims': sims,
            'embedding': feat
        })
    return results, image


def analyze_similarity(detector, extractor, database):
    """分析同人/异人相似度分布"""
    print("\n=== 相似度分布分析 ===")
    same_person_sims = []
    diff_person_sims = []

    person_dirs = [d for d in os.listdir(FACE_INDEX_DIR)
                   if os.path.isdir(os.path.join(FACE_INDEX_DIR, d))]

    # 同人：每人取两张不同图片比较
    for person in person_dirs[:20]:
        pdir = os.path.join(FACE_INDEX_DIR, person)
        images = [f for f in os.listdir(pdir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 2:
            continue
        feats = []
        for img_file in images[:4]:
            img = cv2.imread(os.path.join(pdir, img_file))
            if img is None:
                continue
            aligned = extractor.align_face(img, None)
            feats.append(extractor.l2_normalize(
                extractor.extract_features([aligned]))[0])
        for j in range(len(feats)):
            for k in range(j + 1, len(feats)):
                same_person_sims.append(float(np.dot(feats[j], feats[k])))

    # 异人：不同人之间比较
    persons_sampled = person_dirs[:15]
    for i in range(len(persons_sampled)):
        for j in range(i + 1, len(persons_sampled)):
            pi = os.path.join(FACE_INDEX_DIR, persons_sampled[i])
            pj = os.path.join(FACE_INDEX_DIR, persons_sampled[j])
            fi = [f for f in os.listdir(pi) if f.lower().endswith('.jpg')][0]
            fj = [f for f in os.listdir(pj) if f.lower().endswith('.jpg')][0]
            ii = cv2.imread(os.path.join(pi, fi))
            jj = cv2.imread(os.path.join(pj, fj))
            if ii is None or jj is None:
                continue
            ei = extractor.l2_normalize(extractor.extract_features(
                [extractor.align_face(ii, None)]))[0]
            ej = extractor.l2_normalize(extractor.extract_features(
                [extractor.align_face(jj, None)]))[0]
            diff_person_sims.append(float(np.dot(ei, ej)))

    same_arr = np.array(same_person_sims)
    diff_arr = np.array(diff_person_sims)
    print(f"同人相似度: count={len(same_arr)}, mean={same_arr.mean():.4f}, "
          f"min={same_arr.min():.4f}, max={same_arr.max():.4f}")
    print(f"异人相似度: count={len(diff_arr)}, mean={diff_arr.mean():.4f}, "
          f"min={diff_arr.min():.4f}, max={diff_arr.max():.4f}")
    if len(same_arr) > 0 and len(diff_arr) > 0:
        # 建议阈值：最大化区分度
        best_thresh = 0.5
        best_acc = 0
        for t in np.arange(0.3, 0.9, 0.01):
            tp = (same_arr >= t).mean()
            tn = (diff_arr < t).mean()
            acc = (tp + tn) / 2
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
        print(f"建议识别阈值: {best_thresh:.2f} (平衡准确率 {best_acc:.4f})")
    return same_arr, diff_arr


def main():
    device = 'cuda'
    print("=" * 60)
    print("SCRFD + ArcFace 测试")
    print("=" * 60)

    detector = SCRFDDetector(SCRFD_PATH, device=device, det_thresh=0.5)
    extractor = ArcFaceFeatureExtractor(ARCFACE_PATH, device=device)

    # 1. 构建人脸库
    database = build_local_database(detector, extractor, FACE_INDEX_DIR)
    print(f"\n人脸库构建完成: {len(database)} 人")
    for name in list(database.keys())[:5]:
        print(f"  - {name}")

    # 2. 相似度分析
    analyze_similarity(detector, extractor, database)

    # 3. 对测试图片识别
    print("\n=== 测试图片识别 ===")
    test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, '*.*')))
    for img_path in test_images[:4]:
        results, image = recognize_image(detector, extractor, database, img_path)
        print(f"\n{os.path.basename(img_path)}: 检测到 {len(results)} 张人脸")
        for i, r in enumerate(results):
            if r['best_name'] is not None:
                print(f"  人脸{i+1}: {r['best_name']} (相似度 {r['best_sim']:.3f}, "
                      f"检测分 {r['score']:.3f})")
            else:
                print(f"  人脸{i+1}: 未识别 (检测分 {r['score']:.3f})")


if __name__ == '__main__':
    main()
