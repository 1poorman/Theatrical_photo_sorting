# -*- coding: utf-8 -*-
"""
真实剧照测试：SCRFD 检测 + ArcFace 识别
使用「整理完成」话剧《样式雷》图集（文件名含演员名，可交叉验证）
"""
import os
import sys
import glob
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from face_recognition.scrfd.scrfd_det import SCRFDDetector
from face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRFD_PATH = os.path.join(ROOT, 'weights/scrfd/scrfd_10g_bnkps.onnx')
ARCFACE_PATH = os.path.join(ROOT, 'weights/arcface/Glint100.onnx')
FACE_INDEX_DIR = os.path.join(ROOT, 'face_index-160')
TEST_DIR = os.path.join(ROOT, 'NCPA_test-images/话剧《样式雷》/【整理完成】话剧《样式雷》彩排')


def build_local_database(extractor):
    database = {}
    person_dirs = [d for d in os.listdir(FACE_INDEX_DIR)
                   if os.path.isdir(os.path.join(FACE_INDEX_DIR, d))]
    for person in person_dirs:
        pdir = os.path.join(FACE_INDEX_DIR, person)
        images = [f for f in os.listdir(pdir) if f.lower().endswith('.jpg')]
        feats = []
        for img_file in images:
            img = cv2.imread(os.path.join(pdir, img_file))
            if img is None:
                continue
            aligned = extractor.align_face(img, None)
            feats.append(extractor.l2_normalize(extractor.extract_features([aligned]))[0])
        if feats:
            database[person] = extractor.l2_normalize([np.mean(feats, axis=0)])[0]
    return database


def main():
    detector = SCRFDDetector(SCRFD_PATH, device='cuda', det_thresh=0.5)
    extractor = ArcFaceFeatureExtractor(ARCFACE_PATH, device='cuda')

    database = build_local_database(extractor)
    print(f"人脸库: {len(database)} 人")

    # 收集测试图（限制数量便于查看）
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, '*.JPG')))[:10]
    total_faces = 0
    known_faces = 0
    recognized = []

    for img_path in tqdm(test_images, desc="识别剧照"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        bboxes, kpss = detector.detect(image)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            kps = kpss[i] if kpss is not None else None
            aligned = extractor.align_face(image, kps)
            feat = extractor.l2_normalize(extractor.extract_features([aligned]))[0]
            best_name = None
            best_sim = -1
            for name, db_feat in database.items():
                sim = float(np.dot(feat, db_feat))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            total_faces += 1
            if best_sim >= 0.4:
                known_faces += 1
            recognized.append((os.path.basename(img_path)[:35], best_name, best_sim))

    print(f"\n共检测到 {total_faces} 张人脸，其中识别为已知(>=0.4): {known_faces} "
          f"({known_faces/max(total_faces,1)*100:.1f}%)")
    print("\n前 30 个识别结果（文件名含标注人名可核对）:")
    for fn, name, sim in recognized[:30]:
        print(f"  [{fn}...] -> {name} (sim={sim:.3f})")


if __name__ == '__main__':
    main()
