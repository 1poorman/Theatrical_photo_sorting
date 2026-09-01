# -*- coding: utf-8 -*-
"""face_db_builder - 从整理完成目录半自动构建人脸库（F2 基础）

两轮构建策略：
- 第 1 轮（单人归属）：文件名仅含 1 位「演员饰角色」的照片，最大人脸即该演员。
  含多尺度重检（解决 SCRFD 对超大/过小脸漏检）。
- 第 2 轮（多人锚定传播）：文件名含 2~4 位演员的同框照片，
  用第 1 轮库对每张脸匹配（余弦 ≥ ANCHOR_MATCH_SIM）消去已识别演员，
  剩余未匹配脸中最大者归属唯一未匹配演员（匹配消去法）。
  迭代多轮让新锚定继续传播。

同时跳过：谢幕/合影、归属歧义（次大腿面积 ≈ 最大腿）、无脸照片。
"""
import os
import cv2
import numpy as np

from core_modules.organize.filename_parser import parse_filename
from core_modules.tools.image_io import list_images, resize_max_side
from core_modules.tools.logger import get_app_logger
from config.base import FACE_DB_MIN_AREA, FACE_DB_STANDARD_SIZE

logger = get_app_logger()

SKIP_KEYWORDS = ('谢幕', '合影', '返场')      # 大合影场景人脸归属不可靠
AMBIGUOUS_AREA_RATIO = 0.85                   # 次大腿面积达到最大腿该比例视为归属不确定
RETRY_SCALES = (0.5, 0.25)                    # 多尺度重检缩放
ANCHOR_MATCH_SIM = 0.5                        # 传播轮：锚定匹配的最低余弦相似度
ANCHOR_CONSISTENCY_SIM = 0.35                 # 新锚定与同人现有锚定的最低一致性
PROPAGATE_ROUNDS = 2                          # 传播轮数
MAX_PEOPLE_IN_GROUP = 4                       # 传播轮处理的最大同框人数


class FaceDBBuilder:
    def __init__(self, recognition_system, db_root, max_anchors_per_person=8):
        """
        Args:
            recognition_system: FaceRecognitionSystem 实例
            db_root: 人脸库根目录（data/face_database）
            max_anchors_per_person: 每演员最多锚定张数
        """
        self.system = recognition_system
        self.db_root = db_root
        self.max_anchors = max_anchors_per_person

    # ---------- 基础工具 ----------

    def _person_dir(self, actor, role):
        return f'{actor}（饰{role}）'

    def _detect_faces(self, img):
        """多尺度人脸检测（坐标基于传入图），按面积降序。"""
        faces = self.system.detector._detect_single(img)
        if not faces:
            h, w = img.shape[:2]
            for scale in RETRY_SCALES:
                if min(h, w) * scale < 320:
                    break
                small = cv2.resize(img, (int(w * scale), int(h * scale)))
                faces = self.system.detector._detect_single(small)
                if faces:
                    for f in faces:
                        f.x, f.y, f.r, f.b = (f.x / scale, f.y / scale,
                                              f.r / scale, f.b / scale)
                        if f.landmark is not None:
                            f.landmark = f.landmark / scale
                    break
        faces = [f for f in faces if f.width * f.height > FACE_DB_MIN_AREA]
        faces.sort(key=lambda f: f.width * f.height, reverse=True)
        return faces

    def _face_feature(self, img, face):
        """对单张脸提取 L2 归一化特征；face=None 时按标准图直接对齐。失败返回 (None, None)。"""
        aligned = self.system.extractor.align_face(
            img, face.landmark if face is not None else None)
        if aligned is None:
            return None, None
        if aligned.shape[0] != FACE_DB_STANDARD_SIZE:
            aligned = cv2.resize(aligned, (FACE_DB_STANDARD_SIZE, FACE_DB_STANDARD_SIZE))
        feats = self.system.extractor.extract_features([aligned])
        if feats.shape[0] != 1:
            return None, None
        feat = self.system.extractor.l2_normalize([feats[0]])[0]
        return feat, aligned

    def _library_features(self):
        """读取库内全部锚定图特征。返回 {person_dir: [(feat, img_path), ...]}"""
        lib = {}
        if not os.path.isdir(self.db_root):
            return lib
        for d in os.listdir(self.db_root):
            pdir = os.path.join(self.db_root, d)
            if not os.path.isdir(pdir):
                continue
            for f in sorted(list_images(pdir)):
                img = cv2.imread(os.path.join(pdir, f))
                if img is None:
                    continue
                feat, _ = self._face_feature(img, None)   # 标准图直接对齐
                if feat is not None:
                    lib.setdefault(d, []).append((feat, os.path.join(pdir, f)))
        return lib

    def _match_face(self, feat, library, allowed_persons=None):
        """脸特征与库比对，返回 (best_person_dir, best_sim)。"""
        best_p, best_s = None, -1.0
        for pdir, feats in library.items():
            if allowed_persons is not None and pdir not in allowed_persons:
                continue
            for f, _ in feats:
                s = float(np.dot(feat, f))
                if s > best_s:
                    best_s, best_p = s, pdir
        return best_p, best_s

    # ---------- 第 1 轮：单人归属 ----------

    def _round_single(self, entries, counts, stats, dry_run):
        for img_path, name, done_style in entries:
            info = parse_filename(name, done_style=done_style)
            if len(info['people']) != 1:
                stats['skipped_multi'] += 1
                continue
            if any(k in name for k in SKIP_KEYWORDS):
                stats['skipped_group'] += 1
                continue

            actor, role = info['people'][0]
            pd = self._person_dir(actor, role)
            if counts.get(pd, 0) >= self.max_anchors:
                stats['skipped_max'] += 1
                continue
            stats['candidates'] += 1

            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取: {img_path}")
                stats['skipped_no_face'] += 1
                continue
            if max(img.shape[:2]) > 1920:
                img = resize_max_side(img, 1920)

            faces = self._detect_faces(img)
            if not faces:
                stats['skipped_no_face'] += 1
                continue
            if len(faces) >= 2 and faces[1].width * faces[1].height >= \
                    AMBIGUOUS_AREA_RATIO * faces[0].width * faces[0].height:
                stats['skipped_ambiguous'] += 1
                continue

            self._save_anchor(img, faces[0], pd, name, counts, stats, dry_run)

    # ---------- 第 2 轮：多人锚定传播（匹配消去法） ----------

    def _round_propagate(self, entries, counts, stats, dry_run):
        for rnd in range(PROPAGATE_ROUNDS):
            library = self._library_features()
            if not library:
                return
            propagated = 0
            for img_path, name, done_style in entries:
                info = parse_filename(name, done_style=done_style)
                people = info['people']
                if not (2 <= len(people) <= MAX_PEOPLE_IN_GROUP):
                    continue
                if any(k in name for k in SKIP_KEYWORDS):
                    continue

                person_dirs = []
                for a, r in people:
                    pd = self._person_dir(a, r)
                    if counts.get(pd, 0) < self.max_anchors:
                        person_dirs.append(pd)
                if len(person_dirs) < 1:
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue
                if max(img.shape[:2]) > 1920:
                    img = resize_max_side(img, 1920)
                faces = self._detect_faces(img)
                # 仅处理检出脸数与标注人数一致的照片（多于人数时归属歧义大）
                if len(faces) != len(people):
                    continue

                # 逐脸比对库
                matched_persons = set()
                matched_face_idx = set()
                face_feats = []
                for fi, f in enumerate(faces):
                    feat, _ = self._face_feature(img, f)
                    face_feats.append(feat)
                    if feat is None:
                        continue
                    p, s = self._match_face(feat, library, allowed_persons=person_dirs)
                    if p is not None and s >= ANCHOR_MATCH_SIM and p not in matched_persons:
                        matched_persons.add(p)
                        matched_face_idx.add(fi)

                unmatched_persons = [p for p in person_dirs if p not in matched_persons]
                unmatched_faces = [i for i in range(len(faces)) if i not in matched_face_idx]
                # 唯一未匹配演员 + 唯一未匹配脸 → 直接归属
                if len(unmatched_persons) == 1 and len(unmatched_faces) == 1:
                    fi = unmatched_faces[0]
                    if face_feats[fi] is not None:
                        if self._save_anchor(img, faces[fi], unmatched_persons[0],
                                             name, counts, stats, dry_run, library=library):
                            propagated += 1
            logger.info(f"锚定传播第 {rnd + 1} 轮: 新增 {propagated} 张")
            if propagated == 0:
                break

    # ---------- 保存与入口 ----------

    def _save_anchor(self, img, face, person_dir, src_name, counts, stats, dry_run,
                     library=None):
        feat, aligned = self._face_feature(img, face)
        if aligned is None:
            stats['skipped_no_face'] += 1
            return False
        # 一致性校验：已有锚定时，新脸须与同人现有锚定足够相似（挡异人混入）
        if library and person_dir in library and library[person_dir]:
            max_sim = max(float(np.dot(feat, f)) for f, _ in library[person_dir])
            if max_sim < ANCHOR_CONSISTENCY_SIM:
                stats['skipped_inconsistent'] = stats.get('skipped_inconsistent', 0) + 1
                logger.info(f"锚定一致性不足({max_sim:.3f}<{ANCHOR_CONSISTENCY_SIM})，"
                            f"跳过: {person_dir} <- {src_name[:40]}")
                return False
        if not dry_run:
            pdir = os.path.join(self.db_root, person_dir)
            os.makedirs(pdir, exist_ok=True)
            stem = os.path.splitext(src_name)[0]
            cv2.imwrite(os.path.join(pdir, f'{stem}_face.jpg'), aligned)
            counts[person_dir] = counts.get(person_dir, 0) + 1
        stats['added'] += 1
        stats['added_persons'].add(person_dir)
        # 传播轮内即时更新库缓存，供同轮后续照片匹配
        if library is not None:
            library.setdefault(person_dir, []).append((feat, src_name))
        return True

    def build_from_organized(self, organized_dirs, dry_run=False):
        """从整理完成目录（可含原始目录）构建人脸库。

        Args:
            organized_dirs: 目录或目录列表（整理完成命名 / 原始命名均可）
        Returns:
            dict: 统计信息
        """
        if isinstance(organized_dirs, str):
            organized_dirs = [organized_dirs]
        os.makedirs(self.db_root, exist_ok=True)
        counts = {}
        for d in os.listdir(self.db_root):
            if os.path.isdir(os.path.join(self.db_root, d)):
                counts[d] = len([f for f in os.listdir(os.path.join(self.db_root, d))
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        entries = []
        for root_dir in organized_dirs:
            done = '整理完成' in root_dir
            for name in list_images(root_dir):
                entries.append((os.path.join(root_dir, name), name, done))

        stats = {'scanned': len(entries), 'candidates': 0, 'added': 0,
                 'skipped_multi': 0, 'skipped_group': 0, 'skipped_ambiguous': 0,
                 'skipped_no_face': 0, 'skipped_max': 0, 'added_persons': set()}

        self._round_single(entries, counts, stats, dry_run)
        self._round_propagate(entries, counts, stats, dry_run)

        stats['added_persons'] = sorted(stats['added_persons'])
        logger.info(f"人脸库构建: 扫描 {stats['scanned']}，新增 {stats['added']} 张 / "
                    f"{len(stats['added_persons'])} 人，跳过(多人){stats['skipped_multi']} "
                    f"(合影){stats['skipped_group']} (歧义){stats['skipped_ambiguous']} "
                    f"(无脸){stats['skipped_no_face']} (满额){stats['skipped_max']}")
        return stats


# ---------- 锚定特征缓存（跨请求复用，按库目录文件签名自动失效） ----------

_ANCHOR_CACHE = {}   # db_root 绝对路径 -> {'signature', 'matrix', 'persons'}


def _db_signature(db_root):
    """库目录签名：全部锚定文件的（相对路径, mtime_ns, size）。增删/修改任一文件即失效。"""
    sig = []
    for root, dirs, files in os.walk(db_root):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fp = os.path.join(root, f)
                st = os.stat(fp)
                sig.append((os.path.relpath(fp, db_root), st.st_mtime_ns, st.st_size))
    sig.sort()
    return hash(tuple(sig))


def get_cached_library_matrix(system, db_root):
    """带缓存的库特征矩阵（供智能整理流水线等高频调用方使用）。

    缓存按库目录签名失效：assign 命名入库/人工增删锚定图后自动重建。
    线程安全：GIL 下 dict 赋值原子；并发重建最坏情况是重复提取一次。

    Args:
        system: 人脸识别模块实例
        db_root: 库根目录

    Returns:
        (matrix (N,512), persons list[str], from_cache bool)
    """
    db_root = os.path.abspath(db_root)
    sig = _db_signature(db_root)
    cached = _ANCHOR_CACHE.get(db_root)
    if cached and cached['signature'] == sig:
        return cached['matrix'], cached['persons'], True

    builder = FaceDBBuilder(system, db_root)
    library = builder._library_features()
    feats, persons = [], []
    for pdir, fl in library.items():
        for f, _ in fl:
            feats.append(f)
            persons.append(pdir)
    matrix = np.array(feats) if feats else np.zeros((0, 512))
    _ANCHOR_CACHE[db_root] = {'signature': sig, 'matrix': matrix, 'persons': persons}
    logger.info(f"锚定库特征矩阵重建: {len(persons)} 张 / {len(library)} 人")
    return matrix, persons, False
