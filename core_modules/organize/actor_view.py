# -*- coding: utf-8 -*-
"""actor_view - 按演员整理视图生成（F2）

对人脸库（face_db_builder 产出的目录结构）中每位演员，在目标目录中检索其出现的照片：
逐图 SCRFD 检测 → ArcFace 特征 → 与本地锚定特征比对（余弦 ≥ 阈值）→ 命中复制。

输出：
  actor_views/{演员（饰角色）}/   命中照片副本
  actor_view_report.json         每图命中人物与相似度
"""
import os
import re
import json
import shutil
import cv2
import numpy as np

from core_modules.tools.image_io import list_images, resize_max_side
from core_modules.tools.logger import get_app_logger
from core_modules.organize.face_db_builder import FaceDBBuilder
from core_modules.organize.identity_registry import (IdentityRegistry, registry_path, registry_digest,
                                                      decide_identity, resolve_hits)
from core_modules.organize.filename_parser import parse_filename, parse_dirname
from config.base import KNOWN_FACE_THRESHOLD, FACE_IMAGE_MAX_SIDE, FACE_AMBIGUOUS_MARGIN

logger = get_app_logger()


class ActorViewGenerator:
    def __init__(self, recognition_system, db_root, identity_registry_file=None):
        self.system = recognition_system
        self.db_root = db_root
        self._fdb = FaceDBBuilder(recognition_system, db_root)
        self.identity_file = registry_path(db_root, identity_registry_file)
        self.identity_hash = registry_digest(self.identity_file)
        self.identities = IdentityRegistry.load(self.identity_file)
        self._verify_registry()

    def _verify_registry(self):
        if registry_digest(self.identity_file) != self.identity_hash:
            raise ValueError('identity_registry_content_changed')

    def _library_features(self):
        return self._fdb._library_features()

    def generate(self, input_dir, output_dir, threshold=None, copy_images=True,
                 person_filter=None, margin=FACE_AMBIGUOUS_MARGIN):
        """生成按演员整理视图。

        Args:
            input_dir: 目标照片目录（原始或整理后）
            output_dir: 输出根目录（actor_views/ 与 report 写入此处）
            threshold: 命中阈值（默认 KNOWN_FACE_THRESHOLD=0.55）
            copy_images: 是否复制命中图
            person_filter: 限定演员（可选）。支持：
                - 人名模糊匹配：'陈少云'
                - 完整库目录名：'陈少云（饰颍考叔）'
                - 库子目录路径：'data/face_database/陈少云（饰颍考叔）'（取目录名）
                - 角色表中的演员名或精确 identity_id（区分同名演员）
                - 多人用逗号/顿号分隔：'陈少云,史依弘'
                仅对匹配到的演员建专辑；未匹配时返回 error + 可用人物列表。
        Returns:
            report dict
        """
        threshold = KNOWN_FACE_THRESHOLD if threshold is None else threshold
        if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= upper
               for v, upper in ((threshold, 1), (margin, 2))):
            raise ValueError('invalid_face_decision_threshold')
        self._verify_registry()
        library = self._library_features()
        if not library:
            logger.error(f"人脸库为空: {self.db_root}")
            return {'error': 'empty library'}

        # Filtering chooses output albums, but never removes competing identities.
        selected_ids = None
        if person_filter and person_filter.strip():
            tokens = [t.strip() for t in re.split(r'[,，、]', person_filter) if t.strip()]
            # 传入库子目录路径时取目录名
            tokens = [os.path.basename(t.rstrip('/\\')) if ('/' in t or '\\' in t) else t
                      for t in tokens]
            matched = {}
            for person, features in library.items():
                identity = self.identities.identity(person)
                if any(token in person or token in identity['actor_name'] or
                       token == identity['identity_id'] for token in tokens):
                    matched[person] = features
            if not matched:
                return {'error': 'person not found', 'available': sorted(library)}
            selected_ids = {self.identities.identity(p)['identity_id'] for p in matched}

        names = list_images(input_dir)
        directory_play = parse_dirname(input_dir)['play']
        views_dir = os.path.join(output_dir, 'actor_views')
        if copy_images:
            os.makedirs(views_dir, exist_ok=True)

        # 预取库特征矩阵加速比对
        lib_matrix, lib_person = [], []
        for pdir, feats in library.items():
            for f, _ in feats:
                lib_matrix.append(f)
                lib_person.append(pdir)
        lib_matrix = np.array(lib_matrix)

        per_person = {}
        for person in library:
            identity_id = self.identities.identity(person)['identity_id']
            if selected_ids is None or identity_id in selected_ids:
                display, _ = resolve_hits({person: 0.}, self.identities, directory_play)
                for label in display:
                    per_person.setdefault(label, [])
        image_records = []

        for name in names:
            self._verify_registry()
            img_path = os.path.join(input_dir, name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取: {img_path}")
                continue
            if max(img.shape[:2]) > FACE_IMAGE_MAX_SIDE:
                img = resize_max_side(img, FACE_IMAGE_MAX_SIDE)

            faces = self.system.detector._detect_single(img)
            hits = {}
            decisions = []
            for face_index, f in enumerate(faces):
                feat, _ = self._fdb._face_feature(img, f)
                if feat is None:
                    decisions.append({'face_index': face_index, 'status': 'unknown', 'reason': 'alignment_failed'})
                    continue
                sims = lib_matrix @ feat
                decision = decide_identity(sims, lib_person, self.identities, threshold, margin)
                decisions.append(dict(decision, face_index=face_index))
                if decision['status'] == 'recognized' and (selected_ids is None or decision['identity_id'] in selected_ids):
                    p = decision['anchor_dir']
                    # 每图每人只记最高相似度
                    if p not in hits or decision['score'] > hits[p]:
                        hits[p] = decision['score']

            play = parse_filename(name)['play'] or directory_play
            hits, identities = resolve_hits(hits, self.identities, play)
            self._verify_registry()

            record = {'image': name, 'faces': len(faces),
                      'hits': {p: round(s, 4) for p, s in sorted(hits.items())},
                      'identities': identities, 'face_decisions': decisions}
            image_records.append(record)
            for p, s in hits.items():
                per_person.setdefault(p, []).append({'image': name, 'similarity': round(s, 4)})
                if copy_images:
                    safe_person = re.sub(r'[\\/:*?"<>|]', '', p).strip()
                    if safe_person in ('', '.', '..'):
                        raise ValueError('invalid_actor_display_name')
                    dst = os.path.join(views_dir, safe_person, name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        shutil.copy2(img_path, dst)

        report = {
            'input_dir': input_dir, 'db_root': self.db_root, 'threshold': threshold,
            'person_filter': person_filter,
            'library_persons': len(library), 'margin': margin,
            'identity_registry_sha256': self.identity_hash,
            'total_images': len(names),
            'persons': {p: {'count': len(v), 'images': v}
                        for p, v in sorted(per_person.items())},
            'images': image_records,
        }
        self._verify_registry()
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'actor_view_report.json'), 'w',
                  encoding='utf-8') as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)

        logger.info(f"按演员视图完成: {len(names)} 张, {len(library)} 位演员, "
                    f"命中分布 " + ', '.join(f'{p}:{len(v)}' for p, v in per_person.items()))
        return report
