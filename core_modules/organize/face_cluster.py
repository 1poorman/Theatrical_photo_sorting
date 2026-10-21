# -*- coding: utf-8 -*-
"""face_cluster - 无标注目录的人脸聚类建库（F4，路径3 + 路径2兜底）

流程：
  目录内全部图片 → SCRFD 检测人脸（过滤过小脸）→ ArcFace 对齐特征
  → Agglomerative 层次聚类（余弦距离阈值，average linkage）
  → 每簇：内聚度（簇内平均两两余弦）+ 对齐脸拼图预览
  → cluster_state.json 持久化（簇成员含源图路径/框/landmark）
  → 人工命名簇（assign）后从源图重新对齐裁剪写入人脸库

分诊（路径2兜底）：
  - 内聚度 >= COHESION_OK 且人脸数 >= MIN_FACES_OK：高置信，命名即入库
  - 其余：标记 suggest_manual=True，建议人工收集种子图（/api/face/build_database）
"""
import os
import json
import hashlib
import tempfile
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from core_modules.tools.image_io import list_images, resize_max_side
from core_modules.tools.logger import get_app_logger
from core_modules.face_recognition.face_recognition import FaceRecognitionSystem
from config.base import FACE_DB_STANDARD_SIZE
from core_modules.organize.task_artifacts import atomic_json, copy_verified, fingerprint

logger = get_app_logger()

# 聚类与分诊参数（可被 API 参数覆盖）
DEFAULT_DIST_THRESHOLD = 0.5          # 余弦距离阈值：1 - cos >= 0.5 即 cos <= 0.5 分裂
DEFAULT_MIN_FACE_AREA = 40 * 40       # 过小脸不参与聚类（远景龙套）
MIN_FACES_OK = 3                      # 高置信簇的最少人脸数
COHESION_OK = 0.5                     # 高置信簇的最低内聚度
CLUSTER_COLLAGE_MAX = 12              # 拼图最多展示的对齐脸数
STATE_FILE = 'cluster_state.json'
PREVIEW_DIR = 'previews'


def _make_face_collage(aligned_faces, out_path, cell=160, cols=6):
    """对齐脸拼图预览（不足补灰底）。
    Args:
        aligned_faces: 对齐脸列表
        out_path: 输出路径
        cell: 单元格尺寸
        cols: 列数
    Returns:
        out_path: 输出路径

    """
    n = len(aligned_faces)
    if n == 0:
        return None
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 245, dtype=np.uint8)
    for i, face in enumerate(aligned_faces[:CLUSTER_COLLAGE_MAX]):
        r, c = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = face
    cv2.imwrite(out_path, canvas)
    return out_path


class FaceClusterBuilder:
    """无标注目录的人脸聚类建库器。

    用法：
        b = FaceClusterBuilder(system)
        state = b.scan(input_dir, output_dir, progress_cb=fn)   # 扫描+聚类，写 cluster_state.json
        b.assign(state_path, cluster_id, '演员（饰角色）', db_root)  # 人工命名后入库
    """

    def __init__(self, recognition_system: FaceRecognitionSystem):
        self.system = recognition_system

    # ---------- 扫描 + 聚类 ----------

    def scan_image(self, path, min_face_area=DEFAULT_MIN_FACE_AREA, check=None):
        """Extract bounded batches for one image; retain no decoded face arrays."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError('image_read_failed')
        if max(img.shape[:2]) > 1920:
            img = resize_max_side(img, 1920)
        faces = self.system.detector._detect_single(img)
        members, aligned_batch, pending = [], [], []
        limit = min(32, max(1, int(getattr(self.system.extractor, 'max_batch_size', 32))))
        def flush():
            if not pending:
                return
            if check:
                check()
            feats = self.system.extractor.extract_features(aligned_batch)
            if len(feats) != len(pending):
                raise ValueError('face_feature_count_mismatch')
            feats = self.system.extractor.l2_normalize(feats)
            for member, feat in zip(pending, feats):
                feat = np.asarray(feat, dtype=float).reshape(-1)
                if not feat.size or not np.isfinite(feat).all() or np.linalg.norm(feat) <= 0:
                    raise ValueError('invalid_face_feature')
                members.append(dict(member, feat=feat.tolist()))
            aligned_batch.clear()
            pending.clear()
        for index, face in enumerate(faces):
            if check:
                check()
            if face.width * face.height < min_face_area or face.landmark is None:
                continue
            aligned = self.system.extractor.align_face(img, face.landmark)
            if aligned is None:
                raise ValueError('face_alignment_failed')
            if aligned.shape[:2] != (FACE_DB_STANDARD_SIZE, FACE_DB_STANDARD_SIZE):
                aligned = cv2.resize(aligned, (FACE_DB_STANDARD_SIZE, FACE_DB_STANDARD_SIZE))
            pending.append({'image': os.path.basename(path), 'img_path': os.path.abspath(path),
                            'face_index': index, 'bbox': [int(face.x), int(face.y), int(face.r), int(face.b)],
                            'landmark': np.asarray(face.landmark, dtype=float).tolist()})
            aligned_batch.append(aligned)
            if len(pending) == limit:
                flush()
        flush()
        return members

    @staticmethod
    def cluster_members(members, dist_threshold=DEFAULT_DIST_THRESHOLD):
        """Shared average-linkage grouping without preview decoding or writes."""
        if len(members) < 2:
            return []
        feats = np.asarray([member['feat'] for member in members], dtype=np.float32)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_threshold,
                                             metric='cosine', linkage='average')
        labels = clustering.fit_predict(feats)
        clusters = []
        for label in sorted(set(labels)):
            indices = np.flatnonzero(labels == label)
            count = len(indices)
            cfeats = feats[indices].astype(np.float64)
            # Algebraically equivalent to sum of the similarity matrix, O(n*d).
            total = cfeats.sum(axis=0)
            cohesion = float((total @ total - np.square(cfeats).sum()) / (count * (count - 1))) if count > 1 else 1.0
            cmembers = [members[index] for index in indices]
            clusters.append({'id': f'cluster-{label + 1:02d}', 'size': count,
                             'cohesion': round(cohesion, 3),
                             'suggest_manual': count < MIN_FACES_OK or cohesion < COHESION_OK,
                             'status': 'pending', 'assigned_to': None, 'preview': None,
                             'sample_images': list(dict.fromkeys(member['image'] for member in cmembers))[:10],
                             'members': [{key: value for key, value in member.items() if key != 'feat'} for member in cmembers]})
        return sorted(clusters, key=lambda cluster: -cluster['size'])

    def aligned_member(self, member):
        img = cv2.imread(member['img_path'])
        if img is None:
            raise ValueError('image_read_failed')
        if max(img.shape[:2]) > 1920:
            img = resize_max_side(img, 1920)
        aligned = self.system.extractor.align_face(img, np.asarray(member['landmark'], dtype=float))
        if aligned is None:
            raise ValueError('face_alignment_failed')
        return cv2.resize(aligned, (FACE_DB_STANDARD_SIZE, FACE_DB_STANDARD_SIZE))

    def scan(self, input_dir, output_dir, dist_threshold=DEFAULT_DIST_THRESHOLD,
             min_face_area=DEFAULT_MIN_FACE_AREA, progress_cb=None):
        """扫描目录并聚类。

        Args:
            input_dir: 照片目录（可无任何人物标注）
            output_dir: 结果目录（cluster_state.json + previews/）
            dist_threshold: 余弦距离阈值（越大簇越少）
            min_face_area: 参与聚类的最小人脸面积（像素²）
            progress_cb: 进度回调 fn(done, total, message)
        Returns:
            state dict（同 cluster_state.json 内容）
        """
        os.makedirs(output_dir, exist_ok=True)
        preview_dir = os.path.join(output_dir, PREVIEW_DIR)
        os.makedirs(preview_dir, exist_ok=True)

        names = list_images(input_dir)
        total = len(names)
        if progress_cb:
            progress_cb(0, total, f'开始扫描 {total} 张图片')

        members, errors = [], []
        for i, name in enumerate(names):
            try:
                members.extend(self.scan_image(os.path.join(input_dir, name), min_face_area))
            except Exception as exc:
                errors.append({'image': name, 'error': str(exc)})
                logger.warning(f'人脸提取失败 {name}: {exc}')
            if progress_cb:
                progress_cb(i + 1, total, f'已处理 {i + 1}/{total}')
        clusters = self.cluster_members(members, dist_threshold)
        for cluster in clusters:
            faces = [self.aligned_member(member) for member in cluster['members'][:CLUSTER_COLLAGE_MAX]]
            cluster['preview'] = _make_face_collage(faces, os.path.join(preview_dir, cluster['id'] + '.jpg'))
        state = {'input_dir': input_dir, 'output_dir': output_dir,
                 'dist_threshold': dist_threshold, 'min_face_area': min_face_area,
                 'n_images': total, 'n_faces': len(members),
                 'clusters': clusters, 'failed': len(errors), 'errors': errors}
        self._save_state(output_dir, state)
        if progress_cb:
            n_manual = sum(1 for c in clusters if c['suggest_manual'])
            progress_cb(total, total,
                        f'完成：{len(members)} 张人脸 → {len(clusters)} 个簇'
                        f'（{n_manual} 个建议人工补图）')
        logger.info(f"人脸聚类: {total} 图 / {len(members)} 脸 → {len(clusters)} 簇")
        return state

    # ---------- 命名入库 ----------

    def assign(self, state_path, cluster_id, person_name, db_root):
        """将指定簇的对齐脸写入人脸库目录（路径3 的确认动作）。

        Args:
            state_path: cluster_state.json 路径
            cluster_id: 簇 ID
            person_name: 人名（建议格式「演员（饰角色）」，也接受纯人名）
            db_root: 人脸库根目录
        Returns:
            dict: {person_dir, saved, skipped}
        """
        state = self.load_state(state_path)
        cluster = next((c for c in state['clusters'] if c['id'] == cluster_id), None)
        if cluster is None:
            raise ValueError(f'cluster not found: {cluster_id}')

        if not person_name or person_name in ('.', '..') or any(char in person_name for char in ('/', '\\', '\x00')):
            raise ValueError('person_name must be a single directory name')
        from core_modules.organize.task_versions import model_identity
        from core_modules.organize.task_queue import encode
        current_model = json.loads(encode(model_identity(self.system)))
        if state.get('execution_contract') and state['execution_contract']['face'] != current_model:
            raise ValueError('face_model_changed_since_scan')
        hashes = state.get('input_hashes', {})
        for member in cluster['members']:
            if member['img_path'] in hashes and fingerprint(member['img_path']) != hashes[member['img_path']]:
                raise ValueError('source_content_changed_since_scan')

        person_dir = os.path.join(db_root, person_name)
        from pathlib import Path
        if Path(person_dir).is_symlink() or any(parent.is_symlink() for parent in Path(person_dir).parents):
            raise ValueError('output_directory_symlink')
        os.makedirs(person_dir, exist_ok=True)

        saved = skipped = 0
        for m in cluster['members']:
            stem = os.path.splitext(os.path.basename(m['image']))[0]
            suffix = f"_{m['face_index']:02d}_face.jpg" if 'face_index' in m else '_face.jpg'
            out_path = os.path.join(person_dir, stem + suffix)
            img = cv2.imread(m['img_path'])
            if img is None:
                skipped += 1
                continue
            # 源图可能被缩放过：landmark 是 1920 限边后坐标，重新按同规则缩放
            if max(img.shape[:2]) > 1920:
                img = resize_max_side(img, 1920)
            landmark = np.asarray(m['landmark'], dtype=float)
            aligned = self.system.extractor.align_face(img, landmark)
            if aligned is None:
                skipped += 1
                continue
            if aligned.shape[0] != FACE_DB_STANDARD_SIZE:
                aligned = cv2.resize(aligned, (FACE_DB_STANDARD_SIZE, FACE_DB_STANDARD_SIZE))
            if m['img_path'] in hashes and fingerprint(m['img_path']) != hashes[m['img_path']]:
                raise ValueError('source_content_changed_during_assignment')
            ok, encoded = cv2.imencode('.jpg', aligned)
            if not ok:
                raise ValueError('face_encoding_failed')
            with tempfile.NamedTemporaryFile() as temporary:
                data = encoded.tobytes()
                temporary.write(data)
                temporary.flush()
                new = copy_verified(temporary.name, out_path, hashlib.sha256(data).hexdigest())
            saved += int(new)
            skipped += int(not new)

        cluster['status'] = 'assigned'
        cluster['assigned_to'] = person_name
        self._save_state(os.path.dirname(state_path), state)
        logger.info(f"簇入库: {cluster_id} -> {person_dir}（{saved} 张，跳过 {skipped}）")
        return {'person_dir': person_dir, 'saved': saved, 'skipped': skipped}

    def ignore(self, state_path, cluster_id, ignored=True):
        """标记/取消忽略簇。"""
        state = self.load_state(state_path)
        cluster = next((c for c in state['clusters'] if c['id'] == cluster_id), None)
        if cluster is None:
            raise ValueError(f'cluster not found: {cluster_id}')
        cluster['status'] = 'ignored' if ignored else 'pending'
        self._save_state(os.path.dirname(state_path), state)
        return {'cluster_id': cluster_id, 'status': cluster['status']}

    # ---------- 状态读写 ----------

    @staticmethod
    def _save_state(output_dir, state):
        # clusters 成员仅含可序列化字段（feat/aligned 均不在簇成员内），直接落盘
        slim = dict(state)
        slim['clusters'] = [
            {k: v for k, v in c.items() if k not in ('feat', 'aligned')}
            for c in state.get('clusters', [])
        ]
        atomic_json(os.path.join(output_dir, STATE_FILE), slim)

    @staticmethod
    def load_state(state_path):
        with open(state_path, encoding='utf-8') as f:
            return json.load(f)
