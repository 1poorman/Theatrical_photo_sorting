# -*- coding: utf-8 -*-
"""smart_organizer - 智能整理流水线（F1 编排入口）

流程：
  原始目录
  → filename_parser 元数据解析
  → burst_dedup 连拍去重选优（景别感知）
  → 逐张保留图：人脸识别（本地锚定库）生成人物列表 + 行当分类（实验性）
  → scene_split 场景划分（EXIF 时间分段）
  → 生成规范命名并复制输出
  → organize_report.json

输出结构：
  output_dir/
  ├── organized/          规范命名的整理结果
  ├── best/ discarded/    连拍选优结果（同 burst_dedup）
  └── organize_report.json
"""
import os
import re
import json
import shutil
import time
import cv2
import numpy as np

from core_modules.organize.filename_parser import parse_filename, parse_dirname
from core_modules.organize.burst_dedup import BurstDeduplicator, image_quality
from core_modules.organize.scene_split import split_scenes, apply_labels
from core_modules.tools.image_io import list_images, resize_max_side
from core_modules.tools.logger import get_app_logger
from config.base import (KNOWN_FACE_THRESHOLD, FACE_IMAGE_MAX_SIDE, FACE_AMBIGUOUS_MARGIN,
                         FEATURE_BATCH_SIZE)

logger = get_app_logger()

MAX_PEOPLE_IN_NAME = 3      # 文件名人物列表上限（按相似度取前 N）


def apply_scene_evidence(scene_assignment, evidence):
    """用已审核证据覆盖场景名；未审核/unknown 保持占位 ``scene-XX``。

    Args:
        scene_assignment: {image_path: scene_label}
        evidence: ``review_queue.load_scene_evidence`` 的返回值
    Returns:
        (new_assignment, stats)
    """
    by_image = evidence.get('by_image') or {}
    by_path = evidence.get('by_path') or {}
    legacy = evidence.get('legacy_by_image', by_image)
    by_scene = evidence.get('by_scene') or {}
    names = {}
    for path in scene_assignment:
        names.setdefault(os.path.basename(path), set()).add(os.path.abspath(path))
    if any(len(paths) > 1 and name in legacy for name, paths in names.items()):
        raise ValueError('裸文件名场景标签存在跨目录同名歧义')
    out, applied = {}, 0
    for path, scene in scene_assignment.items():
        label = by_path.get(os.path.abspath(path))
        if label is None:
            label = legacy.get(os.path.basename(path))
        if label is None:
            label = by_scene.get(scene)
        if label:
            out[path] = label
            applied += 1
        else:
            out[path] = scene
    return out, {'applied': applied, 'kept_placeholder': len(out) - applied}


class SmartOrganizer:
    def __init__(self, recognition_system, face_db_root,
                 shot_classifier=None, embedder=None, role_classifier=None,
                 feature_cache_dir=None, feature_batch_size=FEATURE_BATCH_SIZE):
        """
        Args:
            recognition_system: FaceRecognitionSystem（SCRFD+ArcFace）
            face_db_root: 人脸锚定库根目录（face_db_builder 产出）
            shot_classifier: ShotTypeClassifier（景别感知选优）
            embedder: SigLIP2 ImageEmbedder（场景细分，可选）
            role_classifier: RoleClassifier（行当分类，可选，实验性）
        """
        self.system = recognition_system
        self.face_db_root = face_db_root
        self.shot_classifier = shot_classifier
        self.embedder = embedder
        self.role_classifier = role_classifier
        self.feature_cache_dir = feature_cache_dir
        self.feature_batch_size = feature_batch_size
        self._lib_matrix = None
        self._lib_persons = None

    # ---------- 人脸识别（本地锚定库） ----------

    def _load_library(self):
        if self._lib_matrix is not None:
            return
        from core_modules.organize.face_db_builder import get_cached_library_matrix
        matrix, persons, from_cache = get_cached_library_matrix(
            self.system, self.face_db_root)
        self._lib_matrix = matrix
        self._lib_persons = persons
        logger.info(f"锚定库加载: {len(persons)} 张 / {len(set(persons))} 人"
                    f"{'（缓存命中）' if from_cache else '（全量提取）'}")

    def _recognize_image(self, img, threshold):
        """返回 hits: {person_dir: max_sim}。"""
        self._load_library()
        faces = self.system.detector._detect_single(img)
        if self._lib_matrix.shape[0] == 0 or not faces:
            return {}, faces
        # 对齐全部脸后批量提特征
        aligned_list = []
        for f in faces:
            aligned = self.system.extractor.align_face(img, f.landmark)
            if aligned is None:
                continue
            if aligned.shape[0] != 160:
                aligned = cv2.resize(aligned, (160, 160))
            aligned_list.append(aligned)
        if not aligned_list:
            return {}, faces
        # TRT engines accept at most 32 faces; retain the existing alignment and
        # normalization so batching alone does not change recognition semantics.
        limit = min(32, max(1, int(getattr(self.system.extractor, 'max_batch_size', 32))))
        feats = np.concatenate([self.system.extractor.extract_features(aligned_list[i:i + limit])
                                for i in range(0, len(aligned_list), limit)])
        feats = self.system.extractor.l2_normalize(feats)
        hits = {}
        for fv in feats:
            sims = self._lib_matrix @ fv
            person_max = {}
            for i, s in enumerate(sims):
                p = self._lib_persons[i]
                # Several role directories may belong to the same actor.
                identity = re.split(r'[（(]|饰', os.path.basename(p), maxsplit=1)[0].strip()
                if np.isfinite(s) and s > person_max.get(identity, (-1, None))[0]:
                    person_max[identity] = (float(s), p)
            ranked = sorted(person_max.values(), reverse=True)
            if not ranked or ranked[0][0] < threshold:
                continue
            if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < FACE_AMBIGUOUS_MARGIN:
                continue
            score, person = ranked[0]
            hits[person] = max(score, hits.get(person, -1))
        # Merge identities appearing in several detected faces/role directories.
        by_identity = {}
        for person, score in hits.items():
            identity = re.split(r'[（(]|饰', os.path.basename(person), maxsplit=1)[0].strip()
            if score > by_identity.get(identity, (-1, None))[0]:
                by_identity[identity] = (score, person)
        hits = {person: score for score, person in by_identity.values()}
        return hits, faces

    def describe_image(self, path, threshold=None, classify_role=True, strict=False):
        """Compute one photo's metadata and identity using the shared pipeline."""
        name = os.path.basename(path)
        input_dir = os.path.dirname(path)
        threshold = threshold or KNOWN_FACE_THRESHOLD
        info = parse_filename(name, done_style='整理完成' in input_dir)
        img = cv2.imread(path)
        record = {'source': name, 'date': info['date'], 'venue': info['venue'],
                  'play': info['play'], 'event': info['event'],
                  'photographer': info['photographer'], 'burst_seq': info['burst_seq']}
        if img is None:
            record['error'] = 'read_failed'
            return record
        if max(img.shape[:2]) > FACE_IMAGE_MAX_SIDE:
            img = resize_max_side(img, FACE_IMAGE_MAX_SIDE)

        # 人物识别
        try:
            hits, faces = self._recognize_image(img, threshold)
        except Exception as e:
            if strict:
                raise
            logger.warning(f"识别失败 {name}: {e}")
            hits, faces = {}, []
        persons_sorted = sorted(hits.items(), key=lambda kv: -kv[1])
        record['recognized'] = {p: round(s, 4) for p, s in persons_sorted}
        record['people_detected'] = len(faces)

        # 行当分类（最大脸外扩区域，实验性）
        if classify_role and self.role_classifier is not None and faces:
            try:
                f0 = max(faces, key=lambda f: f.width * f.height)
                x1, y1, x2, y2 = int(f0.x), int(f0.y), int(f0.r), int(f0.b)
                fw, fh = x2 - x1, y2 - y1
                h, w = img.shape[:2]
                box = [max(0, x1 - fw), max(0, y1 - fh * 0.6),
                       min(w, x2 + fw), min(h, y2 + fh * 1.5)]
                rc = self.role_classifier.classify_image(img, main_bbox=box)
                record['role'] = rc
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"行当分类失败 {name}: {e}")
        return record

    @staticmethod
    def organized_name(path, scene, record, seq):
        name = os.path.basename(path)
        info = record
        persons_sorted = sorted((record.get('recognized') or {}).items(), key=lambda kv: -kv[1])
        people_str = '、'.join(p.replace('（饰', '饰').rstrip('）')
                              for p, _ in persons_sorted[:MAX_PEOPLE_IN_NAME])
        parts = [info['date'], info['venue'], info['play']]
        if info['event']:
            parts[-1] = parts[-1] + info['event']
        stem = '-'.join([p for p in parts if p])
        stem = f'{stem} {scene}' if scene else stem
        if people_str:
            stem += f'-{people_str}'
        if info['photographer']:
            stem += f"-{info['photographer']}"
        stem = re.sub(r'[\\/:*?"<>|]', '', stem).strip()
        new_name = f'{seq:02d}-{stem}{os.path.splitext(name)[1]}'
        return new_name

    # ---------- 主流程 ----------

    def organize(self, input_dir, output_dir, keep_per_bucket=2,
                 gap_seconds=300, scene_labels=None, threshold=None,
                 classify_role=True, save_discarded=False,
                 scene_evidence_file=None):
        """执行完整整理流水线。

        Returns:
            report dict
        """
        started = time.perf_counter()
        timings = {}
        threshold = threshold or KNOWN_FACE_THRESHOLD
        os.makedirs(output_dir, exist_ok=True)
        organized_dir = os.path.join(output_dir, 'organized')
        os.makedirs(organized_dir, exist_ok=True)

        # 1. 连拍去重选优（景别感知）
        dedup = BurstDeduplicator(shot_classifier=self.shot_classifier,
                                  keep_per_bucket=keep_per_bucket)
        dedup_report = dedup.dedup_folder(input_dir, output_dir,
                                          keep_per_bucket=keep_per_bucket,
                                          save_discarded=save_discarded)
        kept_names = [os.path.basename(m['name']) for g in dedup_report['groups']
                      for m in g['members'] if m['kept']]
        timings['dedup_seconds'] = time.perf_counter() - started

        # 2. 元数据 + 识别 + 行当
        records = []
        embeddings = None
        feature_report = None
        feature_started = time.perf_counter()
        if self.embedder is not None:
            from core_modules.organize.feature_cache import FeatureService
            features = FeatureService(self.embedder, cache_dir=self.feature_cache_dir,
                                      batch_size=self.feature_batch_size)
            embeddings = features.images([os.path.join(input_dir, name) for name in kept_names])
            feature_report = features.report()
            for path, reason in features.image_errors.items():
                logger.warning(f'场景特征提取失败 {os.path.basename(path)}: {reason}')
        timings['scene_features_seconds'] = time.perf_counter() - feature_started

        scene_assignment = split_scenes(
            [os.path.join(input_dir, n) for n in kept_names],
            gap_seconds=gap_seconds, embeddings=embeddings)
        if scene_labels:
            scene_assignment = apply_labels(scene_assignment, scene_labels)
        evidence_stats = None
        if scene_evidence_file:
            from core_modules.organize.review_queue import load_scene_evidence
            evidence = load_scene_evidence(scene_evidence_file)
            scene_assignment, evidence_stats = apply_scene_evidence(
                scene_assignment, evidence)

        recognition_started = time.perf_counter()
        for name in kept_names:
            path = os.path.join(input_dir, name)
            record = self.describe_image(path, threshold, classify_role)
            if record.get('error'):
                records.append(record)
                continue
            scene = scene_assignment.get(path, 'scene-00')
            seq = len(records) + 1
            new_name = self.organized_name(path, scene, record, seq)

            shutil.copy2(path, os.path.join(organized_dir, new_name))
            record['organized_name'] = new_name
            records.append(record)
            logger.info(f"[{seq}] {name[:36]} -> {new_name[:60]}")

        timings['recognition_and_copy_seconds'] = time.perf_counter() - recognition_started
        timings['before_report_seconds'] = time.perf_counter() - started
        report = {
            'input_dir': input_dir, 'output_dir': output_dir,
            'db_root': self.face_db_root,
            'total_input': dedup_report['total'],
            'kept': len(kept_names), 'discarded': dedup_report['discarded'],
            'n_scenes': len(set(scene_assignment.values())),
            'scene_evidence': evidence_stats,
            'scene_features': feature_report,
            'timings': timings,
            'threshold': threshold,
            'burst': {k: dedup_report[k] for k in
                      ('n_groups', 'kept', 'discarded')},
            'records': records,
        }
        with open(os.path.join(output_dir, 'organize_report.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"整理完成: 输入 {report['total_input']} -> 保留 {report['kept']}"
                    f"（剪除 {report['discarded']}），{report['n_scenes']} 个场景")
        return report
