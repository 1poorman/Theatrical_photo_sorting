# -*- coding: utf-8 -*-
"""burst_dedup - 连拍照片分组去重与选优（F3）

流程：
1. 文件名级分组：同一基础名（去掉连拍序号 (N) / -N- 段）的照片为一组连拍
2. 组内逐张计算全图质量分（清晰度/亮度/对比度）+ 景别分类
3. 景别感知保留：组内按景别分桶，每桶保留质量分最高的 top-K 张
4. 输出 best/ 与 discarded/ 目录 + dedup_report.json
"""
import os
import json
import shutil
import cv2
import numpy as np

from core_modules.organize.filename_parser import parse_filename
from core_modules.tools.logger import get_app_logger

logger = get_app_logger()


def normalize_group_key(filename):
    """从文件名提取连拍分组键：解析后的 base（已移除连拍序号与摄影师）。"""
    info = parse_filename(filename)
    return info['base'] or info['raw'].replace(' ', '')


def image_quality(img):
    """全图质量评估（舞台照向）：清晰度 0.5 + 亮度合理度 0.25 + 对比度 0.25。

    Args:
        img: BGR ndarray
    Returns:
        dict: {sharpness, brightness, contrast, score}，score ∈ [0,1]
    """
    if img is None or img.size == 0:
        return {'sharpness': 0.0, 'brightness': 0.0, 'contrast': 0.0, 'score': 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 清晰度：Laplacian 方差，用对数尺度归一（舞台照 var 常见 100~3000）
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(1.0, np.log10(max(lap_var, 1.0)) / 3.5)  # var=3162 -> 1.0

    # 亮度：偏离合理区间惩罚（舞台照 30~235 可接受）
    brightness = float(np.mean(gray))
    if 40 <= brightness <= 200:
        bright_score = 1.0
    elif 25 <= brightness <= 225:
        bright_score = 0.7
    else:
        bright_score = 0.3

    # 对比度
    contrast = float(gray.std())
    contrast_score = min(1.0, contrast / 55.0)

    score = 0.5 * sharpness + 0.25 * bright_score + 0.25 * contrast_score
    return {'sharpness': round(float(lap_var), 1), 'brightness': round(brightness, 1),
            'contrast': round(contrast, 1), 'score': round(float(score), 4)}


class BurstDeduplicator:
    """连拍分组 + 景别感知选优。"""

    def __init__(self, shot_classifier=None, keep_per_bucket=2):
        """
        Args:
            shot_classifier: ShotTypeClassifier 实例（None 则不做景别分桶）
            keep_per_bucket: 每个景别桶保留张数
        """
        self.shot_classifier = shot_classifier
        self.keep_per_bucket = keep_per_bucket

    def group_bursts(self, image_paths):
        """按连拍分组。返回 [(group_key, [paths...]), ...]，组按大小降序。"""
        groups = {}
        for p in image_paths:
            key = normalize_group_key(p)
            groups.setdefault(key, []).append(p)
        return sorted(groups.items(), key=lambda kv: -len(kv[1]))

    def _shot_type(self, img, name):
        """景别分类；无分类器或失败时返回 None。"""
        if self.shot_classifier is None:
            return None
        try:
            h, w = img.shape[:2]
            # 估计人物面积占比（简化：由分类器内部姿态框计算，这里传 0 由其内部处理）
            result, _ = self.shot_classifier.classify_shot_type(img, name, area_ratio=0.0)
            return result.get('shot_type')
        except Exception as e:
            logger.warning(f"景别分类失败 {name}: {e}")
            return None

    def dedup_folder(self, input_dir, output_dir, keep_per_bucket=None,
                     save_discarded=True, recursive=False):
        """对目录执行连拍去重选优。

        Args:
            input_dir: 原始照片目录
            output_dir: 输出目录（best/ discarded/ dedup_report.json）
            keep_per_bucket: 每景别桶保留张数（默认用实例配置）
            save_discarded: 是否复制被剪照片到 discarded/
        Returns:
            report dict
        """
        from core_modules.tools.image_io import list_images
        keep_per_bucket = keep_per_bucket or self.keep_per_bucket

        names = list_images(input_dir, recursive=recursive)
        paths = [os.path.join(input_dir, n) for n in names]
        if not paths:
            logger.warning(f"目录无图片: {input_dir}")
            return {'groups': [], 'total': 0, 'kept': 0, 'discarded': 0}

        best_dir = os.path.join(output_dir, 'best')
        disc_dir = os.path.join(output_dir, 'discarded')
        os.makedirs(best_dir, exist_ok=True)
        if save_discarded:
            os.makedirs(disc_dir, exist_ok=True)

        groups = self.group_bursts(paths)
        report_groups = []
        n_kept = n_disc = 0

        for gi, (key, members) in enumerate(groups):
            entries = []
            for p in members:
                img = cv2.imread(p)
                q = image_quality(img)
                shot = self._shot_type(img, os.path.basename(p)) if img is not None else None
                entries.append({'path': p, 'name': os.path.basename(p),
                                'quality': q, 'shot_type': shot})

            # 景别分桶（None 归入同桶）
            buckets = {}
            for e in entries:
                buckets.setdefault(e['shot_type'], []).append(e)

            # 每桶按质量分保留 top-K
            for e in entries:
                e['kept'] = False
            for bucket in buckets.values():
                bucket.sort(key=lambda e: -e['quality']['score'])
                for e in bucket[:keep_per_bucket]:
                    e['kept'] = True

            for e in entries:
                if e['kept']:
                    dst = os.path.join(best_dir, e['name'])
                    if not os.path.exists(dst):
                        shutil.copy2(e['path'], dst)
                    n_kept += 1
                elif save_discarded:
                    dst = os.path.join(disc_dir, e['name'])
                    if not os.path.exists(dst):
                        shutil.copy2(e['path'], dst)
                    n_disc += 1
                else:
                    n_disc += 1

            report_groups.append({
                'group': gi + 1, 'key': key, 'size': len(members),
                'members': [{'name': e['name'], 'shot_type': e['shot_type'],
                             'quality': e['quality'], 'kept': e['kept']} for e in entries],
            })
            logger.info(f"组 {gi+1}: {len(members)} 张 -> 保留 "
                        f"{sum(1 for e in entries if e['kept'])}")

        report = {
            'input_dir': input_dir, 'keep_per_bucket': keep_per_bucket,
            'total': len(paths), 'kept': n_kept, 'discarded': n_disc,
            'n_groups': len(groups), 'groups': report_groups,
        }
        with open(os.path.join(output_dir, 'dedup_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"连拍去重完成: {len(paths)} 张 -> 保留 {n_kept} / 剪除 {n_disc}"
                    f"（{len(groups)} 组）")
        return report
