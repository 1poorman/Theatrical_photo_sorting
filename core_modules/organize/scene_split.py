# -*- coding: utf-8 -*-
"""scene_split - 场景划分

主策略：EXIF 拍摄时间排序，时间间隔 > GAP 切段（一场拍摄中的场景切换通常伴随
长时间停顿/走位）；辅策略：段内 SigLIP2 embedding 聚类细分布景变化。

输出：每张照片的 scene 标签（scene-01、scene-02…，占位命名，
支持外部 scene_labels.json 人工映射覆盖为幕次场景名）。
"""
import os
import json
from datetime import datetime

from PIL import Image
from core_modules.tools.image_io import list_images
from core_modules.tools.logger import get_app_logger

logger = get_app_logger()

DEFAULT_GAP_SECONDS = 300


def read_exif_datetime(img_path):
    """读取 EXIF 拍摄时间（DateTimeOriginal 优先，退回 DateTime），失败返回 None。"""
    try:
        img = Image.open(img_path)
        exif = img.getexif()
        dt = None
        # 36867 DateTimeOriginal（在 Exif IFD 中）
        try:
            ifd = exif.get_ifd(0x8769)
            dt = ifd.get(36867)
        except Exception:
            pass
        if not dt:
            dt = exif.get(306)   # DateTime
        if not dt:
            return None
        return datetime.strptime(str(dt).strip(), '%Y:%m:%d %H:%M:%S')
    except Exception:
        return None


def split_scenes(image_paths, gap_seconds=DEFAULT_GAP_SECONDS, embeddings=None,
                 max_sub_k=3):
    """场景划分。

    Args:
        image_paths: 图片路径列表
        gap_seconds: 时间间隔阈值（秒）
        embeddings: 可选 SigLIP2 特征矩阵（与 paths 对齐），用于段内细分
        max_sub_k: 段内最大细分簇数
    Returns:
        dict: {image_path: scene_id}，scene_id 形如 'scene-01'
    """
    timed = []
    no_time = []
    for p in image_paths:
        dt = read_exif_datetime(p)
        if dt:
            timed.append((dt, p))
        else:
            no_time.append(p)
    timed.sort(key=lambda x: x[0])
    logger.info(f"EXIF 时间: {len(timed)} 有 / {len(no_time)} 无")

    segments = []
    cur = []
    for dt, p in timed:
        if cur and (dt - cur[-1][0]).total_seconds() > gap_seconds:
            segments.append(cur)
            cur = []
        cur.append((dt, p))
    if cur:
        segments.append(cur)
    # 无时间照片追加为独立段（按文件名排序，包装为 (None, path) 保持结构一致）
    if no_time:
        segments.append([(None, p) for p in sorted(no_time)])

    assignment = {}
    for si, seg in enumerate(segments):
        scene_id = f'scene-{si + 1:02d}'
        seg_paths = [p for _, p in seg]
        # 段内细分（有 embedding 且段较大时）
        if embeddings is not None and len(seg_paths) >= 8 and max_sub_k > 1:
            sub = _subcluster(seg_paths, embeddings, max_sub_k)
            for k, paths in enumerate(sub):
                sid = f'{scene_id}-{chr(96 + k + 1)}' if len(sub) > 1 else scene_id
                for p in paths:
                    assignment[p] = sid
        else:
            for p in seg_paths:
                assignment[p] = scene_id

    n_scenes = len(set(assignment.values()))
    logger.info(f"场景划分: {len(image_paths)} 张 -> {n_scenes} 个场景")
    return assignment


def _subcluster(paths, embeddings, max_k):
    """段内 KMeans 细分（k 由轮廓系数自动选，上限 max_k）。"""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    idx = {p: i for i, p in enumerate(paths)}
    X = np.array([embeddings[idx[p]] for p in paths])
    best_k, best_score = 1, -1.0
    for k in range(2, min(max_k, len(paths) - 1) + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
            s = silhouette_score(X, labels)
            if s > best_score:
                best_score, best_k = s, k
                best_labels = labels
        except Exception:
            continue
    groups = {k: [] for k in range(best_k)}
    if best_k == 1:
        return [list(paths)]
    for p, l in zip(paths, best_labels):
        groups[int(l)].append(p)
    return [groups[k] for k in sorted(groups) if groups[k]]


def apply_labels(assignment, labels_file):
    """应用人工场景标注映射（scene_labels.json: {scene-01: '第1幕克段', ...}）。"""
    if not labels_file or not os.path.exists(labels_file):
        return assignment
    with open(labels_file, encoding='utf-8') as f:
        mapping = json.load(f)
    return {p: mapping.get(s, s) for p, s in assignment.items()}
