# -*- coding: utf-8 -*-
"""tools.image_io - 图像读写公共工具

收敛各模块重复实现的图像解码/缩放/枚举逻辑：
- imread_reduced: JPEG 降采样解码（DCT 域缩放，比全解码后 resize 快约 4 倍）
- resize_max_side: 最长边限制缩放
- list_images: 目录图片枚举
"""
import os
import cv2

# OpenCV 降采样解码倍率映射
_REDUCED_FLAGS = {
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def imread_reduced(path, target_max=1920):
    """降采样解码：保证解码后最长边 >= target_max 的前提下，选择最大降采样倍率。

    若流程后续本就会缩放到 target_max，则该解码与"全分辨率解码 + resize"等价，
    但速度快约 4 倍（大图场景为端到端主要瓶颈）。

    Args:
        path: 图片路径
        target_max: 解码后允许的最小最长边
    Returns:
        np.ndarray (BGR)，读取失败返回 None
    """
    factor = 1
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        for f in (8, 4, 2):
            if max(w, h) // f >= target_max:
                factor = f
                break
    except Exception:
        factor = 1
    if factor == 1:
        return cv2.imread(path)
    return cv2.imread(path, _REDUCED_FLAGS[factor])


def resize_max_side(image, max_side=1920, interpolation=cv2.INTER_LINEAR):
    """等比缩放，使最长边不超过 max_side；已小于等于时原样返回。"""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interpolation)


def list_images(folder, recursive=False):
    """枚举目录下的图片文件（按文件名排序）。

    Args:
        folder: 目录路径
        recursive: 是否递归子目录
    Returns:
        list[str]: 文件名（recursive=False）或相对路径（recursive=True）
    """
    if recursive:
        out = []
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    out.append(os.path.relpath(os.path.join(root, f), folder))
        return sorted(out)
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTENSIONS)
                  and os.path.isfile(os.path.join(folder, f)))
