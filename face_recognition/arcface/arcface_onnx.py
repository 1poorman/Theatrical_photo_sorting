# -*- coding: utf-8 -*-
"""
ArcFace 人脸特征提取器（ONNX 推理）
参考 insightface 官方实现 python-package/insightface/model_zoo/arcface_onnx.py + utils/face_align.py
使用 weights/arcface/Glint100.onnx（输入 112x112 RGB，输出 512 维 L2 归一化特征）
"""
import os
import os.path as osp
import numpy as np
import cv2
import onnx
import onnxruntime


# ArcFace 标准 5 点目标关键点（112x112）（官方 face_align.py）
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def estimate_norm(lmk, image_size=112):
    """计算相似变换矩阵（官方实现）"""
    assert lmk.shape == (5, 2)
    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    # 使用 skimage 的 SimilarityTransform，或手动实现（避免额外依赖）
    M = _estimate_similarity_transform(lmk, dst)
    return M


def _estimate_similarity_transform(src, dst):
    """最小二乘估计相似变换（scale + rotation + translation），返回 2x3 矩阵"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # 计算缩放
    src_norm = np.sqrt(np.sum(src_demean ** 2))
    dst_norm = np.sqrt(np.sum(dst_demean ** 2))
    scale = dst_norm / src_norm if src_norm > 0 else 1.0

    # 计算旋转（通过协方差矩阵的 SVD）
    H = src_demean.T @ dst_demean
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - scale * R @ src_mean
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M[:2, :]


def norm_crop(img, landmark, image_size=112):
    """基于 5 关键点对齐裁剪（官方实现，不依赖 skimage）"""
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class ArcFaceFeatureExtractor:
    """ArcFace 特征提取器（onnxruntime）"""

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: Glint100.onnx 路径
            device: 'cpu' 或 'cuda'
        """
        self.model_file = model_path
        if model_path is None or not osp.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")

        providers = self._select_providers(device)
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

        # 判断预处理参数（官方逻辑：mxnet 模型带 Sub/Mul 节点）
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for node in graph.node[:8]:
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            self.input_mean = 0.0
            self.input_std = 1.0
        else:
            self.input_mean = 127.5
            self.input_std = 127.5

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        self.input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])  # (112, 112)
        self.input_shape = input_shape

        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.output_shape = outputs[0].shape
        assert len(self.output_names) == 1, "ArcFace model should have single output"
        print(f"ArcFace loaded from {model_path} (input_size={self.input_size}, "
              f"mean={self.input_mean}, std={self.input_std}, providers={providers})")

    def _select_providers(self, device):
        available = onnxruntime.get_available_providers()
        if device is None or device.startswith('cuda'):
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def prepare(self, ctx_id, **kwargs):
        """兼容官方接口"""
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get_feat(self, imgs):
        """批量提取特征（官方实现）
        Args:
            imgs: list of aligned face images (BGR, 112x112) 或单个
        Returns:
            np.ndarray: (N, 512) 已 L2 归一化的特征
        """
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def compute_sim(self, feat1, feat2):
        """计算余弦相似度（官方实现）"""
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    # ---- 兼容原 FaceFeatureExtractor 接口 ----
    def align_face(self, image, landmarks):
        """基于 5 关键点对齐裁剪为 112x112（ArcFace 标准对齐）"""
        try:
            if image is None or image.size == 0:
                return None
            if landmarks is None or len(landmarks) < 5:
                # 无关键点时居中裁剪
                h, w = image.shape[:2]
                size = min(h, w)
                y1, x1 = (h - size) // 2, (w - size) // 2
                cropped = image[y1:y1 + size, x1:x1 + size]
                return cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_LINEAR)
            lmk = np.array([[lm[0], lm[1]] for lm in landmarks[:5]], dtype=np.float32)
            return norm_crop(image, lmk, image_size=self.input_size[0])
        except Exception as e:
            print(f"Error in align_face: {e}")
            return cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)

    def extract_features(self, face_images):
        """提取人脸特征向量
        Args:
            face_images: list of aligned face images (BGR 或 RGB, 112*112)
        Returns:
            features: numpy array of shape (n_faces, 512)，已 L2 归一化
        """
        processed = []
        for img in face_images:
            if img is None:
                raise ValueError("Input image is None")
            # Resize 到 112x112
            if tuple(img.shape[:2]) != tuple(self.input_size):
                img = cv2.resize(img, tuple(self.input_size))
            # 确保 3 通道
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            processed.append(img)
        features = self.get_feat(processed)
        return features

    def l2_normalize(self, features):
        """L2 归一化特征向量（ArcFace 输出本身已归一化，此接口保留兼容）"""
        features = np.asarray(features, dtype=np.float32)
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        return features / norm


if __name__ == '__main__':
    extractor = ArcFaceFeatureExtractor(
        model_path='/home/huachenghao/codes/Theatrical_photo_sorting-251212/weights/arcface/Glint100.onnx',
        device='cpu')
    # 测试：随机噪声
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    feat = extractor.extract_features([img])
    print('feature shape:', feat.shape)
    print('feature norm:', np.linalg.norm(feat[0]))
