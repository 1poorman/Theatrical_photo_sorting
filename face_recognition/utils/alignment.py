import cv2
import numpy as np
import math
from typing import List, Tuple, Optional

class FaceAlignment:
    """
    人脸对齐工具类
    实现多种对齐方法：仿射变换、相似变换、透视变换
    """
    
    @staticmethod
    def get_alignment_matrix(src_points: np.ndarray, dst_points: np.ndarray, 
                            method: str = 'similarity') -> np.ndarray:
        """
        根据源点和目标点计算变换矩阵
        
        Args:
            src_points: 源关键点 (n, 2)
            dst_points: 目标关键点 (n, 2)
            method: 变换方法 ['affine', 'similarity', 'perspective']
        
        Returns:
            变换矩阵
        """
        if method == 'affine':
            # 仿射变换 (至少3个点)
            if len(src_points) >= 3:
                matrix = cv2.getAffineTransform(src_points[:3].astype(np.float32),
                                              dst_points[:3].astype(np.float32))
                return matrix
            else:
                raise ValueError("Affine transform requires at least 3 points")
        
        elif method == 'similarity':
            # 相似变换 (旋转、缩放、平移)
            if len(src_points) >= 2:
                # 使用最小二乘法计算相似变换
                src_mean = np.mean(src_points, axis=0)
                dst_mean = np.mean(dst_points, axis=0)
                
                # 中心化
                src_centered = src_points - src_mean
                dst_centered = dst_points - dst_mean
                
                # 计算缩放和旋转
                A = np.dot(src_centered.T, dst_centered)
                U, _, Vt = np.linalg.svd(A)
                R = np.dot(Vt.T, U.T)
                
                # 确保右手坐标系
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = np.dot(Vt.T, U.T)
                
                # 计算缩放因子
                scale = np.trace(np.dot(A, R)) / np.trace(np.dot(src_centered.T, src_centered))
                
                # 构造变换矩阵
                matrix = np.zeros((2, 3), dtype=np.float32)
                matrix[0:2, 0:2] = scale * R
                matrix[:, 2] = dst_mean - scale * np.dot(R, src_mean.T).T
                
                return matrix
            else:
                raise ValueError("Similarity transform requires at least 2 points")
        
        elif method == 'perspective':
            # 透视变换 (至少4个点)
            if len(src_points) >= 4:
                matrix = cv2.getPerspectiveTransform(src_points[:4].astype(np.float32),
                                                   dst_points[:4].astype(np.float32))
                return matrix
            else:
                raise ValueError("Perspective transform requires at least 4 points")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def align_face_by_eyes(image: np.ndarray, landmarks: np.ndarray, 
                          output_size: Tuple[int, int] = (160, 160),
                          left_eye_idx: int = 0, right_eye_idx: int = 1) -> np.ndarray:
        """
        基于眼睛位置进行人脸对齐（常用方法）
        
        Args:
            image: 输入图像
            landmarks: 人脸关键点
            output_size: 输出图像大小
            left_eye_idx: 左眼关键点索引
            right_eye_idx: 右眼关键点索引
        
        Returns:
            对齐后的人脸图像
        """
        # 提取眼睛位置
        left_eye = landmarks[left_eye_idx]
        right_eye = landmarks[right_eye_idx]
        
        # 计算眼睛中心点
        eyes_center = ((left_eye[0] + right_eye[0]) * 0.5,
                      (left_eye[1] + right_eye[1]) * 0.5)
        
        # 计算眼睛连线与水平线的夹角
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.atan2(dy, dx) * 180.0 / math.pi
        
        # 计算缩放因子
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = output_size[0] * 0.4  # 期望的眼睛距离
        scale = desired_dist / dist if dist > 0 else 1.0
        
        # 构造旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # 调整平移量，使眼睛位于指定位置
        desired_eye_center = (output_size[0] * 0.5, output_size[1] * 0.4)
        rotation_matrix[0, 2] += desired_eye_center[0] - eyes_center[0]
        rotation_matrix[1, 2] += desired_eye_center[1] - eyes_center[1]
        
        # 执行旋转和缩放
        aligned_face = cv2.warpAffine(image, rotation_matrix, output_size,
                                     flags=cv2.INTER_CUBIC)
        
        return aligned_face
    
    @staticmethod
    def align_face_advanced(image: np.ndarray, landmarks: np.ndarray,
                           template_type: str = 'default',
                           output_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        高级人脸对齐，基于预定义模板
        
        Args:
            image: 输入图像
            landmarks: 5个关键点 [左眼, 右眼, 鼻子, 左嘴角, 右嘴角]
            template_type: 模板类型 ['default', 'affine', 'similarity']
            output_size: 输出图像大小
        
        Returns:
            对齐后的人脸图像
        """
        # 定义标准人脸模板 (基于MTCNN格式)
        if template_type == 'default':
            # 标准模板 (VGG-Face2)
            template = np.array([
                [0.341916, 0.461574],  # 左眼
                [0.656533, 0.459833],  # 右眼
                [0.500225, 0.640505],  # 鼻子
                [0.370975, 0.785691],  # 左嘴角
                [0.631516, 0.783486]   # 右嘴角
            ])
        elif template_type == 'affine':
            # Affine变换模板
            template = np.array([
                [0.3, 0.4],
                [0.7, 0.4],
                [0.5, 0.6]
            ])[:min(3, len(landmarks))]
        else:
            template = np.array([
                [0.3, 0.4],
                [0.7, 0.4],
                [0.5, 0.6],
                [0.3, 0.8],
                [0.7, 0.8]
            ])[:min(5, len(landmarks))]
        
        # 缩放模板到输出大小
        template[:, 0] *= output_size[0]
        template[:, 1] *= output_size[1]
        
        # 根据关键点数量选择对齐方法
        if len(landmarks) >= 5 and template_type == 'default':
            # 使用5点透视变换
            src_points = landmarks[:5].astype(np.float32)
            dst_points = template[:5].astype(np.float32)
            matrix = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
            aligned_face = cv2.warpAffine(image, matrix, output_size,
                                         flags=cv2.INTER_CUBIC)
        
        elif len(landmarks) >= 3:
            # 使用3点仿射变换
            src_points = landmarks[:3].astype(np.float32)
            dst_points = template[:3].astype(np.float32)
            matrix = cv2.getAffineTransform(src_points, dst_points)
            aligned_face = cv2.warpAffine(image, matrix, output_size,
                                         flags=cv2.INTER_CUBIC)
        
        else:
            # 使用2点相似变换
            aligned_face = FaceAlignment.align_face_by_eyes(
                image, landmarks, output_size)
        
        return aligned_face
    
    @staticmethod
    def detect_face_orientation(landmarks: np.ndarray) -> str:
        """
        检测人脸朝向
        
        Args:
            landmarks: 人脸关键点
        
        Returns:
            orientation: 朝向 ['front', 'left', 'right', 'up', 'down']
        """
        if len(landmarks) < 2:
            return 'unknown'
        
        # 计算眼睛连线角度
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.atan2(dy, dx) * 180.0 / math.pi
        
        # 判断朝向
        if abs(angle) < 15:
            return 'front'
        elif angle > 15:
            return 'right'
        elif angle < -15:
            return 'left'
        else:
            return 'front'
    
    @staticmethod
    def crop_face_with_margin(image: np.ndarray, bbox: List[int], 
                             margin_ratio: float = 0.3) -> np.ndarray:
        """
        裁剪人脸区域，并添加边界
        
        Args:
            image: 输入图像
            bbox: 边界框 [x1, y1, x2, y2]
            margin_ratio: 边界比例
        
        Returns:
            裁剪后的人脸区域
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 计算边界框尺寸
        width = x2 - x1
        height = y2 - y1
        
        # 添加边界
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        
        # 计算新的边界框
        x1_new = max(0, x1 - margin_x)
        y1_new = max(0, y1 - margin_y)
        x2_new = min(w, x2 + margin_x)
        y2_new = min(h, y2 + margin_y)
        
        # 确保宽高比大致为1:1
        crop_width = x2_new - x1_new
        crop_height = y2_new - y1_new
        
        if crop_width > crop_height:
            # 需要增加高度
            diff = crop_width - crop_height
            y1_new = max(0, y1_new - diff // 2)
            y2_new = min(h, y2_new + diff // 2)
        elif crop_height > crop_width:
            # 需要增加宽度
            diff = crop_height - crop_width
            x1_new = max(0, x1_new - diff // 2)
            x2_new = min(w, x2_new + diff // 2)
        
        # 裁剪人脸区域
        face_crop = image[y1_new:y2_new, x1_new:x2_new]
        
        return face_crop