import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2, os
import numpy as np
from facenet.nets.facenet import Facenet
from config.base import  CUDA_DEVICE

class FaceFeatureExtractor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Handle device mapping for CUDA
        if 'cuda' in device:
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = 'cpu'
            else:
                # Ensure device index is valid
                device_index = int(device.split(':')[1]) if ':' in device else 0
                if device_index >= torch.cuda.device_count():
                    print(f"Device {device} not available, using cuda:0")
                    device = 'cuda:0'
        
        self.device = device
        print(f"FaceFeatureExtractor using device: {self.device}")
        
        # 加载预训练FaceNet模型
        self.model = Facenet(backbone="inception_resnetv1", mode="predict").eval()
        
        if model_path and os.path.exists(model_path):
            try:
                # Try to load the model state dict
                state_dict = torch.load(model_path, map_location=device)
                
                # Check if this is a full model or just the backbone
                if any(key.startswith('classifier') for key in state_dict.keys()):
                    # This model includes classifier layers, load only backbone weights
                    backbone_dict = {k: v for k, v in state_dict.items() 
                                   if not k.startswith('classifier')}
                    self.model.load_state_dict(backbone_dict, strict=False)
                    print("Loaded FaceNet backbone weights (ignored classifier layers)")
                else:
                    # This is a backbone-only model
                    self.model.load_state_dict(state_dict)
                    print("Loaded FaceNet model")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using randomly initialized model")
        
        self.model.to(device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def align_face(self, image, landmarks):
        """
        基于关键点进行人脸对齐
        Args:
            image: 输入图像
            landmarks: 人脸关键点坐标 [(x1,y1), (x2,y2), ...]
        Returns:
            aligned_face: 对齐后的人脸图像
        """
        try:
            # Check if image is valid
            if image is None or image.size == 0:
                return None
                
            # 定义标准人脸关键点位置 (基于常见的对齐方式)
            # 目标尺寸为1024x1024
            desired_w, desired_h = 1024, 1024
            
            # 定义目标关键点位置 (经验值)
            # 这些值可以根据实际需求调整
            desired_eye_left = (0.35 * desired_w, 0.35 * desired_h)
            desired_eye_right = (0.65 * desired_w, 0.35 * desired_h)
            desired_nose = (0.50 * desired_w, 0.50 * desired_h)
            
            # 提取原始关键点 (假设landmarks按照特定顺序排列)
            # 通常顺序是: 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角
            if len(landmarks) >= 3:
                # Convert landmarks to the right format if needed
                if isinstance(landmarks[0], (list, tuple)) and len(landmarks[0]) == 2:
                    eye_left = landmarks[0]
                    eye_right = landmarks[1]
                    nose = landmarks[2]
                else:
                    # If landmarks are in a different format, we need to handle accordingly
                    # This is a fallback for when we estimate landmarks for pre-cropped faces
                    eye_left = landmarks[0]
                    eye_right = landmarks[1]
                    nose = landmarks[2]
            else:
                # 如果关键点不足，返回原图的居中裁剪
                h, w = image.shape[:2]
                size = min(h, w)
                y1, x1 = (h - size) // 2, (w - size) // 2
                cropped = image[y1:y1+size, x1:x1+size]
                return cv2.resize(cropped, (desired_w, desired_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建源点和目标点数组
            src_points = np.float32([
                eye_left,
                eye_right,
                nose
            ])
            
            dst_points = np.float32([
                desired_eye_left,
                desired_eye_right,
                desired_nose
            ])
            
            # 计算仿射变换矩阵
            M = cv2.getAffineTransform(src_points, dst_points)
            
            # 应用仿射变换
            aligned_face = cv2.warpAffine(image, M, (desired_w, desired_h), flags=cv2.INTER_LINEAR)
            
            return aligned_face
        except Exception as e:
            print(f"Error in align_face: {str(e)}")
            # Return resized image as fallback
            return cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    def extract_features(self, face_images):
        """
        提取人脸特征向量
        Args:
            face_images: list of aligned face images (RGB, 1024*1024)
        Returns:
            features: numpy array of shape (n_faces, 512)
        """
        batch_tensor = []
        for img in face_images:
            # 添加检查确保img不是None
            if img is None:
                raise ValueError("Input image is None")
                
            if img.shape[:2] != (1024, 1024):
                img = cv2.resize(img, (1024, 1024))
            tensor = self.transform(img).unsqueeze(0)
            batch_tensor.append(tensor)
        
        batch_tensor = torch.cat(batch_tensor, 0).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        return features.cpu().numpy()
    
    def l2_normalize(self, features):
        """L2归一化特征向量"""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norm