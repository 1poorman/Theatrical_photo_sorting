import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2, os
import numpy as np
# Removed facenet_pytorch import since we're implementing the functions ourselves
from config.base import  CUDA_DEVICE


def fixed_image_standardization(image_tensor):
    """Standardize the image tensor for FaceNet processing.
    
    Args:
        image_tensor: Tensor of shape (3, H, W) with pixel values in [0, 1]
        
    Returns:
        Standardized tensor with same shape
    """
    # Standard normalization for FaceNet
    # Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    # Which is equivalent to: (image_tensor - 0.5) / 0.5 = 2*image_tensor - 1
    return (image_tensor - 0.5) / 0.5


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None):
        """Initialize the Inception Resnet V1 model.
        
        Args:
            pretrained (str, optional): If 'vggface2' or 'casia-webface', load pretrained weights
            classify (bool): If True, add a classification layer
            num_classes (int, optional): Number of classes for classification layer
        """
        super(InceptionResnetV1, self).__init__()
        
        self.classify = classify
        self.num_classes = num_classes

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        if self.classify and self.num_classes is not None:
            self.dropout = nn.Dropout(0.6)
            self.last_linear = nn.Linear(1792, 512)
            self.last_bn = nn.BatchNorm1d(512)
            self.logits = nn.Linear(512, self.num_classes)
        else:
            # For feature extraction, add these layers without logits
            self.dropout = nn.Dropout(0.6)
            self.last_linear = nn.Linear(1792, 512)
            self.last_bn = nn.BatchNorm1d(512)
            # No logits layer for feature extraction
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        x = self.last_bn(x)
        
        if self.classify and hasattr(self, 'logits'):
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = self.logits(x)
        else:
            # For feature extraction, L2 normalize the features
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            
        return x


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
        
        # 加载预训练FaceNet模型 (使用InceptionResnetV1直接作为特征提取器)
        self.model = InceptionResnetV1(classify=False).eval()
        
        if model_path and os.path.exists(model_path):
            try:
                # Load trained model weights
                state_dict = torch.load(model_path, map_location=device)
                
                # Check if this is a complete model checkpoint or just state dict
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    # It's a checkpoint with optimizer and other states
                    # Load with strict=False to ignore missing keys for classifier layers
                    self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
                    print(f"Loaded complete model checkpoint from {model_path}")
                else:
                    # It's just the state dict
                    # Load with strict=False to ignore missing keys for classifier layers
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded model state dict from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using default pretrained model")
                # Note: We're not loading pretrained weights since we've removed the dependency
                self.model = InceptionResnetV1(classify=False).eval()
                self.model.to(device)
        else:
            print("No model path provided or model file not found, using default model")
            self.model = InceptionResnetV1(classify=False).eval()
            self.model.to(device)
        
        self.model.to(device)
        
        # 图像预处理 (使用fixed_image_standardization代替自定义normalize)
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
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
            # 目标尺寸为160x160 (FaceNet的标准输入尺寸)
            desired_w, desired_h = 160, 160
            
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
            return cv2.resize(image, (160, 160), interpolation=cv2.INTER_LINEAR)
    
    def extract_features(self, face_images):
        """
        提取人脸特征向量
        Args:
            face_images: list of aligned face images (RGB, 160*160)
        Returns:
            features: numpy array of shape (n_faces, 512)
        """
        batch_tensor = []
        for img in face_images:
            # 添加检查确保img不是None
            if img is None:
                raise ValueError("Input image is None")
                
            # Resize to 160x160 if needed (FaceNet standard size)
            if img.shape[:2] != (160, 160):
                img = cv2.resize(img, (160, 160))
                
            # Convert BGR to RGB if needed
            if img.shape[2] == 3:  # Check if it's a color image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            tensor = self.transform(img).unsqueeze(0)
            batch_tensor.append(tensor)
        
        batch_tensor = torch.cat(batch_tensor, 0).to(self.device)
        
        with torch.no_grad():
            # Extract embeddings (512-dimensional)
            features = self.model(batch_tensor)
        
        return features.cpu().numpy()
    
    def l2_normalize(self, features):
        """L2归一化特征向量"""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norm