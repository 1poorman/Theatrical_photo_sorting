import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Union, Optional

class ImagePreprocessor:
    """
    图像预处理工具类
    包含图像增强、归一化、标准化等操作
    """
    
    @staticmethod
    def resize_with_pad(image: np.ndarray, target_size: Tuple[int, int],
                       keep_aspect_ratio: bool = True, 
                       padding_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        调整图像大小并保持宽高比，必要时填充
        
        Args:
            image: 输入图像
            target_size: 目标大小 (width, height)
            keep_aspect_ratio: 是否保持宽高比
            padding_color: 填充颜色
        
        Returns:
            调整后的图像
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        if keep_aspect_ratio:
            # 计算缩放因子
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 调整大小
            resized = cv2.resize(image, (new_w, new_h))
            
            # 计算填充
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            
            # 创建填充图像
            if len(image.shape) == 3:
                padded = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized
            else:
                padded = np.full((target_h, target_w), padding_color[0], dtype=np.uint8)
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
                
            return padded
        else:
            # 直接调整大小，不保持宽高比
            return cv2.resize(image, target_size)
    
    @staticmethod
    def normalize_image(image: np.ndarray, 
                       mean: List[float] = [0.5, 0.5, 0.5],
                       std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 输入图像 (0-255)
            mean: 均值
            std: 标准差
        
        Returns:
            归一化后的图像
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        if len(image.shape) == 3:
            # RGB图像
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        else:
            # 灰度图像
            image = (image - mean[0]) / std[0]
        
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray,
                         mean: List[float] = [0.5, 0.5, 0.5],
                         std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
        """
        反归一化图像
        
        Args:
            image: 归一化后的图像
            mean: 均值
            std: 标准差
        
        Returns:
            反归一化后的图像 (0-255)
        """
        denormalized = image.copy()
        
        if len(denormalized.shape) == 3:
            # RGB图像
            for i in range(3):
                denormalized[:, :, i] = denormalized[:, :, i] * std[i] + mean[i]
        else:
            # 灰度图像
            denormalized = denormalized * std[0] + mean[0]
        
        # 缩放到0-255范围
        denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
        
        return denormalized
    
    @staticmethod
    def to_tensor(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """
        将numpy图像转换为torch张量
        
        Args:
            image: 输入图像
            normalize: 是否归一化
        
        Returns:
            torch.Tensor
        """
        if normalize:
            image = ImagePreprocessor.normalize_image(image)
        
        # 转换维度: HWC -> CHW
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        else:
            image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image).float()
    
    @staticmethod
    def enhance_image(image: np.ndarray, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> np.ndarray:
        """
        图像增强
        
        Args:
            image: 输入图像
            brightness: 亮度增强因子 (1.0表示不变)
            contrast: 对比度增强因子
            saturation: 饱和度增强因子
            sharpness: 锐化增强因子
        
        Returns:
            增强后的图像
        """
        # 转换为PIL图像
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # 应用增强
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(sharpness)
        
        # 转换回numpy数组
        if len(image.shape) == 3 and image.shape[2] == 3:
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            enhanced = np.array(pil_image)
        
        return enhanced
    
    @staticmethod
    def histogram_equalization(image: np.ndarray, mode: str = 'clahe') -> np.ndarray:
        """
        直方图均衡化
        
        Args:
            image: 输入图像
            mode: 均衡化模式 ['equalize', 'clahe']
        
        Returns:
            均衡化后的图像
        """
        if len(image.shape) == 3:
            # 转换为YUV色彩空间
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            if mode == 'equalize':
                # 全局直方图均衡化
                yuv[:, :, 0] = cv2.equalizeHist(y_channel)
            elif mode == 'clahe':
                # 自适应直方图均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                yuv[:, :, 0] = clahe.apply(y_channel)
            
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图像
            if mode == 'equalize':
                equalized = cv2.equalizeHist(image)
            elif mode == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized = clahe.apply(image)
        
        return equalized
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                      sigma: float = 1.0) -> np.ndarray:
        """
        高斯模糊
        
        Args:
            image: 输入图像
            kernel_size: 卷积核大小
            sigma: 标准差
        
        Returns:
            模糊后的图像
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        中值滤波
        
        Args:
            image: 输入图像
            kernel_size: 卷积核大小
        
        Returns:
            滤波后的图像
        """
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def detect_and_remove_noise(image: np.ndarray, 
                               method: str = 'fastnl') -> np.ndarray:
        """
        检测并去除噪声
        
        Args:
            image: 输入图像
            method: 去噪方法 ['fastnl', 'bilateral', 'gaussian']
        
        Returns:
            去噪后的图像
        """
        if method == 'fastnl':
            # 快速非局部均值去噪
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        elif method == 'bilateral':
            # 双边滤波
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            # 高斯滤波
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            return image
    
    @staticmethod
    def create_preprocessing_pipeline(steps: List[dict]) -> callable:
        """
        创建预处理流水线
        
        Args:
            steps: 预处理步骤列表
        
        Returns:
            预处理函数
        """
        def pipeline(image: np.ndarray) -> np.ndarray:
            result = image.copy()
            for step in steps:
                func_name = step['func']
                params = step.get('params', {})
                
                if func_name == 'resize':
                    result = ImagePreprocessor.resize_with_pad(result, **params)
                elif func_name == 'normalize':
                    result = ImagePreprocessor.normalize_image(result, **params)
                elif func_name == 'enhance':
                    result = ImagePreprocessor.enhance_image(result, **params)
                elif func_name == 'equalize':
                    result = ImagePreprocessor.histogram_equalization(result, **params)
                elif func_name == 'blur':
                    result = ImagePreprocessor.gaussian_blur(result, **params)
                elif func_name == 'denoise':
                    result = ImagePreprocessor.detect_and_remove_noise(result, **params)
            
            return result
        
        return pipeline
    
    @staticmethod
    def get_facenet_transform() -> transforms.Compose:
        """
        获取FaceNet的标准预处理变换
        """
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])