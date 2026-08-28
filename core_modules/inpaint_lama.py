import cv2, time
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

class ImageInpainter:
    def __init__(self, model_path: str = None, max_size: int = 4096):
        """
        初始化图像修复管道。
        
        :param model_path: 模型路径
        :param max_size: 输入图像长边最大尺寸（用于显存控制）
        """
        if not model_path:
            model_path = './weights/cv_fft_inpainting_lama'
        self.inpainting_pipeline = pipeline(Tasks.image_inpainting, model=model_path)
        self.max_size = max_size

    def _resize_if_needed(self, img: np.ndarray, mask: np.ndarray):
        """按比例缩放图像和掩码，使长边不超过 max_size"""
        h, w = img.shape[:2]
        scale = min(self.max_size / max(h, w), 1.0)  # 只缩小，不放大
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            # 使用高质量插值（修复任务建议用 cv2.INTER_AREA 缩小）
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return img_resized, mask_resized, scale
        else:
            return img, mask, 1.0

    def _restore_size(self, repaired: np.ndarray, original_shape: tuple):
        """将修复结果还原到原始尺寸"""
        h_orig, w_orig = original_shape[:2]
        h_rep, w_rep = repaired.shape[:2]
        if (h_rep, w_rep) != (h_orig, w_orig):
            # 使用线性插值还原（视觉更平滑）
            repaired = cv2.resize(repaired, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return repaired

    def inpaint(self, img_path: str, mask_path: str) -> np.ndarray:
        """
        Execute image inpainting (supports auto-resizing for memory control).
        
        :param img_path: Original image path (BGR or RGB, OpenCV default BGR)
        :param mask_path: Mask image path (black/white, white=255 indicates area to be inpainted)
        :return: Inpainted image (OpenCV format, BGR, uint8)
        """
        start_time = time.time()
        # Check if files exist first
        import os
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise ValueError("Image or mask file does not exist")
        
        # Read original images for shape reference
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise ValueError("Failed to read image or mask, please check paths")

        original_shape = img.shape
        
        # Resize if needed
        img_proc, mask_proc, scale = self._resize_if_needed(img, mask)
        # print(f"Original size: {img.shape[:2]}, Resized to: {img_proc.shape[:2]}, Scale factor: {scale:.2f}")
        
        # Save resized images temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask_file:
                cv2.imwrite(tmp_img_file.name, img_proc)
                cv2.imwrite(tmp_mask_file.name, mask_proc)
                
                # Pass file paths to pipeline
                input_data = {
                    'img': tmp_img_file.name,
                    'mask': tmp_mask_file.name,
                }
                result = self.inpainting_pipeline(input_data)
                repaired_small = result[OutputKeys.OUTPUT_IMG]
                
                # Clean up temp files
                os.unlink(tmp_img_file.name)
                os.unlink(tmp_mask_file.name)
        
        if repaired_small is None:
            raise RuntimeError("Model did not return inpainted result")
        
        # Restore to original size
        repaired_full = self._restore_size(repaired_small, original_shape)
        end_time = time.time()
        print(f"✅ Inpainting time: {end_time - start_time:.2f} seconds")
        return repaired_full.astype(img.dtype)