import cv2, os, time
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from typing import Dict, Tuple, List, Optional


class PersonesSegmenter:
    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            model_dir = './weights/yolo11x-seg.pt'
        self.seg_model = YOLO(model_dir)

    def segment_with_yolo(self, image, results, batch_size=4) -> np.ndarray:
        """
        优化版本的分割函数
        """
        start = time.time()
        height, width = image.shape[:2]

        # 初始化全图分割结果为背景（0）
        full_pred_seg = np.zeros((height, width), dtype=np.uint8)

        # 收集所有有效的边界框
        boxes_data = []
        boxes_coords = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # 提取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # 确保坐标不越界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 > x1 and y2 > y1:
                        sub_image = image[y1:y2, x1:x2]
                        if sub_image.size > 0:
                            boxes_data.append(sub_image)
                            boxes_coords.append((x1, y1, x2, y2))

        if not boxes_data:
            print("No valid bounding boxes found.")
            return full_pred_seg, {
                'max_mask_ratio': 0.0,
                'max_mask_box': None,
                'max_mask_area': 0
            }

        # 批量处理
        max_mask_info = {
            'max_mask_ratio': 0.0,
            'max_mask_box': None,
            'max_mask_area': 0
        }
        
        # Track statistics
        total_boxes = len(boxes_data)
        successful_masks = 0
        failed_masks_details = []  # Track details of failed masks
        
        for i in range(0, len(boxes_data), batch_size):
            batch_images = boxes_data[i:i+batch_size]
            batch_coords = boxes_coords[i:i+batch_size]
            
            # 使用YOLO模型直接预测
            with torch.no_grad():
                yolo_results = self.seg_model.predict(
                    source=batch_images, 
                    classes=[0],
                    conf=0.4,
                    verbose=False,
                    retina_masks=True
                )
            
            # 处理每张子图的结果
            for j, (yolo_result, (x1, y1, x2, y2)) in enumerate(zip(yolo_results, batch_coords)):
                target_h, target_w = y2 - y1, x2 - x1
                box_area = target_h * target_w
                box_ratio = box_area / (height * width)
                
                # Debug info for large boxes
                if box_ratio > 0.5:  # If box occupies more than 50% of image
                    print(f"Large box detected: {x1},{y1},{x2},{y2} (area ratio: {box_ratio:.2f})")
                
                if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
                    masks_data = yolo_result.masks.data.cpu().numpy()
                    if len(masks_data) > 0:
                        # 选择最大的掩码
                        if len(masks_data) > 1:
                            # 优化：使用向量化计算面积
                            areas = np.sum(masks_data, axis=(1, 2))
                            largest_idx = np.argmax(areas)
                            pred_sub = masks_data[largest_idx]
                        else:
                            pred_sub = masks_data[0]
                        
                        # 二值化
                        pred_sub = (pred_sub > 0.5).astype(np.uint8) * 255
                        
                        # 调整大小
                        if pred_sub.shape[0] != target_h or pred_sub.shape[1] != target_w:
                            pred_sub = cv2.resize(pred_sub, (target_w, target_h), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        # 计算掩码面积 (修正：使用实际掩码面积而非边界框面积)
                        current_mask_area = np.sum(pred_sub > 0)
                        # 修正：使用掩码面积与整个图像面积的比例
                        current_mask_ratio = current_mask_area / (height * width)
                        
                        # 修正：检查是否是迄今为止最大的掩码
                        if current_mask_area > max_mask_info['max_mask_area']:
                            max_mask_info.update({
                                'max_mask_ratio': current_mask_ratio,
                                'max_mask_box': (x1, y1, x2, y2),
                                'max_mask_area': current_mask_area
                            })
                        
                        # 应用掩码
                        current_region = full_pred_seg[y1:y2, x1:x2]
                        # 只有在当前位置是背景或者新检测是前景时才更新
                        update_mask = (current_region == 0) & (pred_sub > 0)
                        full_pred_seg[y1:y2, x1:x2][update_mask] = 255
                        
                        successful_masks += 1  # Increment counter for successful mask
                    else:
                        failed_masks_details.append({
                            'coords': (x1, y1, x2, y2),
                            'reason': 'No masks in result',
                            'box_ratio': box_ratio
                        })
                else:
                    failed_masks_details.append({
                        'coords': (x1, y1, x2, y2),
                        'reason': 'No masks attribute',
                        'box_ratio': box_ratio
                    })

        end = time.time()
        print(f"✅ 分割结果: Processed {total_boxes} bounding box(es), generated {successful_masks} mask(s). Time: {end - start:.2f} seconds")
        print(f"max_mask_ratio: {max_mask_info['max_mask_ratio']:.3f}, max_mask_area: {max_mask_info['max_mask_area']}")
        # Print details of failed masks, especially large ones
        if failed_masks_details:
            print("Failed mask details:")
            for detail in failed_masks_details:
                coords = detail['coords']
                reason = detail['reason']
                ratio = detail['box_ratio']
                print(f"  Box {coords}: {reason} (area ratio: {ratio:.3f})")
                
                # Special attention to large failed boxes
                if ratio > 0.3:
                    print(f"    ⚠️  Large box ({ratio:.1%} of image) failed segmentation!")
        
        return full_pred_seg, max_mask_info
    def generate_contour_overlay_effect(self, image, pred_seg: np.ndarray, 
                                           overlay_color=(128, 128, 128), alpha=0.8,
                                           speed_level='auto') -> np.ndarray:
        """
        基于图像最大边长的智能自适应版本
        """
        start = time.time()
        
        height, width = image.shape[:2]
        max_side = max(height, width)
        
        # 基于最大边长的处理策略
        if speed_level == 'auto':
            # 根据图像最大边长选择策略
            if max_side > 4000:  # 超长边图像
                processing_size = 1024
            elif max_side > 2500:  # 长边图像
                processing_size = 1280
            elif max_side > 1500:  # 中等边长图像
                processing_size = 1024
            elif max_side > 800:   # 标准图像
                processing_size = None  # 不降采样
            else:                  # 小图像
                processing_size = None  # 不降采样
        
        elif speed_level == 'ultra_fast':
            # 强制下采样到小尺寸
            processing_size = 640 if max_side > 640 else None
        elif speed_level == 'fast':
            processing_size = 768 if max_side > 768 else None
        elif speed_level == 'balanced':
            processing_size = 1024 if max_side > 1024 else None
        elif speed_level == 'quality':
            processing_size = 1536 if max_side > 1536 else None
        elif speed_level == 'high_quality':
            processing_size = 2048 if max_side > 2048 else None
        else:
            processing_size = None
        
        
        # 如果需要下采样
        if processing_size and (height > processing_size or width > processing_size):
            # 计算下采样比例
            scale = min(processing_size / width, processing_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 记录原始尺寸
            original_shape = (height, width)
            
            # 下采样图像和掩码
            image_small = cv2.resize(image, (new_width, new_height), 
                                    interpolation=cv2.INTER_AREA)  # 下采样使用AREA
            pred_seg_small = cv2.resize(pred_seg, (new_width, new_height), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # 处理
            result_small = self._generate_overlay_optimized(image_small, pred_seg_small, 
                                                        overlay_color, alpha)
            
            # 上采样回原始尺寸
            result_image = cv2.resize(result_small, (width, height), 
                                    interpolation=cv2.INTER_LINEAR)
            
            # 如果掩码边界需要更清晰，可以结合原始掩码进行边缘锐化
            if alpha < 1.0:  # 只有半透明时才需要锐化边缘
                # 创建一个边缘锐化的版本
                person_mask = pred_seg > 0
                if person_mask.any():
                    # 获取原始图像的边缘
                    edges = cv2.Canny(person_mask.astype(np.uint8) * 255, 50, 150)
                    edges = edges > 0
                    
                    # 在边缘处使用原图像素
                    result_image[edges] = image[edges]
        
        else:
            # 原始尺寸处理
            result_image = self._generate_overlay_optimized(image, pred_seg, 
                                                        overlay_color, alpha)
        
        end = time.time()
        print(f"✅ 轮廓叠加效果已生成。图像尺寸：{width}x{height}，耗时：{end - start:.4f}秒")
        
        return result_image

    def _generate_overlay_optimized(self, image, pred_seg, overlay_color, alpha):
        """优化的覆盖效果生成函数"""
        # 创建掩码
        mask = pred_seg > 0
        
        if not mask.any():
            return image.copy()
        
        # 创建覆盖层
        overlay = np.full_like(image, overlay_color)
        
        if alpha == 1.0:
            result = image.copy()
            result[mask] = overlay_color
        elif alpha == 0.0:
            result = image.copy()
        else:
            # 使用整数运算加速
            alpha_int = int(alpha * 256)
            inv_alpha_int = 256 - alpha_int
            
            # 只处理掩码区域
            result = image.copy()
            
            # 获取掩码像素
            mask_indices = np.where(mask)
            
            if len(mask_indices[0]) > 0:
                # 提取需要处理的像素
                image_pixels = image[mask]
                overlay_pixels = overlay[mask]
                
                # 整数混合
                blended = ((image_pixels.astype(np.int32) * inv_alpha_int + 
                        overlay_pixels.astype(np.int32) * alpha_int) >> 8)
                
                # 裁剪并赋值
                result[mask] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
