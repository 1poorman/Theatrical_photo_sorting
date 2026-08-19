import cv2, os, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from ultralytics import YOLO

class ClothesSegmenter:
    PALETTE = [
        [0, 0, 0],        # 0: Background
        [128, 0, 128],    # 1: Hat (purple)
        [255, 255, 0],    # 2: Hair (cyan)
        [0, 255, 255],    # 3: Sunglasses (yellow)
        [0, 0, 255],      # 4: Upper-clothes (red)
        [255, 0, 255],    # 5: Skirt (magenta)
        [0, 255, 0],      # 6: Pants (lime)
        [255, 165, 0],    # 7: Dress (orange)
        [255, 255, 255],  # 8: Belt (white)
        [0, 128, 255],    # 9: Left-shoe (orange-blue)
        [255, 128, 0],    # 10: Right-shoe (blue-orange)
        [255, 255, 255],  # 11: Face (white)
        [128, 255, 0],    # 12: Left-leg (green-yellow)
        [0, 255, 128],    # 13: Right-leg (green-cyan)
        [255, 0, 128],    # 14: Left-arm (pink)
        [128, 0, 255],    # 15: Right-arm (violet)
        [128, 128, 0],    # 16: Bag (olive)
        [0, 128, 128],    # 17: Scarf (teal)
    ]
    assert len(PALETTE) == 18, "Palette must have 18 colors"

    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            model_dir = '/home/huachenghao/codes/clothes/models--mattmdjaga--segformer_b2_clothes/snapshots/584abc1e1d260e23c0fc627c5217a09b2b461046'
        self.processor = SegformerImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_dir)
        self.model.eval()
        self.yolo = YOLO("/home/huachenghao/codes/Theatrical_photo_sorting/yolo11x-seg.pt")
        


    def segment(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        height, width = image.shape[:2]

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu()

        upsampled_logits = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred_seg

    def segment_with_yolo(self, image_path: str, results) -> np.ndarray:
        """
        在 YOLO 检测到的人体边界框区域内执行分割，其余区域设为背景（0）。
        
        :param image_path: 输入图像路径
        :param results: YOLO 检测结果（ultralytics YOLOv8 格式）
        :return: 完整图像的语义分割图 (H, W)，dtype=np.uint8，与 self.segment() 返回格式一致
        """
        start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        height, width = image.shape[:2]

        # 初始化全图分割结果为背景（0）
        full_pred_seg = np.zeros((height, width), dtype=np.uint8)

        persons_detected = 0
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # 提取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # 确保坐标不越界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue  # 跳过无效框

                    sub_image = image[y1:y2, x1:x2]
                    if sub_image.size == 0:
                        continue

                    # 分割子图
                    inputs = self.processor(images=sub_image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits.cpu()
                        


                    upsampled_logits = F.interpolate(
                        logits,
                        size=(y2 - y1, x2 - x1),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_sub = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    # 写入到完整分割图对应位置
                    full_pred_seg[y1:y2, x1:x2] = pred_sub
                    persons_detected += 1
        end = time.time()

        print(f"✅ 分割结果已保存: Processed {persons_detected} person(s) with segmentation. Time taken: {end - start:.2f} seconds")
        return full_pred_seg  # 形状 (H, W)，uint8，与 segment() 返回值完全一致

    def segment_with_yolo_batched(self, image, results, batch_size=4) -> np.ndarray:
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
        
        for i in range(0, len(boxes_data), batch_size):
            batch_images = boxes_data[i:i+batch_size]
            batch_coords = boxes_coords[i:i+batch_size]
            
            # 使用YOLO模型直接预测
            with torch.no_grad():
                yolo_results = self.yolo.predict(
                    source=batch_images, 
                    classes=[0],
                    conf=0.5,
                    verbose=False,
                    retina_masks=True
                )
            
            # 处理每张子图的结果
            for j, (yolo_result, (x1, y1, x2, y2)) in enumerate(zip(yolo_results, batch_coords)):
                target_h, target_w = y2 - y1, x2 - x1
                
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
                        
                        # 计算掩码面积
                        current_mask_area = np.sum(pred_sub > 0)
                        current_mask_ratio = current_mask_area / (height * width)
                        
                        if current_mask_ratio > max_mask_info['max_mask_ratio']:
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

        end = time.time()
        print(f"✅ 分割结果: Processed {len(boxes_data)} person(s). Time: {end - start:.2f} seconds")
        
        return full_pred_seg, max_mask_info

    def generate_mask(self, pred_seg: np.ndarray, classes: list) -> np.ndarray:
        mask = np.isin(pred_seg, classes).astype(np.uint8) * 255
        return mask
    
    def extract_segmented_area(self, image_path: str, pred_seg: np.ndarray, classes: list = None) -> np.ndarray:
        """
        Extract the segmented area from the original image, setting background to black.
        
        :param image_path: Path to the original image
        :param pred_seg: Segmentation prediction array from model (H, W)
        :param classes: List of class IDs to keep (if None, keeps all non-background classes)
        :return: Image with background set to black, shape (H, W, 3), dtype=np.uint8
        """
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # If no specific classes provided, use all classes except background (0)
        if classes is None:
            classes = list(range(1, len(self.PALETTE)))
        
        # Generate mask for selected classes
        mask = np.isin(pred_seg, classes).astype(np.uint8) * 255
        
        # Convert mask to 3-channel to match image
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to image (keeping only selected areas, rest becomes black)
        extracted_image = cv2.bitwise_and(image, mask_3channel)
        
        return extracted_image

    def generate_color_mask(self, pred_seg: np.ndarray) -> np.ndarray:
        """
        根据预测结果生成彩色掩码图。
        
        :param pred_seg: 预测类别图 (H, W)
        :return: color_mask - 彩色掩码图 (H, W, 3)，dtype=np.uint8
        """
        height, width = pred_seg.shape
        color_mask = np.zeros()
        for class_id, color in enumerate(self.PALETTE):
            if class_id < len(self.PALETTE):
                color_mask[pred_seg == class_id] = color  # BGR
                color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

        return color_mask
        
    
    def generate_color_selected_mask(self, pred_seg: np.ndarray, interested_classes = None) -> np.ndarray:
        """
        根据预测结果生成彩色掩码图。
        
        :param pred_seg: 预测类别图 (H, W)
        :param interested_classes: 感兴趣的类别，默认为 None (表示所有类别), List[int]
        :return: color_mask - 彩色掩码图 (H, W, 3)，dtype=np.uint8
        """
        if interested_classes is None:
            interested_classes = [1, 4, 6]
        
        height, width = pred_seg.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
        for class_id in interested_classes:
            if class_id < len(self.PALETTE):
                color = self.PALETTE[class_id]
                color_mask[pred_seg == class_id] = color  # BGR
    
        return color_mask
        
    def generate_contour_overlay_effect(self, image, pred_seg: np.ndarray, 
                                             overlay_color=(128, 128, 128), alpha=0.8) -> np.ndarray:
        """
        使用OpenCV优化版本的轮廓叠加效果
        
        :param image: 输入图像 (H, W, 3)
        :param pred_seg: 来自 segment_with_yolo_batched 的完整分割结果
        :param overlay_color: 覆盖颜色 (B, G, R)，默认为灰色
        :param alpha: 覆盖透明度，范围 0-1，默认为 0.8
        :return: 带有轮廓叠加效果的图像 (H, W, 3)，dtype=np.uint8
        """
        start = time.time()
        
        height, width = image.shape[:2]
        
        # 确保 pred_seg 尺寸与图像匹配
        if pred_seg.shape[0] != height or pred_seg.shape[1] != width:
            pred_seg = cv2.resize(pred_seg, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 创建二值掩码
        mask = (pred_seg > 0).astype(np.uint8) * 255
        
        # 如果没有人像区域，直接返回原图
        if not np.any(mask):
            return image.copy()
        
        # 创建覆盖层
        overlay = np.full_like(image, overlay_color, dtype=np.uint8)
        
        # 使用OpenCV的addWeighted进行混合（效率更高）
        if alpha == 1.0:
            # 完全覆盖
            result = image.copy()
            result[mask > 0] = overlay[mask > 0]
        elif alpha == 0.0:
            # 完全不覆盖
            result = image.copy()
        else:
            # 使用掩码进行混合
            result = image.copy()
            
            # 方法1: 使用cv2.addWeighted + 掩码（推荐）
            # 创建一个临时图像用于混合
            blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            
            # 使用掩码将混合后的结果复制到原图
            result = cv2.bitwise_and(blended, blended, mask=mask)
            result = cv2.add(result, cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)))
            
            # 方法2: 使用numpy的where（更简洁）
            # mask_3d = np.stack([mask]*3, axis=-1)
            # result = np.where(mask_3d > 0, blended, image)
        
        end = time.time()
        print(f"✅ 轮廓叠加效果已生成（优化版）。耗时：{end - start:.4f}秒")
        return result


    def generate_contour_outline_effect(self, image_path: str, pred_seg: np.ndarray,
                                    outline_color=(128, 128, 128), outline_thickness=3) -> np.ndarray:
        """
        在检测到的人物轮廓上绘制边界线效果
        
        :param image_path: 输入图像路径
        :param pred_seg: 来自 segment_with_yolo_batched 的完整分割结果
        :param outline_color: 轮廓颜色 (B, G, R)，默认为灰色
        :param outline_thickness: 轮廓线粗细
        :return: 带有轮廓线效果的图像 (H, W, 3)，dtype=np.uint8
        """
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        height, width = image.shape[:2]
        
        # 确保 pred_seg 尺寸与图像匹配
        if pred_seg.shape[0] != height or pred_seg.shape[1] != width:
            pred_seg = cv2.resize(pred_seg, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 创建输出图像副本
        result_image = image.copy()
        
        # 为每个人物实例生成轮廓 (处理可能的多个实例)
        person_instances = pred_seg.copy()
        
        # 查找轮廓
        contours, _ = cv2.findContours((person_instances > 0).astype(np.uint8), 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在原图上绘制轮廓
        cv2.drawContours(result_image, contours, -1, outline_color, outline_thickness)
        
        return result_image


    def generate_blended_background_effect(self, image_path: str, pred_seg: np.ndarray,
                                        background_color=(128, 128, 128), alpha=0.8) -> np.ndarray:
        """
        生成背景着色效果，人物保持原样，背景变为半透明颜色
        
        :param image_path: 输入图像路径
        :param pred_seg: 来自 segment_with_yolo_batched 的完整分割结果
        :param background_color: 背景颜色 (B, G, R)，默认为灰色
        :param alpha: 背景透明度，范围 0-1，默认为 0.8
        :return: 带有背景着色效果的图像 (H, W, 3)，dtype=np.uint8
        """
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        height, width = image.shape[:2]
        
        # 确保 pred_seg 尺寸与图像匹配
        if pred_seg.shape[0] != height or pred_seg.shape[1] != width:
            pred_seg = cv2.resize(pred_seg, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 创建输出图像副本
        result_image = image.copy()
        
        # 创建背景区域掩码（背景区域）
        background_mask = (pred_seg == 0).astype(bool)
        
        # 创建背景颜色层
        background_layer = np.full_like(image, background_color, dtype=np.uint8)
        
        # 只对背景区域应用透明度混合
        for c in range(3):  # B, G, R channels
            result_image[:, :, c] = np.where(
                background_mask,
                (image[:, :, c] * (1 - alpha) + background_layer[:, :, c] * alpha).astype(np.uint8),
                image[:, :, c]
            )
        
        return result_image

# ========== 使用示例 ==========
if __name__ == "__main__":
    model_dir = "./models--mattmdjaga--segformer_b2_clothes/snapshots/61046"
    image_path = "data/sample_images/2.jpg"
    output_dir = "outputs/seg_clothes/image-2/"

    segmenter = ClothesSegmenter(model_dir=model_dir)
    
    # # 获取分割结果
    # pred_seg = segmenter.segment(image_path)
    results = []
    pred_seg = segmenter.segment_with_yolo(image_path, results)

    # 生成彩色掩码
    color_mask = segmenter.generate_color_mask(pred_seg)

    # 保存彩色掩码图
    import os
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "color_mask.png"), color_mask)

    print("Color mask saved successfully.")