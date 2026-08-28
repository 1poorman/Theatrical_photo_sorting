import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO  

class PersonMaskCreator:
    def __init__(self, model_path='./yolo11l.pt', confidence=0.5, imgsz=640):
        self.model = YOLO(model_path)
        self.conf = confidence
        self.imgsz = imgsz

    def detect_persons_in_image(self, image):
        """检测单张图像中的人物，返回 YOLO 检测结果"""
        # image = cv2.imread(image_path)
        # if image is None:
        #     raise ValueError(f"无法读取图像: {image_path}")
        return self.model.predict(source=image, classes=[0], conf=self.conf, imgsz=self.imgsz)

    def batch_detect_persons(self, input_folder):
        """批量检测人物，返回 {image_path: results} 的字典"""
        input_path = Path(input_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(ext))
            image_paths.extend(input_path.glob(ext.upper()))
        
        detections = {}
        for img_path in image_paths:
            print(f"检测: {img_path.name}")
            detections[img_path] = self.detect_persons_in_image(str(img_path))
        return detections

    def generate_and_save_mask_from_results(self, image, results, output_path=None):
        """
        根据检测结果生成掩码，若 output_path 非 None 则保存。
        :return: 生成的 mask (numpy array), persons_detected (int)
        """
        # image = cv2.imread(image_path)
        # if image is None:
        #     raise ValueError(f"无法读取图像用于掩码生成: {image_path}")
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        persons_detected = 0

        for r in results:
            if r.boxes is not None:
                persons_detected += len(r.boxes)
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # 只有检测到人物且指定了输出路径时才保存
        if persons_detected > 0 and output_path is not None:
            cv2.imwrite(output_path, mask)
            print(f"✅ 掩码已保存: {output_path} | 人物数: {persons_detected}")
        elif persons_detected == 0:
            print("⚠️ 未检测到人物，跳过掩码生成")
        else:
            print(f"ℹ️ 检测到 {persons_detected} 个人物，但未指定 output_path，未保存")

        return mask

    def batch_generate_masks_from_existing_results(self, detections_dict, output_folder):
        """从已有的 {img_path: results} 字典生成掩码"""
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        for img_path, results in detections_dict.items():
            output_file = output_dir / f"mask_{img_path.stem}.png"
            self.generate_and_save_mask_from_results(str(img_path), results, str(output_file))