import cv2, time
import numpy as np
from ultralytics import YOLO
import os, sys
sys.path.append(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class ShotTypeClassifier:
    def __init__(self,  pose_model_path="yolo11l-pose.pt"):
        """
        初始化景别分类器
        
        Args:
            pose_model_path (str): 姿态估计模型路径
        """
        # 初始化模型
        self.yolo_pose_model = YOLO(pose_model_path)
        
        # COCO关键点定义 (17个关键点)
        self.COCO_KEYPOINTS = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]


    def draw_pose_result(self, image, results, save_path, offset=(0, 0)):
        """
        绘制并保存姿态估计结果（仅处理第一个人）
        
        Args:
            image: 原始图像
            results: 姿态估计结果
            save_path: 保存路径
            offset: 坐标偏移（用于裁剪图像）
        """
        # 创建姿态估计结果图像
        # image = cv2.imread(img_path)
        pose_image = image.copy()
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                
                # 仅处理第一个人的姿态估计结果
                if len(keypoints) > 0:
                    kps = keypoints[0]  # 选择第一个人的关键点
                    confs = confidences[0] if confidences is not None and len(confidences) > 0 else None
                    
                    # Apply offset to coordinates (for cropped images)
                    offset_kps = kps.copy()
                    offset_kps[:, 0] += offset[0]  # Add x offset
                    offset_kps[:, 1] += offset[1]  # Add y offset
                    
                    # 绘制关键点
                    for j, kp in enumerate(offset_kps):
                        x, y = int(kp[0]), int(kp[1])
                        # 检查置信度
                        conf = confs[j] if confs is not None and j < len(confs) else 1.0
                        if conf > 0.3:  # 降低阈值以显示更多点
                            cv2.circle(pose_image, (x, y), 5, (0, 255, 0), -1)
                            # 添加关键点索引标签
                            cv2.putText(pose_image, str(j), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # 绘制骨架连接线（COCO数据集的标准连接）
                    skeleton_connections = [
                        (0, 1), (0, 2), (1, 3), (2, 4),  # 面部连接
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂连接
                        (5, 11), (6, 12), (11, 12),  # 身体连接
                        (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部连接
                    ]
                    
                    for start_idx, end_idx in skeleton_connections:
                        if (start_idx < len(offset_kps) and end_idx < len(offset_kps) and
                            confs is not None and 
                            start_idx < len(confs) and end_idx < len(confs) and
                            confs[start_idx] > 0.3 and 
                            confs[end_idx] > 0.3):
                            
                            start_point = tuple(map(int, offset_kps[start_idx]))
                            end_point = tuple(map(int, offset_kps[end_idx]))
                            cv2.line(pose_image, start_point, end_point, (255, 0, 0), 2)
                    
                    # 处理完第一个人后就退出循环，不再处理其他人
                    break
        
        # 保存姿态估计结果
        cv2.imwrite(save_path, pose_image)
        print(f"Pose estimation result saved to: {save_path}")
        return pose_image

    def _get_unique_filename(self, directory, base_filename):
        """
        生成唯一的文件名
        
        Args:
            directory: 目录路径
            base_filename: 基础文件名
            
        Returns:
            完整的唯一文件路径
        """
        os.makedirs(directory, exist_ok=True)
        
        # 完整的文件路径
        full_path = os.path.join(directory, base_filename)
        
        # 如果文件已存在，则添加数字后缀
        i = 1
        while os.path.exists(full_path):
            name, ext = os.path.splitext(base_filename)
            full_path = os.path.join(directory, f"{name}_{i}{ext}")
            i += 1
            
        return full_path

    def classify_shot_type(self, image, img_name, area_ratio, main_bbox=None):
        """
        对图像进行景别分类
        
        Args:
            image: 输入图像
            img_path: 图像路径（用于保存结果和日志）
            
        Returns:
            dict: 包含景别分类结果的字典
        """
        start_time = time.time()

        # image = cv2.imread(img_path)
        h, w = image.shape[:2]
        img_area = w * h

        
        # 在主角色区域内进行姿态估计
        visible_kps = []
        keypoints_confidence = 0.0
        
        if main_bbox is not None:
            # 扩展边界框以包含完整的姿态信息
            x1, y1, x2, y2 = map(int, main_bbox)
            # 添加边距确保包含完整的身体部位
            margin = int((x2 - x1 + y2 - y1) / 10)  # 10%的边距
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            # 裁剪主角色区域
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size > 0:  # 确保裁剪区域有效
                # 在裁剪区域进行姿态估计
                pose_results = self.yolo_pose_model(cropped_image, verbose=False)
                
                # 保存姿态估计结果图
                # if img_path:
                #     pose_output_path = self._get_unique_filename("./out", "output_pose.png")
                #     self.draw_pose_result(image, pose_results, pose_output_path, offset=(x1, y1))
                
                # 处理姿态估计结果
                if len(pose_results) > 0:
                    result = pose_results[-1]  # 使用最后一个结果（通常是最显著的检测）
                    if result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                        
                        # 只处理第一个人的关键点
                        if len(keypoints) > 0:
                            kps = keypoints[0]  # 第一个人的关键点
                            
                            if confidences is not None and len(confidences) > 0:
                                confs = confidences[0]  # 第一个人的置信度
                                
                                # 提取可见的关键点
                                for j in range(min(len(kps), len(confs), 17)):  # 确保不超过17个关键点
                                    # 获取相对于裁剪图像的坐标
                                    rel_x, rel_y = kps[j]
                                    # 转换为原始图像中的绝对坐标
                                    abs_x = rel_x + x1
                                    abs_y = rel_y + y1
                                    
                                    # 检查点是否在原始图像范围内并且置信度足够
                                    if (0 <= abs_x < w and 0 <= abs_y < h and 
                                        confs[j] > 0.3):  # 使用较低的阈值
                                        visible_kps.append(j)
                        
                        # 获取置信度
                        if result.boxes is not None and len(result.boxes) > 0:
                            keypoints_confidence = result.boxes.conf.cpu().numpy()[0] if result.boxes.conf is not None else 0.0
        else:
            # 如果没有边界框信息，对整张图片进行姿态估计（备选方案）
            pose_results = self.yolo_pose_model(image, verbose=False)
            
            # 保存姿态估计结果图
            # if img_path:
            #     pose_output_path = self._get_unique_filename("./out", "output_pose.png")
            #     self.draw_pose_result(image, pose_results, pose_output_path)
            
            # 处理姿态估计结果
            if len(pose_results) > 0:
                result = pose_results[-1]  # 使用最后一个结果
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                    
                    # 只处理第一个人的关键点
                    if len(keypoints) > 0:
                        kps = keypoints[0]  # 第一个人的关键点
                        
                        if confidences is not None and len(confidences) > 0:
                            confs = confidences[0]  # 第一个人的置信度
                            
                            # 提取可见的关键点
                            for j in range(min(len(kps), len(confs), 17)):  # 确保不超过17个关键点
                                # 检查置信度
                                if confs[j] > 0.3:  # 使用较低的阈值
                                    visible_kps.append(j)
                    
                    # 获取置信度
                    if result.boxes is not None and len(result.boxes) > 0:
                        keypoints_confidence = result.boxes.conf.cpu().numpy()[0] if result.boxes.conf is not None else 0.0
                        
        # 判断是否包含脚部、髋部等部位以确定景别类型 (基于正确的关键点索引)
        has_ankle = any(kp in [15, 16] for kp in visible_kps)  # 脚踝关键点 (left_ankle, right_ankle)
        has_hip = any(kp in [11, 12] for kp in visible_kps)    # 髋部关键点 (left_hip, right_hip)
        has_shoulder = any(kp in [5, 6] for kp in visible_kps) # 肩膀关键点 (left_shoulder, right_shoulder)
        has_face = any(kp in [0, 1, 2, 3, 4] for kp in visible_kps)  # 面部关键点 (nose, eyes, ears)

        # 根据面积比例和关键点分布判断景别
        if not has_shoulder and has_face:
            shot = "Extreme Close-up"       #只有脸，则为特写
        elif has_shoulder and not has_hip:
            shot = "Close-up" if area_ratio < 0.1 else "Medium Close-up"   #肩部，没有髋部（半身），则为近、中近景
        elif has_hip and not has_ankle:
            shot = "Medium Shot"            #髋部，没有脚踝，则为中景
        elif has_ankle:
            shot = "Full Shot" if area_ratio > 0.2 else "Long Shot"         #脚踝，则为远、全景
        else:
            shot = "Medium Shot" if area_ratio < 0.3 else "Full Shot"        #其他情况，则为全、中景

        conf = min(0.95, keypoints_confidence + 0.1)  # 简单置信度融合
        
        # 记录日志
        result_dict = {
            "shot_type": shot,
            "confidence": round(conf, 2),
            "area_ratio": round(area_ratio, 3),
            "visible_keypoints_count": len(visible_kps),
            "visible_keypoints": list(set(visible_kps))  # 去重并转换为列表用于调试
        }
        end = time.time()
        # img_name = os.path.basename(img_path) if img_path else "unknown"
        log_result = (f" ✅ Shot Classification. Image: {img_name} | Result: {shot} "
                     f"(confidence: {round(conf, 2)}, area_ratio: {round(area_ratio, 3)}, "
                     f"visible_keypoints_count: {len(visible_kps)}, time: {end - start_time:.2f}s)")
        print(log_result)
        print(f"Visible keypoints indices: {list(set(visible_kps))}")  # 调试信息，去重显示

        # 可选：将日志写入文件
        # with open("shot_classification_log.txt", "a") as f:
        #     f.write(log_result + "\n")


        return result_dict, pose_results


# 示例使用
if __name__ == "__main__":
    classifier = ShotTypeClassifier()
    
    img_path = "/home/huachenghao/codes/NCPA_test-images/舞剧《马可·波罗》/【原始3】20141012歌剧院-舞剧《马可·波罗》B组演出-摄影凌风/20141012歌剧院-舞剧《马可·波罗》B演-前右起：苏鹏饰马可·波罗、李祎然饰中国公主- (23)-摄影凌风.JPG"
    save_path = "./out"
   

    result_dict, pose_results = classifier.classify_shot_type(image_path)
    # 绘制姿态估计结果（可选）
    pose_output_path = self._get_unique_filename(save_path, "output_pose.png")
    self.draw_pose_result(image_path, pose_results, pose_output_path)

    print("景别判断结果：", result_dict)