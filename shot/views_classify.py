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
        # 记录模型路径，供服务端缓存判断使用，避免每次请求重复加载模型
        self.model_path = pose_model_path
        
        # COCO关键点定义 (17个关键点)
        self.COCO_KEYPOINTS = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]


    def draw_pose_result(self, image, results, save_path, offset=(0, 0), person_idx=None):
        """
        绘制并保存姿态估计结果（默认绘制指定的人物）

        Args:
            image: 原始图像
            results: 姿态估计结果
            save_path: 保存路径
            offset: 坐标偏移（用于裁剪图像）
            person_idx: 要绘制的人物索引，None 时取第一个人
        """
        # 创建姿态估计结果图像
        # image = cv2.imread(img_path)
        pose_image = image.copy()

        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

                # 仅处理指定人物（默认第一个人）的姿态估计结果
                if len(keypoints) > 0:
                    idx = person_idx if person_idx is not None and 0 <= person_idx < len(keypoints) else 0
                    kps = keypoints[idx]  # 选择指定人物的关键点
                    confs = confidences[idx] if confidences is not None and len(confidences) > idx else None
                    
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

    @staticmethod
    def _compute_iou(box_a, box_b):
        """计算两个边界框 (x1, y1, x2, y2) 的 IoU"""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _select_main_person(self, pose_results, image_size, main_bbox=None):
        """
        从姿态估计结果中挑选画面中的"主要人物"。

        评分综合考虑：
        - 检测置信度
        - 人物框面积占整图比例（主角通常占比更大）
        - 画面中心度（舞台摄影主角多位于画面中心/视觉焦点处）
        - 与外部传入的主角色框 (main_bbox) 的 IoU 加成（与分割结果对齐）
        - 贴边惩罚（贴边人物可能是被裁切的路人/伴舞）

        Args:
            pose_results: YOLO 姿态估计结果列表
            image_size: (width, height)
            main_bbox: 可选的外部主角色框 (x1, y1, x2, y2)

        Returns:
            (main_person_idx, box_confidence)，未检测到人时返回 (None, 0.0)
        """
        if not pose_results or pose_results[0].keypoints is None \
                or pose_results[0].boxes is None or len(pose_results[0].boxes) == 0:
            return None, 0.0

        result = pose_results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        box_confs = result.boxes.conf.cpu().numpy()

        w, h = image_size
        img_area = w * h
        cx_img, cy_img = w / 2, h / 2
        edge_margin = 0.02  # 距画面边缘 2% 以内视为贴边

        best_idx, best_score = None, -1.0
        for i, (box, conf) in enumerate(zip(boxes, box_confs)):
            x1, y1, x2, y2 = box
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            area_ratio = (bw * bh) / img_area

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centrality = 1.0 - (abs(cx - cx_img) / (w / 2) + abs(cy - cy_img) / (h / 2)) / 2

            # 贴边惩罚：贴边人物可能只是路人或被裁切
            touching_border = (x1 < w * edge_margin or y1 < h * edge_margin
                               or x2 > w * (1 - edge_margin) or y2 > h * (1 - edge_margin))
            border_penalty = 0.7 if touching_border else 1.0

            score = (0.35 * min(float(conf), 1.0)
                     + 0.35 * min(area_ratio * 3.0, 1.0)   # 面积占比封顶，避免极端大框独裁
                     + 0.30 * max(centrality, 0.0))
            score *= border_penalty

            # 与分割得到的主角色框对齐时给予强加成
            if main_bbox is not None:
                iou = self._compute_iou((x1, y1, x2, y2), main_bbox)
                score += 0.5 * iou

            if score > best_score:
                best_score, best_idx = score, i

        return best_idx, float(box_confs[best_idx])

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

        
        # 直接在整图上进行姿态估计，避免裁剪导致的坐标换算误差和上下文丢失；
        # 再从所有检测中挑选画面主要人物
        pose_results = self.yolo_pose_model(image, verbose=False)

        main_idx, keypoints_confidence = self._select_main_person(pose_results, (w, h), main_bbox)

        # 提取主要人物的可见关键点（整图坐标系，无需偏移换算）
        visible_kps = []
        kps, confs = None, None
        if main_idx is not None:
            result = pose_results[0]
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            if main_idx < len(keypoints):
                kps = keypoints[main_idx]
                confs = confidences[main_idx] if confidences is not None and main_idx < len(confidences) else None

                for j in range(min(len(kps), 17)):
                    abs_x, abs_y = kps[j]
                    kp_conf = confs[j] if confs is not None and j < len(confs) else 1.0
                    if 0 <= abs_x < w and 0 <= abs_y < h and kp_conf > 0.3:
                        visible_kps.append(j)

        # 判断是否包含脚部、髋部等部位以确定景别类型 (基于正确的关键点索引)
        has_ankle = any(kp in [15, 16] for kp in visible_kps)  # 脚踝关键点 (left_ankle, right_ankle)
        has_hip = any(kp in [11, 12] for kp in visible_kps)    # 髋部关键点 (left_hip, right_hip)
        has_shoulder = any(kp in [5, 6] for kp in visible_kps) # 肩膀关键点 (left_shoulder, right_shoulder)
        has_face = any(kp in [0, 1, 2, 3, 4] for kp in visible_kps)  # 面部关键点 (nose, eyes, ears)

        # 脚踝贴近画面底边，说明脚部大概率被裁切出画，视为不可见，避免误判为全景/远景
        if has_ankle and kps is not None:
            ankle_ys = [kps[j][1] for j in (15, 16) if j in visible_kps]
            if ankle_ys and min(ankle_ys) > h * 0.95:
                has_ankle = False

        # 根据面积比例和主要人物的关键点分布判断景别
        if main_idx is None or not visible_kps:
            shot = "Unknown"                # 未检测到有效人物
        elif not has_shoulder and has_face:
            shot = "Extreme Close-up"       #只有脸，则为特写
        elif has_shoulder and not has_hip:
            shot = "Close-up" if area_ratio < 0.1 else "Medium Close-up"   #肩部，没有髋部（半身），则为近、中近景
        elif has_hip and not has_ankle:
            shot = "Medium Shot"            #髋部，没有脚踝，则为中景
        elif has_ankle:
            shot = "Full Shot" if area_ratio > 0.2 else "Long Shot"         #脚踝，则为远、全景
        else:
            shot = "Medium Shot" if area_ratio < 0.3 else "Full Shot"        #其他情况，则为全、中景

        conf = min(0.95, keypoints_confidence + 0.1) if main_idx is not None else 0.0  # 简单置信度融合

        main_bbox_out = None
        if main_idx is not None:
            main_bbox_out = [round(float(v), 1) for v in pose_results[0].boxes.xyxy.cpu().numpy()[main_idx]]

        # 记录日志
        result_dict = {
            "shot_type": shot,
            "confidence": round(conf, 2),
            "area_ratio": round(area_ratio, 3),
            "main_person_idx": main_idx,
            "main_person_bbox": main_bbox_out,
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

    img_path = "data/ncpa_test/舞剧《马可·波罗》/【原始3】20141012歌剧院-舞剧《马可·波罗》B组演出-摄影凌风/20141012歌剧院-舞剧《马可·波罗》B演-前右起：苏鹏饰马可·波罗、李祎然饰中国公主- (23)-摄影凌风.JPG"
    save_path = "./out"

    image = cv2.imread(img_path)
    if image is None:
        print(f"无法读取图像: {img_path}")
        sys.exit(1)

    result_dict, pose_results = classifier.classify_shot_type(
        image, os.path.basename(img_path), area_ratio=0.0, main_bbox=None)

    # 绘制主要人物的姿态估计结果（可选）
    pose_output_path = classifier._get_unique_filename(save_path, "output_pose.png")
    classifier.draw_pose_result(image, pose_results, pose_output_path,
                                person_idx=result_dict.get("main_person_idx"))

    print("景别判断结果：", result_dict)