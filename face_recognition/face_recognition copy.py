import os, sys, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import torch
from datetime import datetime
from tqdm import tqdm
from core.dbface import DBFaceDetector
from core.database import FaceDatabase
from core.face_extractor import FaceFeatureExtractor
from utils.alignment import FaceAlignment
from utils.visualization import VisualizationUtils
import cv2
import numpy as np

class FaceRecognitionSystem:
    def __init__(self, detect_path=None, extractor_path=None, device=None):
        # 如果没有指定设备，则使用默认设置
        if device is None:
            device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
            
        self.device = device
        self.detector = DBFaceDetector(detect_path, device=device)
        self.extractor = FaceFeatureExtractor(extractor_path, device=device)
        self.database = FaceDatabase()
        self.face_aligner = FaceAlignment()
        self.visualizer = VisualizationUtils()
        
    def _align_cropped_face(self, image):
        """
        对已裁剪的人脸图像进行对齐
        Args:
            image: 人脸图像 (1024x1024)
        Returns:
            aligned_face: 对齐后的人脸图像
        """
        try:
            # 对于已裁剪的人脸图像，我们需要先检测关键点再对齐
            # 使用一个简化的方法来估计关键点位置
            h, w = image.shape[:2]
            
            # 假设标准人脸关键点的大致位置（相对于1024x1024图像）
            # 这些是经验数值，可根据实际情况调整
            landmarks = [
                [0.341916, 0.461574],  # 左眼
                [0.656533, 0.459833],  # 右眼
                [0.500225, 0.640505],  # 鼻子
                [0.370975, 0.785691],  # 左嘴角
                [0.631516, 0.783486]   # 右嘴角
            ]
            
            # 使用提取器的对齐功能
            aligned_face = self.extractor.align_face(image, landmarks)
            return aligned_face
        except Exception as e:
            print(f"Error aligning cropped face: {str(e)}")
            return image  # 返回原始图像作为后备方案
    
    def build_face_database(self, image_folder, label_file=None, first_run=False):
        """
        建立人脸库
        Args:
            image_folder: 包含人脸图片的文件夹
            label_file: 可选的标签文件（person_id, person_name, image_path）
        """
        if label_file:
            # 从标签文件读取
            pass
        else:
            # 从文件夹结构推断：person_name/image.jpg
            total_persons = 0
            successful_persons = 0
            
            person_dirs = [d for d in os.listdir(image_folder) 
                          if os.path.isdir(os.path.join(image_folder, d))]
            
            print(f"Found {len(person_dirs)} persons to process")
            
            for person_name in person_dirs:
                person_folder = os.path.join(image_folder, person_name)
                
                print(f"\n{'='*50}")
                print(f"Processing person: {person_name}")
                print(f"{'='*50}")
                
                total_persons += 1
                person_id = person_name.lower().replace(" ", "_")
                
                image_files = [f for f in os.listdir(person_folder) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                print(f"Found {len(image_files)} images for {person_name}")
                
                processed_count = 0
                success_count = 0
                
                for img_file in tqdm(image_files, desc=f"Processing {person_name}", leave=False):
                    img_path = os.path.join(person_folder, img_file)
                    processed_count += 1
                    
                    # 处理已裁剪的人脸图片
                    try:
                        image = cv2.imread(img_path)
                        if image is None:
                            print(f"  Warning: Could not read image {img_path}")
                            continue
                            
                        # 对于已经是1024*1024的人脸图片，进行对齐处理
                        h, w = image.shape[:2]
                        if h == 160 and w == 160:
                            # 对已裁剪的人脸进行对齐
                            # aligned_face = self._align_cropped_face(image)
                            
                            # 提取特征
                            embeddings = self.extractor.extract_features([image])
                            if embeddings.shape[0] == 0:
                                print(f"  Warning: Failed to extract features from {img_path}")
                                continue
                                
                            embedding = embeddings[0]
                            embedding = self.extractor.l2_normalize([embedding])[0]
                            
                            # 存储到数据库
                            if first_run:
                                self.database.create_index()
                            self.database.add_face(
                                person_id=person_id,
                                person_name=person_name,
                                embedding=embedding,
                                image_path=img_path
                            )
                            success_count += 1
                            print(f"  ✓ Added face to database: {img_path}")
                        else:
                            # 对于非标准尺寸的图片，使用原来的检测流程
                            # 使用DBFace内置的检测方法，包括面积过滤和NMS
                            face_rects, processed_image, ratio = self.detector.detect_image(
                                image, pixel_threshold=100
                            )
                            
                            if len(face_rects) > 0:
                                # 取最大的人脸（按面积）
                                largest_face_idx = 0
                                largest_area = 0
                                for i, (x, y, w, h) in enumerate(face_rects):
                                    area = w * h
                                    if area > largest_area:
                                        largest_area = area
                                        largest_face_idx = i
                                
                                # 获取最大人脸的坐标
                                x, y, w, h = face_rects[largest_face_idx]
                                
                                # 调整坐标到原始图像尺寸
                                x_orig = int(x / ratio)
                                y_orig = int(y / ratio)
                                w_orig = int(w / ratio)
                                h_orig = int(h / ratio)
                                
                                # 提取人脸区域
                                face_image = image[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
                                
                                if face_image is None or face_image.size == 0:
                                    print(f"  Warning: Could not extract face from {img_path}")
                                    continue
                                
                                # 使用关键点进行对齐
                                faces = self.detector._detect_single(image)
                                if len(faces) > 0:
                                    # 过滤面积过小的人脸
                                    filtered_faces = [face for face in faces 
                                                    if face.width * face.height > 300]
                                    if len(filtered_faces) > 0:
                                        # 取最大人脸
                                        face = max(filtered_faces, key=lambda f: f.width * f.height)
                                        
                                        # 对齐人脸
                                        aligned_face = self.extractor.align_face(image, face.landmark)
                                        
                                        # 检查对齐是否成功
                                        if aligned_face is None:
                                            print(f"  Warning: Failed to align face in {img_path}")
                                            continue
                                        
                                        # 提取特征
                                        embeddings = self.extractor.extract_features([aligned_face])
                                        if embeddings.shape[0] == 0:
                                            print(f"  Warning: Failed to extract features from {img_path}")
                                            continue
                                            
                                        embedding = embeddings[0]
                                        embedding = self.extractor.l2_normalize([embedding])[0]
                                        
                                        # 存储到数据库
                                        self.database.add_face(
                                            person_id=person_id,
                                            person_name=person_name,
                                            embedding=embedding,
                                            image_path=img_path
                                        )
                                        success_count += 1
                                        print(f"  ✓ Added face to database: {img_path}")
                            else:
                                print(f"  ✗ No faces detected in {img_path}")
                            
                    except Exception as e:
                        print(f"  Error processing {img_path}: {str(e)}")
                        continue
                
                # Print summary for this person
                print(f"\nSummary for {person_name}:")
                print(f"  Total images: {len(image_files)}")
                print(f"  Processed: {processed_count}")
                print(f"  Successfully added: {success_count}")
                print(f"  Success rate: {success_count/len(image_files)*100:.1f}%")
                
                if success_count > 0:
                    successful_persons += 1
                    print(f"  Status: ✓ SUCCESS")
                else:
                    print(f"  Status: ✗ FAILED (no faces added)")
                
                print(f"{'='*50}\n")
            
            # Print overall summary
            print(f"\n{'#'*60}")
            print(f"BUILD DATABASE COMPLETE")
            print(f"{'#'*60}")
            print(f"Total persons processed: {total_persons}")
            print(f"Successful persons: {successful_persons}")
            print(f"Failed persons: {total_persons - successful_persons}")
            print(f"Overall success rate: {successful_persons/total_persons*100:.1f}%")
            print(f"{'#'*60}\n")
    
    def recognize_face(self, image_path, known_threshold=0.8, unknown_threshold=0.3, 
                   iou_threshold=0.4, min_face_size=20, debug=False):
        """
        人脸识别函数
        Args:
            image_path: 输入图片路径
            known_threshold: 已知人脸的匹配阈值（降低到0.75）
            unknown_threshold: 未知人脸的匹配阈值（降低到0.5）
            iou_threshold: 重复人脸检测的IOU阈值
            min_face_size: 最小人脸尺寸
            debug: 调试模式，显示更多信息
        Returns:
            list of recognition results
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        max_dimension = max(h, w)
        if max_dimension > 4096:
            scale_factor = 4096 / max_dimension
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            h, w = new_height, new_width

        print(f"Processing image: {image_path} ({w}x{h})")
        
        # 步骤1: 检测人脸并进行质量过滤
        try:
            face_objects = self.detector._detect_single(image)
        except Exception as e:
            if debug:
                print(f"Error detecting faces: {e}")
            face_objects = []
        
        # 初始过滤
        filtered_faces = []
        for face_obj in face_objects:
            # 过滤太小的人脸
            if face_obj.width < min_face_size or face_obj.height < min_face_size:
                if debug:
                    print(f"  Skip: Too small {face_obj.width}x{face_obj.height}")
                continue
                
            # 过滤低质量人脸（降低置信度阈值到0.1）
            if face_obj.score < 0.1:
                if debug:
                    print(f"  Skip: Low confidence {face_obj.score}")
                continue
                
            # 计算人脸面积和宽高比
            area = face_obj.width * face_obj.height
            aspect_ratio = face_obj.width / face_obj.height
            
            # 放宽宽高比限制
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                if debug:
                    print(f"  Skip: Bad aspect ratio {aspect_ratio}")
                continue
                
            filtered_faces.append(face_obj)
        
        print(f"Detected {len(face_objects)} faces, after filtering: {len(filtered_faces)}")
        
        # 步骤2: 处理每个人脸
        raw_results = []
        for i, face_obj in enumerate(filtered_faces):
            try:
                # 获取边界框坐标并确保在图像范围内
                x1 = max(0, int(face_obj.x))
                y1 = max(0, int(face_obj.y))
                x2 = min(w, x1 + int(face_obj.width))
                y2 = min(h, y1 + int(face_obj.height))
                
                # 添加边界padding
                padding = int(min(face_obj.width, face_obj.height) * 0.1)
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)
                
                # 裁剪人脸区域
                face_image = image[y1_pad:y2_pad, x1_pad:x2_pad]
                if face_image.size == 0:
                    if debug:
                        print(f"  Face {i+1}: Empty image after cropping")
                    continue
                
                # 计算人脸质量分数（简化评估）
                face_quality = self._assess_face_quality_simple(face_image)
                
                # 调整人脸图像大小以适应特征提取器
                target_size = (160, 160)  # 假设特征提取器输入为112x112
                face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LINEAR)
                # face_resized = face_image
                
                # 颜色空间转换
                if len(face_resized.shape) == 2:
                    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
                elif face_resized.shape[2] == 4:
                    face_resized = face_resized[:, :, :3]
                
                # 人脸对齐（如果有关键点）
                
                # if hasattr(face_obj, 'landmark') and face_obj.landmark is not None:
                #     try:
                #         aligned_face = self.extractor.align_face(image, face_obj.landmark)
                #         if aligned_face is not None:
                #             face_resized = cv2.resize(aligned_face, target_size, interpolation=cv2.INTER_AREA)
                #     except Exception as e:
                #         if debug:
                #             print(f"  Face {i+1}: Alignment failed: {e}")
                
                # 提取特征
                try:
                    embeddings = self.extractor.extract_features([face_resized])
                    if embeddings.shape[0] == 0:
                        if debug:
                            print(f"  Face {i+1}: No embeddings extracted")
                        continue
                    embedding = embeddings[0]
                except Exception as e:
                    if debug:
                        print(f"  Face {i+1}: Feature extraction error: {e}")
                    continue
                
                # L2归一化
                embedding = self.extractor.l2_normalize([embedding])[0]
                
                # 搜索匹配
                matches = self.database.search_face(embedding, top_k=5, threshold=unknown_threshold)
                
                if debug and matches:
                    print(f"  Face {i+1}: Found {len(matches)} matches")
                    for match in matches[:3]:
                        print(f"    Match: {match['person_name']} (Score: {match['score']:.3f})")
                
                # 根据人脸质量调整匹配分数（简化校准）
                calibrated_matches = []
                for match in matches:
                    raw_score = match['score']
                    
                    # 简化校准：不再大幅降低分数
                    if raw_score > 0.8:
                        calibrated = raw_score * 0.95  # 轻微降低
                    elif raw_score > 0.6:
                        calibrated = raw_score * 0.9
                    else:
                        calibrated = raw_score * 0.85
                    
                    # 轻微应用人脸质量因子
                    calibrated *= (0.8 + 0.2 * face_quality)
                    calibrated = min(1.0, max(0.0, calibrated))
                    
                    match_copy = match.copy()
                    # match_copy['score'] = calibrated
                    match_copy['score'] = raw_score
                    match_copy['raw_score'] = raw_score
                    calibrated_matches.append(match_copy)
                
                # 按校准后的分数排序
                calibrated_matches.sort(key=lambda x: x['score'], reverse=True)
                
                
                # 构建结果
                result = {
                    "id": i,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": face_obj.score,
                    "area": face_obj.width * face_obj.height,
                    "quality": face_quality,
                    "matches": calibrated_matches,
                    "embedding": embedding,
                    "face_image": face_resized
                }
                
                raw_results.append(result)
                
                if debug:
                    if calibrated_matches:
                        best_match = calibrated_matches[0]
                        print(f"  Face {i+1}: Best match: {best_match['person_name']} "
                            f"(Raw: {best_match['raw_score']:.3f}, Calibrated: {best_match['score']:.3f})")
                    else:
                        print(f"  Face {i+1}: No matches found")
                
            except Exception as e:
                if debug:
                    print(f"Error processing face {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if not raw_results:
            print("No valid faces found after processing")
            return []
        
        # 步骤3: 使用NMS去除重复检测
        unique_results = self._apply_nms(raw_results, iou_threshold)
        
        # 步骤4: 确定最终识别结果
        final_results = []
        for i, result in enumerate(unique_results):
            if result['matches']:
                best_match = result['matches'][0]
                print(f"DEBUG: Face {i+1} best match score: {best_match['score']}, threshold: {known_threshold}")
                
                # 简化动态阈值：只使用基本阈值
                dynamic_threshold = known_threshold
                
                if best_match['score'] >= dynamic_threshold:
                    identified_as = best_match['person_name']
                    confidence = best_match['score']
                    status = 'identified'
                    is_known = True
                else:
                    # 检查是否有其他候选匹配接近阈值
                    potential_matches = [m for m in result['matches'] 
                                    if m['score'] >= dynamic_threshold * 0.8]  # 降低到0.8倍阈值
                    print(f"DEBUG: Face {i+1} potential matches: {len(potential_matches)}")
                    if potential_matches:
                        identified_as = f"可能: {potential_matches[0]['person_name']}"
                        confidence = potential_matches[0]['score']
                        status = 'possible'
                        is_known = True
                    else:
                        identified_as = '未知'
                        confidence = best_match['score'] if best_match['score'] > 0 else 0.0
                        status = 'unknown'
                        is_known = False
            else:
                print(f"DEBUG: Face {i+1} has no matches")
                identified_as = '未知'
                confidence = 0.0
                status = 'no_match'
                is_known = False
            
            # 添加最终识别信息
            result.update({
                "identified_as": identified_as,
                "identification_confidence": confidence,
                "status": status,
                "is_known": is_known,
                "final_ranking": i + 1
            })
            final_results.append(result)
        
        # 按识别置信度排序
        final_results.sort(key=lambda x: x['identification_confidence'], reverse=True)
        # 在原图上绘制人脸框和标签
        annotated_image = self.visualizer.draw_faces_on_image(image, final_results)
        # 输出统计信息
        print(f"\n=== 识别结果 ===")
        print(f"检测到人脸: {len(final_results)}")
        
        identified = [r for r in final_results if r['is_known']]
        unknown = [r for r in final_results if not r['is_known']]
        
        print(f"已知人脸: {len(identified)}, 未知人脸: {len(unknown)}")
        
        for i, result in enumerate(final_results):
            status_icon = "✓" if result['is_known'] else "?"
            status_text = "已知" if result['is_known'] else "未知"
            
            if result['is_known']:
                print(f"{status_icon} 人脸 {result['final_ranking']}: "
                    f"{result['identified_as']} (置信度: {result['identification_confidence']:.3f}, "
                    f"质量: {result['quality']:.2f}, 面积: {result['area']:.0f})")
                
                # 显示前3个匹配（如果存在）
                if result['matches']:
                    for j, match in enumerate(result['matches'][:3]):
                        match_type = "最佳" if j == 0 else f"备选{j}"
                        print(f"    {match_type}: {match['person_name']} "
                            f"(分数: {match['score']:.3f}, 原始: {match.get('raw_score', 0):.3f})")
            else:
                if result['matches']:
                    best_match = result['matches'][0]
                    print(f"{status_icon} 人脸 {result['final_ranking']}: {status_text} "
                        f"(最佳匹配: {best_match['person_name']}, 分数: {best_match['score']:.3f}, "
                        f"质量: {result['quality']:.2f}, 面积: {result['area']:.0f})")
                else:
                    print(f"{status_icon} 人脸 {result['final_ranking']}: {status_text} "
                        f"(质量: {result['quality']:.2f}, 面积: {result['area']:.0f})")
        
        return final_results, annotated_image

    def _assess_face_quality_simple(self, face_image):
        """
        简化版人脸质量评估
        Returns:
            float: 质量分数 (0.0-1.0)
        """
        if face_image is None or face_image.size == 0:
            return 0.5
        
        try:
            # 转换为灰度图
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # 1. 亮度评估（更宽松）
            brightness = np.mean(gray)
            if brightness < 30 or brightness > 225:
                brightness_score = 0.3
            elif brightness < 50 or brightness > 200:
                brightness_score = 0.6
            elif brightness < 70 or brightness > 180:
                brightness_score = 0.8
            else:
                brightness_score = 1.0
            
            # 2. 简单清晰度评估
            try:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 50:
                    sharpness_score = 0.3
                elif laplacian_var < 100:
                    sharpness_score = 0.6
                elif laplacian_var < 200:
                    sharpness_score = 0.8
                else:
                    sharpness_score = 1.0
            except:
                sharpness_score = 0.7
            
            # 综合质量分数（简单平均）
            quality_score = (brightness_score + sharpness_score) / 2
            
            return min(1.0, max(0.3, quality_score))  # 确保最低0.3
        
        except Exception as e:
            print(f"Face quality assessment error: {e}")
            return 0.5

    def _apply_nms(self, results, iou_threshold):
        """
        应用非极大值抑制去除重复检测
        """
        if len(results) <= 1:
            return results
        
        # 按置信度排序
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        unique_results = []
        used_indices = set()
        
        for i in range(len(sorted_results)):
            if i in used_indices:
                continue
            
            current = sorted_results[i]
            unique_results.append(current)
            used_indices.add(i)
            
            # 比较与后续人脸的相似度
            for j in range(i + 1, len(sorted_results)):
                if j in used_indices:
                    continue
                
                # 计算IOU
                iou = self._calculate_iou(current['bbox'], sorted_results[j]['bbox'])
                
                # 如果IOU很高，则认为是同一个人的重复检测
                if iou > iou_threshold:
                    used_indices.add(j)
                    # 可选：合并特征或取最高分
                    # 这里我们简单跳过重复检测
        
        print(f"NMS: {len(results)} -> {len(unique_results)} faces")
        return unique_results

    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IOU
        """
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        if box1_area + box2_area - inter_area == 0:
            return 0.0
        
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou
    
    def test_embedding_accuracy(self, image_folder, threshold=0.8):
        """
        测试embedding和人像检索的准确率
        使用建立数据库的图片直接输入测试，验证系统能否正确识别
        
        Args:
            image_folder: 包含人脸图片的文件夹 (如 face_index-160)
            threshold: 识别正确的阈值，默认0.8
            
        Returns:
            dict: 准确率统计信息
        """
        print("=" * 60)
        print("开始测试embedding和人像检索准确率")
        print("=" * 60)
        
        # 统计变量
        total_tests = 0
        correct_identifications = 0
        failed_to_detect = 0
        failed_to_recognize = 0
        
        # 获取所有人物目录
        person_dirs = [d for d in os.listdir(image_folder) 
                      if os.path.isdir(os.path.join(image_folder, d))]
        
        print(f"找到 {len(person_dirs)} 个人物目录进行测试")
        
        # 遍历每个人物目录
        for person_name in person_dirs:
            person_folder = os.path.join(image_folder, person_name)
            image_files = [f for f in os.listdir(person_folder) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"\n测试人物 '{person_name}' 的 {len(image_files)} 张图片...")
            
            # 测试每张图片
            for img_file in image_files:
                img_path = os.path.join(person_folder, img_file)
                total_tests += 1
                
                try:
                    print(f"  测试图片: {img_file}")
                    
                    # 直接读取图片并提取特征（因为这些是已经裁剪好的160x160人脸图片）
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"    警告: 无法读取图片 {img_path}")
                        failed_to_detect += 1
                        continue
                    
                    h, w = image.shape[:2]
                    if h == 160 and w == 160:
                        # 对于已经是160x160的人脸图片，直接提取特征
                        embeddings = self.extractor.extract_features([image])
                        if embeddings.shape[0] == 0:
                            print(f"    警告: 无法从 {img_path} 提取特征")
                            failed_to_detect += 1
                            continue
                            
                        embedding = embeddings[0]
                        embedding = self.extractor.l2_normalize([embedding])[0]
                        
                        # 搜索匹配
                        matches = self.database.search_face(embedding, top_k=3, threshold=0.8)
                        
                        if not matches:
                            print(f"    错误: 未能找到任何匹配项")
                            failed_to_recognize += 1
                            continue
                        
                        best_match = matches[0]
                        predicted_person = best_match['person_name']
                        score = best_match['score']
                        
                        # 检查是否正确识别
                        if predicted_person.lower().replace(" ", "_") == person_name.lower().replace(" ", "_"):
                            if score >= threshold:
                                correct_identifications += 1
                                print(f"    ✓ 正确识别: {predicted_person} (置信度: {score:.3f})")
                            else:
                                print(f"    ? 低置信度匹配: {predicted_person} (置信度: {score:.3f})")
                        else:
                            print(f"    ✗ 错误识别: 预期'{person_name}', 实际'{predicted_person}' (置信度: {score:.3f})")
                    else:
                        print(f"    跳过: 图片尺寸不正确 ({h}x{w})，应为160x160")
                        failed_to_detect += 1
                        
                except Exception as e:
                    print(f"    错误: 处理 {img_path} 时发生异常: {str(e)}")
                    failed_to_detect += 1
                    continue
        
        # 计算准确率
        success_rate = correct_identifications / total_tests if total_tests > 0 else 0
        detection_success_rate = (total_tests - failed_to_detect) / total_tests if total_tests > 0 else 0
        
        # 打印结果
        print("\n" + "=" * 60)
        print("测试完成 - 结果统计")
        print("=" * 60)
        print(f"总测试数量: {total_tests}")
        print(f"正确识别数量: {correct_identifications}")
        print(f"检测失败数量: {failed_to_detect}")
        print(f"识别失败数量: {failed_to_recognize}")
        print(f"检测成功率: {detection_success_rate:.2%}")
        print(f"识别准确率: {success_rate:.2%}")
        print("=" * 60)
        
        # 判断系统是否正常工作
        if success_rate >= 0.99:
            print("🎉 系统工作正常！准确率达到预期标准 (>= 99%)")
        elif success_rate >= 0.95:
            print("✅ 系统基本正常，准确率较高 (>= 95%)")
        else:
            print("⚠️  系统可能存在一些问题，准确率低于预期 (< 95%)")
        
        print("=" * 60)
        
        return {
            "total_tests": total_tests,
            "correct_identifications": correct_identifications,
            "failed_to_detect": failed_to_detect,
            "failed_to_recognize": failed_to_recognize,
            "detection_success_rate": detection_success_rate,
            "recognition_accuracy": success_rate,
            "threshold_used": threshold
        }

    def recognize_random_samples(self, input_folder, output_folder, fraction=1/3, known_threshold=0.8):
        """
        随机抽取文件夹中的图片进行人脸检测和识别，并按识别身份保存结果
        
        Args:
            input_folder: 包含待识别人脸图片的文件夹
            output_folder: 保存识别结果的文件夹
            fraction: 抽取比例，默认为1/3
            known_threshold: 识别阈值
            
        Returns:
            dict: 处理统计信息
        """
        print("=" * 60)
        print("开始随机抽样识别人脸")
        print("=" * 60)
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        all_images = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith(image_extensions)]
        
        if not all_images:
            print("警告: 输入文件夹中没有找到图片文件")
            return {}
        
        # 随机抽取指定比例的图片
        sample_size = max(1, int(len(all_images) * fraction))
        sampled_images = random.sample(all_images, sample_size)
        
        print(f"总共找到 {len(all_images)} 张图片")
        print(f"随机抽取 {sample_size} 张图片进行识别")
        
        # 统计变量
        total_processed = 0
        successful_recognitions = 0
        failed_recognitions = 0
        total_faces_detected = 0
        identity_counts = {}  # 记录每个身份被识别的次数
        
        # 处理每张抽样的图片
        for img_file in tqdm(sampled_images, desc="处理图片"):
            img_path = os.path.join(input_folder, img_file)
            total_processed += 1
            
            try:
                # 进行人脸识别
                results, annotated_image = self.recognize_face(
                    img_path, 
                    known_threshold=known_threshold,
                    unknown_threshold=0.3
                )
                
                # 统计检测到的人脸数量
                total_faces_detected += len(results)
                
                # 处理每个识别结果
                for i, result in enumerate(results):
                    if result['is_known']:
                        identity = result['identified_as']
                        successful_recognitions += 1
                        
                        # 更新身份计数
                        if identity not in identity_counts:
                            identity_counts[identity] = 0
                        identity_counts[identity] += 1
                        
                        # 生成保存文件名
                        if identity_counts[identity] == 1:
                            save_filename = f"{identity}.jpg"
                        else:
                            save_filename = f"{identity}_{identity_counts[identity]}.jpg"
                        
                        save_path = os.path.join(output_folder, save_filename)
                        
                        # 保存识别出的人脸图像
                        cv2.imwrite(save_path, result['face_image'])
                        print(f"  保存识别结果: {save_filename}")
                    else:
                        failed_recognitions += 1
                
                # 保存带标注的原图
                annotated_save_path = os.path.join(output_folder, f"annotated_{img_file}")
                cv2.imwrite(annotated_save_path, annotated_image)
                
            except Exception as e:
                print(f"处理图片 {img_file} 时出错: {str(e)}")
                failed_recognitions += 1
                continue
        
        # 计算统计信息
        success_rate = successful_recognitions / max(total_faces_detected, 1)
        
        # 打印结果
        print("\n" + "=" * 60)
        print("随机抽样识别完成 - 结果统计")
        print("=" * 60)
        print(f"处理图片数量: {total_processed}")
        print(f"检测到的人脸总数: {total_faces_detected}")
        print(f"成功识别的人脸数: {successful_recognitions}")
        print(f"未识别的人脸数: {failed_recognitions}")
        print(f"人脸识别成功率: {success_rate:.2%}")
        print("\n各身份识别统计:")
        for identity, count in identity_counts.items():
            print(f"  {identity}: {count} 次")
        print("=" * 60)
        
        return {
            "total_processed": total_processed,
            "total_faces_detected": total_faces_detected,
            "successful_recognitions": successful_recognitions,
            "failed_recognitions": failed_recognitions,
            "success_rate": success_rate,
            "identity_counts": identity_counts
        }

if __name__ == '__main__':
    dbface_model_dir = '/home/huachenghao/codes/Theatrical_photo_sorting/face_recognition/dbface/model/dbface.pth'
    facenet_model_dir = '/home/huachenghao/codes/Theatrical_photo_sorting/face_recognition/facenet/model_data/facenet_inception_resnetv1.pth'
    image_path = '/home/huachenghao/codes/Theatrical_photo_sorting/test_images/4.jpg' 
    
    recognizer = FaceRecognitionSystem(dbface_model_dir, facenet_model_dir)
    # 建立人脸索引（第一次），批量添加人脸
    # recognizer.build_face_database('/home/huachenghao/codes/face_index-160', first_run=True)
    # recognizer.build_face_database('/home/huachenghao/codes/face_index-160')
    
    # 测试embedding和人像检索准确率
    # test_results = recognizer.test_embedding_accuracy('/home/huachenghao/codes/face_index-160')

    # 随机抽样识别人脸
    recognizer.recognize_random_samples(
        input_folder='/home/huachenghao/codes/NCPA_test-images/话剧《样式雷》/【原始】20160609戏剧场-话剧《样式雷》-摄影凌风',
        output_folder='./out/test/',
        fraction=1/2,
        known_threshold=0.8
    )
    # results, annotated_image = recognizer.recognize_face(image_path, known_threshold=0.8, unknown_threshold=0.3)
    # print(f"Found {len(results)} faces:")

    # for i, result in enumerate(results):
    #     cv2.imwrite(f'./out/result_{i+1}.jpg', result['face_image'])
    #     # print(f"Face {i+1}:  Area={result['area']}, Matches={len(result['matches'])}")
    #     if result['is_known']:
    #         print(f"Face {i+1}识别为: {result['identified_as']} (置信度: {result['identification_confidence']:.3f})")
    #     else:
    #         print(f"Face {i+1}未知人脸 (质量: {result['quality']:.2f})")