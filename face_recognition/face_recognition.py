import os, sys, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from tqdm import tqdm
from scrfd.scrfd_det import SCRFDDetector
from core.database import FaceDatabase
from arcface.arcface_onnx import ArcFaceFeatureExtractor
from utils.assess import assess_face_quality_simple
from utils.visualization import VisualizationUtils
from tools.image_io import imread_reduced, list_images
from config.base import (
    FACE_IMAGE_MAX_SIDE, KNOWN_FACE_THRESHOLD, SEARCH_THRESHOLD, SEARCH_TOP_K,
    FACE_DETECT_MIN_SCORE, FACE_MIN_SIZE, FACE_ASPECT_RATIO_RANGE, FACE_NMS_IOU,
    FACE_DB_STANDARD_SIZE, FACE_DB_MIN_AREA,
)
import cv2

class FaceRecognitionSystem:
    def __init__(self, detect_path=None, extractor_path=None, device=None):
        # 如果没有指定设备，则使用默认设置（onnxruntime）
        # 可选值: 'cpu' / 'cuda' / 'tensorrt'（TensorRT 加速，FP16）
        if device is None:
            device = 'cpu'
            try:
                import onnxruntime
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    device = 'cuda'
            except ImportError:
                pass

        self.device = device
        self.detector = SCRFDDetector(detect_path, device=device)
        self.extractor = ArcFaceFeatureExtractor(extractor_path, device=device)
        self.database = FaceDatabase()
        self.visualizer = VisualizationUtils()

    def build_face_database(self, image_folder, label_file=None, first_run=False):
        """
        建立人脸库
        Args:
            image_folder: 包含人脸图片的文件夹（子目录按人物命名）
            label_file: 预留参数（当前仅支持目录结构推断，传入时忽略并警告）
            first_run: 预留参数（索引创建由 FaceDatabase 内部管理）
        """
        if label_file:
            print(f"Warning: label_file mode not implemented, ignored: {label_file}")

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

                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"  Warning: Could not read image {img_path}")
                        continue

                    # 已是标准尺寸人脸图（见 FACE_DB_STANDARD_SIZE）：直接对齐入库，无需检测
                    h, w = image.shape[:2]
                    if h == FACE_DB_STANDARD_SIZE and w == FACE_DB_STANDARD_SIZE:
                        aligned_face = self.extractor.align_face(image, None)
                    else:
                        # 非标准尺寸：检测并取最大人脸（单次检测，基于关键点对齐）
                        faces = self.detector._detect_single(image)
                        # 过滤面积过小的人脸
                        filtered = [f for f in faces if f.width * f.height > FACE_DB_MIN_AREA]
                        if not filtered:
                            print(f"  ✗ No faces detected in {img_path}")
                            continue
                        face = max(filtered, key=lambda f: f.width * f.height)
                        aligned_face = self.extractor.align_face(image, face.landmark)
                        if aligned_face is None:
                            print(f"  Warning: Failed to align face in {img_path}")
                            continue

                    # 提取特征并入库
                    embeddings = self.extractor.extract_features([aligned_face])
                    if embeddings.shape[0] == 0:
                        print(f"  Warning: Failed to extract features from {img_path}")
                        continue

                    embedding = self.extractor.l2_normalize([embeddings[0]])[0]
                    self.database.add_face(
                        person_id=person_id,
                        person_name=person_name,
                        embedding=embedding,
                        image_path=img_path
                    )
                    success_count += 1
                    print(f"  ✓ Added face to database: {img_path}")

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
    
    def recognize_face(self, image_path=None, image_size=FACE_IMAGE_MAX_SIDE, known_threshold=KNOWN_FACE_THRESHOLD,
                   unknown_threshold=None, iou_threshold=FACE_NMS_IOU, min_face_size=FACE_MIN_SIZE,
                   debug=False, image=None):
        """
        人脸识别函数（默认参数统一取自 config/base.py）
        Args:
            image_path: 输入图片路径
            image_size: 图像最长边限制
            known_threshold: 已知人脸匹配阈值（ArcFace 余弦相似度）
            unknown_threshold: 保留参数（历史兼容，当前逻辑未使用）
            iou_threshold: 重复人脸检测的IOU阈值
            min_face_size: 最小人脸尺寸
            debug: 调试模式，显示更多信息
            image: 可选，外部预解码图像（与 image_path 二选一传入）
        Returns:
            (list of recognition results, annotated_image)
        """
        # 加载图像（降采样解码加速；支持外部传入已解码图像，配合批量预解码流水线）
        if image is None:
            image = imread_reduced(image_path, target_max=image_size)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        max_dimension = max(h, w)
        if max_dimension > image_size:
            scale_factor = image_size / max_dimension
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
                
            # 过滤低置信度人脸
            if face_obj.score < FACE_DETECT_MIN_SCORE:
                if debug:
                    print(f"  Skip: Low confidence {face_obj.score}")
                continue

            # 宽高比限制（过滤异常误检）
            aspect_ratio = face_obj.width / face_obj.height
            if not (FACE_ASPECT_RATIO_RANGE[0] <= aspect_ratio <= FACE_ASPECT_RATIO_RANGE[1]):
                if debug:
                    print(f"  Skip: Bad aspect ratio {aspect_ratio}")
                continue
                
            filtered_faces.append(face_obj)
        
        print(f"Detected {len(face_objects)} faces, after filtering: {len(filtered_faces)}")
        
        # 步骤2: 预处理所有过滤后的人脸（裁剪/质量/对齐）
        face_items = []
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
                
                # 计算人脸质量分数（仅作为结果元数据保留，不参与分数调整）
                face_quality = assess_face_quality_simple(face_image)
                
                # 人脸对齐（SCRFD 提供 5 关键点，ArcFace 标准对齐）
                aligned_face = None
                if hasattr(face_obj, 'landmark') and face_obj.landmark is not None:
                    try:
                        aligned_face = self.extractor.align_face(image, face_obj.landmark)
                    except Exception as e:
                        if debug:
                            print(f"  Face {i+1}: Alignment failed: {e}")
                
                if aligned_face is not None:
                    face_resized = aligned_face
                else:
                    # 无关键点时居中裁剪
                    target_size = (112, 112)  # ArcFace 输入尺寸
                    face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LINEAR)
                    # 颜色空间转换
                    if len(face_resized.shape) == 2:
                        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
                    elif face_resized.shape[2] == 4:
                        face_resized = face_resized[:, :, :3]
                
                face_items.append((i, face_obj, face_quality, face_resized, [x1, y1, x2, y2]))
            except Exception as e:
                if debug:
                    print(f"Error preparing face {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

        # 步骤2b: 批量提取特征（单次前向处理全部人脸，减少 GPU 调用与预处理开销）
        raw_results = []
        embeddings_all = None
        if face_items:
            try:
                embeddings_all = self.extractor.extract_features([it[3] for it in face_items])
                if embeddings_all.shape[0] != len(face_items):
                    embeddings_all = None
            except Exception as e:
                if debug:
                    print(f"Batch feature extraction error: {e}")
                embeddings_all = None

        # 步骤2c: 逐脸匹配与结果构建
        for k, (i, face_obj, face_quality, face_resized, bbox) in enumerate(face_items):
            try:
                if embeddings_all is not None:
                    embedding = embeddings_all[k]
                else:
                    # 回退：逐张提取，保证鲁棒性
                    try:
                        emb = self.extractor.extract_features([face_resized])
                        if emb.shape[0] == 0:
                            if debug:
                                print(f"  Face {i+1}: No embeddings extracted")
                            continue
                        embedding = emb[0]
                    except Exception as e:
                        if debug:
                            print(f"  Face {i+1}: Feature extraction error: {e}")
                        continue

                # L2归一化
                embedding = self.extractor.l2_normalize([embedding])[0]
                
                # 搜索匹配（ES ((cos+1)/2)^2 变换，SEARCH_THRESHOLD 对应较低余弦，保证召回）
                matches = self.database.search_face(embedding, top_k=SEARCH_TOP_K, threshold=SEARCH_THRESHOLD)

                # Group matches by person name and keep the best score for each person
                person_matches = {}
                for match in matches:
                    person_name = match['person_name']
                    if person_name not in person_matches or match['score'] > person_matches[person_name]['score']:
                        person_matches[person_name] = match

                # Convert back to list and limit to top 5 unique persons
                # （person_matches 已按人去重并保留每人最高分，无需再次去重）
                matches = sorted(person_matches.values(), key=lambda x: x['score'], reverse=True)[:5]

                if debug:
                    if matches:
                        print(f"  Face {i+1}: Found {len(matches)} matches")
                        for match in matches[:3]:
                            print(f"    Match: {match['person_name']} (Score: {match['score']:.3f})")
                    else:
                        print(f"  Face {i+1}: No matches found in database (database may be empty or no matches above threshold)")

                # 直接使用 ArcFace 余弦相似度作为匹配分数，不做校准/质量乘法。
                # 原分段校准与质量因子是 FaceNet 时代的兜底手段：ArcFace 区分度高，
                # 同人余弦通常 0.5~0.8，乘以 0.85/0.7 等系数会把真实匹配压到阈值之下
                #（实测 raw 0.626 -> calibrated 0.532，threshold 0.55，identified 被降级为 possible）。
                # 模糊人脸的干扰由步骤4的"最佳-次佳分差"歧义检查处理，比质量分数更可靠。
                calibrated_matches = []
                for match in matches:
                    match_copy = match.copy()
                    match_copy['raw_score'] = match['score']  # 保留字段供展示兼容
                    calibrated_matches.append(match_copy)
                
                # 构建结果
                result = {
                    "id": i,
                    "bbox": bbox,
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
            # 返回值与正常路径一致：(results, annotated_image)，避免调用方解包失败
            return [], self.visualizer.draw_faces_on_image(image, [])
        
        # 步骤3: 使用NMS去除重复检测
        unique_results = self._apply_nms(raw_results, iou_threshold)
        
        # 步骤4: 确定最终识别结果
        final_results = []
        for i, result in enumerate(unique_results):
            if result['matches']:
                best_match = result['matches'][0]
                if debug:
                    print(f"DEBUG: Face {i+1} best match score: {best_match['score']}, threshold: {known_threshold}")

                dynamic_threshold = known_threshold
                
                # Check if the best match is significantly better than other candidates
                is_unique_match = True
                if len(result['matches']) > 1:
                    second_best = result['matches'][1]
                    # 最佳与次佳分差过小视为歧义（高置信度除外）
                    if (best_match['score'] - second_best['score']) < FACE_AMBIGUOUS_MARGIN and best_match['score'] < 0.9:
                        is_unique_match = False
                
                if best_match['score'] >= dynamic_threshold and is_unique_match:
                    identified_as = best_match['person_name']
                    confidence = best_match['score']
                    status = 'identified'
                    is_known = True
                elif best_match['score'] >= dynamic_threshold and not is_unique_match:
                    # Ambiguous case - multiple similar matches
                    identified_as = f"可能: {best_match['person_name']} 或 {result['matches'][1]['person_name']}"
                    confidence = best_match['score']
                    status = 'ambiguous'
                    is_known = True
                else:
                    # 检查是否有其他候选匹配接近阈值
                    potential_matches = [m for m in result['matches'] 
                                    if m['score'] >= dynamic_threshold * 0.9]
                    if debug:
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
                if debug:
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
        import logging
        
        # 创建logger
        logger = logging.getLogger('recognize_random_samples')
        logger.setLevel(logging.INFO)
        
        # 清除可能存在的旧handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 创建console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件handler，将日志保存到output_folder下
        os.makedirs(output_folder, exist_ok=True)
        log_file_path = os.path.join(output_folder, 'recognize_random_samples.log')
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加handler到logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        logger.info("=" * 60)
        logger.info("开始随机抽样识别人脸")
        logger.info("=" * 60)
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图片文件
        all_images = list_images(input_folder)
        
        if not all_images:
            logger.warning("警告: 输入文件夹中没有找到图片文件")
            return {}
        
        # 随机抽取指定比例的图片
        sample_size = max(1, int(len(all_images) * fraction))
        sampled_images = random.sample(all_images, sample_size)
        
        logger.info(f"总共找到 {len(all_images)} 张图片")
        logger.info(f"随机抽取 {sample_size} 张图片进行识别")
        
        # 统计变量
        total_processed = 0
        successful_recognitions = 0
        failed_recognitions = 0
        total_faces_detected = 0
        identity_counts = {}  # 记录每个身份被识别的次数
        
        # 处理每张抽样的图片（多线程预解码：cv2 解码释放 GIL，可与 GPU 推理重叠执行）
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        from itertools import islice

        def _decode(fname):
            return cv2.imread(os.path.join(input_folder, fname))

        prefetch_n = min(4, len(sampled_images))
        file_iter = iter(sampled_images)
        pool = ThreadPoolExecutor(max_workers=prefetch_n)
        # 预填充解码队列（保持处理顺序，内存最多多缓存 prefetch_n 张图）
        pending = deque(pool.submit(_decode, f) for f in islice(file_iter, prefetch_n))

        for img_file in tqdm(sampled_images, desc="处理图片"):
            img_path = os.path.join(input_folder, img_file)
            total_processed += 1

            # 取出预解码完成的图像，并补充解码下一张
            image = pending.popleft().result()
            for nf in islice(file_iter, 1):
                pending.append(pool.submit(_decode, nf))

            try:
                # 进行人脸识别（复用预解码图像，跳过重复读取）
                results, annotated_image = self.recognize_face(
                    img_path,
                    known_threshold=known_threshold,
                    unknown_threshold=0.35,
                    image=image
                )
                
                # 统计检测到的人脸数量
                total_faces_detected += len(results)
                
                # 处理每个识别结果
                for i, result in enumerate(results):
                    if result['is_known']:
                        identity = result['identified_as']
                        confidence = result['identification_confidence']
                        quality_score = result['quality'] 
                        successful_recognitions += 1
                        
                        # 更新身份计数
                        if identity not in identity_counts:
                            identity_counts[identity] = 0
                        identity_counts[identity] += 1
                        
                        # 生成保存文件名
                        if identity_counts[identity] == 1:
                            save_filename = f"{identity}_{confidence:.3f}_{quality_score:.3f}.jpg"
                        else:
                            save_filename = f"{identity}_{identity_counts[identity]}_{confidence:.3f}_{quality_score:.3f}.jpg"
                        
                        save_path = os.path.join(output_folder, save_filename)
                        
                        # 保存识别出的人脸图像
                        cv2.imwrite(save_path, result['face_image'])
                        logger.info(f"  保存识别结果: {save_filename}")
                    else:
                        failed_recognitions += 1
                
                # 保存带标注的原图
                annotated_save_path = os.path.join(output_folder, f"annotated_{img_file}")
                cv2.imwrite(annotated_save_path, annotated_image)
                
            except Exception as e:
                logger.error(f"处理图片 {img_file} 时出错: {str(e)}")
                failed_recognitions += 1
                continue

        pool.shutdown()

        # 计算统计信息
        success_rate = successful_recognitions / max(total_faces_detected, 1)
        
        # 打印结果
        logger.info("\n" + "=" * 60)
        logger.info("随机抽样识别完成 - 结果统计")
        logger.info("=" * 60)
        logger.info(f"处理图片数量: {total_processed}")
        logger.info(f"检测到的人脸总数: {total_faces_detected}")
        logger.info(f"成功识别的人脸数: {successful_recognitions}")
        logger.info(f"未识别的人脸数: {failed_recognitions}")
        logger.info(f"人脸识别成功率: {success_rate:.2%}")
        logger.info("\n各身份识别统计:")
        for identity, count in identity_counts.items():
            logger.info(f"  {identity}: {count} 次")
        logger.info("=" * 60)
        
        # 关闭并移除handlers
        file_handler.close()
        console_handler.close()
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)
        
        return {
            "total_processed": total_processed,
            "total_faces_detected": total_faces_detected,
            "successful_recognitions": successful_recognitions,
            "failed_recognitions": failed_recognitions,
            "success_rate": success_rate,
            "identity_counts": identity_counts
        }

if __name__ == '__main__':
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH, FACE_DEVICE
    image_path = 'data/sample_images/4.jpg'

    print(f"Using device: {FACE_DEVICE}")
    recognizer = FaceRecognitionSystem(SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH, device=FACE_DEVICE)
    # 建立人脸索引（第一次），批量添加人脸
    # recognizer.build_face_database('data/face_database', first_run=True)

    # recognizer.build_face_database('data/face_database')

    # 测试embedding和人像检索准确率
    # test_results = recognizer.test_embedding_accuracy('data/face_database')

    # 随机抽样识别人脸
    recognizer.recognize_random_samples(
        input_folder='data/ncpa_test/话剧《样式雷》/【原始】20160609戏剧场-话剧《样式雷》-摄影凌风',
        output_folder='./out/test-7/',
        fraction=1,
        known_threshold=KNOWN_FACE_THRESHOLD
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