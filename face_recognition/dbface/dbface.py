import common as common
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from model.DBFace import DBFace
# from ..config.base import TARGET_SIZE, FACE_THRESH, CUDA_DEVICE
from typing import List, Tuple, Union
import os
import glob
TARGET_SIZE = 1920
FACE_THRESH = 0.4
class DBFaceDetector:
    def __init__(self, model_path: str = None, use_gpu: bool = True, device=None):
        """
        Initialize the DBFace detector.
        
        Args:
            model_path (str): Path to the model weights file
            use_gpu (bool): Whether to use GPU for inference
        """
        if device is None:
            if use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda:1")  # Default to cuda:0
            else:
                self.device = torch.device("cpu")
        else:
            if 'cuda' in device and torch.cuda.is_available():
                device_index = int(device.split(':')[1]) if ':' in device else 0
                if device_index < torch.cuda.device_count():
                    self.device = torch.device(device)
                else:
                    print(f"Device {device} not available, using cuda:1")
                    self.device = torch.device("cuda:1")
            else:
                self.device = torch.device("cpu")
        
        print(f"DBFaceDetector using device: {self.device}")

        self.model = DBFace()
        self.model.to(self.device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def nms(self, objs, iou=0.5):
        """
        Apply non-maximum suppression to detected objects.
        
        Args:
            objs: List of detected objects
            iou: IoU threshold for NMS
            
        Returns:
            Filtered list of objects after NMS
        """
        if objs is None or len(objs) <= 1:
            return objs

        objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):
            if flags[index] != 0:
                continue

            keep.append(obj)
            for j in range(index + 1, len(objs)):
                if flags[j] == 0 and obj.iou(objs[j]) > iou:
                    flags[j] = 1

        return keep

    def _detect_single(self, image: np.ndarray, threshold=FACE_THRESH, nms_iou=0.5):
        """
        Detect faces in a single image.
        
        Args:
            image: Input image as numpy array
            threshold: Detection threshold
            nms_iou: IoU threshold for NMS
            
        Returns:
            List of detected bounding boxes
        """
        mean = [0.408, 0.447, 0.47]
        std = [0.289, 0.274, 0.278]

        image = common.pad(image)
        image = ((image / 255.0 - mean) / std).astype(np.float32)
        image = image.transpose(2, 0, 1)

        torch_image = torch.from_numpy(image)[None]
        torch_image = torch_image.to(self.device)

        hm, box, landmark = self.model(torch_image)
        hm_pool = F.max_pool2d(hm, 3, 1, 1)

        # 对提取不出1000个点的过小的人脸图片进行了过滤
        if hm.shape[2] * hm.shape[3] > 1000:
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]

            scores = scores.squeeze()
            indices = indices.squeeze()
            ys = list((indices / hm_width).int().data.numpy())
            xs = list((indices % hm_width).int().data.numpy())
            scores = list(scores.data.numpy())
            box = box.cpu().squeeze().data.numpy()
            landmark = landmark.cpu().squeeze().data.numpy()

            stride = 4
            objs = []
            for cx, cy, score in zip(xs, ys, scores):
                if score < threshold:
                    break

                x, y, r, b = box[:, cy, cx]
                xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
                x5y5 = landmark[:, cy, cx]
                x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
                box_landmark = list(zip(x5y5[:5], x5y5[5:]))
                objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
            return self.nms(objs, iou=nms_iou)
        else:
            return []

    @staticmethod
    def resize_img_keep_ratio(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio and padding to target size.
        
        Args:
            img: Input image
            target_size: Target size as (height, width)
            
        Returns:
            Resized and padded image
        """
        old_size = img.shape[0:2]

        # 找到 目标尺寸/图片尺寸 的小值，也就是相对长边
        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        # 计算按原比例缩放后的目标尺寸
        new_size = tuple([int(i * ratio) for i in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))

        pad_w = target_size[1] - new_size[1]
        pad_h = target_size[0] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        img_new = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255)
        )
        return img_new

    def detect_image(self, image: np.ndarray, pixel_threshold: int = 0) -> Tuple[List[Tuple], np.ndarray, float]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            Tuple of (face rectangles, processed image, resize ratio)
        """
        image_h, image_w = image.shape[0:2]
        
        # 指定目标最长边大小，用于控制人脸图片的大小和显存占用
        # 如果图像尺寸大于目标阈值，则缩小图像最大边到阈值大小
        # 图片太大的情况，需要先resize再进网络，防止炸显存
        ratio = 1
        if max(image_h, image_w) > TARGET_SIZE:
            old_size = image.shape[0:2]
            # 找出相对长边
            ratio = min(float(TARGET_SIZE) / (old_size[i]) for i in range(len(old_size)))
            # 计算按原比例缩放后的目标尺寸
            new_size = tuple([int(i * ratio) for i in old_size])
            image = cv2.resize(image, (new_size[1], new_size[0]))

        objs = self._detect_single(image)

        # 修改为只画出检测面积最大的
        area_list = []
        for obj in objs:
            area = obj.width * obj.height
            area_list.append(area)
        # 在面积的列表中找出像素大于指定数量的值和索引
        filter_faces = [
            index for index, value in enumerate(area_list) if value > pixel_threshold
        ]  # 这个阈值用来筛选掉不需要的较小的人脸框
        face_rects = []
        for index_num in filter_faces:
            obj = objs[index_num]

            face_rect = (obj.x, obj.y, obj.width, obj.height)
            face_rects.append(face_rect)

        return face_rects, image, ratio
    
    def detect_faces_with_info(self, image: np.ndarray, pixel_threshold: int = 0) -> Tuple[List[object], np.ndarray, float]:
        """
        Detect faces in an image and return comprehensive face information.
        
        Args:
            image: Input image as numpy array
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            Tuple of (face objects with full info, processed image, resize ratio)
        """
        image_h, image_w = image.shape[0:2]
        
        # 指定目标最长边大小，用于控制人脸图片的大小和显存占用
        # 如果图像尺寸大于目标阈值，则缩小图像最大边到阈值大小
        # 图片太大的情况，需要先resize再进网络，防止炸显存
        ratio = 1
        if max(image_h, image_w) > TARGET_SIZE:
            old_size = image.shape[0:2]
            # 找出相对长边
            ratio = min(float(TARGET_SIZE) / (old_size[i]) for i in range(len(old_size)))
            # 计算按原比例缩放后的目标尺寸
            new_size = tuple([int(i * ratio) for i in old_size])
            image = cv2.resize(image, (new_size[1], new_size[0]))

        objs = self._detect_single(image)

        # Filter faces based on area threshold
        filtered_objs = [obj for obj in objs if obj.width * obj.height > pixel_threshold]
        
        # Scale coordinates back to original image size
        scaled_objs = []
        for obj in filtered_objs:
            # Create a copy with adjusted coordinates
            scaled_obj = common.BBox(
                label=obj.label,
                xyrb=(obj.x / ratio, obj.y / ratio, obj.r / ratio, obj.b / ratio),
                score=obj.score,
                landmark=[(pt[0] / ratio, pt[1] / ratio) for pt in obj.landmark] if obj.landmark else None
            )
            scaled_objs.append(scaled_obj)
        return scaled_objs, image, ratio
    
    def detect_single_file(self, image_path: str, pixel_threshold: int = 0) -> Tuple[List[Tuple], np.ndarray, float]:
        """
        Detect faces in a single image file.
        
        Args:
            image_path: Path to the input image
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            Tuple of (face rectangles, processed image, resize ratio)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        return self.detect_image(image, pixel_threshold)
    
    def detect_batch_files(self, image_paths: List[str], pixel_threshold: int = 0) -> List[Tuple[List[Tuple], np.ndarray, float]]:
        """
        Detect faces in multiple image files.
        
        Args:
            image_paths: List of paths to input images
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            List of results for each image, where each result is a tuple of 
            (face rectangles, processed image, resize ratio)
        """
        results = []
        for path in image_paths:
            try:
                result = self.detect_single_file(path, pixel_threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                results.append(([], None, 1.0))  # Return empty result for failed images
        return results
    
    def detect_batch_images(self, images: List[np.ndarray], pixel_threshold: int = 0) -> List[Tuple[List[Tuple], np.ndarray, float]]:
        """
        Detect faces in multiple images.
        
        Args:
            images: List of input images as numpy arrays
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            List of results for each image, where each result is a tuple of 
            (face rectangles, processed image, resize ratio)
        """
        results = []
        for image in images:
            try:
                result = self.detect_image(image, pixel_threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                results.append(([], None, 1.0))  # Return empty result for failed images
        return results
    
    def detect_folder(self, folder_path: str, pixel_threshold: int = 0, extensions: List[str] = None) -> dict:
        """
        Detect faces in all images within a folder.
        
        Args:
            folder_path: Path to the folder containing images
            pixel_threshold: Minimum pixel area threshold for faces
            extensions: List of file extensions to process (default: ['jpg', 'jpeg', 'png'])
            
        Returns:
            Dictionary mapping filenames to detection results
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if extensions is None:
            extensions = ['jpg', 'jpeg', 'png']
        
        # Create patterns for all supported extensions
        patterns = [os.path.join(folder_path, f"*.{ext}") for ext in extensions]
        patterns_upper = [os.path.join(folder_path, f"*.{ext.upper()}") for ext in extensions]
        patterns.extend(patterns_upper)
        
        # Find all matching files
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(pattern))
        
        # Process all images
        results = {}
        for file_path in image_files:
            try:
                result = self.detect_single_file(file_path, pixel_threshold)
                filename = os.path.basename(file_path)
                results[filename] = result
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                filename = os.path.basename(file_path)
                results[filename] = ([], None, 1.0)  # Return empty result for failed images
        
        return results

    def extract_and_save_faces(self, image: np.ndarray, output_dir: str, image_name: str = "face", 
                         target_size: Tuple[int, int] = (1024, 1024), 
                         pixel_threshold: int = 0,
                         save_results: bool = False) -> List[str]:
        """
        Detect all faces in an image, crop them to uniform size, and save to disk.
        
        Args:
            image: Input image as numpy array
            output_dir: Directory to save cropped faces
            image_name: Base name for saved face images
            target_size: Target size for cropped faces (height, width)
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            List of paths to saved face images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect faces in the image
        face_rects, processed_image, ratio = self.detect_image(image, pixel_threshold)
        
        saved_paths = []
        
        # Draw bounding boxes on the original image
        image_with_detections = image.copy()
        
        # Crop and save each face
        for i, (x, y, w, h) in enumerate(face_rects):
            try:
                # Adjust coordinates based on resizing ratio
                x_orig = int(x / ratio)
                y_orig = int(y / ratio)
                w_orig = int(w / ratio)
                h_orig = int(h / ratio)
                
                # Draw bounding box on the full image
                cv2.rectangle(image_with_detections, (x_orig, y_orig), 
                            (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
                cv2.putText(image_with_detections, f'Face {i+1}', 
                        (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1)
                
                # Validate coordinates
                if w_orig <= 0 or h_orig <= 0:
                    print(f"Skipping face {i+1}: Invalid dimensions ({w_orig}x{h_orig})")
                    continue
                    
                # Add some padding around the face (optional)
                padding_x = int(w_orig * 0.1)  # 10% padding
                padding_y = int(h_orig * 0.1)
                
                x_pad = max(0, x_orig - padding_x)
                y_pad = max(0, y_orig - padding_y)
                x2_pad = min(image.shape[1], x_orig + w_orig + padding_x)
                y2_pad = min(image.shape[0], y_orig + h_orig + padding_y)
                
                # Validate final coordinates
                if x2_pad <= x_pad or y2_pad <= y_pad:
                    print(f"Skipping face {i+1}: Invalid crop coordinates")
                    continue
                    
                # Crop the face region
                face_crop = image[y_pad:y2_pad, x_pad:x2_pad]
                
                # Check if crop is valid
                if face_crop is None or face_crop.size == 0:
                    print(f"Skipping face {i+1}: Empty crop")
                    continue
                    
                # Resize to target size
                face_resized = cv2.resize(face_crop, (target_size[1], target_size[0]), 
                                        interpolation=cv2.INTER_LINEAR)
                
                # Generate unique filename
                face_filename = f"{image_name}_face_{i+1}.jpg"
                face_path = os.path.join(output_dir, face_filename)
                
                if save_results:
                    # Save the cropped face
                    success = cv2.imwrite(face_path, face_resized)
                    if success:
                        saved_paths.append(face_path)
                        print(f"Saved face {i+1} to: {face_path}")
                    else:
                        print(f"Failed to save face {i+1}")
                    
            except Exception as e:
                print(f"Error processing face {i+1}: {str(e)}")
                continue
        if save_results:
            # Save the full detection result image
            try:
                full_result_filename = f"{image_name}_detection_result.jpg"
                full_result_path = os.path.join(output_dir, full_result_filename)
                success = cv2.imwrite(full_result_path, image_with_detections)
                if success:
                    print(f"Saved full detection result to: {full_result_path}")
                else:
                    print("Failed to save full detection result")
            except Exception as e:
                print(f"Error saving full detection result: {str(e)}")
        
        print(f"Extracted and saved {len(saved_paths)} faces from image")
        return saved_paths

    def extract_and_save_faces_from_file(self, image_path: str, output_dir: str, 
                                    target_size: Tuple[int, int] = (512, 512),
                                    pixel_threshold: int = 0,
                                    save_results: bool = False) -> List[str]:
        """
        Detect all faces in an image file, crop them to uniform size, and save to disk.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save cropped faces
            target_size: Target size for cropped faces (height, width)
            pixel_threshold: Minimum pixel area threshold for faces
            
        Returns:
            List of paths to saved face images
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Use image filename (without extension) as base name
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        return self.extract_and_save_faces(image, output_dir, image_name, target_size, pixel_threshold, save_results)

    def extract_and_save_faces_from_folder(self, folder_path: str, output_base_dir: str,
                                    target_size: Tuple[int, int] = (1024, 1024),
                                    pixel_threshold: int = 0,
                                    extensions: List[str] = None) -> dict:
        """
        Detect all faces in images within a folder, crop them to uniform size, and save to disk.
        
        Args:
            folder_path: Path to the folder containing images
            output_base_dir: Base directory to save cropped faces (subdirectories will be created per image)
            target_size: Target size for cropped faces (height, width)
            pixel_threshold: Minimum pixel area threshold for faces
            extensions: List of file extensions to process (default: ['jpg', 'jpeg', 'png'])
            
        Returns:
            Dictionary mapping image filenames to lists of saved face paths
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if extensions is None:
            extensions = ['jpg', 'jpeg', 'png']
        
        save_result = True if output_base_dir else False
        # Create patterns for all supported extensions
        patterns = [os.path.join(folder_path, f"*.{ext}") for ext in extensions]
        patterns_upper = [os.path.join(folder_path, f"*.{ext.upper()}") for ext in extensions]
        patterns.extend(patterns_upper)
        
        # Find all matching files
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(pattern))
        
        print(f"Found {len(image_files)} image files to process")
        
        # Process all images
        results = {}
        successful_images = 0
        total_faces = 0
        
        for file_path in image_files:
            try:
                # Create subdirectory for this image's faces
                image_name = os.path.splitext(os.path.basename(file_path))[0]
                image_output_dir = os.path.join(output_base_dir, image_name)
                
                # Extract and save faces
                saved_paths = self.extract_and_save_faces_from_file(
                    file_path, image_output_dir, target_size, pixel_threshold, save_result
                )
                
                results[file_path] = saved_paths
                successful_images += 1
                total_faces += len(saved_paths)
                print(f"Processed {file_path}: {len(saved_paths)} faces extracted")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results[file_path] = []
        
        print(f"Folder processing complete: {successful_images}/{len(image_files)} images processed, {total_faces} faces extracted")
        return results

if __name__ == "__main__":
    # Example usage:
    detector = DBFaceDetector(model_path="./model/dbface.pth")
    # # 1. Extract faces from a single image array
    # try:
    #     # Example usage:
    #     detector = DBFaceDetector(model_path="./model/dbface.pth")
    #     image_path = "/home/huachenghao/codes/NCPA_test-images/话剧《样式雷》/【原始】20160609戏剧场-话剧《样式雷》-摄影凌风/20160609戏剧场-话剧《样式雷》 (77)-摄影凌风.JPG"
        
    #     if not os.path.exists(image_path):
    #         print(f"Image not found: {image_path}")
    #     else:
    #         image = cv2.imread(image_path)
    #         if image is None:
    #             print(f"Could not read image: {image_path}")
    #         else:
    #             print(f"Processing image with shape: {image.shape}")
    #             saved_face_paths = detector.extract_and_save_faces(image, "./output/faces", "my_image")
    #             print(f"Successfully processed {len(saved_face_paths)} faces")
                
    # except Exception as e:
    #     print(f"Error in main execution: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

    # # 2. Extract faces from a single image file
    # try:
    #     saved_face_paths = detector.extract_and_save_faces_from_file(
    #         "/path/to/image.jpg", 
    #         "./output/faces"
    #     )
    #     print(f"Successfully processed single file with {len(saved_face_paths)} faces")
    # except Exception as e:
    #     print(f"Error processing single file: {str(e)}")
    #     traceback.print_exc()

    # 3. Extract faces from all images in a folder
    try:
        results = detector.extract_and_save_faces_from_folder(
            "/home/huachenghao/codes/NCPA_test-images/话剧《样式雷》/【整理完成】话剧《样式雷》彩排", 
            "./output/all_faces_160",
            target_size=(160, 160),
        )
        total_faces = sum(len(paths) for paths in results.values())
        print(f"Successfully processed folder with {total_faces} faces from {len(results)} images")
    except Exception as e:
        print(f"Error processing folder: {str(e)}")
        import traceback
        traceback.print_exc()