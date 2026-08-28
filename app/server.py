# -*- coding: utf-8 -*-
"""
server.py - 模型管理与服务启动

职责：
- 定义全局模型实例与默认路径配置
- 提供模型初始化、临时目录清理、进度追踪等工具函数
- 作为启动入口，加载 main.app 并启动 uvicorn 服务

启动方式：python app/server.py（FastAPI 自带 /docs 交互界面）
"""
import os
import sys
import time
import tempfile
import shutil
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from logger import setup_logging, get_app_logger, log_key_action

logger = get_app_logger()

from detection.PersonMaskCreator import PersonMaskCreator
from inpaint.inpaint_lama_ds import ImageInpainter
# from seg_clothes.ClothesSegment import ClothesSegmenter
from seg_clothes.yolo_seg import PersonesSegmenter
from shot.views_classify import ShotTypeClassifier
from face_recognition.face_recognition import FaceRecognitionSystem

# ---------- 全局配置 ----------

# 设备
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 默认路径
DEFAULT_DETECTION_MODEL_PATH = "./weights/rtdetr-x.pt"
# DEFAULT_SEGCLOTHES_MODEL_PATH = "./weights/segformer_b2_clothes"
DEFAULT_SEGPERSONES_MODEL_PATH = "./weights/yolo11x-seg.pt"
DEFAULT_INPAINTER_MODEL_PATH = "./weights/cv_fft_inpainting_lama"
DEFAULT_POSE_MODEL_PATH = "./weights/yolo11l-pose.pt"
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_SCRFD_MODEL_PATH = "./weights/scrfd/scrfd_10g_bnkps.onnx"
DEFAULT_ARCFACE_MODEL_PATH = "./weights/arcface/Glint100.onnx"
DEFAULT_FACE_DATABASE_PATH = "./outputs/face_index"

# 创建目录
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
TEMP_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 任务进度追踪
progress_tracker = {}

# 最近一次构建的 embedding 索引路径
last_index_path = None

# ---------- 全局模型实例（由 initialize_models 赋值） ----------

face_recognition_model = None
detection_model = None
# segclothes_model = None
segpersones_model = None
inpainter_model = None
pose_model = None


# ---------- 工具函数 ----------

def cleanup_temp_directory():
    """清理临时目录中超过 1 小时的旧文件，并限制总文件数"""
    try:
        current_time = time.time()
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            mod_time = os.path.getmtime(item_path)
            if current_time - mod_time > 3600:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        items = os.listdir(TEMP_DIR)
        if len(items) > 100:
            items_with_time = [(item, os.path.getmtime(os.path.join(TEMP_DIR, item))) for item in items]
            items_with_time.sort(key=lambda x: x[1])
            items_to_remove = items_with_time[:len(items) - 50]
            for item, _ in items_to_remove:
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    except Exception as e:
        logger.warning("Error during temp directory cleanup: %s", e)


def prepare_temp_directory():
    """清理旧文件并返回一个新的临时子目录"""
    cleanup_temp_directory()
    return tempfile.mkdtemp(dir=TEMP_DIR)


def periodic_cleanup():
    """周期性清理临时目录（每 30 分钟）"""
    while True:
        time.sleep(1800)
        cleanup_temp_directory()


def update_progress(task_id, progress, message):
    """更新任务进度"""
    if task_id in progress_tracker:
        progress_tracker[task_id]["progress"] = progress
        progress_tracker[task_id]["message"] = message


def get_error(code=-1, message=""):
    return {"code": code, "message": message}


def serialize_results(results):
    """将 YOLO 结果对象转换为 JSON 可序列化格式"""
    if isinstance(results, list):
        serialized = []
        for result in results:
            if hasattr(result, 'boxes'):
                boxes_data = []
                if result.boxes is not None:
                    for box in result.boxes:
                        box_data = {
                            'xyxy': box.xyxy.cpu().numpy().tolist() if hasattr(box.xyxy, 'cpu') else box.xyxy.tolist(),
                            'conf': box.conf.cpu().item() if hasattr(box.conf, 'cpu') else box.conf.item(),
                            'cls': box.cls.cpu().item() if hasattr(box.cls, 'cpu') else box.cls.item()
                        }
                        boxes_data.append(box_data)
                serialized.append({'boxes': boxes_data})
            else:
                serialized.append(str(result))
        return serialized
    else:
        if hasattr(results, 'boxes'):
            boxes_data = []
            if results.boxes is not None:
                for box in results.boxes:
                    box_data = {
                        'xyxy': box.xyxy.cpu().numpy().tolist() if hasattr(box.xyxy, 'cpu') else box.xyxy.tolist(),
                        'conf': box.conf.cpu().item() if hasattr(box.conf, 'cpu') else box.conf.item(),
                        'cls': box.cls.cpu().item() if hasattr(box.cls, 'cpu') else box.cls.item()
                    }
                    boxes_data.append(box_data)
            return {'boxes': boxes_data}
        else:
            return str(results)


def initialize_models(detection_model_path=DEFAULT_DETECTION_MODEL_PATH,
                      # segclothes_model_path=DEFAULT_SEGCLOTHES_MODEL_PATH,
                      segpersones_model_path=DEFAULT_SEGPERSONES_MODEL_PATH,
                      inpainter_model_path=DEFAULT_INPAINTER_MODEL_PATH,
                      pose_model_path=DEFAULT_POSE_MODEL_PATH,
                      scrfd_model_path=DEFAULT_SCRFD_MODEL_PATH,
                      arcface_model_path=DEFAULT_ARCFACE_MODEL_PATH):
    """初始化所有模型（带关键动作日志与逐模型耗时统计）"""
    global detection_model, segpersones_model, inpainter_model, pose_model, face_recognition_model

    def _load(name, fn):
        start = time.time()
        try:
            fn()
            logger.info("[ACTION] load model %s DONE in %.2fs", name, time.time() - start)
        except Exception as e:
            logger.error("[ACTION] load model %s FAIL (%.2fs): %s", name, time.time() - start, e, exc_info=True)
            raise

    try:
        if torch.cuda.is_available():
            device = "cuda:1"
            logger.info("CUDA is available. Using device: %s (device count: %d)", device, torch.cuda.device_count())
        else:
            device = "cpu"
            logger.warning("CUDA is not available. Using CPU.")

        if os.path.exists(scrfd_model_path) and os.path.exists(arcface_model_path):
            def _load_face_recognition():
                global face_recognition_model
                # TensorRT 加速推理。注意：本进程通过 CUDA_VISIBLE_DEVICES=1 限定可见 GPU，
                # 可见设备内编号固定为 0，故直接用 'tensorrt'（实际用哪块卡由环境变量决定）
                fr_device = "tensorrt" if device.startswith("cuda") else device
                face_recognition_model = FaceRecognitionSystem(
                    scrfd_model_path, arcface_model_path, device=fr_device)
            _load("face_recognition", _load_face_recognition)
        else:
            logger.warning("Face recognition models not found at %s or %s", scrfd_model_path, arcface_model_path)

        if detection_model_path and os.path.exists(detection_model_path):
            def _load_detection():
                global detection_model
                detection_model = PersonMaskCreator(detection_model_path)
            _load("detection", _load_detection)
        else:
            logger.warning("Detection model not found at %s", detection_model_path)

        if segpersones_model_path and os.path.exists(segpersones_model_path):
            def _load_segmentation():
                global segpersones_model
                segpersones_model = PersonesSegmenter(segpersones_model_path)
            _load("segmentation", _load_segmentation)
        else:
            logger.warning("Segmentation model not found at %s", segpersones_model_path)

        if inpainter_model_path and os.path.exists(inpainter_model_path):
            def _load_inpainting():
                global inpainter_model
                inpainter_model = ImageInpainter(inpainter_model_path, max_size=1024)
            _load("inpainting", _load_inpainting)
        else:
            logger.warning("Inpainting model not found at %s", inpainter_model_path)

        if pose_model_path and os.path.exists(pose_model_path):
            def _load_pose():
                global pose_model
                pose_model = ShotTypeClassifier(pose_model_path)
            _load("pose", _load_pose)
        else:
            logger.warning("Pose model not found at %s", pose_model_path)

        log_key_action("initialize_models", device=device)
        return True
    except Exception as e:
        logger.error("Error initializing models: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    import uvicorn
    from main import app

    logger, log_file = setup_logging()
    logger.info("Starting FastAPI server on http://0.0.0.0:8198 (docs: http://0.0.0.0:8198/docs)")
    # log_config=None：交给我们自己的日志配置，uvicorn 日志统一落盘
    uvicorn.run(app, host="0.0.0.0", port=8198, log_config=None, log_level="info")
