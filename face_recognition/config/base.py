# -*- coding: utf-8 -*-
"""config.base - 项目统一参数中心

所有可调参数集中于此，代码中不再散落魔法数字/路径。
支持通过环境变量覆盖（部署时无需改代码），格式：TRANS_<参数名>。

用法：
    from config.base import KNOWN_FACE_THRESHOLD, WEIGHTS_DIR
"""
import os

# ---------- 基础路径 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模型权重根目录
WEIGHTS_DIR = os.environ.get('TRANS_WEIGHTS_DIR', os.path.join(PROJECT_ROOT, 'weights'))

# ---------- 服务 ----------
API_HOST = '0.0.0.0'
API_PORT = 8198
UI_PORT = 8199

# ---------- 模型权重路径 ----------
SCRFD_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'scrfd', 'scrfd_10g_bnkps.onnx')
ARCFACE_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'arcface', 'Glint100.onnx')
DETECTION_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'rtdetr-x.pt')
SEGPERSON_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'yolo11x-seg.pt')
POSE_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'yolo11l-pose.pt')
INPAINTER_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'cv_fft_inpainting_lama')

# ---------- 推理设备 ----------
# 'tensorrt' / 'tensorrt:N' / 'cuda' / 'cuda:N' / 'cpu'
FACE_DEVICE = os.environ.get('TRANS_FACE_DEVICE', 'tensorrt')

# ---------- Elasticsearch 向量库 ----------
ELASTIC_URL = os.environ.get('TRANS_ELASTIC_URL', 'http://172.25.75.23:9200')
ELASTIC_USER = os.environ.get('TRANS_ELASTIC_USER', 'elastic')
ELASTIC_PASSWORD = os.environ.get('TRANS_ELASTIC_PASSWORD', 'elastic')
ELASTIC_FACE_INDEX = 'face_database_512'

# ---------- 人脸识别参数 ----------
# 识别时图像最长边（含降采样解码的目标下限）
FACE_IMAGE_MAX_SIDE = int(os.environ.get('TRANS_FACE_IMAGE_MAX_SIDE', '1920'))
# 已知人脸判定阈值（ArcFace 余弦相似度：同人 0.5~0.8，异人 <0.3）
KNOWN_FACE_THRESHOLD = float(os.environ.get('TRANS_KNOWN_FACE_THRESHOLD', '0.55'))
# 向量库检索阈值（低于 KNOWN_FACE_THRESHOLD，保证召回）
SEARCH_THRESHOLD = 0.4
# 检索返回条数
SEARCH_TOP_K = 10
# 检测置信度下限（低于丢弃）
FACE_DETECT_MIN_SCORE = 0.3
# 人脸最小边长（像素，低于丢弃）
FACE_MIN_SIZE = 20
# 人脸框宽高比允许范围（过滤误检）
FACE_ASPECT_RATIO_RANGE = (0.2, 5.0)
# 结果 NMS 的 IoU 阈值（去除重叠重复人脸）
FACE_NMS_IOU = 0.4
# 歧义判定：最佳与次佳分差小于该值视为 ambiguous
FACE_AMBIGUOUS_MARGIN = 0.05

# ---------- 人脸库构建 ----------
# 已裁剪标准人脸图尺寸（直接对齐入库，跳过检测）
FACE_DB_STANDARD_SIZE = 160
# 建库时人脸最小面积（像素）
FACE_DB_MIN_AREA = 300

# ---------- 日志 ----------
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_RETENTION_DAYS = 30
