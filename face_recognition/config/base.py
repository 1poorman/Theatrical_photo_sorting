# 工程名称，logger配置使用
PROJECT_NAME = "multimodal-helper"

POST_URL = "http://172.25.67.254:8040/"
HOST = "0.0.0.0"
PORT = 8040

# dev-local（120服务器消息对列和es）配置


ELASTIC_URL = "http://172.25.67.120:9200"
ELASTIC_USER = "elastic"
ELASTIC_PASSWORD = "elastic"
 
FACE_INDEX = "face"
CAPTION_INDEX = "caption"
IMAGE_INDEX = "image"
VOICEPRINT_INDEX = "voiceprint"
# FACE_INFO_INDEX = "face_info"

# DBFace输入图像的目标尺寸，根据显卡显存可进行调节
TARGET_SIZE = 1920
# 置信度阈值
FACE_THRESH = 0.4

# dbface识别人脸图片像素阈值
IMAGE_PIXEL_THRESHOLD = 200

# 进行人脸检测的图像最低阈值，小于该阈值则resize到该阈值
# IMAGE
IMAGE_EXPECTED_PIXELS = 300
# 人脸模糊度阈值
BLUR_THRESHOLD = 30

# 人脸对比特征阈值
VIDEO_SIMILAR_THRESHOLD = 0.6
IMAGE_SIMILAR_THRESHOLD = 0.9

VIDEO_TARGET_PATH = "/it-school/adms/nas/archivesdoc/face/split_video"
FACE_TARGET_PATH = "/it-school/adms/nas/archivesdoc/face/split_image"
AUDIO_TARGET_PATH = "/it-school/adms/nas/archivesdoc/audio"
IMAGE_TARGET_PATH = "/it-school/adms/nas/archivesdoc/image"


ASR_MODEL = "pretrained_models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ASR_VAD_MODEL = (
    "pretrained_models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
)
ASR_PUNC_MODEL = "pretrained_models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
ASR_SPK_MODEL = (
    "pretrained_models/iic/speech_campplus_sv_zh-cn_16k-common"
)

# EXTRACT_MODEL_CACHE_DIR = "/home/hepenglin/.cache/modelscope/hub"

TOP_K = 10

FORCE_CPU = False
CUDA_DEVICE = 1

POST_TIMEOUT = 1000


SUPPORT_IMAGE_FORMATS = ["heic", "jpg", "jpeg", "bmp", "png", "tif", "tiff"]

IMAGE_ARCHIVE_TYPE = "image"
