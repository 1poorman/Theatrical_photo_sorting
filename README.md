# Theatrical Photo Sorting（戏剧照片整理工具）

基于深度学习的戏剧照片处理与整理系统，提供**人脸识别**、**人像检测分割**、**背景修复**、**图像 embedding 检索聚类**等能力，通过 FastAPI 提供 RESTful API 服务。

## 功能总览

| 模块 | 功能 | 算法/模型 |
|------|------|-----------|
| 人脸识别 | 人脸检测、5 点关键点对齐、身份识别 | **SCRFD**（检测）+ **ArcFace**（识别，ONNX） |
| 人像检测 | 人物框检测与人像 mask 生成 | RTDETR-X |
| 服装分割 | 服装区域分割、轮廓叠加效果 | YOLO11x-seg |
| 镜头分类 | 镜头景别分类（特写/近景/中景/全景等） | YOLO11l-pose |
| 背景修复 | 人物去除后背景修复 | LAMA (FFT) |
| 图像检索 | 图像 embedding 索引、相似检索、聚类分组 | ResNet50/101、DINOv2 |

## 目录结构

```
├── app/main.py                    # FastAPI API 定义（输入输出）
├── app/server.py                  # 模型管理与服务启动（uvicorn，纯 API）
├── app/server_ui.py               # 可视化界面服务（复用 main.app，端口 8199）
├── face_recognition/
│   ├── face_recognition.py    # 人脸识别系统（SCRFD + ArcFace）
│   ├── scrfd/                 # SCRFD ONNX 检测器
│   ├── arcface/               # ArcFace ONNX 特征提取器
│   ├── core/                  # 人脸库/向量库（Elasticsearch）封装
│   └── utils/                 # 对齐、质量评估、可视化工具
├── detection/                 # 人像检测（PersonMaskCreator）
├── inpaint/                   # 背景修复（ImageInpainter, LAMA）
├── seg_clothes/               # 服装分割（PersonesSegmenter）
├── shot/                      # 镜头分类（ShotTypeClassifier）
├── embedding/                 # 图像检索（image_search_system_module）
├── weights/                   # 模型权重
│   ├── scrfd/scrfd_10g_bnkps.onnx
│   ├── arcface/Glint100.onnx
│   ├── rtdetr-x.pt / yolo11x-seg.pt / yolo11l-pose.pt
│   └── cv_fft_inpainting_lama/
├── outputs/                    # 结果存储目录（检测/分割/人脸库/聚类等输出）
├── data/sample_images/               # 人脸识别测试图
├── data/face_database/            # 人脸库（按人物命名的子目录，160x160）
├── tests/test_scrfd_arcface.py      # SCRFD+ArcFace 相似度与识别测试
├── tests/test_real_images.py        # 真实剧照端到端测试
├── requirements_scrfd_arcface.txt
└── environment_scrfd_arcface.yml
```

## 环境安装

```bash
# 1. 创建 conda 环境（Python 3.10）
conda env create -f environment_scrfd_arcface.yml
# 或
conda create -n face_scrfd_arcface python=3.10

conda activate face_scrfd_arcface

# 2. 安装核心依赖
pip install -r requirements_scrfd_arcface.txt

# 3. 安装 PyTorch（CUDA 13.0）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 依赖清单（requirements_scrfd_arcface.txt）

- **推理框架**：`torch` (2.9.0+cu130)、`torchvision`、`onnxruntime-gpu`
- **图像处理**：`opencv-python`、`numpy`、`pillow`、`scipy`
- **模型依赖**：`ultralytics`、`modelscope`、`faiss-cpu`、`scikit-learn`、`kornia`、`matplotlib`
- **服务**：`fastapi`、`uvicorn`、`python-multipart`、`werkzeug`
- **向量库**：`elasticsearch>=8.12,<9`（对应 ES 8.12 服务器）
- **其他**：`onnx`、`tqdm`、`simplejson`、`sortedcontainers`、`pandas`、`pyarrow`、`huggingface_hub`、`safetensors`

## 启动服务

提供两种启动方式，模型实例完全共享（同一 `main.app`），可同时运行互不冲突：

```bash
conda activate face_scrfd_arcface

# 方式一：可视化界面服务（推荐）
python server_ui.py       # 端口 8199，自带浏览器操作界面

# 方式二：纯 API 服务
python server.py          # 端口 8198，仅提供 RESTful API
```

| 服务 | 地址 | 说明 |
|------|------|------|
| 可视化界面 | `http://localhost:8199/ui` | 浏览器直接操作（人脸识别/检测分割/修复/检索） |
| Swagger 文档 | `http://localhost:8199/docs` 或 `:8198/docs` | 在线调试所有 API |
| 纯 API | `http://localhost:8198` | RESTful 接口（供程序调用） |

> 启动时自动加载全部模型（人脸识别 / 人像检测 / 分割 / 修复 / 姿态），需要 GPU 且耗时约 1 分钟。

### 停止服务

```bash
# 按端口停止对应的服务进程
fuser -k 8199/tcp        # 停止可视化界面服务（app/server_ui.py）
fuser -k 8198/tcp        # 停止纯 API 服务（app/server.py）

# 或按进程名停止
pkill -f "python app/server_ui.py"
pkill -f "python app/server.py"

# 查看服务是否在运行
ps aux | grep -E "server(_ui)?\.py" | grep -v grep
ss -tlnp | grep -E "8198|8199"      # 查看端口监听状态
```

## 可视化界面（app/server_ui.py）

`app/server_ui.py` 复用 `app/main.py` 的全部 API 路由，提供现代化浏览器操作界面：

- **👤 人脸识别**：上传图片，展示每张人脸的识别结果、置信度与标注图
- **📂 构建人脸库**：输入文件夹路径一键构建（子目录按人物命名）
- **🎬 组合处理**：人像检测 / 服装分割 / 镜头分类 / 背景修复结果并排展示
- **📚 构建图像索引**：选择嵌入模型（ResNet50/101、DINOv2）构建检索索引
- **🔍 相似图片检索**：上传查询图，展示相似度与缩略图
- **🗂️ 相似图片聚类**：输入文件夹自动聚类分组
- 顶部状态栏一键健康检查，实时显示服务与设备状态

## API 文档

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/` | 重定向到 `/docs` |
| GET | `/api/health` | 健康检查（返回设备信息） |
| GET | `/api/file?path=` | 访问输出目录中的文件 |
| GET | `/api/progress/{task_id}` | 查询任务进度 |
| POST | `/api/face/build_database` | 构建人脸库（`face_db_folder`） |
| POST | `/api/face/recognize` | 识别图片中的人脸（`image`） |
| POST | `/api/image/process` | 组合处理：检测+分割+镜头分类+修复（`image`） |
| POST | `/api/embedding/build_index` | 构建图像索引（`image_folder`/`index_save_path`/`model_name`） |
| POST | `/api/embedding/search` | 相似图片检索（`query_image`/`index_path`/`top_k`） |
| POST | `/api/embedding/group_similar` | 相似图片聚类分组（`image_folder`/`model_name`/`save_dir`） |

### 示例

```bash
# 人脸识别
curl -X POST http://localhost:8198/api/face/recognize \
     -F "image=@data/sample_images/4.jpg"

# 组合处理
curl -X POST http://localhost:8198/api/image/process \
     -F "image=@data/sample_images/4.jpg"

# 构建人脸库（face_db_folder 下每个子目录对应一个人物）
curl -X POST http://localhost:8198/api/face/build_database \
     -F "face_db_folder=data/face_database"

# 图像检索
curl -X POST http://localhost:8198/api/embedding/build_index \
     -F "image_folder=/path/to/images" -F "index_save_path=/path/to/index" \
     -F "model_name=dinov2_base"
curl -X POST http://localhost:8198/api/embedding/search \
     -F "query_image=@query.jpg" -F "top_k=5"
```

## 人脸识别

基于 **SCRFD + ArcFace**（ONNX 推理）替换了旧版 DBFace + FaceNet：

- **SCRFD** (`face_recognition/scrfd/scrfd_det.py`)：`weights/scrfd/scrfd_10g_bnkps.onnx`，9 输出（3 cls + 3 bbox + 3 kps），带 5 关键点
- **ArcFace** (`face_recognition/arcface/arcface_onnx.py`)：`weights/arcface/Glint100.onnx`，输入 112x112 RGB，输出 512 维特征
- **5 点对齐**：基于 SCRFD 关键点按 ArcFace 标准（`norm_crop`）对齐，无需额外依赖
- **阈值**：已知人脸 `known_threshold=0.55`、未知人脸 `unknown_threshold=0.4`（余弦相似度）

### 人脸库存储

人脸特征向量存储于 **Elasticsearch**（8.12）：

- 配置：`face_recognition/config/base.py`（`ELASTIC_URL` / `ELASTIC_USER` / `ELASTIC_PASSWORD`）
- 索引：`face_database_512`（512 维人脸向量库，独立索引，不影响其他数据）

### 本地测试（不依赖服务）

```bash
# 相似度分布分析 + 测试图识别
python tests/test_scrfd_arcface.py

# 真实剧照端到端识别（验证识别结果与演员标注一致性）
python tests/test_real_images.py
```

## Docker

```bash
docker-compose up -d
```

> Dockerfile 默认 `CMD ["app/server.py"]`；docker-compose 中 `./weights` 以可写方式挂载，便于下载权重。

## 注意事项

- 需 **GPU**（CUDA 13.0，如 RTX 3090）以获得最佳性能；无 GPU 时自动回退 CPU（onnxruntime CPU provider）
- 服务默认使用 `cuda:1`（`app/server.py` 中 `CUDA_VISIBLE_DEVICES=1`，可按需修改）
- 人脸库构建使用 `first_run=True` 时会重建 `face_database_512` 索引（其他 ES 索引不受影响）
- 临时文件存放于 `outputs/temp/`，超过 1 小时自动清理
