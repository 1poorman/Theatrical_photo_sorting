# Theatrical Photo Sorting（戏剧照片整理工具）

基于深度学习的戏剧照片处理与整理系统，通过 FastAPI 提供 RESTful API 与浏览器可视化界面，覆盖五大能力：

| 能力 | 说明 | 算法/模型 |
|------|------|-----------|
| 人脸识别 | 检测 → 5 点对齐 → 特征提取 → ES 向量检索；**TensorRT FP16 加速** | SCRFD + ArcFace (ONNX) |
| 人像检测 | 人物框检测与人像 mask 生成 | RTDETR-X |
| 人像分割 | 人物/服装区域分割 | YOLO11x-seg |
| 镜头分类 | 景别分类（特写/近景/中景/全景） | YOLO11l-pose |
| 背景修复 | 人物去除后背景修复 | LAMA (ModelScope) |
| 图像检索 | embedding 索引、相似检索、聚类分组 | ResNet50/101、DINOv2、SigLIP2 (faiss) |
| 智能整理 | 连拍去重选优、半自动人脸库、按演员视图、场景划分、行当分类、规范命名流水线 | SCRFD+ArcFace、YOLO11l-pose、SigLIP2 zero-shot（详见 `docs/smart_organize_blueprint.md`） |

## 目录结构

```
├── app/                            # FastAPI 服务层
│   ├── main.py                     #   API 路由定义（所有端点）
│   ├── server.py                   #   模型加载/生命周期管理 + 服务启动（端口 8198）
│   └── server_ui.py                #   可视化界面服务（复用 main.app，端口 8199）
│
├── core_modules/                   # ★ 核心功能模块
│   ├── face_recognition/           #   人脸识别（多文件子包）
│   │   ├── face_recognition.py     #     识别系统主类（检测/对齐/特征/匹配/流程编排）
│   │   ├── database.py             #     ES 人脸向量库（索引/写入/检索）
│   │   ├── trt_utils.py            #     TensorRT 支持（库预加载、FP16、引擎缓存）
│   │   ├── trt_cache/              #     TRT 引擎缓存（自动生成，不入库）
│   │   ├── scrfd/                  #     SCRFD 检测器（ONNX/TRT 推理 + 解码）
│   │   └── arcface/                #     ArcFace 特征提取器（含标准 5 点对齐）
│   ├── person_detection.py         #   人像检测（RTDETR，PersonMaskCreator）
│   ├── person_segmentation.py      #   人像分割（YOLO11x-seg）
│   ├── shot_classify.py            #   镜头景别分类（YOLO11l-pose）
│   ├── inpaint_lama.py             #   背景修复（LAMA，ImageInpainter）
│   ├── image_search.py             #   图像检索（faiss 索引/检索/聚类）
│   ├── organize/                   #   智能整理（Smart Organizer 子包）
│   │   ├── filename_parser.py      #     文件名/目录名解析（日期/剧场/剧目/人物/摄影师/连拍号）
│   │   ├── burst_dedup.py          #     连拍分组去重 + 全图质量评分 + 景别感知选优
│   │   ├── face_db_builder.py      #     半自动人脸库（单人归属 + 多人锚定传播两轮构建）
│   │   ├── actor_view.py           #     按演员整理视图生成
│   │   ├── scene_split.py          #     场景划分（EXIF 时间分段为主 + embedding 聚类为辅）
│   │   ├── role_classifier.py      #     戏曲行当 zero-shot 分类（生旦净丑，实验性）
│   │   └── smart_organizer.py      #     整理流水线编排入口
│   └── tools/                      #   跨模块共享工具
│       ├── logger.py               #     集中日志（北京时区、按天滚动、StepTimer 计时）
│       ├── image_io.py             #     降采样解码、最长边缩放、图片枚举
│       ├── face_quality.py         #     人脸质量评估
│       └── visualization.py        #     检测结果可视化
│
├── config/                         # ★ 项目统一参数中心
│   └── base.py                     #   阈值/路径/向量库地址等，支持环境变量覆盖
│
├── tests/                          # 测试脚本（见"测试"章节）
├── docs/                           # 文档（优化测试报告等）
├── data/                           # 数据（face_database/ 人脸库、sample_images/、ncpa_test/）
├── weights/                        # 模型权重（不入库）
├── outputs/ logs/                  # 运行产物（不入库）
├── Dockerfile / docker-compose.yml
├── requirements_scrfd_arcface.txt  # pip 依赖
└── environment_scrfd_arcface.yml   # conda 环境
```

> **config/base.py** 是唯一的参数中心：阈值、模型路径、ES 地址等全部集中于此，支持 `TRANS_<参数名>` 环境变量覆盖。
> **core_modules/tools/** 收敛了原先散落在各模块的重复实现，新功能请优先复用。

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment_scrfd_arcface.yml
conda activate face_scrfd_arcface

# 2. 安装依赖（PyTorch 按需选择 CUDA 版本）
pip install -r requirements_scrfd_arcface.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. 准备权重（weights/ 下，首次运行部分模型自动下载）
#    scrfd/scrfd_10g_bnkps.onnx  arcface/Glint100.onnx  rtdetr-x.pt
#    yolo11x-seg.pt  yolo11l-pose.pt  cv_fft_inpainting_lama/  dinov2_vitb14_pretrain.pth

# 4. 启动服务
python app/server_ui.py    # 可视化界面，端口 8199（推荐）
python app/server.py       # 纯 API，端口 8198
```

| 服务 | 地址 |
|------|------|
| 可视化界面 | `http://localhost:8199/ui` |
| Swagger 文档 | `http://localhost:8199/docs` 或 `:8198/docs` |

启动时自动加载全部模型（需 GPU，约 1 分钟；TensorRT 引擎首次构建约十几秒，之后走缓存秒级加载）。

## API 一览

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（设备信息） |
| GET | `/api/file?path=` | 访问输出目录文件 |
| GET | `/api/progress/{task_id}` | 查询任务进度 |
| POST | `/api/face/build_database` | 构建人脸库（子目录按人物命名） |
| POST | `/api/face/recognize` | 人脸识别 |
| POST | `/api/image/process` | 组合处理：检测+分割+镜头分类+修复 |
| POST | `/api/embedding/build_index` | 构建图像索引 |
| POST | `/api/embedding/search` | 相似图片检索 |
| POST | `/api/embedding/group_similar` | 相似图片聚类分组 |
| POST | `/api/organize/build_face_database` | 半自动人脸库构建（整理完成/原始目录 → 锚定库） |
| POST | `/api/organize/actor_view` | 按演员生成整理视图 |
| POST | `/api/organize/run` | 智能整理流水线（去重→识别→行当→场景→规范命名） |

```bash
curl -X POST http://localhost:8198/api/face/recognize -F "image=@data/sample_images/4.jpg"
```

## 人脸识别

流程：SCRFD 检测（含 5 关键点）→ ArcFace 标准对齐（112×112）→ 512 维特征 → ES 向量检索。

- **推理后端**：`device='tensorrt'`（默认，FP16 加速）/ `'tensorrt:N'` / `'cuda'` / `'cpu'`，TRT 不可用时自动回退
- **依赖版本**：`tensorrt-cu12==10.16.1.11` 必须搭配 onnxruntime 1.23.x（链接 `libnvinfer.so.10`）
- **性能**（RTX 3090）：单图端到端 184ms（含降采样解码、批量特征提取），详见 `docs/optimization_report.md`
- **阈值**：`known_threshold=0.55`（ArcFace 余弦相似度，同人 0.5~0.8、异人 <0.3，无需分数校准）

```python
from core_modules.face_recognition.face_recognition import FaceRecognitionSystem
system = FaceRecognitionSystem(scrfd_path, arcface_path, device='tensorrt')
results, annotated = system.recognize_face('photo.jpg', known_threshold=0.55)
```

人脸向量存储于 Elasticsearch 8.x（索引 `face_database_512`），连接配置见 `config/base.py`（支持环境变量覆盖）。

## 测试

```bash
conda activate face_scrfd_arcface   # 均在项目根目录运行
python tests/test_trt_speed.py      # TRT/CUDA/CPU 基准 + 一致性校验
python tests/test_batch_optim.py    # 批量特征提取验证
python tests/profile_pipeline.py    # 端到端分环节耗时剖析
python tests/test_scrfd_arcface.py  # 相似度分布分析 + 识别
python tests/test_real_images.py    # 真实剧照端到端
python tests/test_seg_clothes_v2.py # 人像分割
python tests/test_siglip2.py        # SigLIP2 embedding
python tests/test_server_load.py    # 服务端模型加载链

# 智能整理（Smart Organizer，见 docs/smart_organize_blueprint.md）
python tests/test_filename_parser.py  # 文件名解析全量回归（无模型依赖）
python tests/test_burst_dedup.py      # 连拍去重选优（含景别分类）
python tests/test_face_db_build.py    # 人脸库两轮构建（需 GPU，产出 outputs/tmp_face_db）
python tests/test_actor_view.py       # 按演员视图（依赖 tmp_face_db）
python tests/test_scene_role.py       # 场景划分 + 行当 zero-shot（依赖 tmp_face_db）
python tests/test_smart_organize.py   # 端到端流水线（单剧目）
python tests/acceptance_m7.py         # 四剧目全量验收（报告 outputs/acceptance/）
```

## Docker

```bash
docker compose up -d --build   # API 端口 8198，需宿主机 NVIDIA 驱动 + nvidia-container-toolkit
```

- 权重只读挂载，`data/outputs/logs` 与 TRT 引擎缓存均为 volume 持久化
- ES 地址等参数通过 `TRANS_*` 环境变量在 compose 中覆盖（见 `docker-compose.yml`）
- 如需可视化界面：映射 `8199:8199` 并将 command 改为 `python app/server_ui.py`

## 注意事项

- 需 **NVIDIA GPU**；无 GPU 自动回退 CPU 推理（速度大幅下降）
- 服务默认使用物理 GPU1（`app/server.py` 中 `CUDA_VISIBLE_DEVICES=1`，可按需修改）
- `outputs/`、`logs/`、`weights/`、`data/` 均不入库；临时文件 `outputs/temp/` 超 1 小时自动清理
- 依赖版本以 `requirements_scrfd_arcface.txt` 为准（ES 客户端 8.x 对应服务端 8.12）
