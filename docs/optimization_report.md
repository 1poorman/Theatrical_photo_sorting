# 人脸识别推理优化测试报告

> 日期：2026-08-28
> 环境：Dell 工作站，2× NVIDIA RTX 3090，驱动 580.126.09（CUDA 13.0）
> 软件栈：Python 3.10（conda `face_scrfd_arcface`）、onnxruntime-gpu 1.23.2、tensorrt-cu12 10.16.1.11、OpenCV (opencv-python)
> 模型：SCRFD `scrfd_10g_bnkps.onnx`（检测）、ArcFace `Glint100.onnx`（识别）

---

## 一、优化背景

人脸识别模块原通过 onnxruntime（CUDA EP）加载 ONNX 权重推理。为提升批量照片整理吞吐，将推理后端升级为 TensorRT，并针对剖析发现的 CPU 侧瓶颈做配套优化。

## 二、TensorRT 接入

### 2.1 版本适配

关键坑：onnxruntime 1.23.2 的 TRT EP 编译时链接 `libnvinfer.so.10`（TensorRT 10.x）。pip 默认安装的最新版 `tensorrt-cu11/12`（11.x）库为 `libnvinfer.so.11`，**二进制不兼容**，必须降级：

```bash
pip install tensorrt-cu12==10.16.1.11   # 提供 libnvinfer.so.10
```

pip 安装的 TensorRT 库不在系统 ld 路径，由 `trt_utils.preload_trt_libs()` 以 `ctypes.CDLL(RTLD_GLOBAL)` 显式预加载解决。

### 2.2 实现要点（`face_recognition/trt_utils.py`）

| 配置 | 值 | 说明 |
|------|-----|------|
| `trt_fp16_enable` | True | RTX 3090 FP16 吞吐约为 FP32 两倍 |
| `trt_engine_cache_enable` | True | 引擎缓存于 `face_recognition/trt_cache/`，首启构建约十几秒，后续秒级加载 |
| `trt_timing_cache_enable` | True | tactics 选择加速，跨引擎复用 |
| `trt_max_workspace_size` | 4 GB | |
| device 语法 | `tensorrt` / `tensorrt:N` | N 为可见设备内编号（受 `CUDA_VISIBLE_DEVICES` 影响） |

TRT EP 不可用时自动回退 CUDA/CPU 并打印警告，不影响服务可用性。

## 三、瓶颈剖析（5030×3357 测试图，缩放至 1920）

| 环节 | 耗时 | 占比 |
|------|------|------|
| `cv2.imread` 大图全分辨率解码 | 169~585 ms | **≈70%** |
| SCRFD CPU 解码+NMS | ~3.3 ms | 1% |
| SCRFD 预处理（resize+blob） | ~2.3 ms | 1% |
| SCRFD GPU 推理（TRT） | ~2.5 ms | 1% |
| ArcFace 单脸（TRT） | ~2.0 ms | 1% |

结论：端到端瓶颈不在 GPU 推理，而在 **CPU 侧图片解码**。

## 四、优化项与实测效果

### 4.1 TensorRT FP16 后端

| 模型 | CPU | CUDA | TensorRT (FP16) | TRT 提速 |
|------|-----:|-----:|----------------:|---------:|
| SCRFD 检测（640×640） | 74.5 ms | 9.75 ms | **7.40 ms** | 比 CUDA 快 24% |
| ArcFace 特征（112×112） | 215.0 ms | 4.43 ms | **1.76 ms** | 比 CUDA 快 60% |

**精度一致性**（TRT FP16 vs CUDA FP32，同一测试图 3 人脸）：
- 检测框：数量一致，平均 IoU 0.9988
- 特征余弦相似度：0.9999（FP16 正常波动）

### 4.2 大图降采样解码（最大单项收益：410→184 ms）

新增 `FaceRecognitionSystem._imread_reduced()`：借助图像头信息选择 `IMREAD_REDUCED_COLOR_2/4/8` 倍率解码，保证解码后最长边 ≥ 1920（`image_size` 参数）。因后续流程本就缩放至 1920，识别精度零损失。

| 解码方式 | 5030×3357 JPEG |
|----------|---------------:|
| `cv2.imread` 全分辨率 | 585 ms |
| `IMREAD_REDUCED_COLOR_2`（→2515×1678） | **149 ms**（4 倍） |

### 4.3 SCRFD 解码优化（`scrfd_det.py`）

原实现对全部 ~8400 个 anchor 做 bbox/kps 解码后再按分数过滤；改为**先过滤、仅解码正样本**（通常 <100 个），并补充空结果保护分支。

### 4.4 ArcFace 批量特征提取（`face_recognition.py`）

`recognize_face` 重构为两阶段：先收集全部过滤后人脸（裁剪/质量/对齐），再一次批量前向提取特征，保留逐张回退路径。

| 方式（3 张脸，TRT） | 耗时 | 一致性 |
|---------------------|-----:|--------|
| 逐张 ×3 | 6.61 ms | 基准 |
| 批量 b=3 | **2.50 ms** | 余弦相似度 1.000000 |

### 4.5 批量识别预解码流水线（`recognize_random_samples`）

`recognize_face` 新增 `image=None` 参数复用外部解码结果；批量流程以 `ThreadPoolExecutor`（4 线程）滑动窗口预解码，与 GPU 推理重叠执行（cv2 解码释放 GIL），保持处理顺序，内存多缓存 ≤4 张。

**10 张图完整批量流程**（含磁盘读取、标注图与人脸图落盘）：共 2.39 s，平均 **239 ms/张**。

### 4.6 端到端汇总

| 阶段 | 单图端到端（`recognize_face`，含解码） |
|------|--------------------------------------:|
| 优化前（CUDA EP + 全分辨率解码 + 逐张特征） | ~410 ms |
| 优化后（TRT + 降采样解码 + 批量特征 + 流水线） | **~184 ms**（**2.2 倍**） |

**识别结果验证**（`data/sample_images/4.jpg`，known_threshold=0.55）：优化前后完全一致——
1. 王文杰（饰桂枝儿）conf≈0.81（identified）
2. 彭国斌（饰雷廷昌）conf≈0.64（identified）
3. 薛天智（饰雷思起）conf≈0.53（possible）

### 4.7 修复的存量 Bug

`app/server.py` 中 `setattr(globals(), "model", ...)` 误用（对 dict 调 setattr），导致所有模型加载报 `'dict' object has no attribute ...`。已全部改为 `global` 声明 + 赋值，服务端人脸识别同步接入 TensorRT（跟随 `CUDA_VISIBLE_DEVICES` 可见设备）。

## 五、测试脚本

| 脚本 | 用途 |
|------|------|
| `tests/test_trt_speed.py` | cpu/cuda/tensorrt 三后端基准 + 结果一致性校验 |
| `tests/test_batch_optim.py` | ArcFace 批量 vs 逐张推理对比 |
| `tests/profile_pipeline.py` | 端到端分环节耗时剖析 |
| `tests/test_server_load.py` | 服务端加载模式验证（模拟 server.py） |

```bash
conda activate face_scrfd_arcface
python tests/test_trt_speed.py    # 从项目根目录运行
```

## 六、后续可选项（未实施）

- 批量模式 `imread` 与 GPU 推理进一步流水线化（当前预解码深度 4）
- TRT INT8 量化校准（检测推理 2.5ms → ~1.5ms，需校准数据）
- `build_face_database` 中 `detect_image` 与 `_detect_single` 的重复检测合并
