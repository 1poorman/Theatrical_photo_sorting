# SwinFace 识别对比与年龄估计

SwinFace 已加入人脸模型对比工具，并提供独立的年龄估计功能。优先使用本地
`weights/swinface/swinface_recognition_age_fp16.pt`（61.8 MiB）；未导出时使用原始
`weights/swinface/checkpoint_step_79999_gpu_0.pt`。首次请求加载，后续请求复用。
模型源码随项目保存，推理不需要下载或安装官方训练工程。

本次 109 张锚定脸、107 个跨原照片查询中，SwinFace Top-1 为 **85.05%**，当前
ArcFace 为 **86.92%**；在开发集错误接收率 ≤2% 的工作点，正确接收率分别为
**48.60% / 82.24%**。当前识别与图库继续使用 ArcFace。
完整条件和其他候选见 [对比报告](../reports/swinface_20261111/README.md)。

## 权重精简与量化

原始文件为 841.5 MiB 的训练 checkpoint，包含优化器和训练分类器。现在可导出只保留
身份识别、年龄两个输出的推理文件；其他属性分支和未使用的分类头不再装载。
原文件不会被覆盖。以下年龄数值是同一批 109 张脸相对原 CPU 模型的**输出偏差**，
没有真实年龄标注，不能当作年龄准确率。

| 版本 | 文件大小 | 比原文件缩小 | 年龄平均 / 最大偏差（岁） | 用途 |
| --- | ---: | ---: | ---: | --- |
| 精简 FP32 | 123.4 MiB | 85.34% | 0 / 0 | 无损精简，CPU / CUDA |
| **FP16** | **61.8 MiB** | **92.65%** | **0.0033 / 0.0167** | 推荐；CPU 还原 FP32 运算，CUDA 半精度混合运算 |
| 动态 INT8 | 41.2 MiB | 95.10% | 0.551 / 4.407 | 可选实验；仅 CPU，58 个 Linear 层量化 |

FP16 在 CPU、CUDA 上的 107 个查询 Top-1 均为 85.05%，身份预测均与原模型一致。
CUDA 年龄相对原 CPU 输出的最大偏差为 0.052 岁。RTX 3090、batch 32 的 PyTorch
峰值已分配显存由原模型约 770 MiB 降至 564 MiB；GPU 速度基本相当。
FP16 在 CPU 上主要节省磁盘，模型参数会展开为 FP32。

INT8 保留卷积、归一化和注意力矩阵乘法为 FP32；当前 PyTorch 2.9 / FBGEMM 下可直接运行。
它在本次 CPU batch 32 的延迟降低约 21%，但动态激活量化会受批次组成影响。
八脸检查中，仅改变分批方式就出现最大 2.60 岁差异，因此不作为年龄功能默认。
识别分数也会变化，切换 INT8 时应重提特征并重新校准阈值。
完整指标、大小口径与复现条件见 [压缩报告](../reports/swinface_quantization_20261111/README.md)。

从官方原文件导出；目标文件已存在时拒绝覆盖：

```bash
python tools/face/export_swinface.py --precision fp16 \
  --output weights/swinface/swinface_recognition_age_fp16.pt

python tools/face/export_swinface.py --precision fp32 \
  --output weights/swinface/swinface_recognition_age_fp32.pt

python tools/face/export_swinface.py --precision int8_dynamic \
  --output weights/swinface/swinface_recognition_age_int8_dynamic.pt
```

本工作区已生成上述三个文件。`--source` 可指定其他位置的官方原 checkpoint。
部署识别与年龄功能只需要选定的推理文件；使用标准文件名时，FP16 会被自动优先选择。
修改配置后重启服务，显式路径始终优先，例如：

```bash
# 回到无损 FP32 精简版
export TRANS_SWINFACE_MODEL_PATH=weights/swinface/swinface_recognition_age_fp32.pt

# 或选择 CPU INT8
export TRANS_SWINFACE_MODEL_PATH=weights/swinface/swinface_recognition_age_int8_dynamic.pt
export TRANS_SWINFACE_DEVICE=cpu
```

INT8 的 `auto` 会选择 CPU；显式指定 CUDA 会报错。API 的 `model.storage_precision` 和
`model.inference_precision` 分别显示文件与实际运算精度，模型签名包含两者所对应的文件摘要和运行模式。

## 界面

更新代码并重启服务后，打开 `/organize`，导入照片目录，选中照片，在详情面板点击
**估计人脸年龄**。页面显示编号、对齐人脸和“约 XX 岁”；点击标注图可放大。
切换照片会保留本页已计算的结果，重新载入项目或刷新页面后可再次计算。

也可在 `/legacy/ui` 的 **人脸年龄估计** 卡片直接上传照片，无需先建立演员身份库。
预览保存到 `outputs/face_age_results/<请求编号>/`，输出根目录遵循 `TRANS_OUTPUT_DIR`。

年龄是模型对拍摄时外观的估计，妆造、遮挡、姿态和画质可能显著影响结果。
小于 40 px 的人脸显示画质提示；回归输出不在 0–120 岁内时显示“暂无法估计”，
API 保留原始回归值。当前没有真实年龄标注，尚未测量本项目的年龄 MAE。

## API

```bash
curl -X POST http://localhost:8199/api/face/age \
  -F 'image=@data/sample_images/4.jpg'

curl -X POST http://localhost:8199/api/face/age \
  --form-string 'input_dir=data/sample_images' --form-string 'name=4.jpg'
```

上传与 `input_dir`/`name` 只能选择一种。目录模式拒绝跨目录和符号链接越界。
损坏图片返回 422，缺失或不可运行的模型返回 503，无人脸返回成功及空列表。

| `data` 字段 | 含义 |
| --- | --- |
| `faces[].face_id` / `bbox` | 从左至右编号及人脸框；坐标对应返回的分析图 |
| `faces[].estimated_age` | 岁，保留一位小数；超出展示范围为 `null` |
| `faces[].age_model_output` | 未裁剪、未四舍五入的回归输出 |
| `faces[].age_status` | `estimated` 或 `out_of_range` |
| `faces[].detection_confidence` | 检测器分数，不是年龄准确概率 |
| `faces[].alignment` / `small_face` | 五点对齐或框缩放，以及是否为小脸 |
| `faces[].face_image_path` | 112×112 对齐人脸预览 |
| `annotated_image_path` | 带编号和年龄的完整标注图 |
| `image_width` / `image_height` | 分析图尺寸，最长边默认 1920，遵循 `TRANS_FACE_IMAGE_MAX_SIDE` |
| `model` | 权重摘要、源码版本、预处理、设备与批大小 |

## 命令行

```bash
python tools/face/estimate_face_age.py \
  --input data/sample_images/4.jpg --output outputs/face_age_demo --device cpu

# 目录逐图检测、估计和生成预览；子目录需加 --recursive。
python tools/face/estimate_face_age.py \
  --input data/sample_images --output outputs/face_age_batch --device auto

# 已是一张脸的对齐裁剪时，显式跳过检测。
python tools/face/estimate_face_age.py \
  --input data/face_database --recursive --aligned \
  --output outputs/face_age_anchors --device cpu
```

输出包含 `age_estimates.json` 和逐图预览。坏图记录错误并继续其余图片；有失败时退出码为 1。
`--checkpoint`、`--batch-size`、`--torch-threads` 可覆盖命令行配置。

## 配置与实现

| 环境变量 | 默认值 |
| --- | --- |
| `TRANS_SWINFACE_MODEL_PATH` | 优先 `weights/swinface/swinface_recognition_age_fp16.pt`，不存在时使用官方原 checkpoint；跟随 `TRANS_WEIGHTS_DIR` |
| `TRANS_SWINFACE_DEVICE` | `auto`：有 CUDA 用 `cuda:0`，否则 CPU；动态 INT8 始终用 CPU。也可指定 `cpu` 或 `cuda:N` |
| `TRANS_SWINFACE_BATCH_SIZE` | `8` |

CUDA 编号基于进程可见设备；现有服务默认映射物理 GPU1。显式请求不可用 CUDA 会报错。
年龄入口的 SCRFD 使用 CPU，SwinFace 使用上述设备。主视觉依赖中的 PyTorch、timm、
OpenCV 已足够，无需引入官方训练工程的 MXNet、分布式训练或数据集依赖。

采用官方 112×112 RGB、`/255 → -0.5 → /0.5` 预处理。原文件严格加载 backbone/FAM/TSS/OM
四份 state dict；精简文件严格校验格式、来源版本和保留的识别/年龄层。
年龄只运行对应分支；同时需要身份和年龄时共享一次 backbone。
使用 `torch.load(weights_only=True, mmap=True)`，保存完整权重摘要并核对加载期间没有变化。

`SwinFaceAnalyzer.extract_features(crops)` 返回 L2 归一化的 512 维向量；
`predict(crops)` 同时返回 `embeddings` 和 `ages`。SwinFace 与 ArcFace 的特征空间不同，
不能将两者向量混入同一索引或直接比较相似度。
官方来源、MIT 许可及摘要见 [源码说明](../../core_modules/face_recognition/swinface/_vendor/README.md)。
