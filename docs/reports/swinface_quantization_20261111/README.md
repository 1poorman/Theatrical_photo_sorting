# SwinFace 权重精简与量化实测

推荐 **FP16：61.8 MiB，比官方训练 checkpoint 缩小 92.65%**。同一批 109 张脸、107 个
跨原照片查询中，CPU / CUDA 的身份预测均与原模型一致，Top-1 保持 85.05%。年龄相对
原 CPU 输出的平均 / 最大偏差为 CPU **0.0033 / 0.0167 岁**、CUDA **0.0096 / 0.0517 岁**。
这是输出稳定性检查；没有真实年龄标注，年龄 MAE 仍为空。

代码优先选择 `weights/swinface/swinface_recognition_age_fp16.pt`，缺失时兼容原始文件；
`TRANS_SWINFACE_MODEL_PATH` 显式设置优先。原始权重保留，未部署到现有服务。
使用与切换方法见[指南](../../guides/swinface_guide.md)。

## 文件大小与计算方式

单位为 MiB（1,048,576 字节），以下为实际导出文件大小。

| 文件 | 字节数 | MiB | 相对原文件缩小 | 计算方式 |
| --- | ---: | ---: | ---: | --- |
| `checkpoint_step_79999_gpu_0.pt` | 882,380,545 | 841.50 | — | 官方完整模型，CPU FP32 / 官方 CUDA 混合精度 |
| `swinface_recognition_age_fp32.pt` | 129,352,905 | 123.36 | 85.34% | 无损保留身份和年龄 |
| `swinface_recognition_age_fp16.pt` | 64,832,905 | 61.83 | 92.65% | CPU 展开 FP32；CUDA 半精度参数与混合精度运算 |
| `swinface_recognition_age_int8_dynamic.pt` | 43,228,473 | 41.23 | 95.10% | CPU FBGEMM 动态 INT8 Linear，其余 FP32 |

原文件含优化器 565,653,392 字节、训练分类器 43,900,928 字节，两者约 581.3 MiB。
精简模型保留 backbone、年龄 FAM 第 1 分支、TSS 第 1 分支和 `om.output_fcs[1]`，
删除其他属性以及未被 `backbone.forward` 调用的 `backbone.head`。参数从 67,809,266
降至 32,035,547。该步骤没有剪除身份或年龄实际使用的层，CPU 与 CUDA 上均验证输出逐元素一致。

FP16 是半精度存储与计算；INT8 使用整数线性层内核。INT8 对 58 个 Linear 按输出通道
对称量化，运行时动态量化激活；卷积、归一化、softmax 和注意力矩阵乘法仍为 FP32。
已验证 PyTorch 2.9.0 / FBGEMM，仅支持 CPU；`auto` 选择 CPU，显式 CUDA 返回错误。
文件仅保存 state dict 和基础元数据，加载保持 `weights_only=True`，导出拒绝覆盖已有文件。

## 识别与年龄偏差

沿用[原始模型对比](../swinface_20261111/README.md)的输入顺序：109 张对齐锚定脸、14 位演员、
47 张原照片、107 个有效 query；排除同原照片与相同内容，Top-5 按锚定脸排序。
没有新增独立测试集。下表为 CPU batch 32，与保存的官方 CPU FP32 输出比较。

| 版本 | Top-1 | Top-5 | 验证 AUC | 同图特征最低余弦 | 年龄平均 / P95 / 最大偏差（岁） |
| --- | ---: | ---: | ---: | ---: | --- |
| 原版 / 精简 FP32 | 85.05% | 95.33% | 85.28% | 1.000000 | 0 / 0 / 0 |
| FP16 | 85.05% | 95.33% | 85.27% | 0.999998 | 0.0033 / 0.0089 / 0.0167 |
| 动态 INT8 | 88.79% | 95.33% | 85.18% | 0.965290 | 0.5509 / 1.3935 / 4.4065 |

FP16 的 107 个身份预测与原版相同，原阈值下接收决定也相同。INT8 改变了 5 个身份预测；
这批开发样本 Top-1 提升不能证明量化提高泛化能力，分数与工作点也发生了变化：

| CPU 版本 | 固定原阈值 0.559370577：正确 / 错误接收 | 重新选阈值，开发集错误接收率 ≤2%：正确 / 错误接收 |
| --- | ---: | ---: |
| 原版 / 精简 FP32 | 48.60% / 1.87% | 48.60% / 1.87% |
| FP16 | 48.60% / 1.87% | 48.60% / 1.87% |
| 动态 INT8 | 42.99% / 1.87% | 51.40% / 1.87% |

重新选择的 INT8 阈值为 0.536474705，不能直接作为部署阈值。不同精度使用新的模型签名；
INT8 应重新提取图库特征并独立校准阈值。

动态 INT8 存在批次依赖：八张分散选择的人脸，batch 1 与 batch 8 的年龄最大差异为
0.98 岁，batch 3 与 batch 8 为 2.60 岁。同一脸结果会受同批其他脸影响，因此保留为可选实验。
FP16 CPU 同项最大差异小于 0.00001 岁。

CUDA 官方模型与精简 FP32 输出完全相同。FP16 与**同一 CUDA 设备的官方模型**相比，
年龄平均 / 最大偏差为 0.0100 / 0.0385 岁；与原 CPU 输出相比最大为 0.0517 岁。
所有 CUDA 版本 Top-1 均为 85.05%，身份预测不变。原版 CUDA 在原 CPU 阈值处有一个边界
接收决定变化，说明跨设备结果也并非逐字节相同。

## 速度与显存

共享主机上的短基准：Montage Jintide C4215R，Torch 4 线程、interop 1、OpenCV 1；
CUDA 为物理 GPU1 / RTX 3090。各版本依次运行，每项热身 2 次、计时 5 次。
计时包括预处理、识别与年龄两个输出以及复制回 CPU，不含检测或文件读取。

| CPU 版本 | 单脸 p50 | batch 32 总 p50 | batch 32 每脸 p50 |
| --- | ---: | ---: | ---: |
| 精简 FP32 | 130.35 ms | 3065.25 ms | 95.79 ms |
| FP16 存储、FP32 运算 | 120.99 ms | 3137.86 ms | 98.06 ms |
| 动态 INT8 | 96.39 ms | 2426.47 ms | 75.83 ms |

INT8 在本次 batch 32 下延迟降低约 21%、吞吐提高约 26%。FP16 在 CPU 上不减少
运算位宽，不宣称 CPU 提速；五次短测不足以形成生产性能保证。

| RTX 3090 版本 | 单脸 p50 | batch 32 总 p50 | batch 32 推理期间 PyTorch 峰值已分配显存 |
| --- | ---: | ---: | ---: |
| 官方完整模型 | 14.49 ms | 28.53 ms | 770.46 MiB |
| 精简 FP32 | 14.46 ms | 28.14 ms | 632.07 MiB |
| FP16 | 14.75 ms | 29.23 ms | 564.11 MiB |

GPU 速度基本相当。FP16 模型状态张量约 61.75 MiB，但激活和工作区仍占空间，不能把
文件缩小 92.65% 解释为显存同幅度下降。以上为 PyTorch 分配统计，不等于 `nvidia-smi`
进程占用。JSON 中 `runtime_state_tensor_bytes` 是逻辑张量字节，不计量化通道参数、
内核打包、分配器和其他进程内存。

## 验证与复现

- 原模型预处理 / 完整官方前向：CPU 4 项通过。
- 新格式、禁止覆盖、错误精度 / 设备及真实重新导出与加载：CPU 6 项通过；CUDA 5 项通过、1 项 CPU 专用 INT8 跳过。
- 年龄服务 / API 8 项通过；真实默认 FP16 HTTP 70 项检查通过，三张剧照脸数 1 / 3 / 6；CPU CLI 默认 FP16 检出 3 张脸。
- 125 项只读审计通过：原文件及输入摘要、导出摘要、源码快照、预测文件、重算识别指标、年龄顺序与数值、默认配置回退。隔离服务已停止。

导出命令见[指南](../../guides/swinface_guide.md#权重精简与量化)。重新测量需选新的输出目录，工具
核对原报告、脸清单、原图、向量和官方权重摘要：

```bash
python tools/face/benchmark_swinface_quantization.py \
  --checkpoint compact_fp32=weights/swinface/swinface_recognition_age_fp32.pt \
  --checkpoint fp16=weights/swinface/swinface_recognition_age_fp16.pt \
  --checkpoint int8_dynamic=weights/swinface/swinface_recognition_age_int8_dynamic.pt \
  --device cpu --output outputs/swinface_quantization_repeat_cpu

python tools/face/benchmark_swinface_quantization.py \
  --checkpoint official=weights/swinface/checkpoint_step_79999_gpu_0.pt \
  --checkpoint compact_fp32=weights/swinface/swinface_recognition_age_fp32.pt \
  --checkpoint fp16=weights/swinface/swinface_recognition_age_fp16.pt \
  --device cuda:1 --output outputs/swinface_quantization_repeat_cuda

RUN_SWINFACE_COMPACT_REAL=1 python tests/test_swinface_compact.py
```

证据：[汇总 JSON](summary.json)、[验证记录](validation.json)。完整原始输出位于
`outputs/acceptance/swinface_quantization_20261111/`，含逐脸预测、NPZ 向量、导出记录、
HTTP/CLI、测试与源码快照。`summarize.py` 从保存数组重算指标，不重新推理。
汇总将余弦归一化两侧统一为 float64，消除原计算约 1e-8 的舍入误差；原始报告保留。

官方源码仍固定到 `ca36196814b54964e663e0991efd7d163c0069d3`，原始权重 SHA256 为
`7ba7a2dcba59500ed8a7836eea8536164ac8c30e1dbdf9211da8378cd4070d98`。
导出文件摘要见汇总 JSON；权重与原图继续留在 Git 忽略目录。
