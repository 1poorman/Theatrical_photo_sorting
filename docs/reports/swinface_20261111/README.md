# SwinFace 识别对比与年龄估计验证

日期：2026-11-11。基线协议来自 [11 月 6 日跨模型实验](../face_stack_20261106/README.md)。

**SwinFace 已完成本地实测与年龄功能接入。它没有超过现有 ArcFace 的 Top-1 和低错误接收
工作点，当前识别默认继续采用 ArcFace。** 年龄功能通过真实照片、CPU/GPU 一致性、HTTP
与界面检查；项目没有真实年龄标注，年龄准确率尚未验收。

## 模型与协议

- 仓库：[lxq1000/SwinFace](https://github.com/lxq1000/SwinFace)，提交
  `ca36196814b54964e663e0991efd7d163c0069d3`。
- 用户提供的官方权重：`weights/swinface/checkpoint_step_79999_gpu_0.pt`，882,380,545 字节。
- 本地 SHA256：`7ba7a2dcba59500ed8a7836eea8536164ac8c30e1dbdf9211da8378cd4070d98`。
- Swin-T 多任务模型 67.81M 参数，backbone 为 28.90M，身份向量 512 维。严格加载四个
  推理组件，保留 MIT 许可。年龄来自回归头，不是 `Young` 属性分类。

重新运行六个单模型及两种既有融合。数据仍为同一话剧的 109 张锚定脸、15 个演员＋角色
目录、14 位演员、47 个原照片来源，107 个 query 具备跨照片正样本。同一原照片和相同内容
哈希均排除在对应 gallery 之外。Top-5 沿用按锚点排序的历史口径，各模型单独校准阈值。

已有裁剪缩放到模型原生尺寸：SwinFace/ArcFace/SFace 为 112×112，FaceNet 为 160×160。
没有重新核验全部裁剪的五点对齐，也没有更换检测器或新增检测真值。

同一主机 CPU 顺序执行，10 次计时，PyTorch 4 线程，ONNX 使用运行时默认线程。
框架线程策略不同，没有声称 CPU 资源完全受控。SwinFace 识别只运行 backbone，
识别＋年龄另列；精简路径与官方完整前向进行了数值核对。

## 识别结果

| 模型 | Top-1 | Top-5 | ROC AUC | EER | CPU 单脸 p50 | batch=32 每脸 p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前 Glint100 / ArcFace | 86.92% | 96.26% | 85.05% | 23.46% | 224.01 ms | 160.93 ms |
| Buffalo-L / R50 | 82.24% | 91.59% | 73.92% | 32.99% | 74.83 ms | 50.45 ms |
| Buffalo-S / MobileFaceNet | 77.57% | 93.46% | 73.67% | 34.14% | 9.98 ms | 5.48 ms |
| OpenCV SFace | 76.64% | 88.79% | 78.87% | 29.43% | 13.63 ms | 13.77 ms |
| FaceNet / VGGFace2 | 90.65% | 95.33% | 91.14% | 17.99% | 48.94 ms | 19.03 ms |
| **SwinFace** | **85.05%** | **95.33%** | **85.28%** | **22.34%** | **111.58 ms** | **90.62 ms** |
| 70% ArcFace + 30% FaceNet | 94.39% | 97.20% | 90.11% | 19.14% | 272.95 ms¹ | 179.96 ms¹ |
| 90% ArcFace + 10% FaceNet | 91.59% | 97.20% | 87.34% | 21.88% | 272.95 ms¹ | 179.96 ms¹ |

¹ 融合速度按两个模型串行耗时相加估算，非直接服务测量。

| 模型 | 校准阈值 | 正确接收 | 错误接收 | 拒识 |
| --- | ---: | ---: | ---: | ---: |
| 当前 ArcFace | 0.428002 | 82.24% | 1.87% | 15.89% |
| Buffalo-L | 0.542544 | 64.49% | 1.87% | 33.64% |
| Buffalo-S | 0.624617 | 39.25% | 0.93% | 59.81% |
| SFace | 0.635408 | 43.93% | 1.87% | 54.21% |
| FaceNet | 0.829662 | 55.14% | 0.93% | 43.93% |
| **SwinFace** | **0.559371** | **48.60%** | **1.87%** | **49.53%** |
| 70% ArcFace + 30% FaceNet | 0.471410 | 86.92% | 1.87% | 11.21% |
| 90% ArcFace + 10% FaceNet | 0.404737 | 87.85% | 1.87% | 10.28% |

SwinFace Top-1 正确 91/107，ArcFace 为 93/107，FaceNet 为 97/107。在错误接收最多
2 个 query 时，SwinFace 正确接收 52 个，ArcFace 为 88 个。SwinFace 的 AUC/EER 略优，
但没有转化为更好的自动命名工作点。

错误接收率以 107 个**已知身份** query 为分母，不是 unknown 开放集 FAR。阈值和融合权重
使用开发数据选择，不能直接作为生产阈值或独立终测结论。

## 年龄与工程验证

- 109/109 张锚定脸产生有限年龄输出，范围 11.24–49.00 岁，中位数 31.64 岁。
  这是预测分布，**不代表真实年龄范围或估计准确**；GT 数量为 0，MAE 为 `null`。
- CPU 单脸识别＋年龄 p50 为 114.00 ms，batch=32 每脸 95.02 ms，比仅识别增加约
  2.42 ms / 4.40 ms；包括预处理与输出回传，不含检测及 HTTP。
- 八张裁剪的精简/官方完整前向对照：CPU 最大特征差 `1.25e-7`、年龄差 `3.81e-6` 岁；
  CUDA 为 `1.54e-4` / `0.00890` 岁。CUDA 保留官方 attention 的 autocast，存在跨批浮点误差。
- 8 项年龄服务/API 回归、CPU 与 CUDA 各 4 项模型检查、27 项现有工作台回归通过。
- 真实 HTTP 三张剧照返回 1/3/6 张脸，91 项检查通过，包括坐标、预览、上传/路径一致、
  坏图、无人脸、原图摘要与加载期间健康检查。隔离 CPU 服务首次请求 6.71 秒，期间健康
  请求最大 23.13 ms；这只是功能观察值。
- 35 图、五种屏宽、万条复核队列的浏览器验证通过；新增年龄按钮、数值、不可估计、
  放大标注和无人脸交互通过。浏览器测试替代年龄响应，真实模型 HTTP 另行验证。
- CLI 对 `data/sample_images/4.jpg` 实测返回 3 张脸，生成 JSON、对齐裁剪和标注图。
- 从保存的向量重算全部识别指标并核对 109 张原脸、六份权重及所有向量，148 项审计通过。

操作方法见 [使用指南](../../swinface_guide.md)。

## 证据与复现

精选证据：[benchmark.json](benchmark.json)、[CPU 一致性](official_parity_cpu.json)、
[CUDA 一致性](official_parity_cuda.json)、[验证汇总](validation.json)。原始产物在 `outputs/acceptance/swinface_20261111/`：
`benchmark.json` 含逐脸年龄，`recognition/` 保存源文件摘要和全部向量，`http_workflow.json`
含真实接口返回，`recognition_audit.json` 含重算及摘要校验。

```bash
PYTHONPATH=/tmp/facenet_deps python tools/benchmark_face_stack.py \
  --device cpu --torch-threads 4 --timing-rounds 10 \
  --recognizer current_glint100=weights/arcface/Glint100.onnx \
  --recognizer buffalo_l_r50=/tmp/face_stack_buffalo_l/w600k_r50.onnx \
  --recognizer buffalo_s_mbf=/tmp/face_stack_buffalo_s/w600k_mbf.onnx \
  --sface-recognizer opencv_sface=/tmp/face_recognition_sface_fresh.onnx \
  --facenet-recognizer facenet_vggface2=/tmp/facenet_vggface2.pt \
  --swinface-recognizer swinface=weights/swinface/checkpoint_step_79999_gpu_0.pt \
  --ensemble arcface70_facenet30=current_glint100,facenet_vggface2,0.7 \
  --ensemble arcface90_facenet10=current_glint100,facenet_vggface2,0.9 \
  --artifacts-dir outputs/swinface_comparison/recognition \
  --output outputs/swinface_comparison/benchmark.json

RUN_SWINFACE_REAL=1 python tests/test_swinface.py
python tests/test_face_age.py
```

其余模型与 FaceNet 临时依赖沿用前一报告的来源和安装方法；仅保留 `--recognizer` 与
`--swinface-recognizer` 可复现 ArcFace/SwinFace 对照。本次未将新向量写入现有 ArcFace 索引。
