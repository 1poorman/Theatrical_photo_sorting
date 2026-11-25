# 戏剧照片人脸检测、识别与向量检索扩展选型实验

日期：2026-11-06

## 结论

原先只比较 SCRFD 不同规模和 ArcFace 不同骨干，不能支持跨算法“最优”结论。本次扩展为 4 个检测候选、5 个单识别模型、2 个双模型融合方案，并复用 3 个检索后端实验。

对本项目“远景小脸优先、自动命名错误接收率低于 2%、在线增量图库”的目标，当前开发集上证据最强的候选组合是：

```text
最长边 1920 解码
  → SCRFD-10G-KPS 检测与 5 点定位
  → 统一对齐后生成模型原生尺寸输入
  → 0.9 × current Glint100/ArcFace + 0.1 × FaceNet/InceptionResNet-V1
  → 两个 L2 特征按 sqrt(0.9)、sqrt(0.1) 缩放后拼接为 1024 维
  → Qdrant exact cosine
  → 按 actor_id 去重，以绝对阈值、最佳/次佳演员分差和质量共同判定
```

各子目标的赢家并不相同：

- **检测召回**：SCRFD-10G 对 1095 个历史开发框保留 **99.54%**，40–79 px 小脸保留 **99.26%**；RetinaFace-R34 为 94.61%/91.86%，YuNet 为 87.58%/82.25%。这项本地口径偏向生成历史框的 SCRFD，不能替代独立人工检测真值。
- **单模型闭集排序**：FaceNet Top-1 **90.65%**、AUC **91.14%**、EER **17.99%**，优于 current ArcFace 的 86.92%/85.05%/23.46%。所以 ArcFace 不是所有指标上的最佳单模型。
- **低误接收自动命名**：各模型分别校准到错误接收率不超过 2% 后，current ArcFace 正确接收 **82.24%**，FaceNet 只有 **55.14%**；FaceNet 的少数错误匹配分数偏高。
- **综合排序融合**：70% ArcFace + 30% FaceNet 的 Top-1 **94.39%**、Top-5 **97.20%**，是已测排序最佳。
- **自动接收融合**：90% ArcFace + 10% FaceNet 在错误接收 1.87% 时正确接收 **87.85%**、拒识 10.28%，是开发集上该业务约束的最佳结果。
- **在线检索**：Qdrant exact 在真实 89×512 人脸库上 p50 **2.253 ms**、recall@10 **100%**；FAISS Flat 以 0.015 ms 更快，但不提供同等在线持久化、增量更新和 payload 过滤。

严格表述是：**SCRFD-10G + ArcFace/FaceNet 小比例融合 + Qdrant exact 是当前项目开发数据上最佳已测组合；融合权重和阈值是在同一开发集上选择的，必须经过独立数据确认后才能替换当前单 ArcFace 基线。**

## 1. 判定目标

端到端系统拆成四个独立问题：

1. 检测器能否找全多人剧照中的远景小脸并输出稳定关键点。
2. 特征模型能否抵抗侧脸、暗光、浓妆、头饰和不同场次。
3. 阈值能否在低错误接收的前提下减少拒识，而不是只提高闭集 Top-1。
4. 检索后端能否保持精确最近邻，同时支持在线更新、过滤和持久化。

本轮排序只看技术指标，优先级为：错误接收约束、小脸漏检、正确接收/Top-1、吞吐、工程复杂度。

## 2. 候选原理与覆盖范围

### 2.1 检测器

| 家族 | 核心思路 | 本轮状态 |
| --- | --- | --- |
| SCRFD | 重新分配训练样本与网络计算量，优化小脸精度—速度前沿 | 本地实测 10G、500M |
| RetinaFace | 单阶段多尺度检测，同时回归框和五点关键点 | 本地实测 R34；引用原论文/公开 WIDER FACE 数据 |
| YuNet | 极小型 CNN 检测器，面向 OpenCV/边缘 CPU | 本地实测 2023mar |
| MTCNN | P/R/O-Net 三级级联 | 原理对比；架构较老，密集小脸通常吞吐不足 |
| BlazeFace/MediaPipe | 移动端轻量 anchor 检测 | 原理对比；目标偏自拍视频而非剧照小脸 |
| CenterFace/YOLO-Face | anchor-free 或通用 YOLO 检测改造 | 原理对比；本轮没有统一来源、统一 640 输入的可复现权重 |

SCRFD 官方单尺度 VGA WIDER FACE Easy/Medium/Hard AP：500M 为 90.97/88.44/69.49，10G 为 95.40/94.01/82.80。RetinaFace 官方 R50 的 96.5/95.6/90.4 使用多尺度测试，不能直接当作与 SCRFD 单尺度数字同口径的胜负；本轮 RetinaFace-R34 实现公开的对应 Hard AP 为 88.90，也需要结合本地结果判断。

### 2.2 识别模型

| 家族 | 训练/网络特点 | 本轮状态 |
| --- | --- | --- |
| ArcFace R100/R50/MobileFaceNet | 加性角度间隔；比较三个训练包/骨干规模 | 本地实测 |
| FaceNet/InceptionResNet-V1 | 160×160 输入，VGGFace2 权重，512 维归一化特征 | 本地实测 |
| SFace | sigmoid-constrained hypersphere loss，OpenCV 128 维模型 | 本地实测 |
| ArcFace + FaceNet | 两个归一化空间的加权余弦分数融合 | 本地实测 70/30、90/10 |
| AdaFace | 根据特征范数估计质量并自适应间隔 | 论文和方法对比，未做同口径本地推理 |
| MagFace | 特征模长同时编码身份和质量 | 论文和方法对比，未做同口径本地推理 |
| CurricularFace/Sub-center ArcFace | 课程式难样本学习；多子中心抗脏标签 | 论文和方法对比，未做同口径本地推理 |

本轮不再用“没有本地实测的论文榜单”证明优于或弱于本项目模型。AdaFace、MagFace 和 CurricularFace 仍是下一批重要候选，但结论只在实际跑过的模型集合内成立。

融合实现不是对分数事后随意相加。若 ArcFace 特征为 `a`、FaceNet 特征为 `f`，二者已分别 L2 归一化，则构造：

```text
g = concat(sqrt(w) × a, sqrt(1-w) × f)
```

`g` 的模长仍为 1，其余弦相似度严格等于 `w·cos(a1,a2) + (1-w)·cos(f1,f2)`，可以直接使用普通 1024 维余弦向量库。

### 2.3 检索后端

- FAISS `IndexFlatIP`：单进程、只读、内存常驻时延最低。
- Qdrant exact：精确余弦，支持持久化、在线增删、payload 过滤和集合隔离。
- Elasticsearch `script_score`：可以精确搜索，但当前实测更慢。
- Qdrant HNSW：只有在真实人脸分布上完成 recall/误识/延迟校准后才能启用。

## 3. 实验协议

原始聚合数据见 [benchmark.json](benchmark.json)，复现脚本见 [benchmark_face_stack.py](../../../tools/face/benchmark_face_stack.py)。

### 3.1 识别

| 项目 | 口径 |
| --- | --- |
| 数据 | `data/face_database` 的 109 张演员锚定脸 |
| 标签 | 15 个“演员＋角色”目录，规范化为 14 名演员 |
| 来源 | 47 张不同原始拍摄文件 |
| 有效 query | 107；只有一个拍摄来源的演员不参加跨照片查询 |
| 防泄漏 | query 不与同一原始照片或相同内容哈希的脸比较 |
| 排序 | 剩余锚点精确余弦，报告 Top-1/Top-5 |
| 验证 | 跨来源正负对，报告 ROC AUC、EER、TAR@FAR |
| 自动接收 | 对每个模型单独扫描阈值，在错误接收率 ≤2% 下最大化正确接收率 |
| 输入 | ArcFace/SFace 使用 112×112；FaceNet 使用原生 160×160；都从同一现有锚定脸缩放 |
| 推理 | 同一主机 CPU，顺序单进程；融合速度按两个模型串行之和计算 |

实测候选：

| 名称 | 家族 | 参数 | 维度 | SHA256 前缀 |
| --- | --- | ---: | ---: | --- |
| current_glint100 | R100/ArcFace 类 | 65.16M | 512 | `4ab1d6435d63` |
| buffalo_l_r50 | R50/ArcFace | 43.59M | 512 | `4c06341c33c2` |
| buffalo_s_mbf | MobileFaceNet/ArcFace | 3.40M | 512 | `9cc6e4a75f0e` |
| opencv_sface | SFace | 9.67M | 128 | `0ba9fbfa01b5` |
| facenet_vggface2 | InceptionResNet-V1/FaceNet | 23.48M | 512 | `281cebca8662` |
| ArcFace/FaceNet 融合 | 加权拼接 | 88.64M | 1024 | 上述两个哈希 |

### 3.2 检测

使用逐脸复核包的 292 张真实戏剧剧照、1095 个历史框，统一最长边 1920、网络输入 640×640、检测阈值 0.3、预测面积至少 1600 px²，IoU≥0.5 视为匹配。

候选为 SCRFD-10G、SCRFD-500M、YuNet-2023mar、RetinaFace-R34。历史框来自此前 SCRFD 流程，因此这是**对 SCRFD 有利的回归一致性测试**，不是独立人工真值。未匹配预测也不能直接称为误检。

### 3.3 检索

复用 [Qdrant 在线向量库对比](../qdrant_20261102/README.md)：真实 89×512 gallery、20 个 query，以及 1万/10万合成扩容测试。以 FAISS 精确结果计算 recall@10，特征提取不计入检索时延。

## 4. 实验结果

### 4.1 单模型与融合识别

| 模型 | Top-1 | Top-5 | ROC AUC | EER | CPU batch=1 | batch=32 每脸 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_glint100 | 86.92% | 96.26% | 85.05% | 23.46% | 227.91 ms | 139.44 ms |
| buffalo_l_r50 | 82.24% | 91.59% | 73.92% | 32.99% | 55.40 ms | 49.86 ms |
| buffalo_s_mbf | 77.57% | 93.46% | 73.67% | 34.14% | 8.71 ms | 5.87 ms |
| opencv_sface | 76.64% | 88.79% | 78.87% | 29.43% | 12.78 ms | 11.94 ms |
| **facenet_vggface2** | 90.65% | 95.33% | **91.14%** | **17.99%** | 49.42 ms | 11.12 ms |
| **70% ArcFace + 30% FaceNet** | **94.39%** | **97.20%** | 90.11% | 19.14% | 277.33 ms | 150.56 ms |
| 90% ArcFace + 10% FaceNet | 91.59% | **97.20%** | 87.34% | 21.88% | 277.33 ms | 150.56 ms |

FaceNet 推翻了“R100 ArcFace 是已测最佳单模型”的原结论：它的 Top-1 高 3.73 个百分点，AUC 高 6.09 个百分点。70/30 融合进一步把 Top-1 提高到 94.39%，说明两种特征空间的错误并不完全重合。

### 4.2 低错误接收工作点

固定使用同一个 `0.55` 阈值跨模型没有意义，因为不同损失的分数尺度不同。下表对每个候选独立扫描开发集阈值，在错误接收率不超过 2% 时最大化正确接收：

| 模型 | 阈值 | 正确接收 | 错误接收 | 拒识 |
| --- | ---: | ---: | ---: | ---: |
| current_glint100 | 0.4280 | 82.24% | 1.87% | 15.89% |
| buffalo_l_r50 | 0.5425 | 64.49% | 1.87% | 33.64% |
| buffalo_s_mbf | 0.6246 | 39.25% | 0.93% | 59.81% |
| opencv_sface | 0.6354 | 43.93% | 1.87% | 54.21% |
| facenet_vggface2 | 0.8297 | 55.14% | 0.93% | 43.93% |
| 70% ArcFace + 30% FaceNet | 0.4714 | 86.92% | 1.87% | 11.21% |
| **90% ArcFace + 10% FaceNet** | **0.4047** | **87.85%** | **1.87%** | **10.28%** |

FaceNet 适合候选排序和人工复核，但单独用于低误识自动接收并不占优。90/10 融合只比 70/30 多正确接收 1 个 query；这个权重差异很小且来自同一开发集，不应在独立验证前固化为生产常量。

### 4.3 检测回归

| 模型 | 参数 | CPU p50 | 吞吐 | 历史框保留率 | 40–79 px | 80–159 px | ≥160 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **SCRFD-10G-KPS** | 4.23M | 87.44 ms | 11.44 图/s | **99.54%** | **99.26%** | **100%** | **100%** |
| RetinaFace-R34 | 22.11M | 162.03 ms | 6.17 图/s | 94.61% | 91.86% | 99.23% | 96.77% |
| SCRFD-500M-KPS | 0.63M | **18.25 ms** | **54.80 图/s** | 92.51% | 89.64% | 97.16% | 96.77% |
| YuNet-2023mar | 0.05M | 18.28 ms | 54.69 图/s | 87.58% | 82.25% | 96.13% | 96.77% |

在这个偏向 SCRFD 的本地回归口径下，10G 同时领先 RetinaFace-R34 4.93 个总体百分点和 7.40 个小脸百分点，且 CPU 快约 1.85 倍。RetinaFace 的公开 WIDER FACE 结果仍说明它可能在独立检测真值上更强，因此不能把本表解释成通用检测器排行榜。

SCRFD-500M 和 YuNet 速度几乎相同，但 500M 的总体/小脸保留率分别高 4.93/7.39 个百分点；在本项目轻量档中也应优先 500M，而不是 YuNet。

### 4.4 检索后端

| 真实 89×512 人脸 | p50 | p95 | recall@10 | 适用性 |
| --- | ---: | ---: | ---: | --- |
| FAISS 常驻精确 | **0.015 ms** | 0.029 ms | 100% | 单进程只读最快 |
| **Qdrant 精确** | **2.253 ms** | **2.821 ms** | **100%** | 在线增删、持久化、过滤的推荐默认 |
| Elasticsearch 精确 | 5.776 ms | 6.655 ms | 100% | 可回退，不是当前首选 |

10 万条 768 维合成向量中，Qdrant ANN p50 10.480 ms，但 recall@10 只有 44.70%；exact p50 24.213 ms、recall@10 100%。ANN 参数未经真实人脸分布校准时不能默认启用。

## 5. 推荐方案

### 5.1 准确率优先候选

1. 检测保留 SCRFD-10G-KPS，统一最长边 1920、输入 640、阈值 0.3。
2. 保存原框、五点关键点、对齐版本和来源哈希；为 ArcFace 生成 112×112，为 FaceNet 生成 160×160，query/gallery 必须完全同口径。
3. 当前 ArcFace 继续作为低误接收基线；新增 FaceNet 特征进行影子评估。
4. 独立验证若确认融合收益，则用 1024 维加权拼接向量新建 Qdrant collection；不得与既有 512 维 ArcFace 集合混用。
5. 以错误接收上限先筛阈值，再比较正确接收率；同时加入最佳/次佳演员分差和质量拒识。
6. 使用稳定 `actor_id` 作为身份主键，剧目、角色、组别作为 payload；检索后按演员去重。
7. Qdrant 保持 Cosine、`exact=true`。只有真实图库扩容实验满足 recall 和误识门槛后才启用 HNSW。

### 5.2 轻量档

SCRFD-500M + MobileFaceNet 可用于 CPU 预览或候选预筛。YuNet 与 500M 速度相近但本地小脸保留更低；SFace 比 MobileFaceNet 慢且本地 Top-1 更低，因此没有形成更优轻量组合。

### 5.3 后续必须补测

- 用独立人工 bbox 比较 SCRFD-10G、RetinaFace-R34/R50、YuNet，消除 SCRFD 历史框偏置。
- 在统一对齐、统一演员划分下加入 AdaFace、MagFace、CurricularFace/Sub-center ArcFace。
- 单独建立包含 unknown 的 validation/test，冻结融合权重、阈值和分差后再评 final test。
- 按小脸、姿态、暗光、遮挡、妆造、剧目和演员分层报告置信区间。

## 6. 证据边界

- 109 张脸只来自单一话剧，类别不均衡；两个演员只有单一拍摄来源。
- 现有锚定脸未在本轮重新人工确认统一五点对齐；FaceNet 与 ArcFace 的输入尺寸不同。
- 检测参考框来自 SCRFD，不是独立人工真值。
- 不同检测器使用相同数值阈值 0.3，但分数并未校准；正式比较应在独立真值上画完整 PR 曲线。
- 70/30、90/10 融合权重和阈值均在这 107 个 query 上选择，存在明显开发集过拟合风险。
- Qdrant 的真实向量实测是 512 维；1024 维融合集合仍需补测延迟、内存和磁盘占用。
- 当前数据已用于选型，不能再充当锁定终测集。

因此本轮证明的是“已测候选在当前开发数据上的相对表现”，不是对所有人脸模型的数学全局最优证明。

## 7. 复现

扩展对照权重均下载到 `/tmp`，不纳入仓库：

- YuNet、SFace：OpenCV Zoo 对应模型目录中的官方 ONNX。
- RetinaFace-R34：`yakhyo/retinaface-pytorch` v0.0.1 release 的 `retinaface_r34.onnx`。
- FaceNet：`timesler/facenet-pytorch` 与其 v2.2.9 VGGFace2 checkpoint。
- ArcFace R50/MobileFaceNet、SCRFD-500M：InsightFace `buffalo_l`、`buffalo_s` 模型包。

将 facenet-pytorch 安装到临时目录，不修改项目依赖环境：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python -m pip install \
  --no-deps --no-build-isolation --target /tmp/facenet_deps \
  /tmp/facenet-pytorch-master
```

然后运行：

```bash
PYTHONPATH=/tmp/facenet_deps \
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/face/benchmark_face_stack.py \
  --device cpu --timing-rounds 10 \
  --recognizer current_glint100=weights/arcface/Glint100.onnx \
  --recognizer buffalo_l_r50=/tmp/face_stack_buffalo_l/w600k_r50.onnx \
  --recognizer buffalo_s_mbf=/tmp/face_stack_buffalo_s/w600k_mbf.onnx \
  --sface-recognizer opencv_sface=/tmp/face_recognition_sface_fresh.onnx \
  --facenet-recognizer facenet_vggface2=/tmp/facenet_vggface2.pt \
  --ensemble arcface70_facenet30=current_glint100,facenet_vggface2,0.7 \
  --ensemble arcface90_facenet10=current_glint100,facenet_vggface2,0.9 \
  --detector scrfd_10g=weights/scrfd/scrfd_10g_bnkps.onnx \
  --detector scrfd_500m=/tmp/face_stack_buffalo_s/det_500m.onnx \
  --yunet-detector yunet_2023mar=/tmp/face_detection_yunet_2023mar.onnx \
  --retinaface-detector retinaface_r34=/tmp/retinaface_r34.onnx \
  --output docs/reports/face_stack_20261106/benchmark.json
```

## 8. 一手资料

访问日期均为 2026-11-06。

- [SCRFD](https://arxiv.org/abs/2105.04714)
- [RetinaFace](https://arxiv.org/abs/1905.00641)
- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [ArcFace](https://arxiv.org/abs/1801.07698)
- [FaceNet](https://arxiv.org/abs/1503.03832) 与 [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [SFace](https://arxiv.org/abs/2205.12010) 与 [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- [CurricularFace](https://arxiv.org/abs/2004.00288)
- [AdaFace](https://arxiv.org/abs/2204.00964)
- [MagFace](https://arxiv.org/abs/2103.06627)
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- [RetinaFace PyTorch/ONNX 对照实现](https://github.com/yakhyo/retinaface-pytorch)
- [Qdrant](https://github.com/qdrant/qdrant)

## 9. 商用结论说明

本轮按要求没有用商业许可问题排除任何候选，所有排名只讨论技术表现。若进入真实商业部署，仍需分别核对 InsightFace 模型包、FaceNet VGGFace2 权重、OpenCV Zoo 模型以及训练数据的授权范围；许可审计不改变本实验数值，但可能改变最终可上线模型。
