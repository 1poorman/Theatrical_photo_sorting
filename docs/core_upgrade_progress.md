# 核心升级执行进度

> 更新：2026-10-21。C1–C4 已提交为 `fdc1fda`，U2 以此为 corrected baseline；本轮提交包含 U2 缓存/批量路径与 U3 界面、复核事务及场景分析后台任务。
> 验收定义见 [核心升级蓝图](core_upgrade_blueprint.md)。U0–U8 独立记录，不改变 M11–M16 的历史状态。

## 1. 当前状态

已完成 C1–C4 的首轮修复及扩展回归，生成全目录清单和修复前后复现报告，接通本地 EmbeddingGemma 2 与现有 Qwen 视觉端点。U2 已接入共享内容特征缓存与整理批量路径，哈希及线程池改进后的 200 图 × 5 次复验内存、输出一致性通过，特征吞吐仍仅 1.151 倍。复核已使用 SQLite 事务与 revision 冲突校验。**U2 性能状态为 not_passed；完整 U0–U8 尚未完成，正式模型质量和整链路性能尚未验收。**

| 阶段 | 当前实现与检查 | 尚缺验收 | 默认采用 |
| --- | --- | --- | --- |
| U0 | 清单、去重、保守划分；4 个 raw/corrected 复现各重复 5 次 | GT 核对、final test 锁定、真实 raw 流水线基准 | 不适用 |
| U1 | C1–C4 核心修复；核心回归 28/28、工作台 9/9、多模态 11/11 | 显式身份 ID/剧目角色映射、身份阈值校准 | 正确性修复已启用 |
| U2 | 分析/整理共用内容缓存；批量图文推理、有界解码、人脸批次上限；缓存 24/24、核心 28/28、多模态 11/11 | 新 200 图 × 5 次完整复验仅特征吞吐未通过；质量 GT、DALI 未验收 | 默认 8 图/批、2 解码线程；正式性能仍 not_passed |
| U3 | 缩略图缓存/ETag/限额；持久草稿、单条更新；复核事务 9/9；场景分析持久队列/检查点/取消/恢复 9/9、工作台 18/18；10000 条分页与浏览器检查 | 整理/建库逐项恢复、真实 GPU 长任务 HTTP 门槛、浏览器取消/部分失败完整验收 | 部分启用，阶段未完成 |
| U4 | Gemma 可选后端；同池 34 图/5 幕对比及 5 次编码计时 | 多剧目独立标注、质量门槛；其他候选未运行 | 保留 SigLIP2 base；Gemma 可选 |
| U5 | 图片观察 → 引用校验 → 闭集判断；1 图真实双请求通过 | 100 图双人事实核对、预算、缓存、OCR、Macro-F1 | “视觉模型辅助”选项启用 |
| U6 | 尚未实施，not_run | 变化点、DINOv3、边界 GT | 未采用 |
| U7a/U7b | 尚未进行模型/聚类/选优升级，not_run | 框级身份、连拍与选优 GT；候选对比 | 未采用 |
| U8 | 35 图浏览器流程、新旧入口、5 种屏宽通过 | U0–U3 完整验收、GPU 整链路与最终选择 | 未验收 |

## 2. 本轮实现

### 2026-10-21：U2 缓存与批量路径

- `FeatureService` 统一分析与整理的图文向量缓存。图像键使用完整内容 SHA-256，复制或改名可复用；等大小改写并恢复 mtime 仍会失效。单条向量带维度/校验和，独立临时文件原子替换；损坏缓存重算，并发写入不覆盖其他条目。
- SigLIP 缓存签名覆盖已加载权重内容、配置、实际 dtype、图像处理器、tokenizer 内容/处理规则及依赖版本。Gemma 签名覆盖本地权重/配置/tokenizer 全内容，加载前后复查。未提供签名的自定义编码器使用实例私有空间。
- 批量输出按有效路径对应；无效/缺失向量逐图回退，失败不会取消其他图片。图像解码统一执行 EXIF 方向校正，解码线程数和张量批次有上限；人脸特征提取遵守 extractor 批量上限且不超过 32。
- `/scene/reason`、整理及智能体入口使用同一后端与 `TRANS_FEATURE_CACHE_DIR`。新增内容缓存不会复用旧路径/mtime 缓存。
- `image_quality` 对 uint8 图片采用 int16 Laplacian 与 OpenCV 统计，保持亮度边界和原质量公式。历史 292 图数值对照完全一致，单次统计阶段约 7.38 倍；这只是已解码图像的探索计时，不代表完整整理提速。
- 基准保存源文件摘要、原图清单、人脸权重/库摘要、驱动与同卡进程记录，使用隔离目录并保留「整理完成」解析语义。输出目录必须为空，避免覆盖已有测量。

24 图 × 2 次探索：2 解码线程的特征阶段约 1.16 倍、RAM 峰值约 1.22 倍；4 线程约 1.52 倍但 RAM 约 2.01 倍，未采用。两组图文最低余弦均 ≥0.9999998，热缓存前向为 0。完整对照见第 3 节，未达性能门槛。其后将解码线程池改为编码器生命周期内复用，并提供 `close()`，单列两次特征复测，不将旧版性能数字归到新版本。

接续复测 4 个解码线程，200 图 × 2 次特征：中位 110.389 / 86.867 秒，1.271 倍；RSS 峰值增加 32.72%，仍未通过，保留 2 线程默认。两次耗时有明显波动，不将它与早先不同时间的 2 线程报告直接当作受控比较。报告位于 `u2_pool4_200x2_20261021_gpu/`。

profiling 显示 GPU 前向仅约 2 秒，CPU 解码/预处理与完整内容哈希占主导。将冷缓存图像从 3 次完整哈希改为查找与发布各 1 次；所有批次结束后才校验并发布新向量，补齐「后续批次改写前面原图」和「冷/热缓存仍完整校验内容」回归，缓存 24/24 通过。`u2_publication_200x5_20261021/` 独立 200 图 × 5 次完整复验已结束，被测源码摘要匹配；结果见第 3 节。

### 2026-10-21：U3 首批工作台响应改进

- 缩略图使用完整源内容、EXIF 规则、尺寸、JPEG 配置和 Pillow 版本作为键，复制/改名复用；命中不解码原图，仍校验源内容。同大小改写/恢复 mtime 后更新 ETag，条件请求返回 304，生成期间改写返回 409 且不发布。
- 缓存条目含 JPEG 校验和，损坏重建；唯一临时文件原子替换。发布和容量清理由跨进程文件锁串行化，默认 256 MiB / 4096 项，只清理新缓存目录中符合命名的普通条目。测试覆盖复制、损坏、并发、容量与越界链接。
- 标签/备注草稿按照片路径与复核目录存入当前浏览器；切图、搜索、刷新、保存失败保留，成功后清除，并恢复选中照片。草稿不会自动提交为人工决定。
- 复核决定后用接口返回的单条记录更新本地状态和当前项目统计，无额外全队列下载；初次载入仍分页读取队列。10000 条模拟记录验证完整分页、单页 ≤36 卡片及决定后 0 次列表 GET。
- 复核以 `review.sqlite3` 为权威存储，记录与不可变修订日志一次事务提交；迁移前保存按完整 SHA-256 命名的原 JSONL 副本。兼容快照原子替换，提交后中断会在下一次访问时恢复。索引分页与摘要来自同一读事务。
- 工作台提交 `expected_revision`，同版本并发修改恰好一个成功，另一个 409 后只刷新该记录并保留草稿。旧客户端可省略 revision，仍为串行最后写入，不能获得冲突保护。
- 复核存储 9/9 覆盖线程/进程并发、提交前后中断、事务回滚、迁移、历史不可变及 10000 条分页；工作台接口 16/16、核心 28/28、多模态聚合 11/11，35 张真实照片、实际 409 冲突与五种屏宽浏览器流程通过。GPU 整理仍使用替代响应，后台任务及 U3 HTTP 延迟门槛未完成。

35 张原图 × 5 次缩略图探索保存于 `u3_thumbnail_20261021/summary.json`：640 像素预览的旧实现、冷缓存、热缓存中位耗时分别为 4.074 / 5.101 / 1.083 秒；JPEG 字节全部一致，热缓存原图解码为 0。内容校验及持久化使首次生成更慢，缓存命中约 3.76 倍；本次为单剧目、文件系统已预热的函数调用，不证明 HTTP 并发或 U3 延迟门槛通过。

10000 条合成复核记录的函数计时位于 `u3_review_10000_20261021/summary.json`：5 次遍历全部 50 页中位 0.121 秒，单次决定（含全量 JSONL 兼容快照）0.146 秒，摘要 0.001 秒。该报告源码摘要早于初始读分支补 `BEGIN` 的一致性修复，不能视作当前源码或长任务 HTTP 并发验收；每次决定仍生成全量兼容快照。

### 2026-10-21：U3 场景分析后台任务

- 新增 `/api/organize/tasks` JSON 创建/列表、单任务查询、`items` 检查点分页及 `cancel` / `resume`。当前只支持 `scene_reason`。队列默认容量 8（含执行项）、一个工作线程；SQLite 保存输入清单、完整图片 SHA-256、知识快照和逐图结果。独占文件锁防止第二服务进程抢占任务；初始化把遗留执行中状态改为 interrupted，等待显式恢复。
- 创建不执行模型，仍需枚举并完整哈希图片，大量高分辨率原图的创建耗时尚待验收。`idempotency_key` 相同请求返回既有任务，不同输入 409；同输出的未完成任务拒绝并发覆盖。
- 每张结果持久提交后再开始下一张；恢复跳过成功和已记录失败项。已完成图片修改后拒绝发布旧结果；未完成图片修改作为该图结构化失败，其他图片继续。单图失败写 unknown/error，最终为 completed_with_errors。原子输出发布可重复执行，不自动确认标签或添加人脸锚定。
- 取消请求保存 cancel_requested；当前单图操作返回后记录结果并停止，最终为 cancelled。取消/中断可继续剩余图片，失败项需新任务重试。若进程在操作结束、检查点提交前崩溃，该项仍可能重算；不承诺外部模型调用恰好一次。
- 场景旧同步 API 与智能体使用同一分析执行器；同步分析、整理、知识构建和受限智能体路由移到线程池。分析/整理/建库共享串行业务锁，暂只对分析提供持久恢复；原聚类扫描仍沿用旧进程内线程。
- 工作台保存分析 task_id，刷新只查询既有任务，运行时可继续看图复核。展示成功/失败/待处理数及取消、继续、重新分析操作；完成后载入复核，保留人工决定。35 图浏览器验证刷新不重复提交、草稿、实际 409、10000 条分页及五屏宽。
- 任务核心 9/9、工作台 18/18、复核事务 9/9、核心 28/28、多模态 11/11 通过。任务检查覆盖真实子进程崩溃、幂等并发、队列限额、工作进程锁、取消/恢复、单图失败及发布失败后不重新推理。

可控慢任务 HTTP 测量位于 `u3_http_controlled_20261021_final/summary.json`，源码摘要与当前被测文件匹配：5 客户端、60 秒、合计 5 请求/秒，10000 条复核记录，每次分页校验 200 条与统计；1000 张生成小图逐项人为等待 250 ms。健康/分页 P95 为 3.71 / 17.40 ms，创建 P95 83.48 ms，取消确认 25.57 ms，查询错误为 0。该报告仅验证隔离的任务与复核路由调度，不包含真实 GPU、生产完整中间件或大图输入哈希；5 次创建中 4 次为幂等重复，不能证明持续唯一任务入队性能。U3 正式长任务门槛仍未验收。

### 2026-10-19：正确性与模型探索

- **C1**：排序、分段前按原始路径建立特征映射；缺失、非有限或维度不齐的特征只停用所在段的细分。单图特征异常不再中断其他图片，EXIF 文件句柄正确关闭。
- **C2**：视觉检索覆盖全部幕次，再融合词法先验；单个文本编码失败仍保留词法候选。输出独立 lexical/visual 分数及降级原因，区分缺失特征与低相似度。
- **C3**：逐脸合并同演员多个角色目录的锚定分，取最佳身份并检查第二名差距，多人结果按演员去重。当前仍以目录演员名作为兼容身份键，并保留最佳锚定目录的展示名称；正式 identity_id、当前剧目角色绑定和阈值校准待补。margin 暂沿用现有配置 `0.05`，不声明框级质量达标。
- **C4**：pending 更新候选并保留版本关系；approved/rejected 保持原样，包括缺少 reviewer 的旧记录。裸文件名首次绑定完整路径后写入 `image_bindings.jsonl`，新实例也拒绝其他目录的同名关联。导出使用绑定路径，整理按完整路径应用标签。
- **人工与分数**：人工修改/拒绝不沿用机器分数，原分数存为 `machine_confidence`；人工自定义标签保留。工作台改称“证据分”，注明不是正确概率。
- **批量回退**：修复 `ImageEmbedder.extract_features_batch` 失败时把 tensor 当路径的错误，逐图回退保持向量与路径对应。
- **视觉请求**：实际部署身份来自用户配置，为 Qwen3.8-27B；请求保留 Qwen3.6-27B 别名，trace 记录身份来源。发送经 EXIF 校正的真实预览图，校验观察 schema 与幕次引用。

真实整链路首次检查发现第二次请求耗尽 2048 completion tokens，返回 `content: null`，客户端计算响应摘要时触发 `AttributeError`。现将空正文视为协议失败，并对该部署的结构化视觉/闭集请求默认设置 `chat_template_kwargs.enable_thinking=false`。复验两请求分别输出 264、229 tokens，均正常结束，总耗时 7.215 秒；结果仍为 pending。此结果只证明链路可用，不证明幕次正确。

## 3. 数据与实测

报告位于 [`outputs/acceptance/core_upgrade/first_pass/`](../outputs/acceptance/core_upgrade/first_pass/)，留在本地。原图、正式人脸库和 `.env` 未修改。

### U0 清单和复现

扫描 **582 个文件、319 张唯一解码图像**。按目录规则暂分为 292 张舞台照、27 张资料图，分类待人工核对；解码失败为 0。历史“555 张”不能作为独立样本数。

| 剧目 | 唯一舞台照 | 有文件名幕次弱标签 |
| --- | ---: | ---: |
| 天下归心 | 35 | 34 |
| 假面舞会 | 119 | 118 |
| 马可·波罗 | 59 | 59 |
| 样式雷 | 79 | 49 |

`manifest.jsonl` 记录内容/像素哈希、方向、来源和待核对标签。近重复采用 dHash、宽高比、平均颜色的启发式检查；318 个检测组在 `splits.json` 中跨集合数为 0。整剧目与检测到的重复组同集合，**不保证所有近重复已找出**。划分为 provisional，final test 未锁定；正式幕次、框级身份、连拍和边界核对数均为 0。

`baseline_raw.json` 保存 `391ef98` 的 4 个错误行为，`baseline_corrected.json` 保存修复后的 4/4 行为通过。各重复 5 次且行为稳定；使用合成向量/EXIF/聚类替身，`real_model_quality` 和 `end_to_end_performance` 明确为 `not_run`，不是完整流水线性能基线。

### U2：200 张原图的真实流水线对照

[`u2_pipeline_200x5_20261021/`](../outputs/acceptance/core_upgrade/u2_pipeline_200x5_20261021/) 保存两方案原始报告、日志、向量、归档记录和摘要。以 `fdc1fda` 为 corrected baseline；4 剧目、8 原目录、200 张内容唯一图片，分辨率约 880 万–3615 万像素；5 条幕次文本。各方案独立进程预热，特征缓存每次清空，重复 5 次，真实 SigLIP2/SCRFD/ArcFace，人脸库使用隔离副本。景别和行当分类未启用，图片分类及场次分组仍为 provisional。

| 指标 | corrected 逐张 | U2 批量（线程池修复前） | 结论 |
| --- | ---: | ---: | --- |
| 特征阶段中位耗时 | 79.507 秒 | 64.667 秒 | 1.229 倍，未达 1.5 倍 |
| 完整整理中位耗时 | 230.060 秒 | 157.612 秒 | 减少 31.49%，达到 ≥20% |
| 特征阶段峰值 RSS | 2,352,615,424 bytes | 4,664,315,904 bytes | 1.983 倍，超标 |
| 完整整理峰值 RSS | 3,286,765,568 bytes | 7,390,732,288 bytes | 2.249 倍，超标 |

200 图与 5 文本特征最低余弦分别为 `0.99999988`、`1.0`，全部 ≥0.999。每轮保留 154 图，两方案的归档名称、识别演员、场景划分、质量分数完全一致，处理及人脸错误均为 0。独立新服务热缓存及「原路径分析→隔离目录整理」热缓存均为图文前向 0；热缓存整理输出保持一致。新增显存峰值约占初始空闲容量的 13.5%/14.7%，无 OOM。

结果为 **performance_status=not_passed**，失败项为 `feature_speedup` 与 `rss_increase`。这是当前所述整理配置的性能和行为一致性测量，不证明幕次/身份 GT 正确，亦不包含 HTTP、UI、DALI、权重下载及冷启动。完整 5 次对照的源码摘要保存在 `provenance.json`，与随后线程池修复版本明确区分。

线程池复用后的 [`u2_pool_200x2_20261021/`](../outputs/acceptance/core_upgrade/u2_pool_200x2_20261021/) 单列 200 图 × 2 次特征复测：baseline 中位 82.492 秒，批量 70.669 秒，1.167 倍；峰值 RSS 为 2,362,568,704 / 2,596,990,976 bytes，仅增加 9.92%。图文特征一致性与热缓存前向 0 仍通过。复用线程池显著减少本次特征阶段的内存累积，但未达到吞吐门槛；这份报告不含完整整理、只有 2 次，不能计为正式 U2 通过。

### U2：哈希与线程池改进后的完整复验

[`u2_publication_200x5_20261021/`](../outputs/acceptance/core_upgrade/u2_publication_200x5_20261021/) 使用同一 corrected baseline、200 张唯一图/4 剧目/8 目录、5 文本及各方案 5 次独立进程测量。

| 指标 | corrected 逐张 | 当前 U2 批量 | 结论 |
| --- | ---: | ---: | --- |
| 特征阶段中位耗时 | 91.750 秒 | 79.689 秒 | 1.151 倍，未达 1.5 倍 |
| 完整整理中位耗时 | 625.165 秒 | 198.152 秒 | 本次观察减少 68.30% |
| 特征阶段峰值 RSS | 2,362,388,480 bytes | 2,611,617,792 bytes | 增加 10.55%，通过 |
| 完整整理峰值 RSS | 3,285,966,848 bytes | 3,470,487,552 bytes | 增加 5.62%，通过 |

图像/文本最低余弦为 `0.99999988` / `1.0`；10 轮均保留 154 图，归档决定和质量分数一致。热缓存及分析→整理缓存图文实际前向均为 0，处理/人脸/特征错误为 0，新增显存占初始空闲容量约 16.0% / 22.9%，无 OOM。**performance_status=not_passed，仅 feature_speedup 未通过。**

两个方案顺序测量，同卡有其他工作；完整整理 baseline 范围 490.690–691.980 秒，batch 159.967–227.851 秒。68.30% 是这次共享 GPU 条件下的观察值，不能归因成隔离负载下的可靠代码收益，也不能和早先 31.49% 直接组成受控版本比较。正式模型质量仍需要独立核对 GT。

### U4 召回探索

同一剧目、34 张有唯一幕次弱标签的照片、5 幕原文文本；排除 1 张没有唯一剧情幕次标签的照片。中性文件名，清除 EXIF/ICC 元数据；两个后端 manifest/text hash 一致。

| 后端 | Recall@1 | Recall@3 | 5 次编码耗时中位数 | PyTorch 峰值 allocated 显存 |
| --- | ---: | ---: | ---: | ---: |
| corrected SigLIP2 base | 4/34（11.8%） | 23/34（67.6%） | 0.661 秒 | 1,698,467,840 bytes |
| EmbeddingGemma 2 | 10/34（29.4%） | 28/34（82.4%） | 2.737 秒 | 1,586,116,096 bytes |

硬件 RTX 3090，PyTorch 2.9.0+cu130。SigLIP2 使用 Transformers 5.15.0；Gemma 使用独立依赖中的 Transformers 5.20.1、SentenceTransformers 6.1.0。计时为**进程预热、无特征缓存的预览图与文本编码**，不含原图清单生成、人脸、决策、归档或下载；不同依赖和模型预处理均已注明。

Gemma 在探索样本上多召回 5 张，但 Recall@3 仍未到蓝图 90% 目标，耗时增加；单剧目弱标签未经独立核对，质量状态为 **insufficient_data**。继续作为可选后端，不替换已修复的 SigLIP2 默认。不能与历史不同文本/文件名条件的结果直接比较。

`embedding_smoke.json` 另记录 3 张真实图的 768 维 BF16 编码：批量/逐张最低余弦 `0.999929`，新检索器命中缓存后图像/文本编码计数均为 0。这不等于 U2 全量缓存、吞吐门槛通过。

## 4. 复现与配置

在项目根目录执行；模型、浏览器及部分 TestClient 检查需要能访问目标机设备、线程和网络的环境。

```bash
# 清单和四项缺陷对照（原图只读）
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/build_core_upgrade_manifest.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/characterize_core_correctness.py

# 核心 28 项、工作台 18 项、复核事务/后台任务各 9 项、多模态 11 个检查脚本
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_core_upgrade.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_workspace.py -v
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_review_store.py -v
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_core_upgrade_tasks.py -v
# HTTP 调度探索（使用新的空目录；不替代真实 GPU 验收）
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/benchmark_task_responsiveness.py --output outputs/acceptance/core_upgrade/u3_http_new_run
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/acceptance_multimodal_m16.py

# U2：缓存契约、统计等价和真实流水线；基准输出必须使用新的空目录
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_feature_cache.py -v
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_quality_statistics.py -v
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/benchmark_feature_pipeline.py --output outputs/acceptance/core_upgrade/u2_pipeline_200x5_20261021 --limit 200 --repeats 5

# 两后端依次运行，默认 repeats=5
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/evaluate_core_upgrade.py --backend siglip2_base --device cuda:0
PYTHONPATH=outputs/core_upgrade_deps /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/evaluate_core_upgrade.py --backend embeddinggemma2 --device cuda:0
PYTHONPATH=outputs/core_upgrade_deps /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/check_multimodal_upgrade.py --component embedding --device cuda:0
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/check_multimodal_upgrade.py --component vision

# 先启动验收服务，再在另一终端运行浏览器检查
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python -m uvicorn server_ui:app --app-dir app --host 127.0.0.1 --port 18301 --lifespan off
WORKSPACE_URL=http://127.0.0.1:18301 node tests/test_workspace_browser.cjs
```

依赖版本见 [`requirements_embeddinggemma.txt`](../requirements_embeddinggemma.txt)。本机已在 `outputs/core_upgrade_deps` 验证；其他机器应在独立环境安装，不能只修改旧 Transformers 环境中的后端名称。

| 配置 | 默认/用法 |
| --- | --- |
| `TRANS_SCENE_EMBEDDING_BACKEND` | 默认 `siglip2_base`；设为 `embeddinggemma2` 并加载对应依赖可启用新后端 |
| `TRANS_FEATURE_CACHE_DIR` | 默认 `outputs/feature_cache`；分析与整理共用，向量位于 `features-v2/` |
| `TRANS_FEATURE_BATCH_SIZE` | 默认 `8`，限制服务每批图像/文本数量 |
| `TRANS_FEATURE_DECODE_WORKERS` | 默认 `2`，限制并发原图解码数量 |
| `TRANS_THUMBNAIL_CACHE_DIR` | 默认 `outputs/thumbnail_cache`，缩略图内容缓存 |
| `TRANS_THUMBNAIL_CACHE_MAX_BYTES` | 默认 `268435456`（256 MiB），仅清理新缓存条目 |
| `TRANS_THUMBNAIL_CACHE_MAX_ENTRIES` | 默认 `4096` |
| `TRANS_EMBEDDINGGEMMA_MODEL_PATH` | `/models/hch/embeddinggemma-2` |
| `TRANS_EMBEDDINGGEMMA_DEVICE` | `cuda:0`；支持时 BF16，否则 FP32，禁止 FP16 |
| `TRANS_VISION_MODEL_IDENTITY` | `Qwen3.8-27B`；请求名称仍取 `.env` 的 `big_model_name` |
| `TRANS_VISION_IMAGE_MAX_SIDE` | `1280` |
| `TRANS_VISION_ENABLE_THINKING` | 默认 `false`；启用后需另验 token 预算和延迟 |

浏览器检查覆盖 35 张真实照片、导入、离线分析、复核、导出、刷新与 1440/1024/768/540/390 宽度。GPU 整理、聚类动作在浏览器测试中拦截，仅验接口交互；不能计为 U8 GPU 整链路通过。

## 5. 下一步

1. 核对场次、资料分类、弱标签和近重复，再锁定独立 final test；现有舞台照数量基本够幕次初评，主要缺口是可信标注。
2. 补齐身份 ID 与剧目角色映射，隔离 gallery/query 并校准 margin；保存真实 raw/corrected 流水线 5 次基准。
3. U2：完整复验已结束且仍 not_passed；继续降低 CPU 预处理成本，保持完整内容改写失效和路径对应契约。正式质量核对和 DALI 实验另行记录。
4. U3：分析已接入有界持久队列、检查点、取消/恢复；补整理/建库逐项恢复、真实 GPU 长任务 HTTP 响应与浏览器取消/部分失败验收。初次载入仍下载分页队列，兼容快照每次决定仍全量生成。
5. 扩展多剧目检索和 U5 预算/事实质量评估，再进入 U6/U7；未达门槛组件保留实验状态。

新增测试已加入 `.gitignore` 例外。每阶段更新命令、代码版本、报告和质量状态，禁止将合成复现、小样本 smoke 或历史结果计为正式效果通过。
