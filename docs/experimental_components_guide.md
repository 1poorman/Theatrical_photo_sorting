# 视觉证据、时序与图聚类实验指南

更新：2026-10-23。这里说明当前实现、实际检查和使用边界。正式验收定义仍以[核心升级蓝图](core_upgrade_blueprint.md)为准，接续状态见[工作记录](../WORK_STATE.md)。

## 当前选择

| 组件 | 实现与实际检查 | 默认选择与质量状态 |
| --- | --- | --- |
| U5 视觉证据 | 条件调用、持久请求/输出 token 预算、内容缓存、严格观察与引用校验；100 图开发集处理、15 图观察/30 次真实请求、重放 0 请求 | 用户启用视觉辅助时采用自动预算；事实质量与 Macro-F1 为 `insufficient_data` |
| U5 OCR | 按明确指定图片运行本地 PaddleOCR，校正 EXIF，保存文字、置信分与框；5 张含字资料图完成真实 CPU 识别，重建提供者全部命中缓存 | 按需使用；无独立逐字参考标注，不报告 OCR 准确率 |
| U6 变化点 | 连续窗口、最短段长、拍摄间隔、原始顺序/拍摄顺序；缺特征只影响相应连续片段，时间倒退禁用间隔证据 | `window_change_points` 为显式实验选项；默认 `legacy` |
| U7a 图聚类 | FAISS 精确互近邻、簇大小上限、成员到中心的相似度检查、同图不同检测的 cannot-link | `graph_experimental` 为显式实验选项；默认 `legacy`，身份质量未通过 |
| U7b 连拍选优 | 时间/视觉双限制、组首与相邻特征防止链式误并、同配额选择、已标记独特画面保护 | 仅离线研究，未接入默认整理选优；偏好 GT 缺失 |

U2 提速与 DALI 继续暂停。本文的预计算 CPU 测量、缓存命中或单次 HTTP 延迟均不能替代 U2 端到端性能门槛。

## 视觉证据与预算

`POST /api/organize/tasks` 的 `scene_reason` 任务可传入 `visual_policy`。现有不含这些字段的请求保持兼容；`use_llm=false` 时不调用视觉端点。

```json
{
  "kind": "scene_reason",
  "idempotency_key": "project-analysis-v2",
  "params": {
    "input_dir": "/path/to/photos",
    "knowledge_dir": "/path/to/knowledge",
    "output_file": "/path/to/results/evidence.jsonl",
    "use_embedder": true,
    "use_llm": true,
    "visual_cache_dir": "/path/to/cache",
    "visual_policy": {
      "mode": "automatic",
      "automatic_fraction": 0.30,
      "max_images": 64,
      "max_output_tokens_per_request": 1024,
      "max_batch_output_tokens": 32768,
      "candidate_margin": 0.03,
      "max_prompt_characters": 24000
    }
  }
}
```

这些是代码默认值，尚未通过独立 validation 校准。调用条件包括视觉候选不足、候选差距小、缺少参考证据，以及人物线索与视觉候选冲突。文件名人物列表仍属于线索，不变成人脸身份 GT。人工已确认记录不会触发新增观察；机器重跑仍经复核队列合并，不覆盖已确认/已拒绝记录。

自动模式同时限制图片数和实际请求数：请求上限为 `min(2 × max_images, floor(N × automatic_fraction))`，一次完整补充分析包含观察和闭集判断两次请求。因此 100 图默认至多 30 次请求，通常最多观察 15 图。小于 7 图的自动批次可能没有足够预算完成两次请求；明确按需分析使用 `mode=on_demand`，单图时在只含该图的独立输入目录中运行，完整流程上限为 2 次请求。当前工作台复选框使用自动模式；按需策略由任务 API 指定。

请求在 HTTP 之前以 SQLite 事务预留。失败、取消或进程中断不会返还已预留额度，恢复不会盲目重发。受预算管理的客户端禁用内部重试和级联，防止一次登记对应多次实际请求。输出 token 预算限制的是声明的最大生成量；输入 token 用量另存 trace，不能当成输入加输出的硬上限。过长闭集提示词单独拒绝。

观察只允许服饰、道具、布景、动作、可见文字和不确定项。格式不合法、候选越界或引用不合法时保留 unknown/候选供复核，不伪造观察。实际图片 trace 保存发送字节摘要、大小、请求/响应摘要、耗时和 token 用量，端点只留摘要。

缓存绑定原图内容、提示词、模型/端点身份、预处理、输出预算及源码版本。OCR 和视觉身份先规范化为持久 JSON 形式，重建后能正确命中。远端只有服务别名且没有经过核验的权重摘要时，视觉缓存限制在当前进程；跨进程恢复含远端调用的任务也会拒绝混用旧执行身份。当前 Qwen 端点的真实权重摘要仍未核验，不能把服务别名当成权重证明。

真实报告：[100 图预算实验](../outputs/acceptance/core_upgrade/u5_visual_budget_host_20261023/report.md)。15 次观察全部通过 schema，15 次闭集判断有效，重放 100 图结果相同且没有新增请求。其余 85 图仍有检索候选，不表示也经过逐图视觉观察。输入来自已使用的开发集，复用冻结 SigLIP2 分数；106.66 秒不包含检索编码及远端模型加载。双人事实核对数为 0，`fact_review_template.json` 中未观察图片明确留空。沙箱网络失败另存 `u5_visual_budget_sandbox_20261023/`。

## 可见文字 OCR

任务可额外指定 `ocr_images`（当前输入内的绝对路径，至多 64 张）和 `ocr_config`：

```json
{
  "ocr_images": ["/path/to/photos/program.jpg"],
  "ocr_config": {
    "python": "/path/to/paddle-env/bin/python",
    "detector_dir": "/path/to/PP-OCRv5_server_det",
    "recognizer_dir": "/path/to/PP-OCRv5_server_rec",
    "cache_dir": "/path/to/ocr-cache"
  }
}
```

使用本地 `.pdiparams` 与配置文件，运行时核对文件摘要；识别途中或缓存复用前权重变化均拒绝。通过独立 Paddle 环境中的 CPU 子进程执行，不强制普通舞台照识别，不自动下载权重。输出坐标属于校正与缩放后的预览，并记录宽高。文字仍是机器证据，`verified_by` 为空；即使读到人名，也不能由此确认照片中人脸身份。

[5 图实际检查](../outputs/acceptance/core_upgrade/u5_ocr_verified_20261023/report.md)覆盖资料头像文字和舞台资料图注。每图真实运行一次，第二个提供者全部复用缓存。缺少人工转录，结果只能证明链路、格式、内容绑定和缓存行为。

## 时序分段与连拍研究

整理任务支持 `scene_strategy=window_change_points`、`scene_order_trusted`（默认 false）和 `scene_change_config`：

```json
{
  "scene_strategy": "window_change_points",
  "scene_order_trusted": false,
  "scene_change_config": {
    "window": 4,
    "min_segment": 4,
    "penalty": 0.12,
    "max_gap_seconds": 300,
    "mode": "rehearsal",
    "sequence_order": "input"
  }
}
```

结果始终是连续片段，回到较早的画面特征会形成新片段，不把不连续照片合并成同一时间段。缺少时间时保留输入位置；时间不可信、倒退或无法比较时禁用间隔证据。`performance`/`rehearsal` 记录使用模式，当前没有应用剧情幕序先验，不能解释为正式演出必须按幕递增。纯视觉突变仍只是切分建议，尚不能区分灯光/机位变化与剧情切幕。

[262 图实际研究](../outputs/acceptance/core_upgrade/u6_u7_temporal_real_20261023/report.md)按精确内容摘要关联现有 SigLIP2 向量和 14 条相机时间线索。窗口与 KMeans 采用相同特征和时间间隔块；人物/幕次标签不参与分组。独立探索分支暂接受未核验 EXIF 线索，得到 260 个单图组和 1 个两图组；保守分支全部保留单图。没有三图以上组，无法满足 60 组连拍选优验收。相同配额检查通过，但不足以评价两种选择器的优劣；边界、首选与必须保留画面的人工标注均为 0。

`temporal_burst.py` 提供离线时间/视觉分组和同配额选择。未知时序单独保留；人工场景标签可用于生产防误并，但研究比较在分组之后才统计标签。已有 `must_keep` 记录不因低画面质量被删除；若超过配额则明确报错。当前选择器使用主体分数（如调用方提供）或本地清晰度/对比度，不伪称接入 TOPIQ/MUSIQ。

## 实验人脸图聚类

`face_cluster` 任务可传入 `clustering_backend=graph_experimental` 与 `graph_config`：

```json
{
  "clustering_backend": "graph_experimental",
  "graph_config": {
    "neighbors": 16,
    "min_similarity": 0.60,
    "min_centroid_similarity": 0.65,
    "max_cluster_size": 512
  },
  "cannot_link_exemptions": []
}
```

例外列表中的每项为 `{"image":"/absolute/input/photo.jpg","face_index":0}`，只用于已经核对的镜像/合成等例外。普通同图不同检测生成 cannot-link，合并时检查整个连通簇，避免传递合并绕过约束。图聚类保留单脸/单元素簇并全部标为待人工复核，不添加演员锚定。

[1095 脸实际向量研究](../outputs/acceptance/core_upgrade/u7_graph_real_1095_20261023/report.md)复用当前 292 图真实检测检查点，不重复检测。453 个旧簇变为 616 个图簇；4200 对“同图不同检测”假定约束的违反数为旧方案 29、图方案 0。镜像/重复检测例外尚未人工核对，不能把这些计数改称身份错认率；更保守的簇也可能增加碎簇。B-cubed、已知召回、unknown FPIR 和 AdaFace 对照仍缺独立数据。

## CPU 规模检查与复现

[10000 条预计算特征检查](../outputs/acceptance/core_upgrade/u6_u7_cpu_scale_20261023/report.md)使用固定随机种子、独立子进程和 4 线程：768 维窗口分段 1.708 秒/330.7 MiB，512 维图聚类 1.276 秒/201.7 MiB，层次聚类 21.781 秒/983.4 MiB。图/层次使用相同特征，约 17.1 倍差值仅属于这组人工生成特征上的标签计算。每方法只跑一次、CPU 共享，不能推广为真实身份质量、完整整理速度或最坏输入复杂度证明。

可重复使用的工具均要求新的空输出目录：

```bash
python tools/evaluate_visual_budget.py --study /path/to/completed/u4/backend --output /path/to/new/vision-run
python tools/evaluate_ocr_evidence.py --manifest /path/to/selected-text-images.json --python /path/to/paddle/bin/python --detector /path/to/det --recognizer /path/to/rec --output /path/to/new/ocr-run
python tools/benchmark_experimental_grouping.py --samples 10000 --output /path/to/new/scale-run
python tools/evaluate_temporal_research.py --study /path/to/completed/u4/backend --review /path/to/review/artifacts --source-root data/ncpa_test --output /path/to/new/temporal-run
python tools/evaluate_face_graph_checkpoint.py --task-db /path/to/tasks.sqlite3 --task-id TASK_ID --output /path/to/new/face-study
```

本机非 OCR 工具使用 `face_scrfd_arcface` 环境，OCR 工作子进程使用 `paddle` 环境。日志/源码摘要/输入摘要保存在各输出目录。真实照片、模型权重和人工标签不纳入 Git；报告链接依赖本地 `outputs/`。DINOv3、AdaFace、TOPIQ/MUSIQ 和其他未运行检索模型均不据此宣称完成对照或切换默认。

## 原始照片整链路检查与配置回退

`tools/check_real_workflow.py`在独立图库/照片副本和本地真实模型上验证分析、并发复核、导出、重跑锁定保护与整理。输入清单须使用原始摄影文件名，包含可解析的日期、剧场和剧目。旧首轮误用已整理副本，虽然功能计数通过，但输出缺日期/剧场，已单独记录输入范围问题；不能将该轮当作规范命名通过。新版工具在加载模型前拒绝这类清单，并逐输出核对元数据、命名、已确认标签和待复核/拒绝占位。

本地原始清单通过完整文件 SHA-256 从已确认照片对应到相同字节的原始路径；标签只供隔离的脚本复核使用，不当成新人工GT。[200图原始输入报告](../outputs/acceptance/core_upgrade/u8_raw_workflow_200_20261023/summary.json)的70项功能检查通过，保留/复制161图、明确跳过39图、0失败；4条确认与4条拒绝探针均保留正确命名，原图与正式图库不变。使用真实本地模型、`use_llm=false`，正式质量/性能门槛仍未通过，范围见[升级进度](core_upgrade_progress.md)。复现时使用新的空目录：

```bash
python tools/check_real_workflow.py --manifest outputs/acceptance/core_upgrade/resume_validation_20261023/u8_original_input_manifest.json --source-root data/ncpa_test --gallery-root data/face_database --registry outputs/acceptance/core_upgrade/verified_labels_scene_segments_v2_20261022/identity_registry.json --output /path/to/new/raw-workflow --max-images 200 --gpu 1
```

恢复已修复的默认配置时，服务设置`TRANS_SCENE_EMBEDDING_BACKEND=siglip2_base`，场景任务设置`use_llm=false`且不指定`ocr_images`，整理设置`scene_strategy=legacy`，人脸任务设置`clustering_backend=legacy`并移除实验配置字段。修改服务环境后重启服务并创建新任务，模型配置详见[核心升级进度](core_upgrade_progress.md)。连拍实验本来就只在离线工具使用。参数、源码或模型版本变化后保留原结果与人工决定，不跨版本复用检查点。原始照片命名流程与已有整理文件名的再次整理是不同输入范围，后者尚未完成命名兼容。
