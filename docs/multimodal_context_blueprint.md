# 多模态剧照整理扩展蓝图

> 版本：v1.0（2026-09-14）  
> 适用范围：在 `docs/smart_organize_blueprint.md` 已完成 M1–M10 的基础上，加入官方剧目信息、剧情文本、花絮图文和本地大模型，提升剧照的幕次/场景标记，并为可审计的高级智能体提供基础。

## 1. 目标与边界

### 1.1 目标

将单张剧照的“看图猜场景”改为可追溯的证据融合：

```
官方网页/离线快照
  → 剧情、幕次、角色、花絮图文知识库
剧照
  → EXIF/文件名 + 人脸/角色/景别 + 图像描述
  → 候选场景召回
  → 本地视觉语言模型结构化判定
  → 人工复核/纠错
  → scene_labels.json、规范文件名、检索索引
```

系统输出必须能回答“为什么标为第 N 幕”：至少保留一个文本证据、一个图像证据或明确的人工决定，并记录来源和模型版本。

### 1.2 本期范围

- 支持从 `剧目信息.txt` 读取官方 URL；当前样例为《天下归心》：`https://www.chncpa.org/subsite/txgx201410/index.html`。
- 抓取 HTML、正文、标题、图片 URL、图片说明和发布时间；保存原始快照，网络不可用时从快照运行。
- 解析幕次/场景/角色/剧情事件/花絮，并建立剧目级知识库。
- 为剧照生成视觉描述、人物/角色候选和场景候选；与剧情段落进行跨模态检索。
- 调用 `.env` 中 OpenAI 兼容的本地模型进行受约束 JSON 判定；小模型优先，失败时由 27B 模型兜底。
- 产出人工审核队列、修订记录和可回放评测集。

不在本期范围：自动改写官方正文、无证据地生成新剧情、替代摄影师最终命名、训练新视觉模型、绕过版权或 robots 规则批量抓站。

## 2. 现有能力复用

| 已有模块 | 多模态扩展中的用途 |
|---|---|
| `filename_parser` | 读取日期、剧目、事项、演员-角色和摄影师；作为弱监督证据 |
| `burst_dedup` | 先去重再做昂贵的视觉语言推理，保留景别桶策略 |
| `face_db_builder` / `face_cluster` | 识别演员、建立角色候选和人工确认入口 |
| `scene_split` | EXIF 时间分段作为场景候选边界；人工映射仍可覆盖 |
| `role_classifier` | 生/旦/净/丑作为辅助特征，低置信度不得单独决定幕次 |
| `image_search.ImageEmbedder` | 图像向量、文本/图像相似检索和缓存 |
| `SmartOrganizer` | 接收最终 `scene_labels.json`，生成现有规范文件名和报告 |

## 3. 总体架构

### 3.1 目录与产物

建议新增以下模块和目录：

```
core_modules/organize/
├── play_source.py          # URL/剧目信息.txt 解析、抓取、快照、robots/限速
├── play_knowledge.py       # HTML 清洗、段落/图片/角色/幕次结构化
├── multimodal_evidence.py  # 剧照描述、OCR、图文向量和证据检索
├── scene_reasoner.py       # 候选召回、LLM JSON 判定、置信度和冲突处理
├── agent_runtime.py        # 受限工具调用、任务状态、审计日志
└── review_queue.py         # 人工复核、修订、评测集导出
```

```
outputs/play_knowledge/{play_slug}/
├── source_manifest.json       # URL、抓取时间、HTTP 状态、内容哈希、许可备注
├── pages/*.html                # 原始页面快照（可选压缩）
├── documents.jsonl             # 清洗后的正文块及 source_ref
├── media.jsonl                 # 图片、说明、页面位置、下载哈希
├── entities.json               # 角色、演员、幕次、地点、道具、事件
├── embeddings/                 # 文本/图像向量及模型签名
├── scene_candidates.jsonl      # 每图候选幕次及证据
├── review_queue.jsonl          # 待审核/已审核记录
└── knowledge_report.json       # 统计、错误和可复现参数
```

### 3.2 规范化数据契约

剧目知识块至少包含：

```json
{
  "play_id": "txgx201410",
  "source_ref": "page:txgx201410/index.html#p12",
  "kind": "plot|scene|role|behind_the_scenes|image_caption",
  "text": "郑庄公与武姜相见……",
  "entities": {"acts": [5], "scenes": ["见母"], "roles": ["郑庄公", "武姜"]},
  "published_at": null,
  "content_hash": "sha256:..."
}
```

场景判定记录至少包含：

```json
{
  "image": "relative/path.jpg",
  "scene_id": "act-05-scene-见母",
  "label": "第5幕见母",
  "confidence": 0.86,
  "decision": "model|human|rule",
  "evidence": [
    {"type": "filename", "ref": "...jpg", "quote": "孟广禄饰郑庄公"},
    {"type": "knowledge", "source_ref": "page:...#p12", "quote": "见母"},
    {"type": "visual", "model": "siglip2_base", "similarity": 0.73}
  ],
  "alternatives": [{"label": "第4幕天问", "confidence": 0.31}],
  "model_trace": {"model": "Qwen3.6-27B-INT4", "prompt_version": "scene-v1"},
  "review": {"status": "pending", "reviewer": null, "updated_at": null}
}
```

### 3.3 证据融合策略

候选生成采用可解释的分层策略，避免让大模型在全剧范围内自由猜测：

1. 规则候选：文件名已有“第 N 幕/场”、人工映射、目录事项；命中即保留为强证据。
2. 时间候选：`scene_split` 的 EXIF 段和拍摄批次，形成相邻图片先验。
3. 人物候选：ArcFace 命中演员后映射到官方角色；角色别名表支持“齐乌利/齐乌力”等变体。
4. 视觉候选：SigLIP2 图像向量与剧情块、官方花絮图片说明进行 top-k 检索；可加入 OCR、景别、舞台区域特征。
5. 语言判定：将最多 5 个候选及其证据交给本地模型，只允许从候选中选择，或返回 `unknown`。

建议初始评分：规则 0.35、人物 0.20、时间 0.15、视觉 0.20、语言一致性 0.10。权重写入配置并随报告记录；没有有效证据时不得输出高于 0.49 的置信度。

## 4. 官方内容采集与知识库

### 4.1 采集协议

- 仅从 `剧目信息.txt` 或显式配置的 URL 开始；域名白名单默认只包含 NCPA 页面。
- 先请求 `robots.txt`，遵守禁止路径；单域名并发 1、请求间隔至少 1 秒、总大小和页面数设上限。
- 保存 URL、抓取时间、状态码、响应哈希、编码、解析器版本；失败时记录错误并继续离线流程。
- 页面和图片仅作为内部整理证据，报告中保留原始链接与版权备注，不把远端资源直接复制进规范目录。
- 图片下载失败不阻塞正文入库；重复图片按内容哈希去重，缩略图和原图 URL 分开记录。

### 4.2 内容抽取

- HTML 清洗保留标题层级、段落顺序、`alt`、`figcaption`、相邻文本和页面锚点。
- 用规则优先识别“第 X 幕/场”“人物介绍”“剧情简介”“排练/花絮”；模型只负责补全实体关系，不覆盖原文。
- 中文分块建议 300–800 字，重叠 80 字；每块保留 `source_ref`，禁止只保存无出处摘要。
- 角色表包含官方角色名、演员、别名、证据段落；幕次表包含顺序、场名、剧情摘要、关键词和证据段落。

### 4.3 《天下归心》首批知识目标

首批验收应覆盖已存在整理目录中的 5 个幕次：`第1幕克段`、`第2幕庆功`、`第3幕私访`、`第4幕天问`、`第5幕见母`，以及“彩排/花絮”作为非剧情事件。网页不可达时，必须使用固定 HTML/JSON fixture 完成同样测试，不以网络状态作为通过条件。

## 5. 本地大模型与高级智能体

### 5.1 模型适配

`.env` 中的 `url`/`key`/`big_model_name` 和 `base_url`/`api_key`/`small_model_name` 被视为 OpenAI 兼容端点配置；密钥不得写入代码、报告或提交。推荐调用顺序：

1. `small_model_name`：页面摘要、实体补全、候选排序和常规 JSON 输出。
2. `big_model_name`：小模型 JSON 解析失败、证据冲突、防幻觉校验失败或端点异常时兜底。
3. `cascade_enabled=false` 时仍必须返回结构化错误和 `unknown`，不得静默猜测。

所有请求设置超时、最大 token、重试次数和请求哈希；提示词版本、模型名、温度、响应原文摘要写入 `model_trace`。生产默认 `temperature=0`，输出使用 JSON Schema 校验：`scene_id` 只能是候选 ID 或 `unknown`，引用必须来自输入证据 ID。

### 5.2 智能体角色与工具边界

智能体由编排器驱动，角色职责固定，工具调用可审计：

- 资料代理：读取快照、索引正文和媒体元数据，不直接联网写文件。
- 视觉代理：读取指定图片、调用现有人脸/景别/embedding 模块，输出观察事实，不生成剧情。
- 场景推理代理：合并候选证据，只能返回候选幕次或 `unknown`。
- 校验代理：检查引用存在性、角色-演员一致性、同一 EXIF 段标签连续性和命名格式。
- 复核代理：生成待人工队列和“需要哪条证据”的解释；不自动修改已人工确认记录。

禁止代理执行任意 shell、访问白名单之外的 URL、删除源照片或直接覆盖人工标签。每个任务有 `task_id`、输入哈希、工具清单、状态机（`queued/running/review/approved/failed`）和可重放日志。

## 6. 接口与流水线接入

建议 API（实现时沿用现有 `/api/progress/{task_id}`）：

| 方法 | 端点 | 作用 |
|---|---|---|
| POST | `/api/organize/play/source` | 从 `剧目信息.txt` 注册 URL 或离线快照 |
| POST | `/api/organize/play/ingest` | 后台抓取、解析、建知识库 |
| GET | `/api/organize/play/knowledge` | 查询幕次、角色、来源和错误 |
| POST | `/api/organize/scene/reason` | 对目录生成候选场景和证据报告 |
| GET | `/api/organize/scene/review` | 分页获取待审核记录 |
| POST | `/api/organize/scene/review` | 接受/修改/拒绝标签并写修订日志 |
| POST | `/api/organize/agent/run` | 执行受限智能体任务，返回 task_id |

`SmartOrganizer.organize()` 增加可选 `scene_evidence_file` 参数：存在时读取已审核标签；未审核或 `unknown` 的图片沿用 `scene-XX` 占位，不把模型猜测写进最终文件名。现有 `scene_labels.json` 格式保持兼容。

## 7. 里程碑

| 里程碑 | 交付内容 | 主要验收测试 | 完成标准 |
|---|---|---|---|
| M11 | `play_source.py`：URL 注册、robots、快照、失败重试、离线 fixture | `tests/test_play_source.py` | URL/快照哈希稳定；断网可复现；越权域名拒绝 |
| M12 | `play_knowledge.py`：正文/图片/角色/幕次抽取与实体别名 | `tests/test_play_knowledge.py` | fixture 五幕召回 ≥95%，每块有 source_ref |
| M13 | `multimodal_evidence.py`：图像描述、OCR、图文向量、候选召回缓存 | `tests/test_multimodal_evidence.py` | 重跑不重复推理；top-5 召回 ≥90%；缺图可降级 |
| M14 | `scene_reasoner.py`：规则+检索+LLM JSON、级联和防幻觉校验 | `tests/test_scene_reasoner.py` | Schema 通过率 ≥99%；越界 scene_id=0；冲突转 review |
| M15 | `review_queue.py`、API、审核 UI、修订日志 | `tests/test_scene_review_api.py` | 接受/修改/拒绝可重放；人工标签不可被重跑覆盖 |
| M16 | 接入 `SmartOrganizer`、智能体编排、全量验收文档 | `tests/acceptance_multimodal_m16.py` | 《天下归心》五幕准确率、引用完整性、性能和回归门槛全部通过 |

建议每个里程碑产出 `outputs/acceptance/mXX/`，包含输入清单、配置快照、测试 JSON、失败样本和模型 trace 摘要。

## 8. 验收测试标准

### 8.1 知识库正确性

- 官方 URL 解析成功率 100%；网络不可用时 fixture 运行成功率 100%。
- 五个剧情幕次均被抽取，幕次顺序正确率 100%，场名 exact match ≥95%。
- 角色-演员映射人工标注集 precision ≥95%；别名合并不得丢失原始名称。
- 每个 `documents.jsonl` 和 `media.jsonl` 条目均有可打开的 `source_ref`、内容哈希和解析器版本。

### 8.2 场景标记质量

建立 50 张以上《天下归心》金标准（每幕至少 8 张，含多人、远景、花絮和无 EXIF 样本），按图片分层抽样，避免同一连拍泄漏到训练/评测两侧：

- Top-1 幕次/场景准确率 ≥85%；宏平均 F1 ≥0.80。
- 高置信（≥0.75）结果 precision ≥90%；`unknown` 或低置信样本召回率 ≥95%。
- 与仅 EXIF 基线相比，准确率提升至少 15 个百分点；若未提升，M16 不通过。
- 人工复核后最终标签准确率 ≥95%，人工修改不得在重跑中丢失。
- 每个自动标签至少有两类证据，或明确标记为单证据/低置信；引用链接可追溯率 100%。

### 8.3 防幻觉与可靠性

- 构造不存在的幕次、角色和错配演员对抗集：模型不得输出候选集之外的实体，越界率 0%。
- 删除或篡改一条证据后，校验代理必须发现引用缺失并转 `review`/`failed`。
- 小模型故障、超时、非法 JSON、网络断开四类故障均能完成任务收尾；级联升级或 `unknown` 比例有报告。
- 同一输入哈希、配置和提示词版本重跑结果可复现（标签一致率 ≥99%，时间戳字段除外）。

### 8.4 性能与成本

- 100 张图片的缓存命中重跑不调用视觉语言模型；端到端 P95 ≤10 分钟（本地 GPU，排除首次下载）。
- 首次运行每图大模型调用 ≤1 次，失败重试总数 ≤2；小模型成功时大模型升级率 ≤20%。
- 采集任务遵守并发/限速配置；测试日志不能泄露 `.env` 密钥。

### 8.5 现有回归

M16 必须运行现有智能整理测试和本扩展测试：

```bash
python tests/test_filename_parser.py
python tests/test_burst_dedup.py
python tests/test_face_db_build.py
python tests/test_face_cluster.py
python tests/test_scene_role.py
python tests/test_smart_organize.py
python tests/acceptance_multimodal_m16.py
```

既有命名合规率、连拍剪除、人脸识别覆盖和报告字段完整性不得低于 `smart_organize_blueprint.md` M7/M10 已记录基线；新增证据字段只能向后兼容地追加。

## 9. 风险、降级与发布门槛

| 风险 | 处理 | 发布门槛 |
|---|---|---|
| 官方页面改版/不可达 | 快照、fixture、解析器版本化 | 离线验收必须通过 |
| 花絮图与剧情图混淆 | `kind` 分离，花絮默认不能决定幕次 | 花絮误标率 ≤5% |
| 戏曲妆造导致人脸失败 | 使用文件名、角色表、视觉描述和人工队列补证 | 不以人脸单项失败阻塞 |
| 本地模型输出幻觉 | JSON Schema、候选闭集、引用校验、级联 | 越界率 0% |
| 密钥/隐私泄露 | `.env` 仅运行时读取，脱敏 trace，禁止提交 | secret scan 通过 |
| 自动标签覆盖人工决定 | review 状态优先，版本化修订日志 | 人工标签保留率 100% |

发布前必须同时满足场景质量、防幻觉、引用完整性、性能和现有回归五组门槛；任一组失败，只允许以“候选/审核模式”运行，不得自动改写最终文件名。

## 10. 首次实施顺序

1. 先用《天下归心》固定快照完成 M11–M12，得到可检查的五幕知识库和角色表。
2. 使用已有 35 张整理完成剧照建立 50 张扩展金标准（可补充花絮和重复组），冻结评测清单。
3. 完成图文候选召回和离线 LLM fixture，再接入 `.env` 本地端点；先验证 JSON/引用，再开放自动标签。
4. 上线审核队列和 `scene_evidence_file` 接口，最后接入 `SmartOrganizer` 与智能体 API。

