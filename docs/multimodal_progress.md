# 多模态剧照整理扩展 - 进度状态文件

> 用途：记录 `docs/multimodal_context_blueprint.md`（M11–M16）的实施进度、关键决策与接续入口。
> 每次继续工作前先读本文件，收工后更新「当前进度」「下一步」和「未提交产物」。
> 最后更新：2026-10-21

## 1. 里程碑状态总览

| 里程碑 | 模块 | 状态 | 说明 |
|---|---|---|---|
| M11 | `play_source.py` | 🟢 离线完成 | URL 白名单、robots 合规、限速重试、页数/大小上限、快照哈希与离线读取 |
| M12 | `play_knowledge.py` | 🟢 离线完成 | 《天下归心》五幕解析、角色别名（郑庄公→庄公）映射回幕次，人物证据生效 |
| M13 | `multimodal_evidence.py` | 🟢 离线完成 | 词法 + 图文向量召回、向量磁盘缓存（重跑不重复推理）、缺图/缺编码器降级；OCR/图像描述待补 |
| M14 | `scene_reasoner.py` | 🟢 完成（历史验收） | 闭集候选、LLM JSON、级联和防幻觉；旧客户端真实端点联调记录见第 5、6 节，不代表新 LangChain 端到端验收 |
| M15 | `review_queue.py` + API + UI | 🟢 离线完成 | 复核队列、接受/修改/拒绝、不可变修订日志、最终标签导出；`/organize` 看图复核，原卡片⑤位于 `/legacy/organize` |
| M16 | `SmartOrganizer` 接入 + 智能体 + 全量验收 | 🟢 完成（历史验收） | 审核证据接入、受限运行时、离线聚合及目标机 gated 回归记录见第 6 节；不表示所有新环境均已验收 |

图例：🟢 完成 / 🟡 进行中或部分完成 / ⚪ 未开始

## 2. 已完成产物与历史验收

2026-10-17 工作台更新，2026-10-19 复验并纳入 Git 提交：`/organize` 与 `/ui` 接入照片网格、剧情建库、图文分析、看图复核与自动标签导出；旧版总入口 `/legacy`，原照片处理与卡片界面分别位于 `/legacy/ui`、`/legacy/organize`，兼容 `/tools` 与 `/organize/advanced`。新增 `app/workspace_api.py`、`app/templates/workspace.html`、`app/static/workspace.*` 和工作台回归测试。工作台接口 9/9、复核 UI、图文召回和人工证据合并回归通过；35 张真实剧照的浏览器离线操作及五种屏宽验收通过；GPU/LLM 质量没有重新评测。详见 [工作台指南](workspace_guide.md)。

下列 M11–M16 源码已有提交记录，并非全部未提交。测试与报告部分仅存在本地；以 `git ls-files` 和 `git status` 为准。当前未发布 LangChain 扩展见 [CHANGELOG](CHANGELOG.md)，目录与部署见 [项目说明](PROJECT_GUIDE.md)。

新增源码：
- `core_modules/organize/play_source.py` — URL 提取、域名校验、robots 合规、限速重试、抓取快照、离线读取
- `core_modules/organize/play_knowledge.py` — 正文/幕次/角色抽取、别名与人物证据映射、知识库落盘
- `core_modules/organize/multimodal_evidence.py` — 词法 + 图文向量召回、向量缓存与降级（M13）
- `core_modules/image_search.py` — `ImageEmbedder.encode_text` 与 `model_signature`（SigLIP 真实文本编码，M13）
- `core_modules/organize/llm_client.py` — OpenAI 兼容 `.env` 端点与级联客户端（M14）；`client_from_env` 优先 `small_diag_url`+`small_diag_model` 显式组合
- `core_modules/organize/scene_reasoner.py` — 闭集场景判定与防幻觉校验（M14）；`_build_prompt` 候选 evidence_ids 按全局证据表 `source_ref` 匹配（修复引用越界）
- `core_modules/organize/review_queue.py` — 复核队列/修订日志/最终标签导出/证据加载（M15、M16）
- `core_modules/organize/agent_runtime.py` — 受限智能体任务状态机、工具白名单、审计日志（M16）
- `core_modules/organize/conversational_agent.py` — 连续对话、会话 JSON 持久化、结构化记忆、自主工具调用与模型/工具兜底
- `app/main.py` — `/api/organize/scene/review`（seed/list/decide/export）、`play/source|ingest|knowledge`、`scene/reason`、`agent/run`、`agent/chat`、`agent/session/{session_id}`、`agent/tools`，`/api/organize/run` 增加 `scene_evidence_file`（M11–M16）
- `app/server_ui.py` — `/organize` 卡片⑤「场景标签人工复核」与卡片③「场景证据文件」入口（M15）
- `core_modules/organize/smart_organizer.py` — 增加 `apply_scene_evidence` 与 `organize(scene_evidence_file=...)`（M16）

新增测试：
- `tests/test_play_source.py` — M11 来源管理/robots/重试/快照（离线，假 opener）
- `tests/test_play_knowledge.py` — M11–M13 离线验收（当前通过）
- `tests/test_multimodal_evidence.py` — M13 图文向量召回/缓存/降级（假 embedder，离线）
- `tests/test_scene_reasoner.py` — M14 离线验收（假 LLM，无网络）
- `tests/test_scene_review_api.py` — M15 复核 API 验收（TestClient 直调）
- `tests/test_scene_review_ui.py` — M15 `/organize` 卡片⑤ UI 冒烟（需 fastapi 环境）
- `tests/test_multimodal_api.py` — M11/M14/M16 多模态 API（play/source|ingest|knowledge、scene/reason、agent/run）
- `tests/test_scene_evidence_merge.py` — M16 已审核证据合并（纯逻辑，离线）
- `tests/test_agent_runtime.py` — M16 受限智能体运行时（纯逻辑，离线）
- `tests/test_conversational_agent.py` — 连续会话、跨实例记忆、自主多步工具调用、级联/确定性兜底、并发与敏感字段测试
- `tests/acceptance_multimodal_m16.py` — M16 离线验收聚合脚本
- `tests/test_e2e_scene_pipeline.py` — M16 端到端离线串联（reason→review→export→organize）

知识库产物：
- `outputs/play_knowledge/txgx201410/` — 《天下归心》`documents.jsonl`/`media.jsonl`/`entities.json`/`scene_candidates.jsonl`/`knowledge_report.json`
- `outputs/play_knowledge/mkbl2015/` — 《马可·波罗》同名产物；初版正文无幕次导致 `n_scenes=0`，已从整理完成文件名派生 3 幕（见第 5、6 节）

验收归档（2026-09-17）：
- `outputs/acceptance/m16/gated_regressions.json` — 六项重型回归 6/6 通过记录
- `outputs/acceptance/m16/siglip_calibration.json` — SigLIP2 图文检索阈值/校准结论
- `outputs/acceptance/m16/llm_endpoint_diag.json` — `.env` 端点级联与 model_trace 诊断
- `outputs/acceptance/m16/siglip_zero_shot_caption_eval.json` — M13 零样本 caption 实验（不可用结论）
- `outputs/tmp_scene_llm_diag/scene_evidence_llm.jsonl` — 真实端点批量 9 图证据（含 model_trace）

## 3. 当前可复现命令

```bash
cd /home/huachenghao/codes/Theatrical_photo_sorting-251212
# M11–M14 仅依赖标准库，base python 即可python tests/test_play_source.py      # M11 来源/robots/重试/快照（通过）
python tests/test_play_knowledge.py   # M11–M13 离线验收（通过）
python tests/test_multimodal_evidence.py  # M13 图文向量/缓存/降级（通过）
python tests/test_scene_reasoner.py   # M14 离线验收，不联网（通过）
# M15 需要 fastapi，使用 face_scrfd_arcface 环境
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_scene_review_api.py  # 通过
# M16 离线验收聚合（跑上面扩展测试 + 列出重型回归）
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/acceptance_multimodal_m16.py  # 11/11 通过（需可正常导入 FastAPI 的目标环境）
```

## 4. 关键决策与约定

- **不改写原文**：`play_knowledge` 只抽取幕次/角色/媒体，文本与 `content_hash` 保留出处。
- **闭集判定**：`scene_reasoner` 只允许从候选 `scene_id` 或 `unknown` 中返回，越界即视为无效。
- **离线优先**：M11–M16 测试全部使用 `data/ncpa_test/` 固定样本或假模型，不依赖网络与真实模型。
- **密钥安全**：`.env` 运行时读取，`model_trace` 只记录模型名/提示词版本/响应摘要，不落密钥。
- **级联升级条件**：仅在小模型 JSON 解析失败/校验失败/端点异常时升级到 `big_model_name`，不依赖模型置信度。

## 5. 已知问题 / 待办

- [x] 2026-10-17：将现有多模态 API 接入以剧照为中心的可视化工作流，补齐真实图片路径、依据查看和复核到导出的自动衔接。

- [x] M14：真实本地端点联调完成（`small_diag`→`big` 级联验证 + `model_trace` 全字段落盘，见 `outputs/acceptance/m16/llm_endpoint_diag.json`）；`client_from_env` 现优先 `small_diag_*` 显式配置。
- [x] M13：真实 SigLIP2 图文链路联调完成（35 图 top1=66%/top3=86%，富文本 top3=100%；act-02 舞台画面系统性偏高，视觉仅作辅助先验，见 `outputs/acceptance/m16/siglip_calibration.json`）。OCR/图像描述特征仍待补。
- [x] M16：目标机 gated 重型回归 6/6 通过（`filename_parser` 已修正信息目录 GT 口径后通过；报告见 `outputs/acceptance/m16/gated_regressions.json`）。
- [ ] M13：补 OCR/图像描述特征 → 已评估（2026-09-17）：SigLIP2 base 零样本 caption top1=0/35、短语无区分度，不作为证据源；官方图注已作 `image_caption` 证据入库替代（`outputs/acceptance/m16/siglip_zero_shot_caption_eval.json`）；真 caption 模型（BLIP2/Qwen-VL）待本地权重可用再接入。
- [x] M12：`马可·波罗` 正文无「第 N 幕/场」已解决：`build_knowledge` 从整理完成目录文件名抽取「第N幕：标题/序：标题」，序幕映射 `act-00`，mkbl2015 知识库 3 幕生成，规则短路判定通过（`tests/test_play_knowledge.py`）。
- [x] 连续对话智能体：JSON 会话持久化、滚动历史摘要、结构化 memory、自主白名单工具调用、非法 JSON/端点/工具失败兜底。
- [x] 端到端串联：`reason → review → export → organize(scene_evidence_file)` 全链路脚本。
- [x] M15 审核 UI：`/organize` 卡片⑤ 复核 + 卡片③ `scene_evidence_file`。
- [x] 蓝图 API 全量：`play/source|ingest|knowledge`、`scene/reason`、`agent/run`。

## 6. 下一步（接续从这里开始）

2026-10-21 接续入口为 [核心升级蓝图](core_upgrade_blueprint.md) 与 [升级进度](core_upgrade_progress.md)。C1–C4 首轮修复已提交；U2 缓存及批量路径已接入，原完整测量仍 `not_passed`。线程池修复与 4 线程复测单列；冷缓存重复哈希已减少，`u2_publication_200x5_20261021/` 完整复验结束：特征吞吐 1.151 倍未达标，内存与输出一致性通过；共享 GPU 下整理耗时观察减少 68.3%，不能当作隔离负载的可靠收益。U3 接入缩略图缓存/ETag/限额、持久草稿、单条更新及 SQLite 事务；复核存储 9/9、工作台 18/18、多模态 11/11，35 图/五屏宽、实际 409 冲突与 10000 条分页通过。场景分析已接入持久队列、检查点、取消和刷新恢复，任务核心 9/9。正式质量、整理/建库逐项恢复和真实 GPU 长任务 HTTP 响应仍待验收。以下保留 M11–M16 历史记录，不代替新阶段验收。

1. ✅ M11 来源/robots/重试/快照通过（`tests/test_play_source.py`）。
2. ✅ M11–M13 知识库与候选召回通过（`tests/test_play_knowledge.py`，含角色别名人物证据）。
3. ✅ M13 图文向量召回/缓存/降级通过（`tests/test_multimodal_evidence.py`）。
4. ✅ M14 闭集判定与级联通过（`tests/test_scene_reasoner.py`）。
5. ✅ M15 复核核心/API/UI 通过（`test_scene_review_api.py`、`test_scene_review_ui.py`）。
6. ✅ M16 离线验收 11/11 通过（`tests/acceptance_multimodal_m16.py`，FastAPI 三项已在 face_scrfd_arcface 环境复核通过）。
7. ✅ 连续对话智能体离线验收（`tests/test_conversational_agent.py`）：跨实例记忆、两步工具链、模型不可用路由、级联和工具失败收尾、白名单边界。
8. ✅ 目标机 gated 重型回归（2026-09-17）：6/6 通过（`filename_parser` 修正「剧组信息/剧目信息」剧情说明图口径后 555/555 通过；见 `outputs/acceptance/m16/gated_regressions.json`）。
9. ✅ 真实 SigLIP2 联调：本地权重加载、图文 768 维向量、磁盘缓存（重跑 0 次推理）；35 图检索 top1=66%/top3=86%，富文本 top3=100%；结论：视觉向量作辅助先验，阈值记录于 `outputs/acceptance/m16/siglip_calibration.json`。
10. ✅ `.env` 本地端点联调：`small_diag`（未启动）→`big`（Qwen3.6-27B）级联实际触发并验证；闭集 unknown 不越界；批量 9 图 `model_trace` 全字段落盘 `outputs/tmp_scene_llm_diag/scene_evidence_llm.jsonl`。
11. ✅ M12《马可·波罗》幕次扩展（2026-09-17）：`play_knowledge.py` 新增 `parse_scene_terms`/`_scenes_from_filenames`，正文无幕次时从文件名建幕次（序=act-00）；`scene_reasoner` 规则短路覆盖新命名；commit `7c91425`/`1e49e74`。
12. ✅ M13 OCR/图像描述评估（2026-09-17）：本地 SigLIP2 零样本 caption 两轮实验（中文短语/英文 prompt）均 top1=0/35，记录结论并归档；官方图注证据已覆盖离线语料，caption 模型待权重。
13. 下一步（可选）：接入 BLIP2/Qwen-VL 本地权重做真实 captioning；扩展更多剧目表达（场/折）。
14. ✅ 2026-10-19 工作台及旧版入口复验完成，源码、文档、截图与三项工作台测试纳入 Git 提交。后续可在目标 GPU 服务上按新界面复测全量识别与归档；本次仅对 GPU 操作做接口交互验证。

## 7. 状态文件维护约定

- 每次收工更新第 1、2、5、6 节；保持「未提交产物」与 `git status` 一致。
- 里程碑完成时在状态表改为 🟢，并补充验收命令与结果摘要。
- 注意：`.gitignore` 含 `tests/*`，本目录新增测试默认不入库（工作区保留）；如需提交测试请 `git add -f`。
