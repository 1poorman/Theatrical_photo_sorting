# 多模态剧照整理扩展 - 进度状态文件

> 用途：记录 `docs/multimodal_context_blueprint.md`（M11–M16）的实施进度、关键决策与接续入口。
> 每次继续工作前先读本文件，收工后更新「当前进度」「下一步」和「未提交产物」。
> 最后更新：2026-09-15

## 1. 里程碑状态总览

| 里程碑 | 模块 | 状态 | 说明 |
|---|---|---|---|
| M11 | `play_source.py` | 🟢 离线完成 | URL 白名单、robots 合规、限速重试、页数/大小上限、快照哈希与离线读取 |
| M12 | `play_knowledge.py` | 🟢 离线完成 | 《天下归心》五幕解析、角色别名（郑庄公→庄公）映射回幕次，人物证据生效 |
| M13 | `multimodal_evidence.py` | 🟢 离线完成 | 词法 + 图文向量召回、向量磁盘缓存（重跑不重复推理）、缺图/缺编码器降级；OCR/图像描述待补 |
| M14 | `scene_reasoner.py` | 🟢 离线完成 | 闭集候选 + LLM JSON + 小/大模型级联 + 防幻觉校验；真实端点联调待做 |
| M15 | `review_queue.py` + API + UI | 🟢 离线完成 | 复核队列、接受/修改/拒绝、不可变修订日志、最终标签导出；`/organize` 卡片⑤ 审核 UI |
| M16 | `SmartOrganizer` 接入 + 智能体 + 全量验收 | 🟢 离线完成 | `scene_evidence_file` 接入、证据合并、受限智能体运行时、离线验收聚合；模型/ES 全量验收待目标机 |

图例：🟢 完成 / 🟡 进行中或部分完成 / ⚪ 未开始

## 2. 已完成产物（工作区，未提交）

新增源码：
- `core_modules/organize/play_source.py` — URL 提取、域名校验、robots 合规、限速重试、抓取快照、离线读取
- `core_modules/organize/play_knowledge.py` — 正文/幕次/角色抽取、别名与人物证据映射、知识库落盘
- `core_modules/organize/multimodal_evidence.py` — 词法 + 图文向量召回、向量缓存与降级（M13）
- `core_modules/image_search.py` — `ImageEmbedder.encode_text` 与 `model_signature`（SigLIP 真实文本编码，M13）
- `core_modules/organize/llm_client.py` — OpenAI 兼容 `.env` 端点与级联客户端（M14）
- `core_modules/organize/scene_reasoner.py` — 闭集场景判定与防幻觉校验（M14）
- `core_modules/organize/review_queue.py` — 复核队列/修订日志/最终标签导出/证据加载（M15、M16）
- `core_modules/organize/agent_runtime.py` — 受限智能体任务状态机、工具白名单、审计日志（M16）
- `app/main.py` — `/api/organize/scene/review`（seed/list/decide/export）、`play/source|ingest|knowledge`、`scene/reason`、`agent/run`，`/api/organize/run` 增加 `scene_evidence_file`（M11–M16）
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
- `tests/acceptance_multimodal_m16.py` — M16 离线验收聚合脚本
- `tests/test_e2e_scene_pipeline.py` — M16 端到端离线串联（reason→review→export→organize）

知识库产物：
- `outputs/play_knowledge/txgx201410/` — 《天下归心》`documents.jsonl`/`media.jsonl`/`entities.json`/`scene_candidates.jsonl`/`knowledge_report.json`
- `outputs/play_knowledge/mkbl2015/` — 《马可·波罗》同名产物（正文无「第 N 幕」，`n_scenes=0`）

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
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/acceptance_multimodal_m16.py  # 10/10 通过
```

## 4. 关键决策与约定

- **不改写原文**：`play_knowledge` 只抽取幕次/角色/媒体，文本与 `content_hash` 保留出处。
- **闭集判定**：`scene_reasoner` 只允许从候选 `scene_id` 或 `unknown` 中返回，越界即视为无效。
- **离线优先**：M11–M16 测试全部使用 `data/ncpa_test/` 固定样本或假模型，不依赖网络与真实模型。
- **密钥安全**：`.env` 运行时读取，`model_trace` 只记录模型名/提示词版本/响应摘要，不落密钥。
- **级联升级条件**：仅在小模型 JSON 解析失败/校验失败/端点异常时升级到 `big_model_name`，不依赖模型置信度。

## 5. 已知问题 / 待办

- [ ] M12：`马可·波罗` 正文无「第 N 幕/场」，需支持其它幕次表达或人工映射。
- [ ] M13：补 OCR/图像描述特征；真实 SigLIP2 文本编码已接入 `ImageEmbedder.encode_text`，阈值待在目标机校准。
- [ ] M14：真实本地端点联调（`small_diag` → `big_model_name` 级联）与 `model_trace` 落盘。
- [ ] M16（离线已过）：在目标机跑重型回归与模型/ES 全量验收。
- [x] 端到端串联：`reason → review → export → organize(scene_evidence_file)` 全链路脚本。
- [x] M15 审核 UI：`/organize` 卡片⑤ 复核 + 卡片③ `scene_evidence_file`。
- [x] 蓝图 API 全量：`play/source|ingest|knowledge`、`scene/reason`、`agent/run`。

## 6. 下一步（接续从这里开始）

1. ✅ M11 来源/robots/重试/快照通过（`tests/test_play_source.py`）。
2. ✅ M11–M13 知识库与候选召回通过（`tests/test_play_knowledge.py`，含角色别名人物证据）。
3. ✅ M13 图文向量召回/缓存/降级通过（`tests/test_multimodal_evidence.py`）。
4. ✅ M14 闭集判定与级联通过（`tests/test_scene_reasoner.py`）。
5. ✅ M15 复核核心/API/UI 通过（`test_scene_review_api.py`、`test_scene_review_ui.py`）。
6. ✅ M16 离线验收 10/10 通过（`tests/acceptance_multimodal_m16.py`），含端到端串联与全量 API。
7. 下一步：用真实 SigLIP2 `ImageEmbedder` 与 `.env` 本地端点联调；在目标机跑 gated 重型回归并归档 `outputs/acceptance/m16/`。

## 7. 状态文件维护约定

- 每次收工更新第 1、2、5、6 节；保持「未提交产物」与 `git status` 一致。
- 里程碑完成时在状态表改为 🟢，并补充验收命令与结果摘要。
- 注意：`.gitignore` 含 `tests/*`，本目录新增测试默认不入库（工作区保留）；如需提交测试请 `git add -f`。
