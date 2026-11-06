# 项目说明与维护指南

更新日期：2026-09-17。本文说明当前工作区，包括尚未发布的 LangChain 高级智能体；历史状态以 [CHANGELOG](CHANGELOG.md) 为准。

## 1. 项目定位与架构

系统面向戏剧照片处理、检索和整理，包含通用图像工具、人脸识别、智能整理、多模态场景复核和对话智能体。

2026-10-19 制定的 [核心升级蓝图](core_upgrade_blueprint.md) 覆盖正确性、批量处理、任务恢复和开源算法比较；已开展首轮修复、数据清单及本地多模态模型验证，当前状态、配置与验收命令见 [升级进度](core_upgrade_progress.md)。

```text
FastAPI / Web UI / 独立智能体 CLI
  ├─ 人脸识别、检测、分割、景别分类、修复、图像检索
  ├─ SmartOrganizer：去重 → 人脸/角色 → 场景分段 → 复制命名
  ├─ 剧目知识 → 闭集场景候选 → 人工复核 → 整理管线
  └─ LangChainPhotoAgent：模型决策 → 白名单工具 → 观察结果 → 继续/回答
       ├─ ConversationStore：会话、上下文、历史摘要
       └─ AgentRuntime：任务状态、结果与审计
```

场景候选不等于人工批准；任务状态 `review` 表示结果等待复核，不表示工具尚未执行。目录占位标注与文件名解析不是视觉识别。

## 2. 文件职责与管理

| 路径 | 职责 / 管理方式 |
| --- | --- |
| `app/main.py` | API 路由、请求参数和响应；依赖 `server` 的模型实例 |
| `app/server.py` | 模型初始化、服务生命周期；API 启动入口 |
| `app/server_ui.py` | Web UI，复用 API 应用 |
| `app/photo_agent_cli.py` | 独立 LangChain 对话入口，无需加载视觉模型 |
| `config/base.py` | 路径、阈值、ES 与部分环境变量默认值 |
| `core_modules/face_recognition/` | SCRFD、ArcFace、ES 人脸库、TRT 支持 |
| `core_modules/*.py` | 人像检测/分割、景别、修复、图像向量检索 |
| `core_modules/organize/` | 文件名解析、去重、建库、演员视图、知识、复核、智能体 |
| `core_modules/tools/` | 图像 I/O、日志、可视化、质量工具，优先复用 |
| `assets/` | 随 Git 保存的文档截图；样例原图位于本地 `data/sample_images/` |
| `data/ncpa_test/` | 本地剧目测试数据，不随普通克隆提供 |
| `data/face_database/` | 人脸锚定库，含敏感数据，不公开提交 |
| `weights/` | 本地权重，默认忽略；不随意重命名 |
| `outputs/`、`logs/` | 运行产物与日志，全部忽略；历史产物保留本地 |
| `tests/test_*.py` | 按功能组织的测试，多数是可直接执行脚本 |
| `tests/acceptance_*.py` | 聚合/全量验收，依赖特定数据与环境 |
| `tests/local/tmp_*.py` | 本地实验、历史修补与冒烟脚本，不是正式迁移程序 |
| `requirements/vision.txt` | 视觉服务依赖；`requirements/environment.yml` 为 Conda 环境定义 |
| `requirements/agent.txt` | LangChain 扩展依赖，不包含视觉服务完整依赖 |
| `Dockerfile`、`docker-compose.yml` | 主服务镜像与 GPU/存储挂载 |

2026-10-28：环境文件集中于 `requirements/`；临时脚本归档至 `tests/local/`。正式代码与测试、原图/图库和历史验收目录保持路径。`tests/.gitignore` 明确列出随 Git 保存的测试；`python tools/audit_repository_scope.py` 检查暂存索引，排除本地数据、权重、缓存、日志和实验输出。文档截图与精选技术报告包继续随 Git 保存。

## 3. 安装与启动

所有命令默认从项目根目录执行。

### 主视觉服务

```bash
conda env create -f requirements/environment.yml
conda activate face_scrfd_arcface
python -m pip install -r requirements/vision.txt
python app/server.py
```

PyTorch 的 CUDA 安装方式以依赖文件顶部说明为准；权重布局见项目 README。服务默认 API 端口 8198，界面启动方式为 `python app/server_ui.py`（8199）。也可在 `app/` 目录执行 `uvicorn main:app --host 127.0.0.1 --port 8198`。不要假设根目录下 `uvicorn app.main:app` 与脚本入口等价：现有代码使用 `import server`。

健康接口：`GET /api/health`，交互式 API：`/docs`。服务开始监听不代表模型全部加载成功；需要检查模型加载日志和实际业务接口。人脸系统初始化会访问所选向量库（Docker 默认 Qdrant，本地默认 Elasticsearch），连接失败可能导致模型实例未赋值。

### 独立智能体

```bash
conda activate agent
python -m pip install -r requirements/agent.txt
python -m app.photo_agent_cli --session demo --model auto --protocol json
```

若没有 `agent` 环境，可先创建 Python 3.11 环境。CLI 不要求本机 GPU，但需要可用模型 HTTP 端点；它不会启动模型服务。要在主 API 使用高级智能体，需要在**主服务的同一 Python 环境**安装 `requirements/agent.txt`，不能只安装到另一个 Conda 环境。

### Docker

`docker compose up -d --build` 启动可视化工作台（8199，包含全部 API）和 Qdrant 1.20.0，应用等待数据库健康后启动。图像与人脸向量在线查询，默认保持精确余弦；图像本地保存小型连接清单，旧 FAISS 仍可读取。宿主机需安装支持 CUDA 12.8 的 NVIDIA 驱动及 NVIDIA Container Toolkit。权重只读挂载，数据、输出、日志和 Qdrant/TRT/模型缓存持久化。Qdrant 仅在 Compose 内网开放 6333；首次使用为空库。旧 ES 数据卷保留，可用 `--profile legacy` 启动迁移或回退（启用 ES 时需 `vm.max_map_count >= 262144`）。配置见 `docker.env.example`，迁移、增删、回退与容量管理见 [在线向量库指南](qdrant_guide.md)，测量见 [Qdrant 对比报告](reports/qdrant_20261102/README.md)。

Dockerfile 安装视觉服务、ModelScope framework 和 `requirements/agent.txt` 的 LangChain 依赖。现有 `.env` 不打包进镜像；启用大模型配置时追加 `-f docker-compose.yml -f docker-compose.llm.yml`，将配置只读挂载至 `/app/.env`，并使用容器可访问的模型端点。不要将含密钥的文件写入镜像。业务表单使用容器内路径，例如 `/app/data/...` 和 `/app/outputs/...`。

## 4. 配置与模型选择

| 配置 | 作用与注意事项 |
| --- | --- |
| `TRANS_WEIGHTS_DIR`、`TRANS_OUTPUT_DIR` | 权重与输出根目录 |
| `TRANS_ELASTIC_URL`、`TRANS_ELASTIC_USER`、`TRANS_ELASTIC_PASSWORD` | ES 端点与凭据；部署时覆盖开发默认值 |
| `TRANS_SIGLIP_WEIGHTS_ROOT` | SigLIP2 本地权重 |
| `TRANS_KNOWN_FACE_THRESHOLD`、`TRANS_FACE_IMAGE_MAX_SIDE` | 识别阈值和图片缩放 |
| `TRANS_FACE_DEVICE` | 人脸推理后端（如 `tensorrt`、`cuda`、`cpu`）；主服务尊重此配置，无可见 GPU 时回退 CPU |
| `.env` 的 `small_diag_url/model/key` | 实际键名为 `small_diag_url`、`small_diag_model`、`small_diag_key` |
| `.env` 的 `url`、`big_model_name`、`key` | 大模型端点、模型名和密钥 |
| `AGENT_SMALL_URL/MODEL/KEY`、`AGENT_BIG_URL/MODEL/KEY` | 新智能体的环境变量覆盖，各斜杠项均有 `AGENT_SMALL_` / `AGENT_BIG_` 前缀 |

`.env` 解析器支持 `key=value` 和 `key: value`，不是可直接 `source` 的 Shell 文件。不要在命令日志或文档输出实际密钥。

本地服务未指定 `CUDA_VISIBLE_DEVICES` 时默认使用物理 GPU1，显式设置优先；Docker 默认只分配物理 GPU0，可通过 `DOCKER_GPU_DEVICE` 调整。进程内第一块可见 GPU 的编号为 0，人脸推理后端由 `TRANS_FACE_DEVICE` 指定，以实际 ORT providers 日志为准。授权外宿主机能使用 GPU 不代表受限沙箱也可使用，反之亦然。

新智能体 `model=auto` 顺序尝试小/大模型，`small` 或 `big` 单独选择；`protocol=json` 适配非原生工具调用模型，`native` 要求端点支持 tools。新智能体 `auto` 的行为与旧客户端 `cascade_enabled` 开关不可混为一谈。

本机已发现的 vLLM 环境名为 `mff_sft_serve`，不是最初提到的 `mff_sft_server`。本地权重在 `~/models/MiniCPM5-2B`；启动命令见智能体蓝图，需在有 GPU 权限的环境运行。

## 5. 业务操作与数据产物

浏览器推荐使用 `/organize` 多模态工作台，操作顺序见 [工作台使用指南](workspace_guide.md)。`/ui` 为同入口；右上角「旧版界面」通往 `/legacy`，保留 `/legacy/ui` 照片处理与 `/legacy/organize` 智能整理，两页可互相切换并返回新版。`/tools` 和 `/organize/advanced` 分别作为旧版两页的兼容入口。项目设置在浏览器保存，照片、知识库和人工复核结果仍保存在服务器目录。

### 标准整理

1. 准备原始照片和本地人脸锚定库；可通过已标注图片建库，或聚类后人工命名。
2. 选择新输出目录，调用 `/api/organize/run`；去重选优、识别人脸、场景分段并复制命名。
3. 查看 `organize_report.json` 与 `organized/`、`best/`、`discarded/`；原图保留。

直接整理 API 与新智能体包装的路径保护不同：智能体包装检查输出不存在；不能把这一检查当作所有旧 API 的全局保证。始终为每次运行使用新目录。

### 场景证据与人工审核

1. `play/source` 注册来源，`play/ingest` 从本地文本及媒体建立知识库。
2. `scene/reason` 生成候选和证据；可选图文编码器/LLM 的能力以接口参数为准。
3. `scene/review/seed` 导入队列，人工接受/修改/拒绝后导出标签。
4. 把已审核证据传给整理 API 的 `scene_evidence_file`；未批准或 `unknown` 不覆盖占位场景。

| 产物 | 内容 |
| --- | --- |
| `entities.json` | 幕次、角色、角色别名 |
| `documents.jsonl`、`media.jsonl` | 文本/媒体证据与来源 |
| `knowledge_report.json` | 知识库统计 |
| `scene_evidence.jsonl` | 每张图的候选、证据、判定和复核信息 |
| `review_queue.jsonl` | 复核记录；`revisions.jsonl` 为人工修订记录 |
| `scene_labels.json` | 已批准的标签映射 |
| `agent/conversations/<session_id>.json` | 会话历史、结构化记忆、摘要和工具结果 |
| `agent/tasks/agent_tasks.json`、`agent/tasks/agent_audit.jsonl` | 对话智能体任务状态及审计；单次 `agent/run` 使用不同的运行根目录 |

### 智能体操作

```bash
python -m app.photo_agent_cli --model big --session demo \
  --message '检查目录中的照片数量和命名信息，不修改文件' \
  --context '{"input_dir":"data/sample_images"}'
```

复用 `session_id` 续聊。高级 API 为 `/api/organize/agent/advanced/chat`，使用 Form 参数 `message`、`session_id`、`context_json`、`model`、`protocol`、`max_steps`；其工具清单在 `/api/organize/agent/advanced/tools`。原对话 API 保留。会话读取复用 `/api/organize/agent/session/{session_id}`。

CLI 注册 7 个工具；主 API 额外注册 `organize_photos`，依赖已加载视觉模型及人脸库。新工具 `scene_reason` 当前使用文件名和知识库证据，并不执行视觉编码；不要与同名旧 API 包装的可选视觉能力混淆。

## 6. 验证、排障与安全

测试分层而非一键执行所有脚本：

| 层级 | 示例命令 | 前提 / 风险 |
| --- | --- | --- |
| 智能体离线 | `python tests/test_langchain_agent.py` | `agent` 环境；真实 LangChain 图、脚本化模型，无真实 LLM 推理 |
| 旧智能体逻辑 | `python tests/test_agent_runtime.py`、`python tests/test_conversational_agent.py` | 本地脚本存在；会重建专用测试输出 |
| 知识与证据 | `python tests/test_play_knowledge.py`、`python tests/test_scene_reasoner.py` | 部分测试依赖本地剧目样本；假模型不等于真实模型验收 |
| API/UI | `python tests/test_multimodal_api.py`、`python tests/test_scene_review_ui.py` | 主服务依赖；部分脚本清理 `outputs/agent`，请先阅读并隔离会话 |
| GPU/ES/全量 | `python tests/acceptance_m7.py`、`python tests/test_face_cluster_api.py` | 权重、剧目数据、GPU/ES；可能写入正式库，不能直接在生产环境运行 |

历史通过结果见多模态进度记录。2026-11-02 容器验收通过 18 项真实 HTTP 检查、21 项浏览器工作流、8 项重启与恢复检查及 14 项 LangChain 回归；已验证当前大模型端点的 JSON 工具调用。小模型与真实 native 协议仍未在本轮验收，详情及证据见 [Docker 功能验收](reports/docker_functional_20261102.md)。

随后完成的 Qdrant 改造通过 14 项存储回归、8 项真实 HTTP、2 项最终重启持久化与 21 项浏览器检查，另验证非空 ES 迁移及源数据保留；真实/合成数据对比见 [Qdrant 报告](reports/qdrant_20261102/README.md)。精确检索保持默认，原始数据和旧 ES 卷保留。

常见问题：

- `No module named server`：使用 `python app/server.py`，或切换到 `app/` 后执行 uvicorn。
- 高级接口返回 503：LangChain 依赖未安装到 API 所使用的环境。
- HTTP 500 / 模型未初始化：检查权重与 ES 连接，不只看健康接口。
- 模型连接失败：区分端点未启动、模型名不匹配、鉴权失败及沙箱网络限制；不要记录完整异常中的凭据。
- TRT/CPU 选择与预期不符：核对主服务实际设备分支和 provider 日志，而非只看环境变量。
- 模型回答不构成成功凭证：核对 `turn_tool_calls`、任务状态、真实产物和审核记录。

这是面向可信本地操作者的工具服务，不是多租户文件沙箱。API 接受本地路径、会话读取缺少用户隔离；如需联网部署，应外置认证、授权、路径白名单及审计策略。结构化记忆仅过滤部分敏感键，不保证对用户消息、嵌套数据和异常全文脱敏。不要把密钥放进聊天上下文。

备份应覆盖原图、锚定库、已批准标签和修订日志；运行输出清理由人工确认保留周期后执行。本次不删除任何文件、不提交 Git、不创建版本 tag。
