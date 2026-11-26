# 高级智能体蓝图（LangChain）

## 目标

基于现有受限工具和任务审计，新建一个可连续对话的高级智能体。智能体分析用户需求，自主选择并串联工具，完成戏剧照片整理相关的只读检查、清单生成、知识库构建、场景候选、复核准备和完整整理。

## 2026-11-25 优化执行蓝图

依据[优化分析](../reports/agent_optimization_20261125/README.md)，本轮先完成后端 P0/P1：共享工具契约、可信执行结果、持久业务任务、视觉能力、项目记忆和模型预算。实现记录与验证入口见[执行报告](../reports/agent_optimization_20261125/implementation.md)。工作台对话界面、增量会话存储、大规模模型比较及 SFT/RL 作为后续阶段分别验收。

| 阶段 | 交付 | 验收重点 |
| --- | --- | --- |
| A：契约与结果 | `agent_contracts.py`；两聊天入口共享注册表 | 范围/枚举/额外参数一致校验；全部失败与部分失败不误报成功 |
| B：持久执行 | `PhotoTaskService` 复用 `submit_business_task` 与 TaskQueue | 跨轮幂等；新输出占用；查询、取消与恢复；提交与完成分开 |
| C：多模态 | 持久场景分析与 `observe_photo` | 真实图片输入；已有视觉预算/缓存；证据类型及人工复核边界 |
| D：项目连续性 | 项目记忆、任务快照、产物映射 | 切换目录不串用输出；新实例读取任务结果并恢复知识库路径 |
| E：成本与评测 | `AgentBudget`、离线回归与真实模型 smoke 入口 | 每次实际端点尝试计数；保留 usage；任务状态与产物验证 |

重型工具在主服务中提交既有持久业务队列；智能体 JSON 记录负责调用关联，TaskQueue 负责逐图检查点、版本检查和业务恢复。等待任务时结束当前聊天，下一轮或调用 `get_task` 读取实际状态，不连续轮询耗费模型预算。原同步业务接口仍保留。

## 运行方式

- 依赖：`requirements/agent.txt`；可用 Conda 环境 `agent`。
- 对话模式：`python -m app.photo_agent_cli --model auto --protocol json --max-steps 6`
- 单轮模式：`python -m app.photo_agent_cli --session live --message '检查目录并生成清单' --context '{"input_dir":"/path/photos","output_file":"/path/new.jsonl"}'`
- 主服务接口：`POST /api/organize/agent/advanced/chat`
- 工具契约：`GET /api/organize/agent/advanced/tools`
- API 参数：`model=auto|small|big`、`protocol=json|native`、`max_steps=1..8`、`session_id`、`context_json`、`turn_timeout=1..900`、`output_token_budget=256..65536`
- 高级入口默认时间预算 180 秒、编排模型最大输出预留 8192 tokens；每个实际端点尝试预留最多 2048 tokens，包括失败与级联，实际 SDK 请求使用剩余时间/输出上限。不是包含输入 tokens 的账单硬上限，也不能强制停止同步文件工具。大流程可显式提高预算。视觉调用另由视觉策略管理，不计入聊天编排模型的 usage；旧聊天入口尚未对齐整轮预算。

## 模型

默认 `auto` 使用小模型优先、大模型兜底。`.env` 已配置：

- 小模型：`small_diag_url`、`small_diag_model`、`small_diag_key`
- 大模型：`url`、`big_model_name`、`key`
- 可用 `AGENT_SMALL_URL`、`AGENT_BIG_URL` 等环境变量覆盖同名运行时值。

本地 2B 权重位于 `~/models/MiniCPM5-2B`。已发现的服务环境为 `mff_sft_serve`（最初提供的 `mff_sft_server` 名称未在环境列表出现）；在有 GPU 权限的环境启动 OpenAI 兼容服务。参考命令：

```bash
/home/huachenghao/.conda/envs/mff_sft_serve/bin/vllm serve ~/models/MiniCPM5-2B \
  --served-model-name mff-sft-minicpm5-2b --host 127.0.0.1 --port 3761 \
  --max-model-len 32768 --gpu-memory-utilization 0.5
```

历史 2B vLLM 启动需要具备 GPU 权限的宿主机或容器。当前大模型联调使用 `.env` 中已部署的服务；无需启动新的本地 GPU 模型。沙箱网络若无法连接该端点，保留失败记录并在允许的宿主网络复验。

`protocol=json` 适合不支持原生 tool calls 的模型：LangChain 循环把模型 JSON 行动转换成 tool calls，把 ToolMessage 转换成数据消息。`protocol=native` 直接使用 OpenAI 兼容工具调用。

## 工具

| 工具 | 用途 | 安全边界 |
| --- | --- | --- |
| `inspect_photos` | 只读目录、计数和文件名元数据 | 非递归、非视觉识别 |
| `scan_dir_and_tag` | 生成新 JSONL，场景为 `unknown` | 输出必须不存在 |
| `knowledge_summary` | 读取实体和角色 | 只读 |
| `build_play_knowledge` | 从本地剧目文本构建知识库；服务模式返回业务任务 | 不抓网络，新目录，查询完成后使用产物 |
| `scene_reason` | 服务模式提交场景分析，支持图文检索和视觉观察；CLI 文件模式使用文件名 | 新 JSONL；视觉服务需配置；候选必须人工复核 |
| `review_summary` | 查询复核状态 | 只读 |
| `prepare_review` | 新建待复核队列 | 模型批准强制重置为 pending |
| `organize_photos` | 提交持久整理任务 | 主 API 环境；新目录；只采纳人工批准场景证据 |
| `get_capabilities` | 查询实际服务能力、模型配置和检索后端 | 不加载模型、不返回凭据 |
| `get_task` | 查询业务状态与产物 | 仅当前会话项目已关联任务；排队/运行不等于完成 |
| `cancel_task` / `resume_task` | 取消或恢复业务任务 | 项目关联校验；复用既有取消边界与版本检查 |
| `observe_photo` | 明确指定单图的画面事实观察 | 实际图像预览输入；视觉预算、内容缓存；不猜身份或批准幕次 |
| `build_face_database` | 从明确演员标注照片提交人脸建库 | 人脸库和报告均为新目录，不修改已有正式图库 |
| `scan_faces` | 提交人脸聚类候选扫描 | 新目录；不自动赋予演员身份 |
| `export_review` | 导出人工批准场景证据到新 JSONL | 没有批准记录则拒绝；保留原图绑定和复核来源 |

`scene_reason` 服务参数包括 `use_embedder`、`use_llm` 和受限 `visual_policy`。显式单图观察可选择 `mode=on_demand`；默认 automatic 策略会限制视觉请求比例，小输入可能没有自动请求。指定视觉能力但服务未配置时返回错误，不把文件名分析当成视觉结果。独立 CLI 默认保留文件工具，重型任务和 `observe_photo` 由主服务提供。

## 会话与记忆

会话持久化在 `outputs/agent/conversations/<session_id>.sqlite3`。每步在同一事务中追加新消息和工具调用，更新会话元数据及有变动的业务任务快照。续聊读取最近的提示和滚动摘要窗口，不加载全部消息或调用。`memory`、`projects` 和已核验产物的含义不变，仍会刷新当前项目关联的业务任务；这不意味着单轮成本与项目任务数无关。

升级前停止所有旧 worker 并备份 `outputs/agent/`。首次访问每个旧会话或运行时目录时，在事务中导入原 JSON；不删除、不覆盖、不再双写旧文件。导入失败回滚，可修复旧文件后重试。迁移后的权威记录是 SQLite；不要混用新旧写入进程。回退旧代码只能看到原快照，不能恢复升级后的新轮次。备份 SQLite 请停写复制或使用 SQLite 备份 API。

`GET /api/organize/agent/session/{session_id}/history?kind=messages&limit=24` 返回最近一页，`items` 按时间正序；将非空 `next_before` 原样作为下次 `before` 获取更早记录。`kind=tool_calls` 独立分页，游标不能混用；默认每页 30 条，最多 100 条，未知会话返回 404。最新消息页还提供当前项目的任务摘要和产物，任务状态以后台队列为准。原无 `/history` 的 GET 接口保留完整快照，仅供显式全量读取。

调用记录与审计保存在 `outputs/agent/tasks/`，实际业务任务保存在 `TRANS_TASK_DIR` 指定的 TaskQueue 数据库。写入幂等键绑定会话、项目、工具 schema、参数及输入文件状态；同 ID 不同参数拒绝，相同写入请求复用记录。后台队列继续执行完整哈希与版本校验。只读查询允许状态刷新；取消/恢复不使用永久写入去重。

返回值中 `outcome.status` 分别为 `submitted`、`succeeded`、`partial`、`failed`、`cancelled`、`interrupted`。`submitted` 仅表示任务已受理，`partial`/`failed` 的 `ok=False`；人工复核通过独立字段表示。两聊天入口的 `tool_calls`、`turn_tool_calls` 和 `operation_report` 只含本轮；`workflow`、`execution_report` 是当前项目状态，`message_count` 仍为完整消息数。`session_file` 现在指向数据库。高级入口的 `model_usage` 保存编排模型尝试、用量及预算。写入操作后的摘要由真实工具结果生成，普通问答仍是模型文本，不能把自由文本当作业务终态。

存储与消息边界递归过滤敏感键，并处理带键名的文本赋值；保留非凭据的幂等键、缓存键和 token usage。该规则不能识别所有无键名凭据或任意自由文本密钥，聊天中仍不应传递凭据。长工具结果先结构化缩减，再生成合法 JSON，产物引用保留；完整结果仍在任务与会话记录中。

## 验收

```bash
/home/huachenghao/.conda/envs/agent/bin/python tests/test_langchain_agent.py
/home/huachenghao/.conda/envs/agent/bin/python tests/test_agent_runtime.py
/home/huachenghao/.conda/envs/agent/bin/python tests/test_conversational_agent.py
python tests/test_agent_optimization.py
python tools/validation/evaluate_photo_agent.py --output /path/new-run --model big --image /path/photo.jpg
```

离线 graph 测试替换模型传输层；新增集成回归使用真实 TaskQueue 和临时图片，检查幂等、项目切换、业务失败、任务/知识库产物、审核导出及视觉缓存。真实模型评测使用新输出目录、隔离任务库和 `.env` 端点，可选发送实际样例图；报告保存源码摘要。真实模型 smoke 只证明链路和任务行为，不替代幕次、身份或画面事实的独立质量评测。
