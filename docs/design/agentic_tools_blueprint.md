# 高级智能体蓝图（LangChain）

## 目标

基于现有受限工具和任务审计，新建一个可连续对话的高级智能体。智能体分析用户需求，自主选择并串联工具，完成戏剧照片整理相关的只读检查、清单生成、知识库构建、场景候选、复核准备和完整整理。

## 运行方式

- 依赖：`requirements/agent.txt`；可用 Conda 环境 `agent`。
- 对话模式：`python -m app.photo_agent_cli --model auto --protocol json --max-steps 6`
- 单轮模式：`python -m app.photo_agent_cli --session live --message '检查目录并生成清单' --context '{"input_dir":"/path/photos","output_file":"/path/new.jsonl"}'`
- 主服务接口：`POST /api/organize/agent/advanced/chat`
- 工具契约：`GET /api/organize/agent/advanced/tools`
- API 参数：`model=auto|small|big`、`protocol=json|native`、`max_steps=1..8`、`session_id`、`context_json`

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

注意：当前 Codex 沙箱无法访问 NVIDIA 驱动（`nvidia-smi` 不可用），因此 2B vLLM 启动和真实模型端到端推理必须在宿主机或 GPU 容器中执行。本仓库内的 LangChain 循环验收通过脚本化模型完成。

`protocol=json` 适合不支持原生 tool calls 的模型：LangChain 循环把模型 JSON 行动转换成 tool calls，把 ToolMessage 转换成数据消息。`protocol=native` 直接使用 OpenAI 兼容工具调用。

## 工具

| 工具 | 用途 | 安全边界 |
| --- | --- | --- |
| `inspect_photos` | 只读目录、计数和文件名元数据 | 非递归、非视觉识别 |
| `scan_dir_and_tag` | 生成新 JSONL，场景为 `unknown` | 输出必须不存在 |
| `knowledge_summary` | 读取实体和角色 | 只读 |
| `build_play_knowledge` | 从本地剧目文本构建知识库 | 不抓网络，新目录 |
| `scene_reason` | 文件名 + 知识库生成场景候选 | 标记非视觉，需人工复核 |
| `review_summary` | 查询复核状态 | 只读 |
| `prepare_review` | 新建待复核队列 | 模型批准强制重置为 pending |
| `organize_photos` | 完整整理管线 | 仅主 API 环境；不删除原图 |

## 会话与记忆

会话持久化在 `outputs/agent/conversations/<session_id>.json`。跨轮记忆保存用户提供的输入、输出、知识库等路径，部分顶层敏感键会过滤；这不等于消息、嵌套数据或异常全文脱敏，不应通过聊天传递密钥。任务结果、工具调用和状态审计分别保存在会话文件与 `outputs/agent/tasks/agent_audit.jsonl`。

## 验收

```bash
/home/huachenghao/.conda/envs/agent/bin/python tests/test_langchain_agent.py
/home/huachenghao/.conda/envs/agent/bin/python tests/test_agent_runtime.py
/home/huachenghao/.conda/envs/agent/bin/python tests/test_conversational_agent.py
```

当前 12 项 LangChain 智能体测试通过：真实 LangChain graph 多步调用、跨实例记忆、JSON 级联、原生 tool calls、参数修复、防覆盖、失败不重放、预算限制、重复成功阻断、人工审核强制、模型错误详情和敏感信息过滤。
