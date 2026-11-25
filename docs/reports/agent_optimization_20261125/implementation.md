# 智能体优化执行记录

本轮在优化分析的基础上实施 P0/P1 的后端工程部分。业务工具沿用既有 TaskQueue、场景执行器和 VisualEvidenceController，原图、正式图库与历史质量报告保持原有版本边界；大规模模型选型与质量验收不包含在本轮完成范围内。

## 已实现

- 共享契约完整校验范围、枚举、必填参数和额外字段；旧入口在兼容类型转换后校验，高级入口使用原类型校验。
- 工具结果区分已提交、执行成功、部分失败、失败、取消与中断；人工审核状态独立表示，写入摘要使用真实结果。
- 运行时原子保护同任务 ID 与跨轮幂等写入，包括并发实例与任务 ID/幂等键交叉冲突；启动恢复把未完成调用标记中断，不自动重放写入。实际后台队列在事务中先检查幂等，再检查新输出，保持旧业务 API 默认输出策略。
- 两聊天入口使用共享工具注册，重型建库、场景分析、整理和人脸任务提交持久队列；支持项目内查询、取消与恢复。
- 视觉观察发送 EXIF 校正 RGB 预览，沿用内容绑定缓存和持久视觉预算；场景分析公开图文检索与视觉策略。
- 项目切换隔离输出参数；新实例刷新业务任务，自动恢复知识库与证据产物路径；已审核证据导出保留原图绑定。
- JSON 适配保留端点 usage，包括返回无效 JSON 后升级模型的用量；高级编排预算对每次实际尝试预留输出、记录失败与级联，并把剩余时间和输出上限传到 SDK 实际请求。旧聊天路由转为线程池执行。
- 敏感键递归过滤；长结果保持合法 JSON；新增离线集成回归与真实模型任务 smoke 工具。

## 验证入口

`tests/test_agent_optimization.py` 的 21 项测试覆盖业务错误状态、无执行完成声明、跨轮去重/输入变化、运行时 ID 冲突/并发/中断恢复、项目重载、脱敏、schema、长结果、失败级联用量和实际 SDK 请求预算、持久知识库任务、取消/恢复、外部任务 ID 拒绝、新输出事务保护、真实预览和视觉缓存、审核证据原图绑定，以及两套真实聊天路由的工具契约和参数响应。

```bash
/home/huachenghao/.conda/envs/agent/bin/python tests/test_langchain_agent.py
/home/huachenghao/.conda/envs/agent/bin/python tests/test_agent_runtime.py
/home/huachenghao/.conda/envs/agent/bin/python tests/test_conversational_agent.py
PYTHONPATH=/home/huachenghao/.conda/envs/agent/lib/python3.11/site-packages /home/huachenghao/.conda/envs/langchain/bin/python tests/test_agent_optimization.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_core_upgrade_tasks.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_organize_tasks.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_visual_evidence.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_workspace.py
```

| 测试 | 通过 | 环境与边界 |
| --- | --- | --- |
| 新智能体集成 | 21 项 | Python 3.11；真实 TaskQueue、临时图片和 ASGI 聊天请求；模型传输替身 |
| LangChain graph | 14 项 | `agent` 环境；模型传输替身 |
| 运行时 / 连续会话 | 6 / 9 组 | `agent` 环境；原有脚本式断言 |
| 持久队列 / 整理执行器 / 视觉证据 | 10 / 18 / 14 项 | 原视觉 Python 3.10；不叠加 Python 3.11 包 |
| 工作台/API | 27 项 | 原视觉 Python 3.10；实际主服务模块导入，不启动 GPU 模型 |

合计 104 项 unittest 与 15 组脚本断言。日志在 `outputs/acceptance/agent_optimization_20261125/regressions/`。

本地 `agent` 环境具备新 LangChain 依赖但没有视觉/API 依赖；`langchain` Python 3.11 环境具备基础视觉/API 依赖但 jsonschema 缺少 attrs，完整服务器还缺少 ultralytics/faiss/onnx。因此集成测试使用同为 Python 3.11 的现有包组合，不修改环境。仅将无关的服务器 GPU 模型管理、图像检索和年龄估计依赖替换为禁止调用的测试替身，实际 `main.py` 聊天路由、参数校验、中间件与智能体实现保持真实；该检查不是全模型服务启动验收。

Python 3.11 ASGI 线程唤醒在沙箱中超时，宿主复验 21 项全部通过，成功日志为 `integration_host.log`；`integration.log` 保留沙箱未完成输出。原视觉 Python 3.10 的 27 项工作台回归首次在沙箱中通过；保存日志的复验在 Matplotlib 字体缓存初始化超时，宿主复验 27 项通过，日志为 `workspace_host.log`，`workspace.log` 保留沙箱未完成输出。工作台日志中的 `test scan error` 是既有失败路径测试主动注入的异常。现有 FastAPI `on_event` 弃用告警不影响断言，本轮未改造其生命周期接口。

部署环境应在同一 Python 环境完整安装 `requirements/vision.txt` 和 `requirements/agent.txt`；不能把不同 Python 小版本的 site-packages 混用。视觉依赖新增 jsonschema，保证旧聊天也具有共享参数校验依赖；纯运行时导入仍不依赖 jsonschema。未替换或安装本机环境、未部署运行镜像。

## 真实模型联调

入口：`tools/validation/evaluate_photo_agent.py`。使用 `.env` 的 `Qwen3.6-27B`，测试目录检查、清单及 pending 状态、防覆盖、缺参数、持久知识库提交/跨轮结果，以及可选真实图片观察。临时目录和任务库位于新评测输出中，不使用正式人脸库，不启动本地 GPU 模型。

沙箱首次调用无法连接端点，失败报告保留在 `outputs/acceptance/agent_optimization_20261125/live_big_sandbox/report.json`；宿主网络初验 `live_big_host/` 为 7/7。此后补充项目名称隔离、恢复状态和实际 SDK 请求预算，因此初验不作为最终源码验收。最终版本另用新目录 `live_big_final/` 复验，7/7 全部通过，不覆盖旧报告；12 份执行源码摘要逐一匹配最终文件。真实图片为 `data/sample_images/2.jpg`（4031×2670），SHA-256 与初验一致；观察输出仍需人工复核，不评价身份或幕次质量。精选验收数据见 [validation.json](validation.json)。

| 真实模型任务 | 通过 | 秒 |
| --- | --- | --- |
| 目录检查 | 是 | 4.617 |
| unknown/pending 清单生成 | 是 | 15.288 |
| 重复写入保持原输出字节 | 是 | 26.245 |
| 缺参数不触发整理 | 是 | 4.497 |
| 持久知识库提交 | 是 | 8.314 |
| 跨轮恢复知识库产物 | 是 | 4.527 |
| 真实图片观察 | 是 | 9.702 |

七个案例的执行时间合计 73.190 秒；编排模型共 13 次端点请求，提供方报告输入 45,359 / 输出 5,629 / 合计 50,988 tokens。视觉模型另有一次实际图片请求，未包含在编排用量内。上述是单次功能 smoke，不是 P50/P95、准确率或相对基线性能结论。

## 预算与恢复边界

- 高级聊天 `model_usage` 记录编排端点尝试和提供方返回的用量；输出预算是保守预留上限，不退还未用额度，不是包含输入 tokens 的账单硬上限。无用量的失败请求仍计入尝试和预留。
- 视觉观察与后台场景视觉调用使用独立的视觉预算，不计入编排模型用量。旧聊天入口本轮只统一契约/业务工具和线程池执行，尚未增加相同的整轮预算接口。
- 时间预算用于限制下一次调用及实际模型请求超时；不能强制终止已执行的任意同步文件工具，持久业务任务也不会随聊天结束被取消。
- 中断的本地写入不自动重放，先核对产物和业务任务；业务恢复沿用 TaskQueue 的逐图检查点与版本校验。当前主服务按既有单任务执行器边界验证，未做多 worker 扩容验收。

## 后续验收

- 构建 60–100 个独立任务比较小模型、直接大模型、级联及 JSON/native，报告任务成功率、人工补充次数、P50/P95 和累计用量；当前 smoke 不足以作模型选型。
- 基于失败类型的精细路由、旧聊天预算对齐以及正式视觉/身份质量评测仍需后续独立验证；本轮不宣称实际准确率或性能提升。
- 工作台对话入口复用当前项目、业务任务轮询和复核页面；本轮尚未实现该界面，也未替换正在运行的应用镜像。
- 会话增量存储和历史分页作为 P2；当前 JSON 会话仍保留全部历史，并提供结构化项目/任务信息。
- SFT/RL 在轨迹版本绑定、可验证终态与独立测试具备后再实施；当前未开展训练或重启历史 U2 性能实验。

历史 `probe.py/findings.json` 保留优化前行为，修复后这些旧缺口断言预期失败；新行为由正式回归验证，不改写历史记录。

## 本地提交

按用户要求分四批使用英文 Conventional Commits：`930417a` 为契约与执行状态，`6a35f56` 为持久业务与视觉预算，`006aea7` 为回归与真实模型评测；本报告、蓝图和工作状态随第四批文档提交保存，完整哈希以 `git log -4` 为准。源码和测试字节与验收摘要一致，提交阶段未重复模型调用；`validation.json` 的 `committed_at_validation=false` 保留验收时的未提交状态。原始评测输出与 `.env` 不纳入 Git，未推送或部署。
