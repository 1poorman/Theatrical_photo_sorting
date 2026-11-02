# Docker 功能验收（2026-11-02）

本轮最终验收全部通过。上轮人脸识别的测试样本问题已澄清，高级智能体的 system 消息兼容性问题已修复，并完成镜像重建及真实接口复测。

| 验证层 | 结果 | 覆盖内容 |
| --- | --- | --- |
| 容器真实 HTTP / 模型 | 18/18 | 人脸建库与识别、检测/分割/姿态/修复、SigLIP2 与 ResNet50 索引检索、相似分组、知识库与场景复核导出、后台整理、建库预演及实际写入、人物视图、人脸聚类、离线和高级智能体 |
| 浏览器工作流 | 21/21 | 35 张真实剧照、草稿恢复、复核冲突、导出、任务交互、独立项目；1440/1024/768/540/390 宽度；10,000 条复核记录按需加载 |
| 重启 / 重建与恢复 | 8/8 | ES 重启后 512 维向量检索、FAISS 重载、7 份文件哈希、3 条复核记录、5 个已完成任务、会话及簇命名保留；取消任务以原 ID 恢复并完成 35 张照片分析；真实智能体工具调用 |
| 智能体回归 | 14/14 | 新增单一 system 消息及结构化 system 内容保留检查；既有级联、参数校验、调用预算、失败不重放和人工复核边界检查 |

人脸检测失败的旧样本是 160×160 已裁剪库图。标准库图入库会跳过检测，因此“能入库”不保证在该裁剪图上再次检出。新验收使用完整单人脸剧照，真实完成 SCRFD 检测、ArcFace 特征提取、ES 入库与身份匹配，未调整识别阈值。

高级智能体原来将 JSON 协议说明和 graph 规则分别作为 system 消息发送。当前模型服务返回 `400 - System message must be at the beginning.`，随后进入回退。修复将系统内容合并至首条消息，保留规则、上下文及工具结果语义。最终 HTTP 验收实际调用 `inspect_photos`，返回 3 张照片且 `fallback=false`。

运行环境沿用 `.env.docker` 和 `docker-compose.llm.yml`：物理 GPU1 在容器内映射为 `cuda:0`，RTX 3090，TensorRT / CUDA 人脸推理；PyTorch 2.9.0+cu128、ONNX Runtime GPU 1.23.2、LangChain 1.4.7、langchain-openai 1.7.3、Elasticsearch 8.12.2。实际验收容器镜像为 `sha256:dc602d52befe9b29604cab344193996efedd0d2f8ac1c847fc5157e5f4068289`，容器内修复文件哈希与工作区一致。结束时应用和 ES 均健康，没有活动后台任务，临时 ES 记录与测试索引已删除。

验收证据保存在本机忽略目录 `outputs/acceptance/docker_20261102/`：

- [最终 HTTP 报告](../../outputs/acceptance/docker_20261102/http_015754/report.json)
- [持久化与恢复报告](../../outputs/acceptance/docker_20261102/persistence_report.json)
- [浏览器报告](../../outputs/acceptance/docker_20261102/browser_report.json)及[桌面截图](../../outputs/acceptance/docker_20261102/browser/review-desktop.png)
- [智能体回归日志](../../outputs/acceptance/docker_20261102/agent_regression.log)
- [运行版本](../../outputs/acceptance/docker_20261102/runtime.json)及[容器状态](../../outputs/acceptance/docker_20261102/container_state.json)

本轮验证功能链路和恢复契约。HTTP 主样本为 3 张缩放剧照，身份匹配使用同一测试照片建库及查询，不代表跨照片识别准确率。浏览器中的 GPU 任务交互使用模拟响应，真实推理由独立 HTTP 验收覆盖。复核标签来自验收脚本，不是新增人工真值。大模型实测覆盖当前已配置端点的 JSON 工具协议；未配置的小模型及真实 native 协议未在本轮验收。
