# 技术与实验报告

[返回文档中心](../README.md) · [使用指南](../guides/README.md)

## 智能体

| 日期 | 报告 | 结论与范围 |
| --- | --- | --- |
| 2026-11-25 | [智能体优化分析](agent_optimization_20261125/README.md) | 工具能力、持久执行、结果校验、记忆及评测；三组离线测试与 11 个边界探针 |
| 2026-11-25 | [智能体优化执行](agent_optimization_20261125/implementation.md) | 共享契约、持久任务、跨轮幂等、项目连续性与真实图片观察；后端工程交付，质量/选型与 UI 另行验收 |

## 模型与存储

| 日期 | 报告 | 结论与范围 |
| --- | --- | --- |
| 2026-11-11 | [SwinFace 权重精简与量化](swinface_quantization_20261111/README.md) | 推荐 61.8 MiB FP16；比较识别一致性、年龄输出偏差、速度与显存 |
| 2026-11-11 | [SwinFace 对比与年龄功能](swinface_20261111/README.md) | 109 脸开发集，识别默认保留 ArcFace；年龄 GT 为 0 |
| 2026-11-06 | [人脸技术栈选型](face_stack_20261106/README.md) | 检测/识别模型族、融合、存储选型与许可边界 |
| 2026-11-02 | [Qdrant 容器对比](qdrant_20261102/README.md) | 真实与合成向量的召回、延迟、内存和迁移验证 |
| 2026-11-02 | [Docker 功能检查](docker_functional_20261102.md) | 容器内工作台、API 与现有模型功能 |

## 性能与流程成本

| 日期 | 报告 | 结论与范围 |
| --- | --- | --- |
| 2026-10-28 | [U2 并行增量](u2_parallel_20261028/README.md) | 200 图 × 5 次，特征 1.527×；共享负载和正式验收边界保留 |
| 2026-10-28 | [U2/U3 流程成本](u2_u3_costs_20261028/README.md) | 复核快照、导出、HTTP 流程与成本口径 |

## 完整技术报告

| 版本 | 阅读入口 | 随附资源 |
| --- | --- | --- |
| 2026-10-26 | [完整技术报告](technical_report_20261026.md) | 29 页、15 幅正文图；[PDF](technical_report_20261026/technical_report.pdf)、[离线 HTML](technical_report_20261026/technical_report.html)、[报告包与重建方法](technical_report_20261026/README.md) |
| 2026-10-24 | [历史技术报告](technical_report_20261024.md) | 26 页、13 幅正文图；[PDF](technical_report_20261024/technical_report.pdf)、[离线 HTML](technical_report_20261024/technical_report.html)、[报告包](technical_report_20261024/README.md) |

每个日期报告包保存正文引用的 JSON、插图和必要脚本；原始运行产物留在 `outputs/`。目录调整时维护相对链接和报告包清单，不改写历史测量数据、来源摘要或原始输出。
