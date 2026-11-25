# 文档中心

更新时间：2026-11-12。按阅读目的分类；当前交接点见 [WORK_STATE](../WORK_STATE.md)，项目入口见 [README](../README.md)。

## 按需求查找

| 需求 | 入口 | 内容 |
| --- | --- | --- |
| 安装、部署、接手维护 | [项目说明](PROJECT_GUIDE.md) | 目录职责、环境、配置与操作 |
| 使用功能或复现实验 | [使用指南](guides/README.md) | 工作台、人脸与年龄、复核、场景检索、向量库、预处理 |
| 理解设计和实施范围 | [设计方案](design/README.md) | 整理流水线、多模态、智能体、核心升级、Qdrant |
| 查执行状态和未完成项 | [进度记录](progress/README.md) | 核心升级、多模态历史、U2 收尾 |
| 查实测数据和技术报告 | [报告索引](reports/README.md) | 模型对比、权重压缩、性能、容器与完整技术报告 |
| 查早期背景 | [历史资料](archive/README.md) | 早期实施计划和优化实验 |
| 查版本变化 | [迭代记录](CHANGELOG.md) | 日期、变更、提交与验证边界 |
| 找命令行脚本 | [工具目录](../tools/README.md) | 47 个脚本，按 8 个主题列出用途和入口 |

## 常用入口

- [工作台操作](guides/workspace_guide.md)、[SwinFace 年龄估计](guides/swinface_guide.md)、[Qdrant 使用与迁移](guides/qdrant_guide.md)。
- [最新完整技术报告](reports/technical_report_20261026.md)：29 页，[PDF](reports/technical_report_20261026/technical_report.pdf) 与离线资源。
- [SwinFace 对比](reports/swinface_20261111/README.md)、[权重精简与量化](reports/swinface_quantization_20261111/README.md)、[人脸技术栈选型](reports/face_stack_20261106/README.md)。
- [目录结构与旧路径迁移](guides/repository_layout.md)：新增文件归类规则、历史命令更新方法。

## 维护约定

顶层保留本索引、项目说明和 CHANGELOG；新增材料放入对应分类并登记到分类索引。指南说明操作，设计说明目标，进度记录状态，报告保存测量条件、结果和局限。

报告包中的 PDF、HTML、插图、来源快照继续一起保存；原始实验输出在 `outputs/`，原图、人脸库和权重分别留在 `data/`、`weights/`。历史实验结论按当时版本阅读；变更目录不构成重新测量或验收通过。
