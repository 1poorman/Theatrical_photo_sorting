# 工具与文档目录

2026-11-12 将原 `tools/` 顶层 47 个脚本分为 8 类，将 `docs/` 顶层 22 份专题文档归入子目录；文档顶层保留 README、PROJECT_GUIDE 和 CHANGELOG。

```text
tools/
  README.md          工具用途与完整清单
  face/              人脸模型、年龄、聚类任务与恢复
  vector/            向量库基准、迁移与维护
  scene/             场景文本、候选池与检索评测
  review/            人工复核包、标签导入与质量统计
  vision/            视觉观察、OCR、时序与分组实验
  performance/       特征流水线、预处理、响应与流程成本
  validation/        数据清单、核心正确性与整链路验收
  maintenance/       仓库范围与远端模型元数据审计
docs/
  README.md          文档入口
  PROJECT_GUIDE.md   项目说明、安装与维护
  CHANGELOG.md       迭代记录
  guides/            操作方法与复现命令
  design/            蓝图、设计与升级方案
  progress/          执行过程、状态与收尾
  reports/           技术报告、实测数据和随附资源
  archive/           早期计划与历史优化背景
```

## 查找旧路径

工具文件名均保留，旧 `tools/<文件名>.py` 改为 `tools/<分类>/<文件名>.py`；[工具清单](../../tools/README.md)逐一列出新位置。Python 导入同步改为 `tools.<分类>.<模块>`。例如：

| 旧路径或导入 | 新路径或导入 |
| --- | --- |
| `tools/estimate_face_age.py` | [tools/face/estimate_face_age.py](../../tools/face/estimate_face_age.py) |
| `tools/benchmark_feature_pipeline.py` | [tools/performance/benchmark_feature_pipeline.py](../../tools/performance/benchmark_feature_pipeline.py) |
| `tools/migrate_vector_store.py` | [tools/vector/migrate_vector_store.py](../../tools/vector/migrate_vector_store.py) |
| `from tools.scene_text_blocks import ...` | `from tools.scene.scene_text_blocks import ...` |

专题文档文件名也保留，旧 `docs/<文件名>.md` 按下表查找：

| 原文件类型 | 新目录与完整索引 |
| --- | --- |
| `*_guide.md` | [guides/](README.md) |
| `*_blueprint.md`、`qdrant_upgrade_plan.md` | [design/](../design/README.md) |
| `*_progress.md`、`u2_closeout_20261029.md` | [progress/](../progress/README.md) |
| `technical_report_20261024.md`、`technical_report_20261026.md` | [reports/](../reports/README.md) |
| `optimization_report.md`、`实施计划.md` | [archive/](../archive/README.md) |

仓库内现行命令、导入、OCR 子进程和 Docker 基准入口均使用新路径。个人脚本、终端历史或自动化中的旧命令需按表更新。

## 历史证据与新增文件

`outputs/` 中的原始运行产物、冻结源码和协议保留原路径与摘要。已完成报告中的测量 JSON、PDF、离线 HTML 及图片保留原始字节；可维护的 Markdown 导航、重建脚本和报告包文件清单随目录调整。复现旧版本时使用报告记录的 Git 提交或源码快照；源码摘要绑定的旧协议仍会拒绝当前不同版本，不能靠改摘要继续旧实验。

新增工具按所服务的主题归类：人脸基准放 `face/`，向量库基准放 `vector/`，跨功能性能放 `performance/`。脚本从项目根目录运行，提供 `--help`；供脚本共用的模块与同主题工具放在一起。更新[工具索引](../../tools/README.md)，避免重新堆积到顶层。

新增文档在分类索引登记；报告使用日期目录，正文、图表和来源快照一起保存。原图、图库、权重、运行输出继续使用现有数据目录。
