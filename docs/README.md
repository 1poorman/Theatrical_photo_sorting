# 文档中心

更新时间：2026-10-26。文档基于当前工作区与 Git 历史整理；代码和测试文件保留原路径，避免破坏导入、样本路径及历史链接。

异常中断后继续或协作接手，先读 [工作状态与中断接续](../WORK_STATE.md)，其中记录当前优先级、未提交改动、验证边界与下一步命令。

## 推荐阅读顺序

1. [项目 README](../README.md)：能力概览、快速开始、API 清单。
2. [项目说明](PROJECT_GUIDE.md)：文件职责、运行配置、业务操作、数据与维护。
3. [版本迭代记录](CHANGELOG.md)：已提交历史与未发布改动。
4. 按需求阅读以下设计与验收资料。

## 文档分层

| 文档 | 定位 | 使用方式 |
| --- | --- | --- |
| [WORK_STATE.md](../WORK_STATE.md) | 当前开发交接快照 | 异常恢复、关键节点、未提交改动与多智能体协作 |
| [technical_report_20261026.md](technical_report_20261026.md) | 最新技术评审报告（29页PDF、离线HTML、15幅正文图） | 功能架构、14种检索变体、Pillow修复、双线程选优与DALI评价；PNG、SVG和31份证据快照随Git保存 |
| [technical_report_20261024.md](technical_report_20261024.md) | 历史技术报告（26页、13幅正文图） | 保留当时数字与结论；PDF、HTML及图片已迁入Git可见归档 |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | 当前项目手册 | 部署、接手开发、排障与文件管理 |
| [workspace_guide.md](workspace_guide.md) | 多模态整理工作台 | 图片浏览、剧情关联、看图复核、整理导出与界面验收 |
| [identity_registry_guide.md](identity_registry_guide.md) | 演员身份与剧目角色表 | JSON 格式、身份与角色来源、专辑筛选、API 参数和任务恢复 |
| [sourced_scene_candidates_guide.md](sourced_scene_candidates_guide.md) | 有出处的分幕候选与U4复现 | 官方/媒体资料、精确引用、短原文块/固定向量25/28对照、来源v1与演出版边界 |
| [photo_partition_review_guide.md](photo_partition_review_guide.md) | 剧照划分与图库来源复核 | 原/整理副本、EXIF相机时间、近重复/连拍线索、复核草稿与262条新版幕次标签 |
| [face_identity_review_guide.md](face_identity_review_guide.md) | 逐脸身份标注 | 1095张脸的原图位置、已知/未知/不可评价与镜像例外、分页草稿和校验 |
| [visual_fact_review_guide.md](visual_fact_review_guide.md) | 画面事实与OCR双人复核 | 100图/1352条描述、5图独立转录、分歧与错误分母、草稿和开发集评测 |
| [experimental_components_guide.md](experimental_components_guide.md) | 视觉证据、时序与图聚类实验 | 请求/token 预算、OCR、窗口分段、图聚类、连拍研究及实际验收边界 |
| [core_upgrade_blueprint.md](core_upgrade_blueprint.md) | 核心算法与性能后续蓝图 | U0–U8 实施依赖、开源选型、数据划分、验收门槛与默认采用规则 |
| [core_upgrade_progress.md](core_upgrade_progress.md) | 核心升级执行状态 | 全目录清单、正确性修复、真实模型探索、配置与后续待验收项 |
| [CHANGELOG.md](CHANGELOG.md) | 可追溯迭代记录 | 日期、提交、变更与验证边界 |
| [agentic_tools_blueprint.md](agentic_tools_blueprint.md) | 当前 LangChain 智能体实现说明 | 独立 CLI、主 API、模型配置、工具契约 |
| [smart_organize_blueprint.md](smart_organize_blueprint.md) | 智能整理设计 | 去重、建库、演员视图与流水线背景 |
| [multimodal_context_blueprint.md](multimodal_context_blueprint.md) | M11–M16 设计与验收目标 | 证据融合、知识库与人工审核边界 |
| [multimodal_progress.md](multimodal_progress.md) | 多模态实施与历史验收记录 | 查找报告、已知问题与后续工作 |
| [optimization_report.md](optimization_report.md) | 历史性能实验 | 理解 TRT、批量检索和性能取舍；不视为当前机器基准 |
| [实施计划.md](实施计划.md) | 早期实施背景 | 保留历史方案；未完成规划不代表现有功能 |

## 文件整理原则

- 入口和总览集中在 README 与本目录；不再在不同蓝图中重复维护完整 API 清单。
- 源码保留在 `app/`、`core_modules/`、`config/`；文档不代替代码契约。
- 原图、人脸库、模型权重和运行输出不迁移、不删除；管理规则见项目说明。
- `tests/tmp_*.py` 属于本地实验/历史补丁，不作为安装步骤，也不自动执行或批量清理。
- 运行报告放在 `outputs/`；文档记录测试条件、报告位置、结果和局限，不复制密钥或原始人脸数据。
- 可公开阅读的技术报告放在 `docs/reports/`：Markdown使用相对PNG链接，PDF、离线HTML、可编辑SVG和被引用的JSON快照一并提交；本地原图、模型及实验数据库仍留在原目录。

## 后续维护

新增功能时同时更新 API/CLI 使用说明、对应蓝图和 CHANGELOG 的“未发布”部分。发布时记录真实 Git tag/commit、环境、测试命令、兼容性和迁移步骤；没有 tag 时使用日期与 commit，不补造语义版本。
