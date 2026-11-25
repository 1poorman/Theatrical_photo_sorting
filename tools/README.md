# 工具目录

47 个脚本按主题归类。命令从项目根目录运行；使用安装了对应依赖的 Python 环境。

[文档中心](../docs/README.md) · [目录迁移说明](../docs/guides/repository_layout.md)

## 按任务选择

| 分类 | 数量 | 用途 |
| --- | ---: | --- |
| [人脸与年龄](face/) | 10 | 模型、权重导出和人脸任务验证 |
| [向量库](vector/) | 4 | 在线存储基准、迁移与容量维护 |
| [场景检索](scene/) | 8 | 有出处文本、候选池和检索协议 |
| [人工复核与标签](review/) | 6 | 准备复核材料、校验决定并计算质量 |
| [视觉证据与分组实验](vision/) | 7 | 观察、OCR、时序与图聚类研究 |
| [性能与流程成本](performance/) | 5 | 跨功能吞吐、预处理、响应和导出成本 |
| [清单与整链路验收](validation/) | 5 | 数据清点、核心正确性和流程检查 |
| [仓库与环境维护](maintenance/) | 2 | 仓库范围和远端模型元数据审计 |

## 运行方式

```bash
python tools/face/estimate_face_age.py --help
python tools/vector/migrate_vector_store.py --help
python -m tools.performance.benchmark_feature_pipeline --help
python tools/maintenance/audit_repository_scope.py
```

Python 导入使用 `tools.<分类>.<模块>`。下表标为“共用模块”的文件由同主题脚本导入，其余脚本提供命令行入口。模型基准和真实流程检查需要本地权重、数据或对应服务，具体参数与前提见 `--help` 及关联指南。

原 `tools/<文件名>.py` 的文件名保持不变，下列链接给出新的分类路径。历史输出中的源码路径和摘要仍记录当时版本；复现方法见[迁移说明](../docs/guides/repository_layout.md)。

## 人脸与年龄

| 脚本 / 模块 | 用途 |
| --- | --- |
| [estimate_face_age.py](face/estimate_face_age.py) | 图片或目录逐脸年龄估计，输出 JSON 与预览 |
| [export_swinface.py](face/export_swinface.py) | 导出识别＋年龄 FP32 / FP16 / CPU INT8 权重 |
| [benchmark_face_stack.py](face/benchmark_face_stack.py) | 人脸检测与识别模型族对比 |
| [benchmark_swinface_quantization.py](face/benchmark_swinface_quantization.py) | 权重大小、识别一致性、年龄偏差及速度对比 |
| [check_face_age_workflow.py](face/check_face_age_workflow.py) | 真实年龄 HTTP 功能检查 |
| [benchmark_face_task_http.py](face/benchmark_face_task_http.py) | 人脸任务 HTTP 响应与隔离输入检查 |
| [benchmark_face_task_completion.py](face/benchmark_face_task_completion.py) | 任务取消、继续和发布恢复 |
| [benchmark_face_cluster_recovery.py](face/benchmark_face_cluster_recovery.py) | 人脸聚类恢复与结果等价检查 |
| [benchmark_face_database_recovery.py](face/benchmark_face_database_recovery.py) | 标注图库建库恢复与结果等价检查 |
| [benchmark_cluster_previews.py](face/benchmark_cluster_previews.py) | 持久任务预览生成对照 |

相关说明：[使用指南](../docs/guides/swinface_guide.md)。

## 向量库

| 脚本 / 模块 | 用途 |
| --- | --- |
| [benchmark_vector_store.py](vector/benchmark_vector_store.py) | FAISS / Qdrant / ES 的真实向量与合成规模对照 |
| [check_qdrant_workflow.py](vector/check_qdrant_workflow.py) | 在线向量、迁移和重启 HTTP 验收 |
| [migrate_vector_store.py](vector/migrate_vector_store.py) | 将历史 FAISS / ES 向量复制到 Qdrant |
| [prune_vector_generations.py](vector/prune_vector_generations.py) | 列出或显式清理索引所属的旧集合 |

相关说明：[使用指南](../docs/guides/qdrant_guide.md)。

## 场景检索

| 脚本 / 模块 | 用途 |
| --- | --- |
| [prepare_scene_text_block_study.py](scene/prepare_scene_text_block_study.py) | 冻结文本分块、候选池和输入协议 |
| [evaluate_scene_text_block_study.py](scene/evaluate_scene_text_block_study.py) | 长文/短块与两种候选池评测 |
| [prepare_scene_retrieval_study.py](scene/prepare_scene_retrieval_study.py) | 准备新版检索研究，绑定历史向量 |
| [evaluate_scene_retrieval_study.py](scene/evaluate_scene_retrieval_study.py) | 检索、分类和冻结向量复用评测 |
| [evaluate_verified_scene_retrieval.py](scene/evaluate_verified_scene_retrieval.py) | 根据已确认图级标签评测跨剧目检索 |
| [scene_text_blocks.py](scene/scene_text_blocks.py) | 共用模块：原文块、向量归一化与候选评分 |
| [scene_retrieval_study.py](scene/scene_retrieval_study.py) | 共用模块：研究协议与分类指标 |
| [sourced_scene_candidates.py](scene/sourced_scene_candidates.py) | 共用模块：来源摘要、引文和候选映射 |

相关说明：[使用指南](../docs/guides/sourced_scene_candidates_guide.md)。

## 人工复核与标签

| 脚本 / 模块 | 用途 |
| --- | --- |
| [prepare_photo_partition_review.py](review/prepare_photo_partition_review.py) | 照片来源、划分、近重复与离线复核包 |
| [import_verified_photo_labels.py](review/import_verified_photo_labels.py) | 从已确认整理文件名导入图级标签 |
| [prepare_face_identity_review.py](review/prepare_face_identity_review.py) | 逐脸框、身份复核材料与决定校验 |
| [evaluate_face_identity_review.py](review/evaluate_face_identity_review.py) | 用人工标签评测冻结的人脸划分 |
| [prepare_visual_fact_review.py](review/prepare_visual_fact_review.py) | 画面事实/OCR 双人复核及分歧评测 |
| [review_quality_metrics.py](review/review_quality_metrics.py) | 共用模块：聚类和文字转录质量指标 |

相关说明：[使用指南](../docs/guides/photo_partition_review_guide.md)。

## 视觉证据与分组实验

| 脚本 / 模块 | 用途 |
| --- | --- |
| [collect_visual_observations.py](vision/collect_visual_observations.py) | 为冻结清单逐图采集画面观察 |
| [evaluate_visual_budget.py](vision/evaluate_visual_budget.py) | 请求/token 预算与视觉证据评测 |
| [evaluate_ocr_evidence.py](vision/evaluate_ocr_evidence.py) | 指定文本图片的本地 OCR 评测 |
| [paddle_ocr_worker.py](vision/paddle_ocr_worker.py) | 独立 Paddle 环境的 OCR 工作进程 |
| [evaluate_temporal_research.py](vision/evaluate_temporal_research.py) | 基于冻结向量和 EXIF 提示研究时序分组 |
| [evaluate_face_graph_checkpoint.py](vision/evaluate_face_graph_checkpoint.py) | 在已有检测检查点上比较图聚类划分 |
| [benchmark_experimental_grouping.py](vision/benchmark_experimental_grouping.py) | 时序与人脸分组的隔离 CPU 规模测试 |

相关说明：[使用指南](../docs/guides/experimental_components_guide.md)。

## 性能与流程成本

| 脚本 / 模块 | 用途 |
| --- | --- |
| [benchmark_feature_pipeline.py](performance/benchmark_feature_pipeline.py) | 顺序/批量特征、原图整理与资源对照 |
| [evaluate_dali_preprocessing.py](performance/evaluate_dali_preprocessing.py) | 在独立输出中比较 DALI 和已完成 Pillow 基准 |
| [benchmark_task_responsiveness.py](performance/benchmark_task_responsiveness.py) | 受控慢任务下的 HTTP 响应 |
| [benchmark_review_snapshots.py](performance/benchmark_review_snapshots.py) | 大复核队列单条写入与快照导出成本 |
| [benchmark_workflow_costs.py](performance/benchmark_workflow_costs.py) | 持久分析、复核、导出、整理的真实 HTTP 成本 |

相关说明：[使用指南](../docs/guides/feature_preprocessing_guide.md)。

## 清单与整链路验收

| 脚本 / 模块 | 用途 |
| --- | --- |
| [build_core_upgrade_manifest.py](validation/build_core_upgrade_manifest.py) | 清点原照片并建立来源清单 |
| [characterize_core_correctness.py](validation/characterize_core_correctness.py) | 记录 C1–C4 修复前后行为 |
| [evaluate_core_upgrade.py](validation/evaluate_core_upgrade.py) | 中性预览上的探索性场景检索 |
| [check_multimodal_upgrade.py](validation/check_multimodal_upgrade.py) | 本地模型和多模态组件冒烟检查 |
| [check_real_workflow.py](validation/check_real_workflow.py) | 真实分析 → 复核 → 导出 → 整理验收 |

相关说明：[使用指南](../docs/guides/workspace_guide.md)。

## 仓库与环境维护

| 脚本 / 模块 | 用途 |
| --- | --- |
| [audit_repository_scope.py](maintenance/audit_repository_scope.py) | 核对 Git 索引，排除本地数据与运行产物 |
| [audit_remote_model.py](maintenance/audit_remote_model.py) | 读取远端模型元数据，不执行模型推理 |

相关说明：[使用指南](../docs/guides/repository_layout.md)。
