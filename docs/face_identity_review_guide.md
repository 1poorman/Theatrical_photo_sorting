# 逐脸身份标注与来源复核

打开本机已有的[逐脸复核页面](../outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023/index.html)。它包含292张真实剧照、1095张检测人脸，提供原图框位置、裁剪、35个剧目内演员身份以及可保存、导出和恢复的浏览器草稿。每页显示12张原图；五种屏宽、分页、草稿和导入导出共12项浏览器检查通过。

先填写复核人，逐脸选择已知身份、未知身份或不可评价，记录确认依据和必要说明。看不清身份时使用不可评价；未知需要证据证明不属于所选身份集合。文件名中的人物列表可能不完整，不能按列表位置自动对应人脸，也不能把未列名的人脸标为未知。镜像、重复检测等例外单独勾选并记录依据。多角色冲突仍保留演员身份，不从当前裁剪猜角色。

984张对齐裁剪直接复用已完成任务中的PNG字节；旧拼图每簇最多保留24张裁剪，剩余111张采用已有检测框从源图截取，记录为`recorded_bbox_unaligned`。未新增检测或特征推理。框坐标对应既有OpenCV解码方向和最长边1920的缩放帧，页面按比例投影至原图预览。面积低于历史门槛的脸、漏检和坏图不在本包内；无检测结果不表示原图无人脸。

全部身份标签初始为`pending`；当前人工框级核对仍为0。浏览器测试中的临时决定不算人工标注。现有图像都已用于开发实验，不能通过填标签把它们变成未使用的final test。图库来源、近重复/连拍隔离和未知身份不入gallery仍需核验；本工具不会修改正式图库、已确认图级标签或锁定数据划分。

在项目根目录检查现有包，无需模型或GPU：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/prepare_face_identity_review.py \
  --output outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023 \
  --verify-existing
```

导出浏览器草稿后验证人脸ID、原图哈希、检测框和协议绑定；已完成决定要求有效身份ID、复核人、依据和带时区时间戳。报告只统计合法人工决定，不运行质量评价，也不应用标注：

```bash
python tools/prepare_face_identity_review.py \
  --output outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023 \
  --decisions /path/to/face_identity_decisions.json \
  --decision-report /path/to/new/decision_validation.json
```

准备新包使用空目录，保留现有报告：

```bash
python tools/prepare_face_identity_review.py \
  --task-db outputs/acceptance/core_upgrade/u3_cluster_http_292_20261023/tasks/tasks.sqlite3 \
  --task-id ea49f74d798e46b9a6c992b0b5a72c0e \
  --input-manifest outputs/acceptance/core_upgrade/u3_cluster_http_292_20261023/input_manifest.json \
  --photo-review outputs/acceptance/core_upgrade/u0_partition_review_20261022/artifacts \
  --registry outputs/acceptance/core_upgrade/verified_labels_scene_segments_v2_20261022/identity_registry.json \
  --output /path/to/new/face_review
```

输入摘要、检测记录、裁剪、模型契约和准备工具版本绑定到协议；跨机器交接应携带整个复核包、JSON草稿及所引用的源资料。原始标注/图片/裁剪只保存在本地`outputs/`，不纳入Git。第一次遗漏旧拼图上限的准备失败和浏览器重绘错误复现均保留；最终包与修复验证见[本轮接续目录](../outputs/acceptance/core_upgrade/resume_followup_20261023/)。

已有独立评测入口可把真实逐脸决定与冻结的legacy/graph分组比较，无需重新检测或提取特征：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/evaluate_face_identity_review.py \
  --review outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023 \
  --task-db outputs/acceptance/core_upgrade/u3_cluster_http_292_20261023/tasks/tasks.sqlite3 \
  --study outputs/acceptance/core_upgrade/u7_graph_real_1095_20261023 \
  --decisions /path/to/face_identity_decisions.json \
  --output /path/to/new/face_quality
```

输出目录必须为空。省略`--decisions`会读取原始待定模板；[现有报告](../outputs/acceptance/core_upgrade/u1_face_quality_pending_20261024/summary.json)中1095张脸均待定，B-cubed、已知身份召回和unknown FPIR均为`null`，不会由聚类数量推算准确率。输入检测顺序、框、特征向量、标签数组和模型契约均需匹配。

B-cubed只计算明确已知、可评价且未标例外的人脸子集。不同照片中的unknown不被当成不同人物。已复核且非例外的同图人脸提供cannot-link假设，同图同身份冲突另列待核验；镜像/重复检测例外不参与该约束。该工具不执行gallery/query身份匹配，也不完成阈值校准或独立final test；分组指标再高也不会自动采用图聚类。画面事实和OCR使用另一份[双人复核指南](visual_fact_review_guide.md)。
