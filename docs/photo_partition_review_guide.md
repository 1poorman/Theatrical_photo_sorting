# 剧照划分与图库来源复核

本工具准备人工核查材料，支持中断后继续和跨会话交接。当前产物位于 [U0复核目录](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/)，异常恢复先读 [工作状态](../WORK_STATE.md)。

直接打开 [复核页面](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/artifacts/index.html) 或 [相机时间序列](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/artifacts/timeline.html)。页面按类型、文字和待核查筛选，保存浏览器草稿，支持导出与恢复JSON。复核人、确认依据、分歧和实际场次/连拍ID分别填写。导出为复核草稿，不会自动修改正式标注、图库或测试集；跨机器交接携带整个产物目录及导出的核查JSON。

## 当前版本与发现

复核包绑定旧 `verified_labels_roles_20261022/` 版本：263张已导入图、260条幕次。修正另存新 `verified_labels_scene_segments_v2_20261022/`：263图、262条幕次。页面中的旧“未导入幕次”状态保留用于发现过程；修正见 [两条标签差异报告](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/import_gap_report.md)。历史U4报告及短块协议继续绑定原260 query，不将262分母写回旧Recall。

| 核查项目 | 实际结果 | 解释 |
| --- | --- | --- |
| 全部图片 | 582文件、319唯一解码图像、263副本组 | 原/整理副本共享像素身份和已知使用记录 |
| 目录分类线索 | 292舞台照、27资料图 | 分类尚需人工核查 |
| 未导入舞台照 | 29张，均样式雷；20张曾用于U2 | 当前复核后均为开发数据 |
| 原始拍摄时间 | 286舞台图有DateTimeOriginal、6张缺失；日期不一致0张 | 编辑时间不代替原始拍摄时间 |
| 相机序列 | 14条；14处整理序号相对EXIF逆序 | 来源目录/机身线索分开，缺序列号时型号可能对应多机身；跨相机时钟未校准 |
| 文件名组 | 49组、17组跨已导入幕次 | 基础名和编号不能直接证明同一连拍 |
| 近重复/时间相邻 | 4个近重复候选对；同流≤2秒1对 | 启发式候选，未人工核验，可能漏检/误报 |
| 边界线索 | 197对 | ≥60秒停顿或已导入幕次变化，尚非真实切幕/停拍GT |
| 图库来源 | 109裁剪名称对应47源图；108裁剪/46图与旧U4 query对应 | 来源/框未核验，现图库不能据此宣称gallery/query隔离 |
| 复核项目 | 375项 | 场次4、近重复4、时间1、边界197、同名49、逆序14、缺幕次32、资料图27、图库47 |

每剧只有一个日期线索，独立演出确认数仍为0。旧 `final_test_provisional` 的79图中至少69图已出现在已提供实验清单；清单仅为已知使用的下界。本轮所有582文件的新使用标记均为 `development_review`，final test未锁定。没有新增框级身份、unknown、连拍偏好或连续序列边界GT。

## 两处幕次漏读修正

沿用已保存的用户确认来源 `user_confirmed_organized_filenames_20261022`。整理完成文件名的两条幕次使用独立连字符段，旧导入器只读取事项后的空格场次：天下归心33号 `彩排-第5幕见母-…`，样式雷39号 `彩排-第8场-…`。

[`tools/import_verified_photo_labels.py`](../tools/import_verified_photo_labels.py) 现读取已确认整理文件名中的明确独立场次段，冲突编号拒绝导入，未用原图文件名猜测幕次。只有两图的 `scene_label`/`scene_id` 改变，旧260标签、演员/角色记录与身份ID保持；身份注册表字节相同，剩余角色冲突不变。原图、旧标注和历史评测保留。

新 [标注JSON](../outputs/acceptance/core_upgrade/verified_labels_scene_segments_v2_20261022/verified_photo_labels.json) SHA256：`a157fb45d7e5ec4a175589ccc599716e768e5c5836819c47d428bcfaa979a42e`。完整差异与12项检查见 [标签审计](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/import_gap_audit.json)。剩余无幕次图为假面舞会指挥照，不从相邻标签补写。

## 准备、恢复与核查

新目录准备示例（本机已有目录已完成，使用空输出目录）：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/prepare_photo_partition_review.py \
  --root data/ncpa_test \
  --inventory outputs/acceptance/core_upgrade/first_pass/manifest.jsonl \
  --legacy-splits outputs/acceptance/core_upgrade/first_pass/splits.json \
  --labels outputs/acceptance/core_upgrade/verified_labels_scene_segments_v2_20261022/verified_photo_labels.json \
  --registry outputs/acceptance/core_upgrade/verified_labels_scene_segments_v2_20261022/identity_registry.json \
  --gallery data/face_database \
  --exposure-manifest outputs/acceptance/core_upgrade/u4_verified_siglip2_final_20261022/input_manifest.json \
  --exposure-manifest outputs/acceptance/core_upgrade/u2_publication_200x5_20261021/inputs.json \
  --output outputs/acceptance/core_upgrade/u0_partition_review_next/artifacts
```

每张唯一图核对原始内容、EXIF方向RGB像素和dHash，保存带校验和的 `image_checkpoints/` 与中性预览。原始EXIF时间、亚秒和时区线索分开，机身序列号仅存摘要ID；预览清除EXIF。原图/图库/输入在结束前再次核对完整摘要，来源增删也拒绝。

未完成目录用同一命令、输入与源码加 `--resume`，核对协议、检查点和预览摘要，避免重新解码已完成图。已完成目录用 `--verify-existing`，不会重建图片或重跑模型。输入/源码/图库变化需新目录；保留失败状态，不绕过版本检查。

本轮协议摘要 `18ea348ae152fdf7e1d378e3986b7bcb875132c749db0b9d35b7b8c2e58c45f1` 绑定旧标注。[commands.json](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/commands.json) 保存完整prepare/resume/verify参数；使用262条新版标签另建协议。

本轮已完成包只读核查：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/prepare_photo_partition_review.py \
  --root data/ncpa_test \
  --inventory outputs/acceptance/core_upgrade/first_pass/manifest.jsonl \
  --legacy-splits outputs/acceptance/core_upgrade/first_pass/splits.json \
  --labels outputs/acceptance/core_upgrade/verified_labels_roles_20261022/verified_photo_labels.json \
  --registry outputs/acceptance/core_upgrade/verified_labels_roles_20261022/identity_registry.json \
  --gallery data/face_database \
  --exposure-manifest outputs/acceptance/core_upgrade/u4_verified_siglip2_final_20261022/input_manifest.json \
  --exposure-manifest outputs/acceptance/core_upgrade/u2_publication_200x5_20261021/inputs.json \
  --output outputs/acceptance/core_upgrade/u0_partition_review_20261022/artifacts \
  --verify-existing
```

## 验证与接续

9项复核契约测试、20项完整性检查、759份输出摘要只读核查通过。本地页面浏览器12项通过：筛选、刷新恢复、JSON导出/导入、错误协议拒绝、三种屏宽、真实预览和时间线。临时浏览器草稿未作人工决定；沙箱Chromium套接字受限后获准宿主重试。日志/截图见 [浏览器摘要](../outputs/acceptance/core_upgrade/u0_partition_review_20261022/browser_summary.json)。

导入修正先复现4失败，修正后6项回归及12项差异审计通过。冻结短块协议及11份绑定源码未改，U4完成项无需重跑。SigLIP2默认、分块仅离线、U2暂停。

后续核验场次、机身时钟、近重复/连拍和图库裁剪来源，补框级身份/未知query后校准身份阈值。马可现有59张舞台原图均有已导入对应图，第3/4幕及尾声仍缺正例；13张资料图待核查。新增独立照片和候选正文带来源/版本，新262图比较另建中性清单与协议。
