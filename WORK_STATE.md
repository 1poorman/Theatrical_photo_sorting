# 工作状态与中断接续

> **2026-10-26 技术报告更新交付：** [最新技术报告](docs/technical_report_20261026.md)为29页、15幅正文图；[PDF](docs/reports/technical_report_20261026/technical_report.pdf)、离线HTML、PNG/JPEG、可编辑SVG、31份证据快照与重建/验证脚本集中在[Git可见报告包](docs/reports/technical_report_20261026/README.md)。新版补充Pillow修复、双线程选优及DALI评价，保留五轮慢样本、P95未改善和正式U2 `not_passed`结论；测量版本仍绑定`4194277`与当时工作区摘要。历史26页报告的资源已迁入`docs/reports/technical_report_20261024/`，实验数字不变。此次提交范围为报告与文档导航，已有U2源码及其测试改动继续留在工作区；发布结果以Git提交和远端分支为准。

> 最后更新：2026-10-24（UTC，按用户指示提交已完成的命名兼容、延后快照和质量复核续作）。这是异常中断、上下文切换和协作交接的首读文件。
> 当时文档交付：面向技术评审与研发的[历史技术报告](docs/technical_report_20261024.md)，26页PDF、离线HTML及13幅正文图；可阅读资源现已迁入`docs/reports/technical_report_20261024/`。当时仅汇编已有实验，未重新推理；模型/分组默认和U2暂停保持。原始生成脚本与排版检查仍保留在本地历史输出。
> 当前交接点：本次提交收录`581f8fa`之后的命名兼容、延后快照、逐脸/事实复核与评测代码，提交哈希以`git log -1`及本地`completion_audit_20261024/commit_receipt.json`为准。100图按需观察已完成100请求/10检查，当前v2包包含1352事实和5图OCR，17项浏览器检查通过；1095脸评测入口完成。双人评测新补完整来源对齐和同图字节去重，先复现7失败子例、修复后15测试通过，旧页面与模型结果保留。13存储/27API、18整理/28核心、6身份评测已有通过日志。最终只读核查5647项通过，3365唯一文件摘要匹配（含582源文件/109图库文件），32语法/文档检查及112本地链接通过，见`completion_audit_20261024/`；无活动任务。用户明确没有新增GT/独立照片/两位复核人，全部人工标注仍0，正式质量缺项保留。
> 当前未完成：375项原照片复核和1095脸人工标注、gallery/query隔离、独立final test、U4剩8正文/新增马可正例、U5双人事实与OCR转录、U6可信边界、U7偏好与模型对照；远端权重及受控整链路性能未验收。U2继续暂停，不以功能/合成规模通过代替质量验收。已有材料和评测入口可直接使用，不重复同一输入的推理或准备。
> 当前模型结果仍沿用262开发集：均值25/28池R@3为SigLIP2 69.85%/53.44%、Gemma76.34%/71.76%，默认SigLIP2/legacy。历史报告保留绑定版本，不原地重跑；本次只作本地提交，未推送，领先数量以`git status --short --branch`为准。

## 1. 接手时先确认

| 项目 | 2026-10-23 现场状态 |
| --- | --- |
| 工作目录 | `/home/huachenghao/codes/Theatrical_photo_sorting-251212` |
| 分支 / 实现提交 | `master` / 当前`HEAD`；前轮实现为`581f8fa`，本次哈希以Git现场和提交回执为准 |
| 上游 | `github/master`；本次未联网核验远端、未推送 |
| 工作区 | 本次提交收录命名兼容、恢复版本补充、延后快照、逐脸/事实复核、评测与文档；第4节的未提交描述保留其提交前范围 |
| 当前优先级 | 已有本轮功能/规模/真实流程证据；接续U0/U1独立数据与框级标注、U4正文/新候选正例、U5事实、U6边界和U7偏好，不重复已完成实验 |
| 暂停项 | U2 提速、解码优化、DALI；保留现有实现和 `not_passed` 结论 |
| 模型默认 | SigLIP2 base；EmbeddingGemma 2 仍为可选实验后端 |
| 活动进程 | 上轮浏览器服务1138882的日志确认退出；当前接续只读CPU审计完成后无活动服务/GPU任务，见`completion_audit_20261024/summary.json`。旧沙箱失败和宿主重试分别保留 |
| 当前执行者 | 主会话；未分派其他智能体；本会话无活动goal对象，U2继续暂停 |

先执行 `git status --short --branch`、`git diff --stat`，再阅读本文件与 [核心升级进度](docs/core_upgrade_progress.md)。验收定义以 [核心升级蓝图](docs/core_upgrade_blueprint.md) 为准。本文件是现场快照；后续用户指令、实际代码及带版本的原始报告优先于过时记录。

## 2. 已确定的约束与标注范围

- 2026-10-22 已有进度记录明确：用户暂停 U2，先做身份/角色契约和标注准备。普通“继续”沿用该优先级。
- 原图、正式人脸库和已有人工决定保留；实验使用独立输出与图库副本。当前导入产物在 `outputs/acceptance/core_upgrade/`，不要误当成已发布到正式图库。
- 既存导入报告记录确认来源为 `user_confirmed_organized_filenames_20261022`：整理完成文件名中的幕次及列出的人物/角色已确认。本次从持久报告恢复这一记录，没有重新进行人工核对。
- 人物列表不保证包含所有可见人脸；未列名不能作为 unknown 身份 GT。框级身份、未知身份 query、连拍偏好和连续序列边界核对数仍均为 0。
- 已保存的别名确认仅适用于舞剧《马可·波罗》中苏鹏的“马可波罗”→“马可·波罗”，来源为 `user_confirmed_alias_20261022`。不要扩展成全局字符串替换。
- 王浩伟的“烩五哥 / 播威利”仍保留为多角色冲突；没有消歧前输出演员，不推断当前照片的角色。同名演员跨剧目也不自动合并身份。
- final test 尚未锁定。现有 260/262 图评测均为开发集比较，不作为最终质量通过；U2 暂停不改变 U8 的验收门槛。
- 后续复核另存新版263图/262幕次：只恢复2条已确认整理文件名的连字符场次，旧260标签/人名角色/身份注册表字节保留。历史U4全部仍绑定旧260 query；不能修改旧报告分母。现有图库108裁剪名称对应46张旧query，框/身份与来源仍未核验；不将现图库当作已隔离身份验收。

## 3. 关键节点

| 日期 / 节点 | 已落地内容 | 证据与边界 |
| --- | --- | --- |
| 2026-10-19 · `fdc1fda` | C1–C4 首轮正确性修复，本地多模态评测入口 | corrected baseline；合成复现不代表真实模型质量 |
| 2026-10-21 · `3ebadf3` | 共享特征缓存、批量路径、复核事务、持久场景任务 | U2 后续 200 图 × 5 次特征吞吐仅 1.151 倍，保持 `not_passed` |
| 2026-10-21 · `f7cb66a` | 整理/知识库阶段恢复、执行版本绑定 | 源码、模型、输入版本变化会阻止混用检查点 |
| 2026-10-21 · `961ffb7` | 人脸聚类与标注照片建库持久任务、多输出占用 | 真实 HTTP、35 图一致性及 200 图恢复记录见阶段进度 |
| 2026-10-21 · `49b9ce0` | 按原图持久保存无损人脸裁剪 | 200 图预览三轮中位 211.60→94.17 秒，JPEG 字节一致；只是预览阶段对照 |
| 2026-10-22 · U3 本地报告 | 新版 200 图 HTTP、计划/裁剪取消、部分发布失败与四次服务恢复 | HTTP 10 项、恢复 17 项检查通过；704 脸、326 簇；未完成整个 U3 |
| 2026-10-22 · U1实现 | 显式身份 ID、当前剧目角色、逐脸 top1/margin、角色表校验及 API/UI；补同名演员跨照片专辑与姓名/ID 筛选 | 身份 15/15、核心 28/28、整理 17/17、工作台 24/24、浏览器通过；正式身份质量未验收 |
| 2026-10-22 · U0 本地报告 | 导入 4 剧目 263 张标注照片，260 张有幕次，249 张列出人物 | 35 个剧目内身份、471 条人物/角色记录、15 个图库目录绑定；保留 1 处角色冲突 |
| 2026-10-22 · U4 本地报告 | 两模型同源码、260 张中性图片、4 剧目、25 个候选幕次的对照已完成 | SigLIP2 / Gemma R@1 18.85% / 17.31%，R@3 55.00% / 58.08%；保留 SigLIP2 默认 |
| 2026-10-22 · U4来源v1 | 三剧目12个候选补有出处正文，独立新增3个马可幕次；固定25/扩展28开发集比较与来源指南完成 | 五份新报告各7检查、比较45检查、来源10测试通过；Gemma R@3 72.31% / 68.08%，final test未锁定 |
| 2026-10-22 · U4短原文块 | 82块≤64 tokens、102去重文档；图/文各编码一次/后端，12变体从固定向量/分数生成 | 两后端各12检查、分块测试10项、数组审计102项；均值25/28池R@3 SigLIP2 70.00%/53.46%，Gemma 76.54%/71.92%；未达正式U4 |
| 2026-10-22 · 演出版交叉核对 | 16来源/21段引用及6份搜索发现：假面2012瑞典版谱系、马可2014升级/2015压缩差异、样式雷2016复排/6月档期 | 10项审计及只读复现通过；未更改冻结模型正文/照片标注，8项缺失正文与照片版本边界保留 |
| 2026-10-22 · U0复核与导入v2 | 319图/263副本、EXIF/相机/近重复/同名/图库375复核项，已确认连字符幕次漏读2条另存263图/262幕次 | 9复核测试、20完整性/759产物核查、浏览器12、6导入回归及12差异审计通过；均未作框级/独立场次/连拍人工标注 |
| 2026-10-22 · 本次交接 | 建立本文件并添加文档入口，补齐 U0/U1/U4 当前摘要 | 报告存在性和摘要已核对；本次没有提交或推送 |

上表已有提交均可在本地 Git 历史找到；推送信息沿用阶段进度记录。U3 两份报告中的 `app/main.py`、`app/organize_tasks.py` 摘要已与当前 U1 改动不同，不能把历史 HTTP 通过当作当前工作区复验通过。

2026-10-23 已另建当前源码的 `u3_cluster_http_292_20261023/`、`u3_cluster_recovery_292_20261023/` 和工作台回归报告，分别通过10/17/25项检查；旧200图报告仍保留其版本范围。

## 4. 本次实现与提交范围

下列第一张表对应前轮已获授权并完成的本地提交 `9c97bb3`；本地标注、原图、图库与实验输出不纳入Git。当前新增未提交范围另列于后。

| 工作块 | 文件 | 当前行为 / 待办 |
| --- | --- | --- |
| 身份契约 | 新增 `core_modules/organize/identity_registry.py` | 验证注册表、聚合同身份锚定分数、逐脸拒识原因、当前剧目唯一已确认角色及来源；旧目录名身份标为未核验 |
| 识别与专辑 | 修改 `smart_organizer.py`、`actor_view.py`（均在 `core_modules/organize/`） | 输出 `identities` / `face_decisions`；专辑筛选保留其他身份参与竞争；角色表改写触发拒绝 |
| 任务 / API / 工具 | 修改 `app/organize_tasks.py`、`app/main.py`、`core_modules/organize/organize_task.py`、`photo_agent_tools.py` | 传递 `identity_registry_file`、`face_threshold`、`face_margin`；绑定角色表哈希与执行版本；旧同步入口兼容 |
| 工作台 | 修改 `app/static/workspace.js`、`app/templates/workspace.html` | 演员与角色表路径输入、项目状态保存、整理及专辑请求参数 |
| 标注与检索工具 | 新增 `tools/import_verified_photo_labels.py`、`tools/evaluate_verified_scene_retrieval.py`、`tools/sourced_scene_candidates.py` | 导入已确认图级标签；中性PNG、剧目分池对照；来源快照/跨度/候选ID绑定与准备清单重建校验 |
| 照片复核与导入修正 | 新增 `tools/prepare_photo_partition_review.py`、`tests/test_photo_partition_review.py`、`tests/test_verified_photo_label_import.py`；扩展独立import工具和.gitignore例外 | 原图/图库/输入只读、逐图检查点与同版resume、原始时间/相机流、375复核页面/草稿JSON；修复2条场次漏读，9/6测试与完整产物核查通过 |
| 短块与固定向量 | 新增 `tools/scene_text_blocks.py`、`tools/prepare_scene_text_block_study.py`、`tools/evaluate_scene_text_block_study.py`、`tests/test_scene_text_blocks.py` | 精确引文块、64-token预算、均值/最大值预声明；一次编码/去重文档、同分数25/28排名；10测试、真实模型/数组审计通过，仅离线研究 |
| 回归 | 新增 `tests/test_identity_registry.py`、`tests/test_sourced_scene_candidates.py`；修改工作台测试与`.gitignore`例外 | U1共84项/浏览器通过；来源契约10项通过，包含引用注入/错误场次/输入改写/扩池 |
| U3 恢复测量 | 修改 `tools/benchmark_face_task_completion.py` | `--cancel-after-crop`、裁剪检查点与发布核对；已有本地报告 |
| 文档 | `docs/core_upgrade_progress.md`、`docs/multimodal_progress.md`、`docs/workspace_guide.md`、`docs/CHANGELOG.md` | 前轮恢复记录、U2 暂停、U1 验证、U4 配对结果与边界 |
| 指南与交接 | 新增 `docs/identity_registry_guide.md`、`docs/sourced_scene_candidates_guide.md`、`docs/photo_partition_review_guide.md`、`WORK_STATE.md`；更新文档入口 | 身份/来源/照片复核契约、冻结输入与新262标签版本、结果边界、恢复与复现命令 |

`git diff` 不显示未跟踪文件内容，代码审查和提交时要同时检查 `git status --short` 中的 `??`。`outputs/` 被忽略，提交源码不会自动携带本地标注、图片或评测报告。

2026-10-23下表范围已纳入`581f8fa`（44文件），当前续作范围另列：

| 工作块 | 文件 / 行为 | 验证与边界 |
| --- | --- | --- |
| U3 复核详情 | `app/main.py`、`workspace.js`、`review_queue.py`、`review_store.py`：简要元数据与当前页批量详情 | API25项、万条记录浏览器36/36/1详情请求，无完整队列请求；JSONL兼容快照仍全量 |
| U5 | 新`visual_evidence.py`、`ocr_evidence.py`、`paddle_ocr_worker.py`；扩展`llm_client.py`、`scene_reasoner.py`与任务参数/身份 | 预算/缓存/权重/重试回归，100图实际30请求及5图OCR；无事实/转录GT |
| U6/U7 | 新`scene_change_points.py`、`face_graph_cluster.py`、`temporal_burst.py`；接入整理/聚类显式实验参数 | 窗口/图/连拍测试、万条特征与真实特征探索；默认legacy、连拍仅离线 |
| 评测工具 | `scene_retrieval_study.py`及prepare/evaluate入口；`evaluate_visual_budget.py`、`evaluate_ocr_evidence.py`、`benchmark_experimental_grouping.py`、`evaluate_temporal_research.py`、`evaluate_face_graph_checkpoint.py`、`check_real_workflow.py` | 新空目录/版本绑定/隔离数据；真实与合成报告各自记录边界 |
| 测试与文档 | 新增各模块测试，扩展工作台/API/聚类测试及.gitignore例外；实验指南、进度、CHANGELOG、入口与本文件 | 失败复现保留；当时最终任务91项通过；已随`581f8fa`提交，未推送 |

当前续作未提交范围：`filename_parser.py`、`burst_dedup.py`、`scene_reasoner.py`、`workspace_api.py`、`app/organize_tasks.py`、`check_real_workflow.py`与解析/整理/API回归；新增`tools/prepare_face_identity_review.py`、10项契约、指南、文档入口和状态。已整理序号不会被当作连拍依据；当前项目未经批准的旧幕次仍输出占位。

2026-10-24新增未提交范围：`review_store.py`、`review_queue.py`、`app/main.py`、`workspace.js`及相应回归支持延后快照；新增`benchmark_review_snapshots.py`、`audit_remote_model.py`、`collect_visual_observations.py`、`prepare_visual_fact_review.py`、`evaluate_face_identity_review.py`、`review_quality_metrics.py`、事实/身份评测与浏览器测试及指南。双人评测拒绝跨协议、来源、事实、OCR预测混用和同任务重复图字节；报告记录评测源码摘要。未变更默认模型、自动请求预算或U2暂停状态。

## 5. 产物定位与当前结果

以下路径均相对 `outputs/acceptance/core_upgrade/`。原始报告中的数字和版本字段优先于文档转述。

| 路径 | 状态 / 用法 |
| --- | --- |
| `completion_audit_20261024/` | 本次评测边界修复：7失败子例与15通过测试、新待定事实报告、只读产物/原图/图库核查脚本与汇总；现有HTML可由当前渲染器逐字节重建，未重跑模型 |
| `completion_followup_20261023/` | 上轮13存储/27API/18整理/28核心/6身份评测日志、35图工作台浏览器与事实17检查、初次失败与最终截图；服务1138882已结束 |
| `u5_fact_observations_host_20261024/` | 100图按需观察/100实际请求、10检查、同进程重放0请求；远端权重未核验，人工核对0；沙箱失败另存 |
| `u5_dual_fact_review_v2_20261024/` | 当前100图1352事实与5图OCR的双人复核页面、A/B待定模板；105任务全部pending，17浏览器检查通过；初版保留 |
| `u1_face_quality_pending_20261024/` | 1095脸/453旧簇/616图簇的框/向量/分组绑定通过；人工标签0，B-cubed/身份召回/unknown FPIR为空，正式质量不足 |
| `u3_review_snapshot_scale_20261023/` | 万条合成队列、5次保存/9检查，中位0.214706→0.001850秒；仅本地函数与摘要耗时。报告早于之后无快照场景证据读取分支，不作当前整链路速度结论 |
| `remote_model_metadata_host_20261023/` | 元数据审计无推理请求；服务别名`Qwen3.6-27B`存在，配置`Qwen3.8-27B`，无权重摘要，不放宽跨进程恢复 |
| `first_pass/manifest.jsonl`、`first_pass/splits.json` | U0 原始清单及 provisional 划分；导入工具使用前者，后者尚非锁定测试集 |
| `verified_labels_20261022/` | 初次导入，有两处角色冲突；其中 `confirmed_role_aliases.json` 是已保存的别名确认输入 |
| `verified_labels_roles_20261022/` | **冻结历史263/260版本**：所有已完成U4和U0复核包绑定此版；应用已确认别名后剩一处冲突，不原地修改 |
| `verified_labels_scene_segments_v2_20261022/` | **当前标注版本263/262**：恢复2处已确认连字符场次，原260标签/actor_roles不变、注册表字节相同；12差异审计通过，version_manifest绑定摘要；新262模型报告见下 |
| `u0_partition_review_20261022/` | **当前U0准备包**：`artifacts/`存319图/263副本/286拍摄时间、14相机流与375复核项，9测试/20完整性/759只读产物/浏览器12检查；`import_gap_audit.json`及报告、新263/262标注差异，6测试/12审计。绑定旧标注，页面不自动应用决定；命令/状态/检查点/日志/3截图均保存 |
| `u4_verified_siglip2_20261022/` | 只有输入清单和中性图，未见 `summary.json`；不能记为模型评测完成，未恢复其停止原因 |
| `u4_verified_siglip2_gpu_20261022/` | 只有输入清单和中性图；后续完成报告实际引用这里的 `neutral_images/`，必须保留 |
| `u4_verified_siglip2_final_20261022/` | 初次完成的 SigLIP2 报告及本轮共同输入清单；名称中的 `final` 不表示 final test 已锁定；身份显示逻辑随后已变化 |
| `u1_regression_initial_20261022/` | 初次测试日志、沙箱首个 HTTP 请求超时及宿主 23 项通过记录；同名问题修复前的失败复现 |
| `u1_regression_final_20261022/` | **2026-10-22 U1 验证**：`summary.json`、四组测试日志、`browser.log`、4 张 `browser_screenshots/`、已关闭的隔离服务日志；当前复验见20261023目录 |
| `u4_verified_siglip2_followup_20261022/` | 来源校验前的SigLIP2报告；全部260图排序复现初次及新入口基线 |
| `u4_verified_gemma2_followup_20261022/` | 来源校验前的Gemma原25候选基线；与旧SigLIP2清单字节相同，与新运行模型/预处理及7份非入口源码一致 |
| `u4_verified_comparison_20261022/` | 原25候选比较（5本地正文+20标签）：`report.md`、`summary.json`、`run_state.json`、`comparison_analysis.py`；历史报告保留 |
| `u4_ncpa_sources_20261022/` | **冻结来源v1**：6份参考HTML/响应头、提取文本、`source_audit.json`、`source_report.md`、两份候选包及两版准备清单；不要原地重跑构建脚本 |
| `u4_sourced_comparison_20261022/` | **当前U4来源比较**：`report.md`、`summary.json`、`run_state.json`、只读分析脚本、`text_window_audit.py/json`、测试/分析日志；45项审计通过 |
| `u4_text_blocks_20261022/` | **短原文块研究完成**：冻结`prepared/{protocol,blocks}.json`；两后端向量/分数/六变体保存；`summary.json`/`report.md`、`metrics.csv`/`scene_metrics.csv`、只读分析脚本/复现；102数组审计、10分块测试、10演出版审计通过。执行器/JSON记录命令/PID/退出码；新网页仅交叉参考，不加入冻结正文 |
| `u4_inputs_scene_segments_v2_20261022/` | **262中性输入已完成**：23检查、原260顺序/字段/PNG字节及25/28文本保持，协议`db061c85…`；仅补两query，无独立test声明 |
| `u4_ncpa_source_followup_v2_20261022/` | **续查已完成**：18原页面、21搜索记录、20精确引文、15审计及只读重现；2016样式雷过场/暗场调整、2015年2月马可再改、两个201608URL实际正文2011；8正文缺项保留，候选未改 |
| `u4_retrieval_study_262_20261022/` | **协议完成、首个GPU尝试失败**：动态协议SHA`e8519c7e…`及82块绑定，10契约测试通过；沙箱CUDA失败日志/退出1保留，不重写此目录 |
| `u4_retrieval_study_262_gpu_retry_20261022/` | **262图完成**：获准宿主GPU1两后端退出0，各13检查、276独立数组/排名/Macro-F1/子集/事件审计通过；`summary.json`/`report.md`和只读复现脚本。历史260/新增2/全部262分别报告，基线重建差异明确、无速度或正式质量声明 |
| `resume_validation_20261023/` | **本轮回归与接续**：初测/6处复现/30项修复、最终任务91项、API25项和浏览器通过；源码与产物1255项预检、配置示例验证、最终独立`completed_acceptance_audit.json`与脚本；`stage_status.json`、`active_work.json`及全部日志/截图 |
| `u5_visual_budget_sandbox_20261023/`、`u5_visual_budget_host_20261023/` | 沙箱HTTP权限错误保留；宿主100图/4剧目完成，15观察+15闭集=30实际请求，14检查通过，重放0请求；权重未核验、同进程缓存，事实人工核对0 |
| `u5_ocr_verified_20261023/` | 5张明确含字图片真实本地PaddleOCR，schema/原图/模型摘要与重建缓存检查通过；无逐字参考 |
| `u6_u7_cpu_scale_20261023/` | 10000条固定人工生成特征、4线程/独立进程/各1次，10检查通过；窗口1.7075秒/330.7MiB，图1.2763秒/201.7MiB，层次21.7812秒/983.4MiB，仅标签计算 |
| `u6_u7_temporal_real_20261023/` | 262源哈希全部关联SigLIP2向量、14未核验相机流，7完整性检查通过；保守全单图，探索260单图+1两图组，无边界/偏好GT |
| `u3_cluster_http_292_20261023/` | 当前源码/真实GPU1，292唯一舞台照/9目录，10检查通过；健康/分页/不同任务创建P95为4.577/10.998/119.771ms，取消确认6.288ms |
| `u3_cluster_recovery_292_20261023/` | **四次恢复完成**：计划取消/首裁剪取消/首发布失败/重启完成，17检查通过；1095脸/453簇/0失败，无重复检测/裁剪/预览/新建，部分文件复用1份 |
| `u7_graph_real_1095_20261023/` | 只读复用上述检测向量，453旧簇/616图簇，4200假定同图cannot-link违反数29/0；镜像例外未核验，不当身份精度 |
| `u8_real_workflow_200_20261023/` | **功能完成但输入范围不合格**：200已整理副本/4项目、每项目12项与汇总6项通过，退出0；独立发现200输出缺日期/剧场及旧幕次混入事项。`input_scope_audit.json`/`runner_snapshot.py`保留，不当规范命名通过 |
| `u8_raw_workflow_200_20261023/` | **原始流程完成**：200图/4项目、70功能检查、12任务完成，保留复制161/跳过39/0失败；命名、8张确认/拒绝探针、并发冲突、重跑保护、原图/图库及被测源码检查通过。262原始匹配/5预检在resume目录；正式U8仍`not_passed` |
| `u4_sourced_{siglip2_baseline,siglip2_samepool,gemma2_samepool,siglip2_expanded,gemma2_expanded}_20261022/` | 五份完成报告，各7项检查、退出0；stdout日志为父目录同名`.stdout.log`；会话38976已结束 |
| `u3_cluster_crops_http_200_20261021/` | 新版 HTTP `summary.json`、任务数据、`server.log`；目录日期沿用前一天 |
| `u3_cluster_crops_completion_200_20261021/` | 四次恢复 `summary.json`、`artifact_audit.json`、`server_0.log` 至 `server_3.log` |
| `u3_cluster_previews_200x3_20261021/` | 无损裁剪预览三轮对照 |
| `u2_publication_200x5_20261021/` | 暂停前完整 U2 复验及 `not_passed` 依据 |

原25候选（5正文+20标签）历史结果：

| 模型 | Recall@1 | Recall@3 | 命中数（Top-1 / Top-3） |
| --- | ---: | ---: | ---: |
| SigLIP2 base | 18.85% | 55.00% | 49 / 143，分母均为 260 |
| EmbeddingGemma 2 | 17.31% | 58.08% | 45 / 151，分母均为 260 |

Gemma R@3 净增 8 张，但 R@1 净减 4 张；《天下归心》R@3 70.59%→55.88%，未呈现跨剧目一致收益。对 4 剧目重采样的 R@3 差值探索性 95% 区间为 −6.62～+7.78 pp，样本和划分不足以支持正式提升声明，默认保留 SigLIP2。完整逐剧目/逐图比较见报告。

原25候选仅5个有分幕正文，其余20个只有标签；《马可·波罗》3候选的R@3必然100%。本地原资料保持原样，新来源v1位于隔离目录，不按整剧梗概猜编号。

当前来源v1结果（分母260，全部原query与GT保留）：

| 模型 / 候选版本 | Recall@1 | Recall@3 | Top-1 / Top-3命中 |
| --- | ---: | ---: | ---: |
| SigLIP2 · 增补25 | 14.62% | 61.54% | 38 / 160 |
| Gemma · 增补25 | 29.62% | 72.31% | 77 / 188 |
| SigLIP2 · 扩展28 | 14.23% | 49.62% | 37 / 129 |
| Gemma · 扩展28 | 23.08% | 68.08% | 60 / 177 |

固定25候选补12正文（假面6、样式雷3、马可3），保留5旧正文及8标签。扩展只追加马可第3/4幕与尾声，没有对应query GT。Gemma原文基线引用较早报告：模型/预处理与7份非入口源码相同，入口版本不同；五份新报告9份源码相同且匹配现场。固定池Gemma净增32/37张、SigLIP2净减11张Top-1并净增17张Top-3。

扩池后马可6候选R@3为SigLIP2 47.46% / Gemma79.66%，R@1为27.12% / 23.73%。Gemma重编码造成原池2张Top-1及1张Top-3新增命中；用已存28候选分数限制回原25得到79/189，再增加候选竞争为60/177，竞争丢失19/12张均发生在马可。SigLIP2原池排名未变，竞争丢失1/31张。分数微变最大0.001310，分批/BF16只是可能原因，未独立消融。

全部12新增文档73–190 tokens，SigLIP2只处理64，证据存于`text_window_audit.json`。马可正文来自2015压缩巡演，而照片文件名提示2014，等同性未核验；样式雷仍缺第2/3/4/5/7场、尾声、谢幕，假面缺序曲。天下归心Gemma相对同文本SigLIP2 R@3下降14.71 pp；当前开发集未达正式U4条件，默认保持SigLIP2。详见新比较与来源指南。

报告仍为开发集状态，final test 未锁定。原始计时 20.309 / 54.789 秒的加载边界不一致（Gemma 在计时内懒加载，SigLIP2 在计时前加载），且共享 GPU、只跑一遍，不能用于速度比较；`scene_macro_recall_at_1` 不能改称 Macro-F1。后续改善文本或模型时保留本轮产物，使用新目录。

新短块协议结果（全部260 query保留；长文控制在本协议重跑）：

| 模型 / 池 | 长文R@1 / R@3 | 均值块R@1 / R@3（主） | 最大值块R@1 / R@3（次） |
| --- | ---: | ---: | ---: |
| SigLIP2 · 25 | 14.62% / 61.54% | 23.46% / 70.00% | 23.08% / 68.46% |
| SigLIP2 · 28 | 14.23% / 49.62% | 23.08% / 53.46% | 21.54% / 58.08% |
| Gemma · 25 | 30.38% / 72.69% | 27.69% / 76.54% | 38.85% / 76.92% |
| Gemma · 28 | 23.08% / 67.31% | 21.54% / 71.92% | 31.92% / 70.00% |

82块覆盖全部v1已选证据和5段旧本地正文，102份去重文档每段batch1/后端，260图各编码一次（内部batch8）。原25候选/28实验池同分数表，去掉新增ID后原排名与分数逐项完全一致；扩池只有马可竞争，均值损失SigLIP2 1/43张、Gemma 16/12张Top-1/Top-3。样式雷Gemma均值相对自身长文R@3下降8.16 pp，正式U4未通过。最大值有块数偏置，均值等权包括短尾块；最高显示结果不直接采用。

本协议长文Gemma 25/28池为79/189和60/175命中；历史v1为77/188、60/177，完整排名分别50/59图变化，最大分数差0.001434。权重/预处理签名一致，但batch1/共同去重与入口协议不同，没有单独证明原因；比较分块只对本协议长文。全部12变体分数/排名/指标、三候选表和向量点乘精确复现，102检查通过。

演出版审计16来源/21段精确跨度、10检查通过；2017假面与2024同2012瑞典版谱系，马可2014官网明确第三幕战争/舞美结尾调整、2015巡演再压缩，样式雷2016复排与6.10–6.19日程获支持。照片日期/彩排及逐场版本等同仍未核验，8项正文缺失保留。新研究官网正文不静默加入模型；原始URL日期与页面日期分开（马可正文2014/08/26、2014/09/25）。完整分析/复现与来源边界见指南。

## 6. 当前验证与仍缺的验收

2026-10-23 当前源码的验证：

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 6处修复 | 视觉14、OCR6、窗口10，共30项通过；初测与失败复现保留 | `resume_validation_20261023/fix_verification.json`及对应日志 |
| 最终任务回归 | 核心28、整理17、任务核心10、聚类19、建库17，共91项通过 | `task_regression_final.json`及5份`.final.log` |
| 当前工作台 | API25项；35图/五屏宽浏览器、万条元数据/按页详情、草稿/并发/单条更新通过 | `workspace_summary.json`、`workspace_api.log`、`browser.log`及截图 |
| 当前U3真实人脸 | 292图/1095脸，HTTP10项、四次服务恢复17项通过；453簇/0失败 | `u3_cluster_{http,recovery}_292_20261023/summary.json` |
| U5真实请求/OCR | 100图中15观察/30请求，14检查；5图OCR的5项完整性检查；重放无新请求/worker | 两项20261023目录及汇总；事实/转录人工标注仍0 |
| U6/U7探索 | 万条特征10项、262图时序7项；1095脸图聚类4200假定约束0违反 | 三项20261023报告；质量仍`insufficient_data` |
| U8实际流程 | 原始200图/4项目70项功能检查通过，161复制/39跳过/0失败；5预检通过。首轮已整理副本仍保留命名范围问题 | `u8_raw_workflow_200_20261023/{summary,run_state}.json`及各项目报告；旧`input_scope_audit.json` |
| 最终独立核查 | 保存的源码/向量/原始路径/图库、逐输出命名与字节、实际模型结果、文档链接/语法/diff | `resume_validation_20261023/audit_completed_acceptance.py`及`completed_acceptance_audit.json`；质量门槛与U2暂停独立记录 |

下表保留 2026-10-22 的历史验证与当时版本范围：

| 检查 | 本轮结果 | 证据 |
| --- | --- | --- |
| 身份 / 核心 / 整理 / 工作台 | 15/15、28/28、17/17、24/24，共 84 项通过，退出码均为 0 | `u1_regression_final_20261022/summary.json` 及逐组日志；含测试与业务源码摘要 |
| 浏览器 | 35 张真实剧照、五种屏宽、身份表参数/刷新、复核/恢复/错误分支等 21 个列出检查通过 | 同目录 `browser.log`、4 张截图；GPU 操作用替代响应，只验 UI/API 交互 |
| 文档示例 | 角色表经真实 `IdentityRegistry` 验证，JSON 请求示例可解析 | `docs/identity_registry_guide.md`；本轮只读检查 |
| 两模型运行（原25历史） | 各自5项检查通过，退出0，评测入口随后增加来源校验 | 两个`followup`目录`summary.json`与上级`.stdout.log` |
| 配对可比性（原25历史） | 12项检查通过；8份源码相同并匹配当时版本；清单字节、标签/角色表、260 query/GT和候选池一致 | `u4_verified_comparison_20261022/summary.json` |
| 来源契约 | 10项通过；原文跨度、场次、query隔离、改写拒绝与扩池保留 | `u4_sourced_comparison_20261022/source_tests.log`；测试已加入Git忽略例外 |
| 新来源模型比较 | 五份新报告各7项完整性检查，45项跨报告审计通过；源码/来源/标签/原图/PNG匹配，冻结准备清单重建一致 | `u4_sourced_comparison_20261022/summary.json`；原Gemma较早入口版本单独解释 |
| 文本窗口 | CPU处理器审计：12新增/15扩展来源文档全部超过64 tokens，处理长度64 | 同目录`text_window_audit.json`、脚本与stdout日志；未加载模型/使用CUDA |
| 短块契约与两模型 | 10测试、82块≤64 tokens；两后端各12检查、六变体/后端，均退出0，102去重文档/260图各编码一次 | `u4_text_blocks_20261022/block_tests.log`及两模型目录`summary.json`/运行JSON；会话22447退出0 |
| 保存数组分析 | 102检查通过；完整向量点乘、均值/最大值/长文候选表、全部12排名/精确分数/指标重建；25/28移除新ID后原评分顺序完全相同 | 研究`summary.json`、`analyze_saved_vectors.py`、`analysis_reproduction.log`；CSV与Markdown一并精确复现 |
| 演出版来源 | 16来源/21原文跨度、6搜索发现、10检查通过；新资料不改变模型正文/GT/人工角色决定，缺失8项保留 | `production_chronology_audit.json`/报告、`chronology_texts/`、`chronology_reproduction.log`；已只读复现 |
| 照片复核准备 | 9契约测试、20完整性；319像素/dHash重建、582原图与109图库内容/成员不变、759产物只读核查；375项均pending | `u0_partition_review_20261022/artifacts/{summary,integrity_checks,artifact_manifest}.json`、`verification.log`；无人工GT/模型运行 |
| 离线复核浏览器 | 12检查通过，分类/文字/pending、刷新保存、JSON导出/恢复、错误协议拒绝、3屏宽、真实预览/时间流；沙箱套接字失败后获准宿主重试 | `browser_summary.json`/3截图；会话33879退出0，临时浏览器草稿不计人工确认 |
| 连字符场次导入修正 | 6回归通过（修正前4失败保留），12差异审计通过；仅两图scene_label/scene_id补写，263/262新版本，注册表字节相同 | `import_gap_reproduction.log`、`import_gap_tests.log`、`import_gap_audit.json`/报告；旧260/冻结U4保留 |
| U0最终摘要/文档核查 | 28检查通过，90本地链接有效；759复核产物、全部582原图/图库、旧/新标注及冻结短块协议/11源码摘要不变；语法与diff通过 | `u0_partition_review_20261022/documentation_checks.json`与`final_audit.py`；会话61551退出0 |

初次工作台测试在沙箱首个请求等待 180 秒后超时；同源码宿主重跑 23/23，通过后新增演员专辑 API 用例，最终宿主 24/24。超时与修复前同名问题的失败日志均保留，没有改写为通过。

U3 历史200图 HTTP/恢复报告保留；2026-10-23 已按当前源码另建292图复验，不跨版本重放旧任务。正式身份阈值/margin 校准、gallery/query 隔离、框级身份、连拍/边界及 final test 仍缺验收数据。

## 7. 下一步执行顺序

**2026-10-24接续覆盖说明：** 上轮列为“正在准备”的100图观察、双人事实/OCR页面和身份评测入口现已完成，先看`completion_audit_20261024/summary.json`及[双人指南](docs/visual_fact_review_guide.md)、[逐脸指南](docs/face_identity_review_guide.md)。原始模板仍pending。用户已明确无新增GT、独立照片和两位复核人，不能自动补人工决定；后续收到真实复核/独立资料后再生成新评测报告。无新代码或数据不重跑已完成模型、浏览器和规模实验。下列历史待办中的兼容快照问题已由延后写入解决，显式导出成本和受控性能仍待验收。

1. **先读当前验收状态，保留已完成结果。** `resume_validation_20261023/stage_status.json`分别记录实现/功能/规模/质量与默认选择；最终独立核查为`completed_acceptance_audit.json`，真实原始流程为`u8_raw_workflow_200_20261023/summary.json`。本轮服务均结束，无需恢复旧会话或重跑相同输入。首轮已整理副本范围问题保留；新200图重处理已通过70项，正式U8仍不通过。
2. **补U0/U1独立数据与人工标注。** 使用[照片复核指南](docs/photo_partition_review_guide.md)处理375项、14相机流/机身时钟、近重复/真实连拍和图库来源/框；王浩伟角色冲突仍需人工核对。图库108裁剪/46图与旧query名称对应，gallery/query隔离未完成。每剧仅1日期线索，现有数据仍development；补框级已知/unknown、独立场次/final test后再校准，不能把图级未列名当unknown。
3. **补U4–U7质量证据。** 8项正文、马可第3/4幕/尾声正例与精确演出版未补齐；U5至少100图双人事实/文字转录、U6边界与可信时序、U7身份/镜像例外/连拍偏好仍缺。262图探索只有1个两图组，不够选优验收；未运行DINOv3/AdaFace/TOPIQ/MUSIQ明确留空。
4. **保持已完成节点和默认选择。** 263/262新版标注及262图模型研究已完成，不再准备或重跑同一协议；旧260报告/短块协议/原图/正式图库保留。U3当前292图/1095脸可靠性复验和U5–U7功能/规模检查已有结果，无新改动不重复运行；远端权重核验仍缺。`581f8fa`已完成前轮44文件提交，本轮命名/复核续作未提交、未推送。默认SigLIP2/legacy，U2提速与DALI继续暂停。

5. **使用新增逐脸复核材料完成人工依赖。** [1095脸复核页面](outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023/index.html)及[指南](docs/face_identity_review_guide.md)已准备，984对齐裁剪/111框上下文、35身份、12原图分页。选择已知/未知/不可评价及镜像例外，填写复核人和依据；草稿导出后校验，不自动应用标注或锁定test。当前人工核对仍0。U3全量兼容JSONL成本/远端权重、U0真实基线等剩余待办保留；不重复已通过GPU输入。

短块与演出版只读复现（已通过，无需GPU；绑定源码不变时使用）：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python outputs/acceptance/core_upgrade/u4_text_blocks_20261022/analyze_saved_vectors.py --verify-existing
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python outputs/acceptance/core_upgrade/u4_text_blocks_20261022/audit_production_chronology.py --verify-existing
```

该目录的`run_models.py`、`launch_commands.txt`、运行JSON保留已执行命令、进程与退出码；勿用旧非空输出重测。文档和哈希检查保存到`documentation_checks.json`。源码/本地来源文件摘要变化会阻止旧协议使用，这是预期版本边界。

回归复现命令（本轮已通过；后续业务代码变化时按影响范围复跑）：

```bash
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_identity_registry.py -v
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_core_upgrade.py -v
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_organize_tasks.py -v
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_workspace.py -v
```

浏览器服务启动方式见 [核心升级进度的运行方式](docs/core_upgrade_progress.md)；确认测试服务与端口后运行 `WORKSPACE_URL=http://127.0.0.1:18301 node tests/test_workspace_browser.cjs`。GPU 业务请求在浏览器回归中有替代，结果仅证明 UI/API 交互。

Gemma 本轮执行命令（已经完成；需要新实验时更换非空输出目录，不能原地覆盖）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=outputs/core_upgrade_deps MPLCONFIGDIR=/tmp/theatre-core-mpl \
  /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/evaluate_verified_scene_retrieval.py \
  --labels outputs/acceptance/core_upgrade/verified_labels_roles_20261022/verified_photo_labels.json \
  --registry outputs/acceptance/core_upgrade/verified_labels_roles_20261022/identity_registry.json \
  --source-root data/ncpa_test \
  --prepared-inputs outputs/acceptance/core_upgrade/u4_verified_siglip2_final_20261022/input_manifest.json \
  --backend embeddinggemma2 --device cuda:0 \
  --output outputs/acceptance/core_upgrade/u4_verified_gemma2_followup_20261022
```

Gemma 的本地权重配置与独立依赖说明见核心升级进度。GPU 编号以现场可见设备为准；网络、GPU 或依赖不可用时，保存真实错误和未完成状态。

## 8. 中断恢复与协作约定

- 每完成一次代码块、测试、长实验或提交，更新“当前交接点”和下表。长任务启动后立即记录输出目录、日志、PID/服务端口或任务 ID，再继续等待。
- 同版本持久任务通过已有任务 API/恢复工具继续；源码、模型、输入或角色表版本不一致时保留旧报告，恢复匹配版本或新建任务，不删版本校验来强行续跑。
- 评测目录非空时工具会拒绝复跑。只有输入而没有`summary.json`的目录不能记完成；引用其他目录图片/来源的清单需连同依赖保存。U4来源v1冻结，不在原目录重跑构建脚本；新增规则/资料用新目录版本。跨机交接同时提供本地报告、标注、来源快照与中性PNG；Git文档链接本身不包含这些文件。
- 多人/多智能体协作时，开始前在下表登记负责人、文件范围、输入版本和产物目录。共享的 `app/main.py`、`app/organize_tasks.py`、角色表、测试服务、GPU 测量由执行者协调，避免同时写同一文件或污染测量负载。
- 可分别负责身份逻辑/测试、标注与模型评测、UI/文档；改公共契约前先同步。集成者检查整个差异和对应回归，保留不属于自己负责范围的改动。
- 状态文件记录事实与待办，不包含密钥、环境变量全集或原始人脸内容。没有运行的检查写“未运行”，只有历史记录的写清对应版本。

| 时间（UTC） | 负责人 / 范围 | 关键节点 | 验证 / 产物 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-10-22 | 主会话 / 状态与文档 | 盘点 `49b9ce0` 上未提交 U1/U0 改动，恢复 U4 完成报告索引 | 第 5–6 节；无活动长任务交接 | U1 回归、使用说明、Gemma 同输入对照 |
| 2026-10-22 02:48 | 主会话 / U1 回归 | 顺序运行身份、核心正确性、整理任务、工作台四组测试 | `outputs/acceptance/core_upgrade/u1_regression_initial_20261022/summary.json`；工具会话 `98529` | 等待结果并审查身份契约；再补使用说明 |
| 2026-10-22 | 主会话 / 身份与专辑 | 初次身份 12/12、核心 28/28、整理 17/17；工作台在沙箱首个请求超时，宿主重跑 23/23 | `u1_regression_initial_20261022/workspace_host_retry.json`；原始失败记录保留 | 新增同名专辑及筛选回归 |
| 2026-10-22 | 主会话 / U1 修复与说明 | 3 个新测试复现 4 处断言失败后修复跨照片同名合并和姓名/ID 筛选；补演员专辑 API 测试与角色表指南 | 初次目录 `same_name_reproduction.log`；新身份测试 15/15；`docs/identity_registry_guide.md` | 修复后完整回归与浏览器检查，再做 U4 同输入对照 |
| 2026-10-22 | 主会话 / 当前版本验收 | 身份 15/15、核心 28/28、整理 17/17、工作台 24/24；指南 JSON 示例验证通过 | `u1_regression_final_20261022/summary.json`；浏览器会话 `97863`，隔离服务 `56203` | 保存浏览器结果并关闭测试服务 |
| 2026-10-22 | 主会话 / U4 同输入对照 | 使用原 260 图中性清单和 25 个候选，先复跑当前源码的 SigLIP2 | `u4_verified_comparison_20261022/run_state.json`；会话 `44831`；输出 `u4_verified_siglip2_followup_20261022/` | 同一源码运行 Gemma，再核对输入/版本并生成比较报告 |
| 2026-10-22 | 主会话 / 浏览器与 Gemma | 浏览器通过并将截图复制到本轮验收目录；SigLIP2 已完成，Gemma 已启动 | `u1_regression_final_20261022/browser.log`、`browser_screenshots/`；Gemma 会话 `73799` | 等待 Gemma、核对两个报告并更新最终接续点 |
| 2026-10-22 | 主会话 / 本轮完成 | Gemma 正常结束，配对 12 项检查通过；R@3 +3.08 pp、R@1 −1.54 pp，保持 SigLIP2 默认；确认三剧目缺分幕正文 | `u4_verified_comparison_20261022/report.md`、`summary.json`、分析脚本；测试服务正常退出 | 补有出处的候选文本与 U0/身份正式标注；U2 暂停 |
| 2026-10-22 03:36 | 主会话 / U4 官方资料采集 | 核对本地三条 NCPA 专题网址、抓取策略和 25 个候选标签；用户明确授权网上搜集剧目资料 | 新目录 `u4_ncpa_sources_20261022/`；尚未采集、未启动 GPU；保留全部旧输入 | 保存官网快照、哈希和原文引用；确认场次映射后冻结候选文本版本 |
| 2026-10-22 | 主会话 / U4 官方来源恢复 | 沙箱 Python DNS 失败后获准宿主重试，三条专题 HTTP 200，快照和 SHA256 已保存；robots.txt 404。找到《假面舞会》2024 演出页六个明确分幕/分景段落 | `source_manifest.json`、`sandbox_fetch_attempt.json`、`raw/ncpa_ball_2024_event.html`；《样式雷》新专题与两部旧票务页也已保存 | 查询官网站内搜索和关联资料；场次仍缺的记录为缺项，不按整剧梗概猜序号 |
| 2026-10-22 04:39 | 主会话 / U4 来源契约 | 官网明确《样式雷》第1、6、8场；中国侨网转北京日报逐幕报道《马可·波罗》，央视节目页核对编号；发现巡演压缩版限制和3个未观测后续幕次 | `tools/sourced_scene_candidates.py`、评测入口；`tests/test_sourced_scene_candidates.py` 10/10，暂未 GPU；样式雷第2场顺序暗示暂不采用 | 保存精确原文跨度、版本限制和缺项；固定25候选对照，再独立28候选完整池实验 |
| 2026-10-22 04:44 | 主会话 / U4 候选版本冻结 | 六个引用来源、原文跨度和12个已映射候选已生成；马可第三、四幕与尾声作为3个额外候选；query、GT、原候选次序全部保留 | `candidate_texts_v1.json`、`candidate_texts_expanded_v1.json`、`prepared_samepool_v1/`、`prepared_expanded_v1/`、`source_audit.json`、重建脚本；10/10日志存入比较目录 | 宿主GPU顺序运行五份新报告；记录64-token文本截断及巡演版本限制 |
| 2026-10-22 05:08 | 主会话 / U4 模型对照启动 | 新入口SigLIP2基线R@1 18.85%、R@3 55.00%，旧260图排名完全复现，7检查通过；四次后续运行获准宿主执行 | 比较目录 `run_remaining_models.py`、`run_state.json`；会话 `38976`，初始SigLIP2 PID `615109`；全部日志在父目录 | 等待四份报告，核对来源/输入/模型签名，分别分析文本与扩池效果 |
| 2026-10-22 05:20 | 主会话 / U4 中断接续 | 核对run_state与会话38976，五份评测均退出0，GPU实验结束；固定25/扩展28候选输入保持冻结 | `u4_sourced_comparison_20261022/run_state.json`；Gemma分别77/188与60/177命中，分母260 | 核对五份报告完整性、旧Gemma可比性；生成可重现分析、来源映射指南并更新进度 |
| 2026-10-22 | 主会话 / U4分析与文档 | 比较45项审计通过，保存逐剧目/幕次配对与扩池诊断、6来源原文/跨度/缺项、CPU64-token证据；更新指南与状态 | `u4_sourced_comparison_20261022/{summary.json,report.md,comparison_analysis.py,text_window_audit.json}`、`u4_ncpa_sources_20261022/source_report.md`、`docs/sourced_scene_candidates_guide.md` | 检查文档链接/哈希与diff，再登记完成；后续核对8项正文、演出版、新文本块及U0独立标注 |
| 2026-10-22 05:51 | 主会话 / U4来源v1完成交接 | 分析、来源报告、指南与状态同步完成；本地文档链接/摘要、五次退出码和diff检查通过；保留所有冻结输入与旧报告 | 比较目录`documentation_checks.json`及`run_state.json`登记完成；来源测试仍10项、比较审计45项；无提交/推送 | 下一次按第7节推进版本核对、新文本块及U0独立标注；U2保持暂停 |
| 2026-10-22 06:25 | 主会话 / U4短原文块接续 | 复核v1所有实验已结束；官网2014升级版与2015巡演压缩差异保留。准备独立短块协议及同分数扩池比较 | 新目录`u4_text_blocks_20261022/run_state.json`；尚未新GPU评测，不改v1来源/评测源码 | 固定token预算、引用跨度及聚合规则；实现/测试后顺序运行两个后端 |
| 2026-10-22 07:02 | 主会话 / U4短块协议冻结 | 260图、28候选、82原文块覆盖全部v1已选证据，最长64 tokens；去重102文档，候选内部无重复索引；实际块数更正为82 | `prepared/protocol.json`及SHA256、11源码绑定、`preflight.json`、10/10测试；`run_models.py`语法通过，计划与日志路径已登记；沙箱CUDA不可用 | 按沙箱规则请求新执行器宿主运行，两后端共12变体；完成后分析数组及演出版来源 |
| 2026-10-22 07:04 | 主会话 / U4短块GPU启动 | 新执行器宿主权限已获准，CUDA可用；SigLIP2已完成，Gemma已启动；不改冻结测量源码 | 会话22447，执行器PID 979682，SigLIP2 PID 981747、Gemma PID 986162；`launch_observation.json`与运行JSON/日志 | 等待两后端及12变体，保存数组复现分析、演出版核对与文档 |
| 2026-10-22 07:06 | 主会话 / U4短块模型完成 | 两模型退出0，102文档/260图各编码一次，全部12预声明变体保存；各12完整性检查通过 | 会话22447退出0；SigLIP2 PID 981747、Gemma PID 986162；各`summary.json`/向量/分数哈希见研究JSON | 逐数组复现分数与排名，配对/逐剧目/场次分析；保存新版来源审计和文档 |
| 2026-10-22 07:23 | 主会话 / 数组与来源审计 | 102项数组审计通过，12变体/CSV/报告精确复现；16来源/21段跨度、10来源审计通过，保留8缺项/版本边界；首次分析Markdown历史指标键错误已修复，错误日志保留 | `analysis_reproduction.log`、`production_chronology_audit.json`、`chronology_reproduction.log`、`analysis_initial_reporting_error.log`；测量源码/输入未改 | 同步指南/进度/交接，核对文档链接、哈希与diff；后续独立标注/补来源 |

| 2026-10-22 07:37 | 主会话 / U4短块完成交接 | 两后端12变体、102数组审计、10来源审计与只读复现完成；20文档/哈希检查通过，70本地链接有效；默认SigLIP2、分块仅离线、U2暂停 | `u4_text_blocks_20261022/run_state.json`状态为`development_study_completed_with_open_validation_gaps`；`documentation_checks.json`、完整报告/CSV；无活动GPU、无提交/推送 | 独立演出/近重复/连拍划分、8正文/新候选正例、精确版本；U1冲突/框级标注与正式校准 |
| 2026-10-22 07:55 | 主会话 / U0划分只读盘点 | 582文件/319图、263已标注；29张未导入舞台照均为样式雷；图库109裁剪名称唯一对应47源图；每剧1日期，原图拍摄时间多数存在 | 原清单/标注/图库只读；新目录`u0_partition_review_20261022/`计划中；未修改冻结源码、未启动GPU | 新离线核查工具、输入摘要/EXIF/图库来源与可浏览近重复/时间序列；不自动确认为独立场次或final test |
| 2026-10-22 08:12 | 主会话 / U0复核准备工具 | 新独立工具、逐图校验和检查点/同版resume、原始EXIF/相机分流、使用记录、近重复/同名/时间及图库来源页面；9契约测试及语法通过 | `tools/prepare_photo_partition_review.py`、`tests/test_photo_partition_review.py`；新目录`run_review.py`/`run_state.json`；旧绑定源码未改 | CPU准备真实319图/109裁剪；保存命令/PID/日志，完成后核查摘要、页面、输入及冻结协议 |
| 2026-10-22 08:36 | 主会话 / U0复核完成与导入缺口 | CPU退出0：286原始拍摄时间、14流/14逆序、4近重复对、49同名组/17跨幕次、375复核项；20完整性/759产物只读核查、宿主浏览器12项通过 | 会话59907/33879均退出0；`u0_partition_review_20261022/artifacts/summary.json`、复核HTML/模板、`verification.log`/`browser_summary.json`/3截图；未作人工决定 | 连字符分隔的已确认幕次2处被旧导入器漏读；修独立import工具并另生成新263图/262幕次标注版本，旧标注和U4冻结产物不改 |
| 2026-10-22 08:55 | 主会话 / U0新版标注与文档 | 连字符幕次先复现4失败，修正后6测试通过；新263图/262幕次12差异审计通过、旧260标签/人物/注册表字节保留；补照片复核指南、阶段/版本/状态入口 | 导入会话53360退出0；新`verified_labels_scene_segments_v2_20261022/`与`import_gap_audit.json`；短块协议及11源码摘要不变；未新GPU | 最终文档链接/源码/产物摘要核查，保存完成状态；后续人工核查/独立数据、新262协议、8正文/新增马可正例 |
| 2026-10-22 09:07 | 主会话 / U0本轮完成交接 | 28摘要/文档检查通过、90本地链接有效；语法/diff通过，全部旧评测/原图/图库/注册表字节保留；新263/262标注与375复核项完成，无活动长任务 | `u0_partition_review_20261022/documentation_checks.json`、运行/交接JSON、指南；61551退出0；无提交/推送 | 从第7节核查/独立数据/8项正文/马可新候选正例接续；262图新比较另建协议，U2暂停 |

| 2026-10-22 09:23 | 主会话 / 提交准备 | 用户授权“提交改动并继续”；核对35份实现/测试/文档范围与现有验证，diff无空白错误；U1全部源码与测试摘要、U0四份工具/测试摘要匹配 | U1 84项/浏览器及U0既有报告继续有效；未重跑已通过测试，outputs/保持忽略 | 显式暂存并本地提交，登记真实哈希；新目录接续8项正文与262query中性协议 |

| 2026-10-22 09:29 | 主会话 / 提交完成与续作 | `9c97bb3`本地提交成功，35文件/3996新增/83删除，暂存diff通过、提交后工作区干净、领先上游1提交，未推送 | 完整哈希`9c97bb3948e58fd11ff8502978d314bdf55a0f16`；旧U1/U0工具摘要匹配 | 新`u4_inputs_scene_segments_v2_20261022/`准备262中性query和25/28同文本协议；新`u4_ncpa_source_followup_v2_20261022/`核查缺失正文，不重跑旧GPU实验 |

| 2026-10-22 14:33 | 主会话 / 持续目标与262评测 | 用户要求持续至待定任务完成，确认data资料、继续暂停U2；全data/测试说明docx读取，未发现视频及框级/unknown/边界/偏好GT。262输入23检查已完成，续查18来源/21搜索/20引文15检查通过/重现 | 新动态评测3工具/10测试通过；协议准备会话4855；源资料和旧报告保留，未新增提交/推送 | 完成262增量编码/数组审计，接续U3/U5–U8可完成项，formal数据缺项明确记录 |

| 2026-10-23 | 主会话 / 中断后实现复核 | 新增模块初测40项39通过；OCR缓存序列化复用失败，另复现权重变化、远端未核验权重跨进程恢复、隐藏HTTP重试、缺特征与倒退时间戳共6处失败并修复；14/6/10项复验通过 | `resume_validation_20261023/{run_state,integration_initial,reproduction,fix_verification}.json`；初次集成六组退出0，修复前后日志保留；无活动GPU/测试服务 | 真实OCR、U6/U7预计算规模与数据探索、当前版工作台/真实服务，完成文档与缺项报告；U2暂停 |

| 2026-10-23 | 主会话 / 当前版本真实检查 | 100图视觉预算30实际请求/重放0新增、5图OCR重建缓存；万条特征10项、262图时序7项、1095脸图聚类假定约束0违反；292图HTTP10项/四次恢复17项完成 | `u5_*_20261023/`、`u6_u7_*_20261023/`、`u7_graph_real_1095_20261023/`、`u3_cluster_*_292_20261023/`；详情见第5节 | U8实际流程、独立产物/文档核对；不将合成速度与功能通过当质量验收 |
| 2026-10-23 03:40 | 主会话 / U8首轮与命名范围 | 首轮200已整理副本/4项目功能检查完成退出0，但独立发现200输出缺日期/剧场，旧幕次混入事项；不作规范命名通过，已结束服务无需终止 | `u8_real_workflow_200_20261023/{summary,input_scope_audit}.json`、`runner_snapshot.py`；当前工作台25项/任务91项通过 | 找真实原始输入、补预检和逐输出命名检查后独立重验 |
| 2026-10-23 03:46 | 主会话 / 原始U8启动 | 全部262已确认照片按完整文件SHA256找到原始路径，无同剧目同名冲突；5项预检通过，错误输入在模型加载前拒绝。相同200图字节、每剧两张保证保留的复核探针，GPU1启动 | 会话14437、服务794754/38913，`u8_raw_workflow_200_20261023/`；原始清单/映射/预检在resume目录 | 等实际结果，最终审计脚本`audit_completed_acceptance.py`已准备；当前1255摘要/105本地链接一致 |
| 2026-10-23 04:16 | 主会话 / 原始U8完成与交接 | 四项目200输入，161保留复制/39跳过/0失败；16×4+6=70功能检查通过，12任务完成，8张复核探针均保留，原图/正式图库/源码不变；服务已结束 | 会话14437退出0；`u8_raw_workflow_200_20261023/summary.json`；最终独立核查脚本/结果、`stage_status.json`和指南记录实际范围 | 独立场次、框级/unknown、事实/边界/偏好GT及剩8正文；U2暂停，默认保持，当前新改动未提交/推送 |

| 2026-10-23 | 主会话 / 提交与续作 | 前轮44文件提交`581f8fa`；用户要求持续处理未完成项，U2保持暂停。修复普通目录中的已整理命名，8解析/18整理/6导入/28核心/15身份/17建库/9照片复核通过；582命名/263幕次审计通过 | 新`resume_followup_20261023/`；真实模型200图/4项目70项、12任务完成，200复制/0失败，原图/图库不变；会话82374退出0 | 补分析恢复解析器摘要、API26项宿主通过，保留真实报告先于该补充的版本边界 |
| 2026-10-23 | 主会话 / 逐脸标注准备 | 新1095脸复核包、984裁剪复用/111既有框上下文截取，零模型前向；初次遗漏拼图上限失败保留，重绘失焦错误修复；10契约/12五屏宽浏览器通过 | `u1_face_identity_review_final_20261023/`、本轮目录截图/脚本/汇总；原图和正式图库只读，人工标签仍0 | 完成文档/独立审计；后续由实际复核与独立数据解除质量验收缺项 |

后续记录模板：`时间｜负责人/文件范围｜完成或失败的节点｜命令/退出码/日志/任务ID/源码版本｜下一步`。更新当前摘要，同时追加节点，保留历史验证的版本范围。

| 2026-10-24 | 主会话 / U3单条保存与模型审计 | API先复现快照重写失败，再加可选延后快照与工作台参数；12存储/27API通过，一万条记录5次/9检查；远端/models为200、别名存在，get_model_info为404，无权重证明 | `completion_followup_20261023/`、`u3_review_snapshot_scale_20261023/`、`remote_model_metadata_{sandbox,host}_20261023/`；所有相应会话结束 | 用户确认无新增标注资料；按需补100图真实观察，继续双人事实/OCR复核与身份评测契约，U2暂停 |
| 2026-10-24 | 主会话 / 质量复核接续收尾 | 确认100图观察/105复核任务和1095脸评测已完成；修复双人完整来源校验与重复图计数，7失败子例→15测试通过，现有待定模板通过新校验；补双人/身份指南和当前状态 | `completion_audit_20261024/`记录测试、新待定报告、原图/图库/包完整性及版本边界；旧v2 HTML和推理结果保留，服务退出已核实 | 等真实复核或独立资料；U2暂停、正式质量不足，当前改动未提交/推送 |
| 2026-10-24 | 主会话 / 用户授权提交 | 35份待提交文件与完成审计时的摘要完全一致；只更新提交交接说明，不重跑无新改动的测试 | 显式暂存源码/测试/文档，`git diff --check`通过；真实提交结果另存`completion_audit_20261024/commit_receipt.json` | 本地提交，图片、模型和实验产物继续忽略；未推送，U2暂停 |
