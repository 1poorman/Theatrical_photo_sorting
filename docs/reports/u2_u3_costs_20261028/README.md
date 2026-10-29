# U2/U3 历史整树与工作流成本测量

本轮完成完整历史业务源码基线、交替顺序、全量导出及真实HTTP五轮成本测量。U2特征仅提速1.363×、19条演员显示/命名不同；HTTP流程检查通过，正式U2/U3/U8仍未通过。聚合指标见[metrics.json](metrics.json)。

## 测量范围

- `fdc1fda`的`core_modules/`与`config/`完整Git归档，在独立进程加载全部业务依赖，校验实际导入来源。旧三入口基线报告保留其原范围。
- 200张唯一原始剧照、4剧目、8目录，0张来自“整理完成”目录。历史逐图processor与当前processor默认配置，各5次；轮次顺序为AB/BA/AB/BA/AB。
- 当前默认为8图/批、2解码/1质量/1哈希线程；两侧Torch、OMP、MKL、OpenBLAS均限制4线程。真实SigLIP2与SCRFD/ArcFace，景别、行当、远端LLM关闭。
- 主计时为模型已预热、特征缓存为空；冷启动加载与签名计算、热缓存另报。每个方案每轮独立进程；每轮保留热缓存验证记录。报告及归档复制包括在整理计时中。
- GPU1共享，连续记录整卡显存、利用率、竞争进程及CPU负载。交替顺序不能证明独占或受控环境；不能把外部负载变化记为代码收益。
- 真实持久HTTP成本另用旧U8清单的200张原始输入新副本，4项目，5次独立空缓存、同一已预热服务。reason→脚本复核→全量导出→organize包含HTTP与轮询；不计准备、模型预热、产物完整性审计。脚本决定仅复用既有照片标签，不新增GT。
- 10000条合成复核记录在即时/延后快照模式下各5次显式导出；单条决定与“决定＋导出”分别报告，验证快照/历史/最终标签与SQLite一致。

## U2：完整历史业务树对照

基线为`fdc1fda48e224d4d573243e1a51f936841bfaf1a`完整业务树；当前实现提交为`40f2a1ebbfbe0e05e651c163031b4dab10062448`，测量还包括本轮未提交的基准工具，实际版本以原始`provenance.json`的工作区源码摘要及快照为准。两侧采用processor，当前线程配置保持默认。

| 200图耗时 | 历史逐图 | 当前批量 |
| --- | ---: | ---: |
| 特征五轮（秒） | 86.374 / 90.679 / 78.617 / 79.183 / 79.512 | 58.353 / 57.247 / 57.346 / 59.799 / 58.458 |
| 特征中位 / P95（秒） | 79.512 / 89.818 | 58.353 / 59.531 |
| 整理五轮（秒） | 813.293 / 389.579 / 362.301 / 323.674 / 315.500 | 151.314 / 152.752 / 150.937 / 151.395 / 152.175 |
| 整理中位 / P95（秒） | 362.301 / 728.550 | 151.395 / 152.637 |
| 特征最大RSS（GiB） | 2.188 | 2.449 |
| 整理最大RSS（GiB） | 3.062 | 3.324 |

特征中位提速**1.363×**，未达1.5×门槛；整理中位观察减少**58.21%**，RSS分别增加11.93%与8.57%，数值检查10/12通过。另一个失败项为历史/当前决定严格等价：19条保留记录的`recognized_actors`与`organized_name`均改变，保留照片集合、输入/保留数与场景数相同。两模式内部各五轮决定及质量分数一致；不将显示差异改记为等价通过，也不以分数一致证明模型质量。

两侧图像/文本最低余弦0.9999998808/1.0，十轮均保留154图、0处理/人脸错误，当前热缓存及分析→整理均0前向。交替worker每次只测一轮，原工具的worker内`repeat_min_cosine`不能证明跨进程重复一致性；本轮另用独立审计直接比较各worker保存的向量与同模式首轮向量，结果见下方完整性核查及`metrics.json`，原始工具报告保持不变。

连续2210个负载采样观察到外部GPU进程6058/16354，CPU一分钟负载2.240–10.573。历史整理315.500–813.293秒波动全部保留，P95仅按五个观察值插值计算，不代表稳定尾延迟或受控收益。本协议及正式U2均为`not_passed`。

## U3：全量导出成本

| 操作中位耗时 | 即时快照 | 延后快照 |
| --- | ---: | ---: |
| 日常单条决定＋摘要 | 195.626 ms | 1.828 ms |
| 显式导出 | 69.236 ms | 281.238 ms |
| 独立导出试验的决定＋导出 | 287.241 ms | 306.034 ms |

两组“决定”载荷不同：日常试验修改5条；独立导出试验逐次修改同一条并立即导出，不把两组中位数相加。延后模式每次导出实际写回约4.28MB队列及修订历史，11项检查通过；合成载荷不能代表真实万人复核、人工用时或受控整链路提速。

## HTTP：真实持久任务五轮成本

200图×5次使用同一已预热服务。服务进程导入、模型加载与预热为19.353秒，单列于主计时之外；输入副本准备、完整生产启动和人工复核用时未计入。每轮4个项目，每项目seed复核队列、修改/拒绝各一条并全量导出，再执行持久整理任务。

| 轮次 | 总耗时（秒） | 分析（秒） | 整理（秒） | 脚本复核（毫秒） | 导出（毫秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1005.244 | 339.069 | 665.931 | 211.784 | 22.815 |
| 2 | 1220.323 | 412.471 | 807.611 | 209.716 | 22.069 |
| 3 | 1167.331 | 398.466 | 768.638 | 194.160 | 22.106 |
| 4 | 1132.266 | 376.981 | 755.013 | 240.772 | 22.385 |
| 5 | 1305.858 | 446.394 | 859.123 | 308.787 | 23.299 |
| 中位 | **1167.331** | **398.466** | **768.638** | **211.784** | **22.385** |

各阶段中位分别计算，不能相加代替总耗时中位；总耗时还包含少量阶段间HTTP开销。每轮分析均200图像/28文本前向，整理图文均0前向。分析仍逐图图像编码，内容缓存复用有效；本轮没有改动这条业务路径。

每轮保留/复制145图、明确跳过55图、0失败，五轮输出决定相同。20个项目×8项功能检查和5项汇总检查，共165项通过；汇总确认原图/副本、正式图库/图库副本与全部被测源码未变。服务RSS采样最大约3.34GiB，整卡显存最大约16.83GiB；整卡数值包含外部GPU进程，不能当成模型独占显存。HTTP负载采样数量、外部进程及CPU范围见`metrics.json`。五轮实验及隔离服务已退出。

U2与HTTP清单只有147张内容相同的照片，原文件总字节分别为2,027,706,451与2,325,859,804，且目录划分和业务步骤不同。不给出两套流程间的提速倍数。下一轮可优先研究持久分析的批量预取，同时保留逐项取消/恢复、证据顺序、版本绑定和缓存发布校验。

## 完整性与验收边界

本轮4项协议测试及13项复核存储回归已通过；业务源码未修改，不重跑此前93项业务回归。独立审计**2956项全部通过**，核查测量源码/快照、整树导入来源、权重、原图/图库、真正跨进程向量重复一致性、各轮热缓存，以及HTTP全部40个任务和输出/`best`副本字节。两模式各五轮跨进程图像/文本最低余弦均为1.0，20次分析均0失败。审计结果记录在`metrics.json`的`integrity_audit`，历史19条显示差异单列诊断，不当作历史/当前决定等价或质量通过。

GPU/CPU/存储共享，本轮未提供受控生产性能对照；景别、行当和远端LLM关闭，完整生产启动未运行。既有照片标签仅供隔离脚本探针，未新增GT、独立照片或双人复核结果。`fdc1fda`整树corrected五轮已经补齐，`391ef98`真实raw五轮、独立final test/质量、远端模型权重核验及受控生产成本仍缺，正式U2/U3/U8保持`not_passed`。

## 原始证据与复现

- [历史整树完整对照](../../../outputs/acceptance/core_upgrade/u2_corrected_tree_200x5_20261028/summary.json)
- [万条记录全量导出](../../../outputs/acceptance/core_upgrade/u3_export_10000x5_20261028/summary.json)
- [200图HTTP五轮](../../../outputs/acceptance/core_upgrade/u3_http_workflow_200x5_20261028/summary.json)
- [独立审计](../../../outputs/maintenance/performance_costs_20261028/completed_audit.json)
- [历史决定差异](../../../outputs/maintenance/performance_costs_20261028/historical_decision_differences.json)
- [12图整树smoke](../../../outputs/acceptance/core_upgrade/u2_cost_smoke_host_20261028/summary.json)
- [测量与维护脚本](../../../outputs/maintenance/performance_costs_20261028/)

```bash
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/benchmark_feature_pipeline.py \
  --output outputs/acceptance/core_upgrade/NEW_EMPTY_DIRECTORY \
  --baseline fdc1fda --baseline-scope tree --schedule interleaved \
  --limit 200 --repeats 5 --torch-threads 4 --decode-workers 2 --quality-workers 1 --hash-workers 1

python tools/benchmark_review_snapshots.py --output outputs/acceptance/core_upgrade/NEW_EXPORT_DIRECTORY

env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/benchmark_workflow_costs.py \
  --input-manifest outputs/acceptance/core_upgrade/u8_raw_workflow_200_20261023/input_manifest.json \
  --output outputs/acceptance/core_upgrade/NEW_HTTP_DIRECTORY --limit 200 --repeats 5
```

所有命令要求新的空输出目录。源码、测试和精选汇总纳入Git；原图、图库、权重、归档源码快照、任务数据库、向量、运行日志及完整产物留在忽略的`outputs/`。上方原始证据链接仅在保有这些本地输出的工作区可用；本轮未提交或推送。
