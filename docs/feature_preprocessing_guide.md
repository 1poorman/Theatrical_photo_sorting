# 图像预处理与 U2 性能复验

## 2026-10-28 并行完整哈希与增量对照

`TRANS_FEATURE_HASH_WORKERS`控制特征查询阶段的完整SHA-256读取线程，默认1。
可设为4以并行读取多个原图；每线程仅持有1 MiB读取块，结果保留摘要与版本信息。
写缓存/返回向量前仍按输入顺序逐图重读完整内容，避免提前完成的并行摘要扩大
原图改写窗口；时间戳不变也不能替代内容相等证明。向量缓存签名和格式保持兼容。

独立基准新增基线和候选的解码/质量线程参数，以及`--baseline-image-backend`。
基线线程参数在独立子进程导入配置前设置；报告记录两方案实际线程数。
因此可比较上一轮已提交的Pillow/双线程优化，而不混用旧processor数字。

```bash
CUDA_VISIBLE_DEVICES=1 /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/benchmark_feature_pipeline.py \
  --output outputs/acceptance/core_upgrade/u2_parallel_new_run \
  --baseline 56baf19 --baseline-mode batch \
  --baseline-image-backend pillow --image-backend pillow \
  --baseline-decode-workers 2 --baseline-quality-workers 2 \
  --decode-workers 4 --quality-workers 4 --hash-workers 4 \
  --limit 200 --repeats 5 --torch-threads 16
```

12图检查中图文向量、归档命名和质量分数一致，热缓存/分析→整理均0前向；
特征3.257→2.496秒、整理16.441→13.347秒，RSS分别增加9.75%/8.15%。
完整200图×5次报告已在`u2_parallel_200x5_20261028/`完成，独立进程退出0；
[Git可见汇总](reports/u2_parallel_20261028/README.md)保留全部五轮、阶段指标及来源摘要。
特征43.270→28.341秒（1.527×），整理169.927→111.627秒（观察减少34.31%）；
RSS分别增加13.81%/18.83%，12项检查全部通过，本增量协议为`passed`。
十轮保留154图、评分/命名一致、0错误；图文向量逐字节相同，热缓存与分析→整理0前向。
670项完整性核查通过，125份本地文件及74份旧摘要保留。GPU整卡占用在两方案
测量前分别2697/12739 MiB，外部负载未隔离；不能据此认定受控收益或尾延迟改善。
本协议只测同步整理；持久任务的逐项选优仍顺序执行。默认processor、
2解码/1质量/1哈希线程保持，DALI不采用，独立质量缺项与正式U2门槛保留。

33项缓存回归覆盖并行读取顺序/坏图/重复字节/热缓存、相同元数据下的原图改写及
前一个缓存写入期间后图变化；28核心、18整理和4质量并行测试通过。
目录与失败/最终回归记录见`outputs/maintenance/layout_20261028/`。

## 既有预处理选项

场景图像特征默认使用模型自带的 `processor`。可通过环境变量
`TRANS_FEATURE_IMAGE_BACKEND=pillow` 或构造参数
`ImageEmbedder('siglip2_base', image_backend='pillow')` 试用 Pillow 缩放。
此选项仅用于固定宽高的 `SiglipImageProcessor`；其他模型和尺寸规则沿用原处理器。

Pillow 路径先执行 EXIF 方向校正和 RGB 转换，再按处理器的目标尺寸与插值方法缩放，
最后由原处理器执行 rescale/normalize。它避免先把原始大图转换成 Torch 张量。
两条路径有独立缓存签名，切换后重新计算；`processor` 的原缓存键保留。
Pillow 签名从 `resize-v1` 升至 `resize-v2`，防止复用此前尺寸检查失效时保存的结果。

共享特征服务一次向支持预取的编码器传入最多两批路径，模型实际前向仍遵守
`TRANS_FEATURE_BATCH_SIZE`（默认 8）；解码线程由
`TRANS_FEATURE_DECODE_WORKERS`（默认 2）控制。坏图、批量失败及输入在编码期间变化
继续按单图处理，不会把其他图片的向量写入错误路径。

## 2026-10-26 修复与验证

此前 200 图 × 5 次实验使用 `4194277` 的批量实现作基线，观察到特征吞吐 1.101×、
整理耗时增加 0.84%。加载模型后的处理器尺寸是 `SizeDict`，其迭代器输出键值对；
旧代码用 `set(size)` 判断键名，导致 Pillow 路径静默回退。这份历史实验无法证明
Pillow 缩放的收益，原报告保留在 `u2_pillow_200x5_20261025/`。

修复将尺寸统一转为字典。新增测试先复现此问题，再检查实际传给处理器的是目标尺寸
且关闭其内部 resize；同时验证其他尺寸规则仍交给原处理器。28 项缓存测试、28 项核心
测试通过。12 图真实 GPU 检查的最低图像特征余弦为 0.99999964，归档决定及质量分数
一致，热缓存和分析→整理缓存的图文前向均为 0。小样本不作为正式性能通过。

诊断、失败复现、测试日志位于 `outputs/acceptance/core_upgrade/u2_resume_20261026/`；
小样本报告位于 `outputs/acceptance/core_upgrade/u2_pillow_fixed_smoke_20261026/`。

修复后的200图×5次完整结果位于 `u2_pillow_fixed_200x5_20261026/`：

| 指标 | 4194277批量实现 | Pillow修复后 |
| --- | ---: | ---: |
| 特征中位耗时 | 64.078秒 | 42.277秒 |
| 整理中位耗时 | 158.099秒 | 142.883秒 |
| 特征峰值RSS | 2,580,369,408 bytes | 2,091,687,936 bytes |
| 整理峰值RSS | 3,506,122,752 bytes | 3,240,427,520 bytes |

特征吞吐为1.516×，整理耗时减少9.624%；12项检查中仅整理耗时降低20%的门槛未通过，
状态仍为`not_passed`。200图最低余弦为0.99999893，全部重跑向量相同；10次整理均保留154图，
归档决定和质量分数一致，处理错误为0，热缓存及分析→整理缓存的图文前向均为0。
两方案先后在共享GPU测量；参考整理首轮264.1秒，其余154.9–162.8秒，
不能将该环境的观察值当作隔离机器结论，也不代表独立质量标签已经通过。

## 复现

在可访问 GPU 的项目环境中运行；指定新的空输出目录：

```bash
CUDA_VISIBLE_DEVICES=1 /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/benchmark_feature_pipeline.py \
  --output outputs/acceptance/core_upgrade/u2_pillow_new_run \
  --baseline 4194277 --baseline-mode batch --image-backend pillow \
  --limit 200 --repeats 5 --torch-threads 16
```

该命令比较冻结的已有批量实现与新预处理/预取实现。`--baseline-mode sequential`
和 `--baseline fdc1fda` 用于原 corrected 顺序基线；两种对照不得混用。
报告保存代码与输入摘要、实际前向、每轮余弦、内存、归档决定及误差。
共享 GPU 的结果需注明其他负载，不能据此宣称隔离机器性能。

DALI 已独立安装于本地 `outputs/u2_dali_deps/`，不属于普通安装依赖。
独立实验必须验证 EXIF、颜色、缩放、归一化和坏图行为；其接入门槛为在优化后批量路径上
再降低至少 15% 的整理耗时。正式质量仍需要独立人工标签。

实验入口是 `tools/evaluate_dali_preprocessing.py`，仅接受同源码、同模型且已经完成的
Pillow 200 图×5 次报告。它先验证13个大小/方向/颜色样例及坏图隔离，再核对全部200张的
实际特征；不满足最低余弦/99.9%有效样本门槛时结束评价。通过后，另开进程复用基准工具
的完整5次特征/整理/热缓存协议，避免预检内存混入测量。不需放大的RGB JPEG且无EXIF旋转使用DALI；
小图放大、其他图片和GPU解码失败逐图回退Pillow，报告记录实际DALI执行与回退数量。

初版DALI在200图上全部达到余弦门槛（最低0.99946177），但小图放大样例仅0.99746233，
原始失败记录保存在 `u2_dali_200_20261026/`，未据此进行性能声明。
后续变体保留这个样例并要求它回退Pillow，新增缩小样例验证真实DALI执行；2项CPU回退测试通过。

最终变体在 `u2_dali_no_upscale_200_20261026/` 完成13样例、坏图、200图和5次完整对照。
全部图像特征达到余弦门槛，缓存复用及处理错误检查通过，但最终状态为 **`not_adopted`**：

| 相对优化后的Pillow路径 | DALI结果 |
| --- | ---: |
| 特征中位耗时 | 34.853秒（减少17.56%） |
| 整理中位耗时 | 138.162秒（仅减少3.304%，低于15%） |
| 特征 / 整理峰值RSS增幅 | 49.00% / 28.00% |
| 归档决定变化 | 9张场景后缀改变，同一目录分段数15→14 |

五轮DALI均复现上述分段变化；保留154张及质量评分相同不能作为输出完全一致的证据。
因此DALI仍只保留研究工具。其源代码快照、首次失败、最终指标及场景差异均已保存，
详见 `u2_resume_20261026/report.md`、`final_audit.json` 和 `dali_decision_differences.json`。

```bash
# 可选依赖只安装到独立目录。
python -m pip install --target outputs/u2_dali_deps -r requirements/dali.txt
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=outputs/u2_dali_deps \
  /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/evaluate_dali_preprocessing.py \
  --reference-run outputs/acceptance/core_upgrade/u2_pillow_fixed_200x5_20261026 \
  --output outputs/acceptance/core_upgrade/u2_dali_new_run
```

此入口不会启用应用中的DALI后端。`evaluation.json`记录实际结论与缺项，
`run_state.json`记录子进程与退出码；旧非空目录拒绝覆盖。

## 全尺寸质量评分的并行选项

`TRANS_QUALITY_DECODE_WORKERS` 控制同步 `BurstDeduplicator.dedup_folder` 的
CPU 质量评分线程数，默认 `1`，设为 `2` 可试用双线程。线程跨目录复用；
每个线程只持有当前正在处理的全尺寸图片，完成后只留下小型评分记录。
评分公式、原图尺寸、分组规则及同分时的输入顺序保持一致。
启用景别分类器时使用顺序路径，避免并发调用分类器的 GPU 会话或可变状态。
持久 `OrganizeTask` 的逐项检查点仍使用原顺序路径，不能将同步整理的速度
直接作为 HTTP 后台任务的收益。

批量图像推理失败时，编码器逐张复用已经准备好的张量，不再重复原图解码和预处理；
失败图片仍独立处理，缓存仍在发布前重新核验完整原图内容。

2026-10-26 的新增回归覆盖原图评分、坏图、同分排序、线程数上限、分类器顺序及
批量失败回退。32 张原图各三次的 CPU 诊断中，顺序/双线程评分中位耗时为
9.183/4.891 秒，评分记录完全一致。这仅是选优环节的测量。
12 图真实 GPU 检查的特征最低余弦为 0.99999964，分数/归档一致，
热缓存及分析→整理缓存前向为零；小样本整理耗时增加 7.71%，不能据此宣称提速。

完整新对照使用新的输出目录，并自动保存测量源码快照、分阶段耗时及实际质量线程数：

```bash
CUDA_VISIBLE_DEVICES=1 TRANS_QUALITY_DECODE_WORKERS=2 \
  /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/benchmark_feature_pipeline.py \
  --output outputs/acceptance/core_upgrade/u2_pillow_quality_new_run \
  --baseline 4194277 --baseline-mode batch --image-backend pillow \
  --limit 200 --repeats 5 --torch-threads 16
```

诊断、失败复现及回归日志在 `u2_cpu_quality_20261026/`；
小样本宿主报告在 `u2_pillow_quality_smoke_host_20261026/`，
沙箱 CUDA 不可用的失败保留在 `u2_pillow_quality_smoke_20261026/`。
200 图五次完整结果以 `u2_pillow_quality_200x5_20261026/summary.json` 为准；
报告已完成且进程退出0，最终538项只读核查通过。默认处理器和单线程质量评分保持。

| 指标 | 4194277批量路径 | Pillow与双线程选优 |
| --- | ---: | ---: |
| 特征中位耗时 | 59.088秒 | 42.655秒 |
| 整理中位耗时 | 167.209秒 | 122.834秒 |
| 选优环节中位耗时 | 62.587秒 | 35.341秒 |
| 特征峰值RSS | 2,636,566,528 bytes | 2,142,519,296 bytes |
| 整理峰值RSS | 3,627,003,904 bytes | 3,552,137,216 bytes |

本次观察到整理总耗时减少26.539%，选优环节减少43.533%；
特征吞吐1.385×仍低于1.5×门槛，12项检查仅`feature_speedup`未过，
本协议性能状态仍为`not_passed`。特征/整理RSS下降18.738%/2.064%，
图像最低余弦0.99999893，十轮保留154图、质量分及归档命名全部一致，
处理/人脸错误0，热缓存和分析→整理缓存实际图文前向均0。

两方案顺序运行于共享GPU；候选整理范围121.004–181.181秒，前两轮较慢，
P95为172.316秒，基线P95为171.675秒。保留全部五轮取中位数，
不能将26.539%作为受控机器的可靠收益或尾延迟改善，
也不证明单独双线程相对于冻结Pillow路径的端到端增益。
本轮81项回归通过，原图/正式图库与47项旧产物摘要不变；
完整说明见`u2_cpu_quality_20261026/report.md`及`final_audit.json`。
原corrected基线、HTTP后台任务成本和独立质量仍待验收，正式U2保持`not_passed`。
