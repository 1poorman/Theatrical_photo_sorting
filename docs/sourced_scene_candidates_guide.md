# 有出处的分幕候选与U4复现

2026-10-22已完成国家大剧院三部剧目的来源v1比较及短原文块对照：原25候选补12正文，28候选版本增加《马可·波罗》的第三、四幕及尾声。短块与长文、25/28候选由同一后端保存的向量和分数表比较，生产默认仍为SigLIP2。

## 短原文块与固定向量结果

新研究保留全部260张中性PNG、query/GT与候选次序，将同一v1已选引用和5段本地正文分为82块，最长64 tokens（包括剧名、幕次和特殊token）。28个长文加82块去重为102份文档；每个后端每份文档以batch1编码一次，260图各编码一次。均值为预声明主方案，最大值为次方案，两种都报告。

| 后端 / 池 | 本协议长文R@1 / R@3 | 分块均值R@1 / R@3（主） | 分块最大值R@1 / R@3（次） |
| --- | ---: | ---: | ---: |
| SigLIP2 · 25 | 14.62% / 61.54% | 23.46% / 70.00% | 23.08% / 68.46% |
| SigLIP2 · 28 | 14.23% / 49.62% | 23.08% / 53.46% | 21.54% / 58.08% |
| Gemma · 25 | 30.38% / 72.69% | 27.69% / 76.54% | 38.85% / 76.92% |
| Gemma · 28 | 23.08% / 67.31% | 21.54% / 71.92% | 31.92% / 70.00% |

两次GPU运行退出0，各12项检查通过；10项分块测试、102项保存数组分析审计通过，全部12变体分数/排名/指标精确复现。固定25池均值使SigLIP2 Top-1/Top-3净增23/22张，Gemma净减7张Top-1、净增10张Top-3。《样式雷》Gemma均值R@3相对自身长文下降8.16个百分点，结果不支持跨剧目一致改进。

移除新候选后，原25项分数与排名逐项完全相同。均值扩池仅因马可竞争丢失SigLIP2 1/43张、Gemma 16/12张Top-1/Top-3命中。原马可3候选的R@3必为100%；新增3候选没有对应正例，当前28池是扩展实验池，尚非经演出版核验的完整官方库存。

最大值随块数提供更多高分机会；均值等权包含短尾块。不能凭最大值最高指标更换默认。Gemma本协议长文25池79/189、28池60/175命中，与历史v1的77/188、60/177有微差；权重/预处理签名一致，文档batch1和共同去重协议不同，没有独立证明微差的唯一成因。

完整逐剧目/幕次、配对变化、历史长文控制及探索性四剧目区间见[短块报告](../outputs/acceptance/core_upgrade/u4_text_blocks_20261022/report.md)和[summary.json](../outputs/acceptance/core_upgrade/u4_text_blocks_20261022/summary.json)。当前开发集已反复查看、final test未锁定，未达U4质量门槛；分块仅用于离线研究，没有接入业务流程。

## 来源v1结果与范围

同一批260张中性PNG、4剧目，按已知剧目分池。3张无幕次标签图片沿用排除项，final test尚未锁定。

| 模型 | 原25候选R@1 / R@3 | 增补正文25候选R@1 / R@3 | 扩展28候选R@1 / R@3 |
| --- | ---: | ---: | ---: |
| SigLIP2 base | 18.85% / 55.00% | 14.62% / 61.54% | 14.23% / 49.62% |
| EmbeddingGemma 2 | 17.31% / 58.08% | 29.62% / 72.31% | 23.08% / 68.08% |

Gemma原25候选引用本日较早的报告：输入、模型/预处理签名和7份非评测入口源码一致，入口随后增加来源校验。五份新报告使用相同9份源码。各新报告7项完整性检查通过，比较审计45项通过。详细逐剧目、逐幕次、配对新增/丢失及哈希见[比较报告](../outputs/acceptance/core_upgrade/u4_sourced_comparison_20261022/report.md)与[summary.json](../outputs/acceptance/core_upgrade/u4_sourced_comparison_20261022/summary.json)。

增补正文提升Gemma开发集结果，但《天下归心》R@3相对SigLIP2仍下降14.71个百分点，扩展池总体R@3也低于U4的90%目标。当前不能据此切换默认模型或称为正式U4通过。

## 来源与缺项

| 剧目 | 明确映射的正文 | 主要来源 | 版本范围 |
| --- | --- | --- | --- |
| 《假面舞会》 | 第1幕1/2场、第2幕、第3幕1/2/3场 | [国家大剧院2024演出页](https://www.chncpa.org/ycxm/202401/t20240105_259790.html) | 页面说明沿用2012瑞典版；query文件名提示2017 |
| 《样式雷》 | 第1、6、8场 | [2015初排报道](https://www.chncpa.org/zxdt_331/zxdtlm/yczx_332/201507/t20150706_130267.shtml)、[2016复排报道](https://www.chncpa.org/zxdt_331/zxdtlm/yczx_332/201602/t20160229_131722.shtml) | 仅取剧情句，演员阵容不同；query提示2016 |
| 《马可·波罗》 | 序、第一至四幕、尾声 | [北京日报报道，中国侨网转载](https://www.chinaqw.com/zhwh/2015/06-01/51505.shtml) | 2015米兰巡演从两小时以上压缩至90分钟；与query提示2014的版本等同性未核验 |

[央视2012节目页](https://tv.cctv.com/2012/08/29/VIDE8gdhjsIhVAsLh1vixa1N120829.shtml)核对《马可·波罗》编号/标题，[国家大剧院2015巡演页](https://www.chncpa.org/zxdt_331/zxdtlm/yczx_332/201505/t20150525_129249.shtml)交叉参考故事。两页没有独立映射成文档正文。日期提示来自文件名，尚非已核验演出边界。

《假面舞会》序曲、《样式雷》第2/3/4/5/7场、尾声与谢幕共8项仍为标签。《样式雷》报道中的“接着”不足以确认第二场，不按顺序推测正文。25候选含5个《天下归心》已有本地正文、12个新增正文和8个仅标签项；来源缺失不等于标签错误。

完整原文、字符跨度、快照摘要及遗漏列表见[来源报告](../outputs/acceptance/core_upgrade/u4_ncpa_sources_20261022/source_report.md)和[source_audit.json](../outputs/acceptance/core_upgrade/u4_ncpa_sources_20261022/source_audit.json)。

后续[演出版审计](../outputs/acceptance/core_upgrade/u4_text_blocks_20261022/production_chronology_report.md)保存16份来源、21段精确原文跨度、6份搜索发现记录，10项检查通过。新资料只作交叉参考，没有加入冻结模型文本：

- [2017《假面舞会》探班报道](https://www.chncpa.org/zxdt_331/zxdtlm/mtsj_334/201705/t20170522_170922.shtml)明确2012制作、瑞典斯德哥尔摩版、2017年5月24–28日第三轮；与2024来源同制作谱系，仍未核验20170522彩排照片及逐场等同性。
- [2014《马可·波罗》升级说明](https://www.chncpa.org/zxdt_331/zxdtlm/yczx_332/201408/t20140825_109288.shtml)正文日期2014/08/26，[探班页](https://www.chncpa.org/zxdt_331/zxdtlm/yczx_332/201409/t20140924_112833.shtml)正文日期2014/09/25；升级版调整结尾、以投影替代实景，并明确第三幕战争场面。2015巡演又压缩，不据谱系推定完整版本等同。
- [2016《样式雷》复排说明](https://www.chncpa.org/zxdt_331/zxdtlm/yczx_332/201602/t20160202_131636.shtml)确认换阵容/打磨；[戏剧季专题](https://www.chncpa.org/subsite/gjxjj2016/)列出6.10–6.19档期。20160609彩排日期尚未核验；[6月专题](https://www.chncpa.org/subsite/ysl201606/index.html)家族代际编号不作为场次。

正文日期与URL/搜索摘要日期分开记录。8项缺失正文继续保留，不从整剧梗概、演员阵容或相邻段落猜场次。

## 输入契约

[来源助手](../tools/sourced_scene_candidates.py)从保存的HTML确定性提取文本，排除script/style等、注释和图片属性。构建规则另外排除可见图注、探班演员名单；候选正文不读取query图注或人物标注。

候选包包含`schema_version`、`version`、基础清单SHA256、`pool_mode`、`sources`、`replacements`与`additions`。每个来源保存URL、类型、快照相对路径、编码和完整SHA256；每个候选保存剧目/幕次ID、原标签、引用与映射。

引用的`start`为0起始字符位置，`end`不含末尾字符，基于提取文本而非HTML字节。引用必须精确匹配快照；映射必须有明确编号、标题，正文引用必须位于对应的映射上下文内。中文/阿拉伯数字及“景/场”仅按编号规范化，《假面舞会》的景号与场号保持对应。

`same_pool`保持所有原ID、标签、顺序、query和GT，只改明确对应的文档。`expanded_pool`保留全部原文档后追加有明确来源的幕次；新增ID由`stable_id('scene', play_id + ':' + label)`生成。当前3个新增幕次没有query GT，只作为干扰候选，不代表它们的召回质量已评估。

[评测入口](../tools/evaluate_verified_scene_retrieval.py)支持`--candidate-texts`，需要同时提供`--prepared-inputs`。直接复用包含来源元数据的准备清单时，会从原清单与候选包重新构建并逐字段比较；引用来源、包或输入改写，包括等大小/等mtime改写，均拒绝混用。

## 冻结文件与复现

以下路径相对`outputs/acceptance/core_upgrade/`。

| 产物 | 用途 |
| --- | --- |
| `u4_ncpa_sources_20261022/raw/`、`extracted_text/` | HTML/响应头及确定性提取文本；搜索快照只作发现证据 |
| `u4_ncpa_sources_20261022/source_manifest.json` | 初始三条专题抓取及SHA256；沙箱DNS失败另存记录 |
| `u4_ncpa_sources_20261022/candidate_texts_v1.json` | 固定25候选，12个来源正文 |
| `u4_ncpa_sources_20261022/candidate_texts_expanded_v1.json` | 保留上述25项并追加3项 |
| `u4_ncpa_sources_20261022/prepared_{samepool,expanded}_v1/input_manifest.json` | 两模型共同使用的冻结准备输入 |
| `u4_sourced_comparison_20261022/run_state.json` | 五次执行的命令、PID、退出码、报告摘要与完成节点 |
| `u4_sourced_comparison_20261022/comparison_analysis.py` | 只读审计、配对分析与报告生成 |
| `u4_sourced_comparison_20261022/text_window_audit.py`、`text_window_audit.json` | CPU处理器复现及token长度证据，不加载模型 |
| `u4_sourced_{siglip2_baseline,siglip2_samepool,gemma2_samepool,siglip2_expanded,gemma2_expanded}_20261022/` | 五份模型报告；stdout日志位于父目录同名`.stdout.log` |
| `u4_text_blocks_20261022/prepared/` | 冻结短块协议、82块/精确来源跨度、输入/源码/tokenizer摘要 |
| `u4_text_blocks_20261022/{siglip2_fixed_vectors,gemma2_fixed_vectors}/` | 图文向量、文档分数、候选表、文档索引、输入、六变体/后端的报告与运行JSON |
| `u4_text_blocks_20261022/` | 两次命令/PID/退出码、102项分析报告/复现、逐场CSV、演出版审计/原文、测试与日志 |

只读复核候选来源、原图与PNG摘要，同时重新生成本目录分析产物：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python outputs/acceptance/core_upgrade/u4_sourced_comparison_20261022/comparison_analysis.py
```

需要新实验时使用新的空输出目录。例：复用固定25候选版本运行SigLIP2，输出名仅作新运行示例：

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/theatre-core-mpl \
  /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python \
  tools/evaluate_verified_scene_retrieval.py \
  --labels outputs/acceptance/core_upgrade/verified_labels_roles_20261022/verified_photo_labels.json \
  --registry outputs/acceptance/core_upgrade/verified_labels_roles_20261022/identity_registry.json \
  --source-root data/ncpa_test \
  --prepared-inputs outputs/acceptance/core_upgrade/u4_ncpa_sources_20261022/prepared_samepool_v1/input_manifest.json \
  --backend siglip2_base --device cuda:0 \
  --output outputs/acceptance/core_upgrade/u4_sourced_siglip2_new_run
```

Gemma使用`--backend embeddinggemma2`及`PYTHONPATH=outputs/core_upgrade_deps`，本地权重为`/models/hch/embeddinggemma-2`。GPU1为本轮物理编号，后续按现场设备确认；本轮五次顺序运行均已结束，无需重启或等待旧会话38976。

10项来源契约测试已通过，日志为`u4_sourced_comparison_20261022/source_tests.log`。助手变化后可运行：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_sourced_scene_candidates.py -v
```

不要直接重跑当前来源目录的`build_candidate_version.py`覆盖冻结版本。修改原文选择、场次映射、短文块或候选池时，先创建新来源/候选目录与新版本，固定规则后再测，保留v1报告。

短块实现为[scene_text_blocks.py](../tools/scene_text_blocks.py)、[准备入口](../tools/prepare_scene_text_block_study.py)及[评测入口](../tools/evaluate_scene_text_block_study.py)。精确引用跨度基于官网提取文本；5段旧本地正文跨度基于冻结场次文本，准备入口另绑定原始文件及解析器SHA256。规则为最长适配前缀，优先句末、其次分句、最后Unicode字符边界，保留全部已选原文，候选输入不读取query图注/人物列表。

本轮协议SHA256：`88cb1f17fc0ad7fbc2e9c9d402a5939fbbd729c6d200672f70d904d9c8dfc218`，绑定11份测量源码。修改绑定文件后不能继续旧协议；准备新目录。中断时先核对运行JSON/PID，完成项不重跑，非空未完成评测目录保留并另建输出。`run_models.py`使用进程锁防止重复启动，stdout日志和每个后端退出码持久保存。

只读重建分析及演出版审计（不加载模型、不改冻结输入）：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python outputs/acceptance/core_upgrade/u4_text_blocks_20261022/analyze_saved_vectors.py --verify-existing
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python outputs/acceptance/core_upgrade/u4_text_blocks_20261022/audit_production_chronology.py --verify-existing
```

新的GPU实验使用新的空目录，并通过准备入口建立匹配源码协议；两模型具体命令见[launch_commands.txt](../outputs/acceptance/core_upgrade/u4_text_blocks_20261022/launch_commands.txt)。本轮会话22447已结束退出0，无需等待旧进程。分块契约变化后按影响范围运行`tests/test_scene_text_blocks.py`；现有10项已通过，不重复无关U1回归。

## 已观察的限制与接续

来源v1中SigLIP2实际将12个新增文档从73–190 tokens截断为64 tokens，扩展版15个来源文档也全部超窗。新短块协议另存版本并覆盖已选原文；旧v1报告与长文控制保持可复现。

Gemma扩池重编码使原候选分数最大改变0.001310，未扩池剧目的2张Top-1、1张Top-3新增命中来自重编码。分析将28候选的已存分数限制回原25项，得到79/189命中，再加入3项竞争后为60/177；竞争丢失19/12张，全部发生于《马可·波罗》。这是固定已存分数的诊断，不是新模型实测；分批与BF16可能影响近分候选，原因未作独立消融。

短原文块、固定编码及扩池对照已完成。下一步核对独立演出/近重复/连拍划分，补剩余正文与新增候选正例；框级身份及锁定测试集仍缺标注。已查看图片不能直接改名为final test。共享GPU、单遍及不同模型加载边界不作速度或U2结论，U2继续暂停。

中断后先读[WORK_STATE.md](../WORK_STATE.md)与运行状态。`outputs/`被Git忽略；跨机器交接需携带来源目录、模型报告、标签/角色表与基础清单所引用的`u4_verified_siglip2_gpu_20261022/neutral_images/`。绝对路径需在新版本中明确迁移，不能绕过哈希校验继续旧实验。
