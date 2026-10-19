# 核心升级执行进度

> 更新：2026-10-19。基线提交：`391ef98`；本轮实现仍在工作区，尚未提交。  
> 验收定义见 [核心升级蓝图](core_upgrade_blueprint.md)。U0–U8 独立记录，不改变 M11–M16 的历史状态。

## 1. 当前状态

已完成 C1–C4 的首轮修复及扩展回归，生成全目录清单和修复前后复现报告，接通本地 EmbeddingGemma 2 与现有 Qwen 视觉端点。工作台、旧版入口和既有多模态回归通过。**完整 U0–U8 尚未完成，正式模型质量和整链路性能尚未验收。**

| 阶段 | 当前实现与检查 | 尚缺验收 | 默认采用 |
| --- | --- | --- | --- |
| U0 | 清单、去重、保守划分；4 个 raw/corrected 复现各重复 5 次 | GT 核对、final test 锁定、真实 raw 流水线基准 | 不适用 |
| U1 | C1–C4 核心修复；核心回归 28/28、工作台 9/9、多模态 11/11 | 显式身份 ID/剧目角色映射、身份阈值校准 | 正确性修复已启用 |
| U2 | 场景推理批量预填、原子缓存写入、批次失败回退；3 图一致性通过 | 整理全批量化、完整缓存契约、吞吐/RAM 验收 | 批量预填已接入；阶段未通过 |
| U3 | 唯一临时文件、旧记录目录绑定；顺序重跑及目录隔离通过 | 后台任务、检查点、事务/并发、草稿和缩略图缓存 | 未完成 |
| U4 | Gemma 可选后端；同池 34 图/5 幕对比及 5 次编码计时 | 多剧目独立标注、质量门槛；其他候选未运行 | 保留 SigLIP2 base；Gemma 可选 |
| U5 | 图片观察 → 引用校验 → 闭集判断；1 图真实双请求通过 | 100 图双人事实核对、预算、缓存、OCR、Macro-F1 | “视觉模型辅助”选项启用 |
| U6 | 尚未实施，not_run | 变化点、DINOv3、边界 GT | 未采用 |
| U7a/U7b | 尚未进行模型/聚类/选优升级，not_run | 框级身份、连拍与选优 GT；候选对比 | 未采用 |
| U8 | 35 图浏览器流程、新旧入口、5 种屏宽通过 | U0–U3 完整验收、GPU 整链路与最终选择 | 未验收 |

## 2. 本轮实现

- **C1**：排序、分段前按原始路径建立特征映射；缺失、非有限或维度不齐的特征只停用所在段的细分。单图特征异常不再中断其他图片，EXIF 文件句柄正确关闭。
- **C2**：视觉检索覆盖全部幕次，再融合词法先验；单个文本编码失败仍保留词法候选。输出独立 lexical/visual 分数及降级原因，区分缺失特征与低相似度。
- **C3**：逐脸合并同演员多个角色目录的锚定分，取最佳身份并检查第二名差距，多人结果按演员去重。当前仍以目录演员名作为兼容身份键，并保留最佳锚定目录的展示名称；正式 identity_id、当前剧目角色绑定和阈值校准待补。margin 暂沿用现有配置 `0.05`，不声明框级质量达标。
- **C4**：pending 更新候选并保留版本关系；approved/rejected 保持原样，包括缺少 reviewer 的旧记录。裸文件名首次绑定完整路径后写入 `image_bindings.jsonl`，新实例也拒绝其他目录的同名关联。导出使用绑定路径，整理按完整路径应用标签。
- **人工与分数**：人工修改/拒绝不沿用机器分数，原分数存为 `machine_confidence`；人工自定义标签保留。工作台改称“证据分”，注明不是正确概率。
- **批量回退**：修复 `ImageEmbedder.extract_features_batch` 失败时把 tensor 当路径的错误，逐图回退保持向量与路径对应。
- **视觉请求**：实际部署身份来自用户配置，为 Qwen3.8-27B；请求保留 Qwen3.6-27B 别名，trace 记录身份来源。发送经 EXIF 校正的真实预览图，校验观察 schema 与幕次引用。

真实整链路首次检查发现第二次请求耗尽 2048 completion tokens，返回 `content: null`，客户端计算响应摘要时触发 `AttributeError`。现将空正文视为协议失败，并对该部署的结构化视觉/闭集请求默认设置 `chat_template_kwargs.enable_thinking=false`。复验两请求分别输出 264、229 tokens，均正常结束，总耗时 7.215 秒；结果仍为 pending。此结果只证明链路可用，不证明幕次正确。

## 3. 数据与实测

报告位于 [`outputs/acceptance/core_upgrade/first_pass/`](../outputs/acceptance/core_upgrade/first_pass/)，留在本地。原图、正式人脸库和 `.env` 未修改。

### U0 清单和复现

扫描 **582 个文件、319 张唯一解码图像**。按目录规则暂分为 292 张舞台照、27 张资料图，分类待人工核对；解码失败为 0。历史“555 张”不能作为独立样本数。

| 剧目 | 唯一舞台照 | 有文件名幕次弱标签 |
| --- | ---: | ---: |
| 天下归心 | 35 | 34 |
| 假面舞会 | 119 | 118 |
| 马可·波罗 | 59 | 59 |
| 样式雷 | 79 | 49 |

`manifest.jsonl` 记录内容/像素哈希、方向、来源和待核对标签。近重复采用 dHash、宽高比、平均颜色的启发式检查；318 个检测组在 `splits.json` 中跨集合数为 0。整剧目与检测到的重复组同集合，**不保证所有近重复已找出**。划分为 provisional，final test 未锁定；正式幕次、框级身份、连拍和边界核对数均为 0。

`baseline_raw.json` 保存 `391ef98` 的 4 个错误行为，`baseline_corrected.json` 保存修复后的 4/4 行为通过。各重复 5 次且行为稳定；使用合成向量/EXIF/聚类替身，`real_model_quality` 和 `end_to_end_performance` 明确为 `not_run`，不是完整流水线性能基线。

### U4 召回探索

同一剧目、34 张有唯一幕次弱标签的照片、5 幕原文文本；排除 1 张没有唯一剧情幕次标签的照片。中性文件名，清除 EXIF/ICC 元数据；两个后端 manifest/text hash 一致。

| 后端 | Recall@1 | Recall@3 | 5 次编码耗时中位数 | PyTorch 峰值 allocated 显存 |
| --- | ---: | ---: | ---: | ---: |
| corrected SigLIP2 base | 4/34（11.8%） | 23/34（67.6%） | 0.661 秒 | 1,698,467,840 bytes |
| EmbeddingGemma 2 | 10/34（29.4%） | 28/34（82.4%） | 2.737 秒 | 1,586,116,096 bytes |

硬件 RTX 3090，PyTorch 2.9.0+cu130。SigLIP2 使用 Transformers 5.15.0；Gemma 使用独立依赖中的 Transformers 5.20.1、SentenceTransformers 6.1.0。计时为**进程预热、无特征缓存的预览图与文本编码**，不含原图清单生成、人脸、决策、归档或下载；不同依赖和模型预处理均已注明。

Gemma 在探索样本上多召回 5 张，但 Recall@3 仍未到蓝图 90% 目标，耗时增加；单剧目弱标签未经独立核对，质量状态为 **insufficient_data**。继续作为可选后端，不替换已修复的 SigLIP2 默认。不能与历史不同文本/文件名条件的结果直接比较。

`embedding_smoke.json` 另记录 3 张真实图的 768 维 BF16 编码：批量/逐张最低余弦 `0.999929`，新检索器命中缓存后图像/文本编码计数均为 0。这不等于 U2 全量缓存、吞吐门槛通过。

## 4. 复现与配置

在项目根目录执行；模型、浏览器及部分 TestClient 检查需要能访问目标机设备、线程和网络的环境。

```bash
# 清单和四项缺陷对照（原图只读）
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/build_core_upgrade_manifest.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/characterize_core_correctness.py

# 核心 28 项、工作台 9 项、既有多模态 11 个检查脚本
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_core_upgrade.py
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/test_workspace.py -v
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tests/acceptance_multimodal_m16.py

# 两后端依次运行，默认 repeats=5
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/evaluate_core_upgrade.py --backend siglip2_base --device cuda:0
PYTHONPATH=outputs/core_upgrade_deps /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/evaluate_core_upgrade.py --backend embeddinggemma2 --device cuda:0
PYTHONPATH=outputs/core_upgrade_deps /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/check_multimodal_upgrade.py --component embedding --device cuda:0
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/check_multimodal_upgrade.py --component vision

# 先启动验收服务，再在另一终端运行浏览器检查
MPLCONFIGDIR=/tmp/theatre-core-mpl /home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python -m uvicorn server_ui:app --app-dir app --host 127.0.0.1 --port 18301 --lifespan off
WORKSPACE_URL=http://127.0.0.1:18301 node tests/test_workspace_browser.cjs
```

依赖版本见 [`requirements_embeddinggemma.txt`](../requirements_embeddinggemma.txt)。本机已在 `outputs/core_upgrade_deps` 验证；其他机器应在独立环境安装，不能只修改旧 Transformers 环境中的后端名称。

| 配置 | 默认/用法 |
| --- | --- |
| `TRANS_SCENE_EMBEDDING_BACKEND` | 默认 `siglip2_base`；设为 `embeddinggemma2` 并加载对应依赖可启用新后端 |
| `TRANS_EMBEDDINGGEMMA_MODEL_PATH` | `/models/hch/embeddinggemma-2` |
| `TRANS_EMBEDDINGGEMMA_DEVICE` | `cuda:0`；支持时 BF16，否则 FP32，禁止 FP16 |
| `TRANS_VISION_MODEL_IDENTITY` | `Qwen3.8-27B`；请求名称仍取 `.env` 的 `big_model_name` |
| `TRANS_VISION_IMAGE_MAX_SIDE` | `1280` |
| `TRANS_VISION_ENABLE_THINKING` | 默认 `false`；启用后需另验 token 预算和延迟 |

浏览器检查覆盖 35 张真实照片、导入、离线分析、复核、导出、刷新与 1440/1024/768/540/390 宽度。GPU 整理、聚类动作在浏览器测试中拦截，仅验接口交互；不能计为 U8 GPU 整链路通过。

## 5. 下一步

1. 核对场次、资料分类、弱标签和近重复，再锁定独立 final test；现有舞台照数量基本够幕次初评，主要缺口是可信标注。
2. 补齐身份 ID 与剧目角色映射，隔离 gallery/query 并校准 margin；保存真实 raw/corrected 流水线 5 次基准。
3. U2：统一内容特征缓存并接通整理全批量路径。当前缓存仍用路径/mtime/size，签名未全面包含权重内容/预处理，不标记通过。
4. U3：后台任务、逐项检查点、事务/版本冲突、草稿和缩略图。唯一临时文件不能替代跨进程事务，队列/修订日志仍有一致性窗口。
5. 扩展多剧目检索和 U5 预算/事实质量评估，再进入 U6/U7；未达门槛组件保留实验状态。

新增测试已加入 `.gitignore` 例外。每阶段更新命令、代码版本、报告和质量状态，禁止将合成复现、小样本 smoke 或历史结果计为正式效果通过。
