# 蓝图：戏剧照片智能整理系统（Smart Organizer）

> 目标：将原始摄影目录自动转化为符合 NCPA 整理规范的目录，形成
> 「原始照片 → 连拍去重 → 人脸/行当标注 → 场景划分 → 规范命名输出」的整理流水线。
> 验收数据集：`data/ncpa_test/`（4 部剧目，原始 296 张 / 整理完成 263 张）。

## 0. 现状与依据

- 人工整理命名规范（从【整理完成】目录归纳）：
  `{序号:02d}-{日期}-{剧场}-{剧目}彩排/演出 {幕次场景}-{人物列表（演员饰角色）}-{摄影师}.jpg`
- 原始命名：`{日期}{剧场}-{剧目}{事项}-{人物列表}-摄影{摄影师} (连拍序号).jpg`
- 已验证：
  - SCRFD+ArcFace 对戏曲妆造可正常检出主演（小脸龙套不识别，符合预期，不做小脸增强）
  - SigLIP2 纯视觉聚类与幕次语义吻合度低 → 幕次划分以 EXIF 时间分段为主、视觉聚类为辅
  - 文件名正则可稳定提取「演员名-角色名」对 → 可反哺人脸库
  - ES 8.12 可达；原始照片含 EXIF 拍摄时间

## 1. 功能范围

| # | 功能 | 说明 |
|---|------|------|
| F1 | 智能整理流水线 | 原始目录 → 规范命名整理目录（含整理报告） |
| F2 | 按演员整理视图 | 人脸库识别后，按「演员（饰角色）」生成照片专辑 |
| F3 | 连拍去重与选优 | 括号连拍 + embedding 相似兜底分组；全图质量评分；**基于景别保留最佳几张** |
| F5 | 戏曲行当分类 | SigLIP2 zero-shot 对主角裁剪区域做生/旦/净/丑分类（不训练新模型） |
| — | 小脸检测增强 | **本期不做**（龙套不识别可接受，主演当前设置可识别） |

## 2. 模块设计（新增 `core_modules/organize/` 子包）

```
core_modules/organize/
├── __init__.py
├── filename_parser.py   # 文件名/目录名解析：日期、剧场、剧目、摄影师、人物列表、连拍序号
├── burst_dedup.py       # 连拍分组 + 全图质量评分 + 景别感知选优
├── face_db_builder.py   # 半自动人脸库构建（单人照片自动归属 → 标准人脸裁剪 → ES 入库）
├── actor_view.py        # 按演员整理视图生成
├── scene_split.py       # EXIF 时间分段（主）+ SigLIP2 视觉聚类（辅）→ 场景标签
├── role_classifier.py   # SigLIP2 zero-shot 生旦净丑分类
└── smart_organizer.py   # 流水线编排（F1 入口，串联 F2/F3/F5）
```

### 2.1 filename_parser
- `parse_dirname(dirname)` → `{date, venue, play, event, photographer}`
- `parse_filename(fname)` → `{date, venue, play, event, people: [(actor, role)], photographer, burst_seq}`
- 人物提取正则：`([\u4e00-\u9fa5]{2,4})饰([\u4e00-\u9fa5·]{2,8})`；摄影师：`摄影([\u4e00-\u9fa5]{2,3})$`
- 纯逻辑、无模型依赖，全量文件名可离线回归。

### 2.2 burst_dedup（F3）
- 分组键：解析后的基础名（去 `(N)` 后缀、去扩展名）；同基础名为一组。
- 组内评分：全图质量分 = 清晰度（Laplacian 方差，归一化）0.5 + 亮度合理度 0.2 + 对比度 0.2 + 人脸质量加成 0.1（有人脸时）。
- **景别感知保留**：组内先按景别（shot_classify 结果）分桶，每桶按质量分保留 top-K（默认 K=2），
  避免连拍组横跨景别变化时误删；输出 `best/`（保留图）+ `discarded/`（被剪图）+ `dedup_report.json`。

### 2.3 face_db_builder（F2 基础）
- 输入：整理完成目录（或人工确认目录）。
- 规则：文件名仅含 1 位演员（`X饰Y`）→ SCRFD 检测取最大脸 → ArcFace 对齐 160 标准图 →
  存入 `data/face_database/{X}（饰{Y}）/`；每人限锚定 `MAX_ANCHORS_PER_PERSON`（默认 8）。
- 多人同框文件名默认跳过（归属不确定，留给人工）；谢幕大合影跳过（人脸太小）。
- 建库后调用现有 `FaceRecognitionSystem.build_face_database` 写入 ES（库已含《样式雷》，增量为其他剧目）。

### 2.4 actor_view（F2）
- 对目标目录逐图识别（现有 `recognize_face`，known_threshold=0.55）。
- 输出 `actor_views/{演员（饰角色）}/`（复制命中图）+ `actor_view_report.json`（每图命中人物及相似度）。

### 2.5 scene_split
- 主：EXIF DateTimeOriginal 排序，间隔 > `SCENE_GAP_SECONDS`（默认 300s）切段，每段为一个候选场景。
- 辅：SigLIP2 embedding 在段内做 KMeans（k≤3）细分布景变化。
- 场景标签生成 `场景-01`（占位），支持外部传入人工标注映射表覆盖（`scene_labels.json`）。

### 2.6 role_classifier（F5）
- SigLIP2 zero-shot：对每图主角人物框裁剪区域，与文本模板
  （生/旦/净/丑，中英双语多模板取平均）算相似度，softmax 输出行当及置信度；
  置信度 < 0.4 标记 `uncertain`。低置信结果仅入报告，不写入文件名。

### 2.7 smart_organizer（F1 编排）
```
输入原始目录
 → filename_parser 解析元数据
 → burst_dedup 连拍去重选优（景别感知）
 → 逐张：SCRFD/ArcFace 识别人物列表（ES 库）+ role_classifier 行当
 → scene_split 场景划分
 → 生成规范文件名（序号按场景内时间序）复制到输出目录
 → organize_report.json（每图：来源、保留/丢弃、人物、行当、场景、新文件名）
```

## 3. 里程碑

| 里程碑 | 内容 | 回归测试 | 状态 |
|--------|------|----------|------|
| M1 | filename_parser（纯逻辑） | tests/test_filename_parser.py（4 部剧目全量断言：日期/剧目 555/555、人物 507/507、连拍 208/208） | ✅ |
| M2 | burst_dedup（F3） | tests/test_burst_dedup.py（47 张 → 33 组、剪除 14、景别桶保留生效） | ✅ |
| M3 | face_db_builder + 三剧目建库 | tests/test_face_db_build.py（两轮构建：单人归属 + 多人锚定传播） | ✅ |
| M4 | actor_view（F2） | tests/test_actor_view.py（5 演员专辑 GT 精确率 82%~100%） | ✅ |
| M5 | scene_split + role_classifier（F5） | tests/test_scene_role.py（行当 4/5=80%；EXIF 分段与幕次对应） | ✅ |
| M6 | smart_organizer 编排 + API 端点 | tests/test_smart_organize.py（端到端）；test_scrfd_arcface.py 回归无异常 | ✅ |
| M7 | 验收 | tests/acceptance_m7.py（四剧目全量，报告 outputs/acceptance/） | ✅ |

### M7 验收结果摘要（2026-08-31，日志 outputs/acceptance/acceptance_run.log）

- 验收库：29 位演员（样式雷正式库并入；四剧目锚定齐全，样式雷主演 8~33 张）
- 流水线（保留/输入，命名违规均为 0）：

| 剧目 | 输入 | 保留 | 剪除 | 场景 | 识别覆盖 |
|------|------|------|------|------|----------|
| 京剧《天下归心》 | 35 | 33 | 2 | 16 | 30/33 |
| 歌剧《假面舞会》 | 119 | 107 | 12 | 21 | 91/107 |
| 话剧《样式雷》 | 79 | 30 | 49 | 6 | 24/30 |
| 舞剧《马可·波罗》 | 59 | 52 | 7 | 11 | 30/52 |

- 行当 zero-shot（京剧锚定图 GT）：4/5 = 80%（李鸣岩老旦判为生行，跨行当边界样本）
- 识别覆盖未满原因：远景小脸龙套与净角脸谱照（SCRFD 泛化限制，符合预期不识别）

### 已知数据瑕疵与限制

- 源文件名对同一演员存在用字变体（齐乌利/齐乌力），库中拆为两个目录，
  需「演员别名合并」功能或人工合并。
- 孟广禄（净角黑脸谱）锚定仅 1 张：单人照均为远景小脸（~40px）+ 黑脸谱，SCRFD 检不出。
- 话剧《样式雷》原始目录大量无括号散件（同基础名单张），景别桶剪除偏激进（79→30），
  实际使用时可调高 keep_per_bucket。

## 4. 验收标准

1. **连拍选优**：括号连拍组识别率 100%；每组保留图人工抽检合理（清晰不糊、景别符合保留策略）。
2. **人脸库**：三剧目主演均 ≥3 张锚定入库；样张识别 top-1 命中本人（余弦 ≥0.55）。
3. **按演员视图**：随机抽 5 名演员，其专辑中同人照片占比 ≥80%。
4. **行当分类**：高置信（≥0.4）样本抽检准确率 ≥70%；低置信样本正确标记 uncertain。
5. **流水线**：4 部剧目输出目录命名 100% 符合格式；输出人物列表与人工整理
   【整理完成】文件名人物列表的主演召回率 ≥70%（受建库覆盖限制，报告实际值）。
6. **回归**：现有 8 个测试脚本可运行，核心链路（TRT 速度、批量优化、SCRFD/ArcFace）无回归。

## 5. 风险与对策

- 行当 zero-shot 精度未知 → 置信度门槛 + 只入报告不写文件名，M5 实验后决定是否保留。
- 幕次语义无法全自动 → 场景标签为占位（场景-01…），支持人工映射覆盖。
- 建库依赖单人照片存在 → 若某演员无单人照，报告中列出待人工补标名单。
