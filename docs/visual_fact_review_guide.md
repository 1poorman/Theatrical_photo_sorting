# 画面事实与 OCR 双人复核

本机已有[复核页面](../outputs/acceptance/core_upgrade/u5_dual_fact_review_v2_20261024/index.html)，包含100张开发集图片的1352条模型描述，以及5张OCR图片。每次显示一张图片，A、B两份草稿分别保存，可导出、恢复；分页、刷新、错误提示和五种屏宽共17项浏览器检查通过。页面与图片放在同一目录，可直接打开`index.html`。

两位复核人各自选择A或B，填写姓名，独立对照图片完成判断并导出JSON。草稿只保存在各自浏览器；同一浏览器中的A/B槽位只是文件隔离，不能证明复核独立性。当前没有两位实际复核人，全部条目仍为`pending`，人工核对数为0。

画面事实按服装、道具、环境、动作和可见文字逐条判断为“画面支持”“事实错误”或“无法评价”。错误项填写更正或说明，重复描述单独勾选，可补录遗漏内容。填写整图复核依据后点击“本图已复核”；无法判断整图时使用“本图不可评价”。模型不确定项不计入事实分母，也不能据此确认演员身份。

OCR任务只展示图片和空白转录框，不展示或预填预测文字。按阅读顺序逐字转录，每行一段；没有可见文字时明确勾选对应选项。字太小或无法辨认时保留不可评价。当前5张图片只足以验证转录流程，不能代替覆盖充分的文字质量验收。

在项目根目录核查已有包及其来源，不调用模型：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/prepare_visual_fact_review.py \
  --output outputs/acceptance/core_upgrade/u5_dual_fact_review_v2_20261024 \
  --verify-existing
```

获得两份真实复核文件后生成新报告，工具要求报告路径尚不存在：

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/prepare_visual_fact_review.py \
  --output outputs/acceptance/core_upgrade/u5_dual_fact_review_v2_20261024 \
  --review-a /path/to/fact_review_A.json \
  --review-b /path/to/fact_review_B.json \
  --report /path/to/new/fact_quality.json
```

校验覆盖协议、A/B槽位、原图及预览摘要、事实内容、OCR预测和完成状态；每种任务内相同原图字节不能重复计数。两人均已完成且判断一致的`supported + error`是事实错误率分母；重复描述仍计入，待定、无法评价和分歧单列。遗漏只统计条数，不推算完整召回率。至少100张具有一致可评价事实的图片且没有事实分歧时，才报告开发集5%错误率检查；正式`quality_gate`始终保留`insufficient_data`。

OCR只使用两人转录一致的图片计算字符编辑距离。文本统一NFC、换行符和首尾空白，保留内部空白；预测采用原始OCR行序，不按参考答案重排。分母为参考字符总数，空参考不生成准确率，插入错误可使CER超过1。原图、标注、图库和数据划分均不由本工具修改。

这100张视觉观察已完成100次按需请求，同进程重放新增请求为0，10项完整性检查通过。该模式为收集复核材料，不改变默认自动预算。服务别名为`Qwen3.6-27B`，配置声明为`Qwen3.8-27B`，没有可核验权重摘要；两者不能当作已证明的模型版本。冻结观察、旧版页面和浏览器失败记录保留，当前v2包通过修正后的评测校验，无需重新推理或重建页面。

最新[只读核查与待定报告](../outputs/acceptance/core_upgrade/completion_audit_20261024/)记录实际产物与版本。现有图片已用于开发；即使补完复核，也不会自动成为独立final test或产生Macro-F1结果。跨机器使用须携带复核目录及其引用的输入资料，`outputs/`不随Git提交。
