# 2026-10-26 技术报告

[GitHub阅读版](../../technical_report_20261026.md) · [包内Markdown](technical_report.md) · [下载PDF](technical_report.pdf) · [下载离线HTML](technical_report.html)

29页、15幅正文插图。新版补充Pillow尺寸修复、同步双线程选优及独立DALI评价；历史基线保持单列。最新200图×5次对照特征加速1.385倍、整理中位耗时减少26.539%，仍未通过特征1.5倍门槛；共享GPU下P95未改善，独立质量未验收。DALI因输出变化及有限整链路收益未采用。

Markdown插图使用仓库内相对PNG/JPEG路径，中文文字已固化到图像中；每幅统计/原理图另有SVG。PDF与内嵌全部图片的HTML可下载后离线阅读，HTML支持目录跳转与看图放大。GitHub不直接执行HTML，在线阅读使用上方Markdown入口。

## 文件与来源

- [report_source.md](report_source.md)：带分页标记的可编辑正文；阅读版由脚本生成。
- [figures/](figures/)：正文插图、舞台示意照片、工作台截图和可编辑SVG。
- [data/evidence_index.json](data/evidence_index.json)：31份证据的原实验路径、SHA256和提交快照路径；对应JSON保存在[data/evidence/](data/evidence/)。原始路径仅用于来源追溯，不是阅读/重建必需路径。
- [检索完整数值CSV](data/retrieval_metrics.csv)与[最新U2五轮数值CSV](data/u2_repeat_metrics.csv)：可用于独立复核图表。
- [报告版本与测量源码摘要](data/report_data.json)：测量使用HEAD `4194277`及当时未提交U2工作区，报告提交不会改变原实验的实现版本。
- [验证摘要](data/verification_summary.json)与[文件摘要清单](artifact_manifest.json)：来源、数值、布局、PDF和Git文件完整性检查。

舞台照片和复核截图来自已有材料，仅用于功能说明。没有重新推理、生成新人工标注或修改原实验。旧版本见[2026-10-24归档](../technical_report_20261024/README.md)。

## 重建

阅读和重建图表无需原图、模型权重、CUDA或实验数据库。需要Python 3.10+、Node.js和可显示中文的字体；建议Noto Sans CJK SC。依赖单独列出，不影响应用环境：

```bash
python -m pip install -r docs/reports/technical_report_20261026/requirements_report.txt
npm install --no-save playwright
npx playwright install chromium
python docs/reports/technical_report_20261026/build_figures.py
python docs/reports/technical_report_20261026/build_report.py
node docs/reports/technical_report_20261026/render_report.cjs
python docs/reports/technical_report_20261026/verify_report.py
```

以上命令在仓库根目录执行。`REPORT_CJK_FONT`可指定Matplotlib使用的中文字体文件；默认先找本机黑体，否则使用Noto Sans CJK SC。`PLAYWRIGHT_MODULE`可指定已有Playwright模块目录，`CHROMIUM_PATH`可指定已有Chromium可执行文件。改动原理SVG后先运行渲染脚本的`--figures-only`选项，再重建正文和PDF。

浏览器诊断默认写入被忽略的`.report_check/`；`REPORT_DIAGNOSTICS`可改写输出目录。验证脚本优先使用本轮诊断，没有时读取随报告保存的布局记录；`--diagnostics 路径`指定外部记录。`--git-ref staged`核对暂存区的链接和文件字节，`--git-ref HEAD --check-only`核对当前提交且不写入新产物。历史实验JSON如在本地存在，会额外核对其字节；在干净克隆中则只使用冻结快照。
