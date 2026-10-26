"""Create editable Markdown and an offline, print-ready technical report."""
import base64
import html
import json
import mimetypes
from pathlib import Path
import re
import markdown

OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'data/report_data.json').read_text())
sources=json.loads((OUT/'data/evidence_index.json').read_text())
source=(OUT/'report_source.md').read_text()
rows=['| 后端 | 文本/候选池 | Top-1 | R@1 | Top-3 | R@3 | Macro-F1 |',
      '|---|---|---:|---:|---:|---:|---:|']
for row in data['data']['retrieval']:
    name='SigLIP2' if row['backend']=='siglip2_base' else 'Gemma 2'
    values=[name,row['label'],str(row['correct_at_1']),f"{100*row['recall_at_1']:.2f}%",str(row['correct_at_3']),
            f"{100*row['recall_at_3']:.2f}%",f"{100*row['macro_f1']:.2f}%"]
    rows.append('| '+' | '.join(values)+' |')
def evidence_table(items):
    rows = ['| 编号 | 证据内容 | 原报告路径 |', '|---|---|---|']
    for ref in items:
        short = ref['path'].removeprefix('outputs/acceptance/core_upgrade/')
        rows.append(f'| <span id="{ref["id"]}">{ref["id"]}</span> | {ref["title"]} | <span class="source-path">{short}</span> |')
    return '\n'.join(rows)

expanded = (source.replace('{{retrieval_table}}', '\n'.join(rows))
            .replace('{{evidence_table_first}}', evidence_table(sources[:18]))
            .replace('{{evidence_table_second}}', evidence_table(sources[18:]))
            .replace('{{commit_short}}', data['commit'][:7]))
header = '# 幕间：戏剧照片智能整理系统技术报告\n\n技术评审与研发版 · 2026-10-26 · HEAD `4194277` 与测量工作区摘要\n\n'

# GitHub renders this version without relying on CSS, SVG fonts or ignored outputs.
readable = expanded
readable = re.sub(r'<div class="kpi-grid">.*?</div></div>',
    '| 检索开发集 | Gemma短块均值R@3·25池 | 可追溯检测人脸 | 真实流程验证 |\n|---|---|---|---|\n|262张|76.34%|1095脸|200张|',
    readable, flags=re.S)
readable = re.sub(r'<div class="(?:note|callout|equation)">', '\n\n> ', readable)
readable = re.sub(r'</?div[^>]*>', '', readable)
readable = re.sub(r'<span[^>]*>|</span>', '', readable)
readable = re.sub(r'\[(S\d{2})\]', r'[\1](reports/technical_report_20261026/data/evidence/\1.json)', readable)
readable = re.sub(r'(?<=\| )(S\d{2})(?= \|)', r'[\1](reports/technical_report_20261026/data/evidence/\1.json)', readable)
readable = re.sub(r'!\[([^\]]+)\]\(figures/([^)]+)\)',
    lambda m: f'![{m[1]}](reports/technical_report_20261026/figures/{m[2]})\n\n*{m[1]}*', readable)
links = ('[PDF](reports/technical_report_20261026/technical_report.pdf) · '
         '[离线HTML](reports/technical_report_20261026/technical_report.html) · '
         '[插图与来源数据](reports/technical_report_20261026/README.md) · '
         '[历史版本](technical_report_20261024.md)\n\n')
(OUT.parents[2] / 'docs/technical_report_20261026.md').write_text(header + links + readable)
bundle_readable = readable.replace('reports/technical_report_20261026/', '')
bundle_links = ('[PDF](technical_report.pdf) · [离线HTML](technical_report.html) · '
                '[插图与来源数据](README.md) · [历史版本](../../technical_report_20261024.md)\n\n')
(OUT / 'technical_report.md').write_text(header + bundle_links + bundle_readable)

def embed(path):
    mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()

def figure(match):
    tag=match[1];alt=html.unescape(re.search(r'alt="([^"]*)"',tag)[1]);rel=re.search(r'src="([^"]*)"',tag)[1]
    p=OUT/rel
    tag=tag.replace('src="'+rel+'"','src="'+embed(p)+'"')
    kind='screenshot' if p.name in ['workspace.jpg','fact_review.jpg'] else 'chart'
    return '<figure class="'+kind+'">'+tag+'<figcaption>'+html.escape(alt)+'</figcaption></figure>'

parts=re.split(r'<!-- PAGE: ([^|]+) \| (.*?) -->',expanded)
pages=[];toc=[]
for i in range(1,len(parts),3):
    slug,title,content=parts[i].strip(),parts[i+1].strip(),parts[i+2].strip()
    rendered=markdown.markdown(content,extensions=['tables','fenced_code','sane_lists'])
    rendered=re.sub(r'<p>(<img [^>]+/?>)</p>',figure,rendered)
    rendered=re.sub(r'\[(S\d{2})\]',r'<a class="cite" href="#\1">[\1]</a>',rendered)
    page_no=len(pages)+2
    classes='page appendix' if slug in ('sources','sources_more') else 'page'
    pages.append(f'<section class="{classes}" id="{slug}" data-page="{page_no}"><header class="running"><span>幕间 / TECHNICAL REPORT</span><span>{html.escape(title)}</span></header><div class="content">{rendered}</div><footer><span>戏剧照片智能整理 · 2026.10</span><span>{page_no:02d} / {{TOTAL}}</span></footer></section>')
    toc.append({'id':slug,'title':title,'page':page_no})
total=len(pages)+1
cover=f'''<section class="page cover" id="cover" data-page="1">
  <div class="cover-top"><span>THEATRICAL PHOTO INTELLIGENCE</span><span>TECHNICAL REPORT / 2026</span></div>
  <div class="cover-title"><div class="cover-line"></div><div class="edition">技术评审与研发版</div><h1>幕间</h1><h2>戏剧照片智能整理系统</h2><p>功能架构 · 模型原理 · 实验设计 · 结果分析</p></div>
  <div class="cover-art"><img src="{embed(OUT/'figures/theatre_montage.jpg')}" alt="四部剧目的本地舞台图像拼图，用于场景说明"/><div class="art-label">THE STAGE, STRUCTURED.</div></div>
  <div class="cover-bottom"><p>以可追溯证据连接图像、演员、剧情与人工决定</p><div><span>2026年10月26日</span><span>测量HEAD {data['commit'][:7]} + 工作区摘要</span></div><small>4部剧目 · 262张幕次开发样本 · 1095张检测人脸</small></div>
</section>'''
css=r'''
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:#263b48;background:#e8ecef;font-family:"Noto Sans CJK SC","Microsoft YaHei","SimHei",sans-serif;font-size:13.1px;line-height:1.72;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.toolbar{position:fixed;top:0;left:0;right:0;height:56px;background:#152d41;color:white;z-index:20;display:flex;align-items:center;gap:18px;padding:0 28px;box-shadow:0 3px 15px #10233522}.brand{font-size:17px;font-weight:700;letter-spacing:2px}.toolbar small{opacity:.65}.toolbar select{margin-left:auto;max-width:330px}.toolbar select,.toolbar button{border:1px solid #627b8d;background:#203e54;color:white;border-radius:5px;padding:7px 12px;font:inherit;cursor:pointer}.toolbar a{color:white}.shell{padding:82px 0 30px}.page{position:relative;background:white;width:210mm;height:297mm;margin:0 auto 25px;padding:15mm 17mm 18mm;box-shadow:0 8px 35px #1c35421b;break-after:page;overflow:visible}.running{display:flex;justify-content:space-between;border-bottom:1px solid #c8d2d9;padding-bottom:9px;margin-bottom:18px;color:#7b8d99;font-size:9px;letter-spacing:.8px}.running span:first-child{color:#007f83}.content h1{font-size:25px;line-height:1.35;margin:0 0 17px;color:#173248;font-weight:700;letter-spacing:.3px}.content h2{font-size:16px;line-height:1.4;margin:18px 0 8px;color:#007f83}.content p{margin:0 0 11px;text-align:left}.content strong{color:#173248}.content ol,.content ul{padding-left:22px;margin:10px 0 12px}.content li{margin:0 0 9px;padding-left:3px}.content a{color:#007f83;text-decoration:none}.content .cite{font-family:Arial,sans-serif;font-size:9px;vertical-align:super;margin-left:2px;white-space:nowrap}.content table{border-collapse:collapse;width:100%;margin:12px 0 15px;font-size:11.8px;line-height:1.58;table-layout:auto}.content th{background:#173248;color:white;text-align:left;padding:8px 9px;font-weight:normal}.content td{border-bottom:1px solid #dbe2e7;padding:7px 9px;vertical-align:top}.content tr:nth-child(even) td{background:#f2f6f7}.content table strong{font-weight:700}.content code{font-family:"DejaVu Sans Mono",Consolas,monospace;font-size:.83em;color:#31586b;background:#edf3f5;border-radius:2px;padding:1px 3px;overflow-wrap:anywhere}.content pre{font-size:11px;line-height:1.7;background:#edf3f5;border-left:3px solid #007f83;padding:13px 15px;white-space:pre-wrap;overflow-wrap:anywhere}.content pre code{background:none;padding:0}.content figure{margin:13px 0 15px;break-inside:avoid}.content figure img{display:block;width:100%;height:auto;max-height:110mm;object-fit:contain}.content figure.screenshot img{max-height:87mm;border:1px solid #d4dfe5;border-radius:5px}.content figcaption{font-size:10px;line-height:1.55;color:#6b7d89;margin-top:7px;padding-left:9px;border-left:2px solid #c38d38;text-align:justify}.equation{font-family:"DejaVu Serif","SimSun",serif;font-size:13px;line-height:1.65;background:#f0f5f6;border-top:1px solid #d9e4e6;border-bottom:1px solid #d9e4e6;text-align:center;padding:12px 8px;margin:15px 0;color:#174c61}.callout{border-left:4px solid #c38d38;padding:12px 15px;background:#faf5ea;margin:15px 0;font-size:12px;line-height:1.7}.note{border-left:3px solid #007f83;padding:9px 13px;background:#edf5f5;margin:13px 0;font-size:11.8px;line-height:1.7}.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}.kpi-grid>div{padding:13px 9px;background:#f0f5f6;border-top:2px solid #007f83}.kpi-grid b{display:block;font-family:Arial,sans-serif;font-size:29px;color:#007f83;line-height:1.3}.kpi-grid span{font-size:9px;color:#6b7d89}footer{position:absolute;bottom:9mm;left:17mm;right:17mm;border-top:1px solid #d8e0e5;padding-top:7px;display:flex;justify-content:space-between;color:#83939e;font-size:9px;letter-spacing:.4px}.cover{background:#132c42;color:white;padding:14mm 17mm 12mm;overflow:hidden}.cover-top{display:flex;justify-content:space-between;color:#b3c4cf;font-size:8px;letter-spacing:1.4px}.cover-title{margin-top:26mm}.cover-line{height:4px;width:42px;background:#c38d38;margin-bottom:14px}.edition{font-size:11px;letter-spacing:4px;color:#caa45f}.cover h1{font-size:92px;letter-spacing:15px;font-weight:700;line-height:1.2;margin:14px 0 10px;color:white}.cover h2{font-size:29px;letter-spacing:3px;font-weight:normal;margin:0 0 15px;color:white}.cover-title p{font-size:12px;letter-spacing:2px;color:#b5c9d6}.cover-art{height:89mm;margin:23mm -17mm 0;position:relative;overflow:hidden}.cover-art img{width:100%;height:100%;object-fit:cover;opacity:.85}.cover-art:after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,#132c4200 30%,#132c4277)}.art-label{position:absolute;bottom:15px;left:17mm;z-index:1;letter-spacing:4px;font-size:10px;color:#dce4e9}.cover-bottom{margin-top:13mm}.cover-bottom p{font-size:17px;letter-spacing:1px;margin:0 0 20px}.cover-bottom>div{display:flex;justify-content:space-between;color:#b5c9d6;font-size:11px}.cover-bottom small{display:block;color:#9cb3c2;font-size:10px;margin-top:9px;letter-spacing:1px}.appendix .content table{font-size:10.4px;line-height:1.4}.appendix .content td{padding:5px 7px}.appendix .content th{padding:6px 7px}.appendix .content p{font-size:11.5px}#references th:nth-child(1){width:9%}#references th:nth-child(2){width:59%}#references th:nth-child(3){width:32%}.source-path{font-family:Consolas,monospace;font-size:9.3px;overflow-wrap:anywhere;word-break:break-word}.appendix th:nth-child(1){width:8%}.appendix th:nth-child(2){width:32%}.appendix th:nth-child(3){width:60%}.zoom{cursor:zoom-in}dialog{padding:10px;border:0;border-radius:7px;max-width:96vw;background:white}dialog img{max-width:92vw;max-height:88vh;object-fit:contain}dialog::backdrop{background:#142b42d9}dialog button{position:absolute;right:10px;top:10px;border:0;background:#173248;color:white;border-radius:30px;width:30px;height:30px;cursor:pointer}
@media screen and (max-width:850px){.toolbar{padding:0 12px;gap:8px}.toolbar small{display:none}.toolbar select{width:46%;font-size:11px}.toolbar button{font-size:11px;padding:7px}.shell{padding-top:69px}.page{width:calc(100% - 20px);height:auto;min-height:0;padding:25px 22px 58px;margin-bottom:15px}.running{font-size:8px;gap:10px}.content h1{font-size:23px}.content table{font-size:10px;line-height:1.6}.content th,.content td{padding:6px}.content figure img{max-height:none}.equation{font-size:10px;overflow-wrap:anywhere}.kpi-grid{grid-template-columns:repeat(2,1fr)}footer{left:22px;right:22px;bottom:15px}.cover{padding:24px 24px 35px}.cover-title{margin-top:45px}.cover h1{font-size:76px}.cover h2{font-size:23px}.cover-art{height:220px;margin:45px -24px 0}.cover-bottom{margin-top:28px}.cover-bottom p{font-size:14px}.cover-top{font-size:6px}.appendix .content td{font-size:9px}.source-path{font-size:8px}.content pre{font-size:9px}}
@page{size:A4;margin:0}@media print{html{scroll-behavior:auto}body{background:white}.toolbar{display:none}.shell{padding:0}.page{margin:0;box-shadow:none;break-after:page}.page:last-child{break-after:auto}.content a{color:inherit}.zoom{cursor:auto}dialog{display:none}}
'''
options='<option value="cover">封面 / Cover</option>'+''.join(f'<option value="{x["id"]}">{x["page"]:02d} · {html.escape(x["title"])}</option>' for x in toc)
script='''const jump=document.querySelector('#jump');jump.onchange=()=>document.getElementById(jump.value).scrollIntoView({behavior:'smooth'});document.querySelector('#print').onclick=()=>window.print();const modal=document.querySelector('dialog');document.querySelectorAll('.content figure img').forEach(img=>{img.classList.add('zoom');img.onclick=()=>{modal.querySelector('img').src=img.src;modal.querySelector('img').alt=img.alt;modal.showModal();}});modal.querySelector('button').onclick=()=>modal.close();modal.onclick=e=>{if(e.target===modal)modal.close()};'''
document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>幕间 · 戏剧照片智能整理系统技术报告</title><style>'+css+'</style></head><body><nav class="toolbar"><span class="brand">幕间</span><small>TECHNICAL REPORT · 2026.10</small><select id="jump" aria-label="跳转章节">'+options+'</select><button id="print">打印 / PDF</button></nav><main class="shell">'+cover+''.join(pages).replace('{TOTAL}',str(total))+'</main><dialog aria-label="放大图表"><button aria-label="关闭">×</button><img alt="放大图表"></dialog><script>'+script+'</script></body></html>'
(OUT/'technical_report.html').write_text(document)
(OUT/'data/page_index.json').write_text(json.dumps([{'id':'cover','title':'封面','page':1}]+toc,ensure_ascii=False,indent=2))
print(json.dumps({'pages':total,'html_bytes':len(document.encode()),'chapters':len(toc),'figures_in_report':len(re.findall('<figure',document))}))
