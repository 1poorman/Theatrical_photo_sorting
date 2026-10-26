"""Rebuild scientific figures from committed evidence; no models or raw photos required."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/theatre-report-mpl')
import csv
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FONT = os.environ.get('REPORT_CJK_FONT', '/usr/share/fonts/chinese/simhei.ttf')
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    font_family = font_manager.FontProperties(fname=FONT).get_name()
else:
    font_family = 'Noto Sans CJK SC'
plt.rcParams.update({'font.family': font_family, 'font.size': 10, 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c4ccd2',
    'axes.labelcolor': '#263b48', 'text.color': '#263b48', 'xtick.color': '#536777',
    'ytick.color': '#536777', 'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
    'svg.fonttype': 'path', 'savefig.facecolor': '#ffffff'})
TEAL, GOLD, SLATE, RED = '#007f83', '#c38d38', '#536c87', '#b5554d'
data = {}
for row in json.loads((OUT / 'data/evidence_index.json').read_text()):
    path = OUT / row['snapshot']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Evidence snapshot changed: ' + row['id'])
    data[row['id']] = json.loads(raw)
variants = ['baseline_samepool', 'long_samepool', 'blocks_mean_samepool', 'blocks_max_samepool',
            'long_expanded', 'blocks_mean_expanded', 'blocks_max_expanded']
names = ['原文基线 · 25', '增补长文 · 25', '短块均值 · 25', '短块最大值 · 25',
         '增补长文 · 28', '短块均值 · 28', '短块最大值 · 28']
backends = ['siglip2_base', 'embeddinggemma2']
retrieval = data['S02']
records = []
for backend in backends:
    for variant, label in zip(variants, names):
        m = retrieval['results'][backend][variant]['all']['metrics']
        records.append({'backend': backend, 'variant': variant, 'label': label, **m})
with (OUT / 'data/retrieval_metrics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(records)

def save(fig, name):
    fig.savefig(FIG / (name + '.svg'), bbox_inches='tight', pad_inches=.16)
    svg = FIG / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(FIG / (name + '.png'), dpi=190, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)

# All 14 variants, data series share identical query count and scoring protocol.
fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.6), sharey=True)
y = np.arange(len(variants)); h=.32
for ax, metric, title in zip(axes, ['recall_at_1','recall_at_3'], ['Top-1 命中率 / %', 'Top-3 召回率 / %']):
    for j, (b, color) in enumerate(zip(backends, [SLATE,TEAL])):
        values = [100*retrieval['results'][b][v]['all']['metrics'][metric] for v in variants]
        bars=ax.barh(y+(j-.5)*h, values, height=h, color=color, label=['SigLIP2','EmbeddingGemma 2'][j])
        ax.bar_label(bars, labels=[f'{x:.2f}' for x in values], padding=3, fontsize=8)
    ax.set_yticks(y, names); ax.set_xlim(0,92 if metric=='recall_at_3' else 47)
    ax.set_title(title,loc='left',fontweight='bold',pad=16); ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
axes[0].invert_yaxis();axes[1].legend(loc='upper right',bbox_to_anchor=(1,-.075),ncol=2,frameon=False,fontsize=9)
fig.tight_layout(w_pad=3);save(fig,'retrieval_variants')

plays = list(retrieval['play_titles'])
short = [retrieval['play_titles'][p].replace('京剧','').replace('歌剧','').replace('舞剧','').replace('话剧','') for p in plays]
heat_variants=['long_samepool','blocks_mean_samepool','blocks_mean_expanded']
fig,axes=plt.subplots(1,2,figsize=(10.6,4.1),sharey=True)
for ax,b,title in zip(axes,backends,['SigLIP2 base','EmbeddingGemma 2']):
    values=np.array([[100*retrieval['results'][b][v]['all']['per_play'][p]['recall_at_3'] for v in heat_variants] for p in plays])
    ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
    for (i,j),value in np.ndenumerate(values): ax.text(j,i,f'{value:.1f}',ha='center',va='center',color='white' if value>=62 else '#263b48',fontsize=12)
    ax.set_xticks(range(3),['增补长文25','短块均值25','短块均值28'],fontsize=9)
    ax.set_yticks(range(4),short);ax.tick_params(length=0);ax.set_title(title,loc='left',pad=12)
fig.tight_layout(w_pad=1.5);save(fig,'per_play_heatmap')

fig,axes=plt.subplots(1,2,figsize=(10.4,3.7))
for ax,metric,title in zip(axes,['recall_at_1','recall_at_3'],['Top-1 命中数 / 262','Top-3 命中数 / 262']):
    for b,color,label in zip(backends,[SLATE,TEAL],['SigLIP2','Gemma 2']):
        key='correct_at_1' if metric=='recall_at_1' else 'correct_at_3'
        counts=[retrieval['results'][b][v]['all']['metrics'][key] for v in ['blocks_mean_samepool','blocks_mean_expanded']]
        ax.plot([0,1],counts,'o-',lw=2.5,ms=8,color=color,label=label)
        for x,v in enumerate(counts):ax.annotate(str(v),(x,v),xytext=(0,10 if b==backends[1] else -17),textcoords='offset points',ha='center',color=color)
    ax.set_xticks([0,1],['25候选','28候选']);ax.set_xlim(-.3,1.3);ax.set_ylim((35,90) if metric=='recall_at_1' else (120,218));ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.16)
axes[1].legend(frameon=False,loc='lower left');fig.tight_layout();save(fig,'candidate_expansion')

fig,axes=plt.subplots(1,2,figsize=(10.4,3.25))
cpu=data['S08']['results']; methods=['window','graph','legacy']; labels=['窗口变化点\n768维','图聚类\n512维','层次聚类\n512维']
for ax,values,title,fmt in [(axes[0],[cpu[m]['seconds'] for m in methods],'CPU算法耗时 / 秒','{:.3f}'),(axes[1],[cpu[m]['process_peak_rss_bytes']/2**20 for m in methods],'独立进程峰值RSS / MiB','{:.1f}')]:
    bars=ax.bar(labels,values,color=[GOLD,TEAL,SLATE],width=.5);ax.bar_label(bars,labels=[fmt.format(v) for v in values],padding=5,fontsize=10)
    ax.set_ylim(0,max(values)*1.25);ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'cpu_scale')

fig,axes=plt.subplots(1,2,figsize=(10.3,3.15))
http=data['S11']['measurements'];keys=['health','page','create','cancel']
values=[http[k]['p95_seconds']*1000 for k in keys]
bars=axes[0].barh(['健康查询 n=150','分页查询 n=150','创建任务 n=5','取消确认 n=1'],values,color=[SLATE,SLATE,TEAL,GOLD],height=.56)
axes[0].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4);axes[0].set_xlim(0,150);axes[0].invert_yaxis();axes[0].set_title('真实任务HTTP / ms',loc='left');axes[0].grid(axis='x',alpha=.15)
snap=data['S13']['results'];values=[snap[m]['median_seconds']*1000 for m in ['eager','deferred']]
bars=axes[1].bar(['即时快照','延后快照'],values,color=[SLATE,TEAL],width=.48)
axes[1].bar_label(bars,labels=[f'{v:.3f} ms' for v in values],padding=5);axes[1].set_ylim(0,265);axes[1].set_title('万条队列：单条保存与摘要中位',loc='left');axes[1].grid(axis='y',alpha=.15)
fig.tight_layout(w_pad=2);save(fig,'review_responsiveness')

fig,axes=plt.subplots(1,2,figsize=(10.4,3.2))
for ax,kind,title in zip(axes,['feature_runs','pipeline_runs'],['特征阶段 · 仅1.151×','报告定义的分析/整理阶段 · 3.155×']):
    t=data['S14']['timing'][kind];vals=[t['baseline_median_seconds'],t['batch_median_seconds']]
    bars=ax.bar(['基线','批量/缓存'],vals,color=[SLATE,TEAL],width=.5);ax.bar_label(bars,labels=[f'{v:.2f} s' for v in vals],padding=5)
    ax.set_ylim(0,max(vals)*1.25);ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'u2_performance')

fig,ax=plt.subplots(figsize=(10.1,3))
raw={p['play_id']:p for p in data['S15']['projects']};order=list(raw)
kept=[raw[p]['copied'] for p in order];skip=[raw[p]['discarded'] for p in order]
ax.barh(range(4),kept,color=TEAL,label='已复制');ax.barh(range(4),skip,left=kept,color='#d8dfe4',label='明确跳过')
for i,(k,s) in enumerate(zip(kept,skip)):
    ax.text(k/2,i,str(k),va='center',ha='center',color='white',fontweight='bold')
    ax.text(k+s/2,i,str(s),va='center',ha='center',color='#364a59')
ax.set_yticks(range(4),[retrieval['play_titles'][p] for p in order]);ax.set_xlim(0,67);ax.invert_yaxis();ax.set_xlabel('原始流程：输入数 = 复制 + 跳过 + 失败（失败均为0）');ax.legend(frameon=False,loc='lower right');fig.tight_layout();save(fig,'workflow_outputs')

fig,axes=plt.subplots(1,2,figsize=(10.4,3))
counts=[retrieval['results']['siglip2_base']['blocks_mean_samepool']['all']['per_play'][p]['count'] for p in plays]
bars=axes[0].barh(short,counts,color=[SLATE,TEAL,GOLD,'#9aaeb4']);axes[0].bar_label(bars,padding=4);axes[0].set_xlim(0,140);axes[0].invert_yaxis();axes[0].set_title('幕次开发集 / 张',loc='left')
axes[1].axis('off')
for i,(value,label) in enumerate([('582','目录文件（包含原始/整理副本）'),('319','去重可解码图像（含资料图）'),('292','舞台图像线索'),('262','已确认幕次的中性检索query')]):
    axes[1].text(.02,.85-i*.235,value,fontsize=25,color=TEAL,fontweight='bold');axes[1].text(.23,.88-i*.235,label,fontsize=10,va='center')
fig.tight_layout();save(fig,'dataset_scope')

# New comparison: both medians and every full-pipeline repeat are shown.
latest = data['S25']
baseline, candidate = data['S26'], data['S27']
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
keys = ['feature_runs', 'pipeline_runs']
labels = ['特征阶段', '完整整理', '选优环节']
old = [latest['timing'][key]['baseline_median_seconds'] for key in keys]
new = [latest['timing'][key]['batch_median_seconds'] for key in keys]
old.append(statistics.median(row['stage_timings']['dedup_seconds'] for row in baseline['pipeline_runs']))
new.append(statistics.median(row['stage_timings']['dedup_seconds'] for row in candidate['pipeline_runs']))
y = np.arange(3)
for values, color, shift, name in [(old, SLATE, -.16, '同期已有批量基线'),
                                  (new, TEAL, .16, 'Pillow + 双线程选优')]:
    bars = axes[0].barh(y + shift, values, height=.3, color=color, label=name)
    axes[0].bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=8)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlim(0, 205)
axes[0].set_xlabel('各环节中位耗时 / 秒')
axes[0].set_title('同期200图 × 5次；已有批量基线', loc='left', fontsize=10)
axes[0].grid(axis='x', alpha=.15)
axes[0].set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
for report, color, name in [(baseline, SLATE, '基线'), (candidate, TEAL, '候选')]:
    values = [row['seconds'] for row in report['pipeline_runs']]
    axes[1].plot(range(1, 6), values, 'o-', color=color, lw=2, label=name)
    for i, value in enumerate(values, 1):
        offset = (10 if name == '候选' else -15) if i == 1 else (8 if name == '基线' else -15)
        axes[1].annotate(f'{value:.1f}', (i, value), xytext=(0, offset),
                         textcoords='offset points', ha='center', fontsize=8, color=color)
axes[1].set_xticks(range(1, 6))
axes[1].set_ylim(95, 205)
axes[1].set_xlabel('重复序号（保留全部五轮）')
axes[1].set_ylabel('完整整理 / 秒')
axes[1].set_title('中位下降26.54%；P95未改善', loc='left', fontsize=10)
axes[1].grid(axis='y', alpha=.15)
axes[1].legend(frameon=False, loc='upper right', fontsize=9)
fig.tight_layout(w_pad=2)
save(fig, 'u2_latest_performance')

with (OUT / 'data/u2_repeat_metrics.csv').open('w') as stream:
    rows = [dict(mode=mode, repeat=i, seconds=row['seconds'], rss_peak_bytes=row['rss_peak_bytes'],
                 **row['stage_timings'])
            for mode, report in [('baseline', baseline), ('candidate', candidate)]
            for i, row in enumerate(report['pipeline_runs'], 1)]
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'scientific_figures': 9, 'sources': len(data), 'retrieval_rows': len(records),
                  'pipeline_repeat_rows': len(rows), 'inference_runs': 0}))
