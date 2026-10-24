"""Prepare and validate independent visual-fact/OCR reviews, without inference.

Every claim, source hash and OCR prediction is immutable. Two reviewer files are
kept separate; their disagreements and unevaluable cases remain explicit. The
tool reports development metrics only and cannot promote an existing test set.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from core_modules.organize.visual_evidence import FIELDS, validate_observation
from tools.review_quality_metrics import character_errors, normalize_transcription

FACT_FIELDS = FIELDS[:-1]
DECISIONS = ('pending', 'supported', 'error', 'not_evaluable')
STATUSES = ('pending', 'reviewed', 'not_evaluable')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text_field(value, limit=4000):
    return isinstance(value, str) and len(value) <= limit


def validate_draft(template, draft):
    if not isinstance(draft, dict) or set(draft) != set(template):
        raise ValueError('fact_review_invalid_format')
    if any(draft[key] != value for key, value in template.items() if key not in ('reviewer', 'rows')):
        raise ValueError('fact_review_protocol_or_slot_changed')
    if not text_field(draft['reviewer'], 200):
        raise ValueError('fact_review_invalid_reviewer')
    rows = draft['rows']
    if not isinstance(rows, list) or len(rows) != len(template['rows']):
        raise ValueError('fact_review_count_changed')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('review_id'), str) or row['review_id'] in indexed:
            raise ValueError('fact_review_invalid_or_duplicate_id')
        indexed[row['review_id']] = row
    clean = []
    for original in template['rows']:
        row = indexed.get(original['review_id'])
        if row is None or set(row) != set(original) or any(row[key] != value for key, value in original.items() if key != 'review'):
            raise ValueError('fact_review_source_or_claim_changed')
        review = row['review']
        if not isinstance(review, dict) or set(review) != set(original['review']):
            raise ValueError('fact_review_invalid_review_fields')
        if review['status'] not in STATUSES or type(review['no_visible_text']) is not bool:
            raise ValueError('fact_review_invalid_status')
        if any(not text_field(review[key]) for key in ('source_ref', 'notes', 'reviewed_at')):
            raise ValueError('fact_review_invalid_note_or_timestamp')
        facts = review['facts']
        if not isinstance(facts, dict) or set(facts) != set(original['review']['facts']):
            raise ValueError('fact_review_fact_ids_changed')
        for assessment in facts.values():
            if not isinstance(assessment, dict) or set(assessment) != {'decision', 'duplicate', 'correction', 'notes'}:
                raise ValueError('fact_review_invalid_assessment')
            if assessment['decision'] not in DECISIONS or type(assessment['duplicate']) is not bool or any(
                    not text_field(assessment[key]) for key in ('correction', 'notes')):
                raise ValueError('fact_review_invalid_assessment_value')
            if assessment['decision'] == 'error' and not (assessment['correction'].strip() or assessment['notes'].strip()):
                raise ValueError('fact_review_error_needs_explanation')
        omissions = review['omissions']
        if not isinstance(omissions, dict) or set(omissions) != set(original['review']['omissions']) or any(
                not isinstance(values, list) or len(values) > 100 or any(not text_field(v) for v in values)
                for values in omissions.values()):
            raise ValueError('fact_review_invalid_omissions')
        if row['kind'] == 'visual':
            if review['reference_text'] is not None or review['no_visible_text']:
                raise ValueError('fact_review_transcription_on_visual_row')
            if review['status'] == 'reviewed' and any(value['decision'] == 'pending' for value in facts.values()):
                raise ValueError('fact_review_unfinished_facts')
        else:
            if not text_field(review['reference_text'], 10000):
                raise ValueError('fact_review_invalid_transcription')
            if review['status'] == 'reviewed' and (bool(normalize_transcription(review['reference_text'])) == review['no_visible_text']):
                raise ValueError('fact_review_empty_text_needs_no_visible_text_flag')
        if review['status'] != 'pending':
            if not draft['reviewer'].strip() or not review['source_ref'].strip():
                raise ValueError('fact_review_reviewer_and_source_required')
            try:
                timestamp = datetime.fromisoformat(review['reviewed_at'].replace('Z', '+00:00'))
                if timestamp.tzinfo is None:
                    raise ValueError()
            except ValueError as exc:
                raise ValueError('fact_review_timezone_timestamp_required') from exc
        clean.append(row)
    return clean


def evaluate_pair(template_a, draft_a, template_b, draft_b):
    rows_a = validate_draft(template_a, draft_a)
    rows_b = validate_draft(template_b, draft_b)
    if (template_a['slot'] != 'A' or template_b['slot'] != 'B' or
            {key: value for key, value in template_a.items() if key not in ('slot', 'rows', 'reviewer')} !=
            {key: value for key, value in template_b.items() if key not in ('slot', 'rows', 'reviewer')}):
        raise ValueError('fact_review_pair_binding_changed')
    if [row['review_id'] for row in rows_a] != [row['review_id'] for row in rows_b]:
        raise ValueError('fact_review_pair_alignment_changed')
    for a, b in zip(rows_a, rows_b):
        if {key: value for key, value in a.items() if key != 'review'} != {
                key: value for key, value in b.items() if key != 'review'}:
            raise ValueError('fact_review_pair_binding_changed')
    validate_unique_images(rows_a)
    active_a = any(row['review']['status'] != 'pending' for row in rows_a)
    active_b = any(row['review']['status'] != 'pending' for row in rows_b)
    name_a = draft_a['reviewer'].strip().casefold()
    name_b = draft_b['reviewer'].strip().casefold()
    if active_a and active_b and name_a == name_b:
        raise ValueError('fact_review_two_distinct_reviewers_required')
    consensus, disagreements, image_states = Counter(), [], Counter()
    per_reviewer = {'A': Counter(), 'B': Counter()}
    paired_evaluable_images = set()
    duplicates, omissions = Counter(), Counter()
    ocr = []
    visual_total = sum(row['kind'] == 'visual' for row in rows_a)
    for a, b in zip(rows_a, rows_b):
        if a['kind'] == 'ocr':
            entry = {'review_id': a['review_id'], 'status_a': a['review']['status'], 'status_b': b['review']['status']}
            if a['review']['status'] == b['review']['status'] == 'reviewed':
                reference_a = normalize_transcription(a['review']['reference_text'])
                reference_b = normalize_transcription(b['review']['reference_text'])
                if reference_a == reference_b:
                    entry.update(status='agreed', **character_errors(reference_a, a['predicted_text']))
                else:
                    entry.update(status='disagreement')
            else:
                entry['status'] = 'incomplete_or_not_evaluable'
            ocr.append(entry)
            continue
        paired = a['review']['status'] == b['review']['status'] == 'reviewed'
        image_states['paired_reviewed' if paired else 'incomplete_or_not_evaluable'] += 1
        for slot, row in (('A', a), ('B', b)):
            if row['review']['status'] == 'reviewed':
                per_reviewer[slot].update(value['decision'] for value in row['review']['facts'].values())
                duplicates[slot] += sum(value['duplicate'] for value in row['review']['facts'].values())
                omissions[slot] += sum(len(values) for values in row['review']['omissions'].values())
        for fact in a['facts']:
            left = a['review']['facts'][fact['fact_id']]['decision']
            right = b['review']['facts'][fact['fact_id']]['decision']
            if not paired:
                consensus['pending'] += 1
            elif left == right:
                consensus[left] += 1
                if left in ('supported', 'error'):
                    paired_evaluable_images.add(a['review_id'])
            else:
                consensus['disagreement'] += 1
                disagreements.append({'review_id': a['review_id'], 'fact_id': fact['fact_id'], 'A': left, 'B': right})
    evaluated = consensus['supported'] + consensus['error']
    error_ratio = consensus['error'] / evaluated if evaluated else None
    enough = len(paired_evaluable_images) >= 100 and not disagreements
    agreed_ocr = [row for row in ocr if row['status'] == 'agreed']
    edits = sum(row['edits'] for row in agreed_ocr)
    chars = sum(row['reference_characters'] for row in agreed_ocr)
    return {'status': 'validated', 'quality_gate': 'insufficient_data', 'final_test_locked': False,
            'distinct_active_reviewers': int(active_a) + int(active_b),
            'visual_images': visual_total, 'image_states': dict(image_states),
            'paired_evaluable_images': len(paired_evaluable_images), 'fact_counts': dict(consensus),
            'consensus_error_ratio': error_ratio, 'disagreements': disagreements,
            'per_reviewer_counts': {key: dict(value) for key, value in per_reviewer.items()},
            'duplicate_flags': dict(duplicates), 'omission_counts': dict(omissions),
            'development_fact_check': ('passed' if error_ratio <= .05 else 'failed') if enough and error_ratio is not None else 'insufficient_data',
            'ocr': {'images': len(ocr), 'agreed_transcriptions': len(agreed_ocr), 'edits': edits,
                    'reference_characters': chars, 'cer': edits / chars if chars else None, 'rows': ocr},
            'definitions': {'fact_denominator': 'supported + error among paired, agreed fact judgements; duplicates included',
                'not_evaluable_pending_and_disagreements': 'reported explicitly, never treated as supported',
                'transcription': 'NFC, CRLF/CR to LF, edge whitespace trimmed; interior whitespace retained',
                'OCR_prediction_order': 'recorded OCR line order, joined by LF; no reference-driven reordering'},
            'limitations': ['Reviewer names alone do not prove independent human work.',
                'Current images were used in development; final-test quality and Macro-F1 remain unevaluated.',
                'Unknown actor identities are not inferred from visual observations or OCR text.']}


def make_row(kind, image_id, source, checksum, observation):
    if not isinstance(image_id, str) or not image_id or len(image_id) > 256:
        raise ValueError('fact_review_invalid_image_id')
    review_id = kind + '-' + digest([kind, image_id])[:32]
    facts = [{'fact_id': digest([review_id, field, index, statement])[:32], 'field': field, 'statement': statement}
             for field in FACT_FIELDS for index, statement in enumerate(observation[field])] if kind == 'visual' else []
    return {'review_id': review_id, 'kind': kind, 'image_id': image_id,
            'source': source, 'source_sha256': checksum, 'preview': 'images/' + review_id + '.jpg',
            'facts': facts, 'uncertain': observation['uncertain'] if kind == 'visual' else [],
            'predicted_text': '\n'.join(line['text'] for line in observation['lines']) if kind == 'ocr' else None,
            'review': {'status': 'pending', 'source_ref': '', 'notes': '', 'reviewed_at': '',
                       'facts': {fact['fact_id']: {'decision': 'pending', 'duplicate': False, 'correction': '', 'notes': ''} for fact in facts},
                       'omissions': {field: [] for field in FACT_FIELDS} if kind == 'visual' else {},
                       'reference_text': '' if kind == 'ocr' else None, 'no_visible_text': False}}


def validate_unique_images(rows):
    # The same source may have both visual and OCR tasks, but duplicate files
    # within either task cannot inflate its sample size or error denominator.
    keys = [(row['kind'], row['source_sha256']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('fact_review_duplicate_image_content')


def verify_existing(root):
    root = root.resolve()
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    for name, sha in manifest['artifacts'].items():
        path = root / name
        if root not in path.resolve().parents or path.is_symlink() or fingerprint(path) != sha:
            raise ValueError('fact_review_artifact_changed:' + name)
    for row in manifest['sources']:
        if fingerprint(row['path']) != row['sha256']:
            raise ValueError('fact_review_source_changed')
    return {slot: json.loads((root / ('reviewer_' + slot + '_template.json')).read_text()) for slot in ('A', 'B')}


def prepare(observations, ocr, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('new_empty_output_required')
    for directory in (observations, ocr):
        if directory and (output.resolve() == directory.resolve() or directory.resolve() in output.resolve().parents):
            raise ValueError('fact_review_output_inside_existing_study')
    summary = json.loads((observations / 'summary.json').read_text())
    if summary.get('status') != 'passed':
        raise ValueError('completed_direct_observation_run_required')
    for name in ('input_manifest.json', 'results.json'):
        if fingerprint(observations / name) != summary['artifact_sha256'][name]:
            raise ValueError('observation_artifact_changed')
    results = json.loads((observations / 'results.json').read_text())
    rows, sources = [], []
    for name in ('summary.json', 'input_manifest.json', 'results.json'):
        path = observations / name
        sources.append({'path': str(path.resolve()), 'sha256': fingerprint(path)})
    for row in results:
        if not validate_observation(row.get('observation')):
            raise ValueError('valid_visual_observation_required')
        rows.append(make_row('visual', row['image_id'], row['neutral_path'], row['neutral_sha256'], row['observation']))
    if ocr:
        path = ocr / 'summary.json'
        ocr_summary = json.loads(path.read_text())
        if ocr_summary.get('status') != 'passed':
            raise ValueError('completed_OCR_run_required')
        sources.append({'path': str(path.resolve()), 'sha256': fingerprint(path)})
        for row in ocr_summary['results']:
            rows.append(make_row('ocr', row['image_id'], row['source'], row['sha256'], row['observation']))
    if len({row['review_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate_review_image')
    validate_unique_images(rows)
    for row in rows:
        source = Path(row['source']).resolve()
        if output.resolve() == source.parent or output.resolve() in source.parents or source.parent in output.resolve().parents:
            raise ValueError('fact_review_output_overlaps_sources')
        if fingerprint(source) != row['source_sha256']:
            raise ValueError('fact_review_image_changed')
        sources.append({'path': str(source), 'sha256': row['source_sha256']})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir()
    for row in rows:
        with Image.open(row['source']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((1280, 1280))
            image.save(output / row['preview'], format='JPEG', quality=90)
            row['preview_size'] = list(image.size)
        row['preview_sha256'] = fingerprint(output / row['preview'])
    binding = {'sources': sources, 'rows_sha256': digest(rows), 'fact_fields': list(FACT_FIELDS),
               'policy': 'dual_independent_development_fact_review_v1',
               'preparation_code_sha256': fingerprint(Path(__file__))}
    protocol = digest(binding)
    templates = {slot: {'schema_version': 1, 'protocol_sha256': protocol, 'slot': slot, 'reviewer': '',
                       'rows': rows, 'final_test_locked': False, 'human_decisions_applied': False} for slot in ('A', 'B')}
    for slot, template in templates.items():
        atomic_json(output / ('reviewer_' + slot + '_template.json'), template)
    atomic_json(output / 'protocol.json', binding)
    render(output, templates)
    for source in sources:
        if fingerprint(source['path']) != source['sha256']:
            raise ValueError('fact_review_input_changed_during_preparation')
    summary = {'status': 'prepared', 'protocol_sha256': protocol, 'rows': len(rows),
               'visual_images': sum(row['kind'] == 'visual' for row in rows),
               'ocr_images': sum(row['kind'] == 'ocr' for row in rows),
               'visual_claims': sum(len(row['facts']) for row in rows),
               'human_reviewed_images': 0, 'quality_gate': 'insufficient_data',
               'final_test_locked': False, 'human_decisions_applied': False}
    atomic_json(output / 'summary.json', summary)
    artifacts = {str(path.relative_to(output)): fingerprint(path) for path in output.rglob('*') if path.is_file()}
    atomic_json(output / 'artifact_manifest.json', {'sources': sources, 'artifacts': artifacts})
    return summary


def render(root, templates):
    encoded = base64.b64encode(json.dumps(templates, ensure_ascii=False).encode()).decode()
    document = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>画面事实与文字复核</title>
<style>body{margin:20px auto;padding:0 16px;max-width:1250px;font:16px/1.55 system-ui;background:#f6f5f2;color:#17212b}header,nav{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}button,input,select,textarea{font:inherit;padding:6px;max-width:100%;box-sizing:border-box}textarea{width:100%;min-height:60px}article{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;background:white;padding:18px;border:1px solid #ddd;border-radius:8px}.photo img{width:100%;height:auto}.photo{min-width:0}.fact{border-bottom:1px solid #ddd;padding:10px 0}.fact p{margin:4px 0;overflow-wrap:anywhere}.fact label{margin-left:10px}section{min-width:0}small{overflow-wrap:anywhere;color:#596574}#message{min-height:24px}h1{font-size:26px}@media(max-width:700px){article{display:block}.photo{margin-bottom:20px}body{padding:0 12px}h1{font-size:22px}}</style>
<h1>画面事实与文字复核</h1><p>两位复核人分别保存 A、B 文件，先独立对照图片，再交由评测工具比较。人物身份不由画面描述或文字确认。</p>
<header><select id="slot"><option>A</option><option>B</option></select><input id="reviewer" placeholder="当前复核人"><select id="kind"><option value="">全部</option><option value="visual">画面事实</option><option value="ocr">文字转录</option></select><button id="export">导出当前复核</button><label>恢复复核 <input id="import" type="file" accept="application/json"></label></header>
<p id="message"></p><nav><button id="prev">上一张</button><span id="page"></span><button id="next">下一张</button><button id="complete">本图已复核</button><button id="unevaluable">本图不可评价</button></nav><div id="card"></div>
<script>
const original=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(''' + json.dumps(encoded) + r'''),c=>c.charCodeAt(0)))), drafts={};let slot='A',page=0,storageMessage='';
const $=id=>document.getElementById(id),el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};const names={clothing:'服饰',props:'道具',setting:'布景',actions:'动作',visible_text:'可见文字'};
const viewKey=original.A.protocol_sha256+':fact-review-view';try{const view=JSON.parse(localStorage.getItem(viewKey));if(view&&original[view.slot]&&Number.isInteger(view.page)&&view.page>=0&&['','visual','ocr'].includes(view.kind)){slot=view.slot;page=view.page;$('slot').value=slot;$('kind').value=view.kind;}}catch(e){storageMessage='本机草稿位置读取失败，请导出当前复核。';}
function validate(value){const t=original[slot];if(!value||value.slot!==slot||value.protocol_sha256!==t.protocol_sha256||value.final_test_locked!==false||value.human_decisions_applied!==false||typeof value.reviewer!=='string'||!Array.isArray(value.rows)||value.rows.length!==t.rows.length)throw Error('复核版本或复核人组别不匹配');const rows=new Map(value.rows.map(r=>[r.review_id,r]));if(rows.size!==t.rows.length)throw Error('图片ID重复');for(const r of t.rows){const v=rows.get(r.review_id);if(!v||Object.keys(v).length!==Object.keys(r).length)throw Error('图片记录不完整');for(const k of Object.keys(r))if(k!=='review'&&JSON.stringify(v[k])!==JSON.stringify(r[k]))throw Error('图片或观察内容已改变');const a=v.review;if(!a||!['pending','reviewed','not_evaluable'].includes(a.status)||JSON.stringify(Object.keys(a).sort())!==JSON.stringify(Object.keys(r.review).sort())||typeof a.no_visible_text!=='boolean'||JSON.stringify(Object.keys(a.facts).sort())!==JSON.stringify(Object.keys(r.review.facts).sort()))throw Error('复核字段无效');for(const f of Object.values(a.facts))if(!['pending','supported','error','not_evaluable'].includes(f.decision)||typeof f.duplicate!=='boolean'||typeof f.correction!=='string'||typeof f.notes!=='string')throw Error('事实决定无效');for(const k of ['source_ref','notes','reviewed_at'])if(typeof a[k]!=='string')throw Error('复核说明无效');if(r.kind==='ocr'&&typeof a.reference_text!=='string')throw Error('转录无效');if(r.kind==='visual'&&(a.reference_text!==null||a.no_visible_text))throw Error('画面记录的转录字段无效');if(JSON.stringify(Object.keys(a.omissions).sort())!==JSON.stringify(Object.keys(r.review.omissions).sort())||Object.values(a.omissions).some(x=>!Array.isArray(x)||x.some(v=>typeof v!=='string')))throw Error('遗漏字段无效');}return value;}
const key=()=>original.A.protocol_sha256+':fact-review:'+slot;function save(){try{localStorage.setItem(key(),JSON.stringify(drafts[slot]));}catch(e){storageMessage='本机保存失败，请导出：'+e.message;$('message').textContent=storageMessage;}}
function init(){if(drafts[slot])return;drafts[slot]=structuredClone(original[slot]);try{const s=localStorage.getItem(key());if(s)drafts[slot]=validate(JSON.parse(s));}catch(e){$('message').textContent='旧草稿未加载：'+e.message;}}
const filtered=()=>drafts[slot].rows.filter(r=>!$('kind').value||r.kind===$('kind').value),current=()=>filtered()[page];
function status(){const rows=drafts[slot].rows,n=rows.filter(r=>r.review.status==='reviewed').length,x=rows.filter(r=>r.review.status==='not_evaluable').length;$('message').textContent=(storageMessage?storageMessage+' ':'')+'复核 '+slot+'：已完成 '+n+' / '+rows.length+'，不可评价 '+x+'；剩余 '+(rows.length-n-x);}
function render(){init();const rows=filtered();page=Math.max(0,Math.min(page,rows.length-1));try{localStorage.setItem(viewKey,JSON.stringify({slot,page,kind:$('kind').value}));}catch(e){storageMessage='本机保存失败，请导出当前复核。';}$('reviewer').value=drafts[slot].reviewer;$('page').textContent=(rows.length?page+1:0)+' / '+rows.length;$('prev').disabled=!page;$('next').disabled=page+1>=rows.length;const holder=$('card');holder.replaceChildren();status();const row=rows[page];if(!row)return;const article=el('article'),photo=el('div');photo.className='photo';const image=el('img');image.src=row.preview;image.alt='待复核图像';photo.append(image,el('small',row.review_id));const section=el('section');section.append(el('h2',row.kind==='visual'?'逐条画面事实':'独立文字转录'));
if(row.kind==='visual'){for(const [field,title]of Object.entries(names)){section.append(el('h3',title));for(const fact of row.facts.filter(f=>f.field===field)){const value=row.review.facts[fact.fact_id],card=el('div');card.className='fact';card.dataset.fact=fact.fact_id;card.append(el('p',fact.statement));const select=el('select');select.dataset.field='decision';for(const [v,t]of [['pending','待复核'],['supported','画面支持'],['error','事实错误'],['not_evaluable','无法评价']]){const o=el('option',t);o.value=v;select.append(o);}select.value=value.decision;card.append(select);const label=el('label'),duplicate=el('input');duplicate.type='checkbox';duplicate.dataset.field='duplicate';duplicate.checked=value.duplicate;label.append(duplicate,document.createTextNode('重复描述'));card.append(label);for(const [f,p]of [['correction','错误时填写正确内容'],['notes','判断说明']]){const input=el('input');input.dataset.field=f;input.placeholder=p;input.value=value[f];card.append(input);}section.append(card);}const omissions=el('textarea');omissions.dataset.omissions=field;omissions.placeholder='遗漏的可见内容，每行一条（可留空）';omissions.value=row.review.omissions[field].join('\n');section.append(omissions);}if(row.uncertain.length)section.append(el('small','模型不确定项：'+row.uncertain.join('；')));}
else{const reference=el('textarea');reference.id='reference-text';reference.placeholder='只对照图像逐字转录，按阅读顺序，每行一段';reference.value=row.review.reference_text;section.append(reference);const label=el('label'),none=el('input');none.id='no-visible-text';none.type='checkbox';none.checked=row.review.no_visible_text;label.append(none,document.createTextNode('图中没有可见文字'));section.append(label);}
for(const [field,p]of [['source_ref','复核依据，例如：对照本页原图'],['notes','整图说明、遮挡或分歧']]){const input=el(field==='notes'?'textarea':'input');input.dataset.review=field;input.placeholder=p;input.value=row.review[field];section.append(input);}section.append(el('small','当前状态：'+row.review.status));article.append(photo,section);holder.append(article);}
$('card').addEventListener('input',e=>{const row=current();if(!row)return;const fact=e.target.closest('[data-fact]'),field=e.target.dataset.field;if(fact&&field)row.review.facts[fact.dataset.fact][field]=field==='duplicate'?e.target.checked:e.target.value;else if(e.target.dataset.omissions)row.review.omissions[e.target.dataset.omissions]=e.target.value.split('\n').filter(v=>v.trim());else if(e.target.dataset.review)row.review[e.target.dataset.review]=e.target.value;else if(e.target.id==='reference-text')row.review.reference_text=e.target.value;else if(e.target.id==='no-visible-text')row.review.no_visible_text=e.target.checked;row.review.status='pending';row.review.reviewed_at='';save();status();});
$('reviewer').addEventListener('input',e=>{drafts[slot].reviewer=e.target.value;save();});$('slot').addEventListener('change',e=>{slot=e.target.value;page=0;render();});$('kind').addEventListener('change',()=>{page=0;render();});for(const[id,step]of [['prev',-1],['next',1]])$(id).addEventListener('click',()=>{page+=step;render();});
function complete(state){const row=current();if(!row)return;if(!drafts[slot].reviewer.trim()||!row.review.source_ref.trim()){$('message').textContent='请填写复核人和复核依据';return;}if(state==='reviewed'){if(row.kind==='visual'&&Object.values(row.review.facts).some(v=>v.decision==='pending'||(v.decision==='error'&&!v.correction.trim()&&!v.notes.trim()))){$('message').textContent='请完成每条事实，错误项需填写说明';return;}if(row.kind==='ocr'&&Boolean(row.review.reference_text.trim())===row.review.no_visible_text){$('message').textContent='请转录文字，或明确勾选没有可见文字';return;}}row.review.status=state;row.review.reviewed_at=new Date().toISOString();save();status();}
$('complete').addEventListener('click',()=>complete('reviewed'));$('unevaluable').addEventListener('click',()=>complete('not_evaluable'));$('export').addEventListener('click',()=>{save();const url=URL.createObjectURL(new Blob([JSON.stringify(drafts[slot],null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download='fact_review_'+slot+'.json';a.click();URL.revokeObjectURL(url);});$('import').addEventListener('change',async e=>{try{if(!e.target.files[0])return;drafts[slot]=structuredClone(validate(JSON.parse(await e.target.files[0].text())));save();render();}catch(error){$('message').textContent='恢复失败：'+error.message;}});render();
</script></html>'''
    (root / 'index.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--ocr', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true')
    parser.add_argument('--review-a', type=Path)
    parser.add_argument('--review-b', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.verify_existing or args.review_a or args.review_b:
        templates = verify_existing(args.output)
        if args.review_a or args.review_b:
            if not args.review_a or not args.review_b or not args.report or args.report.exists():
                raise ValueError('both_reviews_and_new_report_required')
            result = evaluate_pair(templates['A'], json.loads(args.review_a.read_text()),
                                   templates['B'], json.loads(args.review_b.read_text()))
            result['source_sha256'] = {str(path.resolve()): fingerprint(path) for path in
                                      (args.review_a, args.review_b, args.output / 'artifact_manifest.json')}
            result['evaluation_code_sha256'] = {name: fingerprint(ROOT / name) for name in
                ('tools/prepare_visual_fact_review.py', 'tools/review_quality_metrics.py')}
            atomic_json(args.report, result)
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(json.dumps({'status': 'verified', 'rows': len(templates['A']['rows']), 'human_labels_added': 0}))
    else:
        if not args.observations:
            raise ValueError('observations_required')
        print(json.dumps(prepare(args.observations, args.ocr, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
