"""Prepare face-box annotation from completed checkpoints without model inference.

Existing image-level names never label a particular face. Human decisions are
validated and reported separately; this tool cannot lock a final test or write
the official gallery/labels.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import sys

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.identity_registry import IdentityRegistry
from core_modules.organize.task_artifacts import atomic_json, copy_verified, fingerprint

DECISIONS = ('pending', 'known', 'unknown', 'not_evaluable')
EDITABLE = ('decision', 'identity_id', 'exception', 'reviewer', 'source_ref', 'notes', 'reviewed_at')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_decisions(template, draft):
    if not isinstance(draft, dict) or draft.get('protocol_sha256') != template['protocol_sha256']:
        raise ValueError('face_review_protocol_mismatch')
    if draft.get('final_test_locked', False) is not False or draft.get('human_decisions_applied', False) is not False:
        raise ValueError('face_review_cannot_promote_or_lock_test')
    rows = draft.get('faces')
    if not isinstance(rows, list) or len(rows) != len(template['faces']):
        raise ValueError('face_review_count_mismatch')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('face_id') in indexed:
            raise ValueError('face_review_duplicate_or_invalid_id')
        indexed[row.get('face_id')] = row
    identities = {row['identity_id'] for row in template['identities']}
    clean = []
    for original in template['faces']:
        row = indexed.get(original['face_id'])
        if row is None or set(row) != set(original) or any(row[k] != v for k, v in original.items() if k not in EDITABLE):
            raise ValueError('face_review_binding_changed')
        if row['decision'] not in DECISIONS or type(row['exception']) is not bool:
            raise ValueError('face_review_invalid_decision')
        if row['decision'] == 'known':
            if row['identity_id'] not in identities:
                raise ValueError('face_review_unknown_identity_id')
        elif row['identity_id'] is not None:
            raise ValueError('face_review_identity_requires_known')
        for key in ('reviewer', 'source_ref', 'notes', 'reviewed_at'):
            if not isinstance(row[key], str) or len(row[key]) > 4000:
                raise ValueError('face_review_invalid_' + key)
        if row['decision'] != 'pending':
            if not row['reviewer'].strip() or not row['source_ref'].strip():
                raise ValueError('face_review_reviewer_and_source_required')
            try:
                timestamp = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
                if timestamp.tzinfo is None:
                    raise ValueError()
            except ValueError as exc:
                raise ValueError('face_review_timezone_timestamp_required') from exc
        clean.append(dict(original, **{key: row[key] for key in EDITABLE}))
    return clean


def verify_existing(root):
    root = root.resolve()
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    for name, checksum in manifest['artifacts'].items():
        path = root / name
        if path.is_symlink() or root not in path.resolve().parents or fingerprint(path) != checksum:
            raise ValueError('face_review_artifact_changed: ' + name)
    for row in manifest['sources']:
        if fingerprint(row['path']) != row['sha256']:
            raise ValueError('face_review_input_changed')
    return json.loads((root / 'review_template.json').read_text())


def render(root, template):
    data = json.dumps(template, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>逐脸身份复核</title><style>
body{font:16px/1.5 system-ui;max-width:1200px;margin:20px auto;padding:0 16px;background:#f5f4f1;color:#17212b}
button,input,select,textarea{font:inherit;padding:6px;max-width:100%;box-sizing:border-box}header{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
article{background:white;padding:16px;margin:18px 0;border:1px solid #ddd;border-radius:8px}figure{margin:0;position:relative;max-width:720px}figure img{display:block;width:100%}
.box{position:absolute;border:2px solid #ffdb35;box-sizing:border-box;color:#111;background:#ffdb3515}.box b{background:#ffdb35;font-size:12px}
.faces{display:flex;flex-wrap:wrap;gap:12px;margin-top:14px}.face{border:1px solid #ccc;padding:10px;width:260px;max-width:100%;box-sizing:border-box}
.face img{width:120px;height:120px;object-fit:contain;background:#111}.face input,.face select,.face textarea{display:block;width:100%;margin:6px 0}
small{display:block;overflow-wrap:anywhere}textarea{min-height:56px}#message{min-height:24px}nav{display:flex;gap:12px;align-items:center}</style>
<h1>逐脸身份复核</h1><p>依据原图和人脸裁剪逐脸判断。文件名人物列表不代表所有可见人脸；未列名不能直接标为未知。未知必须有不属于所选身份集合的依据；看不清身份时选择不可评价。镜像或重复检测请单独勾选，并说明依据。</p>
<p>草稿保存在当前浏览器，可导出 JSON。当前图片已用于开发实验；人工标注后仍需独立划分和图库来源核验。此页面不会修改图库或锁定测试集。</p>
<header><input id="reviewer" placeholder="复核人"><select id="play"><option value="">全部剧目</option></select><label><input id="pending" type="checkbox">只看未完成</label><button id="export">导出标注</button><label>恢复标注 <input id="import" type="file" accept="application/json"></label></header>
<p id="message" role="status"></p><nav><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></nav><main id="cards"></main>
<script id="data" type="application/json">''' + data + '''</script><script>
const original=JSON.parse(document.getElementById('data').textContent), key='face-identity-review:'+original.protocol_sha256;
let draft=structuredClone(original), page=0;const size=12, message=document.getElementById('message');
function valid(value){if(value.protocol_sha256!==original.protocol_sha256||!Array.isArray(value.faces)||value.faces.length!==original.faces.length)throw Error('版本或人脸数不匹配');const rows=new Map(value.faces.map(r=>[r.face_id,r]));if(rows.size!==original.faces.length)throw Error('人脸ID重复');for(const face of original.faces){const row=rows.get(face.face_id);if(!row||Object.keys(row).sort().join()!=Object.keys(face).sort().join()||Object.keys(face).some(k=>!original.editable_fields.includes(k)&&JSON.stringify(row[k])!==JSON.stringify(face[k])))throw Error('人脸绑定被修改');if(!['pending','known','unknown','not_evaluable'].includes(row.decision)||typeof row.exception!=='boolean')throw Error('决定无效');if(row.identity_id!==null&&!original.identities.some(i=>i.identity_id===row.identity_id))throw Error('身份无效');for(const k of ['reviewer','source_ref','notes','reviewed_at'])if(typeof row[k]!=='string'||row[k].length>4000)throw Error('字段格式无效');}return value;}
try{const saved=localStorage.getItem(key);if(saved)draft=valid(JSON.parse(saved));}catch(e){message.textContent='恢复失败：'+e.message;}
function save(){try{localStorage.setItem(key,JSON.stringify(draft));}catch(e){message.textContent='草稿保存失败，请导出：'+e.message;}}
const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;};
for(const title of [...new Set(original.images.map(r=>r.play_title))].sort()){const o=el('option',title);o.value=title;document.getElementById('play').append(o);}
function render(){const title=document.getElementById('play').value,pending=document.getElementById('pending').checked,byImage=new Map();for(const f of draft.faces){if(!byImage.has(f.image_id))byImage.set(f.image_id,[]);byImage.get(f.image_id).push(f);}const images=original.images.filter(i=>(!title||i.play_title===title)&&(!pending||(byImage.get(i.image_id)||[]).some(f=>f.decision==='pending')));page=Math.max(0,Math.min(page,Math.ceil(images.length/size)-1));const cards=document.getElementById('cards');cards.replaceChildren();document.getElementById('page').textContent=(images.length?page+1:0)+' / '+Math.ceil(images.length/size);document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=(page+1)*size>=images.length;
const counts={};for(const f of draft.faces)counts[f.decision]=(counts[f.decision]||0)+1;message.textContent='共 '+draft.faces.length+' 张脸：待核查 '+(counts.pending||0)+'，已知 '+(counts.known||0)+'，未知 '+(counts.unknown||0)+'，不可评价 '+(counts.not_evaluable||0);
for(const image of images.slice(page*size,(page+1)*size)){const article=el('article'),faces=byImage.get(image.image_id)||[];article.append(el('b',image.play_title),el('small',image.source));const figure=el('figure'),photo=el('img');photo.src=image.preview;photo.alt='原图位置参考';figure.append(photo);for(const face of faces){const b=el('span');b.className='box';const [x,y,w,h]=face.preview_box;b.style.left=x*100+'%';b.style.top=y*100+'%';b.style.width=w*100+'%';b.style.height=h*100+'%';b.append(el('b',String(face.face_index)));figure.append(b);}article.append(figure);if(!faces.length)article.append(el('p','本次检测无符合面积门槛的人脸；不表示图中没有人脸。'));const grid=el('div');grid.className='faces';for(const face of faces){const card=el('div');card.className='face';card.dataset.face=face.face_id;const crop=el('img');crop.src=face.crop;crop.alt='人脸 '+face.face_index;card.append(crop,el('small','人脸 '+face.face_index));const select=el('select');select.dataset.field='decision';for(const [v,t] of [['pending','待核查'],['known','已知身份'],['unknown','未知身份'],['not_evaluable','不可评价']]){const o=el('option',t);o.value=v;select.append(o);}select.value=face.decision;card.append(select);const identities=el('select');identities.dataset.field='identity_id';const blank=el('option','选择已知演员');blank.value='';identities.append(blank);for(const i of original.identities){const o=el('option',i.actor_name+' · '+i.scope_title);o.value=i.identity_id;identities.append(o);}identities.value=face.identity_id||'';card.append(identities);const label=el('label'),exception=el('input');exception.type='checkbox';exception.dataset.field='exception';exception.checked=face.exception;label.append(exception,document.createTextNode('镜像／重复检测例外'));card.append(label);for(const [f,p] of [['source_ref','确认依据（完成时必填）'],['notes','遮挡、浓妆、暗光、侧脸、分歧说明']]){const n=el(f==='notes'?'textarea':'input');n.dataset.field=f;n.placeholder=p;n.value=face[f];card.append(n);}card.append(el('small',face.reviewer?'复核人：'+face.reviewer:'尚未记录复核人'));grid.append(card);}article.append(grid);cards.append(article);}}
document.getElementById('cards').addEventListener('change',e=>{const card=e.target.closest('[data-face]'),field=e.target.dataset.field;if(!card||!field)return;const f=draft.faces.find(r=>r.face_id===card.dataset.face);f[field]=field==='exception'?e.target.checked:field==='identity_id'?(e.target.value||null):e.target.value;if(field==='identity_id'&&f.identity_id)f.decision='known';if(f.decision!=='known')f.identity_id=null;f.reviewer=document.getElementById('reviewer').value;f.reviewed_at=new Date().toISOString();save();queueMicrotask(render);});
document.getElementById('reviewer').value=draft.reviewer||'';document.getElementById('reviewer').addEventListener('change',e=>{draft.reviewer=e.target.value;save();});for(const id of ['play','pending'])document.getElementById(id).addEventListener('change',()=>{page=0;render();});for(const [id,step] of [['prev',-1],['next',1]])document.getElementById(id).addEventListener('click',()=>{page+=step;render();});
document.getElementById('export').addEventListener('click',()=>{save();const url=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='face_identity_decisions.json';a.click();URL.revokeObjectURL(url);});
document.getElementById('import').addEventListener('change',async e=>{try{if(!e.target.files[0])return;const incoming=valid(JSON.parse(await e.target.files[0].text()));draft=structuredClone(original);const rows=new Map(incoming.faces.map(r=>[r.face_id,r]));for(const f of draft.faces)for(const k of original.editable_fields)f[k]=rows.get(f.face_id)[k];draft.reviewer=incoming.reviewer||'';document.getElementById('reviewer').value=draft.reviewer;save();render();}catch(error){message.textContent='恢复失败：'+error.message;}});render();
</script></html>'''
    (root / 'index.html').write_text(document, encoding='utf-8')


def prepare(task_db, task_id, input_manifest, photo_review, registry_file, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('new_empty_output_required')
    inputs = json.loads(input_manifest.read_text())['images']
    for row in inputs:
        for name in ('source', 'path'):
            source_parent = Path(row[name]).resolve().parent
            if output.resolve() == source_parent or output.resolve() in source_parent.parents or source_parent in output.resolve().parents:
                raise ValueError('face_review_output_inside_photo_source')
    catalog = json.loads((photo_review / 'images.json').read_text())
    by_hash = {row['content_hash']: row for row in catalog}
    if len(by_hash) != len(catalog):
        raise ValueError('duplicate_photo_review_content')
    with sqlite3.connect(task_db.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.execute('BEGIN')
        task = connection.execute('SELECT kind,status FROM tasks WHERE id=?', (task_id,)).fetchone()
        if not task or task != ('face_cluster', 'completed'):
            raise ValueError('completed_face_cluster_required')
        rows = [json.loads(raw) for (raw,) in connection.execute(
            "SELECT result FROM items WHERE task_id=? AND status='succeeded' AND json_extract(input,'$.stage')='faces' ORDER BY position", (task_id,))]
        crops = {row['image']: row['crops'] for row in (json.loads(raw) for (raw,) in connection.execute(
            "SELECT result FROM items WHERE task_id=? AND status='succeeded' AND json_extract(input,'$.stage')='crops'", (task_id,)))}
        checkpoints = {name: json.loads(raw) for name, raw in connection.execute('SELECT name,payload FROM checkpoints WHERE task_id=?', (task_id,))}
    hashes = checkpoints['input_fingerprints']['hashes']
    mapping = {row['path']: row for row in inputs}
    if len(mapping) != len(inputs) or {row['image'] for row in rows} != set(mapping):
        raise ValueError('checkpoint_input_manifest_mismatch')
    registry_data = json.loads(registry_file.read_text())
    registry = IdentityRegistry(registry_data)
    identities = [{'identity_id': row['identity_id'], 'actor_name': row['actor_name'],
                   'scope_title': registry.plays.get(row.get('identity_scope'), '已核验身份')}
                  for row in registry_data['identities'] if row['verified']]
    source_rows = [{'path': str(input_manifest.resolve()), 'sha256': fingerprint(input_manifest)},
                   {'path': str((photo_review / 'images.json').resolve()), 'sha256': fingerprint(photo_review / 'images.json')},
                   {'path': str(registry_file.resolve()), 'sha256': fingerprint(registry_file)}]
    binding = {'task_id': task_id, 'detection_digest': digest(rows), 'crop_digest': digest(crops),
               'execution_contract': checkpoints['execution_contract'], 'inputs': list(source_rows),
               'policy': 'development_face_review_v1_no_automatic_labels_or_test_lock',
               'preparation_code_sha256': fingerprint(Path(__file__))}
    protocol = digest(binding)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'run_state.json', {'status': 'preparing', 'protocol_sha256': protocol})
    images, faces = [], []
    for row in rows:
        entry = mapping[row['image']]
        checksum = entry['sha256']
        if hashes[row['image']] != checksum or checksum not in by_hash:
            raise ValueError('face_review_source_binding_mismatch')
        source = by_hash[checksum]
        for path in (entry['source'], entry['path']):
            if fingerprint(path) != checksum:
                raise ValueError('face_review_source_changed')
            source_rows.append({'path': path, 'sha256': checksum})
        preview = photo_review / source['preview']
        if preview.is_symlink() or photo_review.resolve() not in preview.resolve().parents:
            raise ValueError('face_review_preview_path')
        relative = 'images/' + source['image_id'] + '.jpg'
        copy_verified(preview, output / relative, source['preview_sha256'])
        source_rows.append({'path': str(preview.resolve()), 'sha256': source['preview_sha256']})
        images.append({'image_id': source['image_id'], 'source_sha256': checksum, 'source': entry['source'],
                       'preview': relative, 'play_title': source['play_title']})
        width, height = source['width'], source['height']
        scale = min(1, 1920 / max(width, height))
        frame_width, frame_height = int(width * scale), int(height * scale)
        context_frame = None
        for member in row['members']:
            index = member['face_index']
            face_id = 'face-' + digest([checksum, index])[:32]
            if member['img_path'] != row['image']:
                raise ValueError('face_review_member_source_mismatch')
            box = member['bbox']
            if len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('face_review_invalid_box')
            crop = crops.get(row['image'], {}).get(str(index))
            crop_kind = 'checkpoint_aligned'
            if crop is None:
                # Historical collage checkpoints retain at most 24 faces per
                # cluster. For the rest, crop the recorded box without any
                # new detection or embedding; disclose its unaligned form.
                if context_frame is None:
                    with Image.open(entry['path']) as picture:
                        context_frame = ImageOps.exif_transpose(picture).convert('RGB').resize((frame_width, frame_height))
                region = context_frame.crop((max(0, box[0]), max(0, box[1]),
                                             min(frame_width, box[2]), min(frame_height, box[3])))
                padded = ImageOps.pad(region, (160, 160), color=(24, 24, 24))
                stream = io.BytesIO(); padded.save(stream, format='PNG')
                content = stream.getvalue()
                crop = {'png': base64.b64encode(content).decode(), 'sha256': hashlib.sha256(content).hexdigest()}
                crop_kind = 'recorded_bbox_unaligned'
            content = base64.b64decode(crop['png'], validate=True)
            if hashlib.sha256(content).hexdigest() != crop['sha256']:
                raise ValueError('face_review_crop_corrupt')
            with Image.open(io.BytesIO(content)) as picture:
                if picture.size != (160, 160) or picture.format != 'PNG':
                    raise ValueError('face_review_crop_invalid')
                picture.verify()
            relative = 'faces/' + face_id + '.png'
            destination = output / relative
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(content)
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(frame_width, box[2]), min(frame_height, box[3])
            if x2 <= x1 or y2 <= y1:
                raise ValueError('face_review_box_outside_frame')
            faces.append({'face_id': face_id, 'image_id': source['image_id'], 'source_sha256': checksum,
                          'face_index': index, 'bbox': box, 'coordinate_frame': [frame_width, frame_height],
                          'preview_box': [x1 / frame_width, y1 / frame_height, (x2-x1) / frame_width, (y2-y1) / frame_height],
                          'crop': relative, 'crop_sha256': crop['sha256'], 'crop_kind': crop_kind, 'decision': 'pending',
                          'identity_id': None, 'exception': False, 'reviewer': '', 'source_ref': '', 'notes': '', 'reviewed_at': ''})
        if context_frame is not None:
            context_frame.close()
    if len({row['face_id'] for row in faces}) != len(faces):
        raise ValueError('face_review_duplicate_faces')
    template = {'schema_version': 1, 'protocol_sha256': protocol, 'binding': binding,
                'editable_fields': EDITABLE, 'identities': identities, 'images': images, 'faces': faces,
                'reviewer': '', 'final_test_locked': False, 'human_decisions_applied': False}
    atomic_json(output / 'review_template.json', template)
    render(output, template)
    summary = {'status': 'prepared_pending_human_review', 'images': len(images), 'faces': len(faces),
               'known_identities_available': len(identities), 'human_labeled_faces': 0,
               'crop_counts': dict(Counter(row['crop_kind'] for row in faces)),
               'quality_gate': 'insufficient_data', 'model_inferences': 0, 'final_test_locked': False,
               'limitations': ['Detected faces passing the historical 1600-pixel area filter only; missed/small faces not added.',
                              'Aligned checkpoint crops are reused; faces omitted by collage cap use recorded-box context crops.',
                              'Coordinates refer to the existing EXIF-oriented OpenCV frame scaled to max side 1920.',
                              'Listed actors never assign a face identity; all face labels initially pending.',
                              'All images are development-exposed; gallery origins and known/unknown isolation require review.']}
    atomic_json(output / 'summary.json', summary)
    for source in source_rows:
        if fingerprint(source['path']) != source['sha256']:
            raise ValueError('face_review_input_changed_during_preparation')
    atomic_json(output / 'artifact_manifest.json', {'sources': source_rows, 'artifacts': {
        str(path.relative_to(output)): fingerprint(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'run_state.json'}})
    atomic_json(output / 'run_state.json', {'status': summary['status'], 'protocol_sha256': protocol})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-db', type=Path)
    parser.add_argument('--task-id')
    parser.add_argument('--input-manifest', type=Path)
    parser.add_argument('--photo-review', type=Path)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true')
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--decision-report', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.verify_existing or args.decisions:
        template = verify_existing(root)
        result = {'status': 'read_only_verification_passed', 'faces': len(template['faces'])}
        if args.decisions:
            if not args.decision_report or args.decision_report.exists() or root in args.decision_report.resolve().parents:
                raise ValueError('new_external_decision_report_required')
            rows = validate_decisions(template, json.loads(args.decisions.read_text()))
            result = {'status': 'human_decisions_validated_not_applied', 'counts': dict(Counter(r['decision'] for r in rows)),
                      'mirror_or_duplicate_exceptions': sum(r['exception'] for r in rows),
                      'protocol_sha256': template['protocol_sha256'], 'decisions_sha256': fingerprint(args.decisions),
                      'quality_gate': 'not_evaluated', 'final_test_locked': False}
            atomic_json(args.decision_report, result)
    else:
        if any(value is None for value in (args.task_db, args.task_id, args.input_manifest, args.photo_review, args.registry)):
            parser.error('prepare requires task-db, task-id, input-manifest, photo-review and registry')
        protected = [args.task_db.resolve().parent, args.photo_review.resolve(), args.registry.resolve().parent]
        if any(root == p or p in root.parents or root in p.parents for p in protected):
            raise ValueError('face_review_output_must_be_independent')
        result = prepare(args.task_db, args.task_id, args.input_manifest, args.photo_review, args.registry, root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
