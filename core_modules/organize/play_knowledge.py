# -*- coding: utf-8 -*-
"""将剧目信息.txt 和本地剧组/剧照图片说明规范化为剧目知识库。"""
import hashlib
import json
import os
import re
from pathlib import Path

from core_modules.organize.play_source import extract_urls, load_local_sources

ACT_RE = re.compile(r'^\s*第\s*([一二三四五六七八九十百\d]+)\s*幕\s*(.*)$')
ROLE_RE = re.compile(r'([\u4e00-\u9fffA-Za-z·]{2,20})\s*[（(]\s*饰\s*([^）)]+)[）)]')
ROLE_INLINE_RE = re.compile(r'([\u4e00-\u9fffA-Za-z·]{2,20})饰([\u4e00-\u9fffA-Za-z·]{2,20})')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def _hash_text(text):
    return 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest()


def chinese_number(value):
    if value.isdigit():
        return int(value)
    digits = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
              '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
              '百': 100}
    if value == '十':
        return 10
    if '十' in value:
        left, _, right = value.partition('十')
        return (digits.get(left, 1) if left else 1) * 10 + (digits.get(right, 0) if right else 0)
    return digits.get(value)


def _clean_text(text):
    text = text.replace('\ufeff', '').replace('\u3000', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def parse_plot_text(text, source_ref='local:剧目信息.txt'):
    """抽取剧情段落、幕次和角色，不改写原文。"""
    text = _clean_text(text)
    lines = [line.strip() for line in text.splitlines()]
    docs, scenes = [], []
    current = None
    intro = []
    for line in lines:
        if not line:
            continue
        if line.startswith('详见：'):
            continue
        m = ACT_RE.match(line)
        if m:
            if current:
                current['text'] = _clean_text(' '.join(current.pop('_lines')))
                current['content_hash'] = _hash_text(current['text'])
                current['keywords'] = _keywords(current['text'], current['title'])
                scenes.append(current)
            number = chinese_number(m.group(1))
            title = m.group(2).strip() or f'第{number}幕'
            current = {
                'scene_id': f'act-{number:02d}', 'act': number,
                'title': title, 'label': f'第{number}幕{title}',
                'source_ref': f'{source_ref}#act-{number:02d}', '_lines': []}
            continue
        if current is not None:
            current['_lines'].append(line)
        elif not line.startswith('剧目介绍：'):
            intro.append(line)
    if current:
        current['text'] = _clean_text(' '.join(current.pop('_lines')))
        current['content_hash'] = _hash_text(current['text'])
        current['keywords'] = _keywords(current['text'], current['title'])
        scenes.append(current)
    if intro:
        intro_text = _clean_text(' '.join(intro))
        docs.append({'kind': 'synopsis', 'text': intro_text, 'source_ref': source_ref,
                     'content_hash': _hash_text(intro_text), 'entities': {}})
    for scene in scenes:
        entities = {'acts': [scene['act']], 'scenes': [scene['title']],
                    'roles': sorted(_extract_roles(scene['text']))}
        docs.append({'kind': 'scene', 'text': scene['text'],
                     'source_ref': scene['source_ref'],
                     'content_hash': scene['content_hash'], 'entities': entities})
        scene['entities'] = entities
    roles = sorted(_extract_roles(text))
    return docs, scenes, roles


def _extract_roles(text):
    found = set()
    for actor, role in ROLE_RE.findall(text):
        found.add((actor.strip(), role.strip()))
    for actor, role in ROLE_INLINE_RE.findall(text):
        found.add((actor.strip(), role.strip()))
    return [{'actor': a, 'role': r} for a, r in sorted(found)]


def role_aliases(role):
    """角色别名：全长 + 去姓氏前缀（如 郑庄公→庄公），用于剧情文本匹配。"""
    aliases = {role}
    if len(role) >= 3:
        aliases.add(role[1:])
    return aliases


def _merge_cast(text_roles, media):
    """合并正文与剧组图片的（演员, 角色），返回 {角色: {演员, ...}}。"""
    cast = {}
    for item in text_roles:
        if item.get('role') and item.get('actor'):
            cast.setdefault(item['role'], set()).add(item['actor'])
    for record in media:
        for item in record.get('roles', []):
            if item.get('role') and item.get('actor'):
                cast.setdefault(item['role'], set()).add(item['actor'])
    return cast


def _fill_scene_roles(scenes, docs, cast):
    """把角色表映射回幕次文本，作为人物证据；不修改原文。"""
    doc_by_ref = {d.get('source_ref'): d for d in docs}
    for scene in scenes:
        text = scene.get('text', '')
        pairs = sorted(
            {(actor, role) for role, actors in cast.items()
             if any(alias in text for alias in role_aliases(role))
             for actor in actors})
        role_pairs = [{'actor': a, 'role': r} for a, r in pairs]
        entities = scene.setdefault('entities', {'acts': [scene['act']],
                                                 'scenes': [scene['title']]})
        entities['roles'] = role_pairs
        doc = doc_by_ref.get(scene.get('source_ref'))
        if doc is not None:
            doc.setdefault('entities', {})['roles'] = role_pairs


def _keywords(text, title=''):
    words = set(re.findall(r'[\u4e00-\u9fff]{2,8}', f'{title} {text}'))
    stop = {'这是', '一个', '我们', '自己', '他们', '最终', '不禁', '因此', '原来', '时候'}
    return sorted(w for w in words if w not in stop)[:40]


def _media_kind(path):
    parts = set(Path(path).parts)
    name = os.path.basename(path)
    if '主演' in parts or '演员' in parts or '剧组信息' in parts or '剧组' in parts:
        return 'cast' if ('饰' in name or '（饰' in name) else 'image_caption'
    if '剧照' in parts:
        return 'image_caption'
    return 'photo'


def _media_record(path, play_root, scenes):
    rel = os.path.relpath(path, play_root).replace(os.sep, '/')
    caption = os.path.splitext(os.path.basename(path))[0].strip()
    kind = _media_kind(path)
    acts = []
    for scene in scenes:
        terms = [scene['title'], scene['label']] + scene.get('keywords', [])
        if any(term and term in caption for term in terms):
            acts.append(scene['act'])
    roles = _extract_roles(caption)
    if kind == 'image_caption' and not roles:
        # 剧组信息图片通常是官方剧情动作说明，可直接作为证据文本。
        roles = _extract_roles(caption)
    return {'kind': kind, 'path': rel, 'caption': caption,
            'source_ref': f'local-media:{rel}', 'acts': sorted(set(acts)),
            'roles': roles, 'content_hash': _hash_text(caption)}


def build_knowledge(info_path, play_root, output_dir=None):
    """构建剧目知识库；返回完整 dict，output_dir 非空时同时写 JSONL/JSON。"""
    local = load_local_sources(info_path, play_root)
    urls = extract_urls(info_path)
    docs, scenes, roles = parse_plot_text(local['text'])
    media = [_media_record(p, play_root, scenes) for p in local['image_paths']]
    cast = _merge_cast(roles, media)
    _fill_scene_roles(scenes, docs, cast)
    roles = [{'actor': a, 'role': role}
             for role in sorted(cast) for a in sorted(cast[role])]
    # 将媒体图注作为独立证据文档，便于后续图文召回。
    for item in media:
        if item['kind'] in ('image_caption', 'cast'):
            docs.append({'kind': item['kind'], 'text': item['caption'],
                         'source_ref': item['source_ref'],
                         'content_hash': item['content_hash'],
                         'entities': {'acts': item['acts'], 'roles': item['roles']}})
    knowledge = {
        'play_root': os.path.abspath(play_root),
        'info_path': os.path.abspath(info_path),
        'urls': urls,
        'parser': 'play_knowledge/v1',
        'scenes': scenes,
        'roles': roles,
        'role_aliases': {role: sorted(role_aliases(role)) for role in sorted(cast)},
        'documents': docs,
        'media': media,
        'stats': {'n_scenes': len(scenes), 'n_documents': len(docs),
                  'n_media': len(media), 'n_urls': len(urls)},
    }
    if output_dir:
        write_knowledge(output_dir, knowledge)
    return knowledge


def write_knowledge(output_dir, knowledge):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'documents.jsonl'), 'w', encoding='utf-8') as f:
        for row in knowledge['documents']:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(os.path.join(output_dir, 'media.jsonl'), 'w', encoding='utf-8') as f:
        for row in knowledge['media']:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(os.path.join(output_dir, 'entities.json'), 'w', encoding='utf-8') as f:
        json.dump({'scenes': knowledge['scenes'], 'roles': knowledge['roles'],
                   'role_aliases': knowledge.get('role_aliases', {})},
                  f, ensure_ascii=False, indent=2)
    report = {'parser': knowledge['parser'], 'urls': knowledge['urls'],
              'stats': knowledge['stats']}
    with open(os.path.join(output_dir, 'knowledge_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
