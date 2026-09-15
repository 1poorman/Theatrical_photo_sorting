# -*- coding: utf-8 -*-
"""review_queue - 场景标签人工复核、修订日志与最终标签导出。

约定：
  - 队列与修订日志均为 JSONL，追加/原子写，任意时刻可回放。
  - 人工已确认（accept/modify/reject）的记录标记锁定，重跑 seed 不得覆盖。
  - 未审核或 ``unknown`` 不进入最终标签，由 SmartOrganizer 保留占位。
"""
import json
import os
from datetime import datetime, timezone

ACTIONS = ('accept', 'modify', 'reject')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


class ReviewQueue:
    def __init__(self, root):
        self.root = root
        self.queue_path = os.path.join(root, 'review_queue.jsonl')
        self.revisions_path = os.path.join(root, 'revisions.jsonl')
        self.final_path = os.path.join(root, 'scene_labels.json')

    # ---------- 读取 ----------

    def records(self):
        return {r['image']: r for r in _read_jsonl(self.queue_path)}

    def is_locked(self, record):
        review = record.get('review') or {}
        return bool(review.get('reviewer')) and review.get('status') in ('approved', 'rejected')

    # ---------- 写入 ----------

    def seed(self, reasoner_records):
        """并入候选记录；已人工确认的图片保持原样。返回统计。"""
        existing = self.records()
        added = kept = skipped = 0
        for rec in reasoner_records:
            image = rec['image']
            if image in existing:
                kept += 1
                continue
            rec = dict(rec)
            rec.setdefault('review', {'status': 'pending', 'reviewer': None,
                                      'updated_at': None})
            existing[image] = rec
            added += 1
        ordered = [existing[k] for k in sorted(existing)]
        _atomic_write_jsonl(self.queue_path, ordered)
        return {'added': added, 'kept': kept, 'skipped': skipped,
                'total': len(ordered)}

    def list_pending(self, status=None, limit=50, offset=0):
        rows = [r for r in self.records().values()
                if status is None or (r.get('review') or {}).get('status') == status]
        rows.sort(key=lambda r: r['image'])
        return rows[offset:offset + limit]

    def decide(self, image, action, scene_id=None, label=None,
               reviewer='human', note=''):
        """接受/修改/拒绝一条记录，同时追加不可变修订日志。"""
        if action not in ACTIONS:
            raise ValueError(f'未知动作: {action}（应为 {ACTIONS}）')
        records = self.records()
        if image not in records:
            raise KeyError(f'待审核记录不存在: {image}')
        record = records[image]
        before = {'scene_id': record.get('scene_id'), 'label': record.get('label'),
                  'review_status': (record.get('review') or {}).get('status')}

        if action == 'accept':
            record['review'] = {'status': 'approved', 'reviewer': reviewer,
                                'updated_at': _now(), 'action': action, 'note': note,
                                'original_label': before['label']}
        elif action == 'modify':
            if not label and not scene_id:
                raise ValueError('modify 需要提供 label 或 scene_id')
            record['scene_id'] = scene_id or label
            record['label'] = label or scene_id
            record['decision'] = 'human'
            record['review'] = {'status': 'approved', 'reviewer': reviewer,
                                'updated_at': _now(), 'action': action, 'note': note,
                                'original_label': before['label']}
        else:  # reject
            record['scene_id'] = 'unknown'
            record['label'] = 'unknown'
            record['review'] = {'status': 'rejected', 'reviewer': reviewer,
                                'updated_at': _now(), 'action': action, 'note': note,
                                'original_label': before['label']}

        records[image] = record
        _atomic_write_jsonl(self.queue_path, [records[k] for k in sorted(records)])
        after = {'scene_id': record.get('scene_id'), 'label': record.get('label'),
                 'review_status': record['review']['status']}
        with open(self.revisions_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': _now(), 'image': image, 'action': action,
                                'reviewer': reviewer, 'note': note,
                                'before': before, 'after': after},
                               ensure_ascii=False) + '\n')
        return record

    # ---------- 导出 ----------

    def final_labels(self):
        """返回人工已接受/修改且非 unknown 的最终标签。"""
        out = {}
        for record in self.records().values():
            review = record.get('review') or {}
            if review.get('status') == 'approved' and record.get('label') not in (None, 'unknown'):
                out[record['image']] = record['label']
        return out

    def export_final(self, path=None):
        path = path or self.final_path
        labels = self.final_labels()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        return path, len(labels)

    def summary(self):
        counts = {'pending': 0, 'approved': 0, 'rejected': 0}
        for record in self.records().values():
            status = (record.get('review') or {}).get('status', 'pending')
            counts[status] = counts.get(status, 0) + 1
        return counts

    def revisions(self):
        return _read_jsonl(self.revisions_path)


def load_scene_evidence(path):
    """读取已审核场景标签，供 SmartOrganizer 覆盖占位场景名。

    支持两种输入：
      - 复核队列 JSONL：只采纳 ``review.status == 'approved'`` 且非 unknown 的记录；
      - 人工标签 JSON：``{键: 标签}``，键为图片名/相对路径或 ``scene-NN`` 占位。

    Returns:
        dict: {``by_image``: {basename: label}, ``by_scene``: {scene_id: label}}
    """
    by_image, by_scene = {}, {}
    if not path or not os.path.exists(path):
        return {'by_image': by_image, 'by_scene': by_scene}
    if path.endswith('.jsonl'):
        for record in _read_jsonl(path):
            review = record.get('review')
            if review is not None and review.get('status') != 'approved':
                continue
            label = record.get('label')
            image = record.get('image')
            if label and label != 'unknown' and image:
                by_image[os.path.basename(image)] = label
        return {'by_image': by_image, 'by_scene': by_scene}
    with open(path, encoding='utf-8') as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        return {'by_image': by_image, 'by_scene': by_scene}
    for key, label in mapping.items():
        if not label or label == 'unknown':
            continue
        if isinstance(key, str) and key.startswith('scene-'):
            by_scene[key] = label
        else:
            by_image[os.path.basename(key)] = label
    return {'by_image': by_image, 'by_scene': by_scene}

