# -*- coding: utf-8 -*-
"""review_queue - 场景标签人工复核、修订日志与最终标签导出。

约定：
  - SQLite 事务保存队列与修订，JSONL 保留为兼容快照与导入/导出。
  - 人工已确认（accept/modify/reject）的记录标记锁定，重跑 seed 不得覆盖。
  - 未审核或 ``unknown`` 不进入最终标签，由 SmartOrganizer 保留占位。
"""
import json
import os
import hashlib
import tempfile
from datetime import datetime, timezone
from .review_store import ReviewStore, ReviewConflict

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
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=os.path.dirname(os.path.abspath(path)),
                                         delete=False) as f:
            tmp = f.name
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _candidate_version(record):
    machine = {k: v for k, v in record.items()
               if k not in ('review', 'candidate_version', 'previous_candidate_version', 'review_revision')}
    return hashlib.sha256(json.dumps(machine, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()[:24]


def _image_bindings(root):
    return {row['image']: row['source_image']
            for row in _read_jsonl(os.path.join(root, 'image_bindings.jsonl'))}


class ReviewQueue:
    def __init__(self, root):
        self.root = root
        self.queue_path = os.path.join(root, 'review_queue.jsonl')
        self.revisions_path = os.path.join(root, 'revisions.jsonl')
        self.final_path = os.path.join(root, 'scene_labels.json')
        self.bindings_path = os.path.join(root, 'image_bindings.jsonl')
        self.store = ReviewStore(root)

    # ---------- 读取 ----------

    def records(self):
        with self.store.session() as connection:
            return self._records(connection)

    @staticmethod
    def _records(connection):
        return {image: json.loads(payload) for image, payload in
                connection.execute('SELECT image, payload FROM records ORDER BY image')}

    @staticmethod
    def _bindings(connection):
        return dict(connection.execute('SELECT image, source_image FROM bindings'))

    @staticmethod
    def _put(connection, record):
        connection.execute('''INSERT INTO records(image, payload, status) VALUES (?, ?, ?)
            ON CONFLICT(image) DO UPDATE SET payload=excluded.payload,
                status=excluded.status, revision=records.revision+1''',
            (record['image'], ReviewStore.encode(record), (record.get('review') or {}).get('status', 'pending')))

    def is_locked(self, record):
        review = record.get('review') or {}
        # Machines always emit pending; old human records may omit reviewer.
        return review.get('status') in ('approved', 'rejected')

    # ---------- 写入 ----------

    def seed(self, reasoner_records):
        """并入候选记录；已人工确认的图片保持原样。返回统计。"""
        with self.store.session(write=True) as connection:
            return self._seed(connection, reasoner_records)

    def _seed(self, connection, reasoner_records):
        existing = self._records(connection)
        added = kept = skipped = updated = 0
        reasoner_records = list(reasoner_records)
        bindings = self._bindings(connection)
        original_bindings = dict(bindings)
        images = list(existing) + [rec['image'] for rec in reasoner_records]
        # Persist the first unambiguous full path separately so locked records
        # stay byte-for-byte unchanged, including across fresh queue instances.
        locations = {}
        for image in images + list(bindings.values()):
            if os.path.dirname(image):
                locations.setdefault(os.path.basename(image), set()).add(os.path.abspath(image))
        for image in images:
            if image == os.path.basename(image) and image in locations:
                if len(locations[image]) != 1:
                    raise ValueError(f'旧版记录存在跨目录同名歧义: {image}')
                bindings[image] = next(iter(locations[image]))
        for rec in reasoner_records:
            image = rec['image']
            # Older API evidence used bare filenames. Keep those human decisions
            # when the same project is re-analyzed with full image paths.
            legacy_name = os.path.basename(image)
            if image not in existing and image != legacy_name and legacy_name in existing:
                image = legacy_name
            elif image not in existing and image == legacy_name and image in bindings:
                image = next((key for key in existing
                              if os.path.abspath(key) == bindings[image]), image)
            previous = existing.get(image)
            if previous and self.is_locked(previous):
                kept += 1
                continue
            rec = dict(rec)
            rec.pop('review_revision', None)
            rec['image'] = image
            rec.setdefault('review', {'status': 'pending', 'reviewer': None,
                                      'updated_at': None})
            rec['candidate_version'] = _candidate_version(rec)
            if previous:
                old_version = previous.get('candidate_version') or _candidate_version(previous)
                if old_version != rec['candidate_version']:
                    rec['previous_candidate_version'] = old_version
                elif previous.get('previous_candidate_version'):
                    rec['previous_candidate_version'] = previous['previous_candidate_version']
            if previous == rec:
                kept += 1
                continue
            existing[image] = rec
            self._put(connection, rec)
            if previous:
                updated += 1
            else:
                added += 1
        if bindings != original_bindings:
            connection.executemany('INSERT OR REPLACE INTO bindings VALUES (?, ?)', sorted(bindings.items()))
        return {'added': added, 'kept': kept, 'skipped': skipped, 'updated': updated,
                'total': len(existing)}

    def list_pending(self, status=None, limit=50, offset=0):
        if not 1 <= limit <= 10000 or offset < 0:
            raise ValueError('分页数量必须在 1–10000 之间，offset 不可为负')
        with self.store.session() as connection:
            return self._list(connection, status, limit, offset)

    @staticmethod
    def _list(connection, status, limit, offset):
        where, values = (' WHERE status=?', [status]) if status else ('', [])
        rows = connection.execute('SELECT r.payload, r.revision, b.source_image FROM '
            '(SELECT * FROM records' + where + ' ORDER BY image LIMIT ? OFFSET ?) AS r '
            'LEFT JOIN bindings b ON b.image=r.image ORDER BY r.image', values + [limit, offset])
        return [dict(json.loads(payload), review_revision=revision,
                     **({'source_image': source} if source else {})) for payload, revision, source in rows]

    def page(self, status=None, limit=50, offset=0):
        if not 1 <= limit <= 10000 or offset < 0:
            raise ValueError('分页数量必须在 1–10000 之间，offset 不可为负')
        with self.store.session() as connection:
            return {'records': self._list(connection, status, limit, offset),
                    'summary': self._summary(connection)}

    def record(self, image):
        with self.store.session() as connection:
            row = connection.execute('SELECT r.payload, r.revision, b.source_image FROM records r '
                'LEFT JOIN bindings b ON b.image=r.image WHERE r.image=?', (image,)).fetchone()
            if row is None:
                raise KeyError(f'待审核记录不存在: {image}')
            payload, revision, source = row
            return dict(json.loads(payload), review_revision=revision,
                        **({'source_image': source} if source else {}))

    def decide(self, image, action, scene_id=None, label=None,
               reviewer='human', note='', expected_revision=None):
        """接受/修改/拒绝一条记录，同时追加不可变修订日志。"""
        with self.store.session(write=True) as connection:
            return self._decide(connection, image, action, scene_id, label, reviewer, note, expected_revision)

    def _decide(self, connection, image, action, scene_id, label, reviewer, note, expected_revision):
        if action not in ACTIONS:
            raise ValueError(f'未知动作: {action}（应为 {ACTIONS}）')
        reviewer = reviewer or 'human'
        row = connection.execute('SELECT payload, revision FROM records WHERE image=?', (image,)).fetchone()
        if row is None:
            raise KeyError(f'待审核记录不存在: {image}')
        payload, revision = row
        if expected_revision is not None and expected_revision != revision:
            raise ReviewConflict('这条记录已被更新，请核对最新结果后再保存；当前输入草稿已保留。')
        record = json.loads(payload)
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
            record.setdefault('machine_confidence', record.get('confidence'))
            record['confidence'] = None
            record['review'] = {'status': 'approved', 'reviewer': reviewer,
                                'updated_at': _now(), 'action': action, 'note': note,
                                'original_label': before['label']}
        else:  # reject
            record['scene_id'] = 'unknown'
            record['label'] = 'unknown'
            record['decision'] = 'human'
            record.setdefault('machine_confidence', record.get('confidence'))
            record['confidence'] = None
            record['review'] = {'status': 'rejected', 'reviewer': reviewer,
                                'updated_at': _now(), 'action': action, 'note': note,
                                'original_label': before['label']}

        self._put(connection, record)
        after = {'scene_id': record.get('scene_id'), 'label': record.get('label'),
                 'review_status': record['review']['status']}
        entry = {'at': _now(), 'image': image, 'action': action, 'reviewer': reviewer, 'note': note,
                 'before': before, 'after': after, 'before_revision': revision, 'after_revision': revision + 1}
        connection.execute('INSERT INTO revisions(payload) VALUES (?)', (ReviewStore.encode(entry),))
        return dict(record, review_revision=revision + 1)

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
        with self.store.session() as connection:
            records = self._records(connection)
            bindings = self._bindings(connection)
            labels = {bindings.get(image, image): row['label'] for image, row in records.items()
                      if (row.get('review') or {}).get('status') == 'approved'
                      and row.get('label') not in (None, 'unknown')}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=os.path.dirname(os.path.abspath(path)), delete=False) as stream:
                temporary = stream.name
                json.dump(labels, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return path, len(labels)

    def summary(self):
        with self.store.session() as connection:
            return self._summary(connection)

    def metadata(self):
        """Small project index; exclude evidence/model payloads from initial load."""
        with self.store.session() as connection:
            rows = connection.execute('SELECT r.image,r.status,json_extract(r.payload,\'$.label\'),'
                                      'r.revision,b.source_image FROM records r '
                                      'LEFT JOIN bindings b ON b.image=r.image ORDER BY r.image')
            records = [{'image': image, 'label': label, 'review_revision': revision,
                        'review': {'status': status}, '_metadata_only': True,
                        **({'source_image': source} if source else {})}
                       for image, status, label, revision, source in rows]
            return {'records': records, 'summary': self._summary(connection)}

    def records_for(self, images):
        """Fetch visible detail records only; full paths take priority over legacy names."""
        if not isinstance(images, list) or len(images) > 200 or any(not isinstance(image, str) for image in images):
            raise ValueError('最多查询200张图片')
        with self.store.session() as connection:
            result = []
            for image in dict.fromkeys(images):
                row = connection.execute('SELECT r.payload,r.revision,b.source_image FROM records r '
                                         'LEFT JOIN bindings b ON b.image=r.image WHERE r.image=?', (image,)).fetchone()
                if row is None:
                    row = connection.execute('SELECT r.payload,r.revision,b.source_image FROM records r '
                                             'LEFT JOIN bindings b ON b.image=r.image WHERE b.source_image=?', (image,)).fetchone()
                if row is None and os.path.dirname(image):
                    row = connection.execute('SELECT r.payload,r.revision,b.source_image FROM records r '
                                             'LEFT JOIN bindings b ON b.image=r.image WHERE r.image=? '
                                             'AND (b.source_image IS NULL OR b.source_image=?)',
                                             (os.path.basename(image), image)).fetchone()
                if row:
                    payload, revision, source = row
                    result.append(dict(json.loads(payload), review_revision=revision,
                                       **({'source_image': source} if source else {})))
            return {'records': result, 'summary': self._summary(connection)}

    @staticmethod
    def _summary(connection):
        counts = {'pending': 0, 'approved': 0, 'rejected': 0}
        counts.update(connection.execute('SELECT status, COUNT(*) FROM records GROUP BY status'))
        return counts

    def revisions(self):
        with self.store.session() as connection:
            return [json.loads(row[0]) for row in connection.execute('SELECT payload FROM revisions ORDER BY sequence')]


def load_scene_evidence(path):
    """读取已审核场景标签，供 SmartOrganizer 覆盖占位场景名。

    支持两种输入：
      - 复核队列 JSONL：只采纳 ``review.status == 'approved'`` 且非 unknown 的记录；
      - 人工标签 JSON：``{键: 标签}``，键为图片名/相对路径或 ``scene-NN`` 占位。

    Returns:
        dict: {``by_image``: {basename: label}, ``by_scene``: {scene_id: label}}
    """
    by_image, by_scene, by_path, legacy_by_image = {}, {}, {}, {}
    result = {'by_image': by_image, 'by_scene': by_scene,
              'by_path': by_path, 'legacy_by_image': legacy_by_image}
    def add_image(image, label):
        by_image[os.path.basename(image)] = label  # compatibility for old readers
        if os.path.dirname(image):
            by_path[os.path.abspath(image)] = label
        else:
            legacy_by_image[image] = label
    if not path or not os.path.exists(path):
        return result
    if path.endswith('.jsonl'):
        root = os.path.dirname(path)
        if os.path.basename(path) == 'review_queue.jsonl' and os.path.exists(os.path.join(root, 'review.sqlite3')):
            queue = ReviewQueue(root)
            with queue.store.session() as connection:
                bindings = queue._bindings(connection)
                rows = list(queue._records(connection).values())
        else:
            bindings = _image_bindings(root)
            rows = _read_jsonl(path)
        for record in rows:
            review = record.get('review')
            if not review or review.get('status') != 'approved':
                continue
            label = record.get('label')
            image = record.get('image')
            if label and label != 'unknown' and image:
                add_image(bindings.get(image, image), label)
        return result
    with open(path, encoding='utf-8') as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        return result
    for key, label in mapping.items():
        if not label or label == 'unknown':
            continue
        if isinstance(key, str) and key.startswith('scene-'):
            by_scene[key] = label
        else:
            add_image(key, label)
    return result

