"""Conservative temporal/visual burst groups and equal-quota selection research."""
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np


@dataclass(frozen=True)
class BurstConfig:
    max_gap_seconds: float = 2.0
    max_span_seconds: float = 8.0
    min_similarity: float = 0.97
    max_group_size: int = 24
    diversity_weight: float = 0.10

    def __post_init__(self):
        for name in ('max_gap_seconds', 'max_span_seconds', 'min_similarity', 'diversity_weight'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid_temporal_burst:' + name)
        if self.min_similarity > 1 or self.diversity_weight > 1 or self.max_span_seconds < self.max_gap_seconds:
            raise ValueError('invalid_temporal_burst_range')
        if type(self.max_group_size) is not int or not 1 <= self.max_group_size <= 10000:
            raise ValueError('invalid_temporal_burst_size')


def vector(row):
    try:
        value = np.asarray(row.get('embedding'), dtype=np.float32)
        norm = np.linalg.norm(value)
        if value.ndim != 1 or not value.size or not np.isfinite(value).all() or norm <= 0:
            return None
        return value / norm
    except (ValueError, TypeError):
        return None


def temporal_groups(rows, config=None):
    """Group only reliable same-stream neighbours; keep unknown chronology alone.

    Human scene labels can veto merges. They are never inferred from proximity.
    Both last-member and first-member similarity guards prevent drifting chains.
    """
    config = config or BurstConfig()
    rows = list(rows)
    if len({row['image'] for row in rows}) != len(rows):
        raise ValueError('duplicate_burst_image')
    streams, singleton = {}, []
    for row in rows:
        if row.get('order_trusted') is True and row.get('stream_id') and isinstance(row.get('captured_at'), datetime):
            streams.setdefault(row['stream_id'], []).append(row)
        else:
            singleton.append([row])
    groups = list(singleton)
    audit = {'missing_or_untrusted_time': len(singleton), 'visual_rejections': 0,
             'scene_conflict_rejections': 0, 'time_rejections': 0}
    for stream, sequence in sorted(streams.items()):
        sequence.sort(key=lambda row: (row['captured_at'], row['image']))
        current = []
        for row in sequence:
            merge = bool(current)
            if current:
                gap = (row['captured_at'] - current[-1]['captured_at']).total_seconds()
                span = (row['captured_at'] - current[0]['captured_at']).total_seconds()
                if gap > config.max_gap_seconds or span > config.max_span_seconds or len(current) >= config.max_group_size:
                    merge = False
                    audit['time_rejections'] += 1
                known = {r['confirmed_scene'] for r in current if r.get('confirmed_scene')}
                if row.get('confirmed_scene') and known and row['confirmed_scene'] not in known:
                    merge = False
                    audit['scene_conflict_rejections'] += 1
                features = [vector(r) for r in (current[0], current[-1], row)]
                if any(value is None for value in features) or any(value.shape != features[-1].shape for value in features) or min(
                        float(features[0] @ features[-1]), float(features[1] @ features[-1])) < config.min_similarity:
                    merge = False
                    audit['visual_rejections'] += 1
            if not merge and current:
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)
    original_order = {row['image']: i for i, row in enumerate(rows)}
    groups.sort(key=lambda group: min(original_order[row['image']] for row in group))
    return [(f'burst-{i:05d}', [row['image'] for row in group]) for i, group in enumerate(groups)], dict(
        audit, input_count=len(rows), group_count=len(groups), quality_status='experimental_unvalidated',
        boundary_policy='same trusted stream + time limits + two visual guards + confirmed-scene veto')


def selection_score(quality):
    """Compare subject clarity/exposure when supplied; no global dark-stage penalty."""
    if type(quality.get('subject_score')) in (int, float) and math.isfinite(quality['subject_score']):
        return float(np.clip(quality['subject_score'], 0, 1))
    sharpness = max(1.0, float(quality.get('sharpness', 1)))
    contrast = max(0.0, float(quality.get('contrast', 0)))
    return .70 * min(1, math.log10(sharpness) / 3.5) + .30 * min(1, contrast / 55)


def select_equal_quota(rows, quota, config=None):
    config = config or BurstConfig()
    if type(quota) is not int or quota < 0:
        raise ValueError('invalid_selection_quota')
    rows = list(rows)
    if len({row['image'] for row in rows}) != len(rows):
        raise ValueError('duplicate_selection_image')
    protected = [row for row in rows if row.get('must_keep') is True]
    if len(protected) > quota:
        raise ValueError('protected_images_exceed_quota; increase quota explicitly')
    selected = list(protected)
    protected_ids = {row['image'] for row in protected}
    remaining = [row for row in rows if row['image'] not in protected_ids]
    while remaining and len(selected) < quota:
        def score(row):
            value = selection_score(row.get('quality', {}))
            feature = vector(row)
            previous = [vector(item) for item in selected]
            previous = [item for item in previous if item is not None and feature is not None and item.shape == feature.shape]
            if previous:
                novelty = 1 - max(float(feature @ item) for item in previous)
                value += config.diversity_weight * novelty
            return value
        best = min(remaining, key=lambda row: (-score(row), row['image']))
        selected.append(best)
        remaining = [row for row in remaining if row['image'] != best['image']]
    ids = {row['image'] for row in selected}
    return {'selected': [row['image'] for row in rows if row['image'] in ids],
            'quota': quota, 'input_count': len(rows), 'protected_count': len(protected),
            'quality_status': 'experimental_unvalidated', 'quality_model': 'subject_or_local_clarity_contrast',
            'selection_count': len(selected)}
