"""Order-preserving window change points for precomputed visual sequences.

Window scores are proposal evidence, not confirmed plot boundaries. Parameters
must be fixed on validation before quality claims or production adoption.
"""
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np


@dataclass(frozen=True)
class ChangePointConfig:
    window: int = 4
    min_segment: int = 4
    penalty: float = 0.12
    max_gap_seconds: float = 300.0
    mode: str = 'rehearsal'
    sequence_order: str = 'input'

    def __post_init__(self):
        for name in ('window', 'min_segment'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('invalid_change_point:' + name)
        for name in ('penalty', 'max_gap_seconds'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid_change_point:' + name)
        if self.penalty > 2:
            raise ValueError('change_point_penalty_out_of_range')
        if self.mode not in ('rehearsal', 'performance') or self.sequence_order not in ('input', 'capture'):
            raise ValueError('invalid_change_point_sequence_mode')


def window_scores(features, window=4):
    """O(n*d) prefix sums plus O(n*d) bounded array operations; no n*n matrix."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('invalid_sequence_features')
    if type(window) is not int or window < 1:
        raise ValueError('invalid_change_point_window')
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('zero_sequence_feature')
    values = values / norms[:, None]
    prefix = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.float64),
                             np.cumsum(values, axis=0, dtype=np.float64)], axis=0)
    points = np.arange(1, len(values))
    left = prefix[points] - prefix[np.maximum(0, points - window)]
    right = prefix[np.minimum(len(values), points + window)] - prefix[points]
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    scores = np.divide(np.sum(left * right, axis=1), denom,
                       out=np.ones(len(points), dtype=np.float64), where=denom > 1e-8)
    return np.clip(1 - scores, 0, 2)


def change_points(features, config=None):
    config = config or ChangePointConfig()
    scores = window_scores(features, config.window)
    n = len(scores) + 1 if len(features) else 0
    # Greedy peak selection enforces minimum segment size at both ends.
    candidates = sorted((i + 1 for i, score in enumerate(scores) if score >= config.penalty and
                         config.min_segment <= i + 1 <= n - config.min_segment),
                        key=lambda index: (-float(scores[index - 1]), index))
    selected = []
    import bisect
    for point in candidates:
        position = bisect.bisect_left(selected, point)
        left = selected[position - 1] if position else 0
        right = selected[position] if position < len(selected) else n
        if point - left >= config.min_segment and right - point >= config.min_segment:
            selected.insert(position, point)
    return selected, scores


def split_ordered_scenes(paths, embeddings=None, timestamps=None, config=None, order_trusted=True):
    """Keep contiguous spans in an explicit capture or original-input order.

    Missing timestamps keep original order; a mixture is never appended as a new
    artificial tail. Unknown chronology disables time and performance priors.
    """
    config = config or ChangePointConfig()
    paths = list(paths)
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate_sequence_path')
    if timestamps is None:
        timestamps = {path: None for path in paths}
    if not isinstance(timestamps, dict):
        if len(timestamps) != len(paths):
            raise ValueError('sequence_timestamp_alignment')
        timestamps = dict(zip(paths, timestamps))
    if embeddings is not None and not isinstance(embeddings, dict):
        if len(embeddings) != len(paths):
            raise ValueError('sequence_embedding_alignment')
        embeddings = dict(zip(paths, embeddings))
    if type(order_trusted) is not bool:
        raise ValueError('invalid_sequence_order_trust')
    time_complete = all(isinstance(timestamps.get(path), datetime) for path in paths)
    ordered, effective_order = paths, 'input'
    elapsed = []
    if time_complete and order_trusted:
        try:
            if config.sequence_order == 'capture':
                ordered = sorted(paths, key=lambda p: timestamps[p])
                effective_order = 'capture'
            elapsed = [(timestamps[b] - timestamps[a]).total_seconds() for a, b in zip(ordered, ordered[1:])]
        except TypeError:  # Mixed timezone-aware and naive timestamps are not comparable.
            time_complete = False
            ordered, effective_order = paths, 'input'
    time_trusted = bool(time_complete and order_trusted and all(value >= 0 for value in elapsed))
    gaps = []
    if time_trusted:
        gaps = [i for i, value in enumerate(elapsed, 1) if value > config.max_gap_seconds]
    boundaries = set(gaps)
    diagnostics = []
    for start, end in zip([0] + gaps, gaps + [len(ordered)]):
        segment = ordered[start:end]
        if embeddings is None:
            diagnostics.append({'start': start, 'end': end, 'subdivision': 'unavailable_features'})
            continue
        # A missing feature must not discard the rest of a continuous sequence.
        # Score valid contiguous runs separately; never bridge a missing image.
        run, run_start = [], start
        for offset, path in enumerate(segment + [None], start):
            raw = embeddings.get(path) if path is not None else None
            value = None
            if raw is not None:
                try:
                    candidate = np.asarray(raw, dtype=np.float32)
                    norm = np.linalg.norm(candidate)
                    if candidate.ndim == 1 and candidate.size and np.isfinite(candidate).all() and np.isfinite(norm) and norm > 0:
                        value = candidate
                except (ValueError, TypeError):
                    pass
            if run and (value is None or value.shape != run[-1].shape):
                points, scores = change_points(run, config)
                boundaries.update(run_start + point for point in points)
                diagnostics.append({'start': run_start, 'end': offset, 'subdivision': 'window_change_points',
                                    'scores': [float(score) for score in scores]})
                run = []
            if value is not None:
                if not run:
                    run_start = offset
                run.append(value)
            elif path is not None:
                diagnostics.append({'start': offset, 'end': offset + 1,
                                    'subdivision': 'unavailable_features' if raw is None else 'invalid_features'})
    indices = sorted(boundaries)
    assignments = {}
    for number, (start, end) in enumerate(zip([0] + indices, indices + [len(ordered)]), 1):
        for path in ordered[start:end]:
            assignments[path] = f'scene-{number:02d}'
    return {'assignments': assignments, 'ordered_paths': ordered, 'boundaries': indices,
            'time_gap_boundaries': gaps, 'effective_order': effective_order,
            'order_trusted': bool(order_trusted), 'mode': config.mode,
            'temporal_prior_enabled': time_trusted,
            'plot_order_prior_applied': False, 'diagnostics': diagnostics,
            'quality_status': 'experimental_unvalidated',
            'complexity': 'window scoring O(n*d); worst-case peak insertion O(n^2); memory O(n*d)'}
