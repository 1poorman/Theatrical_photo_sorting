"""Conservative mutual-kNN ArcFace grouping with explicit cannot-link guards.

This backend is experimental. It uses FAISS exact inner-product neighbours,
bounded kNN edges and centroid guards; it never builds an n*n Python matrix.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FaceGraphConfig:
    neighbors: int = 16
    min_similarity: float = 0.60
    min_centroid_similarity: float = 0.65
    max_cluster_size: int = 512

    def __post_init__(self):
        if type(self.neighbors) is not int or not 1 <= self.neighbors <= 128:
            raise ValueError('invalid_face_graph_neighbors')
        if type(self.max_cluster_size) is not int or not 1 <= self.max_cluster_size <= 10000:
            raise ValueError('invalid_face_graph_max_cluster_size')
        for name in ('min_similarity', 'min_centroid_similarity'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError('invalid_face_graph:' + name)


def cannot_link_pairs(members):
    """Same source photo, different detections; marked mirrors/composites excluded."""
    groups = {}
    for index, member in enumerate(members):
        if member.get('cannot_link_exempt') is True:
            continue
        source = member.get('source_image_sha256') or member.get('img_path') or member.get('image')
        if source:
            groups.setdefault(source, []).append((index, member.get('face_index', index)))
    pairs = set()
    for items in groups.values():
        for position, (left, face_left) in enumerate(items):
            for right, face_right in items[position + 1:]:
                if face_left != face_right:
                    pairs.add((left, right))
    return sorted(pairs)


def graph_labels(features, cannot_link=(), config=None, threads=4):
    config = config or FaceGraphConfig()
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('invalid_face_graph_features')
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError('zero_face_graph_feature')
    values = np.ascontiguousarray(values / norms[:, None])
    n = len(values)
    cannot = set()
    adjacency = [set() for _ in range(n)]
    for left, right in cannot_link:
        if type(left) not in (int, np.int64) or type(right) not in (int, np.int64) or not 0 <= left < n or not 0 <= right < n or left == right:
            raise ValueError('invalid_cannot_link_pair')
        left, right = int(left), int(right)
        cannot.add((min(left, right), max(left, right)))
        adjacency[left].add(right)
        adjacency[right].add(left)
    if n <= 1:
        return np.arange(n, dtype=np.int32), {'n_faces': n, 'n_edges': 0, 'constraint_violations': 0}
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('face_graph_requires_faiss') from exc
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(threads)
        index = faiss.IndexFlatIP(values.shape[1])
        index.add(values)
        scores, indices = index.search(values, min(n, config.neighbors + 1))
    finally:
        faiss.omp_set_num_threads(previous_threads)
    neighbours = [set(int(j) for j in row if j >= 0 and j != i) for i, row in enumerate(indices)]
    edges = [(float(score), i, int(j)) for i, (row, distances) in enumerate(zip(indices, scores))
             for j, score in zip(row, distances) if j > i and score >= config.min_similarity and i in neighbours[int(j)]]
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    parents = np.arange(n)
    sets = [{i} for i in range(n)]
    forbidden = [set(items) for items in adjacency]
    sums = values.astype(np.float64)
    min_member_similarity = np.ones(n, dtype=np.float64)
    def root(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = int(parents[item])
        return item
    blocked = {'cannot_link': 0, 'centroid': 0, 'cluster_size': 0, 'transitive_drift': 0}
    merged = 0
    for _, left, right in edges:
        a, b = root(left), root(right)
        if a == b:
            continue
        if len(sets[a]) + len(sets[b]) > config.max_cluster_size:
            blocked['cluster_size'] += 1
            continue
        if sets[a].intersection(forbidden[b]) or sets[b].intersection(forbidden[a]):
            blocked['cannot_link'] += 1
            continue
        na, nb = np.linalg.norm(sums[a]), np.linalg.norm(sums[b])
        if na <= 0 or nb <= 0 or sums[a] @ sums[b] / (na * nb) < config.min_centroid_similarity:
            blocked['centroid'] += 1
            continue
        combined = sums[a] + sums[b]
        norm = np.linalg.norm(combined)
        if norm <= 0:
            blocked['centroid'] += 1
            continue
        # A centroid-only chain can drift across identities. Check every member
        # against the new centroid; max_cluster_size bounds each guard.
        ids = list(sets[a]) + list(sets[b])
        minimum = float((values[ids] @ (combined / norm)).min())
        if minimum < config.min_centroid_similarity:
            blocked['transitive_drift'] += 1
            continue
        if len(sets[a]) < len(sets[b]) or (len(sets[a]) == len(sets[b]) and a > b):
            a, b = b, a
        parents[b] = a
        sets[a].update(sets[b])
        sets[b].clear()
        forbidden[a].update(forbidden[b])
        forbidden[b].clear()
        sums[a] = combined
        min_member_similarity[a] = minimum
        merged += 1
    roots = [root(i) for i in range(n)]
    ordered = sorted(set(roots), key=lambda r: min(sets[r]))
    mapping = {r: i for i, r in enumerate(ordered)}
    labels = np.array([mapping[r] for r in roots], dtype=np.int32)
    violations = sum(int(labels[left] == labels[right]) for left, right in cannot)
    if violations:
        raise ValueError('cannot_link_constraint_violated')
    return labels, {'n_faces': n, 'n_edges': len(edges), 'n_clusters': len(ordered), 'merged_edges': merged,
                    'blocked_edges': blocked, 'valid_cannot_links': len(cannot), 'constraint_violations': violations,
                    'backend': 'faiss_exact_mutual_knn', 'faiss_version': faiss.__version__,
                    'threads': threads, 'quality_status': 'experimental_unvalidated',
                    'complexity': 'exact neighbour search O(n^2*d); stored edges O(n*k); merge guards bounded by cluster cap',
                    'centroid_min_member_similarity': [float(min_member_similarity[r]) for r in ordered]}


def graph_clusters(members, config=None):
    if not members:
        return [], {'n_faces': 0, 'n_clusters': 0, 'constraint_violations': 0}
    features = np.asarray([row['feat'] for row in members], dtype=np.float32)
    labels, audit = graph_labels(features, cannot_link_pairs(members), config)
    clusters = []
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    for label in sorted(set(labels)):
        ids = np.flatnonzero(labels == label)
        values = features[ids].astype(np.float64)
        total, count = values.sum(axis=0), len(ids)
        cohesion = float((total @ total - np.square(values).sum()) / (count * (count - 1))) if count > 1 else 1.0
        selected = [members[int(i)] for i in ids]
        clusters.append({'id': f'cluster-{label + 1:02d}', 'size': count, 'cohesion': round(cohesion, 3),
                         'suggest_manual': True, 'status': 'pending', 'assigned_to': None, 'preview': None,
                         'sample_images': list(dict.fromkeys(m['image'] for m in selected))[:10],
                         'members': [{key: value for key, value in m.items() if key != 'feat'} for m in selected],
                         'clustering_backend': 'graph_experimental'})
    return sorted(clusters, key=lambda c: (-c['size'], c['id'])), audit
