"""Metrics over explicitly supplied review labels; empty inputs stay unevaluated."""
from collections import Counter
import unicodedata


def b_cubed(identities, clusters):
    if len(identities) != len(clusters):
        raise ValueError('cluster_annotation_count_mismatch')
    if any(identity is None or identity == '' for identity in identities):
        raise ValueError('unlabelled_identity_in_metric')
    count = len(identities)
    if not count:
        return {'count': 0, 'precision': None, 'recall': None, 'f1': None,
                'mixed_clusters': 0, 'fragmented_identities': 0}
    truth, predicted = Counter(identities), Counter(clusters)
    intersections = Counter(zip(identities, clusters))
    cells_by_cluster = Counter(cluster for _, cluster in intersections)
    cells_by_identity = Counter(identity for identity, _ in intersections)
    precision = sum(n * n / predicted[cluster] for (_, cluster), n in intersections.items()) / count
    recall = sum(n * n / truth[identity] for (identity, _), n in intersections.items()) / count
    return {'count': count, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall),
            'mixed_clusters': sum(count > 1 for count in cells_by_cluster.values()),
            'fragmented_identities': sum(count > 1 for count in cells_by_identity.values())}


def normalize_transcription(value):
    """NFC; CRLF/CR become LF; trim edges, retain interior spaces/newlines."""
    return unicodedata.normalize('NFC', value.replace('\r\n', '\n').replace('\r', '\n')).strip()


def character_errors(reference, prediction):
    reference, prediction = normalize_transcription(reference), normalize_transcription(prediction)
    # Only the shorter row is retained; character insertion/deletion costs are symmetric.
    a, b = (reference, prediction) if len(reference) >= len(prediction) else (prediction, reference)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    errors = previous[-1]
    return {'edits': errors, 'reference_characters': len(reference),
            'prediction_characters': len(prediction),
            'cer': errors / len(reference) if reference else None}
