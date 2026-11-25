"""Benchmark face recognition and detection families on project data.

Recognition uses the labeled, aligned anchors under ``data/face_database``.
Queries never compare against another crop from the same source photograph.
Detection uses the existing face-review boxes as a regression reference; those
boxes were produced by an earlier SCRFD run and are not independent ground truth.

Supported recognition paths are ArcFace-compatible ONNX, OpenCV SFace,
facenet-pytorch InceptionResnetV1, SwinFace, and weighted feature concatenation. Supported
detectors are SCRFD, OpenCV YuNet, and raw-output RetinaFace ONNX exports.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import time

import cv2
import numpy as np
import onnx
import onnxruntime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
from core_modules.tools.image_io import imread_reduced


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def onnx_parameters(path):
    model = onnx.load(str(path), load_external_data=False)
    return sum(int(np.prod(value.dims)) for value in model.graph.initializer)


def source_key(path):
    stem = Path(path).stem
    stem = re.sub(r'_face(?:_\d+)?$', '', stem)
    return stem


def actor_key(identity):
    return re.split(r'[（(]', identity, maxsplit=1)[0].strip()


def load_faces(root):
    rows = []
    for identity in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        for path in sorted(identity.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            rows.append({'path': str(path.resolve()), 'identity': identity.name,
                         'actor_identity': actor_key(identity.name),
                         'source': source_key(path), 'content_sha256': sha256(path), 'image': image})
    return rows


def model_embeddings(model_path, rows, device, batch_size, timing_rounds):
    model = ArcFaceFeatureExtractor(str(model_path), device=device)
    providers = model.session.get_providers()
    images = [cv2.resize(row['image'], model.input_size, interpolation=cv2.INTER_LINEAR) for row in rows]
    started = time.perf_counter()
    features = []
    for offset in range(0, len(images), batch_size):
        features.append(model.extract_features(images[offset:offset + batch_size]))
    elapsed = time.perf_counter() - started
    matrix = model.l2_normalize(np.concatenate(features, axis=0))

    timing = {}
    for size in (1, min(32, len(images))):
        batch = images[:size]
        for _ in range(5):
            model.extract_features(batch)
        samples = []
        for _ in range(timing_rounds):
            tick = time.perf_counter()
            model.extract_features(batch)
            samples.append((time.perf_counter() - tick) * 1000)
        timing[f'batch_{size}'] = {
            'batch_p50_ms': percentile(samples, 50), 'batch_p95_ms': percentile(samples, 95),
            'per_face_p50_ms': percentile(samples, 50) / size,
            'faces_per_second_from_p50': size * 1000 / percentile(samples, 50),
        }
    del model
    return matrix, {'providers': providers, 'extraction_seconds': elapsed,
                    'faces_per_second': len(rows) / elapsed, 'timing': timing}


def sface_embeddings(model_path, rows, timing_rounds):
    model = cv2.FaceRecognizerSF.create(str(model_path), '')
    images = [cv2.resize(row['image'], (112, 112), interpolation=cv2.INTER_LINEAR) for row in rows]

    def extract(batch):
        return np.concatenate([model.feature(image) for image in batch], axis=0)

    started = time.perf_counter()
    matrix = extract(images)
    elapsed = time.perf_counter() - started
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    timing = {}
    for size in (1, min(32, len(images))):
        batch = images[:size]
        for _ in range(5):
            extract(batch)
        samples = []
        for _ in range(timing_rounds):
            tick = time.perf_counter()
            extract(batch)
            samples.append((time.perf_counter() - tick) * 1000)
        timing[f'batch_{size}'] = {
            'batch_p50_ms': percentile(samples, 50), 'batch_p95_ms': percentile(samples, 95),
            'per_face_p50_ms': percentile(samples, 50) / size,
            'faces_per_second_from_p50': size * 1000 / percentile(samples, 50),
        }
    return matrix, {'providers': ['OpenCV DNN CPU'], 'extraction_seconds': elapsed,
                    'faces_per_second': len(rows) / elapsed, 'timing': timing}


def facenet_embeddings(checkpoint, rows, device, batch_size, timing_rounds):
    import torch
    from facenet_pytorch import InceptionResnetV1

    torch_device = torch.device('cuda' if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(pretrained=None).to(torch_device).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or set(unexpected) - {'logits.weight', 'logits.bias'}:
        raise ValueError(f'Unexpected FaceNet checkpoint keys: missing={missing}, unexpected={unexpected}')
    images = []
    for row in rows:
        image = cv2.resize(row['image'], (160, 160), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        images.append(((image - 127.5) / 128.0).transpose(2, 0, 1))
    tensor = torch.from_numpy(np.stack(images))

    def extract(batch):
        with torch.inference_mode():
            return model(batch.to(torch_device)).cpu().numpy()

    started = time.perf_counter()
    matrix = np.concatenate([extract(tensor[offset:offset + batch_size])
                             for offset in range(0, len(tensor), batch_size)], axis=0)
    elapsed = time.perf_counter() - started
    timing = {}
    for size in (1, min(32, len(tensor))):
        batch = tensor[:size]
        for _ in range(3):
            extract(batch)
        samples = []
        for _ in range(timing_rounds):
            tick = time.perf_counter()
            extract(batch)
            samples.append((time.perf_counter() - tick) * 1000)
        timing[f'batch_{size}'] = {
            'batch_p50_ms': percentile(samples, 50), 'batch_p95_ms': percentile(samples, 95),
            'per_face_p50_ms': percentile(samples, 50) / size,
            'faces_per_second_from_p50': size * 1000 / percentile(samples, 50),
        }
    parameters = sum(value.numel() for value in model.parameters())
    return matrix, {'providers': [str(torch_device)], 'extraction_seconds': elapsed,
                    'faces_per_second': len(rows) / elapsed, 'timing': timing,
                    'parameters': parameters}


def swinface_embeddings(checkpoint, rows, device, batch_size, timing_rounds):
    from core_modules.face_recognition.swinface import SwinFaceAnalyzer

    model = SwinFaceAnalyzer(checkpoint, device=device, batch_size=batch_size)
    images = [row['image'] for row in rows]
    started = time.perf_counter()
    matrix = model.extract_features(images)
    elapsed = time.perf_counter() - started
    started = time.perf_counter()
    ages = model.estimate_ages(images)
    age_seconds = time.perf_counter() - started
    timing, joint_timing = {}, {}
    for operation, destination in ((model.extract_features, timing), (model.predict, joint_timing)):
        for size in (1, min(32, len(images))):
            batch = images[:size]
            for _ in range(3):
                operation(batch)
            samples = []
            for _ in range(timing_rounds):
                tick = time.perf_counter()
                operation(batch)
                samples.append((time.perf_counter() - tick) * 1000)
            destination[f'batch_{size}'] = {
                'batch_p50_ms': percentile(samples, 50), 'batch_p95_ms': percentile(samples, 95),
                'per_face_p50_ms': percentile(samples, 50) / size,
                'faces_per_second_from_p50': size * 1000 / percentile(samples, 50),
                'samples_ms': samples,
            }
    metadata = model.metadata()
    age_report = {
        'unit': 'years', 'labeled_age_count': 0, 'mae': None,
        'validation_status': 'functional_only_no_age_ground_truth',
        'extraction_seconds': age_seconds,
        'predictions_count': len(ages), 'minimum': float(ages.min()),
        'median': float(np.median(ages)), 'maximum': float(ages.max()),
        'outside_0_120_count': int(((ages < 0) | (ages > 120)).sum()),
        'predictions': [{'path': row['path'], 'estimated_age': float(age)} for row, age in zip(rows, ages)],
    }
    return matrix, {
        'providers': [str(model.device)], 'extraction_seconds': elapsed,
        'faces_per_second': len(rows) / elapsed, 'timed_operation': 'recognition_only',
        'timing': timing, 'recognition_and_age_timing': joint_timing,
    }, metadata, age_report


def roc_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((~labels).sum()))
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    tpr = tp / positives
    fpr = fp / negatives
    thresholds = scores[order]
    auc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    result = {'auc': auc, 'eer': float((fnr[index] + fpr[index]) / 2),
              'eer_threshold': float(thresholds[index])}
    for target in (0.1, 0.01, 0.001):
        valid = np.where(fpr <= target)[0]
        chosen = int(valid[-1]) if len(valid) else 0
        result[f'tar_at_far_{target:g}'] = float(tpr[chosen])
        result[f'threshold_at_far_{target:g}'] = float(thresholds[chosen])
    return result


def closed_set_operating_points(queries):
    total = len(queries)
    thresholds = [max(row['score'] for row in queries) + 1e-6] + sorted(
        {row['score'] for row in queries}, reverse=True)
    result = {}
    for maximum_wrong_rate in (0, .01, .02):
        candidates = []
        for threshold in thresholds:
            accepted = [row for row in queries if row['score'] >= threshold]
            correct = sum(row['correct'] for row in accepted)
            wrong = len(accepted) - correct
            if wrong / total <= maximum_wrong_rate:
                candidates.append((correct, -wrong, -threshold, threshold, wrong))
        _, _, _, threshold, wrong = max(candidates)
        correct = sum(row['correct'] and row['score'] >= threshold for row in queries)
        result[f'max_wrong_accept_rate_{maximum_wrong_rate:g}'] = {
            'threshold': threshold, 'correct_accept_rate': correct / total,
            'wrong_accept_rate': wrong / total,
            'rejection_rate': 1 - (correct + wrong) / total,
        }
    return result


def evaluate_recognition(rows, embeddings, identity_field):
    identities = np.asarray([row[identity_field] for row in rows])
    sources = np.asarray([row['source'] for row in rows])
    hashes = np.asarray([row['content_sha256'] for row in rows])
    similarities = embeddings @ embeddings.T
    queries = []
    top1 = top5 = 0
    accepted = {threshold: {'correct': 0, 'wrong': 0} for threshold in (.4, .55)}
    margins = []
    for index in range(len(rows)):
        allowed = (sources != sources[index]) & (hashes != hashes[index])
        positive = allowed & (identities == identities[index])
        if not positive.any():
            continue
        candidates = np.where(allowed)[0]
        order = candidates[np.argsort(similarities[index, candidates])[::-1]]
        predicted = identities[order]
        top1 += int(predicted[0] == identities[index])
        top5 += int(identities[index] in predicted[:5])
        for threshold in accepted:
            if similarities[index, order[0]] >= threshold:
                accepted[threshold]['correct' if predicted[0] == identities[index] else 'wrong'] += 1
        best_positive = float(similarities[index, np.where(positive)[0]].max())
        negative = allowed & (identities != identities[index])
        best_negative = float(similarities[index, np.where(negative)[0]].max())
        margins.append(best_positive - best_negative)
        queries.append({'path': rows[index]['path'], 'identity': identities[index],
                        'predicted': predicted[0], 'correct': bool(predicted[0] == identities[index]),
                        'score': float(similarities[index, order[0]]), 'margin': best_positive - best_negative})

    pair_scores, pair_labels = [], []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if sources[left] == sources[right] or hashes[left] == hashes[right]:
                continue
            pair_scores.append(float(similarities[left, right]))
            pair_labels.append(identities[left] == identities[right])
    failures = [row for row in queries if not row['correct']]
    return {
        'anchors': len(rows), 'identities': len(set(identities)), 'source_groups': len(set(sources)),
        'eligible_queries': len(queries), 'top1_accuracy': top1 / len(queries),
        'top5_accuracy': top5 / len(queries), 'margin_p05': percentile(margins, 5),
        'margin_p50': percentile(margins, 50), 'positive_pairs': int(sum(pair_labels)),
        'negative_pairs': int(len(pair_labels) - sum(pair_labels)),
        'verification': roc_metrics(pair_scores, pair_labels),
        'fixed_thresholds': {str(threshold): {
            'correct_accept_rate': value['correct'] / len(queries),
            'wrong_accept_rate': value['wrong'] / len(queries),
            'rejection_rate': 1 - (value['correct'] + value['wrong']) / len(queries),
        } for threshold, value in accepted.items()},
        'calibrated_closed_set_operating_points': closed_set_operating_points(queries),
        'top1_failure_count': len(failures),
        'confusions': [{'actual': actual, 'predicted': predicted, 'count': count}
                       for (actual, predicted), count in Counter(
                           (row['identity'], row['predicted']) for row in failures).most_common()],
    }


def box_iou(one, many):
    one = np.asarray(one, dtype=np.float32)
    many = np.asarray(many, dtype=np.float32)
    x1 = np.maximum(one[0], many[:, 0]); y1 = np.maximum(one[1], many[:, 1])
    x2 = np.minimum(one[2], many[:, 2]); y2 = np.minimum(one[3], many[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area_one = max(0, one[2]-one[0]) * max(0, one[3]-one[1])
    area_many = np.maximum(0, many[:, 2]-many[:, 0]) * np.maximum(0, many[:, 3]-many[:, 1])
    return inter / np.maximum(area_one + area_many - inter, 1e-9)


def match_boxes(reference, predicted, threshold=.5):
    candidates = []
    for left, box in enumerate(reference):
        if len(predicted):
            for right, value in enumerate(box_iou(box, predicted)):
                if value >= threshold:
                    candidates.append((float(value), left, right))
    used_left, used_right, matches = set(), set(), []
    for value, left, right in sorted(candidates, reverse=True):
        if left not in used_left and right not in used_right:
            used_left.add(left); used_right.add(right); matches.append((left, right, value))
    return matches


class BatchedOutputSession:
    """Adapt official InsightFace SCRFD exports that omit output batch axes."""
    def __init__(self, session):
        self.session = session

    def run(self, *args, **kwargs):
        return [value[None, ...] if value.ndim == 2 else value
                for value in self.session.run(*args, **kwargs)]


def review_reference(path, limit=None):
    review = json.loads(Path(path).read_text())
    images = {row['image_id']: row for row in review['images']}
    faces = defaultdict(list)
    for row in review['faces']:
        faces[row['image_id']].append(row['bbox'])
    rows = []
    for image_id in sorted(images):
        if limit is not None and len(rows) >= limit:
            break
        rows.append({'image_id': image_id, 'path': images[image_id]['source'], 'boxes': faces[image_id]})
    return rows


def evaluate_detector_predictions(reference, predict, providers):
    latencies, counts, matched, refs = [], 0, 0, 0
    ious, by_size = [], defaultdict(lambda: [0, 0])
    for row_index, row in enumerate(reference):
        image = imread_reduced(row['path'], target_max=1920)
        if image is None:
            continue
        if max(image.shape[:2]) > 1920:
            scale = 1920 / max(image.shape[:2])
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                               interpolation=cv2.INTER_LINEAR)
        if row_index == 0:
            predict(image)
        tick = time.perf_counter()
        predicted = predict(image)
        latencies.append((time.perf_counter() - tick) * 1000)
        predicted = np.asarray([box for box in predicted if (box[2]-box[0])*(box[3]-box[1]) >= 1600])
        if predicted.size == 0:
            predicted = np.empty((0, 4), dtype=np.float32)
        reference_boxes = np.asarray(row['boxes'], dtype=np.float32)
        matches = match_boxes(reference_boxes, predicted)
        counts += len(predicted); refs += len(reference_boxes); matched += len(matches)
        ious.extend(value for _, _, value in matches)
        for index, box in enumerate(reference_boxes):
            side = np.sqrt(max(0, (box[2]-box[0])*(box[3]-box[1])))
            bucket = '40-79' if side < 80 else ('80-159' if side < 160 else '160+')
            by_size[bucket][1] += 1
            by_size[bucket][0] += int(any(left == index for left, _, _ in matches))
    return {
        'images': len(latencies), 'reference_faces': refs, 'detections': counts, 'matched_iou_0_5': matched,
        'reference_recall': matched / refs if refs else None,
        'prediction_precision_against_reference': matched / counts if counts else None,
        'matched_iou_p50': percentile(ious, 50), 'latency_p50_ms': percentile(latencies, 50),
        'latency_p95_ms': percentile(latencies, 95), 'images_per_second_from_p50': 1000 / percentile(latencies, 50),
        'providers': providers,
        'reference_recall_by_sqrt_area_px': {key: value[0]/value[1] for key, value in sorted(by_size.items()) if value[1]},
    }


def evaluate_scrfd_detector(model_path, reference, device):
    detector = SCRFDDetector(str(model_path), device=device, det_thresh=.3)
    providers = detector.session.get_providers()
    if len(detector.session.get_outputs()[0].shape) == 2:
        detector.session = BatchedOutputSession(detector.session)

    def predict(image):
        detections, _ = detector.detect(image, input_size=(640, 640))
        return detections[:, :4] if len(detections) else np.empty((0, 4), dtype=np.float32)

    return evaluate_detector_predictions(reference, predict, providers)


def evaluate_yunet_detector(model_path, reference):
    input_size = (640, 640)
    detector = cv2.FaceDetectorYN.create(str(model_path), '', input_size, .3, .4, 5000)

    def predict(image):
        im_ratio = image.shape[0] / image.shape[1]
        model_ratio = input_size[1] / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        scale = new_height / image.shape[0]
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        detector.setInputSize(input_size)
        _, faces = detector.detect(canvas)
        if faces is None:
            return np.empty((0, 4), dtype=np.float32)
        boxes = faces[:, :4].copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes / scale

    return evaluate_detector_predictions(reference, predict, ['OpenCV DNN CPU'])


def retinaface_priors(height, width):
    anchors = []
    for min_sizes, step in zip(((16, 32), (64, 128), (256, 512)), (8, 16, 32)):
        map_height = int(np.ceil(height / step))
        map_width = int(np.ceil(width / step))
        for y in range(map_height):
            for x in range(map_width):
                for min_size in min_sizes:
                    anchors.append(((x + .5) * step / width, (y + .5) * step / height,
                                    min_size / width, min_size / height))
    return np.asarray(anchors, dtype=np.float32)


def numpy_nms(detections, threshold):
    if not len(detections):
        return []
    x1, y1, x2, y2, scores = detections.T
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = order[0]
        keep.append(int(index))
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        overlap = width * height / np.maximum(areas[index] + areas[order[1:]] - width * height, 1e-9)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def evaluate_retinaface_detector(model_path, reference, device):
    providers = ['CPUExecutionProvider']
    if device.startswith('cuda') and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    session = onnxruntime.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    input_size = (640, 640)
    priors = retinaface_priors(input_size[1], input_size[0])

    def predict(image):
        im_ratio = image.shape[0] / image.shape[1]
        if im_ratio > 1:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        scale = new_height / image.shape[0]
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.float32)
        canvas[:new_height, :new_width] = resized
        canvas -= (104, 117, 123)
        blob = canvas.transpose(2, 0, 1)[None]
        outputs = session.run(output_names, {input_name: blob})
        located = next(value for value in outputs if value.shape[-1] == 4)[0]
        confidence = next(value for value in outputs if value.shape[-1] == 2)[0]
        centers = priors[:, :2] + located[:, :2] * .1 * priors[:, 2:]
        sizes = priors[:, 2:] * np.exp(located[:, 2:] * .2)
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        boxes *= np.asarray((input_size[0], input_size[1], input_size[0], input_size[1]))
        scores = confidence[:, 1]
        selected = scores >= .3
        boxes, scores = boxes[selected], scores[selected]
        order = scores.argsort()[::-1][:5000]
        detections = np.hstack((boxes[order], scores[order, None])).astype(np.float32)
        detections = detections[numpy_nms(detections, .4)][:750]
        return detections[:, :4] / scale

    return evaluate_detector_predictions(reference, predict, session.get_providers())


def parse_models(values):
    parsed = []
    for value in values:
        name, path = value.split('=', 1)
        parsed.append((name, Path(path)))
    return parsed


def parse_ensembles(values):
    parsed = []
    for value in values:
        name, specification = value.split('=', 1)
        left, right, weight = specification.split(',', 2)
        parsed.append((name, left, right, float(weight)))
    return parsed


def ensemble_speed(left, right, anchors):
    elapsed = left['extraction_seconds'] + right['extraction_seconds']
    timing = {}
    for key in sorted(left['timing'].keys() & right['timing'].keys()):
        size = int(key.split('_', 1)[1])
        p50 = left['timing'][key]['batch_p50_ms'] + right['timing'][key]['batch_p50_ms']
        p95 = left['timing'][key]['batch_p95_ms'] + right['timing'][key]['batch_p95_ms']
        timing[key] = {'batch_p50_ms': p50, 'batch_p95_ms': p95,
                       'per_face_p50_ms': p50 / size,
                       'faces_per_second_from_p50': size * 1000 / p50}
    return {'providers': left['providers'] + right['providers'], 'extraction_seconds': elapsed,
            'faces_per_second': anchors / elapsed, 'timing': timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face-db', type=Path, default=ROOT / 'data/face_database')
    parser.add_argument('--review', type=Path, default=ROOT / 'outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023/review_template.json')
    parser.add_argument('--recognizer', action='append', default=[], metavar='NAME=ONNX')
    parser.add_argument('--sface-recognizer', action='append', default=[], metavar='NAME=ONNX')
    parser.add_argument('--facenet-recognizer', action='append', default=[], metavar='NAME=CHECKPOINT')
    parser.add_argument('--swinface-recognizer', action='append', default=[], metavar='NAME=CHECKPOINT')
    parser.add_argument('--ensemble', action='append', default=[], metavar='NAME=LEFT,RIGHT,LEFT_WEIGHT')
    parser.add_argument('--detector', action='append', default=[], metavar='NAME=ONNX')
    parser.add_argument('--yunet-detector', action='append', default=[], metavar='NAME=ONNX')
    parser.add_argument('--retinaface-detector', action='append', default=[], metavar='NAME=ONNX')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--timing-rounds', type=int, default=10)
    parser.add_argument('--torch-threads', type=int, help='PyTorch CPU threads; ONNX retains its runtime default')
    parser.add_argument('--artifacts-dir', type=Path, help='Save input hashes and per-model embeddings for audit')
    parser.add_argument('--detection-limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timing_rounds < 1:
        parser.error('batch-size and timing-rounds must be positive')
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            parser.error('torch-threads must be positive')
        import torch
        torch.set_num_threads(args.torch_threads)

    faces = load_faces(args.face_db)
    if len(faces) < 2:
        parser.error('at least two labeled faces are required')
    manifest = [{key: value for key, value in row.items() if key != 'image'} for row in faces]
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if args.artifacts_dir:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (args.artifacts_dir / 'face_manifest.json').write_text(manifest_text)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'device': args.device,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'opencv': cv2.__version__, 'numpy': np.__version__,
                        'onnxruntime': onnxruntime.__version__, 'torch_threads_requested': args.torch_threads,
                        'onnx_threads': 'runtime default', 'batch_size': args.batch_size,
                        'timing_rounds': args.timing_rounds},
        'protocol': {'recognition': 'leave-one-source-photograph-out nearest-anchor and verification pairs',
                     'detection': 'IoU>=0.5 agreement with historical SCRFD development boxes; not independent ground truth'},
        'dataset': {'face_database': str(args.face_db.resolve()), 'anchors': len(faces),
                    'manifest_sha256': hashlib.sha256(manifest_text.encode()).hexdigest(),
                    'identity_counts': dict(Counter(row['identity'] for row in faces))},
        'recognizers': {}, 'detectors': {}, 'limitations': [
            'The labeled face anchors are a small development set from one theatrical production, not an independent final test.',
            'Identity folder labels and crop quality were not re-annotated in this run.',
            'Existing anchor crops were not re-verified for a shared five-point alignment before resizing to each native model input.',
            'Historical detection boxes originate from SCRFD and measure regression agreement, not unbiased detector accuracy.',
            'Detector confidence scores are not calibrated across families; the shared numeric threshold is not an equal-precision operating point.',
            'Fusion weights and recognition thresholds were selected on the same development queries reported here.',
            'Latency is single-process sequential inference on the reported host and is not a service SLA.',
        ]}
    embedding_cache = {}
    for name, path in parse_models(args.recognizer):
        print(f'Benchmark recognition: {name}', flush=True)
        embeddings, speed = model_embeddings(path, faces, args.device, args.batch_size, args.timing_rounds)
        embedding_cache[name] = embeddings
        report['recognizers'][name] = {'path': str(path), 'sha256': sha256(path),
                                       'bytes': path.stat().st_size, 'parameters': onnx_parameters(path),
                                       'family': 'ArcFace-compatible ONNX', 'speed': speed,
                                       'quality': {
                                           'folder_identity': evaluate_recognition(faces, embeddings, 'identity'),
                                           'actor_identity': evaluate_recognition(faces, embeddings, 'actor_identity'),
                                       }}
    for name, path in parse_models(args.sface_recognizer):
        print(f'Benchmark recognition: {name}', flush=True)
        embeddings, speed = sface_embeddings(path, faces, args.timing_rounds)
        embedding_cache[name] = embeddings
        report['recognizers'][name] = {'path': str(path), 'sha256': sha256(path),
                                       'bytes': path.stat().st_size, 'parameters': onnx_parameters(path),
                                       'family': 'SFace/OpenCV FaceRecognizerSF', 'speed': speed,
                                       'quality': {
                                           'folder_identity': evaluate_recognition(faces, embeddings, 'identity'),
                                           'actor_identity': evaluate_recognition(faces, embeddings, 'actor_identity'),
                                       }}
    for name, path in parse_models(args.facenet_recognizer):
        print(f'Benchmark recognition: {name}', flush=True)
        embeddings, speed = facenet_embeddings(path, faces, args.device, args.batch_size, args.timing_rounds)
        embedding_cache[name] = embeddings
        report['recognizers'][name] = {'path': str(path), 'sha256': sha256(path),
                                       'bytes': path.stat().st_size, 'parameters': speed.pop('parameters'),
                                       'family': 'FaceNet/InceptionResnetV1 VGGFace2', 'speed': speed,
                                       'quality': {
                                           'folder_identity': evaluate_recognition(faces, embeddings, 'identity'),
                                           'actor_identity': evaluate_recognition(faces, embeddings, 'actor_identity'),
                                       }}
    for name, path in parse_models(args.swinface_recognizer):
        print(f'Benchmark recognition and age: {name}', flush=True)
        embeddings, speed, metadata, age_report = swinface_embeddings(
            path, faces, args.device, args.batch_size, args.timing_rounds)
        embedding_cache[name] = embeddings
        report['recognizers'][name] = {
            'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'parameters': metadata['parameters'], 'family': 'SwinFace/Swin-T multi-task',
            'model': metadata, 'speed': speed, 'age_estimation': age_report,
            'quality': {
                'folder_identity': evaluate_recognition(faces, embeddings, 'identity'),
                'actor_identity': evaluate_recognition(faces, embeddings, 'actor_identity'),
            },
        }
    for name, left, right, left_weight in parse_ensembles(args.ensemble):
        if left not in embedding_cache or right not in embedding_cache:
            raise KeyError(f'Ensemble components must be configured recognizers: {left}, {right}')
        if not 0 <= left_weight <= 1:
            raise ValueError(f'Ensemble weight must be in [0, 1]: {left_weight}')
        embeddings = np.concatenate((np.sqrt(left_weight) * embedding_cache[left],
                                     np.sqrt(1 - left_weight) * embedding_cache[right]), axis=1)
        embedding_cache[name] = embeddings
        left_meta, right_meta = report['recognizers'][left], report['recognizers'][right]
        report['recognizers'][name] = {
            'components': {left: left_weight, right: 1 - left_weight},
            'sha256': {left: left_meta['sha256'], right: right_meta['sha256']},
            'bytes': left_meta['bytes'] + right_meta['bytes'],
            'parameters': left_meta['parameters'] + right_meta['parameters'],
            'family': 'score fusion via weighted concatenation',
            'speed': ensemble_speed(left_meta['speed'], right_meta['speed'], len(faces)),
            'quality': {
                'folder_identity': evaluate_recognition(faces, embeddings, 'identity'),
                'actor_identity': evaluate_recognition(faces, embeddings, 'actor_identity'),
            }}
    needs_reference = args.detector or args.yunet_detector or args.retinaface_detector
    reference = review_reference(args.review, args.detection_limit) if needs_reference else []
    for name, path in parse_models(args.detector):
        report['detectors'][name] = {'path': str(path), 'sha256': sha256(path),
                                     'bytes': path.stat().st_size, 'parameters': onnx_parameters(path),
                                     'family': 'SCRFD',
                                     'agreement': evaluate_scrfd_detector(path, reference, args.device)}
    for name, path in parse_models(args.yunet_detector):
        report['detectors'][name] = {'path': str(path), 'sha256': sha256(path),
                                     'bytes': path.stat().st_size, 'parameters': onnx_parameters(path),
                                     'family': 'YuNet',
                                     'agreement': evaluate_yunet_detector(path, reference)}
    for name, path in parse_models(args.retinaface_detector):
        report['detectors'][name] = {'path': str(path), 'sha256': sha256(path),
                                     'bytes': path.stat().st_size, 'parameters': onnx_parameters(path),
                                     'family': 'RetinaFace',
                                     'agreement': evaluate_retinaface_detector(path, reference, args.device)}
    if args.artifacts_dir:
        for name, embeddings in embedding_cache.items():
            # Hash the user-provided model name so it cannot escape the directory.
            filename = 'embeddings_' + hashlib.sha256(name.encode()).hexdigest()[:16] + '.npz'
            artifact = args.artifacts_dir / filename
            np.savez_compressed(artifact, embeddings=embeddings)
            report['recognizers'][name]['embedding_artifact'] = {
                'path': str(artifact), 'sha256': sha256(artifact), 'shape': list(embeddings.shape),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(args.output), 'recognizers': list(report['recognizers']),
                      'detectors': list(report['detectors'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
