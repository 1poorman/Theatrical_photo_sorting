"""Compare compact SwinFace exports with the frozen official-model face study.

Reports age drift from original predictions, never an age accuracy/MAE claim.
The face manifest and all reference hashes are verified before inference.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_modules.face_recognition.swinface import SwinFaceAnalyzer
from core_modules.face_recognition.swinface.inference import file_sha256
from tools.benchmark_face_stack import evaluate_recognition


def difference(prediction, reference):
    features, original = prediction['embeddings'], reference['embeddings']
    ages = np.abs(prediction['ages'].astype(np.float64) - reference['ages'])
    features64, original64 = features.astype(np.float64), original.astype(np.float64)
    cosine = np.clip(np.sum(features64 * original64, axis=1) / (
        np.linalg.norm(features64, axis=1) * np.linalg.norm(original64, axis=1)), -1, 1)
    return {
        'embedding_cosine_mean': float(cosine.mean()),
        'embedding_cosine_minimum': float(cosine.min()),
        'embedding_maximum_absolute_difference': float(np.max(np.abs(features - original))),
        'pairwise_cosine_maximum_absolute_difference': float(np.max(np.abs(features @ features.T - original @ original.T))),
        'age_mean_absolute_drift_years': float(ages.mean()),
        'age_p95_absolute_drift_years': float(np.percentile(ages, 95)),
        'age_maximum_absolute_drift_years': float(ages.max()),
    }


def query_decisions(rows, embeddings, threshold):
    identities = np.array([row['actor_identity'] for row in rows])
    sources = np.array([row['source'] for row in rows])
    hashes = np.array([row['content_sha256'] for row in rows])
    similarities = embeddings @ embeddings.T
    result = []
    for index, row in enumerate(rows):
        allowed = (sources != sources[index]) & (hashes != hashes[index])
        if not (allowed & (identities == identities[index])).any():
            continue
        candidates = np.where(allowed)[0]
        chosen = candidates[np.argsort(similarities[index, candidates])[::-1]][0]
        score = float(similarities[index, chosen])
        result.append({
            'path': row['path'], 'predicted': str(identities[chosen]), 'score': score,
            'correct': bool(identities[chosen] == identities[index]), 'accepted': score >= threshold,
        })
    return result


def at_fixed_threshold(decisions, threshold):
    correct = sum(row['accepted'] and row['correct'] for row in decisions)
    wrong = sum(row['accepted'] and not row['correct'] for row in decisions)
    total = len(decisions)
    return {
        'threshold': threshold, 'eligible_queries': total,
        'correct_accept_count': correct, 'wrong_accept_count': wrong,
        'correct_accept_rate': correct / total, 'wrong_accept_rate': wrong / total,
        'rejection_rate': (total - correct - wrong) / total,
    }


def load_reference(directory):
    report_path = directory / 'benchmark.json'
    manifest_path = directory / 'recognition/face_manifest.json'
    report = json.loads(report_path.read_text())
    entry = report['recognizers']['swinface']
    artifact = entry['embedding_artifact']
    archive = ROOT / artifact['path']
    if file_sha256(manifest_path) != report['dataset']['manifest_sha256']:
        raise ValueError('Reference face manifest checksum mismatch')
    if file_sha256(archive) != artifact['sha256']:
        raise ValueError('Reference embedding checksum mismatch')
    if file_sha256(ROOT / entry['path']) != entry['sha256']:
        raise ValueError('Original official checkpoint checksum mismatch')
    rows = json.loads(manifest_path.read_text())
    images = []
    for row in rows:
        path = Path(row['path'])
        if file_sha256(path) != row['content_sha256']:
            raise ValueError(f'Input face changed: {path}')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        images.append(image)
    age_rows = entry['age_estimation']['predictions']
    if [row['path'] for row in age_rows] != [row['path'] for row in rows]:
        raise ValueError('Reference age/embedding input order mismatch')
    with np.load(archive, allow_pickle=False) as data:
        embeddings = data['embeddings'].copy()
    if embeddings.shape != (len(rows), 512):
        raise ValueError('Reference embedding shape mismatch')
    return rows, images, {
        'embeddings': embeddings,
        'ages': np.array([row['estimated_age'] for row in age_rows], dtype=np.float32),
    }, {
        'report_path': str(report_path), 'report_sha256': file_sha256(report_path),
        'manifest_path': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
        'embedding_artifact': artifact, 'source_sha256': entry['sha256'],
        'model': entry['model'], 'quality': entry['quality']['actor_identity'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'outputs/acceptance/swinface_20261111')
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--timing-rounds', type=int, default=5)
    args = parser.parse_args()
    if min(args.torch_threads, args.batch_size, args.timing_rounds) < 1:
        parser.error('thread/batch/round counts must be positive')
    if args.output.exists():
        parser.error('output directory already exists; choose a new path to preserve previous results')
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    rows, images, reference, provenance = load_reference(args.baseline_dir)
    threshold = provenance['quality']['calibrated_closed_set_operating_points']['max_wrong_accept_rate_0.02']['threshold']
    reference_decisions = query_decisions(rows, reference['embeddings'], threshold)
    args.output.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / 'core_modules/face_recognition/swinface/inference.py',
                    ROOT / 'core_modules/face_recognition/swinface/compact.py', ROOT / 'tools/benchmark_face_stack.py']
    source_paths.extend(sorted((ROOT / 'core_modules/face_recognition/swinface/_vendor').glob('*.py')))
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'platform': platform.platform(), 'numpy': np.__version__, 'opencv': cv2.__version__,
                        'torch_threads': torch.get_num_threads(), 'interop_threads': 1, 'opencv_threads': 1,
                        'requested_device': args.device, 'batch_size': args.batch_size,
                        'timing_rounds': args.timing_rounds, 'warmup_rounds': 2},
        'protocol': 'same 109 aligned anchors; exclude same source/content; actor identities; development set only',
        'reference': provenance, 'face_count': len(rows),
        'age_ground_truth_count': 0, 'age_mae': None,
        'source_sha256': {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths},
        'variants': {},
    }
    for specification in args.checkpoint:
        label, separator, value = specification.partition('=')
        if not separator or not label or not all(char.isalnum() or char in '_-' for char in label):
            parser.error('checkpoint must use a filename-safe LABEL=PATH')
        if label in report['variants']:
            parser.error('duplicate checkpoint label')
        print(f'{label}: loading {value}', flush=True)
        tick = time.perf_counter()
        model = SwinFaceAnalyzer(value, device=args.device, batch_size=args.batch_size)
        load_seconds = time.perf_counter() - tick
        if model.source_sha256 != provenance['source_sha256']:
            raise ValueError('Candidate was not exported from the reference official checkpoint')
        if model.device.type == 'cuda':
            report['environment']['gpu'] = torch.cuda.get_device_name(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
        tick = time.perf_counter()
        prediction = model.predict(images)
        prediction_seconds = time.perf_counter() - tick
        np.testing.assert_allclose(np.linalg.norm(prediction['embeddings'], axis=1), 1, atol=1e-6)
        decisions = query_decisions(rows, prediction['embeddings'], threshold)
        candidate = {
            'model': model.metadata(), 'path': str(Path(value).resolve()),
            'load_seconds_including_hashes': load_seconds, 'prediction_seconds': prediction_seconds,
            'drift_from_official_cpu': difference(prediction, reference),
            'quality': evaluate_recognition(rows, prediction['embeddings'], 'actor_identity'),
            'fixed_official_threshold': at_fixed_threshold(decisions, threshold),
            'top1_identity_changes_from_official': sum(row['predicted'] != old['predicted'] for row, old in zip(decisions, reference_decisions)),
            'fixed_threshold_acceptance_changes': sum(row['accepted'] != old['accepted'] for row, old in zip(decisions, reference_decisions)),
            'queries': decisions,
            'age_predictions': [{'path': row['path'], 'years': float(age)} for row, age in zip(rows, prediction['ages'])],
        }
        if model.device.type == 'cuda':
            candidate['cuda_peak_allocated_bytes_during_prediction'] = torch.cuda.max_memory_allocated(model.device)
        artifact = args.output / f'{label}_predictions.npz'
        np.savez_compressed(artifact, **prediction)
        candidate['predictions_artifact'] = {'path': str(artifact.resolve()), 'sha256': file_sha256(artifact)}
        print(f'{label}: Top-1 {candidate["quality"]["top1_accuracy"]:.4%}; age mean/max drift '
              f'{candidate["drift_from_official_cpu"]["age_mean_absolute_drift_years"]:.6f}/'
              f'{candidate["drift_from_official_cpu"]["age_maximum_absolute_drift_years"]:.6f} years', flush=True)
        selected = np.linspace(0, len(images) - 1, min(8, len(images)), dtype=int)
        subset = [images[index] for index in selected]
        model.batch_size = 1
        single = model.predict(subset)
        model.batch_size = 3
        uneven = model.predict(subset)
        model.batch_size = len(subset)
        grouped = model.predict(subset)
        separate = {'embeddings': model.extract_features(subset), 'ages': model.estimate_ages(subset)}
        candidate['batch_consistency'] = {
            'faces': len(subset), 'manifest_indices': selected.tolist(),
            'batch1_vs_batch8': difference(single, grouped),
            'batch3_vs_batch8': difference(uneven, grouped),
            'separate_heads_vs_joint': difference(separate, grouped),
        }
        model.batch_size = args.batch_size
        candidate['joint_timing'] = {}
        for size in sorted({1, min(args.batch_size, len(images))}):
            batch = images[:size]
            for _ in range(2):
                model.predict(batch)
            durations = []
            for _ in range(args.timing_rounds):
                tick = time.perf_counter()
                model.predict(batch)
                # predict copies both outputs to CPU, so CUDA is synchronized.
                durations.append((time.perf_counter() - tick) * 1000)
            candidate['joint_timing'][f'batch_{size}'] = {
                'p50_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations, 95)),
                'per_face_p50_ms': float(np.median(durations) / size), 'samples_ms': durations,
            }
            print(f'{label}: batch {size} p50 {np.median(durations):.3f} ms', flush=True)
        if file_sha256(value) != model.checkpoint_sha256:
            raise ValueError('Candidate weights changed during benchmark')
        report['variants'][label] = candidate
        (args.output / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        del model, prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'Report: {args.output / "benchmark.json"}', flush=True)


if __name__ == '__main__':
    main()
