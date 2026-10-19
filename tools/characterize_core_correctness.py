"""Record C1-C4 behavior before/after repair; no model-quality or throughput claim."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_core_upgrade import write_json

MODULES = ('scene_split', 'multimodal_evidence', 'smart_organizer', 'review_queue')


def load_modules(destination, revision=None):
    loaded, hashes = {}, {}
    for name in MODULES:
        relative = f'core_modules/organize/{name}.py'
        content = (subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=ROOT)
                   if revision else (ROOT / relative).read_bytes())
        target = destination / f'{name}.py'
        target.write_bytes(content)
        spec = importlib.util.spec_from_file_location(f'characterize_{destination.name}_{name}', target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return loaded, hashes


def scenarios(modules, temporary):
    observations = {}
    paths = [f'p{i}' for i in range(16)]
    matrix = np.asarray([[i, 1] for i in range(16)])
    captures = []
    def capture(vectors):
        captures.append(np.asarray(vectors).copy())
        return np.arange(len(vectors)) % 2
    def timestamp(path):
        i = int(path[1:])
        return datetime(2020, 1, 1) + timedelta(seconds=i + (1000 if i >= 8 else 0))
    with patch.object(modules['scene_split'], 'read_exif_datetime', side_effect=timestamp), \
         patch('sklearn.cluster.KMeans') as cluster, \
         patch('sklearn.metrics.silhouette_score', return_value=.8):
        cluster.return_value.fit_predict.side_effect = capture
        modules['scene_split'].split_scenes(paths, embeddings=matrix)
    observations['C1'] = {'observed': captures[2][:, 0].astype(int).tolist(),
                          'expected': list(range(8, 16))}

    image = temporary / 'photo.jpg'
    image.touch()
    scenes = [{'scene_id': f'act-{i:02d}', 'act': i, 'title': f'scene{i}',
               'label': f'scene{i}', 'text': f'description{i}', 'source_ref': str(i)} for i in range(1, 7)]
    embedder = SimpleNamespace(model_signature='synthetic', extract_features=lambda path: np.array([1, 0]),
                               encode_text=lambda text: np.array([1, 0] if 'scene6' in text else [0, 1]))
    retriever = modules['multimodal_evidence'].MultimodalRetriever(embedder)
    _, ranked, _ = retriever.rank(str(image), {'scenes': scenes}, top_k=1)
    observations['C2'] = {'observed': ranked[0][1]['scene_id'], 'expected': 'act-06'}

    extractor = SimpleNamespace(align_face=lambda image, landmarks: np.zeros((160, 160, 3)),
                                extract_features=lambda images: np.array([[1., 0.]]),
                                l2_normalize=lambda vectors: vectors)
    detector = SimpleNamespace(_detect_single=lambda image: [SimpleNamespace(landmark=[])])
    organizer = modules['smart_organizer'].SmartOrganizer(SimpleNamespace(extractor=extractor, detector=detector), 'unused')
    organizer._lib_matrix = np.array([[1., 0.], [.8, 0.]])
    organizer._lib_persons = ['A', 'B']
    hits, _ = organizer._recognize_image(np.zeros((160, 160, 3)), .55)
    observations['C3'] = {'observed': sorted(hits), 'expected': ['A']}

    queue = modules['review_queue'].ReviewQueue(str(temporary / 'review'))
    queue.seed([{'image': 'photo.jpg', 'label': 'old'}])
    queue.seed([{'image': 'photo.jpg', 'label': 'new'}])
    observations['C4'] = {'observed': queue.records()['photo.jpg']['label'], 'expected': 'new'}
    for row in observations.values():
        row['passes_fixed_contract'] = row['observed'] == row['expected']
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='391ef98')
    parser.add_argument('--output', default='outputs/acceptance/core_upgrade/first_pass')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats 必须大于零')
    baseline = subprocess.check_output(['git', 'rev-parse', '--verify', args.revision], cwd=ROOT, text=True).strip()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    with tempfile.TemporaryDirectory(prefix='theatre-correctness-') as temporary:
        for kind, revision in [('raw', baseline), ('corrected', None)]:
            destination = Path(temporary) / kind
            destination.mkdir()
            modules, hashes = load_modules(destination, revision)
            rounds, timings = [], []
            for repeat in range(args.repeats):
                fixture = destination / f'round-{repeat}'
                fixture.mkdir()
                started = time.perf_counter()
                rounds.append(scenarios(modules, fixture))
                timings.append(time.perf_counter() - started)
            stable = all(row == rounds[0] for row in rounds)
            passed = sum(row['passes_fixed_contract'] for row in rounds[0].values())
            report = {'scope': 'synthetic_C1_C4_correctness_only', 'baseline_kind': kind,
                      'status': 'passed' if passed == 4 and stable else 'failed',
                      'created_at': datetime.now(timezone.utc).isoformat(),
                      'code_sha': baseline if revision else subprocess.check_output(
                          ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      'worktree_code': revision is None, 'module_sha256': hashes,
                      'cases': rounds[0], 'passes_fixed_contract': passed, 'total_cases': 4,
                      'repeats': args.repeats, 'stable_observations': stable,
                      'fixture_seconds': [round(value, 6) for value in timings],
                      'median_fixture_seconds': round(statistics.median(timings), 6),
                      'real_model_quality': 'not_run', 'end_to_end_performance': 'not_run',
                      'limitations': 'Synthetic vectors and mocked EXIF/clustering; shared dependencies come from the worktree.'}
            write_json(output / f'baseline_{kind}.json', report)
            reports[kind] = {key: report[key] for key in ('status', 'passes_fixed_contract', 'total_cases', 'stable_observations')}
    print(json.dumps(reports, indent=2))
    if reports['raw']['passes_fixed_contract'] != 0 or reports['corrected']['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
