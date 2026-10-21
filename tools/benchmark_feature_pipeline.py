"""Compare corrected sequential and shared batch features on original photos.

Each mode runs in a separate process, with the same weights, photos, gallery,
keep quota and warmed models. Only task-owned temporary input/output directories
are written. Quality labels are not inferred from these performance measurements.
"""
import argparse
from collections import defaultdict
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'theatre-core-mpl'))
from tools.evaluate_core_upgrade import write_json

SOURCE_FILES = ['core_modules/image_search.py', 'core_modules/organize/feature_cache.py',
                'core_modules/organize/smart_organizer.py', 'core_modules/organize/burst_dedup.py',
                'core_modules/organize/multimodal_evidence.py',
                'core_modules/organize/filename_parser.py', 'core_modules/organize/scene_split.py',
                'core_modules/organize/face_db_builder.py', 'core_modules/tools/image_io.py',
                'core_modules/face_recognition/face_recognition.py',
                'core_modules/face_recognition/scrfd/scrfd_det.py',
                'core_modules/face_recognition/arcface/arcface_onnx.py', 'config/base.py',
                'tools/benchmark_feature_pipeline.py']


def source_hashes():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_FILES}


def gpu_environment(index):
    """Record competing GPU work without collecting process command lines."""
    result = {}
    for name, query in [('device', '--query-gpu=index,uuid,driver_version,memory.total,memory.used,utilization.gpu'),
                        ('processes', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')]:
        try:
            result[name] = subprocess.check_output(
                ['nvidia-smi', '-i', str(index), query, '--format=csv'],
                text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = {'unavailable': type(exc).__name__}
    return result


class Resources:
    """Sample process RSS and whole-device GPU memory; no additional dependency."""
    def __init__(self, index):
        self.stop = threading.Event()
        self.peak_rss = self.peak_gpu = 0
        self.nvml = None
        try:
            self.nvml = ctypes.CDLL('libnvidia-ml.so.1')
            if self.nvml.nvmlInit_v2() != 0:
                raise RuntimeError('NVML initialization failed')
            self.handle = ctypes.c_void_p()
            if self.nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
                raise RuntimeError('NVML device unavailable')
        except (OSError, RuntimeError):
            self.nvml = None

    def gpu(self):
        if self.nvml is None:
            return None
        class Memory(ctypes.Structure):
            _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
        memory = Memory()
        if self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) != 0:
            return None
        return {'total': memory.total, 'free': memory.free, 'used': memory.used}

    def sample(self):
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        self.peak_rss = max(self.peak_rss, rss)
        memory = self.gpu()
        if memory is not None:
            self.peak_gpu = max(self.peak_gpu, memory['used'])

    def run(self):
        while not self.stop.wait(.05):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()


class ForwardCounter:
    """Count actual Torch module forwards, including batch and fallback calls."""
    def __init__(self, embedder, device):
        import torch
        self.counts = {'image': 0, 'text': 0}
        self.events = {'image': [], 'text': []}
        self.handles = []
        self.device = device
        for kind, module in [('image', embedder.model.vision_model), ('text', embedder.model.text_model)]:
            stack = []
            def before(module, args, kind=kind, stack=stack):
                self.counts[kind] += 1
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                stack.append(event)
            def after(module, args, output, kind=kind, stack=stack):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.events[kind].append((stack.pop(), end))
            self.handles.extend([module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after, always_call=True)])

    def finish(self):
        import torch
        torch.cuda.synchronize(self.device)
        for handle in self.handles:
            handle.remove()
        return {'calls': self.counts,
                'gpu_seconds': {kind: sum(a.elapsed_time(b) for a, b in pairs) / 1000
                                for kind, pairs in self.events.items()}}


def load_baseline(output, component):
    path = output / f'corrected_{component}.py'
    spec = importlib.util.spec_from_file_location(f'_u2_corrected_{component}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured(call, model, device, index):
    import torch
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    counter = ForwardCounter(model, device)
    try:
        with Resources(index) as resources:
            start = time.perf_counter()
            result = call()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
    finally:
        forwards = counter.finish()
    return result, {'seconds': elapsed, 'forward': forwards,
                    'rss_peak_bytes': resources.peak_rss,
                    'device_used_peak_bytes': resources.peak_gpu or None,
                    'torch_allocated_peak_bytes': torch.cuda.max_memory_allocated(device),
                    'torch_reserved_peak_bytes': torch.cuda.max_memory_reserved(device)}


def decisions_from(results):
    return [{'folder': folder, 'total_input': result['total_input'], 'kept': result['kept'],
             'n_scenes': result['n_scenes'], 'records': [
                 {key: row.get(key) for key in ('source', 'organized_name', 'people_detected', 'error')} |
                 {'recognized_actors': sorted(row.get('recognized', {}))} for row in result['records']]}
            for folder, result in results]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def setup(args, output):
    from core_modules.organize.feature_cache import ContentHasher
    inventory = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    unique = {}
    for row in inventory:
        if row['status'] == 'ok' and row['source_kind'] == 'stage_photo':
            unique.setdefault(row['image_id'], row)
    by_play = defaultdict(list)
    for row in unique.values():
        by_play[row['play_id']].append(row)
    selected = []
    while any(by_play.values()) and len(selected) < args.limit:
        for play in sorted(by_play):
            if by_play[play] and len(selected) < args.limit:
                selected.append(by_play[play].pop(0))
    hasher = ContentHasher()
    for row in selected:
        source = (Path(args.source_root) / row['source']).resolve()
        if hasher.fingerprint(source)[0] != row['content_hash']:
            raise ValueError(f'清单生成后源图已变化: {source.name}')
        row['path'] = str(source)
    if not selected:
        raise ValueError('没有可测量的舞台照')
    scenes = json.loads(Path(args.scenes).read_text())
    texts = [scene.get('label', '') + ' ' + scene.get('text', '') for scene in scenes]
    write_json(output / 'inputs.json', {'images': selected, 'texts': texts})
    for component, path in [('image_search', 'core_modules/image_search.py'),
                            ('burst_dedup', 'core_modules/organize/burst_dedup.py'),
                            ('smart_organizer', 'core_modules/organize/smart_organizer.py')]:
        source = subprocess.check_output(['git', 'show', f'{args.baseline}:{path}'], cwd=ROOT)
        (output / f'corrected_{component}.py').write_bytes(source)
    gallery = output / 'gallery'
    shutil.copytree(args.gallery, gallery, dirs_exist_ok=True)
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
    write_json(output / 'provenance.json', {
        'source_sha256': source_hashes(),
        'gallery_sha256': {path.relative_to(gallery).as_posix(): hasher.fingerprint(path)[0]
                           for path in sorted(gallery.rglob('*')) if path.is_file()},
        'face_weights_sha256': {str(path): hasher.fingerprint(path)[0]
                               for path in (SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH)},
        'baseline_commit': subprocess.check_output(['git', 'rev-parse', args.baseline], cwd=ROOT, text=True).strip(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'inputs_sha256': hashlib.sha256((output / 'inputs.json').read_bytes()).hexdigest(),
        'parameters': vars(args)})


def worker(args, output):
    import torch
    import transformers
    from core_modules.image_search import ImageEmbedder
    from core_modules.organize.feature_cache import FeatureService
    from core_modules.organize.smart_organizer import SmartOrganizer
    from config.base import FEATURE_DECODE_WORKERS
    provenance = json.loads((output / 'provenance.json').read_text())
    if provenance['source_sha256'] != source_hashes():
        raise RuntimeError('测量期间源码已变化，请使用新输出目录重新运行')
    inputs = json.loads((output / 'inputs.json').read_text())
    paths, texts = [row['path'] for row in inputs['images']], inputs['texts']
    index = int(args.device.partition(':')[2] or 0)
    torch.cuda.set_device(index)
    environment_before = gpu_environment(index)
    before_load = Resources(index).gpu()
    started = time.perf_counter()
    model = ImageEmbedder('siglip2_base', device=args.device)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    # Compute the immutable model/processor identity outside warmed inference timing.
    identity = {'model': model.model_signature, 'preprocessing': model.preprocessing_signature}
    signature_seconds = time.perf_counter() - started
    baseline = args.worker == 'baseline'
    if baseline:
        legacy_type = load_baseline(output, 'image_search').ImageEmbedder
        legacy = legacy_type.__new__(legacy_type)
        legacy.__dict__.update(model.__dict__)
        encoder = legacy
        organizer_module = load_baseline(output, 'smart_organizer')
        organizer_module.BurstDeduplicator = load_baseline(output, 'burst_dedup').BurstDeduplicator
        organizer_type = organizer_module.SmartOrganizer
    else:
        encoder, organizer_type = model, SmartOrganizer
    # Prewarm only the mode being measured, then use a fresh feature cache each run.
    if baseline:
        encoder.extract_features(paths[0])
        encoder.encode_text(texts[0])
    else:
        model.extract_features_batch(paths[:args.batch_size], sub_batch_size=args.batch_size)
        model.encode_text_batch(texts[:args.batch_size])
    report = {'mode': args.worker, 'status': 'running', 'identity': identity,
              'sample_count': len(paths), 'repeats': args.repeats,
              'feature_runs': [], 'pipeline_runs': [], 'load_seconds': load_seconds,
              'signature_seconds': signature_seconds,
              'environment': {'device': args.device, 'gpu': torch.cuda.get_device_name(index),
                              'torch': torch.__version__, 'transformers': transformers.__version__,
                              'cpu_count': os.cpu_count(), 'torch_cpu_threads': torch.get_num_threads(),
                              'decode_workers': FEATURE_DECODE_WORKERS, 'batch_size': args.batch_size,
                              'gpu_before': environment_before,
                              'before_load_gpu': before_load, 'memory_sampling_seconds': .05},
              'pipeline_configuration': {'shot_classifier': None, 'classify_role': False,
                                         'keep_per_bucket': 2, 'face_device': args.face_device,
                                         'gallery': 'isolated_copy', 'feature_cache': 'empty_each_run'}}
    target = output / (args.worker + '.json')
    for repeat in range(args.repeats):
        with tempfile.TemporaryDirectory(prefix='features-', dir=output) as cache_dir:
            features = None
            if baseline:
                def encode():
                    return ({path: encoder.extract_features(path) for path in paths},
                            {text: encoder.encode_text(text) for text in texts})
            else:
                def encode():
                    nonlocal features
                    features = FeatureService(model, cache_dir=cache_dir, batch_size=args.batch_size)
                    return features.images(paths), features.texts(texts)
            (images, documents), sample = measured(encode, model, args.device, index)
            if set(images) != set(paths) or set(documents) != set(texts) or any(
                    vector is None for vector in list(images.values()) + list(documents.values())):
                raise RuntimeError('部分特征缺失，停止完整样本性能声明')
            if features:
                sample['features'] = features.report()
            report['feature_runs'].append(sample)
            if repeat == 0:
                np.savez(output / (args.worker + '_vectors.npz'),
                         images=np.stack([images[path] for path in paths]),
                         texts=np.stack([documents[text] for text in texts]))
                if not baseline:
                    fresh = None
                    def cache_hit():
                        nonlocal fresh
                        fresh = FeatureService(model, cache_dir=cache_dir, batch_size=args.batch_size)
                        return fresh.images(paths), fresh.texts(texts)
                    _, hot = measured(cache_hit, model, args.device, index)
                    hot['features'] = fresh.report()
                    report['cache_hit_run'] = hot
        write_json(target, report)
        print(json.dumps({'mode': args.worker, 'stage': 'features', 'repeat': repeat + 1,
                          'seconds': sample['seconds']}, ensure_ascii=False), flush=True)
    if not args.skip_pipeline:
        from core_modules.face_recognition.face_recognition import FaceRecognitionSystem
        from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH
        system = FaceRecognitionSystem(SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH, device=args.face_device)
        report['face_providers'] = system.extractor.session.get_providers()
        organizer_type(system, str(output / 'gallery'))._load_library()
        face_errors = []
        for component, name in [(system.detector, '_detect_single'), (system.extractor, 'extract_features')]:
            original = getattr(component, name)
            def checked(*values, original=original, name=name, **kwargs):
                try:
                    return original(*values, **kwargs)
                except Exception as exc:
                    face_errors.append({'method': name, 'error': type(exc).__name__})
                    raise
            setattr(component, name, checked)
        with tempfile.TemporaryDirectory(prefix='pipeline-input-', dir=output) as input_root:
            groups = defaultdict(list)
            for row in inputs['images']:
                groups[str(Path(row['path']).parent)].append(row)
            folders = []
            for original_dir, rows in sorted(groups.items()):
                # Filename parsing branches on this directory marker. Preserve it
                # when measuring isolated copies of the original workflow.
                marker = '-整理完成' if '整理完成' in original_dir else ''
                folder = Path(input_root) / (hashlib.sha256(original_dir.encode()).hexdigest()[:12] + marker)
                folder.mkdir()
                for row in rows:
                    (folder / Path(row['path']).name).symlink_to(row['path'])
                folders.append(folder)
            # Warm detector + face extractor without performing measured archival copies.
            import cv2
            from core_modules.tools.image_io import resize_max_side
            from config.base import FACE_IMAGE_MAX_SIDE
            warm = organizer_type(system, str(output / 'gallery'))
            warm._recognize_image(resize_max_side(cv2.imread(paths[0]), FACE_IMAGE_MAX_SIDE), .55)
            for repeat in range(args.repeats):
                with tempfile.TemporaryDirectory(prefix='pipeline-output-', dir=output) as run_dir:
                    def organize(destination=None):
                        destination = Path(destination or run_dir)
                        reports = []
                        for folder in folders:
                            options = {} if baseline else {'feature_cache_dir': str(Path(run_dir) / 'features'),
                                                           'feature_batch_size': args.batch_size}
                            organizer = organizer_type(system, str(output / 'gallery'), embedder=encoder, **options)
                            result = organizer.organize(str(folder), str(destination / folder.name), classify_role=False)
                            reports.append((folder.name, result))
                        return reports
                    results, sample = measured(organize, model, args.device, index)
                    decisions = decisions_from(results)
                    sample['decisions_sha256'] = digest(decisions)
                    quality = {}
                    for folder, _ in results:
                        dedup = json.loads((Path(run_dir) / folder / 'dedup_report.json').read_text())
                        quality[folder] = {member['name']: member['quality']
                                           for group in dedup['groups'] for member in group['members']}
                    sample['quality_sha256'] = digest(quality)
                    sample['total_input'] = sum(result['total_input'] for _, result in results)
                    sample['kept'] = sum(result['kept'] for _, result in results)
                    sample['errors'] = sum(bool(row.get('error')) for _, result in results for row in result['records'])
                    sample['face_errors'] = list(face_errors)
                    if not baseline:
                        sample['stage_timings'] = {key: sum(result['timings'].get(key, 0) for _, result in results)
                                                   for key in results[0][1]['timings']}
                        sample['features_by_directory'] = {folder: result['scene_features']
                                                           for folder, result in results}
                        sample['feature_errors'] = {path: error for _, result in results
                                                    for path, error in result['scene_features']['image_errors'].items()}
                    if repeat == 0:
                        write_json(output / (args.worker + '_decisions.json'), decisions)
                        write_json(output / (args.worker + '_quality.json'), quality)
                    report['pipeline_runs'].append(sample)
                    if not baseline and repeat == args.repeats - 1:
                        # Analysis uses original paths; organization uses directory
                        # aliases. Both must reuse the same content-addressed entries.
                        analysis = FeatureService(model, cache_dir=Path(run_dir) / 'features',
                                                  batch_size=args.batch_size)
                        analysis.images(paths)
                        analysis.texts(texts)
                        hot_results, hot = measured(lambda: organize(Path(run_dir) / 'hot'),
                                                    model, args.device, index)
                        hot['decisions_sha256'] = digest(decisions_from(hot_results))
                        hot['features_by_directory'] = {folder: result['scene_features']
                                                        for folder, result in hot_results}
                        report['pipeline_cache_hit_run'] = hot
                write_json(target, report)
                print(json.dumps({'mode': args.worker, 'stage': 'pipeline', 'repeat': repeat + 1,
                                  'seconds': sample['seconds'], 'kept': sample['kept']}, ensure_ascii=False), flush=True)
    report['status'] = 'measured'
    report['environment']['gpu_after'] = gpu_environment(index)
    write_json(target, report)


def summarize(args, output):
    baseline, batch = [json.loads((output / (mode + '.json')).read_text()) for mode in ('baseline', 'batch')]
    inputs = json.loads((output / 'inputs.json').read_text())
    comparisons = {}
    with np.load(output / 'baseline_vectors.npz') as first, np.load(output / 'batch_vectors.npz') as second:
        for kind in ('images', 'texts'):
            a, b = first[kind], second[kind]
            cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            comparisons[kind] = {'min_cosine': float(cosine.min()), 'fraction_at_least_0999': float((cosine >= .999).mean()),
                                 'below_threshold_indices': np.flatnonzero(cosine < .999).tolist()}
    timing = {}
    for stage in ('feature_runs', 'pipeline_runs'):
        if not baseline[stage] or not batch[stage]:
            continue
        old, new = [statistics.median(row['seconds'] for row in report[stage]) for report in (baseline, batch)]
        rss_old, rss_new = [max(row['rss_peak_bytes'] for row in report[stage]) for report in (baseline, batch)]
        timing[stage] = {'baseline_median_seconds': old, 'batch_median_seconds': new,
                         'speedup': old / new, 'wall_time_reduction': 1 - new / old,
                         'images_per_second': {'baseline': len(inputs['images']) / old,
                                               'batch': len(inputs['images']) / new},
                         'rss_peak_bytes': {'baseline': rss_old, 'batch': rss_new},
                         'rss_peak_ratio': rss_new / rss_old,
                         'distribution': {mode: {'p95_seconds': float(np.percentile(
                             [row['seconds'] for row in report[stage]], 95)),
                             'min_seconds': min(row['seconds'] for row in report[stage]),
                             'max_seconds': max(row['seconds'] for row in report[stage])}
                             for mode, report in [('baseline', baseline), ('batch', batch)]}}
    provenance = json.loads((output / 'provenance.json').read_text())
    if provenance['source_sha256'] != source_hashes():
        raise RuntimeError('测量期间源码已变化；保留原始测量，不生成混合版本结论')
    gpu_ratios = []
    for mode in (baseline, batch):
        initial = mode['environment']['before_load_gpu']
        if initial:
            peaks = [row['device_used_peak_bytes'] for row in mode['feature_runs'] + mode['pipeline_runs']]
            if peaks and all(peaks):
                gpu_ratios.append(max(0, max(peaks) - initial['used']) / initial['free'])
    report = {'status': 'measured', 'baseline_commit': provenance['baseline_commit'],
              'code_commit': provenance['code_commit'],
              'source_sha256': provenance['source_sha256'],
              'inputs_sha256': provenance['inputs_sha256'],
              'sample_count': len(inputs['images']), 'play_count': len({row['play_id'] for row in inputs['images']}),
              'directory_count': len({str(Path(row['path']).parent) for row in inputs['images']}),
              'text_count': len(inputs['texts']),
              'repeats': args.repeats, 'comparison': comparisons, 'timing': timing,
              'cache_hit_forward_calls': batch['cache_hit_run']['forward']['calls'],
              'pipeline_cache_hit_forward_calls': batch.get('pipeline_cache_hit_run', {}).get('forward', {}).get('calls'),
              'pipeline_decisions_equal': len({row['decisions_sha256'] for row in
                                               baseline['pipeline_runs'] + batch['pipeline_runs']}) == 1
                                              if batch['pipeline_runs'] else None,
              'quality_scores_equal': len({row['quality_sha256'] for row in
                                            baseline['pipeline_runs'] + batch['pipeline_runs']}) == 1
                                           if batch['pipeline_runs'] else None,
              'peak_fraction_of_initial_free_gpu': gpu_ratios,
              'quality': 'not_evaluated_requires_verified_labels',
              'limitations': ['Stage-photo classification and performance grouping are provisional.',
                              'Pipeline uses real SCRFD/ArcFace and SigLIP2, without shot or role classifiers.',
                              'Device memory includes other processes and is sampled every 50 ms.',
                              'This benchmark does not measure HTTP/UI latency or DALI.']}
    checks = {'minimum_size': report['sample_count'] >= 200 and report['directory_count'] >= 3 and args.repeats >= 5,
              'cosine': all(value['fraction_at_least_0999'] >= .999 for value in comparisons.values()),
              'cache_zero_forwards': report['cache_hit_forward_calls'] == {'image': 0, 'text': 0},
              'analysis_to_organizer_cache': report['pipeline_cache_hit_forward_calls'] == {'image': 0, 'text': 0}
                  and batch['pipeline_cache_hit_run']['decisions_sha256'] == batch['pipeline_runs'][0]['decisions_sha256'],
              'feature_speedup': timing['feature_runs']['speedup'] >= 1.5,
              'pipeline_speedup': timing.get('pipeline_runs', {}).get('wall_time_reduction', -1) >= .2,
              'pipeline_decisions_equal': report['pipeline_decisions_equal'] is True,
              'quality_scores_equal': report['quality_scores_equal'] is True,
              'rss_increase': all(value['rss_peak_ratio'] <= 1.25 for value in timing.values()),
              'gpu_headroom': len(gpu_ratios) == 2 and max(gpu_ratios) <= .85,
              'no_errors': all(not row.get('errors') and not row.get('face_errors') and not row.get('feature_errors')
                               for row in batch['pipeline_runs'] + baseline['pipeline_runs'])}
    report['checks'] = checks
    report['performance_status'] = 'passed' if all(checks.values()) else 'not_passed'
    write_json(output / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/acceptance/core_upgrade/u2')
    parser.add_argument('--manifest', default='outputs/acceptance/core_upgrade/first_pass/manifest.jsonl')
    parser.add_argument('--scenes', default='outputs/acceptance/core_upgrade/first_pass/exploratory_scenes.json')
    parser.add_argument('--source-root', default='data/ncpa_test')
    parser.add_argument('--gallery', default='data/face_database')
    parser.add_argument('--baseline', default='fdc1fda')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--face-device', choices=['cuda', 'tensorrt'], default='cuda')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--skip-pipeline', action='store_true')
    parser.add_argument('--worker', choices=['baseline', 'batch'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.limit, args.repeats, args.batch_size) < 1 or args.device != 'cuda:0':
        parser.error('数量参数必须为正；当前完整人脸流水线固定使用 cuda:0')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, output)
        return
    if any(output.iterdir()):
        parser.error('输出目录必须为空，以免覆盖已有测量；请指定新的 --output')
    setup(args, output)
    for mode in ('baseline', 'batch'):
        print(f'Starting {mode}; progress in {output / (mode + ".log")}', flush=True)
        with (output / (mode + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', mode],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize(args, output)


if __name__ == '__main__':
    main()
