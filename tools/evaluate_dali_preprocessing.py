"""Evaluate optional DALI against a completed Pillow benchmark, in a new directory.

This experimental adapter is not imported by the application. Upscaling, non-RGB
JPEGs, EXIF transforms, other formats, and decoder failures use the validated Pillow
path. A full-image consistency failure stops the experiment before sorting.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/theatre-core-mpl')
from core_modules.image_search import ImageEmbedder
from core_modules.organize.feature_cache import ContentHasher, FeatureService, signature
from tools import benchmark_feature_pipeline as benchmark


class DaliImageEmbedder(ImageEmbedder):
    """Bounded experimental JPEG decoder with explicit per-path fallback."""

    def __init__(self, model_name='siglip2_base', device='cuda:0', image_backend=None):
        super().__init__(model_name, device, image_backend='pillow')
        processor = self.processor.image_processor
        self._dali_size = dict(processor.size)
        if (type(processor).__name__ != 'SiglipImageProcessor'
                or set(self._dali_size) != {'height', 'width'}
                or int(processor.resample) != 2
                or not processor.do_resize or not processor.do_rescale or not processor.do_normalize
                or list(processor.image_mean) != [0.5]*3 or list(processor.image_std) != [0.5]*3
                or processor.rescale_factor != 1/255):
            raise ValueError('DALI experiment requires the validated fixed-size bilinear SigLIP processor')
        self._dali_version = importlib.metadata.version('nvidia-dali-cuda120')
        self._dali_pipeline = None
        self._dali_lock = threading.RLock()
        self.dali_statistics = {'decoded': 0, 'fallback': 0, 'decoder_failures': 0}
        self.dali_errors = []

    @property
    def preprocessing_signature(self):
        return signature({'reference': super().preprocessing_signature,
                          'adapter': 'dali-jpeg-rgb-bilinear-no-upscale-v2',
                          'dali': self._dali_version, 'jpeg_fancy_upsampling': True,
                          'fallback': 'pillow-upscale-exif-other-formats-and-errors'})

    def _pipeline(self):
        if self._dali_pipeline is None:
            from nvidia.dali import fn, types, pipeline_def
            index = int(str(self.device).partition(':')[2] or 0)
            size = self._dali_size

            @pipeline_def(batch_size=8, num_threads=2, device_id=index, prefetch_queue_depth=1,
                          exec_async=False, exec_pipelined=False)
            def build():
                encoded = fn.external_source(name='encoded', device='cpu')
                decoded = fn.decoders.image(encoded, device='mixed', output_type=types.RGB,
                                            affine=False, jpeg_fancy_upsampling=True)
                resized = fn.resize(decoded, device='gpu', resize_x=size['width'], resize_y=size['height'],
                                    interp_type=types.INTERP_LINEAR, antialias=True)
                return fn.crop_mirror_normalize(resized, device='gpu', dtype=types.FLOAT,
                                                output_layout='CHW', mean=[127.5]*3, std=[127.5]*3)
            self._dali_pipeline = build()
            self._dali_pipeline.build()
        return self._dali_pipeline

    def extract_features(self, image_path):
        vectors, valid = self.extract_features_batch([image_path], sub_batch_size=1)
        return vectors[0] if valid else None

    def extract_features_batch(self, image_paths, sub_batch_size=8):
        import torch
        vectors, valid = [], []
        size = min(8, max(1, int(sub_batch_size)))
        with self._dali_lock:
            for start in range(0, len(image_paths), size):
                paths = image_paths[start:start+size]
                encoded, accelerated, fallback, by_path = [], [], [], {}
                for path in paths:
                    try:
                        with Image.open(path) as image:
                            eligible = (image.format == 'JPEG' and image.mode == 'RGB'
                                        and image.getexif().get(274, 1) == 1
                                        and image.width >= self._dali_size['width']
                                        and image.height >= self._dali_size['height'])
                        if eligible:
                            encoded.append(np.fromfile(path, dtype=np.uint8))
                            accelerated.append(path)
                        else:
                            fallback.append(path)
                    except (OSError, ValueError):
                        fallback.append(path)
                if accelerated:
                    try:
                        pipe = self._pipeline()
                        pipe.feed_input('encoded', encoded)
                        # DLPack transfers ownership of the output view and
                        # synchronizes DALI's producer stream with Torch.
                        output = pipe.run()[0]
                        pixels = torch.from_dlpack(output.as_tensor())
                        result = self._encode_prepared([{'pixel_values': pixels}])
                        by_path.update(zip(accelerated, result))
                        self.dali_statistics['decoded'] += len(accelerated)
                    except Exception as exc:
                        self.dali_statistics['decoder_failures'] += len(accelerated)
                        if not hasattr(self, 'dali_errors'):
                            self.dali_errors = []
                        self.dali_errors.append({'type': type(exc).__name__, 'message': str(exc)[:500]})
                        self._dali_pipeline = None
                        fallback.extend(accelerated)
                for path in fallback:
                    # Call the reference implementation directly to avoid
                    # retrying the DALI batch or losing valid neighbors.
                    value = ImageEmbedder.extract_features(self, path)
                    self.dali_statistics['fallback'] += 1
                    if value is not None:
                        by_path[path] = value
                for path in paths:
                    if path in by_path:
                        vectors.append(by_path[path])
                        valid.append(path)
        return (np.stack(vectors), valid) if vectors else (np.array([]), [])

    def close(self):
        self._dali_pipeline = None
        super().close()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine_report(reference, candidate):
    cosine = (reference*candidate).sum(axis=1) / (
        np.linalg.norm(reference, axis=1)*np.linalg.norm(candidate, axis=1))
    return {'min_cosine': float(cosine.min()),
            'fraction_at_least_0999': float((cosine >= .999).mean()),
            'below_threshold_indices': np.flatnonzero(cosine < .999).tolist()}


def fixture_check(model, directory):
    directory.mkdir()
    y, x = np.mgrid[:96, :144]
    image = Image.fromarray(np.stack((x*255/143, y*255/95, ((x//12+y//12)%2)*255), -1).astype('uint8'))
    paths = []
    for orientation in range(1, 9):
        path = directory / f'exif-{orientation}.jpg'
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif)
        paths.append(str(path))
    for mode, extension in [('L', 'png'), ('RGBA', 'png'), ('CMYK', 'jpg'), ('P', 'png')]:
        path = directory / f'mode-{mode}.{extension}'
        image.convert(mode).save(path)
        paths.append(str(path))
    # Keep the original small-image failure case above and add an eligible
    # downscale case, so falling back for every fixture cannot pass this gate.
    large = directory / 'downscale.jpg'
    image.resize((720, 480)).save(large)
    paths.append(str(large))
    reference = np.stack([ImageEmbedder.extract_features(model, path) for path in paths])
    broken = directory / 'broken.jpg'
    broken.write_bytes(b'not an image')
    model.dali_statistics = dict.fromkeys(model.dali_statistics, 0)
    result, valid = model.extract_features_batch(paths[:3]+[str(broken)]+paths[3:])
    report = {'mapping_equal': valid == paths, 'bad_image_isolated': str(broken) not in valid,
              'fixture_names': [Path(path).name for path in paths],
              'statistics': dict(model.dali_statistics), 'decoder_errors': list(model.dali_errors),
              'comparison': cosine_report(reference, result)}
    # One real DALI JPEG plus all twelve explicit size/format/orientation fallbacks
    # must be exercised; a missing DALI library cannot pass via CPU fallback.
    report['passed'] = (report['mapping_equal'] and report['bad_image_isolated']
                        and report['statistics']['decoded'] == 1
                        and report['statistics']['decoder_failures'] == 0
                        and report['comparison']['fraction_at_least_0999'] == 1.)
    return report


def preflight(reference, output, previous, audit):
    inputs = json.loads((output/'inputs.json').read_text())
    import torch
    torch.set_num_threads(previous['environment']['torch_cpu_threads'])
    model = DaliImageEmbedder()
    if model.model_signature != previous['identity']['model']:
        raise ValueError('reference model weights/configuration differ')
    if ImageEmbedder.preprocessing_signature.fget(model) != previous['identity']['preprocessing']:
        raise ValueError('reference Pillow processor or dependency versions differ')
    audit['fixtures'] = fixture_check(model, output/'fixtures')
    paths = [row['path'] for row in inputs['images']]
    model.dali_statistics = dict.fromkeys(model.dali_statistics, 0)
    service = FeatureService(model, batch_size=8)
    audit['preflight_identity'] = service.identity
    vectors = service.images(paths)
    if set(vectors) != set(paths):
        raise ValueError('incomplete DALI vectors')
    current = np.stack([vectors[path] for path in paths])
    with np.load(reference/'batch_vectors.npz', allow_pickle=False) as saved:
        audit['comparison'] = cosine_report(saved['images'], current)
    audit['statistics'] = dict(model.dali_statistics)
    audit['decoder_errors'] = list(model.dali_errors)
    audit['feature_errors'] = service.image_errors
    np.savez(output/'preflight_vectors.npz', images=current)
    audit['consistency_passed'] = (audit['fixtures']['passed']
        and audit['comparison']['fraction_at_least_0999'] >= .999
        and not service.image_errors and model.dali_statistics['decoded'] > 0
        and model.dali_statistics['decoder_failures'] == 0)
    benchmark.write_json(output/'evaluation.json', audit)
    model.close()
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', choices=['preflight', 'measure'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    reference, output = Path(args.reference_run).resolve(), Path(args.output).resolve()
    if not args.worker and output.exists() and any(output.iterdir()):
        parser.error('output directory must be empty')
    provenance = json.loads((reference/'provenance.json').read_text())
    previous = json.loads((reference/'batch.json').read_text())
    if provenance['source_sha256'] != benchmark.source_hashes() or previous['status'] != 'measured':
        raise ValueError('reference must be complete and bound to the current source')
    if previous['image_backend'] != 'pillow' or previous['sample_count'] < 200 or previous['repeats'] < 5:
        raise ValueError('reference must measure Pillow on at least 200 images and five repeats')
    if previous['environment']['batch_size'] != 8 or previous['environment']['decode_workers'] != 2:
        raise ValueError('this DALI protocol requires the same eight-image/two-decoder reference')
    if args.worker:
        audit = json.loads((output/'evaluation.json').read_text())
        if audit['evaluator_sha256'] != sha256(__file__):
            raise ValueError('evaluator changed since preparation')
        if args.worker == 'preflight':
            preflight(reference, output, previous, audit)
        else:
            # The optional encoder is substituted only inside this independent
            # worker, so preflight memory does not affect timed measurements.
            import core_modules.image_search as image_search
            created = []
            def factory(*values, **kwargs):
                model = DaliImageEmbedder(*values, **kwargs)
                created.append(model)
                return model
            image_search.ImageEmbedder = factory
            options = argparse.Namespace(**provenance['parameters'])
            options.output, options.worker = str(output), 'batch'
            options.image_backend = 'dali-experimental'
            benchmark.worker(options, output)
            measured = json.loads((output/'batch.json').read_text())
            measured['image_backend'] = 'dali-experimental'
            measured['dali_statistics'] = [model.dali_statistics for model in created]
            measured['dali_errors'] = [model.dali_errors for model in created]
            benchmark.write_json(output/'batch.json', measured)
        return
    output.mkdir(parents=True, exist_ok=True)
    for name in ('inputs.json', 'provenance.json'):
        shutil.copyfile(reference/name, output/name)
    inputs = json.loads((output/'inputs.json').read_text())
    hasher = ContentHasher()
    for row in inputs['images']:
        if hasher.fingerprint(row['path'])[0] != row['content_hash']:
            raise ValueError('source changed since the reference measurement')
    for path, expected in provenance['gallery_sha256'].items():
        if sha256(reference/'gallery'/path) != expected:
            raise ValueError('reference gallery changed')
    for path, expected in provenance['face_weights_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('face weights changed')
    shutil.copytree(reference/'gallery', output/'gallery')
    audit = {'status': 'running', 'reference_run': str(reference),
             'source_sha256': benchmark.source_hashes(), 'evaluator_sha256': sha256(__file__),
             'reference_artifacts': {name: sha256(reference/name) for name in
                                     ('provenance.json', 'inputs.json', 'batch.json', 'batch_vectors.npz')},
             'dali_version': importlib.metadata.version('nvidia-dali-cuda120'),
             'limitations': ['Sequential measurements on a shared GPU.',
                             'No independent quality labels; no HTTP/UI measurement.']}
    benchmark.write_json(output/'evaluation.json', audit)
    def run_worker(mode):
        with (output/(mode+'.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--reference-run', str(reference), '--output', str(output), '--worker', mode],
                stdout=log, stderr=subprocess.STDOUT)
            benchmark.write_json(output/'run_state.json', {'mode': mode, 'pid': child.pid, 'status': 'running'})
            code = child.wait()
            benchmark.write_json(output/'run_state.json', {'mode': mode, 'pid': child.pid, 'exit_code': code})
            if code:
                raise RuntimeError(f'{mode} failed; see {mode}.log')
    run_worker('preflight')
    audit = json.loads((output/'evaluation.json').read_text())
    if not audit['consistency_passed']:
        audit['status'] = 'not_adopted_preprocessing_consistency'
        benchmark.write_json(output/'evaluation.json', audit)
        print(json.dumps(audit, indent=2), flush=True)
        return

    run_worker('measure')
    measured = json.loads((output/'batch.json').read_text())
    with np.load(reference/'batch_vectors.npz', allow_pickle=False) as old_vectors, \
            np.load(output/'batch_vectors.npz', allow_pickle=False) as new_vectors:
        audit['measured_comparison'] = {kind: cosine_report(old_vectors[kind], new_vectors[kind])
                                         for kind in ('images', 'texts')}
    import statistics
    audit['timing'] = {}
    for stage in ('feature_runs', 'pipeline_runs'):
        old = statistics.median(row['seconds'] for row in previous[stage])
        new = statistics.median(row['seconds'] for row in measured[stage])
        audit['timing'][stage] = {'pillow_seconds': old, 'dali_seconds': new,
            'wall_time_reduction': 1-new/old,
            'rss_peak_ratio': max(row['rss_peak_bytes'] for row in measured[stage]) /
                              max(row['rss_peak_bytes'] for row in previous[stage])}
    runs = previous['pipeline_runs']+measured['pipeline_runs']
    all_runs = measured['pipeline_runs']+measured['feature_runs']
    initial = measured['environment']['before_load_gpu']
    audit['checks'] = {
        'consistency': audit['consistency_passed'],
        'measured_consistency': all(row['fraction_at_least_0999'] >= .999
                                    for row in audit['measured_comparison'].values()),
        'measurement_identity': measured['identity'] == audit['preflight_identity'],
        'pipeline_speedup': audit['timing']['pipeline_runs']['wall_time_reduction'] >= .15,
        'rss_increase': all(row['rss_peak_ratio'] <= 1.25 for row in audit['timing'].values()),
        'gpu_headroom': bool(initial) and max(row['device_used_peak_bytes']-initial['used']
                                            for row in all_runs)/initial['free'] <= .85,
        'decisions_equal': len({row['decisions_sha256'] for row in runs}) == 1,
        'quality_scores_equal': len({row['quality_sha256'] for row in runs}) == 1,
        'repeat_consistency': all(min(row['repeat_min_cosine'].values()) >= .999
                                  for row in measured['feature_runs']),
        'cache_zero_forwards': measured['cache_hit_run']['forward']['calls'] == {'image':0, 'text':0}
                               and measured['cache_hit_run']['vectors_equal'],
        'analysis_cache_zero_forwards': measured['pipeline_cache_hit_run']['forward']['calls'] == {'image':0, 'text':0},
        'analysis_cache_decisions_equal': measured['pipeline_cache_hit_run']['decisions_sha256']
                                          == measured['pipeline_runs'][0]['decisions_sha256'],
        'dali_executed_without_failures': bool(measured['dali_statistics']) and all(
            row['decoded'] > 0 and row['decoder_failures'] == 0 for row in measured['dali_statistics']),
        'no_errors': all(not row.get('errors') and not row.get('face_errors') and not row.get('feature_errors')
                         and not row.get('features',{}).get('image_errors') for row in all_runs),
        'source_unchanged': audit['source_sha256'] == benchmark.source_hashes()
                            and audit['evaluator_sha256'] == sha256(__file__)}
    audit['status'] = 'performance_passed_quality_pending' if all(audit['checks'].values()) else 'not_adopted'
    benchmark.write_json(output/'evaluation.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
