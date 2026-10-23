"""Optional local PaddleOCR evidence, EXIF correction and content cache.

OCR is only requested for explicitly selected text-bearing images. Recognized
text remains machine evidence and never supplies an actor identity or plot GT.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from core_modules.organize.task_artifacts import fingerprint
from core_modules.organize.visual_evidence import EvidenceCache


def validate_ocr(value):
    if not isinstance(value, dict) or set(value) != {'lines', 'width', 'height'}:
        return False
    if any(type(value.get(name)) is not int or value[name] < 1 for name in ('width', 'height')):
        return False
    if not isinstance(value['lines'], list) or len(value['lines']) > 2048:
        return False
    for row in value['lines']:
        if not isinstance(row, dict) or set(row) != {'text', 'score', 'polygon'}:
            return False
        if not isinstance(row['text'], str) or not row['text'] or len(row['text']) > 4096:
            return False
        if type(row['score']) not in (int, float) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            return False
        polygon = row['polygon']
        if not isinstance(polygon, list) or len(polygon) != 4 or any(
                not isinstance(point, list) or len(point) != 2 or any(
                    type(number) not in (int, float) or not math.isfinite(number) for number in point)
                for point in polygon):
            return False
        if any(not 0 <= x <= value['width'] or not 0 <= y <= value['height'] for x, y in polygon):
            return False
    return True


class PaddleOCRProvider:
    def __init__(self, python, detector_dir, recognizer_dir, cache_dir, device='cpu', timeout=120, max_side=1600):
        self.python = str(Path(python).resolve())
        self.detector = Path(detector_dir).resolve()
        self.recognizer = Path(recognizer_dir).resolve()
        self.device, self.timeout, self.max_side = device, timeout, max_side
        if not Path(self.python).is_file():
            raise ValueError('ocr_python_missing')
        if (device != 'cpu' or type(max_side) is not int or not 32 <= max_side <= 4096 or
                type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid_ocr_preprocessing')
        files = self._model_files()
        worker = Path(__file__).resolve().parents[2] / 'tools/paddle_ocr_worker.py'
        try:
            packages = subprocess.run([self.python, '-c',
                'import json\nfrom importlib.metadata import version, PackageNotFoundError\n'
                'versions = {name: version(name) for name in ["paddleocr", "paddlex", "Pillow", "numpy"]}\n'
                'try:\n    versions["paddlepaddle-gpu"] = version("paddlepaddle-gpu")\n'
                'except PackageNotFoundError:\n    versions["paddlepaddle"] = version("paddlepaddle")\n'
                'print(json.dumps(versions))'],
                capture_output=True, text=True, timeout=20, check=True)
            versions = json.loads(packages.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise ValueError('ocr_runtime_versions_unavailable') from exc
        self.identity = {'kind': 'paddleocr_local', 'model_files_sha256': files, 'worker_sha256': fingerprint(worker),
                         'adapter_sha256': fingerprint(__file__), 'python': self.python, 'device': device,
                         'packages': versions, 'preprocessing': ['exif_rgb', max_side], 'schema': 'ocr-lines-v1'}
        self.cache = EvidenceCache(cache_dir, self.identity)

    def _model_files(self):
        files = []
        for name, root in (('detector', self.detector), ('recognizer', self.recognizer)):
            if not root.is_dir():
                raise ValueError('ocr_model_directory_missing')
            if not any(path.is_file() and path.suffix == '.pdiparams' for path in root.iterdir()):
                raise ValueError('ocr_local_weights_missing:' + name)
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.yml', '.yaml', '.pdiparams', '.pdmodel'):
                    files.append([name + '/' + path.name, fingerprint(path)])
        return files

    def _weights_unchanged(self):
        try:
            return self._model_files() == self.identity['model_files_sha256']
        except (OSError, ValueError):
            return False

    def observe(self, path, requested=False):
        if not requested:
            return None, {'called': False, 'skipped': 'text_evidence_not_requested'}
        if not self._weights_unchanged():
            return None, {'called': False, 'cache_hit': False, 'error': 'ocr_model_files_changed'}
        content = fingerprint(path)
        key = self.cache.key(content, 'ocr')
        cached = self.cache.get(key, validate_ocr)
        if cached:
            return self._evidence(path, content, cached['value']), {
                'called': False, 'cache_hit': True, 'cache_key': key, 'identity': self.identity,
                'original_trace': cached['trace']}
        from PIL import Image, ImageOps
        with tempfile.TemporaryDirectory(prefix='theatre-ocr-') as temporary:
            temporary = Path(temporary)
            preview = temporary / 'image.png'
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                image.thumbnail((self.max_side, self.max_side))
                image.save(preview)
            output = temporary / 'result.json'
            command = [self.python, str(Path(__file__).resolve().parents[2] / 'tools/paddle_ocr_worker.py'),
                       '--image', str(preview), '--detector', str(self.detector), '--recognizer', str(self.recognizer),
                       '--output', str(output), '--device', self.device]
            environment = os.environ.copy()
            environment['PADDLE_PDX_CACHE_HOME'] = str(temporary / 'paddle_cache')
            environment['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            environment['MPLCONFIGDIR'] = str(temporary / 'matplotlib')
            environment['OMP_NUM_THREADS'] = '4'
            try:
                result = subprocess.run(command, env=environment, capture_output=True, timeout=self.timeout,
                                        text=True, cwd=Path(__file__).resolve().parents[2])
                payload = json.loads(output.read_text()) if output.exists() else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return None, {'called': True, 'cache_hit': False, 'error': 'ocr_worker_failed', 'identity': self.identity}
        if result.returncode or not isinstance(payload, dict) or not validate_ocr(payload.get('observation')):
            return None, {'called': True, 'cache_hit': False, 'error': 'invalid_ocr_result',
                          'worker_exit_code': result.returncode, 'identity': self.identity,
                          'worker_output_tail': result.stdout[-300:] if result.returncode else ''}
        if fingerprint(path) != content:
            return None, {'called': True, 'cache_hit': False, 'error': 'source_changed_during_ocr'}
        if not self._weights_unchanged():
            return None, {'called': True, 'cache_hit': False, 'error': 'ocr_model_files_changed'}
        trace = {'called': True, 'cache_hit': False, 'cache_key': key, 'identity': self.identity,
                 'source_image_sha256': content, 'worker': payload.get('trace')}
        self.cache.put(key, payload['observation'], trace, validate_ocr)
        return self._evidence(path, content, payload['observation']), trace

    @staticmethod
    def _evidence(path, sha256, value):
        return {'id': 'ocr:' + sha256[:24], 'type': 'ocr', 'ref': Path(path).name,
                'quote': json.dumps(value, ensure_ascii=False), 'source_image_sha256': sha256,
                'provenance': 'direct_image_ocr', 'verified_by': None}
