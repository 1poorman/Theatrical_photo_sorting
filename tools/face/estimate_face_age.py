"""Estimate face ages in local images with SCRFD and the official SwinFace weights."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.base import SWINFACE_MODEL_PATH
from core_modules.face_recognition.age_estimation import AGE_NOTE, FaceAgeEstimator
from core_modules.tools.image_io import IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Image or directory of images')
    parser.add_argument('--output', type=Path, required=True, help='Directory for report and previews')
    parser.add_argument('--checkpoint', type=Path, default=Path(SWINFACE_MODEL_PATH))
    parser.add_argument('--device', default='auto', help='auto, cpu or cuda[:N]')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--aligned', action='store_true', help='Each input is already a single aligned face; skip detection')
    args = parser.parse_args()
    if args.batch_size < 1 or args.torch_threads < 1:
        parser.error('batch-size and torch-threads must be positive')
    source = args.input.expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(path for path in (source.rglob('*') if args.recursive else source.iterdir())
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        parser.error('input does not exist')
    if not paths:
        parser.error('no images found')

    import torch
    torch.set_num_threads(args.torch_threads)
    estimator = FaceAgeEstimator(args.checkpoint, args.device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths):
        try:
            if args.aligned:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('unable to decode image')
                report = {'faces': [{'face_id': 1, **estimator.estimate_aligned([image])[0]}],
                          'face_count': 1, 'input_mode': 'aligned_face'}
            else:
                report, annotated, crops = estimator.estimate(image_path=str(path))
                destination = args.output / f'image_{index:05d}'
                destination.mkdir(exist_ok=True)
                annotated_path = destination / 'annotated.jpg'
                if not cv2.imwrite(str(annotated_path), annotated):
                    raise OSError('unable to write annotated image')
                report['annotated_image_path'] = str(annotated_path.resolve())
                for face, crop in zip(report['faces'], crops):
                    crop_path = destination / f"face_{face['face_id']}.jpg"
                    if not cv2.imwrite(str(crop_path), crop):
                        raise OSError('unable to write face preview')
                    face['face_image_path'] = str(crop_path.resolve())
            rows.append({'image': str(path), 'status': 'ok', **report})
            print(f'{index + 1}/{len(paths)}: {report["face_count"]} faces - {path.name}', flush=True)
        except Exception as exc:
            rows.append({'image': str(path), 'status': 'error', 'error': str(exc)})
            print(f'{index + 1}/{len(paths)}: ERROR - {path.name}: {exc}', file=sys.stderr, flush=True)
    failed = sum(row['status'] == 'error' for row in rows)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'note': AGE_NOTE, 'age_unit': 'years',
        'input_count': len(rows), 'failed_count': failed,
        'face_count': sum(row.get('face_count', 0) for row in rows),
        'age_ground_truth_count': 0, 'age_mae': None,
        'model': estimator.analyzer.metadata() if estimator._analyzer is not None else {'name': 'SwinFace', 'loaded': False},
        'images': rows,
    }
    destination = args.output / 'age_estimates.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Report: {destination}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
