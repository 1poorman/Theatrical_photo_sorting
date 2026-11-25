"""Run locally specified PaddleOCR weights in the isolated Paddle environment."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--recognizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new_ocr_output_required')
    started = time.perf_counter()
    from PIL import Image
    from paddleocr import PaddleOCR
    import numpy as np
    pipeline = PaddleOCR(text_detection_model_dir=str(args.detector),
                         text_recognition_model_dir=str(args.recognizer),
                         use_doc_orientation_classify=False, use_doc_unwarping=False,
                         use_textline_orientation=False, device=args.device,
                         enable_mkldnn=False, cpu_threads=4)
    result = list(pipeline.predict(str(args.image)))
    if len(result) != 1:
        raise ValueError('ocr_result_count_changed')
    row = result[0]
    # PaddleX results expose a dictionary through res, retaining boxes and scores.
    data = row.get('res', row) if hasattr(row, 'get') else row.json.get('res', row.json)
    texts, scores = data.get('rec_texts', []), data.get('rec_scores', [])
    polygons = data.get('rec_polys', data.get('dt_polys', []))
    if len(texts) != len(scores) or len(texts) != len(polygons):
        raise ValueError('ocr_line_alignment_changed')
    with Image.open(args.image) as image:
        width, height = image.size
    observation = {'lines': [{'text': str(text), 'score': float(score),
                               'polygon': np.asarray(poly, dtype=float).tolist()}
                              for text, score, poly in zip(texts, scores, polygons) if text],
                   'width': width, 'height': height}
    payload = {'observation': observation,
               'trace': {'elapsed_seconds': time.perf_counter() - started,
                         'paddleocr_version': version('paddleocr'), 'paddlex_version': version('paddlex'),
                         'device': args.device, 'input_size': [width, height]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'lines': len(observation['lines']), 'elapsed_seconds': payload['trace']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
