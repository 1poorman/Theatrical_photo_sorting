"""Exercise the real age HTTP endpoint with existing theatrical photographs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18304')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=ROOT / 'outputs/acceptance/core_upgrade/u1_face_identity_review_final_20261023/review_template.json')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    def request(endpoint, form=None, upload=None):
        headers = {}
        data = None
        if form is not None:
            data = urllib.parse.urlencode(form).encode()
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        if upload is not None:
            boundary = 'agecheck' + uuid.uuid4().hex
            data = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="photo.jpg"\r\n'
                    'Content-Type: image/jpeg\r\n\r\n').encode() + upload + f'\r\n--{boundary}--\r\n'.encode()
            headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
        with urllib.request.urlopen(urllib.request.Request(base + endpoint, data=data, headers=headers), timeout=180) as response:
            return json.load(response)

    review = json.loads(args.review.read_text())
    counts = Counter(row['image_id'] for row in review['faces'])
    sources = [Path(next(row['source'] for row in review['images'] if low <= counts[row['image_id']] <= high))
               for low, high in ((1, 1), (2, 4), (5, 12))]
    before = {str(path): sha256(path) for path in sources}
    outputs, health_ms = [], []
    started = time.perf_counter()
    first = sources[0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(request, '/api/face/age', {'input_dir': str(first.parent), 'name': first.name})
        while not pending.done():
            tick = time.perf_counter()
            check('health_during_age_load_' + str(len(health_ms)), request('/api/health')['code'] == 200)
            health_ms.append((time.perf_counter() - tick) * 1000)
            time.sleep(0.2)
        outputs.append(pending.result()['data'])
    first_seconds = time.perf_counter() - started
    check('health_remains_responsive', max(health_ms, default=0) < 2000)
    for path in sources[1:]:
        outputs.append(request('/api/face/age', {'input_dir': str(path.parent), 'name': path.name})['data'])
    for index, output in enumerate(outputs):
        check(f'image_{index}_has_faces', output['face_count'] > 0)
        check(f'image_{index}_model', output['model']['name'] == 'SwinFace' and len(output['model']['checkpoint_sha256']) == 64)
        check(f'image_{index}_count', output['face_count'] == len(output['faces']))
        with Image.open(output['annotated_image_path']) as annotated:
            check(f'image_{index}_annotation_dimensions', annotated.size == (output['image_width'], output['image_height']))
        for face in output['faces']:
            x1, y1, x2, y2 = face['bbox']
            prefix = f'image_{index}_face_{face["face_id"]}'
            check(prefix + '_bbox', 0 <= x1 < x2 <= output['image_width'] and 0 <= y1 < y2 <= output['image_height'])
            check(prefix + '_estimate', face['age_status'] in ('estimated', 'out_of_range'))
            if face['estimated_age'] is not None:
                check(prefix + '_age_range', 0 <= face['estimated_age'] <= 120)
            with Image.open(face['face_image_path']) as crop:
                check(prefix + '_crop_size', crop.size == (112, 112))
    uploaded = request('/api/face/age', upload=first.read_bytes())['data']
    check('upload_matches_path_boxes', [face['bbox'] for face in uploaded['faces']] == [face['bbox'] for face in outputs[0]['faces']])
    check('upload_matches_path_ages', [face['estimated_age'] for face in uploaded['faces']] == [face['estimated_age'] for face in outputs[0]['faces']])
    empty_image = io.BytesIO()
    Image.new('RGB', (400, 300), 'black').save(empty_image, format='JPEG')
    empty = request('/api/face/age', upload=empty_image.getvalue())['data']
    check('no_face_response', empty['face_count'] == 0 and empty['faces'] == [])
    try:
        request('/api/face/age', upload=b'broken image')
        check('bad_image_rejected', False)
    except urllib.error.HTTPError as error:
        check('bad_image_rejected', error.code == 422)
    check('source_bytes_preserved', before == {str(path): sha256(path) for path in sources})
    report = {
        'passed': all(row['passed'] for row in checks), 'checks': checks,
        'base_url': base, 'sources_sha256': before, 'outputs': outputs,
        'first_request_seconds': first_seconds,
        'health_during_first_request_max_ms': max(health_ms, default=None),
        'health_during_first_request_samples_ms': health_ms,
        'age_ground_truth_count': 0, 'age_mae': None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(checks), 'faces': [row['face_count'] for row in outputs],
                      'first_request_seconds': first_seconds, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
