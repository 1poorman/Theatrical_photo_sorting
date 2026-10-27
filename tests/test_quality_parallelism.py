"""Full-resolution quality workers preserve selections, ties and classifier order."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.organize import burst_dedup


class QualityParallelismTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        rng = np.random.default_rng(18)
        for index in range(5):
            image = rng.integers(0, 256, (63, 79, 3), dtype=np.uint8)
            cv2.imwrite(str(self.photos / f'burst ({index + 1}).png'), image)
        (self.photos / 'unreadable.jpg').write_bytes(b'broken image')

    def run_folder(self, workers, name, classifier=None):
        with ThreadPoolExecutor(max_workers=workers) as pool, \
                patch.object(burst_dedup, '_quality_executor', pool), \
                patch.object(burst_dedup, 'QUALITY_DECODE_WORKERS', workers):
            output = self.root / name
            report = burst_dedup.BurstDeduplicator(classifier).dedup_folder(
                str(self.photos), str(output), save_discarded=True)
        return report, {str(path.relative_to(output)): path.read_bytes()
                        for path in output.rglob('*') if path.is_file() and path.suffix != '.json'}

    def test_scores_selection_order_and_copied_bytes_match_serial(self):
        serial, serial_files = self.run_folder(1, 'serial')
        parallel, parallel_files = self.run_folder(2, 'parallel')
        self.assertEqual(parallel, serial)
        self.assertEqual(parallel_files, serial_files)
        broken = [member for group in parallel['groups'] for member in group['members']
                  if member['name'] == 'unreadable.jpg'][0]
        self.assertEqual(broken['quality']['score'], 0.)

    def test_equal_score_ties_keep_input_order_when_workers_finish_out_of_order(self):
        def quality(image):
            time.sleep((int(image[0, 0, 0]) % 3) * .005)
            return {'score': .5}
        with patch.object(burst_dedup, 'image_quality', side_effect=lambda image: quality(image) if image is not None else {'score': 0.}):
            serial, _ = self.run_folder(1, 'serial')
            parallel, _ = self.run_folder(2, 'parallel')
        self.assertEqual(parallel['groups'], serial['groups'])

    def test_active_full_resolution_work_is_limited_to_worker_count(self):
        lock = threading.Lock()
        active = peak = 0
        original = burst_dedup.BurstDeduplicator._quality_entry
        def entry(instance, path):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.02)
                return original(instance, path)
            finally:
                with lock:
                    active -= 1
        with patch.object(burst_dedup.BurstDeduplicator, '_quality_entry', entry):
            self.run_folder(2, 'parallel')
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)

    def test_classifier_stays_on_calling_thread_and_failures_keep_fallback(self):
        main_thread = threading.get_ident()
        calls = []
        class Classifier:
            def classify_shot_type(self, image, name, area_ratio):
                calls.append((name, threading.get_ident()))
                if '(2)' in name:
                    raise ValueError('classifier failed')
                return {'shot_type': 'wide'}, None
        serial, _ = self.run_folder(1, 'serial', Classifier())
        expected = list(calls)
        calls.clear()
        parallel, _ = self.run_folder(2, 'parallel', Classifier())
        self.assertEqual(parallel['groups'], serial['groups'])
        self.assertEqual(calls, expected)
        self.assertTrue(all(thread == main_thread for _, thread in calls))


if __name__ == '__main__':
    unittest.main()
