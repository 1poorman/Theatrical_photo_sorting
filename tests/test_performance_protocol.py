"""Integrity checks for frozen-version and deferred-export measurements."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_feature_pipeline import execution_plan, freeze_baseline, hardware_observation
from tools.benchmark_review_snapshots import measure_exports


class PerformanceProtocolTests(unittest.TestCase):
    def test_competing_gpu_process_is_reported_without_claiming_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'hardware.jsonl'
            path.write_text(json.dumps({'worker_pid': 123, 'cpu_load_average': [4, 3, 2],
                'gpu': {'processes': 'gpu_uuid, pid, name, memory\nGPU-1, 123, python, 100\nGPU-1, 456, python, 200'}}) + '\n')
            report = hardware_observation(path)
            self.assertEqual(report['external_gpu_pids'], [456])
            self.assertTrue(report['external_gpu_processes_present'])
            self.assertFalse(report['controlled_environment'])

    def test_alternating_order_has_five_independent_pairs(self):
        plan = execution_plan(5)
        self.assertEqual(len(plan), 10)
        for repeat in range(5):
            modes = [mode for pair, mode in plan if pair == repeat]
            self.assertEqual(modes, ['baseline', 'batch'] if repeat % 2 == 0 else ['batch', 'baseline'])

    def test_frozen_dependencies_match_git_and_load_outside_current_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            hashes = freeze_baseline(output, 'HEAD')
            for name in ('config/base.py', 'core_modules/organize/scene_split.py',
                         'core_modules/tools/image_io.py'):
                data = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                self.assertEqual(hashes[name], hashlib.sha256(data).hexdigest())
                self.assertEqual((output / 'baseline_tree' / name).read_bytes(), data)
            code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                    'import config.base; from pathlib import Path; '
                    'assert Path(config.base.__file__).is_relative_to(sys.argv[1])')
            subprocess.run([sys.executable, '-c', code, str(output / 'baseline_tree')],
                           cwd=ROOT, check=True)

    def test_full_export_includes_dirty_snapshots_history_and_only_approved(self):
        rows = [{'image': f'photo-{index}.jpg', 'label': 'candidate',
                 'evidence': [{'quote': 'synthetic'}]} for index in range(30)]
        with tempfile.TemporaryDirectory() as temporary:
            results = measure_exports(Path(temporary), rows, 3)
            self.assertEqual(len({row['labels_sha256'] for runs in results.values() for row in runs}), 1)
            for mode, runs in results.items():
                self.assertEqual(len(runs), 3)
                for row in runs:
                    self.assertTrue(all(row['checks'].values()), row)
                    self.assertGreater(row['file_bytes']['review_queue.jsonl'], 0)
                    self.assertGreaterEqual(row['decide_plus_export_seconds'], row['seconds'])
                    self.assertEqual(len(row['snapshot_replacements']), 2 if mode == 'deferred' else 0)
            labels = json.loads((Path(temporary) / 'export_deferred/scene_labels.json').read_text())
            self.assertEqual(labels, {'photo-0.jpg': 'export-probe'})


if __name__ == '__main__':
    unittest.main()
