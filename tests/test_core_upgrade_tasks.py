"""Task durability, cancellation boundaries, idempotency and partial failures."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.task_queue import TaskQueue, TaskConflict, TaskQueueFull, TaskUnavailable


TERMINAL = {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'}


def wait_for(queue, task_id, states=TERMINAL, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = queue.get(task_id)
        if task['status'] in states:
            return task
        time.sleep(0.01)
    raise AssertionError(f'task timeout: {queue.get(task_id)}')


def crash_worker(root):
    def handler(params):
        def process(item):
            with open(Path(root) / 'calls.jsonl', 'a') as stream:
                stream.write(json.dumps(item) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            if item['image'] == 'b':
                os._exit(9)
            return {'image': item['image'], 'label': 'act'}
        return process, lambda rows: {'rows': rows}
    queue = TaskQueue(root, {'test': handler})
    queue.submit('test', {}, [{'image': 'a'}, {'image': 'b'}, {'image': 'c'}], 'crash')
    time.sleep(15)
    os._exit(10)


class TaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    def handler(self, params):
        def process(item):
            self.calls.append(item['image'])
            if params.get('block'):
                self.started.set()
                if not self.release.wait(8):
                    raise TimeoutError('test did not release worker')
            if item.get('bad'):
                raise ValueError('bad image')
            return {'image': item['image'], 'label': 'act'}
        return process, lambda rows: {'rows': rows}

    def queue(self, capacity=8, handler=None):
        queue = TaskQueue(self.root, {'test': handler or self.handler}, capacity)
        self.addCleanup(queue.close, 10)
        return queue

    def test_cancel_acknowledged_during_item_then_resume_skips_completed(self):
        queue = self.queue()
        task, _ = queue.submit('test', {'block': True}, [{'image': name} for name in 'abc'])
        self.assertTrue(self.started.wait(5))
        start = time.monotonic()
        self.assertEqual(queue.cancel(task['task_id'])['status'], 'cancel_requested')
        self.assertLess(time.monotonic() - start, 2)
        self.release.set()
        final = wait_for(queue, task['task_id'], {'cancelled'})
        self.assertEqual((final['succeeded'], final['pending']), (1, 2))
        self.assertEqual(self.calls, ['a'])
        queue.resume(task['task_id'])
        final = wait_for(queue, task['task_id'], {'completed'})
        self.assertEqual(self.calls, ['a', 'b', 'c'])
        self.assertEqual(final['succeeded'], 3)

    def test_capacity_and_queued_cancel(self):
        queue = self.queue(capacity=2)
        running, _ = queue.submit('test', {'block': True}, [{'image': 'a'}])
        self.assertTrue(self.started.wait(5))
        pending, _ = queue.submit('test', {}, [{'image': 'b'}])
        with self.assertRaises(TaskQueueFull):
            queue.submit('test', {}, [{'image': 'c'}])
        self.assertEqual(queue.cancel(pending['task_id'])['status'], 'cancelled')
        queue.submit('test', {}, [{'image': 'c'}])
        self.release.set()
        wait_for(queue, running['task_id'])

    def test_concurrent_idempotency_returns_one_task_and_rejects_changed_input(self):
        queue = self.queue()
        def submit(_):
            return queue.submit('test', {'block': True}, [{'image': 'a'}], 'same')
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(submit, range(4)))
        self.assertEqual(sum(created for _, created in values), 1)
        self.assertEqual(len({task['task_id'] for task, _ in values}), 1)
        with self.assertRaises(TaskConflict):
            queue.submit('test', {}, [{'image': 'different'}], 'same')
        self.release.set()

    def test_failed_item_does_not_abort_others_and_is_durable(self):
        queue = self.queue()
        task, _ = queue.submit('test', {}, [{'image': 'a'}, {'image': 'b', 'bad': True}, {'image': 'c'}])
        final = wait_for(queue, task['task_id'])
        self.assertEqual(final['status'], 'completed_with_errors')
        self.assertEqual((final['succeeded'], final['failed'], final['pending']), (2, 1, 0))
        rows = queue.results(task['task_id'])
        self.assertEqual([row['image'] for row in rows], ['a', 'b', 'c'])
        self.assertEqual(rows[1]['error']['type'], 'ValueError')
        self.assertEqual(rows[1]['label'], 'unknown')
        self.assertEqual(len(queue.item_page(task['task_id'], 2, 1)), 2)
        queue.close(10)
        restarted = self.queue()
        self.assertEqual(restarted.get(task['task_id'])['status'], 'completed_with_errors')

    def test_second_worker_cannot_steal_lease_or_interrupt_running_task(self):
        queue = self.queue()
        task, _ = queue.submit('test', {'block': True}, [{'image': 'a'}])
        self.assertTrue(self.started.wait(5))
        with self.assertRaises(TaskUnavailable):
            TaskQueue(self.root, {'test': self.handler})
        self.assertEqual(queue.get(task['task_id'])['status'], 'running')
        self.release.set()

    def test_shutdown_marks_interrupted_then_explicit_resume(self):
        queue = self.queue()
        task, _ = queue.submit('test', {'block': True}, [{'image': 'a'}, {'image': 'b'}])
        self.assertTrue(self.started.wait(5))
        queue.close(0)
        self.release.set()
        queue.close(10)
        restarted = self.queue()
        self.assertEqual(restarted.get(task['task_id'])['status'], 'interrupted')
        self.assertEqual(self.calls, ['a'])
        restarted.resume(task['task_id'])
        wait_for(restarted, task['task_id'], {'completed'})
        self.assertEqual(self.calls, ['a', 'b'])

    def test_real_process_crash_preserves_previous_checkpoint(self):
        context = multiprocessing.get_context('spawn')
        process = context.Process(target=crash_worker, args=(str(self.root),))
        process.start()
        process.join(12)
        self.assertEqual(process.exitcode, 9)
        queue = self.queue()
        task = queue.list()[0]
        self.assertEqual(task['status'], 'interrupted')
        self.assertEqual((task['succeeded'], task['pending']), (1, 2))
        queue.resume(task['task_id'])
        wait_for(queue, task['task_id'], {'completed'})
        self.assertEqual(self.calls, ['b', 'c'])

    def test_failed_finalization_resume_does_not_repeat_inference(self):
        finishes = []
        def handler(params):
            process, _ = self.handler(params)
            def finish(rows):
                finishes.append(rows)
                if len(finishes) == 1:
                    raise OSError('publication failure')
                return {'rows': rows}
            return process, finish
        queue = self.queue(handler=handler)
        task, _ = queue.submit('test', {}, [{'image': 'a'}, {'image': 'b'}])
        self.assertEqual(wait_for(queue, task['task_id'])['status'], 'failed')
        queue.resume(task['task_id'])
        wait_for(queue, task['task_id'], {'completed'})
        self.assertEqual(self.calls, ['a', 'b'])
        self.assertEqual(len(finishes), 2)

    def test_output_ownership_blocks_competing_tasks(self):
        queue = self.queue()
        first, _ = queue.submit('test', {'block': True}, [{'image': 'a'}], output='/result')
        self.assertTrue(self.started.wait(5))
        with self.assertRaises(TaskConflict):
            queue.submit('test', {}, [{'image': 'b'}], output='/result')
        with self.assertRaises(TaskConflict):
            queue.submit('test', {}, [{'image': 'b'}], output='/result/child.json')
        queue.cancel(first['task_id'])
        self.release.set()
        wait_for(queue, first['task_id'], {'cancelled'})
        self.release.clear()
        self.started.clear()
        second, _ = queue.submit('test', {'block': True}, [{'image': 'b'}], output='/result')
        self.assertTrue(self.started.wait(5))
        with self.assertRaises(TaskConflict):
            queue.resume(first['task_id'])
        self.release.set()
        wait_for(queue, second['task_id'])


if __name__ == '__main__':
    unittest.main()
