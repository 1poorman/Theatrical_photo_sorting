"""Confirmed scene labels must survive filename delimiter variants."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.organize.task_artifacts import fingerprint
from tools.import_verified_photo_labels import import_labels


class VerifiedPhotoImportTests(unittest.TestCase):
    def import_names(self, directory, names):
        root = Path(directory) / 'photos'
        folder = root / '话剧《样式雷》' / '【整理完成】话剧《样式雷》彩排'
        folder.mkdir(parents=True)
        gallery = Path(directory) / 'gallery'
        gallery.mkdir()
        inventory = []
        for index, name in enumerate(names):
            photo = folder / name
            photo.write_bytes(('fixture-' + str(index)).encode())
            inventory.append({'file_id': 'file-' + str(index), 'image_id': 'image-' + str(index),
                'source': photo.relative_to(root).as_posix(), 'content_hash': fingerprint(photo),
                'duplicate_group_id': 'duplicate-' + str(index), 'status': 'ok'})
        manifest = Path(directory) / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in inventory))
        return import_labels(root, manifest, gallery, 'user_confirmed_test')

    def test_separate_scene_segment_is_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, summary = self.import_names(directory, [
                '39-20160609-戏剧场-话剧《样式雷》彩排-第8场-中：杨淇饰慈禧-凌风.JPG'])
            self.assertEqual(rows[0]['scene_label'], '第8场')
            self.assertIsNotNone(rows[0]['scene_id'])
            self.assertEqual(rows[0]['actor_roles'][0]['actor_name'], '杨淇')
            self.assertEqual(summary['scene_labeled_files'], 1)

    def test_named_act_separate_segment_is_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _ = self.import_names(directory, [
                '33-20140304-戏剧场-京剧《天下归心》彩排-第5幕见母-左起：李鸣岩饰武姜、孟广禄饰郑庄公-苏岩.jpg'])
            self.assertEqual(rows[0]['scene_label'], '第5幕见母')
            self.assertEqual(len(rows[0]['actor_roles']), 2)

    def test_inline_scene_behavior_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _ = self.import_names(directory, [
                '02-20160609-戏剧场-话剧《样式雷》彩排 第1场-彭国斌饰雷廷昌-凌风.JPG'])
            self.assertEqual(rows[0]['scene_label'], '第1场')

    def test_non_scene_people_and_conductor_do_not_become_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, summary = self.import_names(directory, [
                '01-20170522-歌剧院-歌剧《假面舞会》A组彩排-指挥：雷纳托·帕伦波-牛小北.JPG'])
            self.assertIsNone(rows[0]['scene_label'])
            self.assertEqual(summary['scene_labeled_files'], 0)

    def test_multiple_explicit_scene_segments_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'conflicting_explicit_scene'):
                self.import_names(directory, [
                    '39-20160609-戏剧场-话剧《样式雷》彩排-第8场-第7场-中：杨淇饰慈禧-凌风.JPG'])

    def test_inline_and_separate_conflicting_scene_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'conflicting_explicit_scene'):
                self.import_names(directory, [
                    '39-20160609-戏剧场-话剧《样式雷》彩排 第7场-第8场-中：杨淇饰慈禧-凌风.JPG'])


if __name__ == '__main__':
    unittest.main()
