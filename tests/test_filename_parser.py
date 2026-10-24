"""Raw and organized filenames retain metadata across project copies."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.organize.filename_parser import parse_filename
from core_modules.organize.burst_dedup import normalize_group_key


class FilenameParserTests(unittest.TestCase):
    def test_organized_name_detected_without_directory_hint(self):
        row = parse_filename('03-20170522-歌剧院-歌剧《假面舞会》A组彩排 第1幕1场-王猛饰和恩-王小京.JPG')
        self.assertEqual((row['seq'], row['date'], row['venue'], row['play'], row['event'], row['scene']),
                         (3, '20170522', '歌剧院', '歌剧《假面舞会》', 'A组彩排', '第1幕1场'))
        self.assertEqual(row['people'], [('王猛', '和恩')])
        self.assertEqual(row['photographer'], '王小京')

    def test_large_sequence_and_attached_venue(self):
        row = parse_filename('10001-20160609戏剧场-话剧《样式雷》彩排 第8场-摄影凌风.JPG')
        self.assertEqual((row['seq'], row['date'], row['venue'], row['event'], row['scene']),
                         (10001, '20160609', '戏剧场', '彩排', '第8场'))

    def test_placeholder_roundtrip_with_or_without_event(self):
        for event in ('彩排', ''):
            for scene in ('scene-01', 'scene-02-03', 'scene-08-a', 'scene-05-b'):
                with self.subTest(event=event, scene=scene):
                    row = parse_filename(f'01-20160609-戏剧场-话剧《样式雷》{event} {scene}-摄影凌风.JPG')
                    self.assertEqual(row['event'], event or None)
                    self.assertEqual(row['scene'], scene)

    def test_separate_scene_is_not_event_or_photographer(self):
        for scene in ('第8场', '第5幕见母', '序曲', '尾声', '谢幕'):
            with self.subTest(scene=scene):
                row = parse_filename(f'39-20160609-戏剧场-话剧《样式雷》彩排-{scene}-摄影凌风.JPG')
                self.assertEqual((row['event'], row['scene'], row['photographer']), ('彩排', scene, '凌风'))

    def test_scene_only_tail_without_photographer(self):
        row = parse_filename('39-20160609-戏剧场-话剧《样式雷》彩排-尾声.JPG')
        self.assertEqual(row['scene'], '尾声')
        self.assertIsNone(row['photographer'])

    def test_raw_burst_semantics_preserved(self):
        names = ['20141012歌剧院-舞剧《马可·波罗》B演-前：苏鹏饰马可·波罗-摄影凌风.JPG',
                 '20141012歌剧院-舞剧《马可·波罗》B演-前：苏鹏饰马可·波罗 (13)-摄影凌风.JPG']
        for name in names:
            row = parse_filename(name)
            self.assertIsNone(row['seq'])
            self.assertIsNone(row['scene'])
            self.assertEqual((row['date'], row['venue'], row['event']), ('20141012', '歌剧院', 'B演'))
        self.assertEqual(normalize_group_key(names[0]), normalize_group_key(names[1]))
        self.assertEqual(parse_filename(names[1])['burst_seq'], 13)

    def test_organized_sequence_does_not_invent_burst_membership(self):
        self.assertNotEqual(normalize_group_key('01-20160609-戏剧场-话剧《样式雷》彩排 第1场-摄影凌风.JPG'),
                            normalize_group_key('02-20160609-戏剧场-话剧《样式雷》彩排 第1场-摄影凌风.JPG'))

    def test_arbitrary_numeric_prefix_is_not_an_organized_date(self):
        row = parse_filename('32-演员饰角色-摄影凌风.JPG')
        self.assertIsNone(row['seq'])
        self.assertIsNone(row['date'])


if __name__ == '__main__':
    unittest.main()
