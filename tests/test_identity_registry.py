"""Identity decisions, role provenance and gallery-filter regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT), str(ROOT / 'tests')]
from core_modules.organize.identity_registry import IdentityRegistry, decide_identity, resolve_hits
from core_modules.organize.smart_organizer import SmartOrganizer
from core_modules.organize.actor_view import ActorViewGenerator
from app.organize_tasks import prepare_organize
from core_modules.organize.organize_task import OrganizeTask
from core_modules.organize.task_queue import TaskQueue
from test_core_upgrade_tasks import wait_for


def registry_data():
    return {'schema_version': 1, 'plays': [{'play_id': 'play-a', 'title': '京剧《甲》'},
                                          {'play_id': 'play-b', 'title': '话剧《乙》'}],
            'identities': [
                {'identity_id': 'actor-a', 'actor_name': '甲演员', 'verified': True,
                 'gallery_dirs': ['别名（饰旧角）', '甲演员（饰另角）'],
                 'roles': [{'play_id': 'play-a', 'role_id': 'role-a', 'role_name': '新角',
                            'source_ref': 'human-cast-a', 'verified': True},
                           {'play_id': 'play-b', 'role_id': 'role-b', 'role_name': '乙角',
                            'source_ref': 'human-cast-b', 'verified': True}]},
                {'identity_id': 'actor-b', 'actor_name': '甲演员', 'verified': True,
                 'gallery_dirs': ['同名（饰第三角）'], 'roles': []}]}


class IdentityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.registry_file = self.root / 'identity_registry.json'
        self.registry_file.write_text(json.dumps(registry_data(), ensure_ascii=False))
        self.registry = IdentityRegistry.load(self.registry_file)

    def test_alias_anchors_merge_before_margin_and_current_play_resolves_role(self):
        row = decide_identity([.91, .90, .7], ['别名（饰旧角）', '甲演员（饰另角）', '同名（饰第三角）'],
                              self.registry, .55, .05)
        self.assertEqual(row['identity_id'], 'actor-a')
        self.assertAlmostEqual(row['margin'], .21)
        hits, identities = resolve_hits({row['anchor_dir']: row['score']}, self.registry, '京剧《甲》')
        self.assertEqual(hits, {'甲演员（饰新角）': .91})
        self.assertEqual(identities[0]['role_source'], 'human-cast-a')

    def test_same_name_different_ids_compete_and_never_overwrite(self):
        row = decide_identity([.90, .89], ['别名（饰旧角）', '同名（饰第三角）'], self.registry, .55, .05)
        self.assertEqual(row['reason'], 'ambiguous_identity')
        hits, identities = resolve_hits({'别名（饰旧角）': .9, '同名（饰第三角）': .8}, self.registry)
        self.assertEqual(len(hits), 2)
        self.assertEqual({r['identity_id'] for r in identities}, {'actor-a', 'actor-b'})

    def test_same_named_identities_keep_labels_across_separate_photos(self):
        first, _ = resolve_hits({'别名（饰旧角）': .9}, self.registry)
        second, _ = resolve_hits({'同名（饰第三角）': .8}, self.registry)
        together, _ = resolve_hits({'别名（饰旧角）': .9, '同名（饰第三角）': .8}, self.registry)
        self.assertTrue(set(first).isdisjoint(second))
        self.assertEqual({**first, **second}, together)

        data = registry_data()
        data['identities'][1]['roles'] = [
            {'play_id': 'play-b', 'role_id': 'role-c', 'role_name': '新角',
             'source_ref': 'human-cast-c', 'verified': True}]
        registry = IdentityRegistry(data)
        first, _ = resolve_hits({'别名（饰旧角）': .9}, registry, 'play-a')
        second, _ = resolve_hits({'同名（饰第三角）': .8}, registry, 'play-b')
        self.assertTrue(set(first).isdisjoint(second))

    def test_legacy_name_never_implies_anchor_role(self):
        hits, identities = resolve_hits({'某演员（饰外剧角色）': .9}, IdentityRegistry(), '京剧《甲》')
        self.assertEqual(hits, {'某演员': .9})
        self.assertIsNone(identities[0]['role_id'])
        self.assertEqual(identities[0]['identity_source'], 'gallery_name_unverified')

    def test_unknown_play_multiple_roles_and_unverified_role_stay_actor_only(self):
        self.assertIsNone(self.registry.resolve('别名（饰旧角）', 'unknown')['role_name'])
        data = registry_data()
        extra = copy.deepcopy(data['identities'][0]['roles'][0])
        extra.update(role_id='role-other', role_name='另一角')
        data['identities'][0]['roles'].append(extra)
        self.assertEqual(IdentityRegistry(data).resolve('别名（饰旧角）', '京剧《甲》')['role_status'], 'ambiguous_role')
        data['identities'][0]['roles'][0]['verified'] = False
        data['identities'][0]['roles'][-1]['verified'] = False
        self.assertIsNone(IdentityRegistry(data).resolve('别名（饰旧角）', '京剧《甲》')['role_id'])

    def test_unverified_identity_not_used_and_conflicting_assignments_rejected(self):
        data = registry_data()
        data['identities'][0]['verified'] = False
        self.assertNotEqual(IdentityRegistry(data).identity('别名（饰旧角）')['identity_id'], 'actor-a')
        data['identities'][1]['gallery_dirs'].append('别名（饰旧角）')
        with self.assertRaises(ValueError):
            IdentityRegistry(data)

    def test_invalid_and_nonfinite_scores_are_rejected_without_identity(self):
        result = decide_identity([np.nan, np.inf, .2], ['x', 'y', 'z'], IdentityRegistry(), .55, .05)
        self.assertEqual(result['reason'], 'below_threshold')
        self.assertIsNone(result['identity_id'])
        self.assertEqual(decide_identity([np.nan], ['x'], IdentityRegistry(), .55, .05)['reason'], 'no_valid_anchor')

    def test_organizer_keeps_face_indices_and_resolves_current_role(self):
        extractor = SimpleNamespace(align_face=lambda img, lm: None if lm=='bad' else np.zeros((160,160,3)),
                                    extract_features=lambda values: np.array([[1., 0.]]), l2_normalize=lambda f:f)
        system = SimpleNamespace(extractor=extractor, detector=SimpleNamespace(_detect_single=lambda image:
                                  [SimpleNamespace(landmark='bad'), SimpleNamespace(landmark='good')]))
        organizer = SmartOrganizer(system, str(self.root))
        organizer._lib_matrix = np.array([[.9, 0], [.7, 0]])
        organizer._lib_persons = ['别名（饰旧角）', '同名（饰第三角）']
        path = self.root / '20140304-京剧《甲》-摄影乙.jpg'
        Image.new('RGB', (30,30)).save(path)
        record = organizer.describe_image(str(path), classify_role=False)
        self.assertEqual(record['recognized'], {'甲演员（饰新角）': .9})
        self.assertEqual([r['face_index'] for r in record['face_decisions']], [0,1])
        self.assertEqual(record['face_decisions'][0]['reason'], 'alignment_failed')
        self.assertEqual(record['identities'][0]['identity_id'], 'actor-a')

    def test_album_filter_does_not_remove_competing_identity(self):
        source = self.root / 'photos'
        source.mkdir()
        Image.new('RGB',(20,20)).save(source / 'a.jpg')
        system = SimpleNamespace(detector=SimpleNamespace(_detect_single=lambda img:[object()]))
        generator = ActorViewGenerator(system, str(self.root))
        library = {'别名（饰旧角）':[(np.array([.9,0]),'a')],
                   '同名（饰第三角）':[(np.array([.92,0]),'b')]}
        with patch.object(generator, '_library_features', return_value=library), \
             patch.object(generator._fdb,'_face_feature',return_value=(np.array([1,0]),None)):
            report = generator.generate(str(source),str(self.root/'albums'),person_filter='别名',copy_images=False)
        self.assertEqual(report['images'][0]['hits'], {})
        self.assertEqual(report['images'][0]['face_decisions'][0]['reason'], 'ambiguous_identity')

    def test_separate_photos_of_same_named_actors_have_distinct_albums(self):
        source = self.root / 'photos'
        source.mkdir()
        for name, color in [('a.jpg', 'red'), ('b.jpg', 'blue')]:
            Image.new('RGB', (20, 20), color).save(source / name)
        system = SimpleNamespace(detector=SimpleNamespace(_detect_single=lambda img: [object()]))
        generator = ActorViewGenerator(system, str(self.root))
        library = {'别名（饰旧角）': [(np.array([.9, 0]), 'a')],
                   '同名（饰第三角）': [(np.array([0, .9]), 'b')]}
        output = self.root / 'albums'
        with patch.object(generator, '_library_features', return_value=library), \
             patch.object(generator._fdb, '_face_feature',
                          side_effect=[(np.array([1, 0]), None), (np.array([0, 1]), None)]):
            report = generator.generate(str(source), str(output))
        occupied = {name: album for name, album in report['persons'].items() if album['count']}
        self.assertEqual(len(occupied), 2)
        for name, album in occupied.items():
            self.assertEqual(album['count'], 1)
            image = album['images'][0]['image']
            self.assertEqual((output / 'actor_views' / name / image).read_bytes(),
                             (source / image).read_bytes())

    def test_album_filter_accepts_registered_actor_name_and_identity_id(self):
        data = registry_data()
        data['identities'][0]['actor_name'] = '演员正名'
        self.registry_file.write_text(json.dumps(data, ensure_ascii=False))
        source = self.root / 'photos'
        source.mkdir()
        Image.new('RGB', (20, 20)).save(source / 'a.jpg')
        system = SimpleNamespace(detector=SimpleNamespace(_detect_single=lambda img: [object()]))
        generator = ActorViewGenerator(system, str(self.root))
        library = {'别名（饰旧角）': [(np.array([.9, 0]), 'a')],
                   '同名（饰第三角）': [(np.array([.7, 0]), 'b')]}
        with patch.object(generator, '_library_features', return_value=library), \
             patch.object(generator._fdb, '_face_feature', return_value=(np.array([1, 0]), None)):
            for person in ('演员正名', 'actor-a'):
                with self.subTest(person=person):
                    report = generator.generate(str(source), str(self.root / person),
                                                person_filter=person, copy_images=False)
                    self.assertNotIn('error', report)
                    self.assertEqual(report['images'][0]['hits'], {'演员正名': .9})
                    self.assertEqual(report['images'][0]['identities'][0]['identity_id'], 'actor-a')

    def test_registry_changed_after_prepare_is_rejected_before_business_item(self):
        photos=self.root/'photos'; photos.mkdir()
        Image.new('RGB',(20,20)).save(photos/'a.jpg')
        params,items=prepare_organize({'input_dir':str(photos),'db_root':str(self.root),
                                      'output_dir':str(self.root/'out'),'classify_role':False})
        self.registry_file.write_text(json.dumps(registry_data())+' ')
        organizer=SmartOrganizer(None,str(self.root))
        queue=TaskQueue(self.root/'tasks',{'organize':lambda p:OrganizeTask(p,organizer)})
        self.addCleanup(queue.close,10)
        task,_=queue.submit('organize',params,items)
        result=wait_for(queue,task['task_id'])
        self.assertEqual(result['status'],'failed')
        self.assertIn('identity_registry_content_changed',result['error'])
        self.assertFalse((self.root/'out').exists())

    def test_registry_changed_with_same_size_and_mtime_is_detected(self):
        organizer=SmartOrganizer(None,str(self.root))
        before=self.registry_file.stat()
        self.registry_file.write_bytes(self.registry_file.read_bytes().replace(b'actor-a',b'actor-z'))
        import os
        os.utime(self.registry_file,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'identity_registry_content_changed'):
            organizer.verify_identity_registry()

    def test_resume_rejects_new_registry_without_repeating_recognition(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        photos=self.root/'photos'; photos.mkdir()
        for name in ('a.jpg','b.jpg'):
            Image.new('RGB',(20,20)).save(photos/name)
        params,items=prepare_organize({'input_dir':str(photos),'db_root':str(self.root),
                                      'output_dir':str(self.root/'out'),'classify_role':False})
        organizer=SmartOrganizer(None,str(self.root))
        calls=[]
        def recognize(img,threshold):
            calls.append('recognize'); entered.set(); release.wait(5)
            return {'别名（饰旧角）':.9}, []
        organizer._recognize_image=recognize
        queue=TaskQueue(self.root/'tasks',{'organize':lambda p:OrganizeTask(p,organizer)})
        self.addCleanup(queue.close,10)
        task,_=queue.submit('organize',params,items)
        try:
            self.assertTrue(entered.wait(5))
            queue.cancel(task['task_id'])
        finally:
            release.set()
        wait_for(queue,task['task_id'],{'cancelled'})
        data=registry_data(); data['identities'][0]['actor_name']='改名'
        self.registry_file.write_text(json.dumps(data))
        queue.resume(task['task_id'])
        final=wait_for(queue,task['task_id'])
        self.assertIn('identity_registry_content_changed',final['error'])
        self.assertEqual(len(calls),1)
        self.assertFalse((self.root/'out').exists())

    def test_importer_retains_missing_labels_conflicts_and_confirmed_aliases(self):
        from tools.build_core_upgrade_manifest import inventory
        from tools.import_verified_photo_labels import import_labels
        source=self.root/'dataset'; directory=source/'舞剧《马可·波罗》'/'【整理完成】彩排'
        directory.mkdir(parents=True)
        for name,color in [('01-20141012-歌剧院-舞剧《马可·波罗》B组 第1幕：元都之恋-苏鹏饰马可波罗-凌风.jpg','red'),
                           ('02-20141012-歌剧院-舞剧《马可·波罗》B组 第2幕：古国情韵-苏鹏饰马可·波罗-凌风.jpg','blue'),
                           ('03-20141012-歌剧院-舞剧《马可·波罗》B组-凌风.jpg','green')]:
            Image.new('RGB',(20,20),color).save(directory/name)
        manifest,_=inventory(source)
        manifest_file=self.root/'manifest.jsonl'
        manifest_file.write_text(''.join(json.dumps(row)+'\n' for row in manifest))
        gallery=self.root/'gallery'; gallery.mkdir(); (gallery/'苏鹏（饰马可·波罗）').mkdir()
        rows,registry,summary=import_labels(source,manifest_file,gallery,'user-confirmed')
        self.assertEqual(summary['scene_labeled_files'],2)
        self.assertEqual(len(summary['role_label_conflicts']),1)
        self.assertFalse(summary['final_test_locked'])
        self.assertTrue(all(r['face_boxes'] is None and not r['people_list_exhaustive'] for r in rows))
        aliases=[{'play':'舞剧《马可·波罗》','actor':'苏鹏','alias':'马可波罗','canonical':'马可·波罗','source_ref':'user-alias'}]
        rows,registry,summary=import_labels(source,manifest_file,gallery,'user-confirmed',aliases)
        self.assertEqual(summary['role_label_conflicts'],[])
        self.assertEqual(len(registry['identities'][0]['roles']),1)
        self.assertEqual(rows[0]['actor_roles'][0]['original_role_name'],'马可波罗')
        self.assertEqual(rows[0]['actor_roles'][0]['alias_source'],'user-alias')
        self.assertEqual(IdentityRegistry(registry).resolve('苏鹏（饰马可·波罗）','舞剧《马可·波罗》')['role_name'],'马可·波罗')
        from tools.evaluate_verified_scene_retrieval import prepare_inputs
        labels_file=self.root/'labels.json'; labels_file.write_text(json.dumps(rows))
        registry_file=self.root/'cast.json'; registry_file.write_text(json.dumps(registry))
        evaluation=self.root/'evaluation'; evaluation.mkdir()
        by_play,scenes,excluded=prepare_inputs(labels_file,registry_file,source,evaluation)
        self.assertEqual(sum(map(len,by_play.values())),2)
        self.assertEqual(len(excluded),1)
        self.assertTrue(all('苏鹏' not in scene['text'] for scene in scenes))
        for group in by_play.values():
            for query in group:
                preview=Path(query['neutral_path'])
                self.assertNotIn('幕',preview.name)
                with Image.open(preview) as image:
                    self.assertFalse(image.getexif())
        (directory/rows[0]['source'].split('/')[-1]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'missing_or_changed_inventory_source'):
            import_labels(source,manifest_file,gallery,'user-confirmed',aliases)


if __name__ == '__main__':
    unittest.main()
