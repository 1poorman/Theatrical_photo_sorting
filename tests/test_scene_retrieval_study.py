"""Changed labels, class metrics, and retained historical document contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.scene_retrieval_study import classification_metrics, subset_metrics, validate_inputs
from tools.evaluate_scene_retrieval_study import document_plan, check_vectors


class RetrievalStudyTests(unittest.TestCase):
    def row(self, name, truth, predicted):
        return {'file_id':name,'play_id':'play','expected':truth,'ranked_scenes':predicted,'scores':[.9,.8]}

    def test_macro_f1_is_not_macro_recall(self):
        rows=[self.row('1','a',['a','b']),self.row('2','a',['b','a']),self.row('3','b',['b','a'])]
        metrics=classification_metrics(rows,['a','b'])
        self.assertAlmostEqual(metrics['macro_f1'],2/3)
        self.assertAlmostEqual(metrics['scene_macro_recall_at_1'],.75)

    def test_predicted_extra_candidate_is_false_negative(self):
        metrics=classification_metrics([self.row('1','a',['extra','a'])],['a'])
        self.assertEqual(metrics['macro_f1'],0)
        self.assertEqual(metrics['recall_at_3'],1)

    def test_truth_class_missing_rejected(self):
        with self.assertRaisesRegex(ValueError,'truth_class_missing'):
            classification_metrics([self.row('1','a',['a'])],['b'])

    def test_subsets_and_non_plot_events_have_explicit_denominators(self):
        rows=[self.row('1','a',['a','b']),self.row('2','b',['a','b'])]
        result=subset_metrics(rows,[{'scene_id':'a','label':'第1场'},{'scene_id':'b','label':'谢幕'}],
            {'historical':['1'],'added':['2'],'all':['1','2']})
        self.assertEqual(result['historical']['metrics']['count'],1)
        self.assertEqual(result['added']['metrics']['correct_at_1'],0)
        self.assertEqual(result['all']['non_plot_events']['count'],1)

    def test_document_reuse_keeps_existing_indices(self):
        inputs={'expanded':{'scenes':[{'text':'old'}]},'baseline':{'scenes':[{'text':'label'}]}}
        blocks={'blocks':[{'block_id':'b','text':'old'}],'candidates':[{'block_ids':['b']}]}
        from tools.scene_text_blocks import text_hash
        inherited=[{'text':'old','text_sha256':text_hash('old')}]
        documents,index=document_plan(inputs,blocks,inherited)
        self.assertEqual(documents[0],inherited[0])
        self.assertEqual(index,{'long_indices':[[0]],'block_indices':[[0]],'baseline_indices':[[1]]})
        self.assertEqual(len(documents),2)

    def test_invalid_saved_vectors_rejected_without_renormalization(self):
        for value in (np.ones((1,2),dtype='float32'),np.array([[np.nan,0]],dtype='float32'),np.ones((1,1),dtype='float64')):
            with self.assertRaises(ValueError):check_vectors(value,1)
        value=np.array([[.6,.8]],dtype='float32')
        self.assertIs(check_vectors(value,1),value)

    def manifests(self):
        first={'file_id':'1','image_id':'i1','play_id':'p','scene_id':'s','source':'a','content_hash':'h','neutral_path':'old.png','neutral_sha256':'png','status':'ok'}
        new={**first,'file_id':'2','image_id':'i2','neutral_path':'new.png'}
        scenes=[{'play_id':'p','scene_id':'s','label':'第1场','text':'p 第1场','text_source':'scene_label_only'}]
        old={'rows':[first],'scenes':scenes,'excluded':[],'test_locked':False}
        baseline={**deepcopy(old),'rows':[dict(first,neutral_path='copied.png'),new]}
        labels=[{k:v for k,v in row.items() if not k.startswith('neutral_')} for row in baseline['rows']]
        registry=type('Registry',(),{'plays':{'p':'剧'}})()
        return baseline,deepcopy(baseline),deepcopy(baseline),old,labels,registry

    def test_historical_query_order_and_png_changes_rejected(self):
        for change in ('reverse','png'):
            args=self.manifests()
            for manifest in args[:3]:
                if change=='reverse':manifest['rows'].reverse()
                else:manifest['rows'][0]['neutral_sha256']='different'
            with self.assertRaisesRegex(ValueError,'historical_queries_changed'):validate_inputs(*args)

    def test_new_label_change_rejected(self):
        args=self.manifests()
        args[4][1]['scene_id']='other'
        with self.assertRaisesRegex(ValueError,'query_labels_changed'):validate_inputs(*args)

    def test_incompatible_pool_and_duplicate_image_rejected(self):
        args=self.manifests()
        args[2]['scenes'][0]['label']='other'
        with self.assertRaisesRegex(ValueError,'expanded_pool'):validate_inputs(*args)
        args=self.manifests()
        for manifest in args[:3]:manifest['rows'][1]['image_id']='i1'
        with self.assertRaisesRegex(ValueError,'content_unique'):validate_inputs(*args)

    def test_valid_historical_and_added_subset(self):
        self.assertEqual(validate_inputs(*self.manifests()),{'historical':['1'],'added':['2'],'all':['1','2']})


if __name__=='__main__':unittest.main()
