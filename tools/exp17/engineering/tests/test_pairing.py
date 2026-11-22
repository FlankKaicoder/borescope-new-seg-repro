import copy
import unittest

from tools.exp17.engineering.data import digest
from tools.exp17.engineering.batch_pairing import CoverageAccumulator, bind_plan, plan_pairs
from tools.exp17.engineering.tests.test_data import fixture_index, native_row


def replay(anchor,n_row,n_slot):
    t=anchor.trusted_record
    return {'n_row':n_row,'n_slot':n_slot,'source_id':anchor.source_id,'polygon_id':anchor.polygon_id,
        'source_sha256':t['a_sha256'],'repair_sha256':t['n_sha256'],'matrix_sha256':t['matrix_identity_sha256'],
        'valid_mask_sha256':t['source_valid_mask_sha256'],'augmentation_sha256':anchor.augmentation_sha256,
        'a_tensor_sha256':digest(['augA',anchor.a_slot]),'n_tensor_sha256':digest(['augN',n_slot]),'support_complete':True}


class PairingTests(unittest.TestCase):
    def test_B_uses_same_batch_nonadmitted_real_originals(self):
        index=fixture_index();rows=[native_row(index,'a',0,0,0),native_row(index,'e',0,1,1)]
        p=plan_pairs(rows,index)
        self.assertEqual(len(p.anchors),1);self.assertEqual(p.anchors[0].b_rows_by_group,{'ge':(1,)})
        self.assertEqual(p.coverage['skip']['no_admitted_N'],1);self.assertEqual(set(p.n_slots),{0})

    def test_same_group_or_other_class_is_not_B_and_no_fake_fallback(self):
        index=fixture_index()
        for other in ['c','d']:
            p=plan_pairs([native_row(index,'a',0,0,0),native_row(index,other,0,1,1)],index)
            self.assertEqual(len(p.anchors),0);self.assertEqual(p.n_slots,{})
            self.assertEqual(p.coverage['skip']['no_cross_source_same_class_B'],1)

    def test_fake_B_class_group_or_polygon_fails_native_identity(self):
        index=fixture_index();rows=[native_row(index,'a',0,0,0),native_row(index,'d',0,1,1)]
        for field,value in [('class_id',0),('group_id','other'),('polygon_id','a_000')]:
            changed=copy.deepcopy(rows);changed[1][field]=value
            with self.assertRaises(ValueError):plan_pairs(changed,index)

    def test_multiple_B_regions_share_one_group_mean_bucket(self):
        index=fixture_index();rows=[native_row(index,'a',0,0,0),native_row(index,'a',1,1,0),native_row(index,'b',0,2,1)]
        p=plan_pairs(rows,index)
        b=next(a for a in p.anchors if a.source_id=='b')
        self.assertEqual(b.b_rows_by_group,{'ga':(0,1)});self.assertEqual(p.coverage['N_image_slots'],2)
        self.assertEqual(p.coverage['anchor_reduction'],'mean_class(mean_A_group(mean_polygon(hinge)))')

    def test_sparse_binding_strict_provenance_and_augmented_support(self):
        index=fixture_index();p=plan_pairs([native_row(index,'a',0,0,0),native_row(index,'b',0,1,1)],index)
        data={a.a_row:replay(a,i,i) for i,a in enumerate(p.anchors)}
        bound,report=bind_plan(p,data);self.assertEqual(len(bound),2);self.assertNotEqual(bound[0].identity,bound[1].identity)
        for key in ['repair_sha256','matrix_sha256','valid_mask_sha256','augmentation_sha256','source_sha256']:
            changed=copy.deepcopy(data);changed[0][key]=digest('wrong')
            with self.assertRaises(ValueError):bind_plan(p,changed)
        changed=copy.deepcopy(data);changed[0]['support_complete']=False
        bound,report=bind_plan(p,changed);self.assertEqual(len(bound),1)
        self.assertEqual(report['replay_skip']['incomplete_augmented_target_valid_support'],1)
        bound,report=bind_plan(p,{})
        self.assertEqual(bound,[]);self.assertEqual(report['replay_skip']['missing_replayed_N'],2)

    def test_replay_cannot_inject_or_mix_N_rows_and_sources(self):
        index=fixture_index();p=plan_pairs([native_row(index,'a',0,0,0),native_row(index,'b',0,1,1)],index)
        data={a.a_row:replay(a,i,i) for i,a in enumerate(p.anchors)}
        changed=copy.deepcopy(data);changed[9]=changed[0]
        with self.assertRaises(ValueError):bind_plan(p,changed)
        changed=copy.deepcopy(data);changed[1]['n_slot']=0
        with self.assertRaises(ValueError):bind_plan(p,changed)
        changed=copy.deepcopy(data);changed[1]['n_row']=0
        with self.assertRaises(ValueError):bind_plan(p,changed)

    def test_coverage_counts_anchor_exposures_not_group_independence(self):
        index=fixture_index();p=plan_pairs([native_row(index,'a',0,0,0),native_row(index,'a',1,1,0),native_row(index,'b',0,2,1)],index)
        accumulator=CoverageAccumulator();accumulator.add(p)
        report=accumulator.report();self.assertEqual(report['group_participation_frequency'],{'ga':2,'gb':1})
        self.assertEqual(report['totals']['planned_ABN'],3);self.assertFalse(report['sampler_modified'])


if __name__=='__main__':unittest.main()
