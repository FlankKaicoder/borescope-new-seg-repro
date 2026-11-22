import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.exp15.regions import Regions
from tools.exp17.pair_loss import tensor_identity
from tools.exp17.engineering import runtime
from tools.exp17.engineering.data import digest
from tools.exp17.engineering.tests.test_data import fixture_index


class Trace:
    def __init__(self,sid,group,count,reject_all=False):
        self.original_hw=(8,8);self.reject_all=reject_all
        self.sha256=digest(['synthetic_trace',sid]);self.native_target_rows=[]
        self.native_masks=np.zeros((1,8,8),np.uint8)
        for i in range(count):
            self.native_target_rows.append({'source_id':sid,'group_id':group,'class_id':0,'polygon_id':f'{sid}_{i:03d}',
                'annotation_ordinal':i,'native_label_ordinal':i,'native_ordinal':i,'mask_id':i+1})
            self.native_masks[0,1+i*3:3+i*3,1+i*3:3+i*3]=i+1


class Cache:
    def __init__(self):self.kinds=[]
    def read(self,path,expected_sha,kind):
        self.kinds.append(kind)
        return np.full((8,8,3),64,np.uint8) if kind=='N' else np.full((8,8),255,np.uint8)
    def report(self):return {'fixture_decodes':self.kinds.count('N')}


def support(valid,trace):
    return {'support':[{'native_ordinal':r['native_ordinal'],'polygon_id':r['polygon_id'],
                       'support_complete':not trace.reject_all} for r in trace.native_target_rows]}


def replay(image,valid,trace):
    return {'img':torch.full((3,32,32),64,dtype=torch.uint8),'masks':torch.from_numpy(trace.native_masks.copy())}


def fixture(reject_a=False):
    index=fixture_index();traces=(Trace('a','ga',2,reject_a),Trace('b','gb',1))
    masks=torch.stack([torch.from_numpy(t.native_masks[0].copy()) for t in traces])
    batch={'img':torch.full((2,3,32,32),128,dtype=torch.uint8),'masks':masks,
           'cls':torch.zeros(3,1),'bboxes':torch.ones(3,4)*.2,'batch_idx':torch.tensor([0,0,1]),
           'sem_masks':torch.zeros(2,8,8,dtype=torch.long),'im_file':['/frozen/train/a.png','/frozen/train/b.png'],
           'j2_replay':traces}
    regions=Regions(np.array([0,0,1]),np.array([0,1,0]),np.array([0,0,0]),np.array(['ga','ga','gb']),
                    np.array([4,4,4]),np.ones(3),2)
    return index,batch,regions


class RuntimeTests(unittest.TestCase):
    def test_metadata_coverage_has_no_N_decode_or_native_mutation(self):
        index,batch,regions=fixture();before={k:v.clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        cache=Cache()
        with patch.object(runtime,'inspect_regions',return_value=regions),patch.object(runtime,'support_from_trace',side_effect=support):
            p=runtime.prepare_batch(batch,index,{},decode_N=False,asset_cache=cache)
        self.assertIs(p.batch,batch);self.assertEqual(p.coverage['actual_valid_ABN'],3)
        self.assertEqual(p.coverage['validNslots'],[0,1]);self.assertNotIn('N',cache.kinds)
        self.assertTrue(all(torch.equal(batch[k],v) for k,v in before.items()))
        self.assertIsNone(p.normal_images)

    def test_sparse_compaction_and_float_tensor_provenance(self):
        index,batch,regions=fixture(reject_a=True);cache=Cache()
        with patch.object(runtime,'inspect_regions',return_value=regions),patch.object(runtime,'support_from_trace',side_effect=support),patch.object(runtime,'replay_normal',side_effect=replay):
            p=runtime.prepare_batch(batch,index,{},decode_N=True,device='cpu',asset_cache=cache)
        self.assertEqual(p.coverage['actual_valid_ABN'],1);self.assertEqual(p.coverage['validNslots'],[1])
        self.assertEqual(cache.kinds.count('N'),1);self.assertEqual(tuple(p.normal_images.shape),(1,3,32,32))
        b=p.bindings[0];self.assertEqual((b.a_row,b.n_row,b.a_slot,b.n_slot),(2,0,1,0))
        self.assertEqual(b.a_tensor_sha256,tensor_identity(p.batch['img'][1]))
        self.assertEqual(b.n_tensor_sha256,tensor_identity(p.normal_images[0]))
        self.assertTrue(torch.equal(p.batch['img'],batch['img'].float()/255))
        self.assertEqual(p.normal_regions.slots.tolist(),[0]);self.assertEqual(p.normal_regions.ordinals.tolist(),[0])

    def test_wrong_trace_class_fails_before_N_decode(self):
        index,batch,regions=fixture();batch['j2_replay'][0].native_target_rows[0]['class_id']=1;cache=Cache()
        with patch.object(runtime,'inspect_regions',return_value=regions):
            with self.assertRaises(ValueError):runtime.prepare_batch(batch,index,{},asset_cache=cache)
        self.assertEqual(cache.kinds,[])

    def test_formal_entry_hard_reject_and_normalization_guard(self):
        with self.assertRaises(RuntimeError):runtime.EngineeringTrainer.run_formal(None)
        with self.assertRaises(RuntimeError):runtime.EngineeringTrainer.train(None)
        with self.assertRaises(ValueError):runtime.normalize_batch({'img':torch.full((1,3,4,4),256.)},'cpu')


if __name__=='__main__':unittest.main()
