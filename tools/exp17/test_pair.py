"""CPU proofs for identity, group reduction, replay and BN/RNG semantics."""

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.exp15.regions import Regions
from tools.exp17 import pair_loss as p
from tools.exp17.prototype import reserve, synthetic_batch


def regions(classes=(0,0,1,1),groups=("a","b","c","d")):
    count=len(classes)
    return Regions(np.arange(count),np.zeros(count,dtype=np.int64),np.array(classes),np.array(groups),
                   np.ones(count,dtype=np.int64),np.ones(count),1)


def binding(row=0,group="a",label=0,polygon="poly",version="v1"):
    hashes=[p.digest(["synthetic",key]) for key in range(5)]
    return p.PairBinding(row,row,group,polygon,label,version,*hashes,a_tensor_sha256=hashes[0],n_tensor_sha256=hashes[1],synthetic=True,
                         support_complete=True,contamination_clear=True,unambiguous_region=True)


class PairTests(unittest.TestCase):
    def test_hinge_and_missing_B(self):
        r=regions();a=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],requires_grad=True)
        normal=a.detach().clone().requires_grad_()
        loss,report=p.source_ranking(a,normal,r,[binding()])
        torch.testing.assert_close(loss,torch.tensor(1.2))
        self.assertEqual(report['paired_anchors'],1)
        loss.backward();self.assertGreater(float(a.grad.abs().sum()),0)
        zero,report=p.source_ranking(a,normal,regions((0,1,1,1)),[binding()])
        self.assertEqual(float(zero.detach()),0);self.assertEqual(report['skipped']['no_cross_source_same_class_B'],1)

    def test_B_group_mean_prevents_replica_inflation(self):
        r=regions((0,0,0,1),("a","b","c","d"))
        a=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
        first,_=p.source_ranking(a,a,r,[binding()])
        aa=torch.cat([a,a[1:2].repeat(7,1)])
        rr=regions((0,0,0,1,*([0]*7)),("a","b","c","d",*(["b"]*7)))
        second,_=p.source_ranking(aa,aa,rr,[binding()])
        torch.testing.assert_close(first,second)

    def test_version_selection_and_quality_fail_closed(self):
        r=regions();state=random.getstate()
        chosen,skipped=p.select_synthetic_bindings([binding(),binding(version='v2')],r)
        self.assertEqual(len(chosen),1);self.assertEqual(random.getstate(),state)
        chosen,skipped=p.select_synthetic_bindings([replace(binding(),support_complete=False)],r)
        self.assertFalse(chosen);self.assertEqual(skipped['incomplete_support'],1)
        for item in [replace(binding(),synthetic=False),replace(binding(),n_row=1),replace(binding(),class_id=1),replace(binding(),group='wrong')]:
            with self.assertRaises(ValueError):p.select_synthetic_bindings([item],r)
        with self.assertRaises(ValueError):p.select_synthetic_bindings([binding(),binding()],r)
        with self.assertRaises(ValueError):p.real_pair_training_gate({'user_pass':True,'data_gate':'PASS'})

    def test_synthetic_embedding_optimization_reduces_ranking(self):
        torch.manual_seed(3)
        a=nn.Parameter(torch.tensor([[1.,.1],[.1,1.],[1.,.2],[.2,1.]]))
        normal=nn.Parameter(a.detach().clone())
        optimizer=torch.optim.SGD([a,normal],lr=.2)
        first=None
        for _ in range(50):
            optimizer.zero_grad();loss,_=p.source_ranking(F.normalize(a,dim=1),F.normalize(normal,dim=1),regions(),[binding()])
            if first is None:first=float(loss.detach())
            loss.backward();optimizer.step()
        self.assertLess(float(loss.detach()),first*.1)

    def test_wrong_normal_bytes_and_unpaired_replacements_reject_before_forward(self):
        batch,r,normal,bindings=synthetic_batch('cpu')
        with self.assertRaisesRegex(ValueError,'tensor bytes'):
            p.forward_joint(nn.Identity(),batch,r,normal.roll(1,0),bindings)
        with self.assertRaisesRegex(ValueError,'unpaired/unadmitted'):
            p.forward_joint(nn.Identity(),batch,r,normal,bindings[:-1])

    def test_same_slot_repair_version_conflict_is_not_admitted(self):
        r=regions((0,0,1,1),('a','a','c','d'));r.slots[1]=0
        chosen,skipped=p.select_synthetic_bindings([
            binding(0,'a',0,'p0','v1'),binding(1,'a',0,'p1','v2')],r)
        self.assertEqual(chosen,[])
        self.assertEqual(skipped['counterfactual_slot_version_conflict'],2)

    def test_auxiliary_BN_uses_same_batch_statistics_and_restores_RNG(self):
        torch.manual_seed(5)
        model=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.BatchNorm2d(8),nn.ReLU()).train()
        x=torch.rand(4,3,16,16)
        main=model(x)
        before={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone();py=random.getstate();numpy=np.random.get_state()
        with p.auxiliary_batch_statistics(model):
            same=model(x);torch.rand(7);random.random();np.random.rand(3)
        torch.testing.assert_close(main,same,rtol=0,atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),py)
        self.assertTrue(np.array_equal(np.random.get_state()[1],numpy[1]))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        same.sum().backward();self.assertGreater(float(model[0].weight.grad.abs().sum()),0)
        with self.assertRaisesRegex(RuntimeError,'expected'):
            with p.auxiliary_batch_statistics(model):raise RuntimeError('expected')
        self.assertTrue(model[1].training and model[1].track_running_stats)

    def test_flip_letterbox_and_valid_mask_trace_contract(self):
        image=np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3)
        target=np.zeros((8,12),bool);target[2:6,3:8]=1
        valid=np.ones((8,12),bool)
        # x -> -x+15; y -> y+4. This pads then flips in a common augmented16-square frame.
        trace=p.ReplayTrace((-1,0,15,0,1,4,0,0,1),(16,16))
        a,n,mask,v,report=p.replay_aligned_pair(image,image.copy(),target,valid,trace)
        self.assertTrue(np.array_equal(a,n));self.assertEqual(int(mask.sum()),int(target.sum()))
        self.assertTrue(report['support_complete']);self.assertEqual(report['trace_sha256'],trace.identity)
        self.assertFalse(v[:4].any());self.assertTrue(mask[6:10,8:13].all())
        partial=valid.copy();partial[3,4]=0
        *_,report=p.replay_aligned_pair(image,image,target,partial,trace)
        self.assertFalse(report['support_complete'])
        with self.assertRaises(ValueError):p.replay_aligned_pair(image,image,target,valid,p.ReplayTrace((0,)*9,(16,16)))

    def test_durable_global_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual([reserve(root) for _ in range(3)],[1,2,3])
            with self.assertRaises(ValueError):reserve(root)
            self.assertEqual((root/'synthetic.attempts').read_text().splitlines(),['1','2','3'])


if __name__=='__main__':unittest.main()
