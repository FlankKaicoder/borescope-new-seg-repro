"""Non-model tests: no Torch import, no Autograd, no research calls."""
import ast
import copy
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
from guards import attach_method_counter, validate_dispatch, audit_policy, resolve_at, seal_before_check
from core import Budget, LABELS


class GuardTests(unittest.TestCase):
    def counts(self):
        return {'A':list(range(24)), 'N':list(range(17)), 'projection':2,
                'root_dispatch':[], 'method_dispatch':[]}

    def stub(self, owned=False):
        counts = self.counts()
        class Fake:
            def __call__(self, value):
                counts['root_dispatch'].append('dict' if isinstance(value,dict) else 'tensor')
                return self.forward(value)
            def forward(self, value):
                return self.loss(value) if isinstance(value,dict) else 'SENTINEL'
            def loss(self, batch):
                return self.forward(batch['img'])
        model=Fake()
        if owned:model.forward=model.forward
        old=model.__dict__.get('forward')
        restore=attach_method_counter(model,counts,lambda:True)
        return model, counts, restore, old

    def test_direct_recursion_separate_from_module_hook(self):
        model,c,restore,_=self.stub()
        self.assertEqual(model({'img':'DATA'}),'SENTINEL')
        self.assertEqual(c['root_dispatch'],['dict'])
        self.assertEqual(c['method_dispatch'],['dict','tensor'])
        validate_dispatch(c);restore();self.assertNotIn('forward',model.__dict__)

    def test_old_assertion_reproduced_without_model(self):
        model,c,restore,_=self.stub();model({'img':'DATA'})
        self.assertNotEqual(c['root_dispatch'],['dict','tensor']);restore()

    def test_restore_owned_forward(self):
        model,c,restore,old=self.stub(True);model({'img':'DATA'});restore()
        self.assertIs(model.__dict__['forward'],old)

    def test_extra_method_rejected(self):
        model,c,restore,_=self.stub();model({'img':'DATA'})
        with self.assertRaises(RuntimeError):model.forward('EXTRA')
        restore()

    def test_tensor_first_rejected(self):
        model,c,restore,_=self.stub()
        with self.assertRaises(RuntimeError):model.forward('TENSOR_FIRST')
        restore()

    def test_unclaimed_method_rejected(self):
        model,c,restore,_=self.stub();restore()
        restore=attach_method_counter(model,c,lambda:False)
        with self.assertRaises(RuntimeError):model({'img':'DATA'})
        restore()

    def test_wrong_layer_or_projection_counts_rejected(self):
        for field in ('A','N','projection','root_dispatch','method_dispatch'):
            c={'A':list(range(24)), 'N':list(range(17)), 'projection':2,
               'root_dispatch':['dict'], 'method_dispatch':['dict','tensor']}
            c[field]=0 if field=='projection' else []
            with self.assertRaisesRegex(RuntimeError,'observed='):validate_dispatch(c)

    def test_observation_saved_before_failure(self):
        saved=[]
        def fail():raise RuntimeError('acceptance')
        with self.assertRaises(RuntimeError):seal_before_check(saved.append,fail,{'actual':1})
        self.assertEqual(saved,[{'actual':1}])

    def budget(self, used=None, active=None):
        b=Budget.__new__(Budget);b.used=list(used or []);b.active=active
        b.check=lambda:{};b.event=lambda *a,**kw:None
        return b

    def test_only_new_ten_call_budget(self):
        self.assertEqual(len(LABELS),10)
        self.assertFalse(any(x.startswith('cpu_') for x in LABELS))
        b=self.budget()
        for x in LABELS:b.begin(x);b.commit()
        self.assertEqual(b.used,LABELS)

    def test_duplicate_rejected(self):
        b=self.budget([LABELS[0]])
        with self.assertRaises(RuntimeError):b.begin(LABELS[0])

    def test_pending_rejected(self):
        b=self.budget([LABELS[0]],LABELS[0])
        with self.assertRaises(RuntimeError):b.begin(LABELS[1])

    def test_order_and_excess_rejected(self):
        with self.assertRaises(RuntimeError):self.budget().begin(LABELS[1])
        with self.assertRaises(RuntimeError):self.budget(LABELS).begin(LABELS[0])

    def test_commit_without_attempt_rejected(self):
        with self.assertRaises(RuntimeError):self.budget().commit()

    def root(self):return (Path.cwd()/'FAKE_DIAGNOSTIC_ROOT').resolve()
    def fd(self,fd):return self.root()/'settings/tmp'

    def test_relative_dirfd_cleanup_allowed(self):
        audit_policy(self.root(),'os.remove',('temporary',7),fd_path=self.fd)
        self.assertEqual(resolve_at('temporary',7,fd_path=self.fd),self.root()/'settings/tmp/temporary')

    def test_relative_escape_rejected(self):
        with self.assertRaises(RuntimeError):audit_policy(self.root(),'os.remove',('../../calls.jsonl',7),fd_path=self.fd)

    def test_original_absolute_delete_rejected(self):
        with self.assertRaises(RuntimeError):audit_policy(self.root(),'os.remove',(str(self.root().parent/'original.pt'),-1))

    def test_rename_checks_both_ends(self):
        audit_policy(self.root(),'os.rename',('a','b',7,7),fd_path=self.fd)
        with self.assertRaises(RuntimeError):audit_policy(self.root(),'os.rename',('a','../../calls.jsonl',7,7),fd_path=self.fd)

    def test_original_write_rejected_and_read_allowed(self):
        path=str(self.root().parent/'original.pt')
        audit_policy(self.root(),'open',(path,'r',os.O_RDONLY))
        with self.assertRaises(RuntimeError):audit_policy(self.root(),'open',(path,'w',os.O_WRONLY))
        audit_policy(self.root(),'open',(str(self.root()/'receipt.json'),'x',os.O_CREAT|os.O_WRONLY))

    def test_no_torch_import(self):self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
