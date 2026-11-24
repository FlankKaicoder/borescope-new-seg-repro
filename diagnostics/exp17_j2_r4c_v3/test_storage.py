"""Manual tiny-tensor storage roundtrips; all Autograd/model entrypoints blocked."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from disk_saved import DiskSaved,nonoverlapping

def forbid(*a,**kw):raise RuntimeError('No Autograd or model operations in storage tests')

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent/'settings/tmp')
        self.root=Path(self.tmp.name)
        class Budget:
            def check(self):return {}
        self.budget=Budget()
        self.cache=DiskSaved(torch,self.budget,self.root/'cache',reserve=0)

    def tearDown(self):self.tmp.cleanup()

    def byte_identity(self,t):return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

    def roundtrip(self,x):
        before=self.byte_identity(x);token=self.cache.pack(x);y=self.cache.unpack(token)
        self.assertEqual(self.byte_identity(y),before)
        self.assertEqual(x.dtype,y.dtype);self.assertEqual(x.shape,y.shape);self.assertEqual(x.stride(),y.stride())
        self.assertFalse(y.requires_grad);self.assertEqual(self.byte_identity(x),before)
        return token,y

    def test_contiguous(self):self.roundtrip(torch.arange(24,dtype=torch.float32).reshape(2,3,4))
    def test_transposed(self):self.roundtrip(torch.arange(24,dtype=torch.float32).reshape(4,6).t())
    def test_stride_and_offset(self):self.roundtrip(torch.arange(40,dtype=torch.float32)[3:31:2])
    def test_channels_last(self):self.roundtrip(torch.arange(120,dtype=torch.float32).reshape(2,3,4,5).to(memory_format=torch.channels_last))
    def test_scalar(self):self.roundtrip(torch.tensor(-.0))
    def test_empty(self):self.roundtrip(torch.empty((0,3)))
    def test_bool(self):self.roundtrip(torch.tensor([True,False,True]))
    def test_int64(self):self.roundtrip(torch.tensor([0,2**40,-13],dtype=torch.int64))
    def test_float16(self):self.roundtrip(torch.tensor([1.25,-.0],dtype=torch.float16))
    def test_bfloat16(self):self.roundtrip(torch.tensor([1.25,-.0],dtype=torch.bfloat16))
    def test_nonfinite_bits_preserved(self):self.roundtrip(torch.tensor([float('nan'),float('inf'),-0.,1.],dtype=torch.float32))
    def test_read_repeated_without_forward(self):
        token,_=self.roundtrip(torch.arange(5.))
        self.cache.unpack(token);self.cache.unpack(token)
        self.assertEqual(self.cache.report()['unpack_calls'],3)
        self.assertEqual(self.cache.report()['pack_calls'],1)
    def test_source_mutation_does_not_change_saved_copy(self):
        x=torch.arange(5.);token=self.cache.pack(x);x.fill_(99)
        self.assertEqual(self.cache.unpack(token).tolist(),[0.,1.,2.,3.,4.])
    def test_corrupt_payload_refused(self):
        token=self.cache.pack(torch.arange(5.));p=self.cache.root/'000001.bin'
        b=bytearray(p.read_bytes());b[0]^=1;p.write_bytes(b)
        with self.assertRaisesRegex(RuntimeError,'SHA'):self.cache.unpack(token)
    def test_short_payload_refused(self):
        token=self.cache.pack(torch.arange(5.));(self.cache.root/'000001.bin').write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError,'length'):self.cache.unpack(token)
    def test_metadata_mutation_refused(self):
        token=self.cache.pack(torch.arange(5.));self.cache.records[token[0]]['shape']=[7]
        with self.assertRaisesRegex(RuntimeError,'metadata'):self.cache.unpack(token)
    def test_unknown_token_refused(self):
        with self.assertRaisesRegex(RuntimeError,'Unknown'):self.cache.unpack((999,'0'*64))
    def test_quota_before_write(self):
        small=DiskSaved(torch,self.budget,self.root/'small',limit=1,reserve=0)
        with self.assertRaisesRegex(RuntimeError,'limit'):small.pack(torch.ones(2))
        self.assertEqual(list(small.root.glob('*.bin')),[])
    def test_free_reserve_before_write(self):
        small=DiskSaved(torch,self.budget,self.root/'lowfree',reserve=100,free_bytes=lambda:100)
        with self.assertRaisesRegex(RuntimeError,'reserve'):small.pack(torch.ones(2))
        self.assertEqual(list(small.root.glob('*.bin')),[])
    def test_overlapping_layout_refused(self):
        with self.assertRaisesRegex(RuntimeError,'overlapping'):self.cache.pack(torch.ones(1).expand(4))
    def test_manifest_matches_payload(self):
        token=self.cache.pack(torch.arange(4.))
        row=json.loads(self.cache.manifest.read_text())
        self.assertEqual(row['record_sha256'],token[1])
        self.assertEqual(row['record']['sha256'],hashlib.sha256((self.cache.root/row['record']['path']).read_bytes()).hexdigest())
    def test_existing_directory_refused(self):
        with self.assertRaises(FileExistsError):DiskSaved(torch,self.budget,self.cache.root)
    def test_no_cuda_or_parameter_grad(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    torch.set_num_threads(1)
    with patch.object(torch.autograd,'grad',forbid),patch.object(torch.autograd,'backward',forbid),\
         patch.object(torch.Tensor,'backward',forbid),patch.object(torch.nn.Module,'_call_impl',forbid):
        unittest.main(verbosity=2)
