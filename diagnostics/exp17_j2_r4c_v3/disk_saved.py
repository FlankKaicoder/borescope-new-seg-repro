"""Lossless, layout-preserving disk saved tensors. No model or Autograd calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def nonoverlapping(shape,stride):
    if any(n==0 for n in shape):return True
    span=1
    for step,n in sorted((s,n) for n,s in zip(shape,stride,strict=True) if n>1):
        if step<span:return False
        span+=(n-1)*step
    return True


class DiskSaved:
    def __init__(self,torch,budget,root,*,limit=16*1024**3,reserve=2*1024**3,free_bytes=None):
        self.torch,self.budget,self.root=torch,budget,Path(root).resolve()
        self.root.mkdir(exist_ok=False)
        self.limit,self.reserve=limit,reserve
        self.free_bytes=free_bytes or (lambda:shutil.disk_usage(self.root).free)
        self.records={};self.used_disk=0;self.pack_calls=0;self.unpack_calls=0
        self.write_bytes=0;self.read_bytes=0;self.pack_seconds=0.;self.unpack_seconds=0.;self.fadvise_calls=0
        self.manifest=self.root/'manifest.jsonl'
        self.manifest.touch(exist_ok=False)

    def evict_pages(self,f):
        if hasattr(os,'posix_fadvise'):
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            self.fadvise_calls+=1

    def disk_check(self,additional):
        # Reserve a block for this record as well as rounded payload allocation.
        manifest_blocks=math.ceil(self.manifest.stat().st_size/4096)*4096
        if self.used_disk+manifest_blocks+additional>self.limit:
            raise RuntimeError('Activation disk cache limit would be exceeded')
        if self.free_bytes()-additional<self.reserve:
            raise RuntimeError('Activation disk cache free-space reserve')

    def pack(self,tensor):
        start=time.monotonic();self.budget.check()
        nbytes=tensor.numel()*tensor.element_size()
        self.disk_check(math.ceil(nbytes/4096)*4096+4096)
        # Same CPU conversion as the validated memory offloader, including layout.
        cpu=tensor.detach().cpu()
        shape=list(cpu.shape);stride=list(cpu.stride());offset=cpu.storage_offset()
        if not nonoverlapping(shape,stride):raise RuntimeError('Unsupported overlapping saved CPU layout')
        contiguous=cpu.contiguous().reshape(-1).view(self.torch.uint8)
        payload=memoryview(contiguous.numpy()).cast('B')
        serial=self.pack_calls+1;path=self.root/f'{serial:06d}.bin'
        sha=hashlib.sha256(payload).hexdigest()
        with path.open('xb') as f:
            if f.write(payload)!=nbytes:raise RuntimeError('Short activation-cache write')
            f.flush();os.fsync(f.fileno());self.evict_pages(f)
        record={'serial':serial,'device':str(tensor.device),'dtype':str(cpu.dtype),'shape':shape,'stride':stride,
                'storage_offset':offset,'bytes':nbytes,'sha256':sha,'path':path.name}
        record_digest=digest(record)
        with self.manifest.open('a',encoding='utf8') as f:
            f.write(json.dumps({'record':record,'record_sha256':record_digest},sort_keys=True)+'\n')
            f.flush();os.fsync(f.fileno());self.evict_pages(f)
        self.records[serial]=record
        allocation=getattr(path.stat(),'st_blocks',None)
        self.used_disk+=(allocation*512 if allocation is not None else math.ceil(nbytes/4096)*4096)
        self.pack_calls+=1;self.write_bytes+=nbytes;self.pack_seconds+=time.monotonic()-start
        self.disk_check(0);self.budget.check()
        return serial,record_digest

    def unpack(self,token):
        start=time.monotonic();self.budget.check()
        if not isinstance(token,tuple) or len(token)!=2 or token[0] not in self.records:
            raise RuntimeError('Unknown activation-cache token')
        record=self.records[token[0]]
        if digest(record)!=token[1]:raise RuntimeError('Activation-cache metadata identity mismatch')
        path=self.root/record['path']
        if path.name!=f'{token[0]:06d}.bin' or not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise RuntimeError('Activation-cache path identity mismatch')
        if path.stat().st_size!=record['bytes']:raise RuntimeError('Activation-cache byte length mismatch')
        payload=bytearray(record['bytes'])
        with path.open('rb') as f:
            if f.readinto(payload)!=len(payload):raise RuntimeError('Short activation-cache read')
            self.evict_pages(f)
        if hashlib.sha256(payload).hexdigest()!=record['sha256']:
            raise RuntimeError('Activation-cache content SHA mismatch')
        torch=self.torch;dtype=getattr(torch,record['dtype'].removeprefix('torch.'))
        shape=tuple(record['shape']);stride=tuple(record['stride']);offset=record['storage_offset']
        if record['bytes']==0:
            host=torch.empty_strided(shape,stride,dtype=dtype)
        else:
            logical=torch.frombuffer(payload,dtype=dtype).reshape(shape)
            if logical.stride()==stride and offset==0:host=logical
            else:
                extent=offset+1+sum((n-1)*s for n,s in zip(shape,stride,strict=True))
                storage=torch.empty(extent,dtype=dtype)
                host=torch.as_strided(storage,shape,stride,offset)
                host.copy_(logical)
        if tuple(host.shape)!=shape or host.stride()!=stride or host.dtype!=dtype:
            raise RuntimeError('Activation-cache layout reconstruction failed')
        result=host.to(record['device'])
        self.unpack_calls+=1;self.read_bytes+=record['bytes'];self.unpack_seconds+=time.monotonic()-start
        self.budget.check()
        return result

    def report(self):
        return {'pack_calls':self.pack_calls,'unpack_calls':self.unpack_calls,'payload_written_bytes':self.write_bytes,
                'payload_read_bytes':self.read_bytes,'allocated_data_bytes':self.used_disk,
                'manifest_bytes':self.manifest.stat().st_size,'disk_limit_bytes':self.limit,'disk_reserve_bytes':self.reserve,
                'pack_seconds':self.pack_seconds,'unpack_seconds':self.unpack_seconds,'fadvise_calls':self.fadvise_calls,
                'cache_retained_on_server':True,'forward_recomputations':0,'precision_changes':0,
                'manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest()}
