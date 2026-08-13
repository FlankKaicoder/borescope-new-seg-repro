#!/usr/bin/env python3
"""Exp10 baseline and paired fair-sampler training."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader,seed_worker
from ultralytics.models.yolo.segment.train import SegmentationTrainer

HARD=set(); MODE='control'; SEED=42; CONSUMED=STEPS=HARD_CONSUMED=0
class FairTrainer(SegmentationTrainer):
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
  if mode!='train': return super().get_dataloader(dataset_path,batch_size,rank,mode)
  ds=self.build_dataset(dataset_path,mode,batch_size); weights=[2. if MODE=='treatment' and Path(x).stem in HARD else 1. for x in ds.im_files]
  g=torch.Generator().manual_seed(SEED); sampler=WeightedRandomSampler(weights,len(ds),replacement=True,generator=g); nw=min(self.args.workers,math.ceil(len(ds)/batch_size)); lg=torch.Generator().manual_seed(6148914691236517205+SEED)
  return InfiniteDataLoader(dataset=ds,batch_size=batch_size,shuffle=False,num_workers=nw,sampler=sampler,prefetch_factor=4 if nw else None,pin_memory=True,collate_fn=ds.collate_fn,worker_init_fn=seed_worker,generator=lg,drop_last=False)
 def preprocess_batch(self,batch):
  global CONSUMED,HARD_CONSUMED; CONSUMED+=len(batch['img']); HARD_CONSUMED+=sum(Path(x).stem in HARD for x in batch['im_file']); return super().preprocess_batch(batch)
 def optimizer_step(self):
  global STEPS; super().optimizer_step(); STEPS+=1
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def finite_ckpt(p):
 c=torch.load(p,map_location='cpu',weights_only=False); m=c.get('ema') or c.get('model'); return all(bool(torch.isfinite(v).all()) for v in m.state_dict().values() if torch.is_tensor(v))
def main():
 global HARD,MODE,SEED
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','control','treatment'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--hard-pool',type=Path);p.add_argument('--epochs',type=int);a=p.parse_args();MODE=a.mode;SEED=a.seed
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True); epochs=a.epochs or (100 if a.mode=='baseline' else 30)
 if a.mode!='baseline':
  if not a.hard_pool:raise ValueError('hard pool required')
  HARD={r['image_stem'] for r in csv.DictReader(a.hard_pool.open(encoding='utf-8-sig')) if r['is_hard'].lower()=='true'}
 args={'data':str(a.data),'imgsz':640,'batch':32,'epochs':epochs,'seed':a.seed,'deterministic':True,'amp':True,'optimizer':'AdamW','device':0,'workers':4,'cache':False,'val':True,'plots':True,'save':True,'project':str(a.output/'ultralytics'),'name':a.mode,'exist_ok':False,'verbose':True}
 (a.output/'requested_args.json').write_text(json.dumps(args,indent=2)+'\n'); start=time.monotonic(); y=YOLO(str(a.model)); y.train(**args,**({'trainer':FairTrainer} if a.mode!='baseline' else {})); sd=Path(y.trainer.save_dir)
 rows=list(csv.DictReader((sd/'results.csv').open())); losskeys=[k for k in rows[0] if k.startswith('train/') and k.endswith('loss')]; losses_finite=all(math.isfinite(float(r[k])) for r in rows for k in losskeys); cps={}
 for n in ('best.pt','last.pt'):
  q=sd/'weights'/n;cps[n]={'path':str(q),'sha256':sha(q),'finite':finite_ckpt(q),'reload':bool(YOLO(str(q)))}
 expected=(668*epochs if a.mode!='baseline' else None); status='PASS' if losses_finite and all(x['finite'] for x in cps.values()) and (expected is None or CONSUMED==expected) else 'HARD_GATE'
 summary={'status':status,'test_accessed':False,'mode':a.mode,'seed':a.seed,'initial_checkpoint':str(a.model),'initial_sha256':sha(a.model),'epochs':epochs,'train_losses_finite':losses_finite,'checkpoint_audit':cps,'total_sampled_images':CONSUMED if a.mode!='baseline' else None,'expected_sampled_images':expected,'optimizer_steps':STEPS if a.mode!='baseline' else None,'hard_draws':HARD_CONSUMED if a.mode!='baseline' else None,'save_dir':str(sd),'wall_seconds':time.monotonic()-start}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return 0 if status=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
