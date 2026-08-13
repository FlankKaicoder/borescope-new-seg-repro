#!/usr/bin/env python3
"""Fair 30-epoch fine-tuning with identical weighted-sampler machinery."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader,seed_worker
from ultralytics.models.yolo.segment.train import SegmentationTrainer

ACTIVE_MODE="control";HARD=set();SAMPLED=[];CONSUMED=0;STEPS=0;HARD_CONSUMED=0
class LoggingWeightedSampler(WeightedRandomSampler):
 def __iter__(self):
  vals=list(super().__iter__());SAMPLED.extend(vals);return iter(vals)
class FairSamplerTrainer(SegmentationTrainer):
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
  if mode!="train":return super().get_dataloader(dataset_path,batch_size,rank,mode)
  ds=self.build_dataset(dataset_path,mode,batch_size);weights=[2.0 if ACTIVE_MODE=="treatment" and Path(x).stem in HARD else 1.0 for x in ds.im_files]
  g=torch.Generator();g.manual_seed(42);sampler=LoggingWeightedSampler(weights,num_samples=len(ds),replacement=True,generator=g);nw=min(self.args.workers,math.ceil(len(ds)/batch_size));lg=torch.Generator();lg.manual_seed(6148914691236517205)
  return InfiniteDataLoader(dataset=ds,batch_size=batch_size,shuffle=False,num_workers=nw,sampler=sampler,prefetch_factor=4 if nw else None,pin_memory=True,collate_fn=ds.collate_fn,worker_init_fn=seed_worker,generator=lg,drop_last=False)
 def preprocess_batch(self,batch):
  global CONSUMED,HARD_CONSUMED
  CONSUMED+=len(batch["img"]);HARD_CONSUMED+=sum(Path(x).stem in HARD for x in batch["im_file"])
  return super().preprocess_batch(batch)
 def optimizer_step(self):
  global STEPS
  super().optimizer_step();STEPS+=1
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 global ACTIVE_MODE,HARD
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=["control","treatment"],required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--hard-pool",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();ACTIVE_MODE=a.mode
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 with a.hard_pool.open(encoding="utf-8-sig",newline="") as f:HARD={r["image_stem"] for r in csv.DictReader(f) if r["is_hard"]=="True"}
 start=datetime.now(timezone.utc);wall=time.monotonic();args={"data":str(a.data),"imgsz":640,"batch":32,"epochs":30,"seed":42,"deterministic":True,"amp":True,"optimizer":"AdamW","device":0,"workers":4,"cache":False,"val":True,"plots":True,"save":True,"project":str(a.output/"ultralytics"),"name":a.mode,"exist_ok":False,"verbose":True}
 (a.output/"requested_args.json").write_text(json.dumps(args,indent=2)+"\n");m=YOLO(str(a.model));m.train(trainer=FairSamplerTrainer,**args);sd=Path(m.trainer.save_dir)
 with (sd/"results.csv").open(newline="") as f:rr=list(csv.DictReader(f));tf=[x for x in rr[0] if x.startswith("train/") and x.endswith("loss")];finite=all(math.isfinite(float(r[x])) for r in rr for x in tf)
 cps={}
 for n in ("best.pt","last.pt"):
  q=sd/"weights"/n;c=torch.load(q,map_location="cpu",weights_only=False);mo=c.get("ema") or c.get("model");bad=[k for k,v in mo.state_dict().items() if not bool(torch.isfinite(v).all())];YOLO(str(q));cps[n]={"path":str(q),"sha256":sha(q),"finite":not bad}
 expected=668*30;nominal_batches=math.ceil(668/32)*30;summary={"status":"PASS" if finite and CONSUMED==expected and all(x["finite"] for x in cps.values()) else "HARD_GATE","test_accessed":False,"mode":a.mode,"train_images":668,"samples_per_epoch":668,"epochs":30,"total_sampled_images":CONSUMED,"expected_sampled_images":expected,"optimizer_steps":STEPS,"nominal_train_batches":nominal_batches,"optimizer_step_note":"actual optimizer.step calls; equality is enforced after both arms complete because gradient accumulation is trainer-controlled","wall_seconds":time.monotonic()-wall,"train_losses_finite":finite,"checkpoint_audit":cps,"hard_draws":HARD_CONSUMED,"sampler_indices_generated_including_prefetch":len(SAMPLED),"save_dir":str(sd)}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));raise SystemExit(0 if summary["status"]=="PASS" else 3)
if __name__=="__main__":main()
