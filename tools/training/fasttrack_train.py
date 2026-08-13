#!/usr/bin/env python3
"""FastTrack fixed-budget trainer with checkpoint numerical audit."""
from __future__ import annotations
import argparse,hashlib,json,math,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from ultralytics import YOLO
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--name",required=True);p.add_argument("--epochs",type=int,required=True);p.add_argument("--batch",type=int,default=32);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);start=datetime.now(timezone.utc);wall=time.monotonic();m=YOLO(str(a.model));args={"data":str(a.data),"imgsz":640,"batch":a.batch,"epochs":a.epochs,"seed":42,"deterministic":True,"amp":True,"optimizer":"AdamW","device":0,"workers":4,"cache":False,"val":True,"plots":True,"save":True,"project":str(a.output/"ultralytics"),"name":a.name,"exist_ok":False,"verbose":True}
 (a.output/"requested_args.json").write_text(json.dumps(args,indent=2)+"\n");m.train(**args);sd=Path(m.trainer.save_dir);shutil.copy2(sd/"args.yaml",a.output/"resolved_args.yaml")
 import csv
 with (sd/"results.csv").open(newline="") as f:rr=list(csv.DictReader(f))
 train_fields=[x for x in rr[0] if x.startswith("train/") and x.endswith("loss")];train_finite=all(math.isfinite(float(r[x])) for r in rr for x in train_fields)
 cps={}
 for n in ("best.pt","last.pt"):
  q=sd/"weights"/n;c=torch.load(q,map_location="cpu",weights_only=False);model=c.get("ema") or c.get("model");bad=[k for k,v in model.state_dict().items() if not bool(torch.isfinite(v).all())];YOLO(str(q));cps[n]={"path":str(q),"sha256":sha(q),"finite":not bad,"bad_tensors":bad}
 summary={"status":"PASS" if train_finite and all(x["finite"] for x in cps.values()) else "HARD_GATE","test_accessed":False,"start_utc":start.isoformat(),"end_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.monotonic()-wall,"epochs":len(rr),"train_losses_finite":train_finite,"checkpoint_audit":cps,"save_dir":str(sd)}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));raise SystemExit(0 if summary["status"]=="PASS" else 3)
if __name__=="__main__":main()
