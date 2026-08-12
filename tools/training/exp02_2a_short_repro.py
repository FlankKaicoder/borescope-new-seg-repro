#!/usr/bin/env python3
"""Reproduce the first six epochs while preserving the original 100-epoch schedule."""
from __future__ import annotations
import argparse, copy, hashlib, json, shutil
from pathlib import Path
import torch, ultralytics
from ultralytics import YOLO

def sha(p):
    h=hashlib.sha256();
    with open(p,"rb") as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
dataset_root=a.data.parent; caches=[dataset_root/"labels/train.cache",dataset_root/"labels/val.cache"]; existed={x:x.exists() for x in caches}
model=YOLO(str(a.model))
raw_dir=a.output/"raw_ema_before_validation";raw_dir.mkdir()
def save_raw_ema(trainer):
    logical_epoch=trainer.epoch+1
    snapshot=copy.deepcopy(trainer.ema.ema).float().to(memory_format=torch.contiguous_format)
    if hasattr(snapshot,"criterion"):snapshot.criterion=None
    torch.save({"checkpoint_epoch":logical_epoch,"model":snapshot,"train_args":vars(trainer.args),"ultralytics_version":ultralytics.__version__},raw_dir/f"epoch{logical_epoch}_raw_ema.pt")
def save_initial(trainer):
    initial=a.output/"epoch0_pretrained_7class.pt"
    snapshot=copy.deepcopy(trainer.ema.ema).half().to(memory_format=torch.contiguous_format)
    if hasattr(snapshot,"criterion"):snapshot.criterion=None
    torch.save({"epoch":-1,"best_fitness":None,"model":snapshot,"ema":None,"optimizer":None,"train_args":vars(trainer.args),"version":ultralytics.__version__},initial)
def stop_after_six(trainer):
    if trainer.epoch>=5: trainer.stop=True
model.add_callback("on_pretrain_routine_end",save_initial);model.add_callback("on_train_epoch_end",save_raw_ema);model.add_callback("on_fit_epoch_end",stop_after_six)
args={"data":str(a.data),"imgsz":640,"epochs":100,"batch":32,"seed":42,"deterministic":True,"amp":True,"optimizer":"AdamW","device":0,"project":str(a.output/"ultralytics"),"name":"short_repro","exist_ok":False,"workers":4,"cache":False,"val":True,"plots":True,"save":True,"save_period":1,"fraction":1.0,"verbose":True}
(a.output/"requested_args.json").write_text(json.dumps(args,indent=2)+"\n")
model.train(**args); save_dir=Path(model.trainer.save_dir); shutil.copy2(save_dir/"args.yaml",a.output/"resolved_args.yaml")
entries=[{"checkpoint_epoch":0,"training_file_epoch":None,"path":str(a.output/"epoch0_pretrained_7class.pt")}]
for logical in range(1,7):
    path=save_dir/"weights"/f"epoch{logical-1}.pt"; raw=raw_dir/f"epoch{logical}_raw_ema.pt";entries.append({"checkpoint_epoch":logical,"training_file_epoch":logical-1,"path":str(path),"raw_ema_path":str(raw),"raw_ema_sha256":sha(raw) if raw.is_file() else ""})
for e in entries:
    path=Path(e["path"]);e.update(exists=path.is_file(),sha256=sha(path) if path.is_file() else "",size_bytes=path.stat().st_size if path.is_file() else 0)
(a.output/"checkpoint_manifest.json").write_text(json.dumps(entries,indent=2)+"\n")
removed=[]
for x in caches:
    if not existed[x] and x.exists():x.unlink();removed.append(str(x))
summary={"status":"PASS" if all(x["exists"] for x in entries) else "FAIL","configured_epochs":100,"actual_epochs":len((save_dir/"results.csv").read_text().strip().splitlines())-1,"callback_stop_after_epoch":6,"test_accessed":False,"model_sha256":sha(a.model),"checkpoints":entries,"transient_caches_removed":removed}
(a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary["status"]=="PASS" and summary["actual_epochs"]==6 else 2)
