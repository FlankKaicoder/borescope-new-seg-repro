#!/usr/bin/env python3
"""Probe per-batch segmentation validation loss in AMP-equivalent and FP32 paths."""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from pathlib import Path
from typing import Any

import torch
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import autocast, select_device


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    xf=x.detach(); finite=torch.isfinite(xf); n=xf.numel(); vals=xf[finite].float()
    return {"shape":list(xf.shape),"dtype":str(xf.dtype),"numel":n,"finite_ratio":float(finite.sum().item()/n) if n else 1.0,
            "min":float(vals.min()) if vals.numel() else None,"max":float(vals.max()) if vals.numel() else None,
            "mean":float(vals.mean()) if vals.numel() else None,"nan_count":int(torch.isnan(xf).sum()),"inf_count":int(torch.isinf(xf).sum())}


def flatten_tensors(obj: Any, prefix="raw") -> list[tuple[str,torch.Tensor]]:
    if isinstance(obj,torch.Tensor):return [(prefix,obj)]
    if isinstance(obj,dict):
        return [item for k,v in obj.items() for item in flatten_tensors(v,f"{prefix}.{k}")]
    if isinstance(obj,(tuple,list)):
        return [item for i,v in enumerate(obj) for item in flatten_tensors(v,f"{prefix}[{i}]")]
    return []


def load_model(path: Path, device: torch.device):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    model=ckpt.get("ema") or ckpt.get("model")
    if model is None:raise RuntimeError(f"No model/ema in {path}")
    model=model.float().to(device).eval()
    model.criterion=None
    return model


def make_validator(data_yaml: Path, device: torch.device, amp: bool) -> SegmentationValidator:
    args={"data":str(data_yaml),"imgsz":640,"batch":32,"device":"0","workers":4,"task":"segment","mode":"val","split":"val",
          "rect":False,"cache":False,"overlap_mask":True,"mask_ratio":4,"plots":False,"save_json":False,"save_txt":False,
          "save_conf":False,"compile":False,"quantize":16 if amp else None,"half":amp,"conf":None,"iou":0.7,"max_det":300,
          "single_cls":False,"agnostic_nms":False,"augment":False,"verbose":False}
    v=SegmentationValidator(args=args);v.training=True;v.device=device;v.data=check_det_dataset(str(data_yaml),split="val")
    v.stride=32;v.args.imgsz=check_imgsz(v.args.imgsz,stride=v.stride,max_dim=1)
    v.dataloader=v.get_dataloader(v.data["val"],v.args.batch)
    return v


def audit_gt(batch: dict[str,Any]) -> dict[str,Any]:
    bboxes=batch["bboxes"]; masks=batch["masks"]; cls=batch["cls"]; bi=batch["batch_idx"]
    mask_areas=[]
    if masks.ndim==3:
        if masks.shape[0]==batch["img"].shape[0]:
            for i in range(masks.shape[0]):
                ids=torch.unique(masks[i]);mask_areas.extend(int((masks[i]==j).sum()) for j in ids if int(j)>0)
        else: mask_areas=[int(x.sum()) for x in masks]
    return {"image_stems":[Path(x).stem for x in batch["im_file"]],"image_tensor":tensor_stats(batch["img"]),
            "instance_count":int(cls.shape[0]),"gt_classes":[int(x) for x in cls.view(-1).cpu()],"bboxes":tensor_stats(bboxes),
            "bbox_min":float(bboxes.min()) if bboxes.numel() else None,"bbox_max":float(bboxes.max()) if bboxes.numel() else None,
            "masks":tensor_stats(masks),"mask_pixel_counts":mask_areas,"empty_or_degenerate_mask":any(x<=0 for x in mask_areas),
            "batch_idx":tensor_stats(bi),"cls":tensor_stats(cls)}


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);device=select_device("0")
    entries=json.loads(a.manifest.read_text())
    checkpoints=[]
    for e in entries:
        path=Path(e["path"]) if e["checkpoint_epoch"]==0 else Path(e["raw_ema_path"])
        checkpoints.append((int(e["checkpoint_epoch"]),path))
    rows=[]; first=None
    for epoch,path in checkpoints:
        for mode,amp_on in (("validator_equivalent_amp",True),("fp32",False)):
            model=load_model(path,device);validator=make_validator(a.data,device,amp_on)
            for batch_index,raw_batch in enumerate(validator.dataloader):
                batch=validator.preprocess(raw_batch)
                input_finite=bool(torch.isfinite(batch["img"]).all())
                with torch.inference_mode(), autocast(amp_on,device=device.type):
                    preds=model(batch["img"])
                    pred_tensors=flatten_tensors(preds)
                    raw_finite=all(bool(torch.isfinite(x).all()) for _,x in pred_tensors)
                    _,losses=model.loss(batch,preds)
                vals={k:float(v.detach().float().cpu()) for k,v in losses.items()}
                stems=[Path(x).stem for x in batch["im_file"]]
                row={"checkpoint_epoch":epoch,"checkpoint_sha256":sha256(path),"precision_mode":mode,"batch_index":batch_index,
                     "image_count":len(stems),"instance_count":int(batch["cls"].shape[0]),"image_stems":json.dumps(stems),
                     "box_loss":vals.get("box_loss"),"seg_loss":vals.get("seg_loss"),"cls_loss":vals.get("cls_loss"),"dfl_loss":vals.get("dfl_loss"),
                     "box_loss_finite":math.isfinite(vals.get("box_loss",math.nan)),"seg_loss_finite":math.isfinite(vals.get("seg_loss",math.nan)),
                     "cls_loss_finite":math.isfinite(vals.get("cls_loss",math.nan)),"dfl_loss_finite":math.isfinite(vals.get("dfl_loss",math.nan)),
                     "input_finite":input_finite,"raw_prediction_finite":raw_finite}
                rows.append(row)
                if first is None and not all((row["box_loss_finite"],row["seg_loss_finite"],row["cls_loss_finite"],row["dfl_loss_finite"])):
                    first={"row":row,"checkpoint_path":str(path),"gt_input_audit":audit_gt(batch),
                           "raw_prediction_tensor_stats":{name:tensor_stats(x) for name,x in pred_tensors},
                           "loss_components":vals}
            del model,validator;torch.cuda.empty_cache()
    fields=list(rows[0])
    with (a.output/"val_loss_batch_probe.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (a.output/"first_nonfinite_batch_audit.json").write_text(json.dumps(first,indent=2)+"\n")
    first_comp=None
    if first:
        for c in ("box_loss","seg_loss","cls_loss","dfl_loss"):
            if not first["row"][c+"_finite"]:first_comp=c;break
    summary={"status":"PASS","test_accessed":False,"checkpoints":len(checkpoints),"rows":len(rows),
             "first_nonfinite_checkpoint":first["row"]["checkpoint_epoch"] if first else None,
             "first_nonfinite_precision_mode":first["row"]["precision_mode"] if first else None,
             "first_nonfinite_batch":first["row"]["batch_index"] if first else None,"first_nonfinite_loss_component":first_comp,
             "nonfinite_rows":sum(not all((r["box_loss_finite"],r["seg_loss_finite"],r["cls_loss_finite"],r["dfl_loss_finite"])) for r in rows),
             "raw_prediction_nonfinite_rows":sum(not r["raw_prediction_finite"] for r in rows),"input_nonfinite_rows":sum(not r["input_finite"] for r in rows)}
    (a.output/"val_loss_probe_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
