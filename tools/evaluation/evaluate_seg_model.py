#!/usr/bin/env python3
"""Export authoritative Ultralytics segmentation validation metrics."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--weights",type=Path,required=True); p.add_argument("--data",type=Path,required=True)
    p.add_argument("--split",choices=["train","val"],default="val"); p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--conf",type=float,default=0.001); p.add_argument("--iou",type=float,default=0.7)
    p.add_argument("--batch",type=int,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    cache_paths=[a.data.parent/"labels"/"train.cache",a.data.parent/"labels"/"val.cache"]
    cache_existed={path:path.exists() for path in cache_paths}
    metrics=YOLO(str(a.weights)).val(data=str(a.data),split=a.split,imgsz=a.imgsz,conf=a.conf,iou=a.iou,batch=a.batch,device=0,plots=True,save_json=False,project=str(a.output),name="ultralytics_val",exist_ok=False)
    overall={k:float(v) for k,v in metrics.results_dict.items()}
    (a.output/"overall_metrics.json").write_text(json.dumps(overall,indent=2)+"\n")
    rows=[]
    for i,c in enumerate(metrics.box.ap_class_index):
        bp,br,b50,b95=metrics.box.class_result(i); mp,mr,m50,m95=metrics.seg.class_result(i)
        rows.append({"class_id":int(c),"class_name":metrics.names[int(c)],"gt_images":int(metrics.nt_per_image[int(c)]),"gt_instances":int(metrics.nt_per_class[int(c)]),"box_precision":bp,"box_recall":br,"box_ap50":b50,"box_ap50_95":b95,"mask_precision":mp,"mask_recall":mr,"mask_ap50":m50,"mask_ap50_95":m95})
    with (a.output/"per_class_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    removed=[]
    for path in cache_paths:
        if not cache_existed[path] and path.exists(): path.unlink(); removed.append(str(path))
    print(json.dumps({"status":"PASS","split":a.split,"overall":overall,"classes":len(rows),"transient_caches_removed":removed},indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
