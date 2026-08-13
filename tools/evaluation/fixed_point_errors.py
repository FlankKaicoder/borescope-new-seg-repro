#!/usr/bin/env python3
"""Compute class-aware mask-IoU fixed-point TP/FP/FN on val only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2
from ultralytics import YOLO
from fasttrack_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rs=[r for r in rows(a.manifest) if r["split"]=="val"];paths=[image_label(a.data_root,r)[0] for r in rs];pred=YOLO(str(a.weights)).predict(source=[str(x) for x in paths],imgsz=640,conf=.25,iou=.70,batch=32,device=0,retina_masks=True,verbose=False,stream=True);gt=tp=pd=0
 for row,res in zip(rs,pred,strict=True):
  ip,lp=image_label(a.data_root,row);im=cv2.imread(str(ip));h,w=im.shape[:2];g=label_instances(lp,w,h);pr=predictions(res,h,w);m=class_matches(g,pr);gt+=len(g);pd+=len(pr);tp+=len(m)
 a.output.write_text(json.dumps({"status":"PASS","split":"val","test_accessed":False,"confidence":.25,"TP":tp,"FP":pd-tp,"FN":gt-tp,"GT":gt,"prediction_count":pd},indent=2)+"\n")
if __name__=="__main__":main()
