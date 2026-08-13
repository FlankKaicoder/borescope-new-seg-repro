#!/usr/bin/env python3
"""Score TRAIN images only and freeze the top-30% hard pool."""
from __future__ import annotations
import argparse,math
from collections import Counter
from pathlib import Path
import cv2
from ultralytics import YOLO
from fasttrack_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--batch",type=int,default=32);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);rs=[r for r in rows(a.manifest) if r["split"]=="train"];paths=[image_label(a.data_root,r)[0] for r in rs];model=YOLO(str(a.weights));results=model.predict(source=[str(x) for x in paths],imgsz=640,conf=.25,iou=.70,batch=a.batch,device=0,retina_masks=True,verbose=False,stream=True);out=[]
 for row,res in zip(rs,results,strict=True):
  ip,lp=image_label(a.data_root,row);im=cv2.imread(str(ip));h,w=im.shape[:2];g=label_instances(lp,w,h);pr=predictions(res,h,w);m=class_matches(g,pr);matched=set(m.values());fn=wrong=loc=0
  for gi,gt in enumerate(g):
   if gi in m:continue
   wr=[x for x in pr if x["class_id"]!=gt["class_id"] and (miou(gt["mask"],x["mask"])>=.5 or biou(gt["box"],x["box"])>=.5)]
   same=[x for x in pr if x["class_id"]==gt["class_id"]]
   if wr:wrong+=1
   elif same:loc+=1
   else:fn+=1
  fp=len(pr)-len(m);score=3*fn+2*wrong+2*loc+fp
  out.append({"image_stem":row["stem"],"hard_score":score,"FN":fn,"wrong_class":wrong,"localization_failure":loc,"FP":fp,"labels_present":row["labels_present"],"is_hard":False})
 n=math.ceil(.30*len(out));chosen=sorted(range(len(out)),key=lambda i:(out[i]["hard_score"],out[i]["image_stem"]),reverse=True)[:n]
 for i in chosen:out[i]["is_hard"]=True
 write_csv(a.output/"hard_pool.csv",out);dist=Counter();
 for x in out:
  if x["is_hard"]:
   for c in x["labels_present"].split("|"):dist[c]+=1
 dump(a.output/"summary.json",{"status":"PASS","derivation_split":"train","test_accessed":False,"train_images":len(out),"hard_images":n,"hard_fraction":n/len(out),"weights_sha256":sha256(a.weights),"hard_score":"3*FN + 2*wrong_class + 2*localization_failure + FP","class_presence_in_hard_pool":dist,"score_min":min(x["hard_score"] for x in out),"score_max":max(x["hard_score"] for x in out)})
if __name__=="__main__":main()
