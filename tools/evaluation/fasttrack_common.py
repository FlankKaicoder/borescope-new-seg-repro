#!/usr/bin/env python3
"""Shared train/val-only helpers for FastTrack-A."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import cv2, numpy as np, yaml

NAMES={0:"Burn",1:"Crack",2:"Dent",3:"Material missing",4:"Tears",5:"Tip curl",6:"corrosion"}

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  while b:=f.read(1048576): h.update(b)
 return h.hexdigest()
def rows(path:Path):
 with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write_csv(path:Path,data:list[dict]):
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def dump(path:Path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def image_label(root:Path,row:dict):
 return root/"images"/row["split"]/(row["stem"]+row["image_suffix"]),root/"labels"/row["split"]/(row["stem"]+".txt")
def label_instances(path:Path,w:int,h:int):
 out=[]
 for raw in path.read_text().splitlines():
  v=raw.split();c=int(v[0]);xy=np.array([float(x) for x in v[1:]],np.float32).reshape(-1,2)
  pts=np.rint(xy*np.array([w-1,h-1])).astype(np.int32);m=np.zeros((h,w),np.uint8);cv2.fillPoly(m,[pts],1)
  ys,xs=np.nonzero(m);box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],np.float32) if len(xs) else np.zeros(4,np.float32)
  out.append({"class_id":c,"mask":m.astype(bool),"box":box})
 return out
def predictions(result,h:int,w:int):
 if result.boxes is None or not len(result.boxes):return []
 boxes=result.boxes.xyxy.cpu().numpy();cs=result.boxes.cls.cpu().numpy().astype(int);conf=result.boxes.conf.cpu().numpy()
 masks=[] if result.masks is not None else [np.zeros((h,w),bool) for _ in cs]
 if result.masks is not None:
  for m in result.masks.data.cpu().numpy():
   if m.shape!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
   masks.append(m>=.5)
 return [{"class_id":int(c),"confidence":float(conf[i]),"box":boxes[i],"mask":masks[i]} for i,c in enumerate(cs)]
def miou(a,b):
 u=np.logical_or(a,b).sum();return float(np.logical_and(a,b).sum()/u) if u else 0.
def biou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1)
 u=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
 return float(inter/u) if u else 0.
def class_matches(gts,preds,threshold=.5):
 cand=sorted(((miou(g["mask"],p["mask"]),gi,pi) for gi,g in enumerate(gts) for pi,p in enumerate(preds) if g["class_id"]==p["class_id"]),reverse=True)
 ug=set();up=set();matches={}
 for score,gi,pi in cand:
  if score>=threshold and gi not in ug and pi not in up:ug.add(gi);up.add(pi);matches[gi]=pi
 return matches
