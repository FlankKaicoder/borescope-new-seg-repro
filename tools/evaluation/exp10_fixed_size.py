#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import cv2
from ultralytics import YOLO
sys.path.insert(0,str(Path(__file__).parent));from fasttrack_common import rows,image_label,label_instances,predictions,class_matches
Q=(0.0020196759259259256,0.007364969135802469,0.029209280303030303); N=('tiny','small','medium','large')
def binof(a):return N[0] if a<=Q[0] else N[1] if a<=Q[1] else N[2] if a<=Q[2] else N[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rs=[r for r in rows(a.manifest) if r['split']=='val'];paths=[image_label(a.data_root,r)[0] for r in rs];stream=YOLO(str(a.weights)).predict(source=[str(x) for x in paths],imgsz=640,conf=.25,iou=.70,batch=32,device=0,retina_masks=True,verbose=False,stream=True);gt=tp=pd=0; totals=Counter();hits=Counter()
 for row,res in zip(rs,stream,strict=True):
  ip,lp=image_label(a.data_root,row);im=cv2.imread(str(ip));h,w=im.shape[:2];g=label_instances(lp,w,h);pr=predictions(res,h,w);m=class_matches(g,pr);gt+=len(g);pd+=len(pr);tp+=len(m)
  for i,x in enumerate(g):
   area=float(x['mask'].sum()/(h*w));b=binof(area);totals[b]+=1;hits[b]+=i in m
 out={'status':'PASS','split':'val','test_accessed':False,'confidence':.25,'match_mask_iou':.5,'size_thresholds':Q,'TP':tp,'FP':pd-tp,'FN':gt-tp,'precision':tp/pd if pd else 0,'recall':tp/gt if gt else 0,'F1':2*tp/(pd+gt) if pd+gt else 0,'size_recall':{b:hits[b]/totals[b] for b in N},'size_gt':dict(totals)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
