#!/usr/bin/env python3
"""Direct VAL threshold sweep and baseline-FN recovery classification."""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fasttrack_common import *

THRESHOLDS=[.001,.005,.010,.025,.050,.100,.150,.200,.250,.300,.400,.500]

def infer(model,paths,conf,batch):return list(model.predict(source=[str(x) for x in paths],imgsz=640,conf=conf,iou=.70,batch=batch,device=0,retina_masks=True,verbose=False,stream=True))
def evaluate(data_root,manifest_rows,results):
 total=Counter();per=Counter();details=[]
 for row,res in zip(manifest_rows,results,strict=True):
  ip,lp=image_label(data_root,row);im=cv2.imread(str(ip));h,w=im.shape[:2];g=label_instances(lp,w,h);p=predictions(res,h,w);m=class_matches(g,p)
  total.update(GT=len(g),TP=len(m),PRED=len(p));
  for x in g:per[(x["class_id"],"GT")]+=1
  for gi in m:per[(g[gi]["class_id"],"TP")]+=1
  details.append((row,im,g,p,m))
 total["FN"]=total["GT"]-total["TP"];total["FP"]=total["PRED"]-total["TP"]
 P=total["TP"]/(total["TP"]+total["FP"]) if total["TP"]+total["FP"] else 0;R=total["TP"]/total["GT"];F=2*P*R/(P+R) if P+R else 0
 return {"TP":total["TP"],"FP":total["FP"],"FN":total["FN"],"Precision":P,"Recall":R,"F1":F,"prediction_count":total["PRED"],**{f"recall_{NAMES[c]}":per[(c,"TP")]/per[(c,"GT")] if per[(c,"GT")] else 0 for c in NAMES}},details
def draw_case(path,im,stem,cat,gt,preds):
 x=im.copy();cont,_=cv2.findContours(gt["mask"].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(x,cont,-1,(0,220,0),2)
 for p in preds:
  cont,_=cv2.findContours(p["mask"].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(x,cont,-1,(0,0,230),2)
  xx,yy=map(int,p["box"][:2]);cv2.putText(x,f"{NAMES[p['class_id']]} {p['confidence']:.3f}",(xx,max(18,yy)),0,.45,(0,0,230),1)
 cv2.rectangle(x,(0,0),(x.shape[1],28),(20,20,20),-1);cv2.putText(x,f"{stem} {cat} GT={NAMES[gt['class_id']]}",(5,19),0,.48,(255,255,255),1);cv2.imwrite(str(path),x)
def plots(out,table,b=.25,fbest=None,r95=None):
 t=np.array([r["confidence"] for r in table]);marks=[(b,"conf=.25"),(fbest,"F1-best"),(r95,"recall95")]
 def mark():
  for x,l in marks:plt.axvline(x,ls="--",lw=1,label=l)
 plt.figure();plt.plot(t,[r["Precision"] for r in table],"o-",label="Precision");plt.plot(t,[r["Recall"] for r in table],"o-",label="Recall");plt.xscale("log");mark();plt.legend();plt.grid();plt.tight_layout();plt.savefig(out/"exp03_threshold_precision_recall.png",dpi=160);plt.close()
 plt.figure();plt.plot(t,[r["F1"] for r in table],"o-");plt.xscale("log");mark();plt.legend();plt.grid();plt.tight_layout();plt.savefig(out/"exp03_threshold_f1.png",dpi=160);plt.close()
 plt.figure();
 for k in ("TP","FP","FN"):plt.plot(t,[r[k] for r in table],"o-",label=k)
 plt.xscale("log");mark();plt.legend();plt.grid();plt.tight_layout();plt.savefig(out/"exp03_threshold_tp_fp_fn.png",dpi=160);plt.close()
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--batch",type=int,default=32);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);vis=a.output/"qualitative";vis.mkdir();mrs=[r for r in rows(a.manifest) if r["split"]=="val"];paths=[image_label(a.data_root,r)[0] for r in mrs];model=YOLO(str(a.weights));table=[];all_details={}
 for conf in THRESHOLDS:
  met,det=evaluate(a.data_root,mrs,infer(model,paths,conf,a.batch));table.append({"confidence":conf,**met});all_details[conf]=det
 fixed=next(x for x in table if x["confidence"]==.25)
 if (fixed["TP"],fixed["FP"],fixed["FN"])!=(123,99,173):raise RuntimeError(f"FIXED_POINT_HARD_GATE {fixed}")
 maxf=max(x["F1"] for x in table);fbest=max(x["confidence"] for x in table if abs(x["F1"]-maxf)<1e-12);maxr=max(x["Recall"] for x in table);r95=max(x["confidence"] for x in table if x["Recall"]>=.95*maxr)
 write_csv(a.output/"threshold_sweep.csv",table);plots(a.output,table,.25,fbest,r95)
 low=all_details[.001];base=all_details[.25];rec=[];visual_counts=Counter()
 for (row,im,g,p,m),(row2,_,g2,p2,m2) in zip(low,base,strict=True):
  for gi,gt in enumerate(g):
   if gi in m2:continue
   same=[(pi,miou(gt["mask"],x["mask"]),biou(gt["box"],x["box"])) for pi,x in enumerate(p) if x["class_id"]==gt["class_id"]]
   wrong=[(pi,miou(gt["mask"],x["mask"]),biou(gt["box"],x["box"])) for pi,x in enumerate(p) if x["class_id"]!=gt["class_id"]]
   recover=[x for x in same if x[1]>=.5 and p[x[0]]["confidence"]<.25]
   spatialwrong=[x for x in wrong if x[1]>=.5 or x[2]>=.5]
   if recover:cat="LOW_CONF_RECOVERABLE";chosen=max(recover,key=lambda x:x[1])
   elif spatialwrong:cat="WRONG_CLASS";chosen=max(spatialwrong,key=lambda x:max(x[1],x[2]))
   elif same:cat="LOCALIZATION_FAILURE";chosen=max(same,key=lambda x:x[1])
   else:cat="NO_RESPONSE";chosen=None
   rec.append({"stem":row["stem"],"gt_index":gi,"class_id":gt["class_id"],"class_name":NAMES[gt["class_id"]],"category":cat,"candidate_class":"" if chosen is None else NAMES[p[chosen[0]]["class_id"]],"confidence":"" if chosen is None else p[chosen[0]]["confidence"],"mask_iou":"" if chosen is None else chosen[1],"box_iou":"" if chosen is None else chosen[2]})
   if gt["class_id"] in (0,1,6) and visual_counts[cat]<8:
    draw_case(vis/f"{cat}__{row['stem']}__{gi}.jpg",im,row["stem"],cat,gt,[] if chosen is None else [p[chosen[0]]]);visual_counts[cat]+=1
 write_csv(a.output/"fn_recovery.csv",rec);counts=Counter(x["category"] for x in rec);byclass={NAMES[c]:dict(Counter(x["category"] for x in rec if x["class_id"]==c)) for c in (1,0,6)}
 plt.figure(figsize=(7,4));ks=["LOW_CONF_RECOVERABLE","WRONG_CLASS","LOCALIZATION_FAILURE","NO_RESPONSE"];plt.bar(ks,[counts[k] for k in ks]);plt.xticks(rotation=18);plt.tight_layout();plt.savefig(a.output/"exp03_fn_recovery_summary.png",dpi=160);plt.close()
 conclusion="POSITIVE" if counts["LOW_CONF_RECOVERABLE"]>=.3*len(rec) else "LIMITED" if counts["LOW_CONF_RECOVERABLE"]>=.1*len(rec) else "NEGATIVE"
 dump(a.output/"summary.json",{"status":"PASS","test_accessed":False,"weights_sha256":sha256(a.weights),"fixed_point":fixed,"threshold_f1_best":fbest,"threshold_recall95":r95,"max_recall":maxr,"recall95_metrics":next(x for x in table if x["confidence"]==r95),"fn_total":len(rec),"fn_recovery_counts":counts,"recovery_rate":counts["LOW_CONF_RECOVERABLE"]/len(rec),"difficult_class_recovery":byclass,"conclusion":conclusion})
if __name__=="__main__":main()
