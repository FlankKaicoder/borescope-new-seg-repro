#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import cv2
from ultralytics import YOLO
sys.path.insert(0,str(Path(__file__).parent));from fasttrack_common import rows,image_label,label_instances,predictions,class_matches,miou,biou,sha256,write_csv
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True); rs=[r for r in rows(a.manifest) if r['split']=='train']; paths=[image_label(a.data_root,r)[0] for r in rs]; pred=YOLO(str(a.weights)).predict(source=[str(x) for x in paths],imgsz=640,conf=.25,iou=.70,batch=32,device=0,retina_masks=True,verbose=False,stream=True); out=[]
 for row,res in zip(rs,pred,strict=True):
  ip,lp=image_label(a.data_root,row); im=cv2.imread(str(ip)); h,w=im.shape[:2]; g=label_instances(lp,w,h); pr=predictions(res,h,w); m=class_matches(g,pr); fn=wrong=loc=0
  for gi,gt in enumerate(g):
   if gi in m:continue
   wr=[x for x in pr if x['class_id']!=gt['class_id'] and (miou(gt['mask'],x['mask'])>=.5 or biou(gt['box'],x['box'])>=.5)]; same=[x for x in pr if x['class_id']==gt['class_id']]
   if wr:wrong+=1
   elif same:loc+=1
   else:fn+=1
  fp=len(pr)-len(m);out.append({'seed':a.seed,'image_stem':row['stem'],'hard_score':3*fn+2*wrong+2*loc+fp,'FN':fn,'wrong_class':wrong,'localization_failure':loc,'FP':fp,'is_hard':False})
 n=math.ceil(.30*len(out)); chosen={x['image_stem'] for x in sorted(out,key=lambda x:(-x['hard_score'],x['image_stem']))[:n]}
 for x in out:x['is_hard']=x['image_stem'] in chosen
 write_csv(a.output/'hard_pool.csv',out);(a.output/'summary.json').write_text(json.dumps({'status':'PASS','seed':a.seed,'derivation_split':'train','test_accessed':False,'train_images':len(out),'hard_images':n,'hard_fraction':n/len(out),'weights_sha256':sha256(a.weights),'hard_score':'3*FN + 2*wrong_class + 2*localization_failure + FP','tie_break':'hard_score descending; image_stem ascending'},indent=2)+'\n')
if __name__=='__main__':main()
