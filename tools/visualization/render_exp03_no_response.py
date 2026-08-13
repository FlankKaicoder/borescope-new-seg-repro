#!/usr/bin/env python3
"""Render the sole Exp03 NO_RESPONSE GT instance without model re-evaluation."""
from pathlib import Path
import cv2,numpy as np,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"evaluation"))
from fasttrack_common import label_instances,NAMES
root=Path("/root/autodl-tmp/borescope-new-seg-data/v1");stem="440";im=cv2.imread(str(root/"images/val/440.jpg"));g=label_instances(root/"labels/val/440.txt",im.shape[1],im.shape[0])[0]
cont,_=cv2.findContours(g["mask"].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cont,-1,(0,220,0),2);cv2.rectangle(im,(0,0),(im.shape[1],28),(20,20,20),-1);cv2.putText(im,"440 NO_RESPONSE GT=Tip curl",(5,19),0,.48,(255,255,255),1)
out=Path("results/fast_repro/exp03_low_conf/qualitative/NO_RESPONSE__440__0.jpg");assert cv2.imwrite(str(out),im)
