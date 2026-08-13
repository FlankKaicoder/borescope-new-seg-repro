#!/usr/bin/env python3
"""FastTrack-B train/val-only ROI classifier and Stage2 reproduction."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, random, time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO

ROOT = Path('/root/autodl-tmp/borescope-new-seg-repro')
DATA = Path('/root/autodl-tmp/borescope-new-seg-data/v1')
SPLIT = ROOT/'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv'
BASE = ROOT/'results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline/weights/best.pt'
HARD = ROOT/'results/fast_repro/exp05_hard_mining/treatment/ultralytics/treatment/weights/best.pt'
OUT = ROOT/'results/fast_repro/exp06_roi'
FIG = ROOT/'results/fast_repro/figures'
NAMES = ['Burn','Crack','Dent','Material missing','Tears','Tip curl','corrosion','background']
EXPECTED_SPLIT = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
EXPECTED_BASE = 'c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7'

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def split_rows():return [r for r in readcsv(SPLIT) if r['split'] in ('train','val')]
def paths(r):return DATA/'images'/r['split']/(r['stem']+r['image_suffix']),DATA/'labels'/r['split']/(r['stem']+'.txt')
def labels(lp,w,h):
 out=[]
 for line in lp.read_text().splitlines():
  z=line.split(); c=int(z[0]); xy=np.asarray(list(map(float,z[1:])),np.float32).reshape(-1,2); q=xy*np.array([w,h]);
  x1,y1=q.min(0);x2,y2=q.max(0);out.append((c,np.array([x1,y1,x2,y2],np.float32)))
 return out
def padbox(b,w,h,pad=1.2):
 x1,y1,x2,y2=b;cx=(x1+x2)/2;cy=(y1+y2)/2;bw=max(2,(x2-x1)*pad);bh=max(2,(y2-y1)*pad)
 return np.array([max(0,int(round(cx-bw/2))),max(0,int(round(cy-bh/2))),min(w,int(round(cx+bw/2))),min(h,int(round(cy+bh/2)))])
def biou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);i=max(0,x2-x1)*max(0,y2-y1);u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
 return i/u if u else 0
def savepatch(im,b,p):
 x1,y1,x2,y2=map(int,b);p.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(p),im[y1:y2,x1:x2])

def check_frozen():
 assert sha(BASE)==EXPECTED_BASE
 # Never enumerate/read TEST. Split hash is frozen evidence from Exp01.
 assert SPLIT.exists()

def build(args):
 check_frozen(); random.seed(42); np.random.seed(42)
 if (OUT/'patch_manifest.csv').exists(): raise FileExistsError('patch manifest already exists')
 rows=split_rows(); rec=[]; meta={}; positive_boxes=[]
 for r in rows:
  ip,lp=paths(r);im=cv2.imread(str(ip));h,w=im.shape[:2];gt=labels(lp,w,h);meta[(r['split'],r['stem'])]=(r,im,gt)
  for j,(c,b) in enumerate(gt):
   crop=padbox(b,w,h);pp=OUT/'patches'/r['split']/NAMES[c]/f"{r['stem']}__gt{j}.jpg";savepatch(im,crop,pp)
   rec.append(dict(patch_path=str(pp.relative_to(ROOT)),source_image=str(ip),split=r['split'],class_id=c,class_name=NAMES[c],source_type='gt_positive',bbox=json.dumps(b.tolist()),crop_coordinates=json.dumps(crop.tolist()),source_index=j))
   if r['split']=='train':positive_boxes.append((max(8,b[2]-b[0]),max(8,b[3]-b[1])))
 print('positive patches',len(rec),flush=True)
 model=YOLO(str(BASE))
 for split in ('train','val'):
  sr=[r for r in rows if r['split']==split]; ips=[paths(r)[0] for r in sr]
  results=model.predict(source=[str(x) for x in ips],imgsz=640,conf=.10,iou=.70,batch=32,device=0,retina_masks=True,verbose=False,stream=True)
  hard=[]
  for r,res in zip(sr,results,strict=True):
   _,im,gt=meta[(split,r['stem'])];h,w=im.shape[:2]
   if res.boxes is None:continue
   for j,b in enumerate(res.boxes.xyxy.cpu().numpy()):
    if max([biou(b,g[1]) for g in gt] or [0])<.5: hard.append((float(res.boxes.conf[j]),r,j,b))
  hard.sort(key=lambda x:x[0],reverse=True)
  pos=sum(x['split']==split and x['source_type']=='gt_positive' for x in rec); hard_n=min(len(hard),pos//2); random_n=pos-hard_n
  for conf,r,j,b in hard[:hard_n]:
   _,im,_=meta[(split,r['stem'])];h,w=im.shape[:2];crop=padbox(b,w,h);pp=OUT/'patches'/split/'background'/f"{r['stem']}__hard{j}.jpg";savepatch(im,crop,pp)
   rec.append(dict(patch_path=str(pp.relative_to(ROOT)),source_image=str(paths(r)[0]),split=split,class_id=7,class_name='background',source_type='hard_fp_background',bbox=json.dumps(list(map(float,b))),crop_coordinates=json.dumps(crop.tolist()),source_index=j))
  made=0; attempts=0
  while made<random_n and attempts<random_n*100:
   attempts+=1;r=random.choice(sr);_,im,gt=meta[(split,r['stem'])];h,w=im.shape[:2];bw,bh=random.choice(positive_boxes);bw=min(w,max(8,int(bw*random.uniform(.8,1.2))));bh=min(h,max(8,int(bh*random.uniform(.8,1.2))));x=random.randint(0,max(0,w-bw));y=random.randint(0,max(0,h-bh));b=np.array([x,y,x+bw,y+bh])
   if max([biou(b,g[1]) for g in gt] or [0])>=.05:continue
   pp=OUT/'patches'/split/'background'/f"{r['stem']}__random{made}.jpg";savepatch(im,b,pp)
   rec.append(dict(patch_path=str(pp.relative_to(ROOT)),source_image=str(paths(r)[0]),split=split,class_id=7,class_name='background',source_type='random_background',bbox=json.dumps(b.tolist()),crop_coordinates=json.dumps(b.tolist()),source_index=made));made+=1
  if made!=random_n:raise RuntimeError('random background construction failed')
 writecsv(OUT/'patch_manifest.csv',rec)
 tr={x['source_image'] for x in rec if x['split']=='train'};va={x['source_image'] for x in rec if x['split']=='val'}
 counts=Counter((x['split'],x['class_name'],x['source_type']) for x in rec)
 summary={'status':'PASS','test_accessed':False,'patch_count':len(rec),'train_count':sum(x['split']=='train' for x in rec),'val_count':sum(x['split']=='val' for x in rec),'source_image_leakage':len(tr&va),'counts':{'|'.join(k):v for k,v in counts.items()},'split_sha256':EXPECTED_SPLIT,'baseline_sha256':sha(BASE)}
 if summary['source_image_leakage']:raise RuntimeError('TRAIN_VAL_LEAKAGE')
 dump(OUT/'dataset_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
val_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class PatchDS(Dataset):
 def __init__(self,rows,tf,two=False):self.rows=rows;self.tf=tf;self.two=two
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];im=Image.open(ROOT/r['patch_path']).convert('RGB');y=int(r['class_id'])
  return ((self.tf(im),self.tf(im)),y) if self.two else (self.tf(im),y)
class SupModel(nn.Module):
 def __init__(self):
  super().__init__();m=resnet18(weights=ResNet18_Weights.DEFAULT);d=m.fc.in_features;m.fc=nn.Identity();self.encoder=m;self.fc=nn.Linear(d,8);self.proj=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,128))
 def forward(self,x):z=self.encoder(x);return self.fc(z),F.normalize(self.proj(z),dim=1),z
def supcon(z,y,temp=.07):
 sim=z@z.T/temp;sim=sim-sim.max(1,keepdim=True).values.detach();eye=torch.eye(len(y),device=y.device,dtype=torch.bool);pos=(y[:,None]==y[None,:])&~eye;den=torch.exp(sim).masked_fill(eye,0).sum(1);logp=sim-torch.log(den[:,None]+1e-12);n=pos.sum(1);valid=n>0
 return -(logp*pos).sum(1)[valid].div(n[valid]).mean()
def eval_model(model,loader,device,sup=False):
 model.eval();ys=[];ps=[]
 with torch.no_grad():
  for x,y in loader:
   x=x.to(device);logit=model(x)[0] if sup else model(x);ys.extend(y.tolist());ps.extend(logit.argmax(1).cpu().tolist())
 cm=confusion_matrix(ys,ps,labels=range(8));pr,re,f,_=precision_recall_fscore_support(ys,ps,labels=range(8),zero_division=0)
 return {'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0),'weighted_f1':f1_score(ys,ps,average='weighted',zero_division=0),'per_class':[{'class_id':i,'class_name':NAMES[i],'precision':pr[i],'recall':re[i],'f1':f[i],'support':int(cm[i].sum())} for i in range(8)],'confusion_matrix':cm.tolist()}
def plots_classifier(kind,hist,met):
 d=FIG/('exp06_roi_ce' if kind=='ce' else 'exp06_supcon');d.mkdir(parents=True,exist_ok=True)
 plt.figure();plt.plot([x['epoch'] for x in hist],[x['loss'] for x in hist],label='total');
 if kind=='supcon':plt.plot([x['epoch'] for x in hist],[x['ce_loss'] for x in hist],label='CE');plt.plot([x['epoch'] for x in hist],[x['supcon_loss'] for x in hist],label='SupCon')
 plt.legend();plt.tight_layout();plt.savefig(d/'training_loss.png',dpi=160);plt.close()
 plt.figure();plt.plot([x['epoch'] for x in hist],[x['accuracy'] for x in hist],label='accuracy');plt.plot([x['epoch'] for x in hist],[x['macro_f1'] for x in hist],label='macro F1');plt.legend();plt.tight_layout();plt.savefig(d/'accuracy_macro_f1.png',dpi=160);plt.close()
 cm=np.array(met['confusion_matrix']);
 for norm,name in [(False,'confusion_matrix.png'),(True,'confusion_matrix_normalized.png')]:
  a=cm/cm.sum(1,keepdims=True).clip(min=1) if norm else cm;plt.figure(figsize=(8,7));plt.imshow(a,cmap='Blues');plt.xticks(range(8),NAMES,rotation=45,ha='right');plt.yticks(range(8),NAMES);plt.colorbar();plt.tight_layout();plt.savefig(d/name,dpi=160);plt.close()
 plt.figure(figsize=(8,4));plt.bar(NAMES,[x['f1'] for x in met['per_class']]);plt.xticks(rotation=35,ha='right');plt.ylim(0,1);plt.tight_layout();plt.savefig(d/'per_class_f1.png',dpi=160);plt.close()
 if kind=='supcon':plt.figure();plt.plot([x['epoch'] for x in hist],[x['embedding_std'] for x in hist]);plt.tight_layout();plt.savefig(d/'embedding_std.png',dpi=160);plt.close()
def train(kind,args):
 torch.manual_seed(42);np.random.seed(42);random.seed(42);rows=readcsv(OUT/'patch_manifest.csv');tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='val'];cnt=Counter(int(r['class_id']) for r in tr);weights=[1/cnt[int(r['class_id'])] for r in tr];gen=torch.Generator().manual_seed(42);sampler=WeightedRandomSampler(weights,len(tr),replacement=True,generator=gen);two=kind=='supcon';td=PatchDS(tr,train_tf,two);vd=PatchDS(va,val_tf);tl=DataLoader(td,batch_size=args.batch,sampler=sampler,num_workers=8,pin_memory=True,drop_last=two);vl=DataLoader(vd,batch_size=args.batch,shuffle=False,num_workers=8,pin_memory=True);dev='cuda:0'
 if two:model=SupModel()
 else:model=resnet18(weights=ResNet18_Weights.DEFAULT);model.fc=nn.Linear(model.fc.in_features,8)
 model.to(dev);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);run=OUT/kind;run.mkdir(exist_ok=True);hist=[];best=-1
 for ep in range(1,args.epochs+1):
  model.train();tot=ce_sum=sc_sum=std_sum=0;n=0
  for x,y in tl:
   y=y.to(dev);opt.zero_grad(set_to_none=True)
   if two:
    x1,x2=x;x1=x1.to(dev);x2=x2.to(dev);l1,z1,e1=model(x1);l2,z2,e2=model(x2);ce=(F.cross_entropy(l1,y)+F.cross_entropy(l2,y))/2;sc=supcon(torch.cat([z1,z2]),torch.cat([y,y]));loss=ce+.1*sc;estd=torch.cat([e1,e2]).std(0).mean()
   else:x=x.to(dev);log=model(x);ce=F.cross_entropy(log,y);sc=torch.zeros((),device=dev);loss=ce;estd=torch.zeros((),device=dev)
   if not torch.isfinite(loss):
    raise RuntimeError('NON_FINITE_TRAIN_LOSS')
   loss.backward();opt.step();bs=len(y);tot+=loss.item()*bs;ce_sum+=ce.item()*bs;sc_sum+=sc.item()*bs;std_sum+=estd.item()*bs;n+=bs
  met=eval_model(model,vl,dev,two);row={'epoch':ep,'loss':tot/n,'ce_loss':ce_sum/n,'supcon_loss':sc_sum/n,'embedding_std':std_sum/n,'accuracy':met['accuracy'],'macro_f1':met['macro_f1']};hist.append(row);print(kind,row,flush=True)
  state={'epoch':ep,'model':model.state_dict(),'metrics':met,'kind':kind,'manifest_sha256':sha(OUT/'patch_manifest.csv'),'batch':args.batch,'seed':42}
  torch.save(state,run/'last.pt')
  if met['macro_f1']>best:best=met['macro_f1'];torch.save(state,run/'best.pt')
 beststate=torch.load(run/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(beststate['model']);met=eval_model(model,vl,dev,two);writecsv(run/'history.csv',hist);writecsv(run/'per_class_metrics.csv',met['per_class']);dump(run/'metrics.json',{**met,'status':'PASS','test_accessed':False,'epochs':args.epochs,'batch':args.batch,'checkpoint_sha256':sha(run/'best.pt'),'manifest_sha256':sha(OUT/'patch_manifest.csv'),'loss_finite':all(math.isfinite(x['loss']) for x in hist),'embedding_std_finite_nonzero':(min(x['embedding_std'] for x in hist)>1e-6 if two else 'N/A')});plots_classifier(kind,hist,met)

def load_cls(kind,dev='cuda:0'):
 st=torch.load(OUT/kind/'best.pt',map_location=dev,weights_only=False)
 if kind=='supcon':m=SupModel()
 else:m=resnet18(weights=None);m.fc=nn.Linear(m.fc.in_features,8)
 m.load_state_dict(st['model']);m.to(dev).eval();return m
def crop_tensor(im,b):
 h,w=im.shape[:2];b=padbox(b,w,h);rgb=cv2.cvtColor(im[b[1]:b[3],b[0]:b[2]],cv2.COLOR_BGR2RGB);return val_tf(Image.fromarray(rgb))
def mask_iou(a,b):
 u=np.logical_or(a,b).sum();return np.logical_and(a,b).sum()/u if u else 0
def gt_masks(lp,w,h):
 out=[]
 for line in lp.read_text().splitlines():
  z=line.split();c=int(z[0]);xy=np.asarray(list(map(float,z[1:])),np.float32).reshape(-1,2);pts=np.rint(xy*np.array([w-1,h-1])).astype(np.int32);m=np.zeros((h,w),np.uint8);cv2.fillPoly(m,[pts],1);out.append({'class_id':c,'mask':m.astype(bool)})
 return out
def greedy(g,p,classaware=True):
 cand=sorted(((mask_iou(x['mask'],y['mask']),i,j) for i,x in enumerate(g) for j,y in enumerate(p) if (not classaware or x['class_id']==y['class_id'])),reverse=True);ug=set();up=set();m={}
 for s,i,j in cand:
  if s>=.5 and i not in ug and j not in up:ug.add(i);up.add(j);m[i]=j
 return m
def metrics(details,classaware=True):
 gt=pred=tp=wrong=0
 for d in details:
  g=d['gt'];p=d['pred'];m=greedy(g,p,classaware);gt+=len(g);pred+=len(p);tp+=len(m)
  if classaware:wrong+=sum(1 for i,j in greedy(g,p,False).items() if g[i]['class_id']!=p[j]['class_id'])
 fp=pred-tp;fn=gt-tp;pr=tp/(tp+fp) if tp+fp else 0;re=tp/gt;f=2*pr*re/(pr+re) if pr+re else 0
 return {'TP':tp,'FP':fp,'FN':fn,'Precision':pr,'Recall':re,'F1':f,'WrongClass':wrong,'Mask_AP50':'N/A','Mask_AP50_95':'N/A'}
def draw_stage2(path,d,title):
 im=d['im'].copy()
 for g in d['gt']:
  cs,_=cv2.findContours(g['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,(0,220,0),2)
 for p in d['pred']:
  x1,y1,x2,y2=map(int,p['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,230),2);cv2.putText(im,f"{NAMES[p['class_id']]} {p['confidence']:.2f}",(x1,max(18,y1)),0,.45,(0,0,230),1)
 cv2.rectangle(im,(0,0),(im.shape[1],30),(20,20,20),-1);cv2.putText(im,title,(5,20),0,.48,(255,255,255),1);cv2.imwrite(str(path),im)
def stage2(args):
 check_frozen();kind=args.classifier;cls=load_cls(kind);mrs=[r for r in split_rows() if r['split']=='val'];ips=[paths(r)[0] for r in mrs];yolo=YOLO(str(BASE));t0=time.perf_counter();res=list(yolo.predict(source=[str(x) for x in ips],imgsz=640,conf=.05,iou=.70,batch=32,device=0,retina_masks=True,verbose=False,stream=True));torch.cuda.synchronize();stage1_ms=(time.perf_counter()-t0)*1000/len(mrs);cache=[];cls_times=[]
 for r,rr in zip(mrs,res,strict=True):
  ip,lp=paths(r);im=cv2.imread(str(ip));h,w=im.shape[:2];g=gt_masks(lp,w,h);p=[]
  if rr.boxes is not None and len(rr.boxes):
   boxes=rr.boxes.xyxy.cpu().numpy();cs=rr.boxes.cls.cpu().numpy().astype(int);co=rr.boxes.conf.cpu().numpy();masks=rr.masks.data.cpu().numpy() if rr.masks is not None else []
   xs=torch.stack([crop_tensor(im,b) for b in boxes]).cuda();torch.cuda.synchronize();q=time.perf_counter()
   with torch.no_grad():log=cls(xs)[0] if kind=='supcon' else cls(xs);prob=log.softmax(1).cpu().numpy()
   torch.cuda.synchronize();cls_times.append((time.perf_counter()-q)*1000/len(boxes))
   for i,b in enumerate(boxes):
    mm=masks[i];mm=cv2.resize(mm,(w,h),interpolation=cv2.INTER_NEAREST)>=.5 if mm.shape!=(h,w) else mm>=.5;p.append({'class_id':int(cs[i]),'confidence':float(co[i]),'box':b,'mask':mm,'prob':prob[i]})
  cache.append({'stem':r['stem'],'im':im,'gt':g,'pred':p})
 def filt(s1,s2,mode):
  out=[]
  for d in cache:
   pp=[]
   for p in d['pred']:
    if p['confidence']<s1:continue
    q=p.copy();predc=int(np.argmax(p['prob']));pdef=1-p['prob'][7]
    if mode=='A':
     if pdef<s2:continue
    else:
     if predc==7 or p['prob'][predc]<s2:continue
     q['class_id']=predc
    pp.append(q)
   out.append({'stem':d['stem'],'im':d['im'],'gt':d['gt'],'pred':pp})
  return out
 grids=[]
 for s1 in (.05,.10,.15):
  grids.append({'mode':'stage1_only','stage1_conf':s1,'stage2_threshold':'N/A',**metrics(filt(s1,0,'A'))})
  for mode in ('A','B'):
   for s2 in (.3,.5,.7):grids.append({'mode':mode,'stage1_conf':s1,'stage2_threshold':s2,**metrics(filt(s1,s2,mode))})
 base25={'mode':'baseline','stage1_conf':.25,'stage2_threshold':'N/A',**metrics(filt(.25,0,'A'))};grids.append(base25);writecsv(ROOT/'results/fast_repro/exp07_stage2/grid.csv',grids)
 best={m:max((x for x in grids if x['mode']==m),key=lambda x:(x['F1'],x['stage1_conf'],x['stage2_threshold'])) for m in ('A','B')}
 recover=readcsv(ROOT/'results/fast_repro/exp03_low_conf/fn_recovery.csv');rec={(x['stem'],int(x['gt_index'])) for x in recover if x['category']=='LOW_CONF_RECOVERABLE'}
 analyses={}
 for mode in ('A','B'):
  b=best[mode];before=filt(b['stage1_conf'],0,'A');after=filt(b['stage1_conf'],b['stage2_threshold'],mode);kept=0;eligible=0
  corrected=harmful=0
  for d0,d1 in zip(before,after):
   mb=greedy(d0['gt'],d0['pred'],True);ma=greedy(d1['gt'],d1['pred'],True);sb=greedy(d0['gt'],d0['pred'],False);sa=greedy(d1['gt'],d1['pred'],False)
   for gi in range(len(d1['gt'])):
    if (d1['stem'],gi) in rec:
     eligible+=gi in mb;kept+=gi in ma
    if gi in sb and d0['gt'][gi]['class_id']!=d0['pred'][sb[gi]]['class_id'] and gi in ma:corrected+=1
    if gi in mb and gi in sa and d1['gt'][gi]['class_id']!=d1['pred'][sa[gi]]['class_id']:harmful+=1
  bm=metrics(before);am=metrics(after);spatial_before=bm['WrongClass'];spatial_after=am['WrongClass']
  analyses[mode]={'recoverable_total':91,'recoverable_candidate_at_selected_conf':eligible,'recoverable_stage1_unavailable':91-eligible,'recoverable_retained':kept,'recoverable_filtered_by_stage2':eligible-kept,'FP_before':bm['FP'],'FP_removed':bm['FP']-am['FP'],'FP_retained':am['FP'],'FP_removal_rate':(bm['FP']-am['FP'])/bm['FP'] if bm['FP'] else 0,'wrong_class_before':spatial_before,'wrong_class_after':spatial_after,'correct_reclassification':corrected,'harmful_reclassification':harmful,'net_correction':corrected-harmful}
 # Plots
 fd=FIG/'exp07_stage2';fd.mkdir(parents=True,exist_ok=True)
 st=[x for x in grids if x['mode']=='stage1_only'];plt.figure();plt.plot([x['stage1_conf'] for x in st],[x['Recall'] for x in st],'o-',label='Recall');plt.plot([x['stage1_conf'] for x in st],[x['FP'] for x in st],'o-',label='FP');plt.legend();plt.tight_layout();plt.savefig(fd/'exp07_stage1_threshold_tradeoff.png',dpi=160);plt.close()
 for mode in ('A','B'):
  z=np.array([[next(x['F1'] for x in grids if x['mode']==mode and x['stage1_conf']==a and x['stage2_threshold']==b) for b in (.3,.5,.7)] for a in (.05,.10,.15)]);plt.figure();plt.imshow(z,vmin=0,vmax=max(.6,z.max()),cmap='viridis');plt.xticks(range(3),(.3,.5,.7));plt.yticks(range(3),(.05,.10,.15));plt.colorbar(label='Mask F1');plt.xlabel('Stage2 threshold');plt.ylabel('Stage1 conf');plt.tight_layout();plt.savefig(fd/f'exp07_stage2_mode{mode}_grid.png',dpi=160);plt.close()
 comp=[base25,max(st,key=lambda x:x['F1']),best['A'],best['B']];plt.figure(figsize=(8,4));plt.bar(['YOLO .25','Low-conf','Mode A','Mode B'],[x['F1'] for x in comp]);plt.ylabel('Mask F1');plt.tight_layout();plt.savefig(fd/'exp07_stage2_main_comparison.png',dpi=160);plt.close()
 stage1_samples=[float(x.speed.get('inference',stage1_ms)) for x in res];candidate_counts=[len(x['pred']) for x in cache];e2e=[s+n*t for s,n,t in zip(stage1_samples,candidate_counts,cls_times)]
 latency={'stage1_mean_ms_image':float(np.mean(stage1_samples)),'stage1_median_ms_image':float(np.median(stage1_samples)),'classifier_mean_ms_candidate':float(np.mean(cls_times)),'classifier_median_ms_candidate':float(np.median(cls_times)),'end_to_end_mean_ms_image':float(np.mean(e2e)),'end_to_end_median_ms_image':float(np.median(e2e))}
 # Small qualitative audit set from the selected configuration (up to 5 per category).
 qdir=fd/'qualitative';qdir.mkdir(exist_ok=True);a0=filt(best['A']['stage1_conf'],0,'A');aa=filt(best['A']['stage1_conf'],best['A']['stage2_threshold'],'A');bb=filt(best['B']['stage1_conf'],best['B']['stage2_threshold'],'B');qc=Counter()
 for d0,da,db in zip(a0,aa,bb):
  m0=metrics([d0]);ma=metrics([da]);mb=metrics([db]);cats=[]
  if m0['FP']>ma['FP']:cats.append('success_filter_fp')
  rec_here={gi for stem,gi in rec if stem==d0['stem']};matched_a=set(greedy(da['gt'],da['pred'],True));
  if rec_here&matched_a:cats.append('success_keep_low_conf_tp')
  if m0['WrongClass']>mb['WrongClass']:cats.append('success_reclassify')
  if m0['TP']>ma['TP']:cats.append('error_filter_tp')
  if m0['TP']>mb['TP'] and mb['WrongClass']>=m0['WrongClass']:cats.append('error_reclassify')
  for cat in cats:
   if qc[cat]<5:draw_stage2(qdir/f"{cat}__{d0['stem']}.jpg",db if 'reclass' in cat else da,f"{cat} {d0['stem']}");qc[cat]+=1
 conclusion='POSITIVE_CANDIDATE' if max(best['A']['F1'],best['B']['F1'])>base25['F1']+.02 else ('TRADEOFF_ONLY' if max(best['A']['F1'],best['B']['F1'])>=base25['F1'] else 'NEGATIVE')
 dump(ROOT/'results/fast_repro/exp07_stage2/summary.json',{'status':'PASS','test_accessed':False,'classifier':kind,'baseline':base25,'best':best,'analysis':analyses,'latency':latency,'qualitative_counts':dict(qc),'AP_note':'N/A: current post-processing fixed-point evaluator has no reliable arbitrary-prediction COCO AP adapter; deferred by Fast Repro rule.','conclusion':conclusion})
 print(json.dumps({'baseline':base25,'best':best,'analysis':analyses,'latency':latency,'conclusion':conclusion},indent=2),flush=True)

def compare(args):
 ce=json.loads((OUT/'ce/metrics.json').read_text());su=json.loads((OUT/'supcon/metrics.json').read_text());d=FIG/'exp06_supcon';labels=['accuracy','macro_f1','weighted_f1'];cv=[ce[x] for x in labels];sv=[su[x] for x in labels];x=np.arange(3);plt.figure();plt.bar(x-.18,cv,.36,label='CE');plt.bar(x+.18,sv,.36,label='SupCon');plt.xticks(x,labels);plt.legend();plt.tight_layout();plt.savefig(d/'exp06_ce_vs_supcon.png',dpi=160);plt.close();delta=su['macro_f1']-ce['macro_f1'];hard=['Burn','Crack','corrosion','background'];dam={n:next(x['recall'] for x in su['per_class'] if x['class_name']==n)-next(x['recall'] for x in ce['per_class'] if x['class_name']==n) for n in hard};con='SUPCON_POSITIVE' if delta>0 and min(dam.values())>-.1 else ('NO_CLEAR_GAIN' if abs(delta)<.01 else 'NEGATIVE');dump(OUT/'ce_vs_supcon.json',{'macro_f1_delta':delta,'difficult_recall_delta':dam,'conclusion':con,'stage2_classifier':'supcon' if con=='SUPCON_POSITIVE' else 'ce'});print(con)

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True);sp.add_parser('build');
 for n in ('train-ce','train-supcon'):
  q=sp.add_parser(n);q.add_argument('--epochs',type=int,default=50);q.add_argument('--batch',type=int,default=64)
 sp.add_parser('compare');q=sp.add_parser('stage2');q.add_argument('--classifier',choices=['ce','supcon'],default='ce')
 a=p.parse_args();{'build':build,'train-ce':lambda x:train('ce',x),'train-supcon':lambda x:train('supcon',x),'compare':compare,'stage2':stage2}[a.cmd](a)
if __name__=='__main__':main()
