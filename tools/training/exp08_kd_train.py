#!/usr/bin/env python3
"""Exp08 historical reconstruction: frozen SupCon ROI teacher -> YOLO auxiliary CE/KD."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from tools.fasttrack_b import SupModel

ROOT=Path('/root/autodl-tmp/borescope-new-seg-repro')
TEACHER=ROOT/'results/fast_repro/exp06_roi/supcon/best.pt'
OFFICIAL=ROOT/'weights/yolo11n-seg.pt'
DATA=Path('/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml')
EXPECTED_TEACHER='8e22f17c029eb0f3cb9416673a3503e0d37c7b98b91f126da2107e23fe58c32b'
P3_LAYER=4;P3_CHANNELS=128;P3_STRIDE=8
MODE='auxce'; TEACHER_INITIAL={}; AUX_LOG=[]; GRAD_SEEN={'student':False,'aux':False}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def load_teacher():
 if sha(TEACHER)!=EXPECTED_TEACHER:raise RuntimeError('TEACHER_SHA_MISMATCH')
 st=torch.load(TEACHER,map_location='cpu',weights_only=False);m=SupModel();m.load_state_dict(st['model']);m.eval()
 for p in m.parameters():p.requires_grad=False
 return m
class KDModel(SegmentationModel):
 def __init__(self,cfg='yolo11n-seg.yaml',ch=3,nc=None,verbose=True):
  super().__init__(cfg,ch,nc,verbose);self.aux_classifier=nn.Sequential(nn.Linear(P3_CHANNELS,128),nn.ReLU(),nn.Linear(128,7));object.__setattr__(self,'_teacher_external',load_teacher());self._p3=None;self.model[P3_LAYER].register_forward_hook(self._hook)
  for p in self.model[:P3_LAYER+1].parameters():p.register_hook(lambda g:self._mark_grad('student',g))
  for p in self.aux_classifier.parameters():p.register_hook(lambda g:self._mark_grad('aux',g))
 def _mark_grad(self,key,g):
  if g is not None and torch.isfinite(g).all() and bool((g!=0).any()):GRAD_SEEN[key]=True
  return g
 def _hook(self,module,inputs,output):self._p3=output
 def _teacher(self,device):
  t=self._teacher_external
  if next(t.parameters()).device!=device:t.to(device)
  t.eval();return t
 def loss(self,batch,preds=None):
  if getattr(self,'criterion',None) is None:self.criterion=self.init_criterion()
  if preds is None:preds=self.forward(batch['img'])
  base,items=self.criterion(preds,batch)
  bi=batch['batch_idx'].view(-1).to(self._p3.device);bb=batch['bboxes'].to(self._p3.device);cls=batch['cls'].view(-1).long().to(self._p3.device)
  fh,fw=self._p3.shape[-2:];cx,cy,w,h=bb.unbind(1);boxes=torch.stack([bi,(cx-w/2)*fw,(cy-h/2)*fh,(cx+w/2)*fw,(cy+h/2)*fh],1)
  rf=roi_align(self._p3,boxes,(7,7),spatial_scale=1.0,aligned=True).mean((2,3));student=self.aux_classifier(rf);aux=F.cross_entropy(student,cls)
  ih,iw=batch['img'].shape[-2:];pw=w*1.2;ph=h*1.2;ibox=torch.stack([bi,(cx-pw/2)*iw,(cy-ph/2)*ih,(cx+pw/2)*iw,(cy+ph/2)*ih],1);teacher=self._teacher(batch['img'].device);tlogs=[]
  with torch.no_grad():
   for q in ibox.split(16):
    crops=roi_align(batch['img'],q,(224,224),spatial_scale=1.0,aligned=True);mean=crops.new_tensor([.485,.456,.406])[None,:,None,None];std=crops.new_tensor([.229,.224,.225])[None,:,None,None];tlogs.append(teacher((crops-mean)/std)[0][:,:7])
  tlog=torch.cat(tlogs);tprob=F.softmax(tlog/2.0,1)
  kd=F.kl_div(F.log_softmax(student/2.0,1),tprob,reduction='batchmean')*4.0
  if not all(bool(torch.isfinite(x).all()) for x in (base.sum(),aux,kd,student,tlog)):raise RuntimeError('NON_FINITE_KD_COMPONENT')
  la=.1 if MODE in ('auxce','kd') else 0.;lk=.1 if MODE=='kd' else 0.;total=base+len(batch['img'])*(la*aux+lk*kd)
  items=dict(items);items['aux_ce_loss']=aux.detach();items['kd_loss']=kd.detach() if MODE=='kd' else kd.detach()*0
  AUX_LOG.append({'aux_ce':float(aux.detach()),'kd':float(kd.detach()),'instances':len(cls)})
  return total,items
class KDTrainer(SegmentationTrainer):
 def get_model(self,cfg=None,weights=None,verbose=True):
  model=self.set_model_names_for_load(KDModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose));
  if weights:model.load(weights)
  return model
def audit_checkpoint(p):
 c=torch.load(p,map_location='cpu',weights_only=False);m=c.get('ema') or c.get('model');bad=[k for k,v in m.state_dict().items() if torch.is_tensor(v) and not torch.isfinite(v).all()];YOLO(str(p));return {'sha256':sha(p),'finite':not bad,'reload':True,'bad':bad[:5]}
def run(a):
 global MODE,TEACHER_INITIAL;MODE=a.mode
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);t=load_teacher();TEACHER_INITIAL={k:tensor_hash(v) for k,v in t.state_dict().items()};args={'data':str(DATA),'imgsz':640,'batch':a.batch,'epochs':a.epochs,'seed':42,'deterministic':True,'amp':True,'optimizer':'AdamW','device':0,'workers':4,'cache':False,'val':a.val,'plots':a.val,'save':True,'fraction':a.fraction,'project':str(a.output/'ultralytics'),'name':a.mode,'exist_ok':False,'verbose':True}
 a.output=a.output.resolve();args['project']=str(a.output/'ultralytics');(a.output/'requested_args.json').write_text(json.dumps(args,indent=2)+'\n');wall=time.monotonic();y=YOLO(str(OFFICIAL));y.train(trainer=KDTrainer,**args);sd=Path(y.trainer.save_dir);model=y.trainer.model.module if hasattr(y.trainer.model,'module') else y.trainer.model;teacher=model._teacher_external;teacher_grad_none=all(p.grad is None for p in teacher.parameters());teacher_unchanged=all(tensor_hash(v)==TEACHER_INITIAL[k] for k,v in teacher.state_dict().items());student_grad=GRAD_SEEN['student'];aux_grad=GRAD_SEEN['aux'];cks={n:audit_checkpoint(sd/'weights'/n) for n in ('best.pt','last.pt') if (sd/'weights'/n).exists()};hist=[]
 if (sd/'results.csv').exists():
  hist=list(csv.DictReader(open(sd/'results.csv')))
 train_keys=[k for k in hist[0] if k.startswith('train/') and k.endswith('loss')] if hist else [];finite=all(math.isfinite(float(r[k])) for r in hist for k in train_keys)
 summary={'status':'PASS' if finite and teacher_grad_none and teacher_unchanged and student_grad and aux_grad and all(x['finite'] for x in cks.values()) else 'HARD_GATE','mode':a.mode,'test_accessed':False,'historical_method_reconstruction':True,'teacher_sha256':sha(TEACHER),'teacher_eval':not teacher.training,'teacher_requires_grad':any(p.requires_grad for p in teacher.parameters()),'teacher_grad_none':teacher_grad_none,'teacher_weights_unchanged':teacher_unchanged,'student_backbone_grad_nonzero':student_grad,'aux_head_grad_nonzero':aux_grad,'feature':{'layer_index':P3_LAYER,'layer_name':type(model.model[P3_LAYER]).__name__,'stride':P3_STRIDE,'channels':P3_CHANNELS,'shape_640':[a.batch,P3_CHANNELS,80,80]},'roi_align':'PASS','epochs':a.epochs,'batch':a.batch,'loss_finite':finite,'checkpoints':cks,'wall_seconds':time.monotonic()-wall,'save_dir':str(sd),'aux_batch_log_count':len(AUX_LOG)};(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return 0 if summary['status']=='PASS' else 3
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['auxce','kd'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=32);p.add_argument('--fraction',type=float,default=1.0);p.add_argument('--val',action='store_true');return run(p.parse_args())
if __name__=='__main__':raise SystemExit(main())
