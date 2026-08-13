#!/usr/bin/env python3
"""Exp09 TRAIN-only SimSiam adaptation of YOLO11 backbone and downstream transfer."""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from ultralytics import YOLO

ROOT=Path('/root/autodl-tmp/borescope-new-seg-repro');DATA=Path('/root/autodl-tmp/borescope-new-seg-data/v1');OFFICIAL=ROOT/'weights/yolo11n-seg.pt';SPLIT=ROOT/'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv';OUT=ROOT/'results/fast_repro/exp09_simsiam';BACKBONE_END=10
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def train_paths():
 with open(SPLIT,encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 tr=[DATA/'images'/'train'/(x['stem']+x['image_suffix']) for x in r if x['split']=='train']
 if len(tr)!=668 or len(set(tr))!=668 or not all(p.parent.name=='train' for p in tr):raise RuntimeError('SSL_TRAIN_SET_GATE')
 return tr
aug=transforms.Compose([transforms.RandomResizedCrop(512,scale=(.2,1.0)),transforms.RandomHorizontalFlip(),transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.1)],p=.8),transforms.RandomGrayscale(.2),transforms.RandomApply([transforms.GaussianBlur(23,sigma=(.1,2.0))],p=.5),transforms.ToTensor()])
class SSLDS(Dataset):
 def __init__(self,paths):self.paths=paths
 def __len__(self):return len(self.paths)
 def __getitem__(self,i):
  im=Image.open(self.paths[i]).convert('RGB');return aug(im),aug(im)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();y=YOLO(str(OFFICIAL));self.layers=nn.Sequential(*[copy.deepcopy(y.model.model[i]) for i in range(BACKBONE_END+1)])
 def forward(self,x):return self.layers(x)
class SimSiam(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=Encoder();self.projector=nn.Sequential(nn.Linear(256,2048,bias=False),nn.BatchNorm1d(2048),nn.ReLU(),nn.Linear(2048,2048,bias=False),nn.BatchNorm1d(2048),nn.ReLU(),nn.Linear(2048,2048,bias=False),nn.BatchNorm1d(2048,affine=False));self.predictor=nn.Sequential(nn.Linear(2048,512,bias=False),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,2048))
 def branch(self,x):f=self.encoder(x).mean((2,3));z=self.projector(f);p=self.predictor(z);return f,z,p
def negcos(p,z):return -F.cosine_similarity(p,z.detach(),dim=1).mean()
def finite_model(m):return all(bool(torch.isfinite(v).all()) for v in m.state_dict().values() if torch.is_tensor(v))
def ssl(a):
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);torch.manual_seed(42);np.random.seed(42);random.seed(42);paths=train_paths();ds=SSLDS(paths);g=torch.Generator().manual_seed(42);dl=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,generator=g);m=SimSiam().cuda();opt=torch.optim.SGD(m.parameters(),lr=.05*a.batch/256,momentum=.9,weight_decay=1e-4);hist=[];wall=time.monotonic();best=1e9
 for ep in range(1,a.epochs+1):
  m.train();ls=[];fs=[];zs=[]
  for x1,x2 in dl:
   x1=x1.cuda(non_blocking=True);x2=x2.cuda(non_blocking=True);f1,z1,p1=m.branch(x1);f2,z2,p2=m.branch(x2);loss=(negcos(p1,z2)+negcos(p2,z1))/2
   if not torch.isfinite(loss):
    raise RuntimeError('SIMSIAM_NONFINITE')
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls.append(loss.item());fs.append(torch.cat([f1,f2]).std(0).mean().item());zs.append(torch.cat([z1,z2]).std(0).mean().item())
  row={'epoch':ep,'loss':float(np.mean(ls)),'feature_std':float(np.mean(fs)),'embedding_std':float(np.mean(zs))};hist.append(row);print(row,flush=True);state={'epoch':ep,'model':m.state_dict(),'batch':a.batch,'seed':42,'train_images':668}
  torch.save(state,a.output/'last.pt')
  if row['loss']<best:best=row['loss'];torch.save(state,a.output/'best.pt')
 with open(a.output/'history.csv','w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
 collapse=min(x['feature_std'] for x in hist)<1e-4 or min(x['embedding_std'] for x in hist)<1e-4;st=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);bb={k.removeprefix('encoder.layers.'):v for k,v in st['model'].items() if k.startswith('encoder.layers.')};torch.save({'backbone':bb,'layers':[0,BACKBONE_END],'source_official_sha256':sha(OFFICIAL),'train_images':668},a.output/'adapted_backbone.pt')
 summary={'status':'FAIL_COLLAPSE' if collapse else 'PASS','test_accessed':False,'ssl_train_images':668,'val_images_seen':0,'test_images_seen':0,'encoder_layers':[0,BACKBONE_END],'encoder_parameter_count':sum(p.numel() for p in m.encoder.parameters()),'expected_backbone_tensors':len(bb),'epochs':a.epochs,'batch':a.batch,'optimizer':'SGD','lr':.05*a.batch/256,'loss_finite':all(math.isfinite(x['loss']) for x in hist),'feature_std_min':min(x['feature_std'] for x in hist),'embedding_std_min':min(x['embedding_std'] for x in hist),'collapse':collapse,'checkpoint_finite':finite_model(m),'adapted_backbone_sha256':sha(a.output/'adapted_backbone.pt'),'wall_seconds':time.monotonic()-wall};(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return 0 if summary['status']=='PASS' else 3
def transfer(a):
 st=torch.load(a.backbone,map_location='cpu',weights_only=False)['backbone'];y=YOLO(str(OFFICIAL));model=y.model;expected={k:v for k,v in model.model[:BACKBONE_END+1].state_dict().items()};before={k:thash(v) for k,v in expected.items()};missing=[k for k in expected if k not in st];unexpected=[k for k in st if k not in expected]
 if missing or unexpected:raise RuntimeError(f'TRANSFER_KEYS {len(missing)} {len(unexpected)}')
 model.model[:BACKBONE_END+1].load_state_dict(st,strict=True);after={k:thash(v) for k,v in model.model[:BACKBONE_END+1].state_dict().items()};changed=[k for k in before if before[k]!=after[k]]
 if not changed:raise RuntimeError('BACKBONE_NOT_CHANGED');a.output.parent.mkdir(parents=True,exist_ok=True);y.save(a.output)
 check=YOLO(str(a.output)).model;loaded=check.model[:BACKBONE_END+1].state_dict();verified=sum(thash(loaded[k])==after[k] for k in after);rep={'status':'PASS' if verified==len(expected) else 'HARD_GATE','test_accessed':False,'expected_tensor_count':len(expected),'loaded_tensor_count':verified,'missing_tensor_count':len(missing),'unexpected_tensor_count':len(unexpected),'changed_tensor_count':len(changed),'sample_layer':changed[0],'coco_hash':before[changed[0]],'simsiam_hash':after[changed[0]],'downstream_hash':thash(loaded[changed[0]]),'output_sha256':sha(a.output)};(a.report).write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2));return 0 if rep['status']=='PASS' else 3
def downstream(a):
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);args={'data':str(DATA/'data.yaml'),'imgsz':640,'batch':32,'epochs':100,'seed':42,'deterministic':True,'amp':True,'optimizer':'AdamW','device':0,'workers':4,'cache':False,'val':True,'plots':True,'save':True,'project':str((a.output/'ultralytics').resolve()),'name':'simsiam_init','exist_ok':False,'verbose':True};(a.output/'requested_args.json').write_text(json.dumps(args,indent=2)+'\n');wall=time.monotonic();y=YOLO(str(a.model));y.train(**args);sd=Path(y.trainer.save_dir);cks={}
 for n in ('best.pt','last.pt'):
  p=sd/'weights'/n;c=torch.load(p,map_location='cpu',weights_only=False);mo=c.get('ema') or c.get('model');finite=all(bool(torch.isfinite(v).all()) for v in mo.state_dict().values());YOLO(str(p));cks[n]={'sha256':sha(p),'finite':finite,'reload':True,'path':str(p)}
 rows=list(csv.DictReader(open(sd/'results.csv')));keys=[k for k in rows[0] if k.startswith('train/') and k.endswith('loss')];summary={'status':'PASS' if all(math.isfinite(float(r[k])) for r in rows for k in keys) and all(x['finite'] for x in cks.values()) else 'HARD_GATE','test_accessed':False,'epochs':100,'batch':32,'checkpoints':cks,'save_dir':str(sd),'wall_seconds':time.monotonic()-wall};(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return 0 if summary['status']=='PASS' else 3
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True);q=sp.add_parser('ssl');q.add_argument('--output',type=Path,required=True);q.add_argument('--epochs',type=int,default=100);q.add_argument('--batch',type=int,required=True);q=sp.add_parser('transfer');q.add_argument('--backbone',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--report',type=Path,required=True);q=sp.add_parser('downstream');q.add_argument('--model',type=Path,required=True);q.add_argument('--output',type=Path,required=True);a=p.parse_args();return {'ssl':ssl,'transfer':transfer,'downstream':downstream}[a.cmd](a)
if __name__=='__main__':raise SystemExit(main())
