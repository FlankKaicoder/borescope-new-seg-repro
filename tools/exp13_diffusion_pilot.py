#!/usr/bin/env python3
import argparse,csv,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFilter,ImageOps
REPO=Path(__file__).resolve().parents[1]; AUDIT=REPO/'results/exp13_vlm_semantic_audit'; MODEL='runwayml/stable-diffusion-inpainting'; REV='main'; W=H=512; STEPS=20; GUIDANCE=7.5; STRENGTH=.95; DILATION=3; FEATHER=1.0
SEEDS={1:13001,2:13002}
PROMPTS={
 'Crack':{'prompt_id':'crack_human_reviewed_v1','text':'A realistic thin industrial crack or fracture on a metallic engine interior surface, appearing as a narrow dark fracture line with thin linear or slightly curved morphology and optional subtle branching, irregular but natural sharp boundaries, preserving the original metallic surface texture, component geometry, and realistic borescope illumination. Preserve the surrounding structure. Do not create a scratch, paint mark, or stain.','negative':'scratch, paint mark, stain, text, watermark, new object, changed component geometry, smooth plastic, cartoon, illustration, blurry, duplicated defect'},
 'corrosion':{'prompt_id':'corrosion_human_reviewed_v1','text':'A realistic localized corrosion region on a metallic engine interior surface, an irregular oxidized patch with rough granular surface degradation and an uneven or diffuse natural boundary, preserving the original metallic geometry, surrounding structure, surface context, and realistic borescope inspection illumination.','negative':'scratch, paint mark, stain, text, watermark, new object, changed component geometry, smooth plastic, cartoon, illustration, blurry, duplicated defect'}}
def load_records():
 rows=list(csv.DictReader((AUDIT/'sample_manifest.csv').open(newline=''))); ctx={str(x['source_id']):x for x in json.loads((AUDIT/'metadata/polygon_context.json').read_text())}
 if len(rows)!=40 or any(x['split']!='train' for x in rows): raise RuntimeError('source manifest is not 40 TRAIN-only rows')
 out=[]
 for cls in ('Crack','corrosion'):
  xs=[x for x in rows if x['class']==cls]; xs.sort(key=lambda x:int(x['source_id']) if x['source_id'].isdigit() else x['source_id'])
  if len(xs)!=20: raise RuntimeError('expected 20 '+cls)
  out.extend(xs[:10])
 return out,ctx
def make_mask(rec,ctx,original,derived):
 src=Image.open(rec['image_path']).convert('RGB'); w,h=src.size; m=Image.new('L',src.size,0); d=ImageDraw.Draw(m)
 for a in ctx['annotations']:
  if a['class']==rec['class']:
   pts=[(round(x*w),round(y*h)) for x,y in a['points']]
   if len(pts)>=3:d.polygon(pts,fill=255)
 if m.getbbox() is None: raise RuntimeError('empty mask '+rec['source_id'])
 m.save(original); m.filter(ImageFilter.MaxFilter(DILATION*2+1)).filter(ImageFilter.GaussianBlur(FEATHER)).save(derived)
def prepare(run):
 recs,ctx=load_records();
 for rel in ('environment','prompts','masks/original','masks/inpaint','crack/inputs','crack/outputs','corrosion/inputs','corrosion/outputs','manifests','review_assets','logs','summary'):(run/rel).mkdir(parents=True,exist_ok=False)
 for cls,s in PROMPTS.items():(run/'prompts'/('crack_prompts.json' if cls=='Crack' else 'corrosion_prompts.json')).write_text(json.dumps(s,indent=2)+'\n')
 (run/'prompt_config.json').write_text(json.dumps({'source':'human-reviewed VLM-oriented semantic templates; not per-image automatic VLM output','templates':PROMPTS},indent=2)+'\n')
 rows=[]; seeds=[]
 for i,r in enumerate(recs,1):
  key='crack' if r['class']=='Crack' else 'corrosion'; iid=f"vlm_{i:02d}_{r['source_id']}"; src=Path(r['image_path']); inp=run/key/'inputs'/(iid+src.suffix.lower()); shutil.copy2(src,inp); om=run/'masks/original'/(iid+'_mask.png'); im=run/'masks/inpaint'/(iid+'_mask.png'); make_mask(r,ctx[str(r['source_id'])],om,im); s=PROMPTS[r['class']]
  for v,seed in SEEDS.items():
   pid=f'{key}_{r["source_id"]}_v{v}'; out=run/key/'outputs'/(pid+'.png'); seeds.append({'pilot_id':pid,'source_id':r['source_id'],'class':r['class'],'variant':v,'seed':seed}); rows.append({'pilot_id':pid,'class':r['class'],'source_id':r['source_id'],'source_image_path':str(inp),'original_mask_path':str(om),'inpaint_mask_path':str(im),'prompt_id':s['prompt_id'],'prompt_text':s['text'],'seed':seed,'model_name':MODEL,'model_revision':REV,'inference_width':W,'inference_height':H,'num_inference_steps':STEPS,'guidance_scale':GUIDANCE,'strength_if_applicable':STRENGTH,'output_path':str(out),'status':'PENDING','failure_reason':''})
 with (run/'manifests/pilot_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (run/'manifests/seed_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(seeds[0]));w.writeheader();w.writerows(seeds)
 (run/'environment/environment.txt').write_text('Preparation recorded; runtime details are in handoff.\n'); return rows
def pipe_load():
 import torch
 from diffusers import StableDiffusionInpaintPipeline
 p=StableDiffusionInpaintPipeline.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float16,safety_checker=None).to('cuda'); p.enable_attention_slicing(); return p
def fit(path,mode):return ImageOps.fit(Image.open(path).convert(mode),(W,H),method=Image.Resampling.LANCZOS if mode=='RGB' else Image.Resampling.NEAREST,centering=(.5,.5))
def generate(run,only=None):
 import torch
 p=pipe_load(); mf=run/'manifests/pilot_manifest.csv'; rows=list(csv.DictReader(mf.open(newline=''))); log=(run/'logs/generation_log.txt').open('a')
 for r in rows:
  if only and r['pilot_id'] not in only:continue
  try:
   g=torch.Generator(device='cuda').manual_seed(int(r['seed'])); img=fit(r['source_image_path'],'RGB'); mask=fit(r['inpaint_mask_path'],'L'); out=p(prompt=r['prompt_text'],negative_prompt=PROMPTS[r['class']]['negative'],image=img,mask_image=mask,height=H,width=W,num_inference_steps=STEPS,guidance_scale=GUIDANCE,strength=STRENGTH,generator=g).images[0]; out.save(r['output_path']); r['status']='SUCCESS'; log.write(r['pilot_id']+' SUCCESS\n')
  except Exception as e:r['status']='FAILED';r['failure_reason']=repr(e);log.write(r['pilot_id']+' FAILED '+repr(e)+'\n')
  with mf.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerows(rows)
 log.close(); del p; torch.cuda.empty_cache(); return rows
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--generate',action='store_true');a.add_argument('--only',nargs='*');x=a.parse_args();run=Path(x.run)
 if x.prepare:prepare(run)
 if x.generate:generate(run,set(x.only) if x.only else None)
if __name__=='__main__':main()
