#!/usr/bin/env python3
import argparse,csv,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFilter,ImageOps
REPO=Path(__file__).resolve().parents[1]; R1=REPO/'results/exp13_diffusion_pilot/run_20260921_124440'; MODEL='/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting'; MODEL_ID='aiplayground/dreamshaper-8-inpainting'; REV='master'; W=H=512; STEPS=20; GUIDANCE=7.5; SEED=17001; FEATHER=1.0
SOURCES=[('Crack','260'),('Crack','469'),('corrosion','9'),('corrosion','23')]
P1={
 'Crack': 'A realistic thin industrial crack or fracture on a metallic engine interior surface, appearing as a narrow dark fracture line with thin linear or slightly curved morphology and optional subtle branching, irregular but natural sharp boundaries, preserving the original metallic surface texture, component geometry, and realistic borescope illumination. Preserve the surrounding structure. Do not create a scratch, paint mark, or stain.',
 'corrosion': 'A realistic localized corrosion region on a metallic engine interior surface, an irregular oxidized patch with rough granular surface degradation and an uneven or diffuse natural boundary, preserving the original metallic geometry, surrounding structure, surface context, and realistic borescope inspection illumination.'}
P2={
 'Crack': 'Generate a clearly visible realistic crack on the metallic engine interior surface. Keep the defect as a crack or fracture, not a repaired clean surface. Make it a thin dark fracture line with natural irregular boundaries while preserving the surrounding geometry and borescope illumination.',
 'corrosion': 'Generate a clearly visible realistic corrosion region on the metallic engine interior surface. Keep it as corrosion, not a clean repaired area, hole, fastener, or seam. Use an irregular rough oxidized patch with natural uneven boundaries while preserving the surrounding geometry and borescope illumination.'}
NEG={'Crack':'scratch, paint mark, stain, text, watermark, changed geometry, smooth clean surface, cartoon, blurry','corrosion':'scratch, paint mark, stain, hole, fastener, seam, text, watermark, changed geometry, smooth clean surface, cartoon, blurry'}
def fit(path,mode):return ImageOps.fit(Image.open(path).convert(mode),(W,H),method=Image.Resampling.LANCZOS if mode=='RGB' else Image.Resampling.NEAREST,centering=(.5,.5))
def dilate(src,dst,px):src.filter(ImageFilter.MaxFilter(px*2+1)).filter(ImageFilter.GaussianBlur(FEATHER)).save(dst)
def prepare(run):
 for rel in ['environment','prompts','masks/original','masks/inpaint_A','masks/inpaint_B','crack/inputs','crack/outputs','corrosion/inputs','corrosion/outputs','manifests','review_assets','logs','summary']:(run/rel).mkdir(parents=True,exist_ok=False)
 (run/'prompts/prompt_variants.json').write_text(json.dumps({'v1':P1,'v2':P2,'negative':NEG},indent=2)+'\n')
 r1=list(csv.DictReader((R1/'manifests/pilot_manifest.csv').open(newline=''))); selected=[]
 for cls,src in SOURCES:
  row=next(x for x in r1 if x['class']==cls and x['source_id']==src and x['seed']=='13001')
  selected.append(row)
 rows=[]
 for cls,src in SOURCES:
  r=next(x for x in selected if x['class']==cls and x['source_id']==src); key='crack' if cls=='Crack' else 'corrosion'; iid=f'{key}_{src}'
  inp=run/key/'inputs'/(iid+Path(r['source_image_path']).suffix); shutil.copy2(REPO/r['source_image_path'],inp)
  orig=run/'masks/original'/(iid+'_mask.png'); shutil.copy2(REPO/r['original_mask_path'],orig)
  ma=run/'masks/inpaint_A'/(iid+'_mask.png'); shutil.copy2(REPO/r['inpaint_mask_path'],ma)
  mb=run/'masks/inpaint_B'/(iid+'_mask.png'); dilate(Image.open(orig).convert('L'),mb,5)
  for strategy,mp in [('mask_A',ma),('mask_B',mb)]:
   for strength in [0.55,0.75,0.95]:
    for pv,prompt in [('v1',P1[cls]),('v2',P2[cls])]:
     did=f'{iid}_{strategy}_s{str(strength).replace(".","")}_{pv}'
     rows.append({'diag_id':did,'class':cls,'source_id':src,'source_image_path':str(inp),'original_mask_path':str(orig),'mask_strategy':strategy,'inpaint_mask_path':str(mp),'mask_dilation_px':'3 (R1)' if strategy=='mask_A' else '5','prompt_id':f'{key}_prompt_{pv}','prompt_text':prompt,'strength':strength,'seed':SEED,'model_name':MODEL_ID,'model_revision':REV,'inference_width':W,'inference_height':H,'num_inference_steps':STEPS,'guidance_scale':GUIDANCE,'output_path':str(run/key/'outputs'/(did+'.png')),'status':'PENDING','failure_reason':'','parent_quality_failed_run':'run_20260921_124440'})
 with (run/'manifests/diagnosis_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (run/'environment/environment.txt').write_text('model='+MODEL_ID+'\nrevision='+REV+'\nseed=17001\nval_access=0\ntest_access=0\n')
 return rows
def generate(run):
 import torch
 from diffusers import StableDiffusionInpaintPipeline
 p=StableDiffusionInpaintPipeline.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float16,safety_checker=None).to('cuda');p.enable_attention_slicing();mf=run/'manifests/diagnosis_manifest.csv';rows=list(csv.DictReader(mf.open(newline='')));log=(run/'logs/generation_log.txt').open('a')
 for r in rows:
  try:
   g=torch.Generator(device='cuda').manual_seed(SEED);out=p(prompt=r['prompt_text'],negative_prompt=NEG[r['class']],image=fit(r['source_image_path'],'RGB'),mask_image=fit(r['inpaint_mask_path'],'L'),height=H,width=W,num_inference_steps=STEPS,guidance_scale=GUIDANCE,strength=float(r['strength']),generator=g).images[0];out.save(r['output_path']);r['status']='SUCCESS';log.write(r['diag_id']+' SUCCESS\n')
  except Exception as e:r['status']='FAILED';r['failure_reason']=repr(e);log.write(r['diag_id']+' FAILED '+repr(e)+'\n')
  with mf.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerows(rows)
 log.close();del p;torch.cuda.empty_cache();return rows
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--generate',action='store_true');x=a.parse_args();run=Path(x.run)
 if x.prepare:prepare(run)
 if x.generate:generate(run)
if __name__=='__main__':main()
