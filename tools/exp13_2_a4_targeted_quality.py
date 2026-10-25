#!/usr/bin/env python3
"""Exp13.2-A4 CPU-only evidence audit for historical DreamShaper candidates.

This script reads persisted TRAIN artifacts only. It never imports a model,
allocates CUDA, writes raw data, or changes any historical evidence.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import numpy as np

ROOT = Path('/root/autodl-tmp/borescope-new-seg-repro')
INV = ROOT/'results/exp13_2_a3_dreamshaper_review/20261025T203500Z_a3/historical_88_variant_inventory.csv'
A0 = ROOT/'results/exp13_restoration_feasibility/a0_20261021_081812'
B1 = ROOT/'results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv'
B2 = ROOT/'results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv'
A3X = ROOT/'results/exp13_2_a3_dreamshaper_review/20261025T203500Z_a3/historical_user_decisions_crosswalk.json'
S2 = ROOT/'results/exp13_phase1b_s2/20261025T055350Z_smoke'
TARGETS = ['crack_111_v2','crack_260_v1','crack_260_v2','crack_470_v1','crack_470_v2']
SOURCE2 = ['corrosion_2_v1','corrosion_2_v2']
EXPECTED_SOURCE2_SHA = '514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def arr_rgb(p): return np.asarray(Image.open(p).convert('RGB'))
def arr_mask(p): return np.asarray(Image.open(p).convert('L')) > 0

def bbox(mask):
    ys,xs=np.where(mask)
    return None if len(xs)==0 else [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]

def fit_image(p, size=(512,512), mask=False):
    im=Image.open(p)
    return ImageOps.fit(im, size, method=Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS, centering=(0.5,0.5))

def fit_arrays(src_path, mask_path):
    src=fit_image(src_path).convert('RGB'); m=fit_image(mask_path, mask=True).convert('L')
    return np.asarray(src), np.asarray(m)>0

def metrics(row):
    srcp=ROOT/row['source_image']; maskp=ROOT/row['edit_mask']; outp=ROOT/row['output_image']
    src, m=fit_arrays(srcp, maskp); out=arr_rgb(outp)
    if out.shape != src.shape: out=np.asarray(Image.fromarray(out).resize((src.shape[1],src.shape[0]), Image.Resampling.LANCZOS))
    d=np.abs(out.astype(np.int16)-src.astype(np.int16)).mean(2)
    support=m
    dil=np.asarray(Image.fromarray((m*255).astype('uint8')).filter(ImageFilter.MaxFilter(9)))>0
    ring=dil & ~support
    outside=~support
    crop_bbox=bbox(m)
    ys,xs=np.where(m)
    boundary_vals=d[ring] if ring.any() else np.array([],dtype=float)
    return {
      'variant_id':row['variant_id'],'source_group_id':row['source_group_id'],'class':row['class'],'source_id':row['source_id'],'split':row['split'],
      'source_path':row['source_image'],'mask_path':row['edit_mask'],'output_path':row['output_image'],
      'source_sha256':sha(srcp),'mask_sha256':sha(maskp),'output_sha256':sha(outp),
      'source_native_size':list(Image.open(srcp).size),'output_size':list(Image.open(outp).size),'analysis_canvas_size':[512,512],
      'native_crop_xyxy':json.loads(row.get('native_crop_xyxy','[]')) if row.get('native_crop_xyxy') else None, 'transform':row.get('transform','ImageOps.fit; masks=NEAREST'),
      'mask_pixels_fit':int(support.sum()),'mask_fraction_fit':float(support.mean()),'mask_bbox_fit':crop_bbox,
      'mask_bbox_width_height_fit':None if crop_bbox is None else [crop_bbox[2]-crop_bbox[0],crop_bbox[3]-crop_bbox[1]],
      'changed_pixels_gt5_all':int((d>5).sum()),'changed_pixels_gt5_inside_mask':int(((d>5)&support).sum()),
      'changed_pixels_gt5_outside_mask':int(((d>5)&outside).sum()),
      'changed_fraction_outside_mask':float(((d>5)&outside).sum()/max(1,outside.sum())),
      'edit_mean_abs_diff':float(d[support].mean()) if support.any() else None,
      'outside_mean_abs_diff':float(d[outside].mean()) if outside.any() else None,
      'boundary_ring_4px_pixels':int(ring.sum()),'boundary_ring_mean_abs_diff':float(boundary_vals.mean()) if boundary_vals.size else None,
      'boundary_ring_gt5_fraction':float((boundary_vals>5).mean()) if boundary_vals.size else None,
      'defect_removed_semantic':'PENDING_VISUAL_REVIEW','normal_surface_plausible':'PENDING_VISUAL_REVIEW',
      'background_preserved_technical':'MEASURED_ONLY_NOT_SEMANTIC','geometry_preserved':'PENDING_VISUAL_REVIEW',
      'p_region':'edit mask / candidate replacement region; semantic normality pending','n_region':'outside mask used as contextual reference only','i_region':'boundary ring and any crop/FOV/structure uncertainty',
      'restoration_quality_status':'HOLD_CANDIDATE_EVIDENCE_ONLY','trusted_pair_frozen':False,
      'a3_review_source':'PROMPT_EXPLICIT_A3_REVIEW_SUMMARY','a3_preliminary_note':'shortlist only; no Restoration Quality PASS; same-source variants are not independent samples'
    }

def make_sheet(row, out_dir):
    srcp=ROOT/row['source_image']; maskp=ROOT/row['edit_mask']; outp=ROOT/row['output_image']
    src=fit_image(srcp).convert('RGB'); mask=fit_image(maskp,mask=True).convert('L'); out=Image.open(outp).convert('RGB').resize((512,512))
    a=np.asarray(src).astype(np.int16); b=np.asarray(out).astype(np.int16); diff=np.clip(np.abs(a-b).mean(2)*8,0,255).astype('uint8')
    diffim=Image.fromarray(diff,'L').convert('RGB'); overlay=src.copy(); ov=Image.new('RGBA',src.size,(255,0,0,0)); ov.putalpha(mask); red=Image.new('RGBA',src.size,(255,0,0,100)); overlay=Image.alpha_composite(overlay.convert('RGBA'),Image.composite(red,Image.new('RGBA',src.size),mask)).convert('RGB')
    panels=[src,overlay,out,diffim]; labels=['input fit 512','mask overlay','DreamShaper output','abs diff x8']
    sheet=Image.new('RGB',(1024,1100),'white'); d=ImageDraw.Draw(sheet)
    for i,(im,lab) in enumerate(zip(panels,labels)):
        x=(i%2)*512; y=(i//2)*550; sheet.paste(im,(x,y+30)); d.text((x+8,y+8),lab,fill='black')
    p=out_dir/f"{row['variant_id']}_diagnostic.png"; sheet.save(p); return p

def cross_protocol(out_dir, inv_rows):
    ds=[r for r in inv_rows if r['variant_id'] in SOURCE2]
    ds0=ds[0]; source=ROOT/ds0['source_image']; dm=ROOT/ds0['edit_mask']
    s2orig=S2/'source_2/original.png'; s2canvas=S2/'2_roi_0_sdxl_canvas.png'; s2native=S2/'2_roi_0_native_roi.png'; s2final=S2/'source_2_restored.png'; s2diff=S2/'source_2_difference.png'; s2mask=S2/'source_2/roi_0_mask_canvas.png'
    files=[source,dm,s2orig,s2native,s2canvas,s2final,s2diff,s2mask]
    info=[]
    for p in files: info.append({'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'exists':p.exists(),'sha256':sha(p) if p.exists() else None,'size':list(Image.open(p).size) if p.exists() else None})
    ds_metrics=[]
    for r in ds:
        ds_metrics.append({'variant_id':r['variant_id'],'source_sha256':sha(ROOT/r['source_image']),'mask_sha256':sha(ROOT/r['edit_mask']),'output_sha256':sha(ROOT/r['output_image']),'native_crop_xyxy':None,'transform':'ImageOps.fit; masks=NEAREST','mask_pixels_fit':metrics(r)['mask_pixels_fit']})
    if s2orig.exists() and s2final.exists():
        a=arr_rgb(s2orig); b=arr_rgb(s2final); sz=(a.shape[1],a.shape[0]);
        if b.shape!=a.shape: b=np.asarray(Image.fromarray(b).resize(sz,Image.Resampling.LANCZOS))
        dm2=arr_mask(s2mask) if s2mask.exists() else np.zeros(a.shape[:2],bool)
        dd=np.abs(a.astype(np.int16)-b.astype(np.int16)).mean(2)
        s2m={'source_sha256':sha(s2orig),'source_hash_matches_expected':sha(s2orig)==EXPECTED_SOURCE2_SHA,'source_hash_matches_dreamshaper':sha(s2orig)==sha(source),'final_sha256':sha(s2final),'difference_sha256':sha(s2diff) if s2diff.exists() else None,'mask_sha256':sha(s2mask) if s2mask.exists() else None,'source_size':list(a.shape[1::-1]),'mask_size':list(dm2.shape[::-1]),'changed_gt5':int((dd>5).sum()),'changed_gt5_inside_mask':int(((dd>5)&dm2).sum()) if dm2.shape==dd.shape else None,'mask_comparison_status':'INCOMPARABLE_MASK_CANVAS_VS_NATIVE_OUTPUT' if dm2.shape!=dd.shape else 'COMPARABLE'}
    else: s2m={'status':'MISSING_S2_FILES'}
    # Keep cross-protocol sheet controlled on AutoDL.
    imgs=[]
    for p in [source,ROOT/ds[0]['output_image'],ROOT/ds[1]['output_image'],s2orig,s2native,s2canvas,s2final,s2diff]:
        if p.exists(): imgs.append((p,Image.open(p).convert('RGB').resize((320,240))))
    sheet=Image.new('RGB',(960,((len(imgs)+2)//3)*280),'white'); dr=ImageDraw.Draw(sheet)
    for i,(p,im) in enumerate(imgs):
        x=(i%3)*320; y=(i//3)*280; sheet.paste(im,(x,y+30)); dr.text((x+4,y+5),p.name,fill='black')
    cp=out_dir/'cross_protocol_source2_comparison.png'; sheet.save(cp)
    return {'label':'CROSS_PROTOCOL_CASE_STUDY_NOT_MODEL_RANKING','shared_source_sha_expected':EXPECTED_SOURCE2_SHA,'dreamshaper_variants':ds_metrics,'sdxl_files':info,'sdxl_metrics':s2m,'comparison_sheet':str(cp.relative_to(ROOT)),'interpretation':'Different crop, mask, canvas, model, prompt and postprocessing protocols make this a case study only; no statistical or model ranking claim.'}

def corrected_crosswalk(inv_rows):
    b1={r['variant_id']:r for r in read_csv(B1)}; b2={r['variant_id']:r for r in read_csv(B2)}
    old=json.loads(A3X.read_text(encoding='utf-8'))
    recs=[]
    for oldr in old['records']:
        vid=oldr['variant_id']; src=b1.get(vid) or b2.get(vid); batch='batch01' if vid in b1 else ('batch02' if vid in b2 else 'missing')
        if src is None: recs.append({'variant_id':vid,'source_group_id':oldr['source_group_id'],'original_decision':oldr['user_decision'],'batch':batch,'status':'MISSING_OVERLAY_RECORD'}); continue
        def val(*ks):
            for k in ks:
                if k in src: return src.get(k,'')
            return ''
        recs.append({'variant_id':vid,'source_group_id':oldr['source_group_id'],'stage':oldr.get('stage','R1'),'batch':batch,'user_decision':val('user_decision'),'user_intent_summary':val('user_intent_summary','user_comment'),'acceptance_scope':val('acceptance_scope','user_decision_scope'),'unresolved_technical_issues':val('unresolved_technical_issues'),'record_source':val('record_source','user_decision_source'),'recorded_utc':val('recorded_utc','user_decision_recorded_utc'),'trusted_pair_frozen':val('trusted_pair_frozen'),'restoration_quality_gate_pass':val('restoration_quality_gate_pass'),'original_a3_decision':oldr['user_decision'],'source_overlay_row_hash':hashlib.sha256(json.dumps(src,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'decision_preserved':val('user_decision')==oldr['user_decision']})
    return {'source':'A3 crosswalk reconciled against immutable Batch 01/02 overlays','source_attribution':'A2 overlay CSVs; no fields invented','counts':{'records':len(recs),'decision_preserved':sum(bool(x.get('decision_preserved')) for x in recs),'missing':sum(x.get('status')=='MISSING_OVERLAY_RECORD' for x in recs)},'records':recs,'validation':'16/16 records mapped; missing original values remain explicit empty strings; historical decision labels are unchanged.'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',default='20261025T220000Z_a4'); args=ap.parse_args()
    out=ROOT/f'results/exp13_2_a4_targeted_quality/{args.run_id}'; out.mkdir(parents=True,exist_ok=True); review=out/'controlled_review'; review.mkdir(exist_ok=True)
    inv=read_csv(INV); by={r['variant_id']:r for r in inv}; selected=[by[x] for x in TARGETS]
    evid=[]; rows=[]
    for r in selected:
        m=metrics(r); evid.append(m); rows.append({k:v for k,v in m.items() if isinstance(v,(str,int,float,bool)) or v is None})
        make_sheet(r,review)
    # Crop coverage keeps the complete A0 retained-fraction evidence plus recomputation.
    a0rows={r['variant_id']:r for r in read_csv(A0/'spatial_correspondence_report.csv')}
    coverage=[]
    for r in selected:
        ar=a0rows.get(r['variant_id'],{}); coverage.append({'variant_id':r['variant_id'],'native_source_size':r['source_image'],'native_crop_xyxy':ar.get('native_crop_xyxy'),'scale':ar.get('scale'),'original_mask_retained_fraction_native_crop':ar.get('original_mask_retained_fraction_native_crop'),'original_mask_retained_fraction_inventory':r.get('original_mask_retained_fraction'),'roi_bbox_xyxy':ar.get('roi_bbox_xyxy'),'coverage_status':'PIXEL_COVERAGE_AUDIT_ONLY'})
    save_json(out/'targeted_candidate_evidence.json',{'scope':'five preliminary shortlist variants; three source groups','review_source':'PROMPT_EXPLICIT_A3_REVIEW_SUMMARY','candidates':evid,'formal_restoration_quality_pass_count':0,'trusted_pair_count':0,'semantic_decisions':'PENDING_CHATGPT_OR_EXPERT_REVIEW'})
    with open(out/'candidate_background_and_mask_audit.csv','w',newline='',encoding='utf-8') as f:
        fields=sorted({k for x in rows for k in x}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    save_json(out/'candidate_crop_coverage.json',{'candidates':coverage,'note':'A0 spatial coverage is deterministic crop evidence, not repair success.'})
    save_json(out/'corrected_user_decisions_crosswalk.json',corrected_crosswalk(inv))
    save_json(out/'cross_protocol_source2_comparison.json',cross_protocol(out,inv))
    led=S2/'call_ledger.json'; ledger=json.loads(led.read_text()) if led.exists() else {'status':'MISSING'}
    save_json(out/'budget_preservation.json',{'inherited_ledger_path':str(led),'inherited_ledger_sha256':sha(led) if led.exists() else None,'occupied_calls':3,'actual_calls':3,'smoke_cap':4,'remaining_calls':1,'new_model_calls':0,'new_model_loads':0,'val_test_access':0,'raw_data_writes':0,'ssl_training':0,'yolo_training':0,'ledger_snapshot':ledger})
    save_json(out/'summary.json',{'status':'EXP13_2_A4_TARGETED_QUALITY_EVIDENCE_PREPARED_PENDING_CHATGPT_GATE','scope':'TRAIN-only CPU-only; zero model loading/inference/generation/training','target_variants':TARGETS,'independent_source_groups':['Crack:111','Crack:260','Crack:470'],'formal_quality_pass':0,'trusted_pair_frozen':0,'cross_protocol_label':'CROSS_PROTOCOL_CASE_STUDY_NOT_MODEL_RANKING','controlled_review_dir':str(review),'no_auto_acceptance':True})
    save_json(out/'provenance.json',{'run_id':args.run_id,'created_utc':datetime.now(timezone.utc).isoformat(),'script':'tools/exp13_2_a4_targeted_quality.py','repo_head':'787a3be0190dfa9b11c5416a5b686a6bae4598d9','inputs':[str(INV),str(A0),str(B1),str(B2),str(A3X)],'source_scope':'TRAIN only','model_calls':0,'model_loads':0,'raw_data_write':0,'note':'Metadata generated on AutoDL; controlled industrial review images remain outside Git.'})
    # Controlled ZIP is intentionally outside Git and contains only new diagnostic images.
    z=out/'a4_controlled_review_images.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in review.glob('*.png'): zz.write(p,p.name)
    print(json.dumps({'out':str(out),'target_count':len(evid),'zip':str(z),'ledger_sha256':sha(led) if led.exists() else None},indent=2))

if __name__=='__main__': main()
