#!/usr/bin/env python3
"""H1 single historical failure case. One offline SDXL call; no retry/resume.

Historical defect-generation prompt is intentionally preserved. This diagnoses
hallucinations under that prompt, not pseudo-normal restoration effectiveness.
All image evidence stays on controlled AutoDL. Public export is JSON only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
RAW = Path('/root/autodl-tmp/损伤训练数据集')
HIST = REPO/'results/exp13_diffusion_diagnosis/r2_20260921_135903'
A0 = REPO/'results/exp13_restoration_feasibility/a0_20261021_081812'
M1 = REPO/'results/exp13_2_m1_matched_restoration_benchmark/20261026T064000Z_m1'
SPLIT = REPO/'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv'
SPLIT_SHA = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
START = '88cf8da39862d49bbf4c301b796a2c9fdebb9cae'
CASE = 'corrosion_23_mask_A_s055_v2'
REV = '115134f363124c53c7d878647567d04daf26e41e'
FP = 'd7fb8e6a3d8c4ac6538b13f2f2575a902662364c051716bd3961bf94a465fd14'
PROTOCOL_SHA = 'd8cd3b4e978f69e7f179dc8766d547fe119e43343ab64e6f88c054a0bd27f4cd'
SCRIPT = Path(__file__).resolve()
CODE = SCRIPT.parent.parent
sys.path.insert(0, str(CODE))
from tools.exp13_restoration.protocol import BudgetProtocol


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pix(im):
    return hashlib.sha256(np.asarray(im).tobytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def fit(p, mode):
    return ImageOps.fit(Image.open(p).convert(mode), (512,512),
                        method=Image.Resampling.LANCZOS if mode=='RGB' else Image.Resampling.NEAREST,
                        centering=(.5,.5))


def record(p):
    p = Path(p)
    r = {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
    if p.suffix.lower()=='.png':
        with Image.open(p) as im:
            r.update({'mode':im.mode,'size_wh':list(im.size),'pixel_sha256':pix(im)})
    return r


def old_untracked():
    # Stat every existing untracked file; do not open unrelated image content.
    names = subprocess.check_output(['git','-C',str(REPO),'ls-files','--others','--exclude-standard','-z']).decode().split('\0')
    out = {}
    for name in names:
        if not name: continue
        p=REPO/name
        if p.is_file():
            st=p.stat(); out[name]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    return out


def historic_records():
    row = next(r for r in rows(HIST/'manifests/diagnosis_manifest.csv') if r['diag_id']==CASE)
    assert row['status']=='SUCCESS' and row['source_id']=='23' and row['class']=='corrosion'
    assert [row[k] for k in ['strength','seed','inference_width','inference_height','num_inference_steps','guidance_scale']]==['0.55','17001','512','512','20','7.5']
    expected = {r['file_role']:r for r in rows(A0/'file_integrity.csv') if r['variant_id']==CASE}
    assert len(expected)==4
    out={}
    for key, e in expected.items():
        p=REPO/row[key]
        assert p.resolve().is_relative_to(HIST) and str(p.relative_to(REPO))==e['path']
        r=record(p)
        assert r['sha256']==e['sha256'] and r['bytes']==int(e['bytes'])
        assert r['size_wh']==[int(e['width']),int(e['height'])] and r['mode']==e['mode']
        out[key]=r
    assert out['output_path']['sha256']=='a2cbbc62211fd725404704f32fff002c421243159d16cc3cbbbd66f823c9f5c9'
    return row,out


def assert_budget(run):
    old=read(M1/'budget_successor_ledger.json')
    inherit=read(run/'budget_inheritance.json')
    assert sha(M1/'budget_successor_ledger.json')==inherit['parent_sha256']
    s=BudgetProtocol(ledger_path=run/'budget_successor_ledger.json').snapshot()
    assert all(s['calls'][k]==v for k,v in old['calls'].items())
    new=[v for k,v in s['calls'].items() if k not in old['calls']]
    assert len(new)<=1 and s['smoke_reserved']==4 and s['remaining']['smoke']==0
    assert s['gpt_fallback_calls']==s['retry_calls']==0
    assert all(c['stage']=='revision' and c['source_id']=='23' and not c['retry'] for c in new)
    return s,new


def preflight(run):
    run.mkdir(parents=True,exist_ok=False)
    (run/'inputs').mkdir()
    dump(run/'server_git_before.json',{'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
         'status':git('status','--porcelain=v1','--untracked-files=all').splitlines(),
         'untracked_stat_inventory':old_untracked()})
    snap=read(CODE/'snapshot_manifest.json')
    assert snap['head']==snap['remote_head']==START
    for rel,h in snap['files'].items(): assert sha(CODE/rel)==h
    assert sha(CODE/'tools/exp13_restoration/protocol.py')==PROTOCOL_SHA
    row,art=historic_records()
    assert sha(SPLIT)==SPLIT_SHA
    train={r['stem']:r for r in rows(SPLIT) if r['split']=='train'}
    tr=train['23']; assert tr['labels_present']=='corrosion'
    raw=RAW/tr['image_path']; annotation=RAW/tr['json_path']
    assert raw.resolve().is_relative_to(RAW) and annotation.resolve().is_relative_to(RAW)
    assert sha(raw)==tr['sha256']==art['source_image_path']['sha256']
    assert sha(annotation)==tr['json_sha256']
    rgb=fit(art['source_image_path']['path'],'RGB'); mask=fit(art['inpaint_mask_path']['path'],'L')
    # fit delegates to resize(box=...), whose LANCZOS support can sample
    # across the box boundary. A physically cropped image is NOT equivalent.
    box=(80,0,560,480)
    for role,mode,actual,method in [('source_image_path','RGB',rgb,Image.Resampling.LANCZOS),('inpaint_mask_path','L',mask,Image.Resampling.NEAREST)]:
        native=Image.open(art[role]['path']).convert(mode)
        assert native.size==(640,480)
        explicit=native.resize((512,512),method,box=box)
        assert pix(actual)==pix(explicit)
    rgb.save(run/'inputs/exact_historical_rgb_512.png')
    mask.save(run/'inputs/exact_historical_edit_mask_512.png')
    poly=fit(art['original_mask_path']['path'],'L'); poly.save(run/'inputs/historical_polygon_512.png')
    effective=Image.fromarray(((np.asarray(mask)>=128)*255).astype(np.uint8))
    effective.save(run/'inputs/effective_mask_512.png')
    prompts=read(HIST/'prompts/prompt_variants.json')
    assert prompts['v2']['corrosion']==row['prompt_text']
    evidence={'source_id':'23','variant_id':CASE,'historical_run':str(HIST),'historical_stage':'R2',
       'selection_source':'H1_CODEX_DIRECT_ARTIFACT_INSPECTION_WITH_ARCHIVED_SOURCE_LEVEL_REVIEW',
       'historical_source_level_evidence':['docs/exp13_1_r2_final_quality_gate_review.md','docs/exp13_2_a0_source_visual_screening_review.md'],
       'not_a_preexisting_variant_level_user_rating':True,
       'reason':'Visible invented cyan rectangular panel and circular wheel/lens-like mechanical components on the upper target band, absent in input; priority source23 yields a uniquely frozen exact manifest variant.',
       'historical_output_sha256':art['output_path']['sha256'],
       'observed_suspected_mechanical_bbox_canvas_xyxy':[55,135,415,255],
       'not_claimed_unique_hallucination_among_all_history':True,'second_case_allowed':False}
    dump(run/'case_identification.json',evidence)
    guard_files={str(p):sha(p) for p in [HIST/'manifests/diagnosis_manifest.csv',HIST/'prompts/prompt_variants.json',A0/'file_integrity.csv',SPLIT,M1/'budget_successor_ledger.json']}
    guard_files.update({r['path']:r['sha256'] for r in art.values()})
    guard_files.update({str(raw):sha(raw),str(annotation):sha(annotation)})
    dump(run/'historical_artifact_integrity.json',{'status':'PASS','artifacts':art,'historical_manifest_row':row,
         'a0_independent_hash_match':True,'protected_file_sha256_before':guard_files,
         'train_provenance':{'source_id':'23','split':'train','raw_sha256':sha(raw),'annotation_sha256':sha(annotation),
                             'split_manifest_sha256':SPLIT_SHA,'val_test_images_opened':0}})
    orig=Image.open(art['original_mask_path']['path']).convert('L')
    retained=int((np.asarray(orig.crop(box))>0).sum()); total=int((np.asarray(orig)>0).sum())
    dump(run/'input_reconstruction_verification.json',{'status':'VERIFIABLY_IDENTICAL_TO_SAVED_HISTORICAL_ALGORITHM',
         'historical_512_tensor_not_saved':True,'historical_native_assets_hash_verified':True,
         'operation':'ImageOps.fit center(.5,.5), RGB LANCZOS / mask NEAREST, 512x512',
         'independent_resize_box_pixel_equal':True,'crop_xyxy_half_open':box,'padding':0,
         'crop_then_resize_not_equivalent':'Rejected during initial CPU preflight; original ImageOps.fit retained unchanged.',
         'crop_target_native_pixels':retained,'total_target_native_pixels':total,
         'target_retention':retained/total,'polygon_complete':retained==total,
         'production_eligibility':'CASE_STUDY_ONLY / NOT_PRODUCTION_ELIGIBLE',
         'inputs':{p.name:record(p) for p in (run/'inputs').iterdir()},
         'approximate_reconstruction':False,'dependencies_current':{'Pillow':importlib.metadata.version('Pillow')},
         'limitation':'Historical tensor dump/runtime lock unavailable; identity is verified against saved assets and exact archived transform, not an unavailable serialized tensor.'})
    env={'timestamp':now(),'python':sys.version,'platform':platform.platform(),
       'components':{n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers','accelerate','numpy','Pillow']},
       'disk_free_bytes':shutil.disk_usage(REPO).free,
       'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],text=True).strip(),
       'available_ram_bytes':next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))}
    assert env['disk_free_bytes']>2*1024**3 and env['available_ram_bytes']>16*1024**3
    assert int(env['gpu'].split(',')[-1].strip().split()[0])>=10240
    dump(run/'environment_preflight.json',env)
    asset=read(M1/'model_integrity_preflight.json')['SDXL']
    assert asset['revision']==REV and asset['content_fingerprint']==FP
    root=Path(asset['local_path'])
    for f in asset['files']:
        p=root/f['path']; assert p.resolve().is_relative_to(root)
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert hashlib.sha256(json.dumps(asset['files'],sort_keys=True).encode()).hexdigest()==FP
    dump(run/'model_integrity_preflight.json',asset)
    config={'source_id':'23','variant_id':CASE,'requested_research_goal':'RESTORE_TO_PSEUDO_NORMAL_METAL',
       'executed_condition':'EXACT_HISTORICAL_DEFECT_GENERATION_PROMPT_REPLAY_FOR_HALLUCINATION_ENDPOINT',
       'task_mismatch_disclosed':True,'restoration_effectiveness_not_tested':True,
       'prompt':row['prompt_text'],'negative_prompt':prompts['negative']['corrosion'],
       'seed':17001,'steps':20,'guidance_scale':7.5,'strength':.55,'precision':'float16','height':512,'width':512,
       'sdxl_resolution_reason':'Exact same 512 input pixels prioritized over SDXL native1024; no upscale, padding, prompt or mask adaptation.',
       'model_id':asset['model_id'],'revision':REV,'model_fingerprint':FP,'scheduler':'EulerDiscreteScheduler',
       'cpu_offload':True,'vae_tiling':True,'vae_slicing':True,'vae_dtype_policy':'Unmodified model force_upcast policy; no fp16 VAE hypothesis tested.',
       'target_new_calls':1,'max_authorized_new_calls':2,'executable_new_calls':1,'dreamshaper_replay':False,
       'replay_not_needed':'Original output, input, mask, manifest and independent A0 hashes all exist.',
       'retry':False,'parameter_search':False,'production_eligible':False,
       'limitations':['single case','historical failure selected deliberately','defect-generation prompt conflicts with normal-metal goal',
                      'SDXL512 below native1024','architecture/runtime/scheduler differ','historical DreamShaper immutable upstream revision unknown'],
       'inputs':{p.name:record(p) for p in (run/'inputs').iterdir()}}
    dump(run/'frozen_execution_config.json',config)
    parent=M1/'budget_successor_ledger.json'; old=read(parent)
    assert old['occupied_calls']==old['actual_calls']==16 and len(old['calls'])==16
    assert old['remaining']['smoke']==0 and old['remaining']['smoke_pilot_revision']==104 and old['remaining']['all_local']==284
    assert all(c['status']=='SUCCEEDED' for c in old['calls'].values())
    dump(run/'budget_inheritance.json',{'parent_path':str(parent),'parent_sha256':sha(parent),'snapshot':old,'new_stage':'revision',
          'h1_executable_cap':1,'user_max_including_optional_ds':2,'optional_ds_not_used':True})
    successor=copy.deepcopy(old)
    successor['history'].append({'event':'H1_SUCCESSOR_INHERITS_M1','timestamp':now(),'parent_sha256':sha(parent),
                                'target_new_calls':1,'max_executable_new_calls':1,'stage':'revision'})
    dump(run/'budget_successor_ledger.json',successor)
    assert_budget(run)
    dump(run/'preflight_summary.json',{'status':'PASS','timestamp':now(),'script_sha256':sha(SCRIPT),
       'frozen_config_sha256':sha(run/'frozen_execution_config.json'),'snapshot':snap,'model_loads':0,'new_calls':0})
    print('H1 PREFLIGHT PASS; one frozen source23 R2 variant; calls0; DS replay unnecessary',flush=True)


def execute(run):
    with (run/'execution_process.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        dump(run/'execution_attempt.json',{'timestamp':now(),'pid':os.getpid(),'no_retry_or_resume':True})
        pre=read(run/'preflight_summary.json')
        assert pre['status']=='PASS' and pre['script_sha256']==sha(SCRIPT)
        assert pre['frozen_config_sha256']==sha(run/'frozen_execution_config.json')
        cfg=read(run/'frozen_execution_config.json'); _,art=historic_records()
        for r in cfg['inputs'].values(): assert record(r['path'])==r
        snap,new=assert_budget(run); assert len(new)==0
        for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']: os.environ[k]='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        log={'case':CASE,'source_id':'23','new_sdxl_calls':0,'new_dreamshaper_calls':0,'status':'LOADING','started_at':now()}
        metrics={}; load={'status':'STARTED','model_fingerprint':FP,'revision':REV}
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline
            assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=10*1024**3
            torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
            asset=read(run/'model_integrity_preflight.json')
            # Recheck every component immediately before loading; offline only.
            for f in asset['files']:
                assert sha(Path(asset['local_path'])/f['path'])==f['sha256']
            pipe=StableDiffusionXLInpaintPipeline.from_pretrained(asset['local_path'],torch_dtype=torch.float16,
                    variant='fp16',use_safetensors=True,local_files_only=True)
            pipe.enable_model_cpu_offload(gpu_id=0); pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
            assert type(pipe.scheduler).__name__==cfg['scheduler']
            pipe.set_progress_bar_config(disable=True)
            load.update({'status':'SUCCEEDED','seconds':time.perf_counter()-t,'pipeline':type(pipe).__name__,
               'scheduler':type(pipe.scheduler).__name__,'scheduler_config':dict(pipe.scheduler.config),
               'component_dtypes':{k:str(getattr(v,'dtype',None)) for k,v in pipe.components.items()},
               'vae_force_upcast':pipe.vae.config.force_upcast,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            dump(run/'model_load.json',load)
            inp=Image.open(run/'inputs/exact_historical_rgb_512.png').convert('RGB')
            mask=Image.open(run/'inputs/exact_historical_edit_mask_512.png').convert('L')
            assert inp.size==mask.size==(512,512)
            processed=pipe.mask_processor.preprocess(mask,height=512,width=512).numpy()[0,0]
            assert np.array_equal(processed,np.asarray(mask)>=128)
            generator=torch.Generator(device='cuda').manual_seed(cfg['seed'])
            budget=BudgetProtocol(ledger_path=run/'budget_successor_ledger.json')
            cid='h1_'+CASE+'_SDXL_'+pre['frozen_config_sha256'][:16]
            rec=budget.reserve_call(call_id=cid,source_id='23',roi_id='h1_exact_historical_512',stage='revision',
                   strength=.55,model_config_id='h1_'+pre['frozen_config_sha256'])
            assert rec['status']=='RESERVED' and len(assert_budget(run)[1])==1
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            budget.start_call(cid,real_model_call=True)
            assert budget.snapshot()['calls'][cid]['status']=='STARTED'
            log.update({'status':'STARTED','call_id':cid,'new_sdxl_calls':1,'started_persisted_before_inference':True})
            dump(run/'call_started_evidence.json',{'timestamp':now(),'call':budget.snapshot()['calls'][cid]})
            t=time.perf_counter()
            try:
                with torch.inference_mode():
                    result=pipe(prompt=cfg['prompt'],negative_prompt=cfg['negative_prompt'],image=inp,mask_image=mask,
                         height=512,width=512,num_inference_steps=20,guidance_scale=7.5,strength=.55,
                         generator=generator,output_type='np')
                torch.cuda.synchronize(); a=np.asarray(result.images[0])
                metrics={'inference_seconds':time.perf_counter()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'input_wh':[512,512],'mask_wh':[512,512],
                   'output_float_shape':list(a.shape),'output_float_dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),
                   'mask_polarity':'white>=128 means edit; checked against actual Diffusers mask_processor',
                   'vae_dtype_after':str(pipe.vae.dtype)}
                np.save(run/'sdxl_raw_decoded_float.npy',a,allow_pickle=False)
                assert metrics['finite'] and list(a.shape)==[512,512,3]
                out=(a*255).round().clip(0,255).astype(np.uint8)
                Image.fromarray(out).save(run/'sdxl_raw_output_512.png')
                assert out.mean()>2 and out.std()>0, 'BLACK_OR_CONSTANT_OUTPUT'
                metrics.update({'float_min':float(a.min()),'float_max':float(a.max()),'float_mean':float(a.mean()),
                                'output_uint8_dtype':str(out.dtype),'output_wh':[512,512]})
                budget.finish_call(cid,success=True); log.update({'status':'SUCCEEDED','finished_at':now()})
                print('H1 SINGLE SDXL CALL SUCCEEDED',flush=True)
            except BaseException as exc:
                budget.finish_call(cid,success=False,error=f'{type(exc).__name__}: {exc}')
                log.update({'status':'FAILED','error':traceback.format_exc()}); raise
        except BaseException:
            log.update({'status':'EXECUTION_HOLD','error':traceback.format_exc(),'finished_at':now()})
            raise
        finally:
            dump(run/'new_call_execution_log.json',log)
            dump(run/'runtime_metrics.json',metrics)
            if not (run/'model_load.json').exists():
                load['error']=log.get('error'); dump(run/'model_load.json',load)
            assert_budget(run)


def sheet(panels,path):
    out=Image.new('RGB',(1536,50+550*((len(panels)+2)//3)),'#eeeeee'); d=ImageDraw.Draw(out)
    d.text((10,10),'H1 SOURCE23 / HISTORICAL CASE ONLY / NOT MODEL RANKING / HUMAN REVIEW PENDING',fill='black')
    for i,(title,im) in enumerate(panels):
        x=(i%3)*512; y=50+(i//3)*550
        d.text((x+8,y+5),title,fill='black')
        fitted=ImageOps.contain(im.convert('RGB'),(504,504))
        out.paste(fitted,(x+(512-fitted.width)//2,y+32+(504-fitted.height)//2))
    out.save(path)


def finalize(run):
    assert read(run/'new_call_execution_log.json')['status']=='SUCCEEDED'
    cfg=read(run/'frozen_execution_config.json'); row,art=historic_records()
    rgb=Image.open(run/'inputs/exact_historical_rgb_512.png').convert('RGB')
    mask=Image.open(run/'inputs/exact_historical_edit_mask_512.png').convert('L')
    ds=Image.open(art['output_path']['path']).convert('RGB')
    sd=Image.open(run/'sdxl_raw_output_512.png').convert('RGB')
    effective=np.asarray(mask)>=128; base=np.asarray(rgb)
    package=run/'review_package'; package.mkdir(exist_ok=False)
    native=Image.open(art['source_image_path']['path']).convert('RGB')
    native.save(package/'source_native.png'); rgb.save(package/'exact_input_512.png')
    mask.save(package/'historical_edit_mask_512.png')
    Image.open(run/'inputs/effective_mask_512.png').save(package/'effective_mask_512.png')
    ds.save(package/'historical_DreamShaper.png'); sd.save(package/'new_SDXL.png')
    views={}; panels=[('Native source (crop x80..560 y0..480)',native),('Exact common historical input 512',rgb),('Historical edit mask (soft grayscale)',mask),
                     ('DreamShaper historical RAW output',ds),('SDXL new RAW output (one call)',sd),('Effective edit mask (white edits)',Image.fromarray((effective*255).astype(np.uint8)))]
    bbox=(55,135,415,255)
    for name,im in [('Input',rgb),('DreamShaper',ds),('SDXL',sd)]:
        ann=im.copy(); ImageDraw.Draw(ann).rectangle(bbox,outline='red',width=2)
        ann.save(package/f'{name}_annotation.png')
        zoom=im.crop(bbox).resize((1440,480),Image.Resampling.NEAREST)
        zoom.save(package/f'{name}_mechanical_region_zoom_NEAREST.png')
        panels.append((f'{name} annotated suspicious band',ann))
    for name,im in [('DreamShaper',ds),('SDXL',sd)]:
        arr=np.asarray(im); delta=np.abs(arr.astype(np.int16)-base.astype(np.int16))
        dm=delta.max(axis=2); changed=np.any(arr!=base,axis=2)
        diff=Image.fromarray(np.minimum(dm*4,255).astype(np.uint8))
        diff.save(package/f'{name}_absolute_difference_x4.png')
        protected=base.copy(); protected[effective]=arr[effective]
        Image.fromarray(protected).save(package/f'{name}_protected_POSTPROCESS_ONLY.png')
        views[name]={'changed_outside_effective_mask_pixels':int((changed&~effective).sum()),
             'changed_gt5_outside_effective_mask_pixels':int(((dm>5)&~effective).sum()),
             'outside_effective_mask_mean_abs_channel_delta':float(delta[~effective].mean()),
             'inside_effective_mask_mean_abs_channel_delta':float(delta[effective].mean()),
             'max_channel_delta_outside_effective_mask':int(delta[~effective].max()),
             'protected_outside_effective_mask_changes':int(np.any(protected!=base,axis=2)[~effective].sum()),
             'raw_output':record(package/('historical_DreamShaper.png' if name=='DreamShaper' else 'new_SDXL.png'))}
        panels.append((f'{name} absolute difference x4',diff))
    panels.append(('Input local detail, nearest pixels',rgb.crop(bbox).resize((504,168),Image.Resampling.NEAREST)))
    sheet(panels,package/'H1_comparison_contact_sheet.png')
    with (package/'human_review_template.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['SOURCE_ID','VARIANT','SDXL_SIMILAR_MECHANICAL_HALLUCINATION','SDXL_DEFECT_REDUCED','METAL_TEXTURE_NATURAL',
                'REAL_GEOMETRY_PRESERVED','ERROR_SAME_TYPE','REVIEWER_COMMENT']
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerow(dict(zip(fields,['23',CASE]+['PENDING']*6)))
    dump(package/'case_and_protocol.json',{'case':read(run/'case_identification.json'),'config':cfg,
         'note':'CASE_STUDY_ONLY / NOT_PRODUCTION_ELIGIBLE. Historical prompt explicitly asks to GENERATE corrosion; this is not a normal-metal restoration test. DS and SDXL architecture, scheduler and runtime differ. No ranking or general inference.'})
    protected=read(run/'historical_artifact_integrity.json')['protected_file_sha256_before']
    assert all(sha(p)==h for p,h in protected.items())
    before=read(run/'server_git_before.json')
    changed=[]
    for rel,st in before['untracked_stat_inventory'].items():
        p=REPO/rel
        if not p.is_file() or {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}!=st: changed.append(rel)
    assert not changed
    assert git('rev-parse','HEAD')==before['head']
    tracked_before=[s for s in before['status'] if not s.startswith('?? ')]
    tracked_now=[s for s in git('status','--porcelain=v1','--untracked-files=all').splitlines() if not s.startswith('?? ')]
    assert tracked_before==tracked_now
    snap,new=assert_budget(run)
    assert len(new)==1 and new[0]['status']=='SUCCEEDED' and snap['actual_calls']==snap['occupied_calls']==17
    dump(run/'budget_preservation.json',{'parent_unchanged':True,'inherited_calls_preserved':16,'h1_new_actual_calls':1,
       'new_dreamshaper_calls':0,'new_sdxl_calls':1,'new_failures':0,'new_retries':0,'smoke_occupied':4,'smoke_remaining':0,
       'cumulative_occupied':17,'cumulative_actual':17,'remaining':snap['remaining'],'gpt_fallback':0,
       'ledger_path':str(run/'budget_successor_ledger.json'),'ledger_sha256':sha(run/'budget_successor_ledger.json')})
    dump(run/'data_git_preservation.json',{'historical_protected_hashes_equal':True,'protected_file_count':len(protected),
       'previous_untracked_file_count':len(before['untracked_stat_inventory']),'untracked_stat_changes':changed,
       'unrelated_untracked_verification':'path,size,mtime; contents not opened; selected historical artifacts and parent ledger SHA256 checked',
       'server_head_before':before['head'],'server_head_after':git('rev-parse','HEAD'),'tracked_changes_preserved':True,
       'val_test_image_access':0,'raw_data_writes':0,'split_writes':0,'downloads':0,'training':0,'trusted_pairs_frozen':0,
       'paper_repository_access':0})
    dump(run/'comparison_summary.json',{'case':CASE,'source_id':'23','status':'PENDING_HUMAN_REVIEW',
       'primary_endpoint':'SDXL suspected newly invented mechanical structures of the same kind as historical DreamShaper',
       'automatic_semantic_verdict':'NOT_ASSIGNED','measurements':views,'bbox_canvas_xyxy':bbox,
       'conclusion_scope':'one selected historical failure at seed17001; not general model comparison or pseudo-normal efficacy',
       'task_mismatch_disclosed':True,'native_inverse_mapping':'NOT_PERFORMED; compare exact512 canvases only',
       'protected_composites':'POSTPROCESS_DIAGNOSTIC_ONLY; no trusted-pair admission'})
    zi=run/'H1_hallucination_case_review.zip'
    with zipfile.ZipFile(zi,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(package))
    dump(run/'review_package_manifest.json',{'zip_path':str(zi),'zip_bytes':zi.stat().st_size,'zip_sha256':sha(zi),
       'image_count':len(list(package.glob('*.png'))),'contact_sheet_count':1,
       'files':[record(p) for p in sorted(package.iterdir()) if p.is_file()],'controlled_server_only':True})
    dump(run/'summary.json',{'status':'EXP13_2_H1_COMPLETED_PENDING_CHATGPT_GATE','source_id':'23','variant_id':CASE,
       'new_calls':1,'sdxl_calls':1,'dreamshaper_replay_calls':0,'successes':1,'failures':0,'retries':0,
       'semantic_quality':'PENDING_USER_AND_CHATGPT_REVIEW','trusted_pairs':0,'production_eligible':False,
       'task_mismatch':'historical corrosion-generation prompt replayed; not normal-metal restoration efficacy',
       'cumulative_budget_calls':17,'smoke_remaining':0,'aggregate_remaining':103,'all_local_remaining':283,
       'run_directory':str(run),'review_zip':str(zi),'review_zip_bytes':zi.stat().st_size,'review_zip_sha256':sha(zi),
       'finished_at':now()})
    dump(run/'provenance.json',{'report_start_head':START,'server_execution_checkout_head':git('rev-parse','HEAD'),
       'execution_snapshot':read(CODE/'snapshot_manifest.json'),'script_sha256':sha(SCRIPT),'protocol_sha256':PROTOCOL_SHA,
       'historical_artifacts':art,'source_case':CASE,'frozen_config_sha256':sha(run/'frozen_execution_config.json'),
       'authoritative_ledger':str(run/'budget_successor_ledger.json'),'parent_ledger':str(M1/'budget_successor_ledger.json'),
       'model_id':cfg['model_id'],'model_revision':REV,'model_fingerprint':FP,'stdout_stderr_exit_capture':'separate supervisor subprocess files; no pipe exit-code ambiguity',
       'train_only':True,'new_calls':1,'maximum_executable_calls':1,'post_run_next_stage_authorized':False})
    export=run/'public_export'; export.mkdir()
    # Deliberately never export image pixels, decoded arrays, ZIP or review assets.
    for p in run.glob('*.json'):
        if p.name=='server_git_before.json': continue
        shutil.copy2(p,export/p.name)
    print(json.dumps(read(run/'summary.json')),flush=True)


def supervise(run):
    # Exclusive logs/exit evidence and attempt guard prevent accidental reruns.
    dump(run/'supervisor_attempt.json',{'timestamp':now(),'no_retry':True})
    with (run/'execution_stdout.log').open('x') as out, (run/'execution_stderr.log').open('x') as err:
        proc=subprocess.run([sys.executable,str(SCRIPT),'--mode','execute','--run',str(run)],stdout=out,stderr=err)
    dump(run/'execution_exit.json',{'returncode':proc.returncode,'timestamp':now(),'method':'subprocess.run returncode; separate stdout/stderr files'})
    return proc.returncode


def main():
    a=argparse.ArgumentParser(); a.add_argument('--run',type=Path,required=True)
    a.add_argument('--mode',choices=['preflight','execute','supervise','finalize'],required=True)
    args=a.parse_args()
    assert args.run.resolve().is_relative_to(REPO/'results/exp13_2_h1_hallucination_replay')
    if args.mode=='supervise': sys.exit(supervise(args.run))
    {'preflight':preflight,'execute':execute,'finalize':finalize}[args.mode](args.run)


if __name__=='__main__':
    main()
