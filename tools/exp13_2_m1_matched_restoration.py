#!/usr/bin/env python3
"""Bounded M1: immutable M0-R1 inputs, one locked successor, no retry/resume.

Run preflight first. Execute is supervised with separate stdout/stderr/exit files.
All image assets and the precommitted blind key stay on controlled AutoDL.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
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
M0 = REPO/'results/exp13_2_m0_matched_model_benchmark/20261025T115110Z_m0'
R1 = REPO/'results/exp13_2_m0_r1/20261025T122430Z_m0r1'
S5 = REPO/'results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json'
SPLIT = REPO/'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv'
SPLIT_SHA = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
S5_SHA = '3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94'
START = '01b730cd0471f982d5d17263cc254b04bb937da8'
SELECTION = [('469','Crack','alt_2'), ('260','Crack','default'),
             ('470','Crack','default'), ('8','corrosion','default'),
             ('1','corrosion','alt_1'), ('2','corrosion','alt_1')]
EXPECTED_BOXES = [[240,0,960,720],[90,0,630,540],[120,0,840,720],
                  [90,0,630,540],[48,0,588,540],[0,0,480,480]]
MODELS = ['DreamShaper','SDXL']
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
    """New evidence only, durable exclusive creation; ledger uses its own fsync."""
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def csvout(p, rows, fields=None):
    with Path(p).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def pix(im):
    return hashlib.sha256(np.asarray(im).tobytes()).hexdigest()


def verify(rec):
    p = (REPO/rec['path']).resolve()
    assert p.is_relative_to(M0) or p.is_relative_to(R1), p
    assert sha(p) == rec['file_sha256'], p
    with Image.open(p) as im:
        a = np.asarray(im)
        assert pix(im) == rec['pixel_sha256'] and im.mode == rec['mode'], p
        assert list(a.shape) == rec['shape'] and str(a.dtype) == rec['dtype'], p


def image(rec):
    verify(rec)
    with Image.open(REPO/rec['path']) as im:
        return im.copy()


def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args], text=True).strip()


def inventory():
    roots = [M0, R1]
    return {str(p.relative_to(REPO)): sha(p) for root in roots for p in root.rglob('*') if p.is_file()}


def verify_inputs():
    assert sha(SPLIT) == SPLIT_SHA and sha(S5) == S5_SHA
    assert sha(R1/'human_input_protocol_review_manifest.json') == '30ec2545755412be969bb4ef3791e547635ff1126343637ef99ccf945c805de7'
    candidates = read(R1/'human_input_protocol_review_manifest.json')['all_sources']
    with SPLIT.open(encoding='utf-8-sig', newline='') as f:
        # The manifest is metadata; only TRAIN image/annotation paths are opened.
        train = {r['stem']:r for r in csv.DictReader(f) if r['split']=='train'}
    selected = []
    for (sid, cls, cid), box in zip(SELECTION, EXPECTED_BOXES):
        row = train[sid]
        s = next(s for s in candidates if s['source_id']==sid)
        opt = next(o for o in s['options'] if o['crop_id']==cid)
        assert row['labels_present']==cls==s['class']
        assert s['source_sha256']==row['sha256'] and s['annotation_sha256']==row['json_sha256']
        raw_paths = [RAW/row['image_path'],RAW/row['json_path']]
        assert all(p.resolve().is_relative_to(RAW) for p in raw_paths)
        assert [sha(p) for p in raw_paths] == [row['sha256'],row['json_sha256']]
        assert opt['box_xyxy_half_open']==box and opt['geometry_checks']['technical_pass']
        assert opt['geometry_checks']['target_retention']==1 and opt['geometry_checks']['edit_support_retention']==1
        for rec in opt['files'].values():
            verify(rec)
        fs = opt['files']
        rgb, mask = image(fs['common_rgb_512']), image(fs['common_edit_mask_512'])
        assert np.array_equal(np.asarray(rgb.resize((1024,1024),Image.Resampling.LANCZOS)), np.asarray(image(fs['sdxl_rgb_1024'])))
        assert np.array_equal(np.asarray(mask.resize((1024,1024),Image.Resampling.NEAREST)), np.asarray(image(fs['sdxl_mask_1024'])))
        for size,key,effkey in [(512,'common_edit_mask_512','effective_edit_mask_512'),(1024,'sdxl_mask_1024','effective_edit_mask_1024')]:
            assert np.array_equal((np.asarray(image(fs[key]))>=128)*255,np.asarray(image(fs[effkey])))
        with Image.open(raw_paths[0]) as native:
            assert pix(native.convert('RGB'))==fs['native_rgb']['pixel_sha256']
            assert pix(native.convert('RGB').crop(box))==fs['native_crop']['pixel_sha256']
        selected.append({'source_id':sid,'class':cls,'crop_id':cid,'crop_xyxy_half_open':box,
                         'decision':'PASS','decision_source':'USER_EXPLICIT_M1_PROMPT',
                         'risk':'HIGH_RISK_BUT_USER_ACCEPTED' if sid in {'1','2'} else 'USER_ACCEPTED',
                         'source_sha256':row['sha256'],'annotation_sha256':row['json_sha256'],
                         'raw_paths':list(map(str,raw_paths)), 'group_id':row['group_id'],
                         'files':fs,'geometry_checks':opt['geometry_checks']})
    assert len(selected)==6 and len({s['source_sha256'] for s in selected})==6
    return selected


def protocol_check():
    assert sha(M0/'frozen_model_comparison_protocol.json')=='949c863df202ce15e39cc86495098727f669f4d3779110fcac7b8d61ea4de8e7'
    assert sha(M0/'model_asset_preflight.json')=='7bca788c1e65419958c7a8b945d5fc93efd61a77b3393c44faf7a9f6300ac26f'
    proto = read(M0/'frozen_model_comparison_protocol.json')
    assert [proto[k] for k in ['negative_prompt','steps','guidance_scale','strength','seed','precision']]==[None,20,7.5,.8,130500,'float16']
    assert proto['prompt']=='restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting'
    assets = read(M0/'model_asset_preflight.json')
    for name,m in assets.items():
        root = Path(m['local_path']).resolve()
        for f in m['files']:
            p = (root/f['path']).resolve()
            assert p.is_relative_to(root) and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'], p
        assert hashlib.sha256(json.dumps(m['files'],sort_keys=True).encode()).hexdigest()==m['content_fingerprint']==proto['assets'][name]['fingerprint']
    for f in read(M0/'cpu_validation.json')['installed_diffusers_source']:
        assert sha(f['path'])==f['sha256']
    return proto, assets


def preflight(run):
    run.mkdir(parents=True, exist_ok=False)
    dump(run/'server_git_before.json', {'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
                                       'status':git('status','--porcelain=v1','--untracked-files=all').splitlines()})
    dump(run/'historical_artifact_inventory.json',inventory())
    code_manifest = read(CODE/'snapshot_manifest.json')
    assert code_manifest['head']==START and code_manifest['remote_head']==START
    for rel,h in code_manifest['files'].items():
        assert sha(CODE/rel)==h, rel
    selected = verify_inputs()
    proto, assets = protocol_check()
    old = read(S5)
    assert old['occupied_calls']==old['actual_calls']==len(old['calls'])==4
    assert old['remaining']['smoke']==0 and old['remaining']['smoke_pilot_revision']==116 and old['remaining']['all_local']==296
    assert old['gpt_fallback_calls']==0 and all(c['status']=='SUCCEEDED' for c in old['calls'].values())
    env = {'timestamp':now(),'python':sys.version,'platform':platform.platform(),
           'components':{n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers','accelerate','numpy','Pillow']},
           'disk_free_bytes':shutil.disk_usage(REPO).free,
           'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],text=True).strip(),
           'available_ram_bytes':next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')),
           'model_loads':0,'real_calls':0,'network_downloads':0}
    assert env['disk_free_bytes']>2*1024**3 and env['available_ram_bytes']>16*1024**3
    free_mib=int(env['nvidia_smi'].split(',')[-1].strip().split()[0])
    assert free_mib>=10240
    dump(run/'user_input_decision_overlay.json', {'source':'USER_EXPLICIT_M1_PROMPT','recorded_at':now(),
        'input_gate':'PASS / BOUNDED','authorization':'YES / BOUNDED','scope':'INPUT_ONLY_NOT_GENERATED_QUALITY',
        'decisions':[{k:s[k] for k in ['source_id','class','crop_id','crop_xyxy_half_open','decision','decision_source','risk']} for s in selected],
        'pending_history_preserved':True,'objective_technical_errors_not_waived':True})
    dump(run/'frozen_source_crop_manifest.json',selected)
    dump(run/'model_integrity_preflight.json',assets)
    dump(run/'frozen_execution_config.json', {'m0_protocol':proto,'m1_authority':'USER_M1_PROMPT_12_CALLS_NO_RETRY',
         'execution_order':'DreamShaper six sources in frozen order; unload; SDXL six same sources',
         'pair_reservation':'atomic both models before first model call for each source',
         'output_type':'np, finite floating decode inspected before standard rounded uint8 PNG conversion',
         'stop_on_inference_failure':True,'config_sha_binding':'this file SHA256 used in every call record',
         'practical_comparison_not_pure_weight_single_variable':True})
    dump(run/'environment_preflight.json',env)
    dump(run/'budget_inheritance_preflight.json', {'path':str(S5),'sha256':S5_SHA,'snapshot':old,'new_calls_cap':12})
    # Precommit independent random A/B labels before any generated output exists.
    private = run/'private'; private.mkdir(mode=0o700)
    mapping = {sid:dict(zip(['A','B'],secrets.SystemRandom().sample(MODELS,2))) for sid,_,_ in SELECTION}
    key = private/'blind_mapping_key.json'
    dump(key,{'created_at':now(),'randomization':'independent SystemRandom permutation per source','mapping':mapping})
    key.chmod(0o600)
    dump(run/'blind_mapping_commitment.json',{'timestamp':now(),'sha256':sha(key),'private_key_location':'controlled server private directory; excluded from public Git and blind ZIP',
                                            'committed_before_model_load_and_inference':True})
    dump(run/'preflight_summary.json',{'status':'PASS','timestamp':now(),'source_count':6,'approved_inputs':6,
                                     'new_calls':0,'model_loads':0,'max_authorized_calls':12,'script_sha256':sha(SCRIPT),
                                     'snapshot':code_manifest})
    print('PREFLIGHT_PASS: six accepted frozen inputs, both weight sets and S5 verified',flush=True)


def assert_budget(snapshot):
    old=read(S5)
    assert sha(S5)==S5_SHA
    assert all(snapshot['calls'][k]==v for k,v in old['calls'].items())
    new=[v for k,v in snapshot['calls'].items() if k not in old['calls']]
    assert len(new)<=12 and snapshot['smoke_reserved']==4 and snapshot['remaining']['smoke']==0
    assert snapshot['retry_calls']==snapshot['gpt_fallback_calls']==0
    assert all(c['stage']=='pilot' and not c['retry'] for c in new)
    return new


def execute_worker(run):
    with (run/'execution_process.lock').open('a+b') as lf:
        fcntl.flock(lf.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        dump(run/'execution_attempt.json',{'started_at':now(),'pid':os.getpid(),'no_resume_or_retry':True})
        assert read(run/'preflight_summary.json')['status']=='PASS'
        assert read(run/'preflight_summary.json')['script_sha256']==sha(SCRIPT)
        selected=verify_inputs()
        assert selected==read(run/'frozen_source_crop_manifest.json')
        config=read(run/'frozen_execution_config.json'); proto=config['m0_protocol']
        assets=read(run/'model_integrity_preflight.json')
        config_id='exp13_2_m1_'+sha(run/'frozen_execution_config.json')
        old=read(S5); assert sha(S5)==S5_SHA
        successor=run/'budget_successor_ledger.json'
        payload=copy.deepcopy(old)
        payload['history'].append({'event':'M1_SUCCESSOR_INHERITS_S5','timestamp':now(),'source_sha256':S5_SHA,'source_ledger':str(S5),'new_call_authorization':12,'stage':'pilot'})
        dump(successor,payload)
        budget=BudgetProtocol(ledger_path=successor)
        assert_budget(budget.snapshot())
        for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:
            os.environ[k]='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        import torch
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
        metrics=[]; loads=[]
        runtime={'started_at':now(),'status':'RUNNING'}
        try:
            assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=10*1024**3
            for name in MODELS:
                load={'model':name,'attempted_at':now(),'success':False}
                loads.append(load)
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                if name=='DreamShaper':
                    pipe=StableDiffusionInpaintPipeline.from_pretrained(assets[name]['local_path'],torch_dtype=torch.float16,
                           use_safetensors=True,safety_checker=None,local_files_only=True)
                    pipe=pipe.to('cuda'); pipe.enable_attention_slicing()
                else:
                    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(assets[name]['local_path'],torch_dtype=torch.float16,
                           variant='fp16',use_safetensors=True,local_files_only=True)
                    pipe.enable_model_cpu_offload(gpu_id=0); pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
                assert type(pipe.scheduler).__name__==proto['scheduler'][name]
                load.update({'success':True,'seconds':time.perf_counter()-start,'pipeline':type(pipe).__name__,
                             'scheduler':type(pipe.scheduler).__name__,'scheduler_config':dict(pipe.scheduler.config),
                             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                             'component_dtypes':{k:str(getattr(v,'dtype',None)) for k,v in pipe.components.items()},
                             'local_fingerprint':assets[name]['content_fingerprint'],'revision':assets[name].get('revision',assets[name].get('immutable_remote_revision'))})
                dump(run/f'model_load_{name}.json',load)
                pipe.set_progress_bar_config(disable=True)
                for s in selected:
                    sid=s['source_id']; fs=s['files']; size=proto['resolution'][name]
                    # No images are rebuilt: only immutable M0-R1 assets are read.
                    inp=image(fs['common_rgb_512' if name=='DreamShaper' else 'sdxl_rgb_1024'])
                    mask=image(fs['common_edit_mask_512' if name=='DreamShaper' else 'sdxl_mask_1024'])
                    assert inp.size==mask.size==(size,size)
                    assert np.array_equal(pipe.mask_processor.preprocess(mask,height=size,width=size).numpy()[0,0],np.asarray(mask)>=128)
                    ids=[f'm1_{m}_{s["crop_id"]}' for m in MODELS]
                    if name=='DreamShaper':
                        assert len(assert_budget(budget.snapshot()))<=10
                        records=budget.reserve_source_group(source_id=sid,roi_ids=ids,stage='pilot',strength=.8,model_config_id=config_id)
                        assert len(records)==2 and all(r['status']=='RESERVED' for r in records)
                    snap=budget.snapshot(); assert_budget(snap)
                    cid=f'{sid}:m1_{name}_{s["crop_id"]}:s0.8'
                    assert snap['calls'][cid]['status']=='RESERVED'
                    outdir=run/'outputs'/sid/name; outdir.mkdir(parents=True,exist_ok=False)
                    rec={'call_id':cid,'source_id':sid,'model':name,'crop_id':s['crop_id'],'status':'STARTED',
                         'input_wh':[size,size],'input_rgb_sha256':fs['common_rgb_512' if name=='DreamShaper' else 'sdxl_rgb_1024']['file_sha256'],
                         'input_mask_sha256':fs['common_edit_mask_512' if name=='DreamShaper' else 'sdxl_mask_1024']['file_sha256'],
                         'mask_effective_pixels':int((np.asarray(mask)>=128).sum()),'started_at':now()}
                    generator=torch.Generator(device='cuda').manual_seed(130500)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    budget.start_call(cid,real_model_call=True)
                    assert budget.snapshot()['calls'][cid]['status']=='STARTED'
                    t=time.perf_counter()
                    try:
                        with torch.inference_mode():
                            result=pipe(prompt=proto['prompt'],negative_prompt=None,image=inp,mask_image=mask,
                                height=size,width=size,num_inference_steps=20,guidance_scale=7.5,strength=.8,generator=generator,
                                output_type='np')
                        torch.cuda.synchronize()
                        a=np.asarray(result.images[0])
                        rec.update({'seconds':time.perf_counter()-t,'output_float_shape':list(a.shape),'output_float_dtype':str(a.dtype),
                                    'finite':bool(np.isfinite(a).all()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                        np.save(outdir/'raw_decoded_float.npy',a,allow_pickle=False)
                        assert rec['finite'] and list(a.shape)==[size,size,3]
                        rec.update({'float_min':float(a.min()),'float_max':float(a.max()),'float_mean':float(a.mean())})
                        raw=Image.fromarray((a*255).round().clip(0,255).astype(np.uint8))
                        raw.save(outdir/'raw_output.png')
                        arr=np.asarray(raw)
                        assert arr.std()>0 and arr.mean()>=2, 'BLACK_OR_CONSTANT_OUTPUT'
                        aligned=raw if size==512 else raw.resize((512,512),Image.Resampling.LANCZOS)
                        aligned.save(outdir/'aligned_512.png')
                        base=np.asarray(image(fs['common_rgb_512']))
                        aa=np.asarray(aligned); support=np.asarray(image(fs['effective_edit_mask_512']))>0
                        delta=np.abs(aa.astype(np.int16)-base.astype(np.int16)).max(axis=2)
                        Image.fromarray(delta.astype(np.uint8)).save(outdir/'difference_512.png')
                        # Hard support protection applied equally to aligned outputs; diagnostic only.
                        protected=base.copy(); protected[support]=aa[support]
                        Image.fromarray(protected).save(outdir/'protected_comparison_only_512.png')
                        rec.update({'output_wh':list(raw.size),'output_uint8_shape':list(arr.shape),'output_uint8_dtype':str(arr.dtype),
                                    'changed_gt5_inside_effective_mask':int(((delta>5)&support).sum()),
                                    'changed_gt5_outside_effective_mask':int(((delta>5)&~support).sum()),
                                    'protected_outside_effective_mask_changes':int(np.any(protected!=base,axis=2)[~support].sum()),
                                    'status':'SUCCEEDED','finished_at':now(),'semantic_quality':'PENDING_USER_BLIND_REVIEW'})
                        budget.finish_call(cid,success=True)
                        print(f'CALL SUCCEEDED {sid} {name} {rec["seconds"]:.3f}s',flush=True)
                    except BaseException as exc:
                        rec.update({'status':'FAILED','finished_at':now(),'seconds':time.perf_counter()-t,'error':traceback.format_exc()})
                        budget.finish_call(cid,success=False,error=f'{type(exc).__name__}: {exc}')
                        raise
                    finally:
                        metrics.append(rec); dump(outdir/'runtime.json',rec)
                del pipe; gc.collect(); torch.cuda.empty_cache()
            runtime.update({'status':'SUCCEEDED','finished_at':now()})
        except BaseException:
            runtime.update({'status':'TECHNICAL_HOLD','finished_at':now(),'error':traceback.format_exc()})
            raise
        finally:
            dump(run/'execution_runtime.json',runtime)
            dump(run/'call_runtime_records.json',metrics)
            dump(run/'model_load_records.json',loads)
            assert_budget(budget.snapshot())


def sheet(panels,path,heading):
    width=3*512; ph=550
    out=Image.new('RGB',(width,60+ph*((len(panels)+2)//3)),'#eeeeee')
    d=ImageDraw.Draw(out); d.text((12,12),heading,fill='black')
    for i,(title,im) in enumerate(panels):
        x=(i%3)*512; y=60+(i//3)*ph
        d.text((x+8,y+4),title,fill='black')
        fitted=ImageOps.contain(im.convert('RGB'),(504,504))
        out.paste(fitted,(x+(512-fitted.width)//2,y+30+(504-fitted.height)//2))
    out.save(path)


def finalize(run):
    """CPU-only existing outputs; cannot import/load/call diffusion models."""
    selected=read(run/'frozen_source_crop_manifest.json')
    snap=BudgetProtocol(ledger_path=run/'budget_successor_ledger.json').snapshot()
    calls=assert_budget(snap)
    key=run/'private/blind_mapping_key.json'
    assert sha(key)==read(run/'blind_mapping_commitment.json')['sha256']
    mapping=read(key)['mapping']
    package=run/'blind_review_package'; package.mkdir(exist_ok=False)
    private_inventory=[]; review=[]; complete=[]
    for s in selected:
        sid=s['source_id']; fs=s['files']
        base=image(fs['common_rgb_512']); mask=image(fs['common_edit_mask_512'])
        support=image(fs['effective_edit_mask_512'])
        poly=image(fs['native_polygon_crop']).resize((512,512),Image.Resampling.NEAREST)
        native=image(fs['native_rgb']); native_poly=image(fs['native_polygon'])
        native_overlay=np.asarray(native).copy(); pm=np.asarray(native_poly)>0
        native_overlay[pm]=((native_overlay[pm].astype(np.uint16)+np.array([255,0,0]))//2).astype(np.uint8)
        native_overlay=Image.fromarray(native_overlay)
        ImageDraw.Draw(native_overlay).rectangle(s['crop_xyxy_half_open'],outline='cyan',width=3)
        original_overlay=np.asarray(base).copy(); p=np.asarray(poly)>0
        original_overlay[p]=((original_overlay[p].astype(np.uint16)+np.array([255,0,0]))//2).astype(np.uint8)
        valid=all(any(c['source_id']==sid and c['roi_id']==f'm1_{m}_{s["crop_id"]}' and c['status']=='SUCCEEDED' for c in calls) for m in MODELS)
        review.append({'SOURCE_ID':sid,'CLASS':s['class'],'CROP_ID':s['crop_id'],
                       'PAIR_COMPLETE':valid,'A_QUALITY':'PENDING','B_QUALITY':'PENDING',
                       'PREFERENCE':'PENDING','REVIEWER_COMMENT':'PENDING'})
        if not valid:
            continue
        complete.append(sid)
        sd=package/sid; sd.mkdir()
        base.save(sd/'input_512.png'); mask.save(sd/'edit_mask_512.png'); support.save(sd/'effective_mask_512.png')
        Image.fromarray(original_overlay).save(sd/'input_polygon_512.png')
        panels=[('Native Polygon / Frozen Crop',native_overlay),('Common input 512',base),('Polygon / actual edit support',Image.fromarray(original_overlay)),
                ('Edit mask (gray)',mask),('Effective binary mask',support),('Normal-context and structure: review required',base)]
        bbox=support.getbbox(); zoom=[max(0,bbox[0]-24),max(0,bbox[1]-24),min(512,bbox[2]+24),min(512,bbox[3]+24)]
        rowraw=[]; rowdiff=[]; rowzoom=[]; rowcomp=[]
        for slot in ['A','B']:
            name=mapping[sid][slot]; origin=run/'outputs'/sid/name
            aligned=Image.open(origin/'aligned_512.png').convert('RGB')
            diff=Image.open(origin/'difference_512.png').convert('RGB')
            protected=Image.open(origin/'protected_comparison_only_512.png').convert('RGB')
            for filename,im in [(f'{slot}_raw_aligned_512.png',aligned),(f'{slot}_difference_512.png',diff),
                                (f'{slot}_protected_POSTPROCESS_ONLY_512.png',protected)]:
                # Re-encode in both slots identically, no model or inference metadata.
                im.save(sd/filename)
            rowraw.append((f'{slot} raw output aligned to 512',aligned))
            rowdiff.append((f'{slot} absolute difference, same scale',diff))
            rowzoom.append((f'{slot} local zoom (nearest pixels)',aligned.crop(zoom).resize((512,512),Image.Resampling.NEAREST)))
            rowcomp.append((f'{slot} protected POSTPROCESS COMPARISON ONLY',protected))
        for row in [rowraw,rowdiff,rowzoom,rowcomp]:
            panels.extend(row+[('Input reference',base if row is not rowzoom else base.crop(zoom).resize((512,512),Image.Resampling.NEAREST))])
        sheet(panels,sd/'blind_contact_sheet.png',f'Source {sid} / {s["class"]} / same field of view / A-B labels random per source')
    csvout(package/'human_blind_review_template.csv',review)
    with (package/'README.txt').open('x',encoding='utf-8') as f:
        f.write('Blind practical restoration comparison. Both slots shown at 512, same crop/mask.\n'
                'A/B mapping independent per source and frozen before inference.\n'
                'Judge raw results first. Protected images are POSTPROCESS COMPARISON ONLY.\n'
                'A_QUALITY/B_QUALITY: PASS, FAIL, UNCERTAIN. PREFERENCE: A_WIN, B_WIN, TIE.\n'
                'All human decisions PENDING. No training-pair eligibility or automatic winner.\n'
                'Model architectures, schedulers and native output resolution differ.\n')
    public_files={str(p.relative_to(package)): {'sha256':sha(p),'bytes':p.stat().st_size} for p in package.rglob('*') if p.is_file()}
    dump(package/'blind_file_manifest.json',public_files)
    zp=run/'anonymous_blind_review_package.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file(): z.write(p,str(p.relative_to(package)))
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
        for rel,rec in public_files.items():
            assert hashlib.sha256(z.read(rel)).hexdigest()==rec['sha256']
    for p in (run/'outputs').rglob('*'):
        if p.is_file(): private_inventory.append({'path':str(p.relative_to(run)),'sha256':sha(p),'bytes':p.stat().st_size})
    dump(run/'private/output_file_integrity.json',private_inventory)
    hist=read(run/'historical_artifact_inventory.json')
    assert all(sha(REPO/p)==h for p,h in hist.items())
    verify_inputs()
    before=read(run/'server_git_before.json')
    def filtered(lines):
        return [l for l in lines if str(run.relative_to(REPO)) not in l]
    after=git('status','--porcelain=v1','--untracked-files=all').splitlines()
    assert filtered(before['status'])==filtered(after)
    assert git('rev-parse','HEAD')==before['head']
    runtime=read(run/'call_runtime_records.json')
    safe=[{k:v for k,v in r.items() if k not in {'error'}} for r in runtime]
    fields=['call_id','source_id','model','crop_id','status','seconds','peak_allocated_bytes','peak_reserved_bytes',
            'changed_gt5_inside_effective_mask','changed_gt5_outside_effective_mask','protected_outside_effective_mask_changes']
    csvout(run/'runtime_metrics.csv',[{k:r.get(k) for k in fields} for r in safe],fields)
    dump(run/'input_output_integrity_audit.json',{'input_file_and_pixel_hashes':'PASS','historical_files_unchanged':len(hist),
         'pipeline_threshold_matches_effective_mask':True,'raw_float_outputs_retained':True,'raw_outputs_untouched_by_postprocess':True,
         'output_checks':safe,'private_model_labeled_output_hash_inventory':'controlled only to avoid matching anonymous slots',
         'native_composition':'not a training pair; common 512 background-protection comparison only',
         'complete_pairs':complete,'incomplete_pairs_excluded_from_future_win_statistics':True})
    new_actual=sum(c['status'] in {'STARTED','SUCCEEDED','FAILED','UNKNOWN'} for c in calls)
    success=sum(c['status']=='SUCCEEDED' for c in calls); failed=sum(c['status']=='FAILED' for c in calls)
    status='EXP13_2_M1_COMPLETED_PENDING_CHATGPT_GATE' if success==12 and len(complete)==6 else 'EXP13_2_M1_TECHNICAL_HOLD'
    protection={'smoke_occupied':4,'smoke_remaining':0,'new_occupied':len(calls),'new_actual':new_actual,
                'cumulative_occupied':snap['occupied_calls'],'cumulative_actual':snap['actual_calls'],
                'aggregate_remaining':snap['remaining']['smoke_pilot_revision'],'all_local_remaining':snap['remaining']['all_local'],
                'retry':0,'gpt':0,'val_access':0,'test_access':0,'raw_data_write':0,'split_write':0,
                'ssl_yolo_training':0,'trusted_pair_frozen':0,'historical_ledger_sha256':sha(S5),
                'historical_files_unchanged':len(hist),'historical_git_entries_preserved':len(filtered(before['status'])),
                'server_head_unchanged':before['head'],'disk_free_bytes':shutil.disk_usage(REPO).free}
    dump(run/'budget_preservation.json',protection)
    dump(run/'blind_review_package_manifest.json',{'path':str(zp),'sha256':sha(zp),'bytes':zp.stat().st_size,
         'sheet_count':len(complete),'source_count':len(complete),'zip_entries':len(public_files)+1,
         'mapping_commitment_sha256':sha(key),'mapping_excluded_from_zip_and_public_git':True,
         'human_fields_pending':True,'both_slots_same_512_dimensions':True})
    dump(run/'summary.json',{'status':status,'finished_at':now(),'new_successful_calls':success,'new_failed_calls':failed,
         'new_actual_calls':new_actual,'new_reserved_unstarted':sum(c['status']=='RESERVED' for c in calls),
         'complete_pairs':len(complete),'input_approved':6,'semantic_quality':'PENDING_USER_BLIND_REVIEW',
         'winner':'NOT_SELECTED','budget':protection,'blind_package':read(run/'blind_review_package_manifest.json')})
    dump(run/'provenance.json',{'phase':'Exp13.2-M1','report_start_head':START,'execution_snapshot':read(CODE/'snapshot_manifest.json'),
         'execution_script_sha256':sha(SCRIPT),'server_historical_head':before['head'],'successor_ledger':str(run/'budget_successor_ledger.json'),
         's5_ledger_sha256':S5_SHA,'config_sha256':sha(run/'frozen_execution_config.json'),
         'input_manifest_sha256':sha(run/'frozen_source_crop_manifest.json'),'sources':'six frozen TRAIN sources only',
         'limitations':['distinct image/group IDs do not establish physical independence','architecture/scheduler/native resolution differ',
                       'DreamShaper immutable upstream revision unknown; local fingerprint reused as explicitly authorized',
                       'input PASS is not generated-quality PASS','no automatic model selection or pair freeze'],
         'stopped_for_chatgpt_and_user_blind_gate':True})
    print(json.dumps(read(run/'summary.json'),ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True)
    ap.add_argument('--mode',choices=['preflight','execute','worker','finalize'],required=True)
    args=ap.parse_args(); run=Path(args.run_dir).resolve()
    assert run.parent==REPO/'results/exp13_2_m1_matched_restoration_benchmark'
    if args.mode=='preflight': preflight(run)
    elif args.mode=='worker': execute_worker(run)
    elif args.mode=='finalize': finalize(run)
    else:
        started=now()
        with (run/'execution_stdout.log').open('xb') as out, (run/'execution_stderr.log').open('xb') as err:
            proc=subprocess.run([sys.executable,'-u',str(SCRIPT),'--run-dir',str(run),'--mode','worker'],stdout=out,stderr=err)
        dump(run/'execution_exit.json',{'started_at':started,'finished_at':now(),'exit_code':proc.returncode,
             'command':[sys.executable,str(SCRIPT),'--run-dir',str(run),'--mode','worker'],
             'stdout_sha256':sha(run/'execution_stdout.log'),'stderr_sha256':sha(run/'execution_stderr.log')})
        print(f'Worker exit_code={proc.returncode}',flush=True)
        sys.exit(proc.returncode)


if __name__=='__main__':
    main()
