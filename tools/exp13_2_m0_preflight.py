#!/usr/bin/env python3
"""M0 read-only TRAIN/CPU preflight. No torch/diffusers imports or inference path."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
RAW = Path('/root/autodl-tmp/损伤训练数据集')
SPLIT = REPO / 'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv'
SPLIT_SHA = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
LEDGER = REPO / 'results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json'
DS = Path('/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting')
XL = Path('/root/autodl-tmp/models_staging/exp13_phase1b_r3_20261025T042354Z/sdxl-inpainting-0.1')
XL_MANIFEST = Path('/root/autodl-tmp/sdxl_fp16_manifest.csv')
REV = '115134f363124c53c7d878647567d04daf26e41e'
POOL = [('469', 'Crack'), ('260', 'Crack'), ('470', 'Crack'), ('9', 'corrosion'), ('23', 'corrosion'), ('8', 'corrosion')]
RULES = {
    'crop': 'L=min(W,H); integer half-open box inside native; clamp center origin to feasible interval containing ALL target polygons plus margin',
    'minimum_context_margin_native_px': 'max(16,ceil(0.03*L))',
    'margin_status': 'PREREGISTERED_ENGINEERING_RULE_NOT_VALIDATED_SEMANTIC_SUFFICIENCY',
    'tie_rule': 'center origin=floor((dimension-L)/2); clamp independently on x/y',
    'raster': 'Pillow ImageDraw polygon round(x),round(y), union of ALL target-class polygons',
    'native_edit_mask': 'native polygon union -> MaxFilter(7) -> GaussianBlur(1.0)',
    'common': 'identical square crop; RGB LANCZOS->512; grayscale edit mask NEAREST->512',
    'sdxl': 'common RGB LANCZOS->1024; common mask NEAREST->1024; no padding',
    'mask_polarity': 'white=edit, black=preserve; gray=feather (pipeline mask binarization separately disclosed)',
    'semantic_gate': 'whole square UI/FOV/gray border; real normal-metal continuity; physical edge crossing; unannotated defects; all PENDING unless explicit new review',
    'rejection': 'pairwise exclusion, no replacement/target reduction/relaxation; >=4 sources and >=2 per class plus semantic review required',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, d):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write('\n')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def csvout(p, rows):
    with p.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True)


def save(im, path):
    assert not path.exists()
    im.save(path)
    a = np.asarray(im)
    return {'path': str(path.relative_to(REPO)), 'file_sha256': digest(path),
            'pixel_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'mode': im.mode,
            'shape': list(a.shape), 'dtype': str(a.dtype)}


def sheet(panels, titles, path, heading):
    out = Image.new('RGB', (1024, 70 + 390 * math.ceil(len(panels)/3)), '#eeeeee')
    d = ImageDraw.Draw(out)
    d.text((12, 12), heading, fill='black')
    for i, (im, title) in enumerate(zip(panels, titles)):
        x, y = (i % 3)*340, 70 + (i//3)*390
        d.text((x+6, y), title, fill='black')
        fitted = ImageOps.contain(im.convert('RGB'), (332, 350))
        out.paste(fitted, (x+4, y+26))
    return save(out, path)


def safetensors_header(p):
    with p.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 0 < n < 32*1024*1024:
            raise ValueError('invalid safetensors header size')
        header = json.loads(f.read(n))
    tensors = [v for k, v in header.items() if k != '__metadata__']
    assert tensors
    for t in tensors:
        a, b = t['data_offsets']
        assert 0 <= a <= b <= p.stat().st_size - 8 - n
    return {'tensor_count': len(tensors), 'dtypes': sorted({t['dtype'] for t in tensors}),
            'header_valid': True, 'tensor_values_loaded': False}


def modelaudit():
    print('Static model SHA256/header inspection; no tensor or pipeline load', flush=True)
    expected_manifest_sha = 'a1a0302f7f8f9a0e2bf21e9b47a7757fd4e1ba9f246bf0c59bfcef1d5eb9607e'
    assert digest(XL_MANIFEST) == expected_manifest_sha
    xr = list(csv.DictReader(XL_MANIFEST.open(encoding='utf-8-sig')))
    assert len(xr) == 19
    files = []
    for row in xr:
        p = XL / row['Path']
        assert p.is_relative_to(XL) and p.is_file()
        h = digest(p)
        assert p.stat().st_size == int(row['Bytes']) and h == row['SHA256']
        files.append({'path': row['Path'], 'bytes': p.stat().st_size, 'sha256': h,
                      'header': safetensors_header(p) if p.suffix == '.safetensors' else None})
    assert sum(f['bytes'] for f in files) == 6941223330
    common = ['model_index.json', 'scheduler/scheduler_config.json', 'tokenizer/merges.txt',
              'tokenizer/vocab.json', 'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json',
              'text_encoder/config.json', 'unet/config.json', 'vae/config.json',
              'feature_extractor/preprocessor_config.json', 'safety_checker/config.json']
    # Historical loader used default full safetensors cast to FP16, no variant.
    for component, name in [('text_encoder', 'model'), ('unet', 'diffusion_pytorch_model'),
                            ('vae', 'diffusion_pytorch_model'), ('safety_checker', 'model')]:
        common += [f'{component}/{name}.safetensors', f'{component}/{name}.fp16.safetensors']
    df = []
    for rel in common:
        p = DS / rel
        assert p.is_file(), rel
        df.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': digest(p),
                   'header': safetensors_header(p) if p.suffix == '.safetensors' else None})
    recovery = REPO/'results/exp13_diffusion_pilot/run_20260921_124440/recovery_config.json'
    result = {}
    for name, root, fs in [('DreamShaper', DS, df), ('SDXL', XL, files)]:
        configs = {k: read(root/k) for k in ['model_index.json', 'scheduler/scheduler_config.json', 'vae/config.json', 'unet/config.json']}
        result[name] = {'local_path': str(root), 'status': 'LOCAL_FILES_COMPLETE_STATIC_ONLY',
                        'files': fs, 'verified_bytes': sum(f['bytes'] for f in fs), 'configs': configs,
                        'content_fingerprint': hashlib.sha256(json.dumps(fs, sort_keys=True).encode()).hexdigest(),
                        'runtime_load_test': 'NOT_EXECUTED_M0'}
    assert result['DreamShaper']['configs']['model_index.json']['_class_name'] == 'StableDiffusionInpaintPipeline'
    assert result['SDXL']['configs']['model_index.json']['_class_name'] == 'StableDiffusionXLInpaintPipeline'
    result['DreamShaper'].update({'model_id': 'aiplayground/dreamshaper-8-inpainting', 'historical_revision': 'master',
        'immutable_remote_revision': None, 'acquisition': read(recovery), 'acquisition_record_sha256': digest(recovery),
        'integrity_limit': 'New local hash freeze; no authenticated historic upstream hash baseline. Embedded config paths do not establish downloaded repository revision.',
        'loader': {'variant': None, 'use_safetensors': True, 'torch_dtype': 'float16', 'safety_checker': None,
                   'local_files_only': True, 'enable_attention_slicing': True},
        'license': 'Not revalidated against immutable upstream metadata in this offline phase; retain historical restrictions, review before M1.'})
    result['SDXL'].update({'model_id': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', 'revision': REV,
        'manifest_sha256': expected_manifest_sha, 'loader': {'variant': 'fp16', 'use_safetensors': True,
        'local_files_only': True, 'torch_dtype': 'float16', 'cpu_offload': True, 'vae_tiling': True, 'vae_slicing': True}})
    return result


def audit_source(run, row, label):
    sid = row['stem']
    p, j = RAW/row['image_path'], RAW/row['json_path']
    assert p.resolve().is_relative_to(RAW) and j.resolve().is_relative_to(RAW)
    before = [digest(p), digest(j)]
    assert before == [row['sha256'], row['json_sha256']]
    im = Image.open(p).convert('RGB')
    w, h = im.size
    assert [w,h] == [int(row['image_width']), int(row['image_height'])]
    ann = read(j)
    shapes = ann['shapes']
    assert all(s.get('shape_type', 'polygon') == 'polygon' and s['label'] == label for s in shapes)
    polys, union = [], Image.new('L', im.size)
    for i, s in enumerate(shapes):
        pts = [(float(x), float(y)) for x,y in s['points']]
        assert len(set(pts)) >= 3 and all(math.isfinite(x) and math.isfinite(y) and 0<=x<w and 0<=y<h for x,y in pts)
        area = abs(sum(pts[k][0]*pts[(k+1)%len(pts)][1]-pts[(k+1)%len(pts)][0]*pts[k][1] for k in range(len(pts))))/2
        assert area > 0
        mask = Image.new('L', im.size)
        ImageDraw.Draw(mask).polygon([(round(x), round(y)) for x,y in pts], fill=255)
        polys.append((mask, {'polygon_id': i, 'points_native_xy': pts, 'shoelace_area': area,
                              'raster_pixels': int(np.count_nonzero(mask)), 'bbox_xyxy_half_open': list(mask.getbbox())}))
        ImageDraw.Draw(union).polygon([(round(x), round(y)) for x,y in pts], fill=255)
    L, bbox = min(w,h), union.getbbox()
    m = max(16, math.ceil(.03*L))
    cx,cy = (w-L)//2,(h-L)//2
    intervals = [[max(0,bbox[2]+m-L), min(w-L,bbox[0]-m)], [max(0,bbox[3]+m-L), min(h-L,bbox[1]-m)]]
    feasible = all(lo<=hi for lo,hi in intervals)
    x,y = [max(lo,min(hi,c)) for (lo,hi),c in zip(intervals,[cx,cy])] if feasible else [cx,cy]
    box = (x,y,x+L,y+L)
    reason = None if feasible else ('ALL_TARGET_SPAN_EXCEEDS_MAX_SQUARE' if bbox[2]-bbox[0]>L or bbox[3]-bbox[1]>L else 'NO_ALL_SIDES_NORMAL_CONTEXT_MARGIN')
    edit = union.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1))
    out = run/sid
    out.mkdir()
    files = {}
    for key, img in [('native_rgb', im), ('native_polygon', union), ('native_edit_mask', edit),
                     ('common_rgb_512', im.crop(box).resize((512,512), Image.Resampling.LANCZOS)),
                     ('common_edit_mask_512', edit.crop(box).resize((512,512), Image.Resampling.NEAREST))]:
        files[key] = save(img, out/(key+'.png'))
    rgb = Image.open(REPO/files['common_rgb_512']['path'])
    mask = Image.open(REPO/files['common_edit_mask_512']['path'])
    xlmask = mask.resize((1024,1024), Image.Resampling.NEAREST)
    files['sdxl_rgb_1024'] = save(rgb.resize((1024,1024), Image.Resampling.LANCZOS), out/'sdxl_rgb_1024.png')
    files['sdxl_mask_1024'] = save(xlmask, out/'sdxl_mask_1024.png')
    exact = bool(np.array_equal(np.asarray(xlmask), np.repeat(np.repeat(np.asarray(mask),2,0),2,1)))
    assert exact
    center = (cx,cy,cx+L,cy+L)
    details = []
    for pm, rec in polys:
        n = rec['raster_pixels']
        details.append(dict(rec, retained_crop_pixels=int(np.count_nonzero(pm.crop(box))),
                            center_crop_retention=float(np.count_nonzero(pm.crop(center))/n),
                            crop_retention=float(np.count_nonzero(pm.crop(box))/n)))
    retained = int(np.count_nonzero(union.crop(box)))
    support_retained = int(np.count_nonzero(edit.crop(box)))
    target_total, support_total = int(np.count_nonzero(union)), int(np.count_nonzero(edit))
    if feasible:
        assert retained == target_total and support_retained == support_total
    # Pixel-center coordinate mapping used by Pillow resize; edges scale without .5 offset.
    scale = 512/L
    matrix = [[scale,0,scale*(.5-x)-.5],[0,scale,scale*(.5-y)-.5],[0,0,1]]
    inv = np.linalg.inv(np.asarray(matrix))
    pts = np.array([[px,py,1] for _,r in polys for px,py in r['points_native_xy']]).T
    err = float(np.max(np.abs(inv @ (np.asarray(matrix) @ pts)-pts)))
    assert err < 1e-9
    overlay = im.copy()
    d = ImageDraw.Draw(overlay)
    for _,rec in polys:
        d.line(rec['points_native_xy']+[rec['points_native_xy'][0]], fill='red', width=3)
    d.rectangle((x,y,x+L-1,y+L-1), outline='cyan', width=3)
    preview = sheet([im,overlay,union,rgb,mask,rgb.resize((1024,1024),Image.Resampling.LANCZOS)],
                    ['Native RGB','All polygons / square','Native polygon union','Common RGB 512','Common mask 512','SDXL input 1024'],
                    out/'input_review.png', f'{sid} {label}: '+('GEOMETRY_ONLY / SEMANTICS_PENDING' if feasible else 'DIAGNOSTIC_ONLY / NOT_FROZEN_FOR_INFERENCE'))
    assert [digest(p),digest(j)] == before
    return {'source_id': sid, 'class': label, 'split': 'train', 'group_id': row['group_id'],
        'source_path': str(p), 'annotation_path': str(j), 'source_sha256': before[0], 'annotation_sha256': before[1],
        'raw_hashes_unchanged': True, 'native_wh': [w,h], 'polygon_count': len(polys), 'polygon_union_bbox': list(bbox),
        'polygon_details': details, 'geometry_status': 'PASS_GEOMETRY_ONLY' if feasible else 'HOLD',
        'rejection_reason': reason, 'semantic_status': 'PENDING_WHOLE_CONTEXT_REVIEW', 'restoration_eligible': False,
        'crop_xyxy_half_open': list(box) if feasible else None, 'diagnostic_crop_xyxy': list(box),
        'center_crop_xyxy': list(center), 'center_crop_union_retention': float(np.count_nonzero(union.crop(center))/target_total),
        'minimum_context_margin': m, 'feasible_origin_intervals_xy': intervals,
        'polygon_pixels_total': target_total, 'polygon_pixels_retained': retained, 'edit_support_pixels_total': support_total,
        'edit_support_pixels_retained': support_retained, 'target_retention': retained/target_total,
        'normal_context_proxy': {'unmasked_fraction_512': float(np.mean(np.asarray(mask)==0)), 'meaning': 'Unmasked pixels are NOT verified normal metal'},
        'input_content_fraction': 1, 'padding_fraction': 0, 'native_to_512_scale': scale,
        'pixel_center_native_to_512_matrix': matrix, 'pixel_center_inverse_matrix': inv.tolist(),
        'coordinate_roundtrip_max_abs_error': err, 'mask_2x_exact_all_blocks': exact,
        'model_preprocessor_mask_threshold': 'Diffusers 0.35.1 default mask binarization at 0.5; same support threshold >=128 for both pipelines',
        'files': files, 'review_sheet': preview, 'input_use': 'GEOMETRY_CONDITIONAL_NOT_APPROVED' if feasible else 'DIAGNOSTIC_ONLY_NOT_FROZEN_FOR_INFERENCE'}


def execute(run, args):
    started = now()
    baseline = git('status','--porcelain=v1','--untracked-files=all').splitlines()
    dump(run/'server_git_before.json', {'branch': git('branch','--show-current').strip(), 'head': git('rev-parse','HEAD').strip(),
                                       'status': baseline, 'remote': git('remote','get-url','origin').strip()})
    assert digest(SPLIT) == SPLIT_SHA
    ledger_sha = digest(LEDGER)
    ledger = read(LEDGER)
    assert ledger['occupied_calls'] == ledger['actual_calls'] == 4
    assert ledger['remaining']['smoke'] == 0 and ledger['remaining']['all_local'] == 296 and ledger['remaining']['smoke_pilot_revision'] == 116
    assert len(ledger['calls']) == 4 and not any(c['status'] in ['RESERVED','STARTED'] for c in ledger['calls'].values())
    dump(run/'preregistered_input_rules.json', {'frozen_at': now(), 'rules': RULES, 'new_model_calls_authorized': 0})
    rows = list(csv.DictReader(SPLIT.open(encoding='utf-8-sig')))
    train = [r for r in rows if r['split']=='train']
    candidates = []
    for sid,label in POOL:
        found = [r for r in train if r['stem']==sid]
        assert len(found) == 1 and found[0]['labels_present']==label
        print('TRAIN audit', sid, flush=True)
        candidates.append(audit_source(run,found[0],label))
    assert len({c['source_sha256'] for c in candidates})==6 and len({c['group_id'] for c in candidates})==6
    geom = [c for c in candidates if c['geometry_status']=='PASS_GEOMETRY_ONLY']
    counts = {label:sum(c['class']==label for c in geom) for label in ['Crack','corrosion']}
    selection_hold = len(geom)<4 or min(counts.values())<2
    dump(run/'source_selection_freeze.json', {'status': 'SOURCE_SELECTION_HOLD' if selection_hold else 'SEMANTIC_REVIEW_HOLD',
        'pool': POOL, 'geometry_only_candidates': [c['source_id'] for c in geom], 'geometry_class_counts': counts,
        'excluded': [{'source_id':c['source_id'],'reason':c['rejection_reason']} for c in candidates if c not in geom],
        'approved_sources': [], 'executable_call_count': 0, 'physical_part_independence': 'UNPROVEN',
        'distinct_file_hashes': 6, 'distinct_train_groups': 6, 'rules': RULES})
    dump(run/'shared_crop_manifest.json', {'candidates': candidates, 'all_images_controlled_server_only': True})
    fields = ['source_id','class','group_id','source_sha256','annotation_sha256','polygon_count','geometry_status','rejection_reason','semantic_status','crop_xyxy_half_open','target_retention','center_crop_union_retention','minimum_context_margin']
    csvout(run/'candidate_preflight.csv', [{k:c[k] for k in fields} for c in candidates])
    dump(run/'shared_mask_alignment.json', {'rules': RULES, 'candidates': [{k:c[k] for k in ['source_id','target_retention','edit_support_pixels_total','edit_support_pixels_retained','coordinate_roundtrip_max_abs_error','mask_2x_exact_all_blocks','pixel_center_native_to_512_matrix','pixel_center_inverse_matrix']} for c in candidates],
          'resampling_limit': '512 rasterization may alter narrow-boundary sampling; nearest 2x is exact relative to frozen 512 baseline. Geometry is not semantic repair assurance.'})
    panels = [Image.open(REPO/c['review_sheet']['path']) for c in candidates]
    sheet(panels, [f"{c['source_id']} {c['geometry_status']}" for c in candidates], run/'six_source_overview.png', 'M0 CPU input overview / NO INFERENCE / SOURCE HOLD')
    models = modelaudit()
    dump(run/'model_asset_preflight.json', models)
    config = {'study': 'PAIRED_PRACTICAL_MODEL_ADAPTATION_STUDY', 'objective': 'RESTORE_TO_PSEUDO_NORMAL_METAL',
        'prompt': 'restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting',
        'negative_prompt': None, 'steps':20, 'guidance_scale':7.5, 'strength':.8, 'seed':130500, 'precision':'float16',
        'execution':'single GPU sequential; no search/retry/download; no implicit crop; padding_mask_crop=None',
        'resolution': {'DreamShaper':512,'SDXL':1024},
        'scheduler':{n:m['configs']['scheduler/scheduler_config.json']['_class_name'] for n,m in models.items()},
        'assets':{n:{'fingerprint':m['content_fingerprint'],'loader':m['loader']} for n,m in models.items()},
        'residual_factors':['native model resolution','native scheduler and architecture/VAE','internal noise schedule differs despite equal numerical seed/strength/guidance'],
        'authorization':'PROPOSED_M1_ONLY_SEPARATE_CHATGPT_GATE_REQUIRED', 'input_rules':RULES}
    dump(run/'frozen_model_comparison_protocol.json', config)
    csvout(run/'planned_call_matrix.csv', [{'source_id':c['source_id'],'class':c['class'],'model':model,
        'resolution':config['resolution'][model], 'status':'BLOCKED_GLOBAL_SOURCE_SELECTION_HOLD',
        'source_geometry':c['geometry_status'], 'new_calls_now':0, 'future_calls_if_all_gates_pass':1}
        for c in candidates for model in ['DreamShaper','SDXL']])
    dump(run/'blind_review_protocol.json', {'status':'PREREGISTERED_NOT_EXECUTED','raw_outputs':'immutable; compare outputs aligned to512 by same LANCZOS; additionally preserve native512/1024 outputs',
        'protected_composite':'POSTPROCESS_COMPARISON_ONLY; base=common RGB512; alpha=common grayscale mask/255 clipped to support(mask>0); outside support exact original; no model-specific blend',
        'differences':'max abs RGB channel >5 uint8; inside/outside mask>0 and >=128 separately; same native-pixel zoom and scale for both',
        'review_panels':'same FOV original, polygon, mask, aligned raw output, protected composite, abs difference, target 1:1 and 4x NEAREST zoom; no model/runtime labels',
        'blinding':'After future source approval and successful pair completion, secrets.SystemRandom shuffle balanced model A/B assignment across sources; save private key outside review package/Git, publish SHA256 commitment before review; fail or incomplete pairs excluded from wins',
        'fields':['source_id','A_PASS_FAIL_UNCERTAIN','B_PASS_FAIL_UNCERTAIN','A_WIN_B_WIN_TIE','reason','severe_hallucination','structure_destruction','defect_removal','metal_texture','seam','raw_background_change','composite_acceptability'],
        'runtime_metrics':'separate nonblinded audit: per call seconds, peak allocated/reserved memory, failure charged',
        'decision_owner':'USER', 'statistical_scope':'source-level practical recommendation only; no pixel independence, no significance or seven-class superiority claim',
        'key_created':False})
    assert digest(LEDGER)==ledger_sha
    dump(run/'budget_preservation.json', {'authoritative_ledger':str(LEDGER),'sha256_before':ledger_sha,'sha256_after':digest(LEDGER),
        'occupied_calls':4,'actual_calls':4,'smoke_remaining':0,'aggregate_remaining':116,'all_local_remaining':296,
        'new_model_calls':0,'model_loads':0,'new_downloads':0,'gpt_fallback_calls':0,'retry_calls':0,
        'requested_future_ceiling':12,'geometry_conditional_future_max':len(geom)*2,'executable_now':0,
        'historical_DreamShaper':'88 outputs /20 source groups separate historical provenance, no recharge or S5 backfill',
        'future_ledger':'After separate authorization only, one persistent locked successor inherits ALL S5 history/calls/counters, hashes parent, models share accounting; reserve complete two-call source pair atomically, persist STARTED before each call; failures occupy/charge; no auto retry; no incomplete-pair wins',
        'implementation_reference':'tools/exp13_restoration/protocol.py BudgetProtocol; future pair-reservation integration must be checked before M1'})
    dump(run/'latest_user_decision_overlay.json', {'source':'USER_M0_PROMPT','recorded_at':now(),
        'DreamShaper_source_visual_acceptance':{'Crack':['86','111','146','260','470'],'corrosion':['2','8','13']},
        'scope':'source visual level only; preserve crop limitations and per-variant historical decisions; no all-variant quality inheritance',
        'SDXL_S5_Crack260_visual_effect':'USER_ACCEPTED','SDXL_S2_compact_ROI_failures':'PRESERVED',
        'A5_and_Local_Change_training':'PAUSED','trusted_pair_freeze':0,'historical_records_modified':False})
    forbidden_imports = sorted(n for n in __import__('sys').modules if n.split('.')[0] in ['torch','diffusers','transformers'])
    assert not forbidden_imports
    after = git('status','--porcelain=v1','--untracked-files=all').splitlines()
    # New M0 files are the only allowed change to the server checkout.
    def historical(status):
        return [s for s in status if 'exp13_2_m0' not in s]
    assert historical(baseline)==historical(after)
    dump(run/'server_git_preservation.json', {'status_before':baseline,'status_after':after,
        'historical_entries_unchanged':True,'server_head_after':git('rev-parse','HEAD').strip(),'sync_performed':False})
    dump(run/'provenance.json', {'started_at':started,'finished_at':now(),'report_start_head':args.report_head,
        'execution_server_head':git('rev-parse','HEAD').strip(),'script_sha256':digest(__file__), 'split_manifest_sha256':digest(SPLIT),
        'python':platform.python_version(), 'packages':{n:importlib.metadata.version(n) for n in ['pillow','numpy','torch','diffusers','transformers','accelerate']},
        'imported_model_packages':forbidden_imports,'disk_free_bytes_after':shutil.disk_usage(run).free,
        'model_load':0,'inference':0,'val_access':0,'test_access':0,'raw_write':0,'ssl_training':0,'yolo_training':0,
        'scope':'Only six TRAIN image/JSON files opened; split metadata read for filtering; static local model bytes hashed without loading tensors',
        'git_safety':'No pull/reset/clean/add-all; server old checkout and historical untracked preserved; no paper repo',
        'industrial_previews':'controlled server only, not for Git'})
    dump(run/'summary.json', {'status':'EXP13_2_M0_SOURCE_OR_PROTOCOL_HOLD','reason':'SOURCE_SELECTION_HOLD',
        'geometry_candidates':[c['source_id'] for c in geom], 'geometry_class_counts':counts,'approved_sources':0,
        'recommend_M1':False,'proposed_call_ceiling':12,'calls_authorized_now':0,'new_calls':0,
        'model_static_files':'COMPLETE; DreamShaper immutable upstream revision not established; runtime compatibility not tested',
        'source_gate':'corrosion9 all-target span impossible; corrosion23 target reaches native border and lacks required context; only corrosion8 remains',
        'stop':'Await ChatGPT Gate; no M1/Pilot/pair freeze/SSL/YOLO'})
    print('Completed CPU preflight: SOURCE_SELECTION_HOLD',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-head', required=True)
    args = parser.parse_args()
    if not all(c.isalnum() or c in '_-' for c in args.run_id):
        raise ValueError('invalid run id')
    run = REPO/'results/exp13_2_m0_matched_model_benchmark'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    code = 1
    with (run/'stdout.log').open('x') as out, (run/'stderr.log').open('x') as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                execute(run,args)
                code = 0
            except Exception:
                traceback.print_exc()
    dump(run/'execution_exit.json', {'exit_code':code,'captured_at':now(),'stdout':'stdout.log','stderr':'stderr.log'})
    print(str(run), 'exit_code=',code)
    raise SystemExit(code)
