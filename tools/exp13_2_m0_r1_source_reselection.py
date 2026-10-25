#!/usr/bin/env python3
"""M0-R1 CPU/TRAIN candidate freeze and human INPUT review; no model imports.

Existing M0 default PNGs are verified and referenced, never regenerated on disk.
Alternative crops are proposals, not replacements or semantic approvals.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
RAW = Path('/root/autodl-tmp/损伤训练数据集')
SPLIT = REPO/'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv'
SPLIT_SHA = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
LEDGER = REPO/'results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json'
LEDGER_SHA = '3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94'
M0 = REPO/'results/exp13_2_m0_matched_model_benchmark/20261025T115110Z_m0'
ANCHORS = {
    'shared_crop_manifest.json': '44146f7fa660e41221982dd215ee6ba4ba6ad1adc80722a46e626e1b1ba1d7f1',
    'frozen_model_comparison_protocol.json': '949c863df202ce15e39cc86495098727f669f4d3779110fcac7b8d61ea4de8e7',
    'preregistered_input_rules.json': '525e6e84e53f12edb7cd62ccfb5ddbd2fd96142f265164019a2dd290223fbd8e',
}
FIXED = ['469', '260', '470', '8']
PRIORITY = ['1', '2', '6', '7', '12', '13', '27']
PENDING = 'PENDING'
REVIEW_FIELDS = ['SOURCE_ID', 'SELECTED_CROP_ID', 'CROP_CONTEXT_ACCEPTABLE',
                 'MASK_COVERAGE_ACCEPTABLE', 'METAL_STRUCTURE_PRESERVED_BY_INPUT',
                 'UI_FOV_CONTAMINATION_ACCEPTABLE', 'NORMAL_METAL_REFERENCE_SUFFICIENT',
                 'FINAL_INPUT_DECISION', 'REVIEWER_COMMENT']
RISKS = ['Whole ROI GE/UI/FOV contamination', 'Gray border / invalid area',
         'Real metal seam, cavity or physical structure edge',
         'Enough valid normal metal texture (unmasked does NOT mean normal)',
         'Unannotated additional defects / plausible normal restoration target']


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


def dump(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def csvout(p, fields, rows):
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def pixel(im):
    return hashlib.sha256(np.asarray(im).tobytes()).hexdigest()


def save(im, path):
    assert not path.exists(), str(path)
    im.save(path)
    a = np.asarray(im)
    return {'path': str(path.relative_to(REPO)), 'file_sha256': sha(path),
            'pixel_sha256': pixel(im), 'mode': im.mode, 'shape': list(a.shape), 'dtype': str(a.dtype)}


def verify_file(rec):
    p = (REPO/rec['path']).resolve()
    assert p.is_relative_to(M0) and sha(p) == rec['file_sha256'], str(p)
    with Image.open(p) as im:
        a = np.asarray(im)
        assert pixel(im) == rec['pixel_sha256'] and im.mode == rec['mode']
        assert list(a.shape) == rec['shape'] and str(a.dtype) == rec['dtype']


def crossing(a, b, c, d):
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def on(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
    o1, o2, o3, o4 = orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)
    return ((o1*o2 < 0 and o3*o4 < 0) or any(abs(o) < 1e-9 and on(p,q,r)
            for o,p,q,r in [(o1,a,b,c),(o2,a,b,d),(o3,c,d,a),(o4,c,d,b)]))


def analyze(row, label, kind):
    """Open only a previously TRAIN-filtered row; do not write any raw path."""
    assert row['split'] == 'train' and row['labels_present'] == label
    ip, jp = (RAW/row['image_path']).resolve(), (RAW/row['json_path']).resolve()
    assert ip.is_relative_to(RAW) and jp.is_relative_to(RAW)
    assert [sha(ip), sha(jp)] == [row['sha256'], row['json_sha256']]
    im = Image.open(ip).convert('RGB')
    assert list(im.size) == [int(row['image_width']), int(row['image_height'])]
    ann = read(jp)
    rec = {'source_id': row['stem'], 'class': label, 'candidate_kind': kind, 'split': 'train',
           'group_id': row['group_id'], 'near_duplicate_group_id': row['near_duplicate_group_id'],
           'source_path': str(ip), 'annotation_path': str(jp), 'source_sha256': row['sha256'],
           'annotation_sha256': row['json_sha256'], 'native_wh': list(im.size),
           'geometry_status': 'REJECT_GEOMETRY', 'semantic_status': 'PENDING_USER_REVIEW',
           'input_semantic_eligibility': 'PENDING_USER_REVIEW', 'restoration_eligible': False}
    shapes = ann.get('shapes', [])
    if not shapes or not all(s.get('shape_type', 'polygon') == 'polygon' and s.get('label') == label for s in shapes):
        rec['rejection_reason'] = 'UNHANDLED_OR_MIXED_ANNOTATION_SHAPES'
        return rec, None
    w,h = im.size
    union = Image.new('L', im.size)
    polys, masks, intersections = [], [], []
    for i,s in enumerate(shapes):
        pts = [(float(x), float(y)) for x,y in s['points']]
        if pts and pts[-1] == pts[0]:
            pts = pts[:-1]
        if len(set(pts)) < 3 or not all(math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h for x,y in pts):
            rec['rejection_reason'] = 'INVALID_POLYGON_COORDINATES'
            return rec, None
        area = abs(sum(pts[k][0]*pts[(k+1)%len(pts)][1]-pts[(k+1)%len(pts)][0]*pts[k][1] for k in range(len(pts))))/2
        for a in range(len(pts)):
            for b in range(a+1, len(pts)):
                if b == a+1 or (a == 0 and b == len(pts)-1):
                    continue
                if crossing(pts[a], pts[(a+1)%len(pts)], pts[b], pts[(b+1)%len(pts)]):
                    intersections.append({'polygon': i, 'edges': [a,b]})
        pm = Image.new('L', im.size)
        rp = [(round(x), round(y)) for x,y in pts]
        ImageDraw.Draw(pm).polygon(rp, fill=255)
        ImageDraw.Draw(union).polygon(rp, fill=255)
        polys.append({'polygon_id': i, 'points_native_xy': pts, 'raster_pixels': int(np.count_nonzero(pm)),
                      'bbox_xyxy_half_open': list(pm.getbbox()), 'shoelace_area': area})
        masks.append(pm)
    L = min(w,h)
    margin = max(16, math.ceil(.03*L))
    bbox = union.getbbox()
    intervals = [[max(0,bbox[2]+margin-L), min(w-L,bbox[0]-margin)],
                 [max(0,bbox[3]+margin-L), min(h-L,bbox[1]-margin)]]
    feasible = all(lo <= hi for lo,hi in intervals)
    center = [(w-L)//2, (h-L)//2]
    origin = [max(lo,min(hi,c)) for (lo,hi),c in zip(intervals,center)] if feasible else center
    box = [origin[0], origin[1], origin[0]+L, origin[1]+L]
    edit = union.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.0))
    rec.update({'polygon_count': len(polys), 'polygon_details': polys, 'polygon_union_bbox': list(bbox),
                'nonadjacent_edge_intersections': intersections, 'minimum_context_margin': margin,
                'feasible_origin_intervals_xy': intervals, 'crop_xyxy_half_open': box if feasible else None,
                'polygon_pixels_total': int(np.count_nonzero(union)),
                'edit_support_pixels_total': int(np.count_nonzero(edit)),
                'padding_fraction': 0, 'input_content_fraction': 1,
                'normal_context_proxy_warning': 'Bounding-box margin and unmasked pixels do not establish normal metal',
                'rejection_reason': None})
    reason = ('ALL_TARGET_SPAN_EXCEEDS_MAX_SQUARE' if bbox[2]-bbox[0] > L or bbox[3]-bbox[1] > L
              else 'NO_ALL_SIDES_NORMAL_CONTEXT_MARGIN') if not feasible else None
    if intersections or any(p['shoelace_area'] == 0 for p in polys):
        reason = 'POLYGON_TOPOLOGY_REVIEW_REQUIRED'
    data = (im, union, edit, masks)
    if reason is None:
        check = crop_checks(rec, data, box)
        rec['default_geometry_checks'] = check
        if not check['technical_pass']:
            reason = 'MASK_OR_COORDINATE_VALIDATION_FAILED'
    rec.update({'geometry_status': 'PASS_GEOMETRY_ONLY' if reason is None else 'REJECT_GEOMETRY',
                'rejection_reason': reason})
    assert [sha(ip),sha(jp)] == [row['sha256'],row['json_sha256']]
    rec['raw_hashes_unchanged'] = True
    return rec, data


def crop_checks(rec, data, box):
    im,union,edit,polys = data
    w,h = im.size
    x,y,x1,y1 = box
    L = min(w,h)
    mask = edit.crop(box).resize((512,512), Image.Resampling.NEAREST)
    target = union.crop(box)
    binary = np.asarray(mask) >= 128
    target512 = np.asarray(target.resize((512,512), Image.Resampling.NEAREST)) > 0
    inverse = np.asarray(Image.fromarray(binary.astype(np.uint8)*255).resize((L,L),Image.Resampling.NEAREST)) > 0
    bbox = union.getbbox()
    margins = [bbox[0]-x,bbox[1]-y,x1-bbox[2],y1-bbox[3]]
    checks = {'in_bounds_square': 0 <= x < x1 <= w and 0 <= y < y1 <= h and x1-x == y1-y == L,
              'actual_bbox_context_margins_ltrb': margins, 'context_margin_pass': min(margins) >= rec['minimum_context_margin'],
              'per_polygon_retention': [int(np.count_nonzero(pm.crop(box)))/int(np.count_nonzero(pm)) for pm in polys],
              'target_retention': int(np.count_nonzero(target))/int(np.count_nonzero(union)),
              'edit_support_retention': int(np.count_nonzero(edit.crop(box)))/int(np.count_nonzero(edit)),
              'polygon512_pixels_missing_from_effective_mask': int(np.count_nonzero(target512 & ~binary)),
              'native_polygon_pixels_missing_after_effective_mask_roundtrip': int(np.count_nonzero((np.asarray(target)>0) & ~inverse)),
              'effective_edit_pixels_512': int(binary.sum()),
              'grayscale_edit_support_pixels_512': int(np.count_nonzero(mask)),
              'unmasked_fraction_512_NOT_normal_metal_fraction': float(np.mean(np.asarray(mask)==0)),
              'mask_polarity': 'white=edit/black=preserve; Diffusers >=0.5 (uint8 >=128)',
              'sdxl_mask_2x_exact': bool(np.array_equal(np.asarray(mask.resize((1024,1024),Image.Resampling.NEAREST)),
                                                       np.repeat(np.repeat(np.asarray(mask),2,0),2,1)))}
    s = 512/L
    matrix = np.asarray([[s,0,s*(.5-x)-.5],[0,s,s*(.5-y)-.5],[0,0,1]])
    pts = np.array([[px,py,1] for p in rec['polygon_details'] for px,py in p['points_native_xy']]).T
    err = float(np.max(np.abs(np.linalg.inv(matrix)@(matrix@pts)-pts)))
    checks.update({'pixel_center_native_to_512_matrix': matrix.tolist(), 'pixel_center_inverse_matrix': np.linalg.inv(matrix).tolist(),
                   'coordinate_roundtrip_max_abs_error': err})
    checks['technical_pass'] = bool(checks['in_bounds_square'] and checks['context_margin_pass'] and
        checks['target_retention'] == checks['edit_support_retention'] == 1 and
        all(v == 1 for v in checks['per_polygon_retention']) and
        checks['polygon512_pixels_missing_from_effective_mask'] == checks['native_polygon_pixels_missing_after_effective_mask_roundtrip'] == 0 and
        checks['sdxl_mask_2x_exact'] and err < 1e-9)
    return checks


def verify_m0(trains):
    for name,h in ANCHORS.items():
        assert sha(M0/name) == h, 'M0 anchor mismatch: '+name
    m = read(M0/'shared_crop_manifest.json')
    records, data_by_id, proof = {}, {}, []
    for old in m['candidates']:
        sid = old['source_id']
        if sid not in FIXED:
            continue
        row = trains[sid]
        assert row['sha256'] == old['source_sha256'] and row['json_sha256'] == old['annotation_sha256']
        rec,data = analyze(row, old['class'], 'M0_RETAINED_IMMUTABLE_DEFAULT')
        assert rec['geometry_status'] == old['geometry_status'] == 'PASS_GEOMETRY_ONLY'
        assert rec['crop_xyxy_half_open'] == old['crop_xyxy_half_open'] and rec['group_id'] == old['group_id']
        assert rec['polygon_count'] == old['polygon_count']
        for item in list(old['files'].values())+[old['review_sheet']]:
            verify_file(item)
        im,union,edit,_ = data
        box = old['crop_xyxy_half_open']
        rgb = im.crop(box).resize((512,512),Image.Resampling.LANCZOS)
        mask = edit.crop(box).resize((512,512),Image.Resampling.NEAREST)
        reconstructed_in_memory = {'native_rgb': im, 'native_polygon': union, 'native_edit_mask': edit,
            'common_rgb_512': rgb, 'common_edit_mask_512': mask,
            'sdxl_rgb_1024': rgb.resize((1024,1024),Image.Resampling.LANCZOS),
            'sdxl_mask_1024': mask.resize((1024,1024),Image.Resampling.NEAREST)}
        for name,img in reconstructed_in_memory.items():
            assert pixel(img) == old['files'][name]['pixel_sha256'], 'M0 rule/pixel mismatch '+sid+'/'+name
        rec['m0_files'] = old['files']
        records[sid],data_by_id[sid] = rec,data
        proof.append({'source_id': sid, 'status': 'ACTUAL_FILE_AND_PIXEL_HASHES_VERIFIED_NO_REBUILD',
                      'source_sha256': rec['source_sha256'], 'annotation_sha256': rec['annotation_sha256'],
                      'crop_xyxy_half_open': box, 'files': old['files'], 'review_sheet': old['review_sheet'],
                      'same_rule_in_memory_pixel_comparison': True, 'checks': rec['default_geometry_checks']})
    assert set(records) == set(FIXED)
    excluded = {c['source_id']:c for c in m['candidates'] if c['source_id'] in ['9','23']}
    assert excluded['9']['rejection_reason'] == 'ALL_TARGET_SPAN_EXCEEDS_MAX_SQUARE'
    assert excluded['23']['rejection_reason'] == 'NO_ALL_SIDES_NORMAL_CONTEXT_MARGIN'
    return records,data_by_id,{'anchors': ANCHORS, 'retained': proof, 'historical_exclusions': excluded,
                             'changes_to_m0_inputs': 0, 'no_exclusion_reaudit_or_relaxation': True}


def sheet(panels, titles, path, header, footer):
    lines = [part for line in footer for part in textwrap.wrap(line, width=145)]
    top,cellw,cellh = 76,512,540
    out = Image.new('RGB',(1536,top+cellh*math.ceil(len(panels)/3)+24+20*len(lines)), '#eeeeee')
    d = ImageDraw.Draw(out)
    d.text((12,12),header,fill='black',font_size=18)
    d.text((12,42),'INPUT PROPOSALS ONLY / NO GENERATION / ALL HUMAN DECISIONS PENDING',fill='darkred',font_size=16)
    for i,(im,title) in enumerate(zip(panels,titles)):
        x,y = (i%3)*cellw,top+(i//3)*cellh
        d.text((x+8,y+2),title,fill='black',font_size=15)
        out.paste(ImageOps.contain(im.convert('RGB'),(496,504)),(x+8,y+28))
    fy = top+cellh*math.ceil(len(panels)/3)+12
    for i,line in enumerate(lines):
        d.text((12,fy+20*i),line,fill='black',font_size=15)
    return save(out,path)


def build_review_option(run, rec, data, crop_id, box, fixed_default=False):
    assert rec['geometry_status'] == 'PASS_GEOMETRY_ONLY'
    checks = crop_checks(rec,data,box)
    assert checks['technical_pass']
    im,union,edit,_ = data
    out = run/'input_assets'/rec['source_id']/crop_id
    out.mkdir(parents=True,exist_ok=False)
    rgb = im.crop(box).resize((512,512),Image.Resampling.LANCZOS)
    mask = edit.crop(box).resize((512,512),Image.Resampling.NEAREST)
    xr = rgb.resize((1024,1024),Image.Resampling.LANCZOS)
    xm = mask.resize((1024,1024),Image.Resampling.NEAREST)
    images = {'native_rgb': im, 'native_polygon': union, 'native_edit_mask': edit,
              'common_rgb_512': rgb, 'common_edit_mask_512': mask, 'sdxl_rgb_1024': xr,'sdxl_mask_1024': xm}
    files = rec['m0_files'].copy() if fixed_default else {k:save(v,out/(k+'.png')) for k,v in images.items()}
    native_crop,poly_crop,edit_crop = im.crop(box),union.crop(box),edit.crop(box)
    eff = Image.fromarray((np.asarray(mask)>=128).astype(np.uint8)*255)
    eff1024 = eff.resize((1024,1024),Image.Resampling.NEAREST)
    auxiliary = {'native_crop': native_crop, 'native_polygon_crop': poly_crop,
                 'native_edit_mask_crop': edit_crop, 'effective_edit_mask_512': eff,
                 'effective_edit_mask_1024': eff1024}
    for name,img in auxiliary.items():
        files[name] = save(img,out/(name+'.png'))
    overlay = im.copy()
    d = ImageDraw.Draw(overlay)
    for p in rec['polygon_details']:
        d.line(p['points_native_xy']+[p['points_native_xy'][0]],fill='red',width=3)
    d.rectangle((box[0],box[1],box[2]-1,box[3]-1),outline='cyan',width=3)
    a = np.asarray(rgb).copy()
    active = np.asarray(eff)>0
    a[active] = (.45*a[active]+.55*np.array([255,0,0])).astype(np.uint8)
    overlay_effective = Image.fromarray(a)
    files['effective_mask_on_rgb_512'] = save(overlay_effective,out/'effective_mask_on_rgb_512.png')
    panels = [im,overlay,native_crop,union,edit,poly_crop,edit_crop,eff,overlay_effective,rgb,mask,xr,xm,eff1024]
    titles = ['1 Native Original RGB','2 Polygons RED / frozen crop CYAN','3 Native Crop (real pixels)',
              '4 Original Polygon Mask native','5 Native Edit: dilation7 + blur1','6 Polygon Mask in crop',
              '7 Grayscale Edit Mask in crop','8 Effective Mask512 >=128','9 Effective repair pixels RED',
              '10 DreamShaper planned RGB512','11 DreamShaper planned Mask512','12 SDXL planned RGB1024',
              '13 SDXL planned Mask1024','14 SDXL Effective Mask1024']
    footer = [f"Crop {crop_id} xyxy half-open={box}; all polygons/support retained=100%; padding=0; whole-context semantics PENDING.",
              'Masks: white=edit; black=preserve. Gray feather becomes binary at >=128 in installed Diffusers; no inference performed.',
              '512 and 1024 panels are previews; exact PNGs and hashes are included in ZIP for pixel-level checking.',
              'Native image SHA256: '+rec['source_sha256'], 'Annotation SHA256: '+rec['annotation_sha256'],
              'Native polygon pixel SHA256: '+pixel(union), 'RGB512 pixel SHA256: '+files['common_rgb_512']['pixel_sha256'],
              'Mask512 pixel SHA256: '+files['common_edit_mask_512']['pixel_sha256'],
              'Selection reason: fixed deterministic geometry only; no generated-output ranking; all targets retained.',
              *['PENDING: '+s for s in RISKS]]
    preview = sheet(panels,titles,run/'input_protocol_review_contact_sheets'/f"{rec['source_id']}_{crop_id}.png",
                    f"Source {rec['source_id']} / {rec['class']} / {crop_id} / GEOMETRY PASS ONLY",footer)
    return {'crop_id': crop_id, 'box_xyxy_half_open': box,
            'kind': 'IMMUTABLE_M0_DEFAULT_REFERENCE' if fixed_default else ('DETERMINISTIC_DEFAULT' if crop_id == 'default' else 'HUMAN_COMPARISON_PROPOSAL_ONLY'),
            'generation_rule': 'M0 L=min(W,H); no padding; all polygons; edit=MaxFilter7/GaussianBlur1; RGB LANCZOS; mask NEAREST',
            'files': files,'geometry_checks': checks, 'contact_sheet': preview,
            'human_decision': 'PENDING', 'used_for_inference': False, 'cross_model_same_crop': True}


def static_protocol_check():
    proto = read(M0/'frozen_model_comparison_protocol.json')
    assert proto['prompt'] == 'restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting'
    assert [proto[k] for k in ['negative_prompt','steps','guidance_scale','strength','seed','precision']] == [None,20,7.5,.8,130500,'float16']
    assert proto['resolution'] == {'DreamShaper':512,'SDXL':1024}
    assert proto['scheduler'] == {'DreamShaper':'DEISMultistepScheduler','SDXL':'EulerDiscreteScheduler'}
    models = read(M0/'model_asset_preflight.json')
    verifications = []
    for name,m in models.items():
        root = Path(m['local_path'])
        for f in m['files']:
            p = root/f['path']
            assert p.resolve().is_relative_to(root) and p.stat().st_size == f['bytes'] and sha(p) == f['sha256']
        assert proto['assets'][name]['fingerprint'] == m['content_fingerprint']
        verifications.append({'model': name, 'files_rehashed': len(m['files']), 'bytes_rehashed': sum(f['bytes'] for f in m['files']),
                              'fingerprint': m['content_fingerprint'], 'all_file_hashes_equal_M0': True})
    src = read(M0/'cpu_validation.json')['installed_diffusers_source']
    for f in src:
        assert sha(f['path']) == f['sha256'], 'Installed mask processor/pipeline source changed'
    return {'changed': False, 'm0_protocol_sha256': sha(M0/'frozen_model_comparison_protocol.json'), 'protocol': proto,
            'static_model_files': verifications, 'mask_threshold_source_verification': src,
            'DreamShaper_provenance_limit': 'ModelScope/master; immutable upstream revision not independently established',
            'model_loads': 0, 'model_package_imports': 0, 'new_inference': 0}


def main(run, report_head):
    started = now()
    baseline = git('status','--porcelain=v1','--untracked-files=all').splitlines()
    m0_inventory = {str(p.relative_to(M0)):sha(p) for p in M0.rglob('*') if p.is_file()}
    assert sha(SPLIT) == SPLIT_SHA and sha(LEDGER) == LEDGER_SHA
    ledger = read(LEDGER)
    assert ledger['occupied_calls'] == ledger['actual_calls'] == len(ledger['calls']) == 4
    assert ledger['remaining']['smoke'] == 0 and ledger['remaining']['smoke_pilot_revision'] == 116 and ledger['remaining']['all_local'] == 296
    assert not any(c['status'] in ['RESERVED','STARTED'] for c in ledger['calls'].values())
    rows = list(csv.DictReader(SPLIT.open(encoding='utf-8-sig',newline='')))
    train_rows = [r for r in rows if r['split'] == 'train']
    assert len({r['stem'] for r in train_rows}) == len(train_rows)
    trains = {r['stem']:r for r in train_rows}
    extra_order = sorted([r['stem'] for r in train_rows if r['labels_present'] == 'corrosion' and r['stem'].isdigit()
                          and r['stem'] not in set(PRIORITY+['8','9','23'])], key=int)
    dump(run/'candidate_selection_preregistration.json', {'frozen_at': now(), 'priority_order': PRIORITY,
        'supplement_order': extra_order, 'supplement_max_actual_inspections': 30,
        'stop': 'Stop as soon as TWO new unique geometry-qualified corrosion sources are found',
        'excluded': {'8':'M0 retained','9':'ALL_TARGET_SPAN_EXCEEDS_MAX_SQUARE','23':'NO_ALL_SIDES_NORMAL_CONTEXT_MARGIN'},
        'duplicate_rule': 'Exclude repeated frozen TRAIN group_id or image SHA256 vs retained/accepted candidates',
        'source_count_rule': '2 new =>3+3; 1 new =>3+2; 0 new =>SOURCE_SELECTION_HOLD (3+1)',
        'fallback_after_future_human_review': 'If only 2+2 approved: Crack469 then260;470 held; first two human-accepted corrosion in frozen order',
        'alternative_crop_rule': 'At most two unique interval endpoint origins, lexicographic xy; same L/margins/masks; no output-based selection',
        'new_inference_authorized': 0})
    fixed,data_by_id,m0_proof = verify_m0(trains)
    dump(run/'m0_original_hold_preservation.json',m0_proof)
    print('Verified all four immutable M0 defaults and original exclusions',flush=True)
    protocol = static_protocol_check()
    dump(run/'model_protocol_unchanged.json',protocol)
    print('Rehashed local model bytes without importing model packages',flush=True)
    used_groups = {r['group_id'] for r in fixed.values()}
    used_hashes = {r['source_sha256'] for r in fixed.values()}
    assert len(used_groups) == len(used_hashes) == 4
    checked,qualified = [],[]
    supplements = 0
    for sid in PRIORITY+extra_order:
        is_extra = sid not in PRIORITY
        if is_extra and supplements >= 30:
            break
        row = trains.get(sid)
        if not row or row['labels_present'] != 'corrosion':
            checked.append({'source_id':sid,'candidate_kind':'METADATA_ONLY_SKIP', 'geometry_status':'NOT_INSPECTED',
                            'rejection_reason':'NOT_UNIQUE_PURE_CORROSION_TRAIN'})
            continue
        if row['group_id'] in used_groups or row['sha256'] in used_hashes:
            checked.append({'source_id':sid,'candidate_kind':'METADATA_ONLY_SKIP','group_id':row['group_id'],
                            'geometry_status':'NOT_INSPECTED','rejection_reason':'DUPLICATE_FROZEN_GROUP_OR_IMAGE'})
            continue
        if is_extra:
            supplements += 1
        rec,data = analyze(row,'corrosion','SUPPLEMENT_ASCENDING' if is_extra else 'HISTORICAL_FIXED_ORDER')
        checked.append(rec)
        print('Corrosion candidate',sid,rec['geometry_status'],rec['rejection_reason'],flush=True)
        if rec['geometry_status'] == 'PASS_GEOMETRY_ONLY':
            qualified.append(rec)
            data_by_id[sid] = data
            used_groups.add(rec['group_id'])
            used_hashes.add(rec['source_sha256'])
        if len(qualified) == 2:
            break
    selected = [fixed[x] for x in FIXED]+qualified
    final = [r['source_id'] for r in selected]
    counts = {'Crack':3,'corrosion':1+len(qualified)}
    enough = counts['corrosion'] >= 2
    status = 'EXP13_2_M0_R1_PREPARED_AWAITING_HUMAN_INPUT_REVIEW' if enough else 'EXP13_2_M0_R1_SOURCE_SELECTION_HOLD'
    plan = 'PREFERRED_3_PLUS_3' if len(qualified) == 2 else ('ALLOWED_3_PLUS_2' if len(qualified) == 1 else 'HOLD_3_PLUS_1')
    dump(run/'corrosion_replacement_candidate_details.json', {'inspected_or_skipped':checked})
    audit_fields = ['source_id','candidate_kind','group_id','native_wh','polygon_count','polygon_union_bbox',
                    'geometry_status','rejection_reason','crop_xyxy_half_open','source_sha256','annotation_sha256','default_geometry_checks']
    csvout(run/'corrosion_replacement_candidate_audit.csv',audit_fields,[{k:c.get(k) for k in audit_fields} for c in checked])
    dump(run/'revised_source_selection.json', {'status':status,'plan':plan,'final_source_ids':final,'class_counts':counts,
        'retained_m0':FIXED,'new_corrosion_geometry_candidates':[r['source_id'] for r in qualified],
        'selected_group_ids':[r['group_id'] for r in selected],'unique_group_and_image_hashes':True,
        'physical_part_independence':'UNPROVEN; different TRAIN groups/files are not proven different physical parts',
        'supplement_actual_inspections':supplements,'priority_raw_inspections':sum(r['candidate_kind']=='HISTORICAL_FIXED_ORDER' for r in checked),
        'all_semantic_eligibility':'PENDING_USER_REVIEW','semantic_approved_sources':[],
        'fallback_2_plus_2':{'Crack_keep':['469','260'],'Crack_hold':['470'],
                           'corrosion_preference':['8']+[r['source_id'] for r in qualified],
                           'condition':'Only after explicit human decisions leave >=2 acceptable corrosion and469/260 acceptable; otherwise HOLD'},
        'new_calls_authorized':0,'future_proposed_calls':2*len(selected) if enough else 0})
    (run/'input_protocol_review_contact_sheets').mkdir()
    reviews,alt_audit = [],[]
    for rec in selected:
        sid = rec['source_id']
        default = rec['crop_xyxy_half_open']
        opts = [build_review_option(run,rec,data_by_id[sid],'default',default,sid in FIXED)]
        ix,iy = rec['feasible_origin_intervals_xy']
        L = min(rec['native_wh'])
        origins = sorted({(x,y) for x in ix for y in iy} - {(default[0],default[1])})[:2]
        for i,(x,y) in enumerate(origins,1):
            opts.append(build_review_option(run,rec,data_by_id[sid],f'alt_{i}',[x,y,x+L,y+L]))
        reviews.append({'source_id':sid,'class':rec['class'],'source_sha256':rec['source_sha256'],
            'annotation_sha256':rec['annotation_sha256'],'group_id':rec['group_id'],
            'polygon_details':rec['polygon_details'],'geometry_status':'PASS_GEOMETRY_ONLY',
            'options':opts,'selected_crop_id':'PENDING','input_semantic_eligibility':'PENDING_USER_REVIEW',
            'context_and_fov_checks':{r:'PENDING' for r in RISKS},
            'reason':'Retained M0 default' if sid in FIXED else 'First fixed-order unique geometry pass',
            'technical_errors_not_overridable_by_human_pass':True})
        alt_audit.append({'source_id':sid,'default_box':default,'feasible_origin_intervals_xy':rec['feasible_origin_intervals_xy'],
            'alternative_status':'LEGAL_PROPOSALS_PENDING_REVIEW' if origins else 'NO_LEGAL_ALTERNATIVE',
            'default_files_unchanged':True,'proposed_alternatives':[{'crop_id':o['crop_id'],'box':o['box_xyxy_half_open'],
                                                                 'checks':o['geometry_checks'],'files':o['files']} for o in opts[1:]]})
    manifest = {'status':'PENDING_USER_INPUT_PROTOCOL_REVIEW','study_status':status,
        'no_semantic_auto_pass':True,'whole_context_risks':RISKS,'threshold':'uint8 >=128 from verified installed Diffusers 0.35.1 source',
        'geometry_cannot_be_waived':True,'all_sources':reviews,
        'scientific_context':'S2 compact ROI/reflect padding failed; user accepted S5 Crack260 under historical center crop; multiple factors differ, crop-only causality not established',
        'review_authority':'USER highest semantic decision source; separate ChatGPT Gate and explicit M1 authorization still required',
        'no_generation_outputs_in_package':True}
    dump(run/'human_input_protocol_review_manifest.json',manifest)
    dump(run/'semantic_input_review_manifest.json',{'status':'PENDING_USER_REVIEW','selection':final,
         'all_sources':reviews,'review_owner':'USER','approved_sources':[],'executable_now':0})
    dump(run/'input_crop_alternative_audit.json',{'rules_unchanged':True,'maximum_alternatives_per_source':2,'sources':alt_audit})
    csvout(run/'human_input_review_template.csv',REVIEW_FIELDS,
           [{k:(sid if k=='SOURCE_ID' else PENDING) for k in REVIEW_FIELDS} for sid in final])
    dump(run/'shared_input_hash_verification.json',{'existing_m0':m0_proof['retained'],
         'all_selected_crop_options':[{'source_id':r['source_id'],'options':r['options']} for r in reviews],
         'no_existing_input_changed':True,'raw_source_and_annotation_hashes_unchanged':True,
         'model_protocol_unchanged':True,'cross_model_crops_identical':True})
    call_fields = ['source_id','class','model','resolution','crop_id','new_calls_now','future_status']
    csvout(run/'updated_planned_call_matrix.csv',call_fields,
           [{'source_id':r['source_id'],'class':r['class'],'model':m,'resolution':res,'crop_id':'PENDING_USER_CHOICE',
             'new_calls_now':0,'future_status':'BLOCKED_USER_INPUT_REVIEW_AND_SEPARATE_M1_GATE'}
            for r in selected for m,res in [('DreamShaper',512),('SDXL',1024)]])
    assert sha(LEDGER) == LEDGER_SHA
    dump(run/'budget_preservation.json',{'authoritative_ledger':str(LEDGER),'sha256_before':LEDGER_SHA,'sha256_after':sha(LEDGER),
         'occupied_calls':4,'actual_calls':4,'smoke_remaining':0,'aggregate_remaining':116,'all_local_remaining':296,
         'new_model_calls':0,'model_loads':0,'downloads':0,'gpt_fallback':0,'retry':0,'reservations_created':0,'new_successor_created':False,
         'proposed_future_calls':2*len(selected) if enough else 0,'executable_now':0,
         'future_rule':'Separate M1 authorization only; single locked successor inherits complete S5 history; atomic two-call pair reserve; STARTED persisted before inference; failure charged; no retry/no replacement/no incomplete-pair win; Pilot/Revision not Smoke'})
    zip_path = run/'input_protocol_review_package.zip'
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as z:
        for name in ['human_input_protocol_review_manifest.json','semantic_input_review_manifest.json','input_crop_alternative_audit.json',
                     'human_input_review_template.csv','revised_source_selection.json','corrosion_replacement_candidate_audit.csv']:
            z.write(run/name,name)
        for review in reviews:
            for opt in review['options']:
                prefix = f"sources/{review['source_id']}/{opt['crop_id']}"
                for key,f in opt['files'].items():
                    z.write(REPO/f['path'],f'{prefix}/{key}.png')
                z.write(REPO/opt['contact_sheet']['path'],f"input_protocol_review_contact_sheets/{review['source_id']}_{opt['crop_id']}.png")
        assert z.testzip() is None
    zip_info = {'path':str(zip_path),'sha256':sha(zip_path),'bytes':zip_path.stat().st_size,
                'review_sheet_count':sum(len(r['options']) for r in reviews),'source_count':len(selected),'controlled_server_only':True}
    # Verify every opened TRAIN source again after all exports. No raw file is opened for writing.
    for rec in list(fixed.values())+[r for r in checked if 'source_path' in r]:
        assert sha(rec['source_path']) == rec['source_sha256'] and sha(rec['annotation_path']) == rec['annotation_sha256']
    assert m0_inventory == {str(p.relative_to(M0)):sha(p) for p in M0.rglob('*') if p.is_file()}
    assert sha(SPLIT) == SPLIT_SHA and sha(LEDGER) == LEDGER_SHA
    forbidden = sorted(n for n in sys.modules if n.split('.')[0] in ['torch','diffusers','transformers','accelerate'])
    assert not forbidden
    after = git('status','--porcelain=v1','--untracked-files=all').splitlines()
    prefix = 'results/exp13_2_m0_r1/'+run.name+'/'
    historical = lambda entries: [e for e in entries if prefix not in e]
    assert historical(baseline) == historical(after), 'Historical Git status entries changed'
    dump(run/'server_git_preservation.json',{'status_before':baseline,'status_after':after,
         'excluded_only_new_result_prefix':prefix,'historical_entries_unchanged':True,
         'm0_file_hashes_unchanged':True,'m0_inventory':m0_inventory,
         'server_head':git('rev-parse','HEAD'),'server_branch':git('branch','--show-current'),'sync_performed':False})
    dump(run/'provenance.json',{'started_at':started,'finished_at':now(),'report_start_head':report_head,
         'execution_server_head':git('rev-parse','HEAD'),'script_sha256':sha(__file__),'split_manifest_sha256':sha(SPLIT),
         'packages':{n:importlib.metadata.version(n) for n in ['numpy','pillow']},'python':platform.python_version(),
         'model_packages_imported':forbidden,'opened_train_source_ids':FIXED+[r['source_id'] for r in checked if 'source_path' in r],
         'val_access':0,'test_access':0,'raw_write':0,'split_write':0,'model_load':0,'new_inference':0,'model_download':0,
         'ssl_training':0,'yolo_training':0,'trusted_pair_freeze':0,'historical_git_entries_unchanged':True,
         'm0_hash_inventory_unchanged':True,'disk_free_bytes_after':shutil.disk_usage(run).free,
         'controlled_zip':zip_info,'no_paper_repo_access':True,'no_git_cleanup_or_server_sync':True})
    dump(run/'summary.json',{'status':status,'plan':plan,'final_source_ids':final,'class_counts':counts,
         'new_corrosion_sources':[r['source_id'] for r in qualified],'geometry_pass_count':len(selected),
         'minimum_2_plus_2_met':enough,'preferred_3_plus_3_met':len(qualified)==2,
         'semantic_pending_count':len(selected),'semantically_approved':0,'human_input_review_gate':'PENDING',
         'proposed_future_calls':2*len(selected) if enough else 0,'executable_calls':0,'new_calls':0,
         'm0_defaults_preserved':FIXED,'m0_exclusions_preserved':['9','23'],'review_zip':zip_info,
         'recommend_or_authorize_m1':False,'stop':'Await ChatGPT and user input protocol review; M1 needs separate authorization'})
    print(json.dumps({'status':status,'selected':final,'review_zip':zip_info},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--report-head',required=True)
    a = ap.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id) and re.fullmatch(r'[a-f0-9]{40}',a.report_head)
    run = REPO/'results/exp13_2_m0_r1'/a.run_id
    run.mkdir(parents=True,exist_ok=False)
    code = 1
    with (run/'stdout.log').open('x') as out,(run/'stderr.log').open('x') as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:
                main(run,a.report_head)
                code = 0
            except Exception as exc:
                traceback.print_exc()
                if not (run/'summary.json').exists():
                    dump(run/'summary.json',{'status':'EXP13_2_M0_R1_PROTOCOL_HOLD','error':str(exc),
                         'model_load':0,'new_inference':0,'execution_not_authorized':True})
    dump(run/'execution_exit.json',{'exit_code':code,'captured_at':now(),'stdout':'stdout.log','stderr':'stderr.log'})
    print(str(run),'exit_code=',code)
    raise SystemExit(code)
