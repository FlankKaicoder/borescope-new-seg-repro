#!/usr/bin/env python3
"""Prepare a CPU-only, TRAIN-only audit and bounded review package for historical R1/R2 outputs."""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def font(size):
    for candidate in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not args.run_id.startswith('20261025T') or not args.run_id.endswith('_a3'):
        raise SystemExit('Use a unique 20261025T*Z_a3 run id')
    a0 = root / 'results/exp13_restoration_feasibility/a0_20261021_081812'
    qa = root / 'results/exp13_restoration_feasibility'
    r1 = root / 'results/exp13_diffusion_pilot/run_20260921_124440'
    r2 = root / 'results/exp13_diffusion_diagnosis/r2_20260921_135903'
    ledger = root / 'results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json'
    out = root / 'results/exp13_2_a3_dreamshaper_review' / args.run_id
    out.mkdir(parents=True, exist_ok=False)

    idx = read_csv(a0 / 'source_variant_review_index.csv')
    integrity = read_csv(a0 / 'file_integrity.csv')
    spatial = read_csv(a0 / 'spatial_correspondence_report.csv')
    diffs = read_csv(a0 / 'quantitative_differences.csv')
    decisions = {}
    decision_files = [qa/'qa_20261023_batch01_user_review_overlay.csv', qa/'qa_20261023_batch02_user_review_overlay.csv']
    for p in decision_files:
        for row in read_csv(p):
            decisions[row['variant_id']] = {k: row.get(k, '') for k in ['user_decision', 'user_intent_summary', 'acceptance_scope', 'unresolved_technical_issues', 'record_source', 'recorded_utc']}

    # Crosswalk original A0 source/variant records to immutable R1/R2 manifests and A2 decision overlays.
    inventory = []
    sources = defaultdict(list)
    for row in idx:
        v = dict(row)
        v['decision_record'] = decisions.get(row['variant_id'], {})
        v['decision_status'] = v['decision_record'].get('user_decision', 'NO_A2_DECISION_RECORD')
        v['independence_unit'] = row['source_group_id']
        v['formal_quality_status'] = 'PENDING; historical visual decision is not a quality PASS'
        inventory.append(v)
        sources[row['source_group_id']].append(v)
    if len(inventory) != 88 or len(sources) != 20:
        raise RuntimeError(f'Historical inventory mismatch: {len(inventory)} variants, {len(sources)} sources')
    r1_rows = [x for x in inventory if x['historical_stage'] == 'R1']
    r2_rows = [x for x in inventory if x['historical_stage'] == 'R2']
    if len(r1_rows) != 40 or len(r2_rows) != 48:
        raise RuntimeError('R1/R2 count mismatch')
    if sum(bool(x['decision_record']) for x in inventory) != 16:
        raise RuntimeError('Expected exactly 16 A2 decision links')

    fields = list(inventory[0].keys())
    write_csv(out/'historical_88_variant_inventory.csv', inventory, fields)
    write_json(out/'reviewed_16_variant_manifest.json', {
        'source': 'A0 index joined to Batch01/02 user decision overlays by variant_id',
        'decision_source_classification': 'USER_REVIEW_DECISION_HISTORY_NOT_INDEPENDENT_EXPERT_REVIEW',
        'variants': [{**{k:v.get(k,'') for k in ['class','source_id','source_group_id','historical_stage','run_id','variant_id','model','model_revision','strength','prompt_id','seed','mask_type','mask_dilation_px','source_image','original_defect_mask','edit_mask','output_image']}, **v['decision_record'], 'trusted_pair_frozen':'false','restoration_quality_gate_pass':'false'} for v in inventory if v['decision_record']]
    })
    write_json(out/'historical_user_decisions_crosswalk.json', {
        'source':'CHATGPT_PRELIMINARY_VISUAL_REVIEW plus explicit user decisions archived in A2 overlays',
        'scope':'8 source groups / 16 R1 variants only; R2 outputs are diagnostic candidates without A2 decisions',
        'counts':dict(Counter(v['decision_status'] for v in inventory if v['decision_record'])),
        'records':[{'variant_id':v['variant_id'],'source_group_id':v['source_group_id'],'stage':v['historical_stage'],**v['decision_record']} for v in inventory if v['decision_record']],
        'constraints':['No historical decision is promoted to formal restoration-quality PASS.','Crop-only acceptance stays crop-only.','Corrosion 8 v1 rejection and v2 reservation remain separate.','Crack 470 historical warning/text conflict remains unresolved; no decision is rewritten.']
    })

    # A0 audit is the authoritative immutable technical source; summarize retained masks and background-change statistics without semantic classification.
    spatial_by_variant = {x['variant_id']:x for x in spatial}
    diff_by_variant = {x['variant_id']:x for x in diffs}
    crop_items=[]
    for v in inventory:
        s=spatial_by_variant[v['variant_id']]
        d=diff_by_variant[v['variant_id']]
        crop_items.append({'variant_id':v['variant_id'],'source_group_id':v['source_group_id'],'stage':v['historical_stage'],'native_crop_xyxy':s.get('native_crop_xyxy'),'native_source_size':s.get('source_size'),'original_mask_pixels':None,'mask_pixels_inside_crop':None,'mask_pixels_outside_crop':None,'retained_fraction':s.get('original_mask_retained_fraction_native_crop'),'retention_definition':'A0 native polygon raster area retained after historical center crop; not restoration success','full_image_mae_rgb_0_255':d.get('full_mae_rgb_0_255'),'defect_mask_mae':d.get('defect_mask_mae'),'edit_support_mae':d.get('edit_support_mae'),'outside_edit_support_mae':d.get('outside_edit_support_mae'),'outside_support_plus_8px_changed_gt5_fraction':d.get('outside_support_plus_8px_changed_gt5_fraction'),'outside_roi_bbox_mae':d.get('outside_roi_bbox_mae'),'note':'Pixel difference is not semantic quality; historical native images may be centrally cropped.'})
    write_json(out/'crop_and_mask_coverage_audit.json', {'source':'A0 spatial correspondence report; crop fractions are read from its archived independently-recomputed column','candidate_count':len(crop_items),'partially_cropped_source_groups':sorted({x['source_group_id'] for x in crop_items if x['retained_fraction'] and float(x['retained_fraction'])<1.0}),'candidates':crop_items,'limitations':['Detailed mask counts and crop coordinates remain in the linked A0 spatial report; no new raster audit is claimed here.','Retained fraction measures annotation raster area retained by center crop, not restoration success.','Cropped-away annotations are outside accepted restoration scope.']})
    write_json(out/'background_preservation_audit.json', {'source':'A0 quantitative_differences.csv','candidate_count':len(diffs),'metrics':{'pixel_definition':'mean(abs(RGB source - RGB output)), range 0..255','available_fields':list(diffs[0].keys()),'nonzero_outside_support_candidates':sum(float(x['outside_edit_support_mae'])>0 for x in diffs),'semantic_background_certification':'NOT_PERFORMED'},'candidate_metrics':diffs,'limitations':['Mask-outside pixel differences are nonzero for historical candidates; edit support is not the only changed region.','Historical resampling, feathering, model reconstruction and compression cannot be fully separated by these summaries.','No metric assigns candidate semantic PASS.']})

    # Build source review pages from pre-existing A0 sheets; do not recalculate or mutate historical sheet contents.
    review_root=out/'review_sheets'
    review_root.mkdir()
    by_source={}
    for key, rows in sources.items():
        if not any(v['decision_record'] for v in rows):
            continue
        first=rows[0]; cls=first['class']; sid=first['source_id']
        base_sheet=a0/'contact_sheets'/f'{cls.lower()}_{sid}_review.png'
        if not base_sheet.is_file():
            raise FileNotFoundError(base_sheet)
        with Image.open(base_sheet) as im:
            original=im.convert('RGB')
        variants=sorted(rows,key=lambda x:(x['historical_stage'],x['variant_id']))
        decision_lines=[]
        for v in variants:
            dr=v['decision_record']
            if dr:
                decision_lines.append(f"{v['variant_id']}: {dr['user_decision']} | {dr['acceptance_scope']}")
        header_h=120+30*len(decision_lines)
        sheet=Image.new('RGB',(original.width,original.height+header_h),'#e8edf0')
        sheet.paste(original,(0,header_h))
        draw=ImageDraw.Draw(sheet)
        draw.text((18,12),f'Exp13.2-A3 historical quality review | {cls} source {sid}',font=font(27),fill='#17212b')
        draw.text((18,49),'A0 source sheet below is an immutable historical view. R1 v1/v2 and any R2 diagnostics remain separate; variants are correlated.',font=font(18),fill='#17212b')
        draw.text((18,77),'User decisions are history only; crop limits/reservations persist. Formal quality fields = PENDING; no candidate is admitted.',font=font(18),fill='#8b1e1e')
        y=112
        for line in decision_lines:
            draw.text((28,y),line,font=font(16),fill='#17212b'); y+=28
        name=f'{cls.lower()}_{sid}_a3_review.png'
        p=review_root/name
        sheet.save(p,optimize=True)
        by_source[key]={'class':cls,'source_id':sid,'independence_unit':key,'review_sheet':str(p.relative_to(out)).replace('\\','/'),'review_sheet_sha256':sha(p),'variants':[{'variant_id':v['variant_id'],'stage':v['historical_stage'],'decision':v['decision_status'],'formal_quality':'PENDING'} for v in variants]}

    template=[]
    human_fields=['defect_removed','normal_surface_plausible','background_preserved','geometry_preserved','hallucination_present','editing_artifact_present','crop_scope_satisfied','mask_boundary_acceptable','restoration_quality','reviewer_id','reviewer_comment','final_accept']
    for v in inventory:
        template.append({'class':v['class'],'source_id':v['source_id'],'source_group_id':v['source_group_id'],'historical_stage':v['historical_stage'],'variant_id':v['variant_id'],'prior_user_decision':v['decision_status'],'prior_decision_scope':v['decision_record'].get('acceptance_scope',''),'review_evidence_source':'CHATGPT_PRELIMINARY_VISUAL_REVIEW; user decisions remain separately attributed',**{f:'PENDING' for f in human_fields},'trusted_pair_frozen':'false','restoration_quality_gate_pass':'false'})
    write_csv(out/'restoration_quality_review_template.csv',template,list(template[0]))
    write_json(out/'historical_20_source_summary.json',{'sources_total':len(sources),'sources_by_class':dict(Counter(k.split(':')[0] for k in sources)),'r1_variants':len(r1_rows),'r2_diagnostic_variants':len(r2_rows),'a2_decision_variants':16,'independent_unit':'(class, source_id)','source_groups':[{'source_group_id':k,'candidate_count':len(v),'r1_count':sum(x['historical_stage']=='R1' for x in v),'r2_count':sum(x['historical_stage']=='R2' for x in v),'a2_decision_count':sum(bool(x['decision_record']) for x in v),'variant_ids':[x['variant_id'] for x in v]} for k,v in sorted(sources.items())]})
    write_json(out/'trusted_pair_readiness_matrix.json',{'trusted_pair_count':0,'quality_gate_pass_count':0,'candidate_rows':88,'source_groups':20,'historical_user_decisions_are_admission':'false','proposed_a1_threshold':{'total_independent_sources':10,'Crack':5,'corrosion':5,'status':'PROPOSED_NOT_FROZEN'},'gaps':['No complete multi-axis variant-level quality decision across defect removal, plausible metal, structure, background, crop scope and artifacts.','P/N/I regions and reviewer-confirmed repair/background support are not frozen.','Same-process normal pseudo-edits and non-Diffusion controls are absent.','Annotation-derived polygon supervision must be disclosed.'],'authorization':{'formal_training':'NO','SSL':'NO','Exp13.2-B':'NO','new_generation':'NO'}})

    ledger_data=json.loads(ledger.read_text(encoding='utf-8'))
    budget={'inherited_from':str(ledger.relative_to(root)),'source_ledger_sha256':sha(ledger),'occupied_calls':ledger_data['occupied_calls'],'actual_calls':ledger_data['actual_calls'],'remaining_smoke':ledger_data['remaining']['smoke'],'new_sdxl_calls':0,'new_dreamshaper_calls':0,'gpu_model_loads':0,'gpt_fallback_calls':ledger_data['gpt_fallback_calls'],'val_test_access':0,'raw_dataset_write':0,'ssl_yolo_training':0,'ledger_mutated':False}
    if budget['occupied_calls']!=3 or budget['actual_calls']!=3 or budget['remaining_smoke']!=1:
        raise RuntimeError('Inherited S2 budget does not match expected 3/4')
    write_json(out/'budget_preservation.json',budget)

    # Package is confined to controlled AutoDL results; raw industrial images and generated outputs never enter Git.
    package=out/'dreamshaper_a3_review_package.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(review_root.glob('*.png')):
            z.write(p,arcname='review_sheets/'+p.name)
        z.write(out/'restoration_quality_review_template.csv',arcname='restoration_quality_review_template.csv')
        z.write(out/'historical_user_decisions_crosswalk.json',arcname='historical_user_decisions_crosswalk.json')
    package_info={'path':str(package),'bytes':package.stat().st_size,'sha256':sha(package),'contents':[f'review_sheets/{p.name}' for p in sorted(review_root.glob('*.png'))]+['restoration_quality_review_template.csv','historical_user_decisions_crosswalk.json'],'controlled_location':'AutoDL repository results; excluded from Git'}
    write_json(out/'review_package_manifest.json',{'source_sheets':str(a0/'contact_sheets'),'source_sheet_hashes':{f'{k}':sha(a0/'contact_sheets'/f'{v[0]["class"].lower()}_{v[0]["source_id"]}_review.png') for k,v in sources.items() if any(x['decision_record'] for x in v)},'package':package_info,'source_level_pages':list(by_source.values()),'human_review_note':'ChatGPT preliminary opinions are not user acceptance, expert confirmation, or formal rejection.'})
    write_json(out/'summary.json',{'status':'EXP13_2_A3_DREAMSHAPER_REVIEW_PREPARED_PENDING_CHATGPT_GATE','run_id':args.run_id,'inventory':{'R1':40,'R2_diagnostic':48,'total':88,'source_groups':20,'A2_user_decisions':16,'reviewed_source_groups':8},'trusted_pairs':0,'quality_gate_pass':0,'package':package_info,'budget':budget,'scope':{'train_only':True,'cpu_only':True,'new_inference':0,'new_generation':0,'training':0,'val_test_access':0,'raw_dataset_write':0}})
    prov={'created_utc':datetime.now(timezone.utc).isoformat(),'host':platform.node(),'python':platform.python_version(),'pillow':Image.__version__ if hasattr(Image,'__version__') else 'unknown','numpy':np.__version__,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'inputs':{str(p.relative_to(root)):sha(p) for p in [a0/'source_variant_review_index.csv',a0/'file_integrity.csv',a0/'spatial_correspondence_report.csv',a0/'quantitative_differences.csv',*decision_files,ledger]},'historical_directories_read_only':True,'dataset_images_read':0,'val_test_access':0,'models_loaded':0,'inference_calls':0,'training_steps':0,'raw_data_writes':0,'output_directory':str(out)}
    write_json(out/'provenance.json',prov)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
