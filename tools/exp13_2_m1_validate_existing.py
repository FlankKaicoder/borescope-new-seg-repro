#!/usr/bin/env python3
"""M1 independent CPU-only checks and public export, never diffusion inference.

Model-linked output hashes and pixel statistics remain controlled to prevent
reconstructing blind labels from public numeric fingerprints before review.
Native composites are extra POSTPROCESS_COMPARISON_ONLY diagnostics.
"""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2); f.write('\n')


def img(p):
    with Image.open(p) as im:return np.asarray(im).copy()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); args=ap.parse_args()
    run=Path(args.run_dir).resolve(); repo=run.parents[2]
    assert repo==Path('/root/autodl-tmp/borescope-new-seg-repro')
    ledger=read(run/'budget_successor_ledger.json')
    s5=repo/'results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json'
    old=read(s5); assert sha(s5)=='3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94'
    assert ledger['occupied_calls']==ledger['actual_calls']==16
    assert ledger['remaining']['smoke']==0 and ledger['remaining']['smoke_pilot_revision']==104 and ledger['remaining']['all_local']==284
    assert ledger['caps']==old['caps'] and ledger['history'][:len(old['history'])]==old['history']
    assert all(ledger['calls'][k]==v for k,v in old['calls'].items())
    new=[v for k,v in ledger['calls'].items() if k not in old['calls']]
    assert len(new)==12 and all(c['status']=='SUCCEEDED' and c['stage']=='pilot' and c['real_model_call'] for c in new)
    assert read(run/'execution_exit.json')['exit_code']==0
    metrics=read(run/'call_runtime_records.json'); assert len(metrics)==12
    sources=read(run/'frozen_source_crop_manifest.json')
    commitment=read(run/'blind_mapping_commitment.json')
    key=run/'private/blind_mapping_key.json'
    assert sha(key)==commitment['sha256']
    mapping=read(key)['mapping']
    assert datetime.fromisoformat(commitment['timestamp'])<datetime.fromisoformat(read(run/'execution_attempt.json')['started_at'])
    for c in new:
        assert datetime.fromisoformat(commitment['timestamp'])<datetime.fromisoformat(c['reserved_at'])<=datetime.fromisoformat(c['started_at'])<=datetime.fromisoformat(c['finished_at'])
        events=[h['event'] for h in ledger['history'] if h.get('call_id')==c['call_id']]
        assert events==['RESERVED','STARTED','SUCCEEDED']
    for rec in read(run/'private/output_file_integrity.json'):
        assert (run/rec['path']).stat().st_size==rec['bytes'] and sha(run/rec['path'])==rec['sha256']
    native_records=[]; raw_count=0
    diag=run/'native_postprocess_diagnostics'; diag.mkdir(exist_ok=False)
    for s in sources:
        sid=s['source_id']; fs=s['files']; box=s['crop_xyxy_half_open']
        for rec in fs.values():assert sha(repo/rec['path'])==rec['file_sha256']
        native=img(repo/fs['native_rgb']['path']); support=img(repo/fs['native_edit_mask']['path'])>0
        native_poly=img(repo/fs['native_polygon']['path'])>0
        mask512=img(repo/fs['common_edit_mask_512']['path'])
        effective=img(repo/fs['effective_edit_mask_512']['path'])>0
        assert np.array_equal(mask512>=128,effective)
        pair=[c for c in new if c['source_id']==sid]; assert len(pair)==2
        # Both pair reservations precede either STARTED, and occupy one uninterrupted history group.
        assert max(c['reserved_at'] for c in pair)<=min(c['started_at'] for c in pair)
        reservation_indices=[i for i,h in enumerate(ledger['history']) if h.get('call_id') in {c['call_id'] for c in pair} and h['event']=='RESERVED']
        assert reservation_indices[1]==reservation_indices[0]+1
        for name in ['DreamShaper','SDXL']:
            out=run/'outputs'/sid/name; size=512 if name=='DreamShaper' else 1024
            a=np.load(out/'raw_decoded_float.npy',allow_pickle=False)
            assert a.shape==(size,size,3) and np.isfinite(a).all()
            raw=img(out/'raw_output.png')
            assert np.array_equal(raw,(a*255).round().clip(0,255).astype(np.uint8))
            assert raw.dtype==np.uint8 and raw.std()>0 and raw.mean()>=2
            aligned=img(out/'aligned_512.png')
            expected=np.asarray(Image.fromarray(raw).resize((512,512),Image.Resampling.LANCZOS))
            assert np.array_equal(aligned,expected)
            base=img(repo/fs['common_rgb_512']['path'])
            protected=img(out/'protected_comparison_only_512.png')
            assert np.array_equal(protected[~effective],base[~effective])
            assert np.array_equal(protected[effective],aligned[effective])
            crop=np.asarray(Image.fromarray(aligned).resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS))
            expected_native=native.copy()
            patch=expected_native[box[1]:box[3],box[0]:box[2]]
            patch_support=support[box[1]:box[3],box[0]:box[2]]
            patch[patch_support]=crop[patch_support]
            assert int(np.any(expected_native!=native,axis=2)[~support].sum())==0
            assert np.array_equal(expected_native[native_poly&~support],native[native_poly&~support])
            path=diag/f'{sid}_{name}_native_POSTPROCESS_COMPARISON_ONLY.png'
            Image.fromarray(expected_native).save(path)
            native_records.append({'source_id':sid,'model':name,'shape':list(expected_native.shape),'dtype':str(expected_native.dtype),
                                   'outside_native_edit_support_changes':0,'native_polygon_pixels_outside_support':int((native_poly&~support).sum()),
                                   'path':str(path),'label':'POSTPROCESS_COMPARISON_ONLY_NOT_TRAINING_PAIR'})
            raw_count+=1
    with zipfile.ZipFile(run/'anonymous_blind_review_package.zip') as z:
        assert z.testzip() is None
        names=z.namelist(); assert len(names)==len(set(names))
        assert all('DreamShaper' not in n and 'SDXL' not in n and 'mapping' not in n for n in names)
        for sid in mapping:
            for slot in ['A','B']:
                p=f'{sid}/{slot}_raw_aligned_512.png'
                with Image.open(BytesIO(z.read(p))) as im:
                    assert im.size==(512,512) and im.mode=='RGB' and not im.info
                    assert np.array_equal(np.asarray(im),img(run/'outputs'/sid/mapping[sid][slot]/'aligned_512.png'))
        manifest=json.loads(z.read('blind_file_manifest.json'))
        for p,r in manifest.items():assert hashlib.sha256(z.read(p)).hexdigest()==r['sha256']
        rows=list(csv.DictReader(z.read('human_blind_review_template.csv').decode('utf-8').splitlines()))
        assert len(rows)==6
        assert all(row[f]=='PENDING' for row in rows for f in ['A_QUALITY','B_QUALITY','PREFERENCE','REVIEWER_COMMENT'])
        assert all(row['PAIR_COMPLETE']=='True' for row in rows)
    dump(run/'native_postprocess_geometry_audit.json',{'status':'PASS_POSTPROCESS_DIAGNOSTICS_ONLY','new_inference':0,'records':native_records})
    public=run/'public_export'; public.mkdir(exist_ok=False)
    # Do not publish per-model output hashes, luminance or difference fingerprints before blind review.
    source_files=['snapshot_manifest.json','user_input_decision_overlay.json','frozen_source_crop_manifest.json',
                  'frozen_execution_config.json','environment_preflight.json','preflight_summary.json',
                  'budget_inheritance_preflight.json','model_integrity_preflight.json','blind_mapping_commitment.json',
                  'budget_successor_ledger.json','budget_preservation.json','execution_exit.json','execution_attempt.json',
                  'execution_runtime.json','model_load_records.json','blind_review_package_manifest.json','summary.json',
                  'provenance.json','execution_stdout.log','execution_stderr.log']
    # snapshot_manifest lives in the isolated source snapshot, not the result root.
    code=Path('/root/autodl-tmp/exp13_m1_workspace/20261026T064000Z_m1/code')
    for name in source_files:
        src=code/name if name=='snapshot_manifest.json' else run/name
        shutil.copyfile(src,public/name)
    keep=['call_id','source_id','model','crop_id','status','input_wh','input_rgb_sha256','input_mask_sha256',
          'mask_effective_pixels','started_at','finished_at','seconds','output_float_shape','output_float_dtype',
          'finite','peak_allocated_bytes','peak_reserved_bytes','output_wh','output_uint8_shape','output_uint8_dtype','semantic_quality']
    dump(public/'call_execution_records.json',[{k:v for k,v in r.items() if k in keep} for r in metrics])
    with (public/'runtime_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        fields=['call_id','source_id','model','crop_id','status','seconds','peak_allocated_bytes','peak_reserved_bytes']
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in metrics])
    fields=list(rows[0])
    with (public/'human_blind_review_template.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    report={'status':'PASS_TECHNICAL_ONLY','timestamp':datetime.now(timezone.utc).isoformat(),
            'existing_raw_outputs_verified':raw_count,'finite_decodes':12,'shape_dtype_checks_pass':12,
            'exact_float_to_png_checks_pass':12,'exact_alignment_checks_pass':12,
            'common_512_protected_outside_effective_support_changes':0,
            'native_protected_outside_native_edit_support_changes':0,'native_composites':12,
            'historical_s5_calls_and_history_unchanged':True,'source_pair_reservations_atomic':True,
            'all_started_durable_with_terminal_success':True,'new_calls':12,'inference_in_validator':0,
            'mask_threshold_128_and_2x_correspondence_verified':True,
            'blind_key_commitment_before_inference_verified':True,'blind_slots_match_private_key':True,
            'blind_zip_same_512_slot_dimensions_no_model_names_or_metadata':True,
            'blind_pending_decision_cells':24,'complete_pairs':6,
            'semantic_quality':'PENDING_USER_BLIND_REVIEW','model_selection':'NOT_PERFORMED',
            'public_disclosure':'model-linked output pixel statistics/hashes remain controlled until blind review',
            'public_export_contains_no_images_weights_zip_or_mapping':True}
    dump(run/'independent_validation.json',report); dump(public/'input_output_integrity_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
