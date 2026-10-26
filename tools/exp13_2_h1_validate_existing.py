#!/usr/bin/env python3
"""Independent CPU-only validation of completed H1 artifacts; no model imports."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run
    assert run.resolve().is_relative_to(Path('/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_h1_hallucination_replay'))
    cfg=read(run/'frozen_execution_config.json'); pre=read(run/'preflight_summary.json')
    assert sha(run/'frozen_execution_config.json')==pre['frozen_config_sha256']
    assert cfg['source_id']=='23' and cfg['variant_id']=='corrosion_23_mask_A_s055_v2'
    assert cfg['task_mismatch_disclosed'] and cfg['restoration_effectiveness_not_tested']
    assert cfg['executable_new_calls']==1 and not cfg['dreamshaper_replay']
    assert [cfg[k] for k in ['steps','guidance_scale','strength','seed','height','width']]==[20,7.5,.55,17001,512,512]
    artifact=read(run/'historical_artifact_integrity.json')
    for p,h in artifact['protected_file_sha256_before'].items(): assert sha(p)==h
    for rec in cfg['inputs'].values(): assert sha(rec['path'])==rec['sha256']
    art=artifact['artifacts']
    for role,mode,filename,resample in [('source_image_path','RGB','exact_historical_rgb_512.png',Image.Resampling.LANCZOS),
                                     ('inpaint_mask_path','L','exact_historical_edit_mask_512.png',Image.Resampling.NEAREST)]:
        expected=ImageOps.fit(Image.open(art[role]['path']).convert(mode),(512,512),method=resample,centering=(.5,.5))
        assert np.array_equal(np.asarray(expected),np.asarray(Image.open(run/'inputs'/filename)))
    log=read(run/'new_call_execution_log.json'); assert log['status']=='SUCCEEDED' and log['new_sdxl_calls']==1 and log['new_dreamshaper_calls']==0
    assert read(run/'execution_exit.json')['returncode']==0
    inherit=read(run/'budget_inheritance.json'); parent=read(inherit['parent_path'])
    assert sha(inherit['parent_path'])==inherit['parent_sha256']
    ledger=read(run/'budget_successor_ledger.json')
    assert all(ledger['calls'][k]==v for k,v in parent['calls'].items())
    new=[c for k,c in ledger['calls'].items() if k not in parent['calls']]
    assert len(new)==1 and new[0]['status']=='SUCCEEDED' and new[0]['stage']=='revision' and not new[0]['retry']
    events=[h['event'] for h in ledger['history'] if h.get('call_id')==new[0]['call_id']]
    assert events==['RESERVED','STARTED','SUCCEEDED']
    assert read(run/'call_started_evidence.json')['call']['status']=='STARTED'
    assert ledger['occupied_calls']==ledger['actual_calls']==17 and ledger['smoke_reserved']==4
    assert ledger['remaining']['smoke']==0 and ledger['remaining']['smoke_pilot_revision']==103 and ledger['remaining']['all_local']==283
    a=np.load(run/'sdxl_raw_decoded_float.npy',allow_pickle=False)
    assert a.shape==(512,512,3) and np.isfinite(a).all()
    out=np.asarray(Image.open(run/'sdxl_raw_output_512.png'))
    assert np.array_equal((a*255).round().clip(0,255).astype(np.uint8),out)
    base=np.asarray(Image.open(run/'inputs/exact_historical_rgb_512.png'))
    mask=np.asarray(Image.open(run/'inputs/exact_historical_edit_mask_512.png'))>=128
    comparison=read(run/'comparison_summary.json')['measurements']
    for name,arr in [('DreamShaper',np.asarray(Image.open(art['output_path']['path']))),('SDXL',out)]:
        d=np.abs(arr.astype(np.int16)-base.astype(np.int16)).max(axis=2)
        assert int(((d>5)&~mask).sum())==comparison[name]['changed_gt5_outside_effective_mask_pixels']
        p=np.asarray(Image.open(run/'review_package'/f'{name}_protected_POSTPROCESS_ONLY.png'))
        assert np.array_equal(p[~mask],base[~mask]) and np.array_equal(p[mask],arr[mask])
    package=read(run/'review_package_manifest.json')
    assert sha(package['zip_path'])==package['zip_sha256']
    with zipfile.ZipFile(package['zip_path']) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={Path(f['path']).name for f in package['files']}
        for f in package['files']:
            assert hashlib.sha256(z.read(Path(f['path']).name)).hexdigest()==f['sha256']==sha(f['path'])
    assert not read(run/'data_git_preservation.json')['untracked_stat_changes']
    result={'status':'PASS_TECHNICAL_ONLY','checks':['historical protected hashes','exact archived transform',
       'frozen configuration','single real call with zero exit','inherited ledger unchanged','RESERVED STARTED SUCCEEDED order',
       'float output finite and PNG exact','mask-outside measurements','postprocess-only protection','ZIP CRC and member hashes'],
       'model_imports':0,'new_calls':0,'reviewer_semantic_pass':False,'validator_sha256':sha(__file__)}
    with (run/'independent_validation.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
