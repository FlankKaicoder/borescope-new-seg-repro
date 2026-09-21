#!/usr/bin/env python3
import csv, importlib.util, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pilot_base', HERE/'exp13_diffusion_pilot.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
MODEL_PATH='/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting'
MODEL_ID='aiplayground/dreamshaper-8-inpainting'
REVISION='master'

def prepare_r1(run):
    base.MODEL=MODEL_PATH; base.REV=REVISION
    rows=base.prepare(run)
    manifest=run/'manifests/pilot_manifest.csv'
    rows=list(csv.DictReader(manifest.open(newline='')))
    for r in rows:
        r['model_name']=MODEL_ID
        r['model_revision']=REVISION
        r['parent_failed_run']='run_20260921_120819'
        r['recovery_reason']='HF_NETWORK_TIMEOUT'
        r['model_acquisition_source']='ModelScope'
    fields=list(rows[0])
    for f in ('parent_failed_run','recovery_reason','model_acquisition_source'):
        if f not in fields: fields.append(f)
    with manifest.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    (run/'recovery_config.json').write_text(json.dumps({'parent_failed_run':'run_20260921_120819','recovery_reason':'HF_NETWORK_TIMEOUT','model_acquisition_source':'ModelScope','model_id':MODEL_ID,'revision':REVISION,'snapshot_path':MODEL_PATH,'model_identity_changed':True},indent=2)+'\n')
    return rows

def generate_r1(run,only=None):
    base.MODEL=MODEL_PATH; base.REV=REVISION
    return base.generate(run,only)

def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--generate',action='store_true');a.add_argument('--only',nargs='*');x=a.parse_args();run=Path(x.run)
    if x.prepare: prepare_r1(run)
    if x.generate: generate_r1(run,set(x.only) if x.only else None)
if __name__=='__main__': main()
