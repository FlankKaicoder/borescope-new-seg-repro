#!/usr/bin/env python3
"""Additional M0 polygon topology, sampling and installed-source checks; CPU only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def crossing(a,b,c,d):
    def orient(p,q,r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    # Include touching/collinear nonadjacent segments as a review risk.
    def on(p,q,r):
        return min(p[0],q[0])<=r[0]<=max(p[0],q[0]) and min(p[1],q[1])<=r[1]<=max(p[1],q[1])
    o1,o2,o3,o4 = orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
    return ((o1*o2<0 and o3*o4<0) or any(abs(o)<1e-9 and on(p,q,r) for o,p,q,r in
            [(o1,a,b,c),(o2,a,b,d),(o3,c,d,a),(o4,c,d,b)]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    repo=Path('/root/autodl-tmp/borescope-new-seg-repro')
    run=Path(args.run).resolve()
    assert run.is_relative_to(repo/'results/exp13_2_m0_matched_model_benchmark')
    manifest=json.loads((run/'shared_crop_manifest.json').read_text())
    records=[]
    for c in manifest['candidates']:
        cross=[]
        for p in c['polygon_details']:
            pts=p['points_native_xy']
            if pts[-1]==pts[0]: pts=pts[:-1]
            n=len(pts)
            for i in range(n):
                for j in range(i+1,n):
                    if j==i+1 or (i==0 and j==n-1): continue
                    if crossing(pts[i],pts[(i+1)%n],pts[j],pts[(j+1)%n]):
                        cross.append({'polygon':p['polygon_id'],'edges':[i,j]})
        root=run/c['source_id']
        target=Image.open(root/'native_polygon.png').crop(tuple(c['diagnostic_crop_xyxy']))
        mask=Image.open(root/'common_edit_mask_512.png')
        target512=np.asarray(target.resize((512,512),Image.Resampling.NEAREST))>0
        binary=np.asarray(mask)>=128
        missing512=int(np.count_nonzero(target512 & ~binary))
        native_binary=np.asarray(Image.fromarray(binary.astype(np.uint8)*255).resize(target.size,Image.Resampling.NEAREST))>0
        missing_native=int(np.count_nonzero((np.asarray(target)>0)&~native_binary))
        assert missing512==0 and missing_native==0
        records.append({'source_id':c['source_id'],'nonadjacent_edge_intersections':cross,
            'topology_status':'PASS_SIMPLE_POLYGON' if not cross else 'REVIEW_REQUIRED',
            'polygon512_pixels_missing_from_pipeline_threshold128':missing512,
            'native_cropped_polygon_pixels_missing_after_mask_roundtrip':missing_native,
            'scope':'All polygons' if c['geometry_status']=='PASS_GEOMETRY_ONLY' else 'DIAGNOSTIC_CROP_ONLY_EXCLUDES_LOST_NATIVE_PIXELS',
            'input_png_count':len(list(root.glob('*.png'))),
            'review_png_wh':list(Image.open(root/'input_review.png').size)})
    dist=importlib.metadata.distribution('diffusers')
    paths=['diffusers/image_processor.py','diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py',
           'diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py']
    installed=[]
    for rel in paths:
        p=Path(dist.locate_file(rel)); lines=p.read_text().splitlines()
        hits=[]
        for i,line in enumerate(lines):
            if ('do_binarize=True' in line or 'image[image < 0.5]' in line or
                'image[image >= 0.5]' in line or 'padding_mask_crop: Optional[int] = None' in line):
                hits.append({'line':i+1,'text':line.strip()})
        assert hits
        installed.append({'path':str(p),'sha256':sha(p),'relevant_source_lines':hits})
    ds=Path('/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting/README.md')
    license_lines=[l for l in ds.read_text().splitlines() if l.startswith(('license:','duplicated_from:'))]
    payload={'timestamp':datetime.now(timezone.utc).isoformat(),'new_inference':0,'model_load':0,
        'candidates':records,'installed_diffusers_source':installed,
        'DreamShaper_local_model_card':{'path':str(ds),'sha256':sha(ds),'metadata':license_lines,
            'scope':'Local model card evidence only, immutable upstream revision not proven'},
        'summary':'Both pipelines binarize masks; all retained targets covered at threshold128 before/after common-mask roundtrip. Excluded crop diagnostics are not full-target inputs.',
        'script_sha256':sha(Path(__file__))}
    with (run/'cpu_validation.json').open('x') as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2))
