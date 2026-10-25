#!/usr/bin/env python3
"""Independent CPU validation of an existing M0-R1 input-review ZIP (no rebuild)."""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
LEDGER = REPO/'results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json'
LEDGER_SHA = '3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94'


def sha(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    assert run.is_relative_to(REPO/'results/exp13_2_m0_r1')
    load = lambda n: json.loads((run/n).read_text())
    summary = load('summary.json')
    manifest = load('human_input_protocol_review_manifest.json')
    started = datetime.now(timezone.utc).isoformat()
    zp = Path(summary['review_zip']['path'])
    assert sha(zp.read_bytes()) == summary['review_zip']['sha256']
    expected, checked = set(), []
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        for n in ['human_input_protocol_review_manifest.json','semantic_input_review_manifest.json',
                  'input_crop_alternative_audit.json','human_input_review_template.csv',
                  'revised_source_selection.json','corrosion_replacement_candidate_audit.csv']:
            expected.add(n)
            assert sha(z.read(n)) == sha((run/n).read_bytes())
        for source in manifest['all_sources']:
            assert source['input_semantic_eligibility'] == 'PENDING_USER_REVIEW' and source['selected_crop_id'] == 'PENDING'
            for opt in source['options']:
                assert opt['geometry_checks']['technical_pass'] and opt['cross_model_same_crop']
                assert opt['geometry_checks']['native_polygon_pixels_missing_after_effective_mask_roundtrip'] == 0
                assert opt['geometry_checks']['polygon512_pixels_missing_from_effective_mask'] == 0
                pixels = {}
                for key,f in opt['files'].items():
                    name = f"sources/{source['source_id']}/{opt['crop_id']}/{key}.png"
                    expected.add(name)
                    data = z.read(name)
                    assert sha(data) == f['file_sha256'] == sha((REPO/f['path']).read_bytes())
                    with Image.open(io.BytesIO(data)) as im:
                        arr = np.asarray(im)
                        assert sha(arr.tobytes()) == f['pixel_sha256']
                        assert im.mode == f['mode'] and list(arr.shape) == f['shape'] and str(arr.dtype) == f['dtype']
                        pixels[key] = arr.copy()
                assert np.array_equal(pixels['effective_edit_mask_512'],(pixels['common_edit_mask_512']>=128).astype(np.uint8)*255)
                assert np.array_equal(pixels['effective_edit_mask_1024'],(pixels['sdxl_mask_1024']>=128).astype(np.uint8)*255)
                assert np.array_equal(pixels['sdxl_mask_1024'],np.repeat(np.repeat(pixels['common_edit_mask_512'],2,0),2,1))
                assert np.array_equal(pixels['effective_edit_mask_1024'],np.repeat(np.repeat(pixels['effective_edit_mask_512'],2,0),2,1))
                name = f"input_protocol_review_contact_sheets/{source['source_id']}_{opt['crop_id']}.png"
                expected.add(name)
                assert sha(z.read(name)) == opt['contact_sheet']['file_sha256']
                checked.append({'source_id': source['source_id'], 'crop_id': opt['crop_id'],
                                'all_zip_and_disk_file_pixel_hashes_equal': True, 'effective_masks_exact': True})
        assert set(z.namelist()) == expected
        entry_count = len(expected)
    template = list(csv.DictReader((run/'human_input_review_template.csv').open()))
    assert [r['SOURCE_ID'] for r in template] == summary['final_source_ids']
    assert all(v == 'PENDING' for r in template for k,v in r.items() if k != 'SOURCE_ID')
    matrix = list(csv.DictReader((run/'updated_planned_call_matrix.csv').open()))
    assert len(matrix) == 2*len(template) == summary['proposed_future_calls']
    assert all(r['new_calls_now'] == '0' and r['crop_id'] == 'PENDING_USER_CHOICE' for r in matrix)
    assert len(checked) == summary['review_zip']['review_sheet_count']
    assert sha(LEDGER.read_bytes()) == LEDGER_SHA
    forbidden = sorted(n for n in sys.modules if n.split('.')[0] in ['torch','diffusers','transformers','accelerate'])
    assert not forbidden
    payload = {'status':'PASS_PACKAGE_TECHNICAL_ONLY', 'started_at':started,
               'finished_at':datetime.now(timezone.utc).isoformat(),
               'script_sha256':sha(Path(__file__).read_bytes()), 'zip_sha256':sha(zp.read_bytes()),
               'zip_entries':entry_count, 'source_count':len(template),'review_sheet_count':len(checked),
               'source_crop_options':checked, 'all_human_fields_pending':True,
               'pending_human_cells':sum(k != 'SOURCE_ID' for r in template for k in r),
               'future_planned_slots':len(matrix),'slots_executable_now':0,
               'authoritative_ledger_sha256_unchanged':LEDGER_SHA,
               'model_packages_imported':forbidden,'new_calls':0,'model_loads':0,
               'val_test_files_opened':0,'raw_files_written':0,
               'visual_layout_QA':{'inspected_sheets':['1_default','2_default','260_alt_2'],
                   'scope':'Assistant layout/context-risk spot check only; not USER semantic approval',
                   'observations':{'1_default':'UI crosshair overlay near one target and logo inside crop; multiple real structure edges require human review',
                                   '2_default':'GE/UI overlays, dark cavity and real metal edge remain in whole crop; geometric retention is not eligibility',
                                   '260_alt_2':'Alternate crop changes context and still includes a real dark-cavity/metal boundary; all decisions pending'}},
               'semantic_approved_sources':0,'no_zip_or_historical_artifact_modified':True}
    target = run/'input_review_package_validation.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(payload,f,indent=2)
        f.write('\n')
    print(json.dumps({'status':payload['status'],'entries':entry_count,'options':len(checked),
                      'human_pending_cells':payload['pending_human_cells'],'zip_sha256':payload['zip_sha256']}))
