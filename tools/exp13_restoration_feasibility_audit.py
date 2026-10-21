#!/usr/bin/env python
"""CPU-only, read-only historical inputs; exclusive creation of A0 review assets.

No diffusion/torch imports, model calls, dataset image reads or quality decisions.
Historical preprocessing is reproduced from the two generation scripts as data,
without importing or executing those scripts.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / 'results/exp13_diffusion_pilot/run_20260921_124440'
R2 = ROOT / 'results/exp13_diffusion_diagnosis/r2_20260921_135903'
TRAIN_META = ROOT / 'results/exp13_vlm_semantic_audit/sample_manifest.csv'
GATE = 'RESTORATION_REVIEW_ASSETS_PREPARED_AWAITING_HUMAN_REVIEW'
HUMAN = ['defect_removed', 'normal_surface_plausible', 'background_preserved',
         'geometry_preserved', 'hallucination_present', 'editing_artifact_present',
         'restoration_quality', 'reviewer_comment', 'final_accept']


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def relative(path):
    return str(path.relative_to(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def checked_path(value, run):
    path = Path(value)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError('Manifest reference outside allowed historical run: ' + str(path))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def fit(image, mask=False):
    return ImageOps.fit(image, (512, 512), method=Image.Resampling.NEAREST if mask
                        else Image.Resampling.LANCZOS, centering=(.5, .5))


def load(path, mode):
    with Image.open(path) as im:
        return im.convert(mode)


def crop_box(size):
    w, h = size
    side = min(w, h)
    return ((w-side)/2, (h-side)/2, (w+side)/2, (h+side)/2)


def overlay(image, mask):
    rgb = np.asarray(image).copy()
    alpha = (np.asarray(mask) > 0)[..., None] * .45
    return Image.fromarray((rgb*(1-alpha) + np.array([255, 35, 35])*alpha).astype('uint8'))


def mean(values):
    return round(float(values.mean()), 6) if values.size else ''


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


F16, F20, F26 = font(16), font(20), font(26)


def paste_cell(canvas, image, x, y, size=256):
    tile = ImageOps.contain(image.convert('RGB'), (size, size), Image.Resampling.LANCZOS)
    canvas.paste(tile, (x+(size-tile.width)//2, y+(size-tile.height)//2))


def paired_zoom(raw, out, box):
    raw = ImageOps.contain(raw.crop(box), (252, 256), Image.Resampling.LANCZOS)
    out = ImageOps.contain(out.crop(box), (252, 256), Image.Resampling.LANCZOS)
    image = Image.new('RGB', (512, 280), '#eeeeee')
    image.paste(raw, ((252-raw.width)//2, 24+(256-raw.height)//2))
    image.paste(out, (260+(252-out.width)//2, 24+(256-out.height)//2))
    draw = ImageDraw.Draw(image)
    draw.text((6, 2), 'RAW ROI', font=F16, fill='black')
    draw.text((266, 2), 'CANDIDATE ROI', font=F16, fill='black')
    return image


def review_sheet(rows, destination):
    # The 512px paired ROI column is twice the normal tile width.
    width, header, row_h = 1900, 410, 340
    canvas = Image.new('RGB', (width, header+len(rows)*row_h), '#eceff1')
    draw = ImageDraw.Draw(canvas)
    first = rows[0]
    draw.text((16, 12), f"{first['class']} source {first['source_id']} - grouped historical candidates", font=F26, fill='black')
    raw_native = load(Path(first['_source']), 'RGB')
    orig_native = load(Path(first['_original']), 'L')
    raw_box = raw_native.copy()
    ImageDraw.Draw(raw_box).rectangle(crop_box(raw_native.size), outline='yellow', width=4)
    paste_cell(canvas, raw_box, 16, 88)
    paste_cell(canvas, overlay(raw_native, orig_native), 288, 88)
    draw.text((16, 62), f'Full RAW {raw_native.width}x{raw_native.height}', font=F16, fill='black')
    draw.text((288, 62), 'Original polygon annotation', font=F16, fill='black')
    lines = ['Yellow box: historical central crop before 512x512 scaling.',
             'Below: RAW/masks fitted with the exact historical ImageOps.fit transform.',
             'Absolute difference uses a fixed 4x display gain; 0=black, >=63.75=white.',
             'Same source variants are correlated. No candidate was selected or scored.',
             'RGB LANCZOS, masks NEAREST. Native masks include Gaussian feather sigma 1.',
             'Review cracks/corrosion, invented holes/components, leakage, smoothing and grey borders.']
    for n, line in enumerate(lines):
        draw.text((580, 82+n*32), line, font=F20, fill='black')
    headings = ['RAW (inference view)', 'Original defect overlay', 'Edit mask (soft)',
                'Restoration candidate', 'Absolute difference x4', 'Paired defect ROI zoom']
    for c, label in enumerate(headings):
        draw.text((16+272*c, 376), label, font=F16, fill='black')
    for i, row in enumerate(rows):
        y = header+i*row_h
        draw.text((16, y), f"{row['run_id']} / {row['variant_id']}", font=F20, fill='black')
        draw.text((16, y+27), f"seed={row['seed']}  strength={row['strength']}  prompt={row['prompt_id']}  mask={row['mask_type']}  native dilation={row['mask_dilation_px']}px", font=F16, fill='black')
        raw = fit(load(Path(row['_source']), 'RGB'))
        orig = fit(load(Path(row['_original']), 'L'), True)
        edit = fit(load(Path(row['_edit']), 'L'), True)
        out = load(Path(row['_output']), 'RGB')
        diff = np.abs(np.asarray(raw, dtype='float32')-np.asarray(out, dtype='float32')).mean(2)
        diff_img = Image.fromarray(np.clip(diff*4, 0, 255).astype('uint8'))
        tiles = [raw, overlay(raw, orig), edit, out, diff_img]
        for c, image in enumerate(tiles):
            paste_cell(canvas, image, 16+c*272, y+65)
        canvas.paste(paired_zoom(raw, out, tuple(row['_roi_box'])), (1376, y+55))
    canvas.save(destination, optimize=True)


def overview(groups, class_name, destination):
    # A whole row per source, then R1 and R2 candidates in recorded parameter order.
    selected = [(k, v) for k, v in groups.items() if k[0] == class_name]
    height = 80+sum(64+((len(v)+1+5)//6)*274 for _, v in selected)
    canvas = Image.new('RGB', (1648, height), '#eceff1')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), class_name+' overview - all historical candidates, grouped by source', font=F26, fill='black')
    y = 80
    for key, rows in selected:
        draw.text((16, y), f"SOURCE {key[1]}: {len(rows)} candidates (not independent samples)", font=F20, fill='black')
        y += 56
        items = [('RAW inference view', fit(load(Path(rows[0]['_source']), 'RGB')), '')]
        for r in rows:
            short = f"{r['historical_stage']} {r['mask_type']} s{r['strength']} {r['prompt_id'].rsplit('_',1)[-1]}"
            items.append((short, load(Path(r['_output']), 'RGB'), 'seed='+r['seed']))
        for n, (label, image, seed) in enumerate(items):
            x, top = 16+(n%6)*272, y+(n//6)*274
            draw.text((x, top), label, font=F16, fill='black')
            draw.text((x, top+22), seed, font=F16, fill='black')
            paste_cell(canvas, image, x, top+46, size=218)
        y += ((len(items)+5)//6)*274+8
    canvas.save(destination, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'a0_\d{8}_\d{6}', args.run_id):
        raise SystemExit('Use a unique a0_YYYYMMDD_HHMMSS run id')
    if str(ROOT) != '/root/autodl-tmp/borescope-new-seg-repro':
        raise SystemExit('This audit must execute in the designated AutoDL repository')
    if git('branch', '--show-current') != 'exp13/vlm-semantic-audit':
        raise SystemExit('Unexpected branch')
    if git('rev-parse', 'HEAD') != '1d85844362862993e358dfdf943be4951f95eb65':
        raise SystemExit('Unexpected evidence-base HEAD')
    destination = ROOT / 'results/exp13_restoration_feasibility' / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    started = utc()
    errors, inventory, rows = [], [], []
    protected_before = {relative(p): digest(p) for run in (R1, R2) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(destination/'historical_sha256_before.json', protected_before)
    train_rows = read_csv(TRAIN_META)  # metadata only, no dataset image opened
    train_lookup = {(r['class'], r['source_id']): r for r in train_rows if r['split'] == 'train'}
    dependencies = [TRAIN_META, ROOT/'tools/exp13_diffusion_pilot.py', ROOT/'tools/exp13_diffusion_pilot_r1.py', ROOT/'tools/exp13_diffusion_diagnosis.py']
    write_json(destination/'provenance.json', {relative(p): digest(p) for p in dependencies})
    for stage, run, manifest, idfield, expected in [
            ('R1', R1, R1/'manifests/pilot_manifest.csv', 'pilot_id', 40),
            ('R2', R2, R2/'manifests/diagnosis_manifest.csv', 'diag_id', 48)]:
        raw_rows = read_csv(manifest)
        if len(raw_rows) != expected:
            errors.append(f'{stage}: expected {expected} rows, found {len(raw_rows)}')
        for raw in raw_rows:
            try:
                if raw['status'] != 'SUCCESS':
                    raise ValueError('Historical generation status is not SUCCESS')
                key = (raw['class'], raw['source_id'])
                if key not in train_lookup:
                    raise ValueError('No TRAIN-only provenance for source')
                row = dict(historical_stage=stage, run_id=run.name, **{'class':raw['class']},
                           source_id=raw['source_id'], source_group_id=':'.join(key),
                           variant_id=raw[idfield], condition=raw[idfield],
                           source_image=raw['source_image_path'], original_defect_mask=raw['original_mask_path'],
                           edit_mask=raw['inpaint_mask_path'], output_image=raw['output_path'],
                           model=raw['model_name'], model_revision=raw['model_revision'],
                           strength=raw.get('strength', raw.get('strength_if_applicable', '')),
                           prompt_id=raw['prompt_id'], prompt_text=raw['prompt_text'], seed=raw['seed'],
                           mask_type=raw.get('mask_strategy', 'R1_inpaint'),
                           mask_dilation_px='5' if raw.get('mask_strategy')=='mask_B' else '3',
                           native_feather_sigma_px=1.0, split='train', train_provenance=relative(TRAIN_META))
                for field, internal in [('source_image_path', '_source'), ('original_mask_path', '_original'),
                                        ('inpaint_mask_path', '_edit'), ('output_path', '_output')]:
                    path = checked_path(raw[field], run)
                    row[internal] = str(path)
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        im.load()
                        if im.getexif().get(274, 1) != 1:
                            raise ValueError('Nontrivial EXIF orientation requires manual correspondence check')
                        inventory.append(dict(variant_id=row['variant_id'], file_role=field,
                                              path=relative(path), bytes=path.stat().st_size, sha256=digest(path),
                                              width=im.width, height=im.height, mode=im.mode, format=im.format))
                src, orig, edit, out = (load(Path(row[k]), mode) for k, mode in
                    [('_source','RGB'),('_original','L'),('_edit','L'),('_output','RGB')])
                if not src.size == orig.size == edit.size or out.size != (512, 512):
                    raise ValueError('Unexpected input/mask/output raster size')
                if orig.getbbox() is None or not set(np.unique(orig)).issubset({0, 255}):
                    raise ValueError('Original mask is empty or nonbinary')
                reconstructed = orig.filter(ImageFilter.MaxFilter(2*int(row['mask_dilation_px'])+1)).filter(ImageFilter.GaussianBlur(1.0))
                if not np.array_equal(reconstructed, edit):
                    raise ValueError('Historical edit mask does not match recorded dilation/feather recipe')
                raw_fit, orig_fit, edit_fit = fit(src), fit(orig, True), fit(edit, True)
                if orig_fit.getbbox() is None:
                    raise ValueError('Defect completely lost in historical crop')
                mask = np.asarray(edit_fit)
                defect = np.asarray(orig_fit)>0
                core, support = mask>=128, mask>0
                safe_support = np.asarray(edit_fit.filter(ImageFilter.MaxFilter(17)))>0
                box = Image.fromarray(((support | defect)*255).astype('uint8')).getbbox()
                box = (max(0,box[0]-16), max(0,box[1]-16), min(512,box[2]+16), min(512,box[3]+16))
                row['_roi_box'] = list(box)
                diff = np.abs(np.asarray(raw_fit, dtype='float32')-np.asarray(out, dtype='float32')).mean(2)
                outside_roi = np.ones((512,512), dtype=bool)
                outside_roi[box[1]:box[3],box[0]:box[2]] = False
                outer_border = np.ones((512,512), dtype=bool)
                outer_border[20:-20,20:-20] = False
                crop = crop_box(src.size)
                retained = np.count_nonzero(np.asarray(orig.crop(crop))) / np.count_nonzero(np.asarray(orig))
                row['_spatial'] = dict(source_size=f'{src.width}x{src.height}', output_size='512x512',
                     masks_match_native_source=True, native_crop_xyxy=json.dumps(crop), scale=512/min(src.size),
                     transform='ImageOps.fit; center=(0.5,0.5); RGB=LANCZOS; masks=NEAREST',
                     original_mask_retained_fraction_native_crop=retained,
                     original_defect_px_fit=int(defect.sum()), edit_support_px_fit=int(support.sum()),
                     edit_core_px_fit=int(core.sum()), feather_ring_px_fit=int(((mask>0)&(mask<128)).sum()),
                     recipe_matches_saved_edit_mask=True, roi_bbox_xyxy=json.dumps(box))
                values = dict(full_mae_rgb_0_255=mean(diff), defect_mask_mae=mean(diff[defect]),
                              edit_core_mae=mean(diff[core]), edit_support_mae=mean(diff[support]),
                              outside_edit_support_mae=mean(diff[~support]),
                              outside_support_plus_8px_mae=mean(diff[~safe_support]),
                              outside_roi_bbox_mae=mean(diff[outside_roi]), outer_20px_frame_mae=mean(diff[outer_border]),
                              outside_support_plus_8px_changed_gt5_fraction=mean(diff[~safe_support]>5),
                              outside_support_plus_8px_pixel_count=int((~safe_support).sum()),
                              roi_bbox_padding_px=16, diagnostic_halo_output_px=8)
                row['_metrics'] = values
                row['_source_sha'] = digest(Path(row['_source']))
                row['_original_sha'] = digest(Path(row['_original']))
                row['_output_sha'] = digest(Path(row['_output']))
                rows.append(row)
            except Exception as e:
                errors.append(stage+'/'+raw.get(idfield,'UNKNOWN')+': '+str(e))
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r:(r['class'],int(r['source_id']),r['historical_stage'],r['mask_type'],float(r['strength']),r['prompt_id'],r['seed'])):
        groups[(row['class'],row['source_id'])].append(row)
    for key, variants in groups.items():
        if len({r['_source_sha'] for r in variants}) != 1 or len({r['_original_sha'] for r in variants}) != 1:
            errors.append(str(key)+': cross-run source/original-mask hashes disagree')
        r1_edits = {digest(Path(r['_edit'])) for r in variants if r['historical_stage']=='R1'}
        if any(digest(Path(r['_edit'])) not in r1_edits for r in variants if r['mask_type']=='mask_A'):
            errors.append(str(key)+': R2 mask_A differs from R1 edit mask')
    if len(groups)!=20 or len(rows)!=88 or len({r['_source_sha'] for r in rows})!=20:
        errors.append('Unexpected combined source/hash/variant counts')
    write_csv(destination/'file_integrity.csv', inventory)
    write_json(destination/'validation_report.json', dict(errors=errors, candidate_rows=len(rows), source_groups=len(groups)))
    if errors:
        write_json(destination/'summary.json', dict(gate='HARD_STOP_MATERIAL_OR_SPATIAL_CORRESPONDENCE_ERROR', errors=errors))
        raise SystemExit('HARD_STOP: see validation_report.json')
    index = [{k:v for k,v in r.items() if not k.startswith('_')} for variants in groups.values() for r in variants]
    ordered = [r for variants in groups.values() for r in variants]
    write_csv(destination/'source_variant_review_index.csv', index)
    write_csv(destination/'spatial_correspondence_report.csv', [dict(run_id=r['run_id'],class_name=r['class'],source_id=r['source_id'],variant_id=r['variant_id'],**r['_spatial']) for r in ordered])
    write_csv(destination/'quantitative_differences.csv', [dict(run_id=r['run_id'],class_name=r['class'],source_id=r['source_id'],variant_id=r['variant_id'],**r['_metrics']) for r in ordered])
    write_csv(destination/'human_quality_review_template.csv', [{**{k:r[k] for k in ['source_id','class','run_id','variant_id']},**{k:'' for k in HUMAN}} for r in ordered])
    write_csv(destination/'source_group_summary.csv', [dict(source_group_id=':'.join(key),class_name=key[0],source_id=key[1],candidate_count=len(v),r1_count=sum(r['historical_stage']=='R1' for r in v),r2_count=sum(r['historical_stage']=='R2' for r in v),unique_output_sha256_count=len({r['_output_sha'] for r in v}),source_sha256=v[0]['_source_sha'],independence='one source group; variants are not independent') for key,v in groups.items()])
    # Descriptive ranges across candidates only. No p-values, rankings or quality pass rate.
    aggregates = []
    for stage in ['R1','R2']:
        for cls in ['Crack','corrosion']:
            subset = [r for r in ordered if r['historical_stage']==stage and r['class']==cls]
            for metric in ['defect_mask_mae','edit_support_mae','outside_edit_support_mae','outside_support_plus_8px_mae','outside_roi_bbox_mae','outer_20px_frame_mae']:
                values = [r['_metrics'][metric] for r in subset if r['_metrics'][metric]!='']
                aggregates.append(dict(stage=stage,class_name=cls,metric=metric,candidate_count=len(subset),source_count=len({r['source_id'] for r in subset}),minimum=min(values),median=float(np.median(values)),maximum=max(values),unit='mean absolute RGB difference; 0..255; descriptive only'))
    write_csv(destination/'descriptive_metric_ranges.csv', aggregates)
    sheets = destination/'contact_sheets'
    sheets.mkdir()
    for key, variants in groups.items():
        review_sheet(variants, sheets/(key[0].lower()+'_'+key[1]+'_review.png'))
    for cls in ['Crack','corrosion']:
        overview(groups, cls, sheets/(cls.lower()+'_overview.png'))
    # Browser-friendly, source-grouped links; no standalone input/output dataset copies.
    links = ['# Review navigation', '', 'Open full-resolution PNGs, then enter judgments in the blank CSV.', '',
             '- [Crack overview](contact_sheets/crack_overview.png)', '- [corrosion overview](contact_sheets/corrosion_overview.png)', '']
    for key in groups:
        links.append(f"- [{key[0]} {key[1]}](contact_sheets/{key[0].lower()}_{key[1]}_review.png)")
    with (destination/'review_navigation.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(links)+'\n')
    after = {relative(p): digest(p) for run in (R1,R2) for p in sorted(run.rglob('*')) if p.is_file()}
    unchanged = after == protected_before
    write_json(destination/'historical_preservation_report.json', dict(all_historical_files_unchanged=unchanged,protected_file_count=len(after),before_after_sha256_equal=unchanged))
    if not unchanged:
        raise SystemExit('HARD_STOP_HISTORICAL_INPUT_CHANGED')
    summary = dict(run_id=args.run_id,gate=GATE,started_utc=started,completed_utc=utc(),hostname=platform.node(),
                   branch=git('branch','--show-current'),evidence_base_head=git('rev-parse','HEAD'),
                   python=sys.version,numpy=np.__version__,pillow=PIL.__version__,
                   r1_outputs=40,r1_sources=20,r2_outputs=48,r2_sources=4,combined_candidates=88,
                   combined_unique_class_source_groups=len(groups),combined_unique_source_file_hashes=20,
                   cross_run_repeated_sources=4,unique_output_file_hashes=len({r['_output_sha'] for r in rows}),
                   source_class_counts=dict(Counter(k[0] for k in groups)),
                   candidate_class_counts=dict(Counter(r['class'] for r in rows)),contact_sheets=22,
                   human_judgment_cells_blank=88*len(HUMAN),validation_errors=errors,historical_files_unchanged=True,
                   generation=False,training=False,val_test_image_access=False,restoration_quality_gate_pass=False,
                   exp13_2_b_authorized=False)
    write_json(destination/'summary.json', summary)
    with (destination/'execution_log.txt').open('x', encoding='utf-8') as f:
        f.write(f'{started} START CPU-only historical audit preparation\n')
        f.write('Parsed R1=40 and R2=48 manifest rows; all SUCCESS and TRAIN-provenance matched.\n')
        f.write('Validated referenced file decodes, raster sizes, saved mask recipe and source grouping.\n')
        f.write('Replayed historical central crop/fit, computed descriptive differences and rendered 22 review PNGs.\n')
        f.write(f'Historical before/after SHA256 equal for {len(after)} files.\n')
        f.write(f'{summary["completed_utc"]} END {GATE}\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
