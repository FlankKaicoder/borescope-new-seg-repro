"""Run the bounded, offline Exp13 Phase 1-B-S2 SDXL engineering smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLInpaintPipeline

from tools.exp13_restoration.geometry import build_geometry, map_mask_to_canvas, pillow_polygon_mask
from tools.exp13_restoration.protocol import BudgetProtocol
from tools.exp13_restoration.roi_composition import RGBROIEdit, compose_full_image
from tools.exp13_restoration.run_phase1a_r2 import MANIFEST, MANIFEST_SHA, RAW, sha256

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
MODEL = Path('/root/autodl-tmp/models_staging/exp13_phase1b_r3_20261025T042354Z/sdxl-inpainting-0.1')
MODEL_MANIFEST = Path('/root/autodl-tmp/sdxl_fp16_manifest.csv')
MODEL_MANIFEST_SHA = 'a1a0302f7f8f9a0e2bf21e9b47a7757fd4e1ba9f246bf0c59bfcef1d5eb9607e'
REVISION = '115134f363124c53c7d878647567d04daf26e41e'
PROMPT = 'restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting'
CONFIG = {'canvas': 1024, 'precision': 'float16', 'steps': 20, 'guidance_scale': 7.5, 'strength': 0.8, 'seed': 130500, 'prompt': PROMPT, 'negative_prompt': None, 'cpu_offload': 'enable_model_cpu_offload(gpu_id=0)', 'vae_tiling': True, 'vae_slicing': True, 'image_resize': 'Pillow LANCZOS', 'mask_resize': 'nearest', 'mask_polarity': 'white=redraw, black=preserve', 'padding': 'reflect image; black mask'}


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> list[dict]:
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('frozen TRAIN manifest SHA mismatch')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        return [r for r in csv.DictReader(f) if r.get('split') == 'train']


def load_source(row: dict) -> tuple[np.ndarray, dict, list[dict]]:
    path = RAW / row['image_path']
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    ann_path = RAW / row['json_path']
    ann = json.loads(ann_path.read_text(encoding='utf-8'))
    shapes = [s for s in ann.get('shapes', []) if s.get('shape_type', 'polygon') == 'polygon']
    return rgb, ann, shapes


def geometry_record(points: tuple[tuple[float, float], ...], width: int, height: int, source_id: str, roi_id: str, label: str) -> tuple:
    geom = build_geometry(points, width, height, context=16, halo_radius=2, roi_id=roi_id, source_id=source_id)
    box = geom.roi_box
    return geom, {'source_id': source_id, 'roi_id': roi_id, 'class_name': label, 'points_xy': points,
                  'polygon_mask_sha256': hash_bytes(geom.polygon_mask.astype(np.uint8).tobytes()),
                  'edit_support_sha256': hash_bytes(geom.edit_support.astype(np.uint8).tobytes()),
                  'polygon_pixels': int(geom.polygon_mask.sum()), 'edit_support_pixels': int(geom.edit_support.sum()),
                  'roi_box_xyxy': [box.x0, box.y0, box.x1, box.y1], 'transform': geom.transform.__dict__,
                  'roi_clamped': geom.provenance['roi_clamped']}


def canvas_inputs(crop: np.ndarray, support: np.ndarray, geom) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    t = geom.transform
    resized = np.asarray(Image.fromarray(crop).resize((t.resized_width, t.resized_height), Image.Resampling.LANCZOS))
    image_canvas = np.pad(resized, ((t.pad_top, t.pad_bottom), (t.pad_left, t.pad_right), (0, 0)), mode='reflect')
    mask_canvas = map_mask_to_canvas(support, t)
    if mask_canvas[:t.pad_top].any() or mask_canvas[t.pad_top+t.resized_height:].any() or mask_canvas[:, :t.pad_left].any() or mask_canvas[:, t.pad_left+t.resized_width:].any():
        raise RuntimeError('canvas padding unexpectedly included edit mask')
    return Image.fromarray(image_canvas, 'RGB'), Image.fromarray(mask_canvas.astype(np.uint8) * 255, 'L'), image_canvas, mask_canvas


def verify_model_files() -> dict:
    if sha256(MODEL_MANIFEST) != MODEL_MANIFEST_SHA:
        raise RuntimeError('model manifest hash mismatch')
    checked = []
    with MODEL_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            path = MODEL / row['Path']
            if not path.is_file() or path.stat().st_size != int(row['Bytes']) or sha256(path) != row['SHA256']:
                raise RuntimeError(f'model file integrity mismatch: {row["Path"]}')
            checked.append({'path': row['Path'], 'bytes': int(row['Bytes']), 'sha256': row['SHA256']})
    if len(checked) != 19 or sum(x['bytes'] for x in checked) != 6941223330:
        raise RuntimeError('model file set/size differs from R3 verified snapshot')
    return {'revision_reference': REVISION, 'manifest_sha256': MODEL_MANIFEST_SHA, 'files_verified': len(checked), 'verified_bytes': sum(x['bytes'] for x in checked), 'files': checked}


def save_input_review(out: Path, source_id: str, native: np.ndarray, rois: list[dict], geoms: list, polygons: list) -> None:
    source_dir = out / f'source_{source_id}'
    source_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(native).save(source_dir / 'original.png')
    overlay = Image.fromarray(native.copy())
    draw = ImageDraw.Draw(overlay)
    for i, pts in enumerate(polygons):
        draw.line(list(pts) + [pts[0]], fill=('red', 'lime')[i % 2], width=2)
    overlay.save(source_dir / 'polygon_overlay_review_only.png')
    for roi, geom in zip(rois, geoms):
        box = geom.roi_box
        crop = native[box.y0:box.y1, box.x0:box.x1]
        local_support = geom.edit_support[box.y0:box.y1, box.x0:box.x1]
        ci, cm, _, _ = canvas_inputs(crop, local_support, geom)
        ci.save(source_dir / f'{roi["roi_id"]}_input_canvas.png')
        cm.save(source_dir / f'{roi["roi_id"]}_mask_canvas.png')
        points = np.asarray(roi['points_xy'], dtype=float)
        pad = 12
        x0 = max(0, int(np.floor(points[:, 0].min()))-pad); y0 = max(0, int(np.floor(points[:, 1].min()))-pad)
        x1 = min(native.shape[1], int(np.ceil(points[:, 0].max()))+pad+1); y1 = min(native.shape[0], int(np.ceil(points[:, 1].max()))+pad+1)
        Image.fromarray(native[y0:y1, x0:x1]).resize((max(1, (x1-x0)*3), max(1, (y1-y0)*3)), Image.Resampling.NEAREST).save(source_dir / f'{roi["roi_id"]}_defect_zoom.png')


def make_contact_sheet(source_dir: Path, source_id: str, roi_ids: list[str]) -> None:
    paths = [source_dir / 'original.png', source_dir / 'polygon_overlay_review_only.png']
    titles = ['Original RGB', 'Polygon overlay (review only)']
    for roi_id in roi_ids:
        paths.extend([source_dir / f'{roi_id}_defect_zoom.png', source_dir / f'{roi_id}_mask_canvas.png', source_dir / f'{roi_id}_input_canvas.png',
                      source_dir.parent / f'{source_id}_{roi_id}_sdxl_canvas.png', source_dir.parent / f'{source_id}_{roi_id}_native_roi.png'])
        titles.extend([f'{roi_id} defect zoom', f'{roi_id} mask (white=edit)', f'{roi_id} input canvas', f'{roi_id} SDXL canvas', f'{roi_id} native ROI'])
    paths.extend([source_dir.parent / f'source_{source_id}_restored.png', source_dir.parent / f'source_{source_id}_difference.png'])
    titles.extend(['Restored full image', 'Difference heatmap'])
    panels = []
    for path, title in zip(paths, titles):
        with Image.open(path) as image:
            thumb = image.convert('RGB')
            thumb.thumbnail((420, 300), Image.Resampling.LANCZOS)
            panel = Image.new('RGB', (440, 340), 'white')
            panel.paste(thumb, ((440-thumb.width)//2, 32+(300-thumb.height)//2))
            ImageDraw.Draw(panel).text((8, 8), title, fill='black')
            panels.append(panel)
    cols = 3
    rows = (len(panels)+cols-1)//cols
    sheet = Image.new('RGB', (cols*440, rows*340), (235, 235, 235))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i%cols)*440, (i//cols)*340))
    sheet.save(source_dir.parent / f'source_{source_id}_contact_sheet.png')


def memory_record() -> dict:
    return {'gpu_name': torch.cuda.get_device_name(0), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'gpu_free_bytes_before_load': torch.cuda.mem_get_info(0)[0], 'cpu_ram_bytes_available': _available_ram(),
            'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'diffusers': __import__('diffusers').__version__, 'transformers': __import__('transformers').__version__,
            'accelerate': __import__('accelerate').__version__}


def _available_ram() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def run_call(pipe, budget: BudgetProtocol, call_id: str, source_id: str, roi: dict, geom, native: np.ndarray, out: Path) -> tuple[np.ndarray, dict]:
    box = geom.roi_box
    crop = native[box.y0:box.y1, box.x0:box.x1].copy()
    support = geom.edit_support[box.y0:box.y1, box.x0:box.x1]
    image_canvas, mask_canvas, _, mask_array = canvas_inputs(crop, support, geom)
    polygon_local = geom.polygon_mask[box.y0:box.y1, box.x0:box.x1]
    if not mask_array.any() or np.any(polygon_local & ~support):
        raise RuntimeError('invalid native mask or ROI geometry')
    record = {'call_id': call_id, 'source_id': source_id, 'roi_id': roi['roi_id'], 'input_size': list(image_canvas.size),
              'mask_size': list(mask_canvas.size), 'mask_white_pixels': int(mask_array.sum()), 'mask_black_pixels': int(mask_array.size-mask_array.sum()),
              'mask_polarity': 'white=redraw; black=preserve', 'native_roi_xyxy': [box.x0, box.y0, box.x1, box.y1]}
    budget.start_call(call_id, real_model_call=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        generator = torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
        result = pipe(prompt=PROMPT, image=image_canvas, mask_image=mask_canvas, height=1024, width=1024,
                      num_inference_steps=20, guidance_scale=7.5, strength=0.8, generator=generator).images[0]
    except BaseException as exc:
        budget.finish_call(call_id, success=False, error=f'{type(exc).__name__}: {exc}')
        record.update({'status': 'FAILED', 'elapsed_seconds': time.perf_counter()-started, 'exception': traceback.format_exc()})
        raise
    elapsed = time.perf_counter() - started
    result_array = np.asarray(result.convert('RGB'), dtype=np.uint8)
    if result_array.shape != (1024, 1024, 3):
        budget.finish_call(call_id, success=False, error='unexpected model output shape')
        raise RuntimeError(f'unexpected output shape {result_array.shape}')
    t = geom.transform
    content = result_array[t.pad_top:t.pad_top+t.resized_height, t.pad_left:t.pad_left+t.resized_width]
    native_result = np.asarray(Image.fromarray(content).resize((box.width, box.height), Image.Resampling.LANCZOS), dtype=np.uint8)
    changed = np.any(native_result != crop, axis=2)
    outside_support_local = int(np.count_nonzero(changed & ~support))
    full_roi = crop.copy()
    full_roi[support] = native_result[support]
    bad_values = not np.isfinite(result_array).all()
    near_black = float(result_array.mean()) < 2.0 or float(result_array.std()) < 0.25
    budget.finish_call(call_id, success=True)
    record.update({'status': 'SUCCEEDED', 'elapsed_seconds': elapsed, 'output_canvas_size': list(result.size),
                   'output_native_roi_shape_hwc': list(full_roi.shape), 'output_native_roi_dtype': str(full_roi.dtype),
                   'raw_model_roi_changed_pixels_before_support_clip': int(changed.sum()), 'raw_model_roi_changes_outside_native_support_clipped': outside_support_local,
                   'native_support_pixels': int(support.sum()), 'native_support_changed_pixels': int(np.count_nonzero(changed & support)),
                   'nan_inf': bool(bad_values), 'black_or_constant_output': bool(near_black),
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    Image.fromarray(result_array).save(out / f'{source_id}_{roi["roi_id"]}_sdxl_canvas.png')
    Image.fromarray(full_roi).save(out / f'{source_id}_{roi["roi_id"]}_native_roi.png')
    return full_roi, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-run-id', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = REPO / 'results/exp13_phase1b_s2' / f'{run_id}_smoke'
    out.mkdir(parents=True, exist_ok=False)
    audit_path = REPO / 'results/exp13_phase1b_s2' / args.audit_run_id / 'safe_source_candidate_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    selected = next((x for x in audit['dual_roi_candidates_in_screening_order'] if x['source_id'] == '65'), None)
    if not audit['source2']['geometry_eligible'] or not selected or selected['native_roundtrip_mock'] != 'PASS':
        raise RuntimeError('frozen source preflight failed')
    if audit['val_access'] != 0 or audit['test_access'] != 0 or audit['raw_dataset_write'] != 0:
        raise RuntimeError('data protection counters are not zero')
    git = {'branch': os.popen(f'git -C {REPO} branch --show-current').read().strip(), 'head': os.popen(f'git -C {REPO} rev-parse HEAD').read().strip()}
    if git['branch'] != 'exp13/vlm-semantic-audit':
        raise RuntimeError(f'wrong Git branch: {git}')
    previous_ledgers = []
    for path in REPO.glob('results/exp13_phase1b*/**/call_ledger.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        calls = data.get('calls', {})
        if any(rec.get('status') in {'STARTED', 'SUCCEEDED', 'FAILED', 'UNKNOWN', 'RESERVED'} for rec in calls.values()):
            raise RuntimeError(f'previous nonzero/unknown ledger occupancy: {path}')
        previous_ledgers.append({'path': str(path), 'calls': len(calls), 'actual_calls': data.get('actual_calls', 0), 'occupied_calls': data.get('occupied_calls', 0)})
    manifest_check = verify_model_files()
    env = memory_record()
    if env['gpu_free_bytes_before_load'] < 10 * 1024**3 or env['cpu_ram_bytes_available'] < 16 * 1024**3:
        raise RuntimeError(f'insufficient preflight memory: {env}')
    dump(out / 'environment_preflight.json', {'git': git, 'previous_ledgers': previous_ledgers, 'model_files': manifest_check, 'environment': env, 'raw_data_write': 0, 'val_access': 0, 'test_access': 0})
    frozen = {'model_id': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', 'revision': REVISION, 'local_path': str(MODEL), 'model_manifest_sha256': MODEL_MANIFEST_SHA, 'config': CONFIG,
              'source_ids': ['2', '65'], 'source2_visual_review': 'PASS: edit polygon lies on metal and does not intersect displayed UI; source UI remains outside edit support',
              'source65_visual_review': 'PASS: first geometry-eligible dual candidate; both polygons lie on visible metal, no edge/FOV/gray-border contact observed', 'audit_run_id': args.audit_run_id}
    dump(out / 'frozen_smoke_config.json', frozen)
    source_rows = {str(r['stem']): r for r in load_manifest()}
    frozen_sources = []
    loaded: dict[str, tuple[np.ndarray, list[dict], list, list]] = {}
    for sid in ('2', '65'):
        row = source_rows[sid]
        native, ann, shapes = load_source(row)
        audit_rec = audit['source2'] if sid == '2' else selected
        if sha256(RAW / row['image_path']) != audit_rec['image_sha256'] or sha256(RAW / row['json_path']) != audit_rec['annotation_sha256']:
            raise RuntimeError(f'source hash changed since candidate audit: {sid}')
        geoms, rois, polygons = [], [], []
        for i, shape in enumerate(shapes):
            points = tuple((float(x), float(y)) for x, y in shape['points'])
            geom, rec = geometry_record(points, native.shape[1], native.shape[0], sid, f'roi_{i}', str(shape.get('label', 'unknown')))
            if i >= len(audit_rec['rois']) or rec['polygon_mask_sha256'] != audit_rec['rois'][i]['polygon_mask_sha256'] or rec['roi_box_xyxy'] != audit_rec['rois'][i]['roi_box_xyxy']:
                raise RuntimeError(f'geometry hash/box mismatch since audit: {sid}/roi_{i}')
            geoms.append(geom); rois.append(rec); polygons.append(points)
        frozen_sources.append({'source_id': sid, 'split': row['split'], 'image_path': row['image_path'], 'annotation_path': row['json_path'],
                               'image_sha256': audit_rec['image_sha256'], 'annotation_sha256': audit_rec['annotation_sha256'], 'native_shape_hwc': list(native.shape),
                               'native_dtype': str(native.dtype), 'rois': rois})
        loaded[sid] = (native, shapes, geoms, polygons)
        save_input_review(out, sid, native, rois, geoms, polygons)
    dump(out / 'smoke_source_manifest.json', {'train_manifest_sha256': MANIFEST_SHA, 'sources': frozen_sources, 'val_access': 0, 'test_access': 0, 'raw_data_write': 0})
    if args.preflight_only:
        dump(out / 'preflight_only_summary.json', {'status': 'PREFLIGHT_PASS_MODEL_NOT_LOADED', 'git': git, 'model_files': manifest_check,
                                                   'environment': env, 'source_ids': ['2', '65'], 'verified_rois': sum(len(x['rois']) for x in frozen_sources),
                                                   'val_access': 0, 'test_access': 0, 'raw_data_write': 0, 'model_load': 0, 'inference_calls': 0})
        print(json.dumps({'run_dir': str(out), 'status': 'PREFLIGHT_PASS_MODEL_NOT_LOADED', 'model_load': 0, 'inference_calls': 0}, ensure_ascii=False))
        return
    ledger_path = out / 'call_ledger.json'
    ledger = BudgetProtocol(ledger_path=ledger_path)
    if ledger_path.exists():
        raise RuntimeError('new S2 ledger unexpectedly exists')
    initial = ledger.snapshot()
    initial['history'].append({'event': 'MIGRATED_FROM_S1_ZERO_OCCUPANCY', 'source_ledger': str(REPO / 'results/exp13_phase1b_s1/20261025T044024Z/call_ledger.json'), 'verified_actual_calls': 0, 'verified_occupied_calls': 0, 'timestamp': datetime.now(timezone.utc).isoformat()})
    tmp = ledger_path.with_name(ledger_path.name + '.tmp')
    tmp.write_text(json.dumps(initial, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, ledger_path)
    ledger = BudgetProtocol(ledger_path=ledger_path)
    records = []
    metrics = []
    status = 'EXP13_PHASE1B_S2_RUNTIME_HOLD'
    load_info = {'attempted': False, 'success': False, 'revision': REVISION}
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        load_info['attempted'] = True
        load_start = time.perf_counter()
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(MODEL), torch_dtype=torch.float16, variant='fp16', use_safetensors=True, local_files_only=True)
        pipe.enable_model_cpu_offload(gpu_id=0)
        pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
        load_info.update({'success': True, 'load_seconds': time.perf_counter()-load_start, 'pipeline_class': type(pipe).__name__, 'offline': True, 'cpu_offload': True, 'vae_tiling': True, 'vae_slicing': True})
        for sid in ('2', '65'):
            native, shapes, geoms, polygons = loaded[sid]
            roi_records = frozen_sources[0 if sid == '2' else 1]['rois']
            if sid == '2':
                reserved = ledger.reserve_source_group(source_id=sid, roi_ids=['roi_0'], stage='smoke', strength=0.8, model_config_id='exp13_phase1b_s2_sdxl_fp16_1024_s130500')
            else:
                reserved = ledger.reserve_source_group(source_id=sid, roi_ids=['roi_0', 'roi_1'], stage='smoke', strength=0.8, model_config_id='exp13_phase1b_s2_sdxl_fp16_1024_s130500')
            if len(reserved) != (1 if sid == '2' else 2) or any(x['status'] != 'RESERVED' for x in reserved):
                raise RuntimeError(f'budget group pre-reservation invalid for source {sid}')
            edits = []
            try:
                for roi_record, shape, geom in zip(roi_records, shapes, geoms):
                    call_id = f'{sid}:{roi_record["roi_id"]}:s0.8'
                    roi_rgb, call_record = run_call(pipe, ledger, call_id, sid, roi_record, geom, native, out)
                    records.append(call_record); metrics.append({'source_id': sid, 'roi_id': roi_record['roi_id'], 'elapsed_seconds': call_record['elapsed_seconds'], 'gpu_peak_allocated_bytes': call_record['gpu_peak_allocated_bytes'], 'gpu_peak_reserved_bytes': call_record['gpu_peak_reserved_bytes']})
                    if call_record.get('nan_inf') or call_record.get('black_or_constant_output'):
                        raise RuntimeError(f'ROI engineering validation failed: {sid}/{roi_record["roi_id"]}')
                    edits.append(RGBROIEdit(roi_record['roi_id'], sid, tuple((float(x), float(y)) for x, y in shape['points']), roi_rgb, geom.edit_support))
            except BaseException:
                for remaining in reserved:
                    if ledger.snapshot()['calls'][remaining['call_id']]['status'] == 'RESERVED':
                        ledger.cancel_call(remaining['call_id'], reason='S2 stop condition prevented this reserved ROI from starting')
                raise
            composed = compose_full_image(native, sid, edits, context=16, halo_radius=2, unresolved_overlap_policy='hold_on_overlap')
            changed = np.any(composed.image != native, axis=2)
            outside = int(np.count_nonzero(changed & ~composed.edit_support))
            if composed.hold or outside != 0 or composed.overlap_pixels != 0:
                raise RuntimeError(f'full-image composition failed: {sid}, outside={outside}, overlap={composed.overlap_pixels}, hold={composed.hold}')
            order_check = compose_full_image(native, sid, list(reversed(edits)), context=16, halo_radius=2, unresolved_overlap_policy='hold_on_overlap')
            if not np.array_equal(composed.image, order_check.image):
                raise RuntimeError(f'ROI composition is order-dependent: {sid}')
            Image.fromarray(composed.image).save(out / f'source_{sid}_restored.png')
            diff = np.max(np.abs(composed.image.astype(np.int16)-native.astype(np.int16)), axis=2).astype(np.uint8)
            Image.fromarray(diff).save(out / f'source_{sid}_difference.png')
            summary = {'source_id': sid, 'output_shape_hwc': list(composed.image.shape), 'output_dtype': str(composed.image.dtype), 'changed_pixels': int(changed.sum()), 'support_pixels': int(composed.edit_support.sum()), 'outside_support_changed_pixels': outside, 'overlap_pixels': composed.overlap_pixels, 'order_independent': True, 'background_immutable': bool(np.array_equal(composed.image[~composed.edit_support], native[~composed.edit_support]))}
            dump(out / f'source_{sid}_geometry_validation.json', summary)
            make_contact_sheet(out / f'source_{sid}', sid, [r['roi_id'] for r in roi_records])
            if sid == '2' and (not summary['background_immutable'] or summary['changed_pixels'] == 0):
                raise RuntimeError('source 2 failed before dual-ROI execution')
        status = 'EXP13_PHASE1B_S2_COMPLETED_PENDING_CHATGPT_GATE'
    except BaseException as exc:
        load_info['exception'] = traceback.format_exc()
        if not load_info['success']:
            status = 'EXP13_PHASE1B_S2_MODEL_LOAD_HOLD'
        else:
            status = 'EXP13_PHASE1B_S2_RUNTIME_HOLD'
        dump(out / 'runtime_exception.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    finally:
        dump(out / 'model_load_report.json', load_info)
        dump(out / 'runtime_metrics.json', {'calls': records, 'summary': metrics, 'gpu_peak_allocated_bytes': max((x['gpu_peak_allocated_bytes'] for x in metrics), default=None), 'gpu_peak_reserved_bytes': max((x['gpu_peak_reserved_bytes'] for x in metrics), default=None)})
        dump(out / 'call_ledger_snapshot.json', ledger.snapshot())
        dump(out / 'summary.json', {'status': status, 'real_roi_calls': ledger.snapshot()['actual_calls'], 'occupied_calls': ledger.snapshot()['occupied_calls'], 'source_ids': ['2', '65'], 'model_load': load_info, 'val_access': 0, 'test_access': 0, 'raw_dataset_write': 0, 'ssl_training': 0, 'yolo_training': 0, 'gpt_fallback': 0})
    print(json.dumps({'run_dir': str(out), 'status': status, 'real_roi_calls': ledger.snapshot()['actual_calls'], 'occupied_calls': ledger.snapshot()['occupied_calls']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
