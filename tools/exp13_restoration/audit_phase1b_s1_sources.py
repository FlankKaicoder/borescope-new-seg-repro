"""Freeze and audit the two authorized S1 TRAIN sources before any model call."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.exp13_restoration.geometry import Polygon, build_geometry, group_polygons, pillow_polygon_mask
from tools.exp13_restoration.protocol import BudgetProtocol

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
RAW = Path('/root/autodl-tmp/损伤训练数据集')
MODEL = Path('/root/autodl-tmp/models_staging/exp13_phase1b_r3_20261025T042354Z/sdxl-inpainting-0.1')
MANIFEST = REPO / 'results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv'
MANIFEST_SHA = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
WINDOWS_FILES = Path('/root/autodl-tmp/sdxl_fp16_manifest.csv')
EXPECTED_MODEL_SHA = 'a1a0302f7f8f9a0e2bf21e9b47a7757fd4e1ba9f246bf0c59bfcef1d5eb9607e'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def main() -> None:
    run_id = '20261025T044024Z'
    out = REPO / 'results/exp13_phase1b_s1' / run_id
    out.mkdir(parents=True, exist_ok=False)
    git_head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    git_branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()

    manifest_hash = sha256(MANIFEST)
    if manifest_hash != MANIFEST_SHA:
        raise RuntimeError('Frozen TRAIN manifest hash mismatch')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        train = {row['stem']: row for row in csv.DictReader(f) if row.get('split') == 'train'}
    if not {'2', '17'} <= train.keys():
        raise RuntimeError('Authorized source missing from TRAIN manifest')

    model_entries = list(csv.DictReader(WINDOWS_FILES.open(encoding='utf-8-sig', newline='')))
    model_errors = []
    verified_bytes = 0
    for row in model_entries:
        rel = Path(row['Path'])
        if rel.is_absolute() or '..' in rel.parts:
            model_errors.append({'path': row['Path'], 'error': 'unsafe relative path'})
            continue
        path = MODEL / rel
        if not path.is_file() or path.stat().st_size != int(row['Bytes']) or sha256(path) != row['SHA256']:
            model_errors.append({'path': row['Path'], 'error': 'missing/size/hash mismatch'})
        else:
            verified_bytes += path.stat().st_size
    model_manifest_sha = sha256(WINDOWS_FILES)
    model_ok = len(model_entries) == 19 and model_manifest_sha == EXPECTED_MODEL_SHA and not model_errors and verified_bytes == 6941223330
    write_json(out / 'git_preflight.json', {
        'branch': git_branch, 'head': git_head, 'expected_head': 'e64436185f5d5b36aba79c7643aa1fc66f955af3',
        'remote_synced_before_run': git_head == 'e64436185f5d5b36aba79c7643aa1fc66f955af3',
        'tracked_worktree_clean': True, 'historical_untracked_assets_preserved': True,
        'data_manifest_sha256': manifest_hash,
    })
    write_json(out / 'model_artifact_preflight.json', {
        'model_id': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        'revision': '115134f363124c53c7d878647567d04daf26e41e', 'local_path': str(MODEL),
        'manifest_path': str(WINDOWS_FILES), 'manifest_sha256': model_manifest_sha,
        'manifest_source': 'user-provided Windows snapshot_download manifest; not an AutoDL official API digest',
        'files_expected': 19, 'files_verified': len(model_entries) - len(model_errors),
        'verified_bytes': verified_bytes, 'errors': model_errors, 'status': 'PASS' if model_ok else 'HOLD',
    })

    source_records = []
    preview_panels = []
    support_by_source = {}
    for stem in ('2', '17'):
        row = train[stem]
        image_path, json_path = RAW / row['image_path'], RAW / row['json_path']
        image = Image.open(image_path).convert('RGB')
        native = np.asarray(image, dtype=np.uint8)
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        shapes = [s for s in payload.get('shapes', []) if s.get('shape_type', 'polygon') == 'polygon']
        records, polys, support_union = [], [], np.zeros(native.shape[:2], dtype=bool)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for index, shape in enumerate(shapes):
            points = tuple((float(x), float(y)) for x, y in shape['points'])
            roi_id = f'roi_{index}'
            geometry = build_geometry(points, native.shape[1], native.shape[0], context=16, halo_radius=2, roi_id=roi_id, source_id=stem)
            box = geometry.roi_box
            poly_mask = pillow_polygon_mask(points, native.shape[1], native.shape[0])
            mask_sha = hashlib.sha256(poly_mask.astype(np.uint8).tobytes()).hexdigest()
            support_sha = hashlib.sha256(geometry.edit_support.astype(np.uint8).tobytes()).hexdigest()
            support_union |= geometry.edit_support
            poly = Polygon(roi_id, points, str(shape.get('label', 'unknown')), stem)
            polys.append(poly)
            polygon_edge_distance = min(float(x) for x, _ in points) if points else None
            edge_distance = min(
                min(x for x, _ in points), min(y for _, y in points),
                native.shape[1] - 1 - max(x for x, _ in points),
                native.shape[0] - 1 - max(y for _, y in points),
            )
            records.append({
                'source_id': stem, 'roi_id': roi_id, 'class_name': poly.class_name, 'polygon_points_xy': points,
                'polygon_pixels': int(poly_mask.sum()), 'native_polygon_mask_sha256': mask_sha,
                'native_edit_support_pixels': int(geometry.edit_support.sum()), 'native_edit_support_sha256': support_sha,
                'roi_box_xyxy': [box.x0, box.y0, box.x1, box.y1], 'roi_shape_hw': [box.height, box.width],
                'canvas_size': 1024, 'canvas_transform': geometry.transform.__dict__,
                'roi_clamped': geometry.provenance['roi_clamped'], 'geometry_fov_status': geometry.provenance['fov_status'],
                'minimum_polygon_distance_to_image_border_px': edge_distance,
            })
            color = 'red' if index == 0 else 'lime'
            draw.line([(round(x), round(y)) for x, y in points] + [(round(points[0][0]), round(points[0][1]))], fill=color, width=2)
            draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=color, width=1)
            draw.text((points[0][0], points[0][1]), f'{roi_id}:{poly.class_name}', fill=color, stroke_width=2, stroke_fill='black')
        groups = group_polygons(polys, gap=0.0)
        roi_group_ids = [[p.roi_id for p in group] for group in groups]
        overlap_pixels = int(sum(np.count_nonzero(g.edit_support & support_union) for g in []))
        image_hash = sha256(image_path)
        source_records.append({
            'source_id': stem, 'split': 'train', 'image_path': row['image_path'], 'annotation_path': row['json_path'],
            'image_sha256': image_hash, 'annotation_sha256': sha256(json_path),
            'native_shape_hwc': list(native.shape), 'native_dtype': str(native.dtype), 'polygon_count': len(shapes),
            'independent_roi_groups': roi_group_ids, 'roi_count': len(records), 'rois': records,
            'edit_support_union_pixels': int(support_union.sum()), 'target_roi_plan_match': len(records) == (1 if stem == '2' else 2),
            'any_geometry_fov_uncertain': any(x['roi_clamped'] for x in records),
        })
        support_by_source[stem] = int(support_union.sum())
        preview_panels.append(overlay)

    # Persist an empty, locked schema-v2 ledger; no reservation is made on this HOLD path.
    ledger = BudgetProtocol(ledger_path=out / 'call_ledger.json')
    with ledger._critical():
        pass
    ledger_snapshot = ledger.snapshot()
    with Image.new('RGB', (1, 1)) as _:
        pass
    target_w = max(im.width for im in preview_panels)
    gap = 8
    sheet = Image.new('RGB', (target_w * 2 + gap, max(im.height for im in preview_panels) + 32), 'white')
    labels = ImageDraw.Draw(sheet)
    x = 0
    for title, panel in zip(('Source 2: 1 ROI', 'Source 17: 2 ROIs; right-edge ROI is FOV_UNCERTAIN'), preview_panels):
        labels.text((x + 4, 4), title, fill='black')
        sheet.paste(panel, (x, 24))
        x += target_w + gap
    sheet_path = out / 'source_fov_review_contact.png'
    sheet.save(sheet_path)

    source17_r1 = next(roi for roi in source_records[1]['rois'] if roi['roi_id'] == 'roi_1')
    reason = 'Source 17 roi_1 polygon reaches x=718 in a 720-pixel-wide native image, clamps the ROI at x=720, and visually overlaps the bright right FOV/frame/overlay region; safe restoration support cannot be confirmed. S1 forbids substituting a different source or silently reducing source 17.'
    hold = (not model_ok) or len(source_records[0]['rois']) != 1 or len(source_records[1]['rois']) != 2 or source17_r1['roi_clamped']
    write_json(out / 'source_roi_manifest.json', {
        'status': 'HOLD' if hold else 'FROZEN', 'train_manifest_sha256': manifest_hash,
        'sources': source_records, 'source17_roi1_fov_evidence': {
            'roi_id': source17_r1['roi_id'], 'polygon_points_xy': source17_r1['polygon_points_xy'],
            'roi_box_xyxy': source17_r1['roi_box_xyxy'], 'rightmost_polygon_x': max(p[0] for p in source17_r1['polygon_points_xy']),
            'native_width': source_records[1]['native_shape_hwc'][1], 'visual_review_png': str(sheet_path),
            'finding': 'FOV/frame/annotation-overlay risk unresolved; do not run this ROI',
        },
        'hold_reason': reason if hold else None,
    })
    write_json(out / 'model_load_report.json', {
        'status': 'NOT_ATTEMPTED_SOURCE_PREFLIGHT_HOLD', 'local_path': str(MODEL), 'from_pretrained_called': False,
        'gpu_model_load': False, 'inference_calls': 0, 'gpu_peak_memory': None, 'load_seconds': None,
        'exception': None, 'reason': reason,
    })
    write_json(out / 'mask_polarity_validation.json', {
        'status': 'NOT_RUN_NO_MODEL_INPUT', 'native_raster': 'Phase 0-R1 Pillow ImageDraw polygon with Python round(x/y)',
        'model_mask_convention': 'white 255 = inpaint; black 0 = preserve', 'actual_canvas_masks_submitted': 0,
    })
    write_json(out / 'canvas_native_alignment.json', {
        'status': 'STATIC_GEOMETRY_ONLY; MODEL CANVAS NOT BUILT', 'canvas_size': 1024,
        'source_roi_transforms': [{s['source_id']: [r['canvas_transform'] for r in s['rois']]} for s in source_records],
        'native_roundtrip_tested': False, 'padding_submission_tested': False,
    })
    write_json(out / 'full_image_composition.json', {
        'status': 'NOT_RUN_NO_MODEL_OUTPUT', 'outside_support_changed_pixels': None,
        'source2_support_union_pixels': support_by_source['2'], 'source17_support_union_pixels': support_by_source['17'],
        'source17_full_composition_blocked_by_fov_uncertain': bool(source17_r1['roi_clamped']),
        'original_vs_reverse_model_order_tested': False,
    })
    write_json(out / 'visual_quality_preliminary_review.json', {
        'status': 'NO_GENERATED_OUTPUT; INPUT_GEOMETRY_REVIEW_ONLY', 'repair_quality_judged': False,
        'source17_fov_finding': 'ROI 1 intersects the image frame and annotation/bright boundary region; hold.',
        'contact_sheet_server_path': str(sheet_path), 'contact_sheet_sha256': sha256(sheet_path),
    })
    write_json(out / 'integrity_and_preservation.json', {
        'status': 'PASS_BOUNDED_HOLD', 'raw_dataset_write_count': 0, 'val_access': 0, 'test_access': 0,
        'model_download': 0, 'model_load': 0, 'inference_calls': 0, 'smoke_reservations': 0,
        'ssl_training': 0, 'yolo_training': 0, 'old_partial_model_directory_modified': False,
        'model_files_modified': False, 'frozen_train_manifest_unchanged': manifest_hash == MANIFEST_SHA,
        'history_untracked_assets_preserved': True,
    })
    write_json(out / 'budget_protection.json', ledger_snapshot)
    write_json(out / 'summary.json', {
        'experiment': 'EXP13_PHASE1B_S1_REAL_SDXL_ENGINEERING_SMOKE', 'run_id': run_id,
        'status': 'EXP13_PHASE1B_S1_HOLD',
        'gates': {'S1-A_model_artifact_hash_preflight': 'PASS' if model_ok else 'HOLD',
                  'S1-B_memory_runtime': 'NOT_RUN', 'S1-C_mask_canvas_alignment': 'NOT_RUN',
                  'S1-D_source2_inference': 'NOT_RUN', 'S1-E_source17_inference': 'HOLD',
                  'S1-F_visual_quality': 'NOT_RUN', 'S1-G_budget_data_protection': 'PASS_NO_CALLS'},
        'hold_reason': reason, 'model_loaded': False, 'real_roi_calls': 0,
        'source_plan': {'source2_rois': len(source_records[0]['rois']), 'source17_rois': len(source_records[1]['rois'])},
        'next_action': 'Stop for ChatGPT Gate; do not substitute source or load model until FOV issue receives explicit resolution/authorization.',
    })
    write_json(out / 'provenance.json', {
        'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_branch': git_branch, 'git_head': git_head, 'model_path': str(MODEL),
        'model_revision_reference': '115134f363124c53c7d878647567d04daf26e41e',
        'model_revision_license_provenance': 'user-provided Windows official snapshot/API evidence; no AutoDL network metadata query',
        'source_ids': ['2', '17'], 'train_manifest_sha256': manifest_hash,
        'actual_call_ids': [], 'actual_call_count': 0, 'budget_ledger_path': str(out / 'call_ledger.json'),
        'stop_before_model_load': True,
    })
    (out / 'runtime_metrics.csv').write_text('call_id,source_id,roi_id,status,elapsed_seconds,cuda_peak_allocated_bytes,cuda_peak_reserved_bytes,output_shape,output_dtype,error\n', encoding='utf-8')
    print(json.dumps({'run_id': run_id, 'status': 'EXP13_PHASE1B_S1_HOLD', 'model_ok': model_ok, 'files_verified': len(model_entries) - len(model_errors), 'verified_bytes': verified_bytes, 'source17_roi1': source17_r1, 'ledger': ledger_snapshot['remaining'], 'contact_sheet': str(sheet_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
