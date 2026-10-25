"""Deterministically screen authorized TRAIN sources for Exp13 Phase 1-B-S2."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.exp13_restoration.geometry import Polygon, build_geometry, group_polygons, pillow_polygon_mask
from tools.exp13_restoration.run_phase1a_r2 import MANIFEST, MANIFEST_SHA, RAW, sha256

REPO = Path('/root/autodl-tmp/borescope-new-seg-repro')
RUN_ID = '20261025T060000Z'
OUT = REPO / 'results/exp13_phase1b_s2' / RUN_ID


def numeric_key(row: dict) -> tuple:
    stem = str(row['stem'])
    try:
        return (0, int(stem), stem)
    except ValueError:
        return (1, stem.casefold(), stem)


def read_shapes(row: dict) -> tuple[Image.Image, dict, list[dict]]:
    image_path, annotation_path = RAW / row['image_path'], RAW / row['json_path']
    with Image.open(image_path) as src:
        image = src.convert('RGB').copy()
    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    shapes = [shape for shape in payload.get('shapes', []) if shape.get('shape_type', 'polygon') == 'polygon']
    return image, payload, shapes


def assess(row: dict, shapes: list[dict]) -> tuple[dict, list[np.ndarray], list]:
    image, _, _ = read_shapes(row)
    w, h = image.size
    masks: list[np.ndarray] = []
    geometries = []
    reasons: list[str] = []
    polys = []
    for i, shape in enumerate(shapes):
        try:
            pts = tuple((float(x), float(y)) for x, y in shape['points'])
            if len(pts) < 3 or not np.isfinite(np.asarray(pts)).all():
                reasons.append(f'polygon_{i}: invalid point list')
                continue
            if any(x < 0 or y < 0 or x >= w or y >= h for x, y in pts):
                reasons.append(f'polygon_{i}: coordinate outside native image')
                continue
            mask = pillow_polygon_mask(pts, w, h)
            if not mask.any():
                reasons.append(f'polygon_{i}: empty normative Pillow mask')
                continue
            geom = build_geometry(pts, w, h, context=32, halo_radius=2, roi_id=f'roi_{i}', source_id=str(row['stem']))
            edge = min(min(x for x, _ in pts), min(y for _, y in pts), w - 1 - max(x for x, _ in pts), h - 1 - max(y for _, y in pts))
            if edge < 24:
                reasons.append(f'polygon_{i}: border distance {edge:.3f}px < 24px')
            if geom.provenance['roi_clamped']:
                reasons.append(f'polygon_{i}: context ROI clamped to native image edge')
            masks.append(geom.edit_support)
            geometries.append({'roi_id': f'roi_{i}', 'class_name': str(shape.get('label', 'unknown')), 'points': pts, 'mask': mask, 'geometry': geom, 'edge_distance_px': edge})
            polys.append(Polygon(f'roi_{i}', pts, str(shape.get('label', 'unknown')), str(row['stem'])))
        except Exception as exc:
            reasons.append(f'polygon_{i}: invalid geometry: {type(exc).__name__}: {exc}')

    groups = group_polygons(polys, gap=0.0) if len(polys) == len(shapes) else []
    if len(shapes) == 2 and len(groups) != 2:
        reasons.append(f'independent ROI groups={len(groups)}; expected exactly 2')
    if len(masks) == 2:
        overlap = int(np.count_nonzero(masks[0] & masks[1]))
        if overlap:
            reasons.append(f'native edit-support overlap={overlap}px')
    else:
        overlap = None

    # Run the Phase 1-A-R2 native ROI composition mock on the exact source.
    mock_status = 'NOT_RUN'
    mock_reason = None
    if len(shapes) == 2 and not reasons:
        try:
            from tools.exp13_restoration.run_phase1a_r2 import validate_source
            mock = validate_source(row)
            mock_status = str(mock.get('status'))
            if mock_status != 'PASS':
                reasons.append(f'Phase 1-A-R2 native ROI composition mock={mock_status}')
        except Exception as exc:
            mock_status = 'ERROR'
            mock_reason = f'{type(exc).__name__}: {exc}'
            reasons.append(f'Phase 1-A-R2 native ROI composition mock errored: {mock_reason}')

    record = {
        'source_id': str(row['stem']), 'split': row.get('split'), 'image_path': row['image_path'], 'annotation_path': row['json_path'],
        'image_sha256': sha256(RAW / row['image_path']), 'annotation_sha256': sha256(RAW / row['json_path']),
        'native_size_wh': [w, h], 'polygon_count': len(shapes), 'roi_group_count': len(groups),
        'support_overlap_pixels': overlap, 'native_roundtrip_mock': mock_status,
        'rejection_reasons': reasons, 'geometry_eligible': len(shapes) == 2 and len(groups) == 2 and not reasons,
        'rois': [{k: v for k, v in item.items() if k not in {'mask', 'geometry'}} | {
            'polygon_mask_sha256': hashlib.sha256(item['mask'].astype(np.uint8).tobytes()).hexdigest(),
            'edit_support_sha256': hashlib.sha256(item['geometry'].edit_support.astype(np.uint8).tobytes()).hexdigest(),
            'polygon_pixels': int(item['mask'].sum()), 'edit_support_pixels': int(item['geometry'].edit_support.sum()),
            'roi_box_xyxy': [item['geometry'].roi_box.x0, item['geometry'].roi_box.y0, item['geometry'].roi_box.x1, item['geometry'].roi_box.y1],
            'canvas_transform': item['geometry'].transform.__dict__, 'roi_clamped': item['geometry'].provenance['roi_clamped'],
        } for item in geometries],
    }
    return record, [item['mask'] for item in geometries], geometries


def make_sheet(row: dict, shapes: list[dict], record: dict, path: Path) -> None:
    image, _, _ = read_shapes(row)
    w, h = image.size
    scale = min(620 / w, 620 / h, 1.0)
    size = (round(w * scale), round(h * scale))
    original = image.resize(size, Image.Resampling.LANCZOS)
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    masks = []
    for index, shape in enumerate(shapes):
        pts = [(round(float(x) * scale), round(float(y) * scale)) for x, y in shape['points']]
        color = ('red', 'lime', 'cyan', 'yellow')[index % 4]
        draw.line(pts + [pts[0]], fill=color, width=2)
        draw.text(pts[0], f"roi_{index}:{shape.get('label', 'unknown')}", fill='white', stroke_width=2, stroke_fill='black')
        mask = pillow_polygon_mask(shape['points'], w, h)
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST).convert('RGB')
        masks.append(mask_img)
    panels = [original, overlay] + masks
    titles = ['Original RGB', 'Polygon overlay (review only)'] + [f'Native polygon mask {i}' for i in range(len(masks))]
    canvas = Image.new('RGB', (size[0] * len(panels), size[1] + 30), 'white')
    d = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x = i * size[0]
        d.text((x + 4, 4), title, fill='black')
        canvas.paste(panel, (x, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    record['contact_sheet_path'] = str(path)
    record['contact_sheet_sha256'] = sha256(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('Frozen TRAIN manifest SHA256 mismatch')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        train = sorted((row for row in csv.DictReader(f) if row.get('split') == 'train'), key=numeric_key)
    source2 = next((r for r in train if str(r['stem']) == '2'), None)
    if source2 is None:
        raise RuntimeError('Authorized source 2 is absent from TRAIN')
    source2_image, _, source2_shapes = read_shapes(source2)
    source2_record, _, _ = assess(source2, source2_shapes)
    source2_record['geometry_eligible'] = len(source2_shapes) == 1 and not source2_record['rejection_reasons']
    if len(source2_shapes) != 1:
        source2_record['rejection_reasons'].append('source 2 polygon count does not equal 1')
    make_sheet(source2, source2_shapes, source2_record, OUT / 'source2_fov_review.png')

    candidates = []
    two_poly_rows = []
    for row in train:
        if str(row['stem']) in {'2', '13', '17', '50'}:
            continue
        # Candidate annotation must exist and contain exactly two polygon shapes.
        try:
            image, _, shapes = read_shapes(row)
        except Exception as exc:
            continue
        if len(shapes) != 2:
            continue
        record, _, _ = assess(row, shapes)
        two_poly_rows.append((row, shapes, record))
        if record['geometry_eligible']:
            candidates.append((row, shapes, record))
        if len(candidates) >= 8:
            break

    for i, (row, shapes, record) in enumerate(candidates, start=1):
        make_sheet(row, shapes, record, OUT / 'candidate_contact_sheets' / f'{i:02d}_source_{row["stem"]}.png')

    audit = {
        'run_id': RUN_ID, 'status': 'VISUAL_REVIEW_REQUIRED', 'train_manifest_sha256': sha256(MANIFEST),
        'candidate_order': 'numeric filename stem ascending; nonnumeric stable casefolded lexical fallback',
        'excluded_source_ids': {'13': 'bottom-edge risk not independently approved', '17': 'prior S1 right-edge/FOV risk', '50': 'complex overlap reserved', '2': 'authorized single-ROI source, not a dual candidate'},
        'source2': source2_record,
        'dual_roi_candidates_in_screening_order': [item[2] for item in candidates],
        'geometry_rejected_two_polygon_candidates': [record for _, _, record in two_poly_rows if not record['geometry_eligible']],
        'candidate_limit': 8, 'candidate_count': len(candidates),
        'selection': None, 'reason': 'Visual original-RGB/FOV review is required before any source can be frozen or any model call can occur.',
        'val_access': 0, 'test_access': 0, 'raw_dataset_write': 0, 'model_download': 0, 'model_load': 0, 'inference_calls': 0,
    }
    (OUT / 'safe_source_candidate_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({'run_id': RUN_ID, 'candidate_count': len(candidates), 'source2_contact': source2_record['contact_sheet_path'], 'candidate_sheets': [x[2]['contact_sheet_path'] for x in candidates]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
