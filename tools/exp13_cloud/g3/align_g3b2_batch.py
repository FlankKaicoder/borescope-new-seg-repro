"""Offline SIFT/RANSAC alignment and target-only composites for a frozen G3 batch."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


EXPECTED_MANIFEST_SHA256 = "2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa"
EXPECTED_SOURCES = 36
EXPECTED_PER_CLASS = 6
FEATHER_WIDTH = 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    return cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def read_target_mask(path, expected_size):
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA mask: {path}")
    if (rgba.shape[1], rgba.shape[0]) != expected_size:
        raise ValueError(f"Mask dimensions mismatch: {path}")
    # The G3 selected-only convention places the target in transparent pixels.
    target = np.uint8(rgba[:, :, 3] == 0) * 255
    if cv2.countNonZero(target) == 0 or cv2.countNonZero(target) == target.size:
        raise ValueError(f"Empty or full-frame target mask: {path}")
    return target


def support_mask(image, target):
    height, width = target.shape
    allowed = np.full((height, width), 255, np.uint8)
    margin_x, margin_y = max(2, round(width * 0.02)), max(2, round(height * 0.02))
    allowed[:margin_y] = 0
    allowed[-margin_y:] = 0
    allowed[:, :margin_x] = 0
    allowed[:, -margin_x:] = 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    allowed[gray < 12] = 0
    dilate = max(3, round(min(width, height) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
    allowed[cv2.dilate(target, kernel) > 0] = 0
    return allowed


def extract_features(image, allowed):
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                           edgeThreshold=10, sigma=1.6)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keys, desc = sift.detectAndCompute(gray, allowed)
    return keys or [], desc


def collect_matches(keys_a, desc_a, keys_b, desc_b, size_a, size_b):
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return []
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    passed = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            a, b = pair
            passed.append((a.distance / pair[1].distance, a.distance, a.queryIdx, a.trainIdx))
    used_a, used_b, result = set(), set(), []
    norm_a, norm_b = max(size_a), max(size_b)
    for ratio, distance, ia, ib in sorted(passed):
        pa, pb = keys_a[ia].pt, keys_b[ib].pt
        cell_a = tuple(int(v / norm_a * 500) for v in pa)
        cell_b = tuple(int(v / norm_b * 500) for v in pb)
        if cell_a in used_a or cell_b in used_b:
            continue
        used_a.add(cell_a)
        used_b.add(cell_b)
        result.append({"source": list(pa), "output": list(pb), "ratio": float(ratio),
                       "descriptor_l2": float(distance)})
    return result


def project(matrix, points):
    return np.asarray(points, np.float64) @ matrix[:, :2].T + matrix[:, 2]


def distribution(points, width, height):
    if len(points) < 3:
        return {"hull_fraction": 0.0, "occupied_cells": 0, "separated_cells": False,
                "span_fraction": [0.0, 0.0]}
    p = np.asarray(points, np.float32)
    hull = cv2.contourArea(cv2.convexHull(p)) / float(width * height)
    cells = {}
    for x, y in p:
        key = (min(2, int(x / width * 3)), min(2, int(y / height * 3)))
        cells.setdefault(key, []).append((float(x), float(y)))
    supported = [np.mean(v, axis=0) for v in cells.values() if len(v) >= 2]
    separated = any(np.linalg.norm(a - b) >= 0.2 * math.hypot(width, height)
                    for i, a in enumerate(supported) for b in supported[i + 1:])
    span = p.max(axis=0) - p.min(axis=0)
    return {"hull_fraction": float(hull), "occupied_cells": len(supported),
            "separated_cells": bool(separated),
            "span_fraction": [float(span[0] / width), float(span[1] / height)]}


def evaluate_matrix(matrix, inliers, pairs, output_size, source_size, method):
    sw, sh = source_size
    ow, oh = output_size
    source_points = np.float64([p["output"] for p in pairs])
    target_points = np.float64([p["source"] for p in pairs])
    errors = np.linalg.norm(project(matrix, source_points) - target_points, axis=1)
    keep = inliers.ravel().astype(bool)
    count = int(keep.sum())
    target_inliers = target_points[keep]
    dist = distribution(target_inliers, sw, sh)
    median = float(np.median(errors[keep])) if count else None
    p90 = float(np.percentile(errors[keep], 90)) if count else None
    threshold = max(2.0, 0.005 * math.hypot(sw, sh))
    corners = np.float64([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    mapped = project(matrix, corners)
    homography = cv2.convexHull(mapped.astype(np.float32))
    overlap = cv2.intersectConvexConvex(homography, np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]))[0]
    overlap_fraction = float(overlap / (sw * sh))
    det = float(np.linalg.det(matrix[:, :2]))
    scales = np.linalg.svd(matrix[:, :2], compute_uv=False)
    checks = {
        "minimum_12_inliers": count >= 12,
        "inlier_ratio_at_least_0_30": count / max(1, len(pairs)) >= 0.30,
        "median_error_within_0_5pct_diagonal": median is not None and median <= threshold,
        "p90_error_within_1pct_diagonal": p90 is not None and p90 <= threshold * 2,
        "distributed_inliers": dist["hull_fraction"] >= 0.05 and dist["separated_cells"]
                               and min(dist["span_fraction"]) >= 0.20,
        "valid_overlap_at_least_half": overlap_fraction >= 0.5,
        "positive_orientation": det > 0,
        "bounded_scale": bool(len(scales) == 2 and scales[-1] >= 0.35 and scales[0] <= 2.8
                               and scales[0] / scales[-1] <= 1.25),
    }
    return {"method": method, "status": "GEOMETRY_CANDIDATE" if all(checks.values()) else "ALIGNMENT_UNVERIFIED",
            "matrix_output_to_source": matrix.tolist(), "match_count": len(pairs),
            "inlier_count": count, "inlier_ratio": count / max(1, len(pairs)),
            "ransac_threshold_source_px": threshold, "median_reprojection_error_source_px": median,
            "p90_reprojection_error_source_px": p90, "valid_overlap_fraction_estimate": overlap_fraction,
            "inlier_distribution": dist, "singular_scales_output_to_source": scales.tolist(),
            "determinant": det, "criteria": checks}, keep


def fit_models(pairs, output_size, source_size):
    if len(pairs) < 3:
        return []
    source = np.float64([p["output"] for p in pairs])
    target = np.float64([p["source"] for p in pairs])
    threshold = max(2.0, 0.005 * math.hypot(*source_size))
    fits = []
    for name, estimator in (("SIFT_RANSAC_SIMILARITY", cv2.estimateAffinePartial2D),
                            ("SIFT_RANSAC_AFFINE", cv2.estimateAffine2D)):
        cv2.setRNGSeed(20261108)
        matrix, inliers = estimator(source, target, method=cv2.RANSAC,
                                    ransacReprojThreshold=threshold, maxIters=3000,
                                    confidence=0.99, refineIters=10)
        if matrix is not None and inliers is not None and np.isfinite(matrix).all():
            fits.append(evaluate_matrix(matrix, inliers, pairs, output_size, source_size, name))
    return fits


def resize_candidate(source_size, output_size):
    sw, sh = source_size
    ow, oh = output_size
    scale_x, scale_y = sw / ow, sh / oh
    return np.float64([[scale_x, 0, 0], [0, scale_y, 0]])


def validate_resize_candidate(pairs, matrix, source_size, output_size):
    sw, sh = source_size
    ow, oh = output_size
    sp = np.float64([p["output"] for p in pairs])
    tp = np.float64([p["source"] for p in pairs])
    error = np.linalg.norm(project(matrix, sp) - tp, axis=1) if len(pairs) else np.array([])
    threshold = max(2.0, 0.005 * math.hypot(sw, sh))
    keep = error <= threshold if len(error) else np.zeros(0, bool)
    dist = distribution(tp[keep], sw, sh)
    passed = (int(keep.sum()) >= 12 and keep.sum() / max(1, len(pairs)) >= 0.30
              and float(np.median(error[keep])) <= threshold if keep.any() else False)
    passed = bool(passed and dist["hull_fraction"] >= 0.05 and dist["separated_cells"]
                  and min(dist["span_fraction"]) >= 0.20)
    return {"method": "ASPECT_RATIO_RESIZE_VALIDATED_BY_SIFT" if passed else "ASPECT_RATIO_RESIZE_REJECTED",
            "status": "GEOMETRY_CANDIDATE" if passed else "ALIGNMENT_UNVERIFIED",
            "matrix_output_to_source": matrix.tolist(), "match_count": len(pairs),
            "inlier_count": int(keep.sum()), "inlier_ratio": float(keep.sum() / max(1, len(pairs))),
            "median_reprojection_error_source_px": float(np.median(error[keep])) if keep.any() else None,
            "p90_reprojection_error_source_px": float(np.percentile(error[keep], 90)) if keep.any() else None,
            "valid_overlap_fraction_estimate": 1.0,
            "inlier_distribution": dist,
            "criteria": {"aspect_ratio_difference_le_0_5pct": True,
                         "minimum_12_consistent_sift_matches": int(keep.sum()) >= 12,
                         "spatially_distributed_matches": bool(passed)}}


def valid_support(matrix, output_size, source_size):
    ow, oh = output_size
    sw, sh = source_size
    support = np.full((oh, ow), 255, np.uint8)
    aligned = cv2.warpAffine(support, matrix, (sw, sh), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned = cv2.erode(aligned, np.ones((3, 3), np.uint8), iterations=1)
    return aligned


def composites(original, aligned, valid, target):
    target_valid = cv2.bitwise_and(target, valid)
    hard = original.copy()
    hard[target_valid > 0] = aligned[target_valid > 0]
    distance = cv2.distanceTransform(target, cv2.DIST_L2, 5)
    alpha = np.uint8(np.clip(distance / FEATHER_WIDTH, 0, 1) * 255)
    alpha[valid == 0] = 0
    alpha[target == 0] = 0
    mix = alpha.astype(np.float32)[:, :, None] / 255.0
    feather = np.uint8(np.clip(aligned.astype(np.float32) * mix + original.astype(np.float32) * (1 - mix), 0, 255))
    return target_valid, hard, feather


def json_polygon_ids(json_path, source_id):
    data = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    result = []
    for index, shape in enumerate(data.get("shapes", [])):
        if shape.get("shape_type") == "polygon":
            result.append(f"{source_id}_{index:03d}")
    return result


def overlay_image(original, json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    overlay = original.copy()
    colors = [(255, 70, 70), (70, 220, 255), (255, 210, 40), (130, 255, 120),
              (220, 120, 255), (255, 150, 60), (100, 180, 255)]
    for i, shape in enumerate(data.get("shapes", [])):
        if shape.get("shape_type") != "polygon":
            continue
        pts = np.int32(np.round(shape["points"])).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, colors[i % len(colors)], max(2, round(min(original.shape[:2]) / 240)))
    return overlay


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_html(out, records):
    cards = []
    for item in records:
        rel = lambda name: html.escape(f"{item['source_id']}/{name}", quote=True)
        geometry = html.escape(item["geometry_status"])
        panels = [("Original", "original.png"), ("Polygon Overlay", "polygon_overlay.png"),
                  ("Raw Cloud Output", "raw_output.png"), ("Aligned Output", "aligned_output.png"),
                  ("Hard Composite", "hard_composite.png"), ("Feather Composite", "feather_composite.png"),
                  ("Resize Preview ONLY - NOT ALIGNED", "resize_preview_unverified.png"),
                  ("Local Comparison", "local_comparison.png")]
        figures = []
        for title, filename in panels:
            p = out / item["source_id"] / filename
            if p.exists():
                figures.append(f'<figure><figcaption>{html.escape(title)}</figcaption><img loading="lazy" src="{rel(filename)}"></figure>')
        diag = html.escape(json.dumps({k: item.get(k) for k in (
            "source_size", "output_size", "aspect_ratio_difference", "mapping_method", "match_count",
            "inlier_count", "inlier_ratio", "median_reprojection_error_source_px",
            "p90_reprojection_error_source_px", "valid_pixel_coverage", "nondefect_alignment",
            "defect_edge_risk", "geometry_status")}, ensure_ascii=False))
        cards.append(f'<article data-class="{html.escape(item["class"], quote=True)}" data-source="{html.escape(item["source_id"], quote=True)}"><h2>{html.escape(item["source_id"])} | {html.escape(item["class"])} | {geometry}</h2><details><summary>Geometry diagnostics</summary><pre>{diag}</pre></details><div class="grid">{"".join(figures)}</div><label>Visual <select data-field="visual"><option>UNREVIEWED</option><option>VISUAL_PASS</option><option>VISUAL_FAIL</option><option>NEEDS_RETRY</option></select></label> <label>Geometry <select data-field="geometry"><option>{geometry}</option><option>GEOMETRY_CANDIDATE</option><option>ALIGNMENT_UNVERIFIED</option></select></label> <label>Notes <input data-field="notes"></label></article>')
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G3-C0 Alignment Review</title><style>body{font:14px system-ui;margin:0;background:#f2f4f5;color:#202428}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #bbb;z-index:3;display:flex;gap:12px;align-items:center;flex-wrap:wrap}article{margin:14px;padding:14px;background:white;border:1px solid #bbc2c6;border-radius:4px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}figure{margin:0}figcaption{font-weight:650}img{width:100%;height:300px;object-fit:contain;background:#202428}pre{white-space:pre-wrap;overflow-wrap:anywhere}input[type=search]{min-width:180px}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.grid{grid-template-columns:1fr}img{height:auto;max-height:60vh}}</style><header><strong>G3-C0 Geometry Review</strong><label>Class <select id="filter"><option value="">All</option></select></label><label>Source ID <input id="search" type="search"></label><button id="download">Download CSV</button><span id="count"></span></header><main>''' + "".join(cards) + '''</main><script>
const cards=[...document.querySelectorAll('article')], classes=[...new Set(cards.map(x=>x.dataset.class))];for(const c of classes){const o=document.createElement('option');o.value=c;o.textContent=c;document.querySelector('#filter').append(o)}function filter(){let c=document.querySelector('#filter').value,s=document.querySelector('#search').value.toLowerCase();for(const x of cards)x.hidden=(c&&x.dataset.class!==c)||!x.dataset.source.toLowerCase().includes(s)}document.querySelector('#filter').onchange=filter;document.querySelector('#search').oninput=filter;function update(){document.querySelector('#count').textContent=cards.length+' sources; '+cards.filter(x=>x.querySelector('[data-field=visual]').value!=='UNREVIEWED').length+' visually reviewed'}cards.forEach(x=>x.querySelectorAll('select,input').forEach(y=>y.addEventListener('change',update)));update();document.querySelector('#download').onclick=()=>{const rows=[...cards].map(x=>[x.dataset.source,x.querySelector('[data-field=visual]').value,x.querySelector('[data-field=geometry]').value,x.querySelector('[data-field=notes]').value]);const q=v=>'"'+String(v).replaceAll('"','""')+'"',text='source_id,visual_review_status,geometry_status,review_notes\r\n'+rows.map(r=>r.map(q).join(',')).join('\r\n');const a=document.createElement('a'),u=URL.createObjectURL(new Blob(['\ufeff'+text],{type:'text/csv;charset=utf-8'}));a.href=u;a.download='g3c0_review.csv';document.body.append(a);a.click();setTimeout(()=>{a.remove();URL.revokeObjectURL(u)},1000)};
</script>'''
    from build_g3c0_review_page import enhance_html
    page, _ = enhance_html(page, HERE / "review_ui.js")
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")


def registration_candidates(original, raw, target):
    source_size = (original.shape[1], original.shape[0])
    output_size = (raw.shape[1], raw.shape[0])
    aspect_delta = abs((source_size[0] / source_size[1]) / (output_size[0] / output_size[1]) - 1)
    original_allowed = support_mask(original, target)
    raw_target_prior = cv2.resize(target, output_size, interpolation=cv2.INTER_NEAREST)
    raw_allowed = support_mask(raw, raw_target_prior)
    keys_a, desc_a = extract_features(original, original_allowed)
    keys_b, desc_b = extract_features(raw, raw_allowed)
    pairs = collect_matches(keys_a, desc_a, keys_b, desc_b, source_size, output_size)
    candidates = []
    if aspect_delta <= 0.005:
        candidates.append(validate_resize_candidate(pairs, resize_candidate(source_size, output_size),
                                                    source_size, output_size))
    candidates.extend(fit for fit, _ in fit_models(pairs, output_size, source_size))
    passes = [candidate for candidate in candidates if candidate["status"] == "GEOMETRY_CANDIDATE"]
    selected = min(passes, key=lambda candidate: candidate["median_reprojection_error_source_px"]) if passes else None
    return candidates, selected, len(pairs), aspect_delta


def process_row(row, expected_raw_sha256, train_root, atlas_root, generation_root, out):
    source_id = str(row["source_id"])
    image_path = train_root / row["image_path"]
    json_path = train_root / row["json_path"]
    mask_path = atlas_root / row["mask_path"]
    raw_path = generation_root / "live" / source_id / "raw_output.png"
    required = [image_path, json_path, mask_path, raw_path]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError(f"Missing frozen batch input for source {source_id}")
    if sha256(image_path) != row["source_image_sha256"]:
        raise ValueError(f"Source image SHA256 mismatch: {source_id}")
    if sha256(json_path) != row["source_json_sha256"]:
        raise ValueError(f"Source JSON SHA256 mismatch: {source_id}")
    if sha256(mask_path) != row["mask_sha256"]:
        raise ValueError(f"Target mask SHA256 mismatch: {source_id}")
    if sha256(raw_path) != expected_raw_sha256:
        raise ValueError(f"Raw output SHA256 differs from generation ledger: {source_id}")
    original, raw = read_rgb(image_path), read_rgb(raw_path)
    source_size, output_size = (original.shape[1], original.shape[0]), (raw.shape[1], raw.shape[0])
    if [int(row["image_width"]), int(row["image_height"])] != list(source_size):
        raise ValueError(f"Manifest source dimensions mismatch: {source_id}")
    annotation = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if [annotation.get("imageWidth"), annotation.get("imageHeight")] != list(source_size):
        raise ValueError(f"Professional JSON dimensions mismatch: {source_id}")
    if Path(annotation.get("imagePath", "")).name != image_path.name:
        raise ValueError(f"Professional JSON imagePath identity mismatch: {source_id}")
    target = read_target_mask(mask_path, source_size)
    polygon_ids = json_polygon_ids(json_path, source_id)
    if set(row["selected_polygon_ids"].split("|")) - set(polygon_ids):
        raise ValueError(f"Selected Polygon ID mismatch: {source_id}")
    candidates, selected, match_count, aspect_delta = registration_candidates(original, raw, target)
    folder = out / source_id
    folder.mkdir(exist_ok=False)
    write_rgb(folder / "original.png", original)
    write_rgb(folder / "polygon_overlay.png", overlay_image(original, json_path))
    write_rgb(folder / "raw_output.png", raw)
    preview = cv2.resize(raw, source_size, interpolation=cv2.INTER_LANCZOS4)
    write_rgb(folder / "resize_preview_unverified.png", preview)
    record = {"source_id": source_id, "class": row["class"], "source_image_sha256": row["source_image_sha256"],
              "raw_output_sha256": sha256(raw_path), "mask_sha256": row["mask_sha256"],
              "polygon_ids": row["selected_polygon_ids"], "source_size": list(source_size),
              "output_size": list(output_size), "aspect_ratio_difference": aspect_delta,
              "mapping_method": selected["method"] if selected else "NONE",
              "match_count": match_count, "geometry_status": "GEOMETRY_CANDIDATE" if selected else "ALIGNMENT_UNVERIFIED",
              "valid_pixel_coverage": 0.0, "nondefect_alignment": "UNVERIFIED",
              "defect_edge_risk": "PENDING_VISUAL_REVIEW",
              "trusted_training_pair_status": "NOT_GRANTED", "visual_review_status": "UNREVIEWED",
              "candidate_diagnostics": candidates}
    if selected:
        matrix = np.float64(selected["matrix_output_to_source"])
        aligned = cv2.warpAffine(raw, matrix, source_size, flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        valid = valid_support(matrix, output_size, source_size)
        target_valid, hard, feather = composites(original, aligned, valid, target)
        record.update({"inlier_count": selected["inlier_count"], "inlier_ratio": selected["inlier_ratio"],
                       "median_reprojection_error_source_px": selected.get("median_reprojection_error_source_px"),
                       "p90_reprojection_error_source_px": selected.get("p90_reprojection_error_source_px"),
                       "valid_pixel_coverage": float(cv2.countNonZero(valid) / valid.size),
                       "nondefect_alignment": "FEATURE_SCREEN_CANDIDATE_PENDING_HUMAN",
                       "defect_edge_risk": "TARGET_ONLY; VERIFY_VISUALLY"})
        for name, image in (("aligned_output.png", aligned), ("hard_composite.png", hard),
                            ("feather_composite.png", feather), ("valid_mask.png", valid),
                            ("valid_target_mask.png", target_valid)):
            saved = write_rgb(folder / name, image) if image.ndim == 3 else cv2.imwrite(str(folder / name), image)
            if not saved:
                raise OSError(f"Failed writing {folder / name}")
        record["derived_sha256"] = {name: sha256(folder / name) for name in (
            "aligned_output.png", "hard_composite.png", "feather_composite.png",
            "valid_mask.png", "valid_target_mask.png")}
        x, y, w, h = cv2.boundingRect(target)
        pad = max(8, round(min(source_size) * 0.04))
        box = (max(0, x-pad), max(0, y-pad), min(source_size[0], x+w+pad), min(source_size[1], y+h+pad))
        local_panels = [original[box[1]:box[3], box[0]:box[2]],
                        aligned[box[1]:box[3], box[0]:box[2]],
                        hard[box[1]:box[3], box[0]:box[2]],
                        feather[box[1]:box[3], box[0]:box[2]]]
        for panel, title in zip(local_panels, ("ORIGINAL", "ALIGNED", "HARD", "FEATHER")):
            cv2.putText(panel, title, (8, min(24, panel.shape[0] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, min(source_size) / 900), (255, 255, 255), 1, cv2.LINE_AA)
        local = np.concatenate(local_panels, axis=1)
        write_rgb(folder / "local_comparison.png", local)
    else:
        # This is explicitly a display-only resize, never treated as aligned evidence.
        write_rgb(folder / "local_comparison.png", preview)
    (folder / "geometry_diagnostics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                                        encoding="utf-8", newline="\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--generation-review-csv", required=True, type=Path)
    parser.add_argument("--manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if sha256(manifest) != args.manifest_sha256:
        raise SystemExit("Expanded batch Manifest SHA256 mismatch")
    train_root, atlas_root, generation_root = map(Path.resolve, (args.train_root, args.atlas_root, args.generation_root))
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refuse existing output directory: {output}")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with args.generation_review_csv.resolve().open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_SOURCES or len({r["source_id"] for r in rows}) != EXPECTED_SOURCES:
        raise SystemExit("Manifest is not the expected 36-source exact set")
    review_map = {r["source_id"]: r for r in review_rows}
    if len(review_map) != EXPECTED_SOURCES or set(review_map) != {r["source_id"] for r in rows}:
        raise SystemExit("Generation review ledger and batch Manifest source sets differ")
    if any(r["generation_status"] != "SUCCEEDED" or r["http_status"] != "200"
           or not r["raw_output_sha256"] for r in review_map.values()):
        raise SystemExit("Generation ledger contains a non-success or missing output identity")
    from collections import Counter
    counts = Counter(row["class"] for row in rows)
    if len(counts) != 6 or any(value != EXPECTED_PER_CLASS for value in counts.values()):
        raise SystemExit(f"Unexpected source class distribution: {counts}")
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for row in rows:
        records.append(process_row(row, review_map[row["source_id"]]["raw_output_sha256"],
                                   train_root, atlas_root, generation_root, output))
    fields = ["source_id", "class", "source_image_sha256", "raw_output_sha256", "mask_sha256",
              "polygon_ids", "source_size", "output_size", "aspect_ratio_difference", "mapping_method",
              "match_count", "inlier_count", "inlier_ratio", "median_reprojection_error_source_px",
              "p90_reprojection_error_source_px", "valid_pixel_coverage", "nondefect_alignment",
              "defect_edge_risk", "geometry_status", "visual_review_status", "trusted_training_pair_status",
              "derived_sha256"]
    csv_rows = []
    for item in records:
        csv_rows.append({**item, "polygon_ids": item["polygon_ids"], "source_size": "x".join(map(str, item["source_size"])),
                         "output_size": "x".join(map(str, item["output_size"])),
                         "candidate_diagnostics": json.dumps(item["candidate_diagnostics"], ensure_ascii=False)})
    write_csv(output / "geometry_review.csv", csv_rows, fields)
    write_csv(output / "review.csv", [{"source_id": r["source_id"], "visual_review_status": "UNREVIEWED",
              "geometry_status": r["geometry_status"], "review_notes": ""} for r in records],
              ["source_id", "visual_review_status", "geometry_status", "review_notes"])
    summary = {"status": "G3C0_OFFLINE_ALIGNMENT_COMPLETE", "manifest_sha256": sha256(manifest),
               "source_count": len(records), "geometry_candidate_count": sum(r["geometry_status"] == "GEOMETRY_CANDIDATE" for r in records),
               "alignment_unverified_count": sum(r["geometry_status"] == "ALIGNMENT_UNVERIFIED" for r in records),
               "aligned_output_count": sum((output / r["source_id"] / "aligned_output.png").is_file() for r in records),
               "hard_composite_count": sum((output / r["source_id"] / "hard_composite.png").is_file() for r in records),
               "feather_composite_count": sum((output / r["source_id"] / "feather_composite.png").is_file() for r in records),
               "api_posts": 0, "gpu_training": 0, "test_image_access": 0,
               "trusted_training_pair_pass": 0, "records": records}
    (output / "geometry_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    render_html(output, records)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
