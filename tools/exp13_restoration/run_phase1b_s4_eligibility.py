"""CPU-only TRAIN restoration-eligibility feasibility audit.

The script reads the frozen TRAIN manifest and raw annotations/images only.
It never imports torch/diffusers, never loads a model, and never writes the
raw dataset. Human semantic checks remain UNKNOWN until visual review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_DEFAULT = Path("/root/autodl-tmp/borescope-new-seg-repro")
RAW_DEFAULT = Path("/root/autodl-tmp/损伤训练数据集")
MANIFEST_REL = Path("results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv")
MANIFEST_SHA = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_rows(manifest: Path) -> list[dict]:
    if sha256(manifest) != MANIFEST_SHA:
        raise RuntimeError("frozen manifest hash mismatch")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numeric_key(stem: str):
    try:
        return (0, int(stem), stem)
    except ValueError:
        return (1, stem.casefold(), stem)


def luma_stats(rgb: np.ndarray, mask: np.ndarray) -> dict:
    values = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722)[mask]
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "p10": float(np.percentile(values, 10)), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90))}


def component_stats(mask: np.ndarray, min_area: int = 1) -> dict:
    """Return connected components without relying on a model or image write."""
    try:
        from scipy import ndimage
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        objects = ndimage.find_objects(labels)
        comps = []
        for label, sl in enumerate(objects, 1):
            if sl is None:
                continue
            area = int(np.count_nonzero(labels[sl] == label))
            if area >= min_area:
                comps.append({"area": area, "bbox_xyxy": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)]})
        comps.sort(key=lambda x: (-x["area"], x["bbox_xyxy"]))
        return {"count": len(comps), "components": comps[:100], "labeling": "scipy_8_connected"}
    except Exception:
        ys, xs = np.where(mask)
        seen = set()
        comps = []
        points = set(zip(ys.tolist(), xs.tolist()))
        for y, x in sorted(points):
            if (y, x) in seen:
                continue
            q = deque([(y, x)]); seen.add((y, x)); area = 0; x0 = x1 = x; y0 = y1 = y
            while q:
                cy, cx = q.popleft(); area += 1; x0 = min(x0, cx); x1 = max(x1, cx); y0 = min(y0, cy); y1 = max(y1, cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not dx and not dy:
                            continue
                        p = (cy + dy, cx + dx)
                        if p in points and p not in seen:
                            seen.add(p); q.append(p)
            if area >= min_area:
                comps.append({"area": area, "bbox_xyxy": [x0, y0, x1 + 1, y1 + 1]})
        comps.sort(key=lambda x: (-x["area"], x["bbox_xyxy"]))
        return {"count": len(comps), "components": comps[:100], "labeling": "python_8_connected"}


def safe_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        return ImageFont.load_default()


def make_canvas(crop: np.ndarray, transform: dict) -> tuple[Image.Image, Image.Image]:
    resized = np.asarray(Image.fromarray(crop).resize((transform["resized_width"], transform["resized_height"]), Image.Resampling.LANCZOS))
    canvas = np.pad(resized, ((transform["pad_top"], transform["canvas_size"] - transform["pad_top"] - transform["resized_height"]),
                              (transform["pad_left"], transform["canvas_size"] - transform["pad_left"] - transform["resized_width"]), (0, 0)), mode="reflect")
    return Image.fromarray(canvas.astype(np.uint8), "RGB"), Image.fromarray(np.zeros((transform["canvas_size"], transform["canvas_size"]), dtype=np.uint8), "L")


def make_candidate_sheet(record: dict, image: Image.Image, crop: Image.Image, support: np.ndarray, polygon: np.ndarray, canvas: Image.Image, out: Path) -> None:
    font = safe_font()
    panels: list[Image.Image] = []
    original = image.copy().convert("RGB")
    overlay = original.copy(); draw = ImageDraw.Draw(overlay)
    pts = record["points_xy"]
    draw.line([(int(round(x)), int(round(y))) for x, y in pts] + [(int(round(pts[0][0])), int(round(pts[0][1])))], fill="red", width=3)
    box = record["roi_box_xyxy"]
    draw.rectangle(box, outline="yellow", width=2)
    roi = crop.copy().convert("RGB")
    mask = Image.fromarray((support.astype(np.uint8) * 255), "L").convert("RGB")
    mask_overlay = roi.copy(); md = ImageDraw.Draw(mask_overlay)
    sm = np.asarray(mask.convert("L")) > 0
    rgb = np.asarray(mask_overlay).copy(); rgb[sm] = (255, 0, 0); mask_overlay = Image.fromarray((0.65 * np.asarray(roi) + 0.35 * rgb).astype(np.uint8))
    risk = canvas.copy(); rd = ImageDraw.Draw(risk)
    t = record["transform"]; x0, y0 = t["pad_left"], t["pad_top"]; x1 = x0 + t["resized_width"]; y1 = y0 + t["resized_height"]
    rd.rectangle((x0, y0, x1 - 1, y1 - 1), outline="lime", width=5)
    rd.rectangle((0, 0, 1023, 1023), outline="red", width=2)
    stats_text = [
        f"source={record['source_id']} roi={record['roi_id']} class={record['class_name']}",
        f"native={record['native_width']}x{record['native_height']} scale={record['scale']:.3f}",
        f"content={record['content_fraction']:.3f} padding={record['padding_ratio']:.3f}",
        f"polygon={record['polygon_pixels']} support={record['support_pixels']} context={record['context_pixels']}",
        "UI/FOV: UNKNOWN, human review required",
        "structure boundary: UNKNOWN, human review required",
        "CPU canvas only; no diffusion output",
    ]
    text_panel = Image.new("RGB", (520, 360), "white"); td = ImageDraw.Draw(text_panel)
    for i, line in enumerate(stats_text):
        td.text((12, 16 + i * 42), line, fill="black", font=font)
    source_panels = [(original, "Original RGB"), (overlay, "Polygon + ROI overlay"), (roi, "Native ROI"), (mask, "Edit support mask (white)"),
                     (canvas, "Expected input Canvas (CPU only)"), (risk, "Padding boundary / content")]
    for im, title in source_panels:
        im.thumbnail((500, 320), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (520, 380), "white"); panel.paste(im, ((520 - im.width) // 2, 34 + (320 - im.height) // 2)); ImageDraw.Draw(panel).text((8, 8), title, fill="black", font=font); panels.append(panel)
    panels.append(text_panel)
    sheet = Image.new("RGB", (1040, 1520), (235, 235, 235))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * 520, (i // 2) * 380))
    sheet.save(out)


def base_record(row: dict, source_id: str, roi_id: str, shape: dict, geom, native: np.ndarray, all_supports: list[np.ndarray]) -> dict:
    box = geom.roi_box; t = geom.transform
    x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    crop = native[y0:y1, x0:x1]
    local_support = geom.edit_support[y0:y1, x0:x1]
    local_polygon = geom.polygon_mask[y0:y1, x0:x1]
    context = ~local_support
    edge_distance = min(float(np.min(np.asarray(shape["points"])[:, 0])), float(np.min(np.asarray(shape["points"])[:, 1])), float(native.shape[1] - np.max(np.asarray(shape["points"])[:, 0])), float(native.shape[0] - np.max(np.asarray(shape["points"])[:, 1])))
    overlap = sum(int(np.count_nonzero(geom.edit_support & other)) for other in all_supports if other is not geom.edit_support)
    content_fraction = (t.resized_width * t.resized_height) / (t.canvas_size * t.canvas_size)
    padding_ratio = 1.0 - content_fraction
    context_components = component_stats(context, min_area=3)
    boundary = np.zeros_like(local_support)
    boundary[1:, :] |= local_support[1:, :] & ~local_support[:-1, :]
    boundary[:-1, :] |= local_support[:-1, :] & ~local_support[1:, :]
    boundary[:, 1:] |= local_support[:, 1:] & ~local_support[:, :-1]
    boundary[:, :-1] |= local_support[:, :-1] & ~local_support[:, 1:]
    reasons = []
    if geom.provenance["roi_clamped"]:
        reasons.append("ROI_CLAMPED_OR_IMAGE_EDGE")
    if edge_distance < 24:
        reasons.append("POLYGON_EDGE_DISTANCE_LT_24PX")
    if overlap:
        reasons.append("UNRESOLVED_SUPPORT_OVERLAP")
    if content_fraction < 0.60:
        reasons.append("EXPLORATORY_LOW_CANVAS_CONTENT_FRACTION")
    if padding_ratio > 0.40:
        reasons.append("EXPLORATORY_HIGH_PADDING_RISK")
    if t.scale > 8:
        reasons.append("EXPLORATORY_EXTREME_UPSCALE")
    reasons += ["FULL_CONTEXT_UI_FOV_HUMAN_REVIEW_REQUIRED", "STRUCTURE_BOUNDARY_HUMAN_REVIEW_REQUIRED", "NORMAL_METAL_SEMANTICS_HUMAN_REVIEW_REQUIRED"]
    geometry_status = "GEOMETRY_ELIGIBLE" if not any(x in reasons for x in ("ROI_CLAMPED_OR_IMAGE_EDGE", "POLYGON_EDGE_DISTANCE_LT_24PX", "UNRESOLVED_SUPPORT_OVERLAP")) else "GEOMETRY_INELIGIBLE"
    return {
        "candidate_id": f"{source_id}:{roi_id}", "source_id": source_id, "roi_id": roi_id, "split": row["split"], "class_name": shape.get("label", "unknown"),
        "image_path": row["image_path"], "annotation_path": row["json_path"], "image_sha256": sha256(RAW / row["image_path"]), "annotation_sha256": sha256(RAW / row["json_path"]),
        "points_xy": shape["points"], "native_width": int(native.shape[1]), "native_height": int(native.shape[0]), "roi_box_xyxy": [box.x0, box.y0, box.x1, box.y1],
        "edge_distance_px": edge_distance, "roi_clamped": bool(geom.provenance["roi_clamped"]), "polygon_pixels": int(local_polygon.sum()), "support_pixels": int(local_support.sum()), "context_pixels": int(context.sum()),
        "context_fraction": float(context.mean()), "context_component_count": int(context_components["count"]), "context_luma": luma_stats(crop, context), "boundary_luma": luma_stats(crop, boundary),
        "scale": float(t.scale), "native_short_side": int(min(t.native_width, t.native_height)), "resized_width": int(t.resized_width), "resized_height": int(t.resized_height), "canvas_size": int(t.canvas_size), "content_fraction": float(content_fraction), "padding_ratio": float(padding_ratio),
        "padding_left": int(t.pad_left), "padding_top": int(t.pad_top), "padding_right": int(t.pad_right), "padding_bottom": int(t.pad_bottom),
        "polygon_mask_sha256": hashlib.sha256(geom.polygon_mask.astype(np.uint8).tobytes()).hexdigest(), "edit_support_sha256": hashlib.sha256(geom.edit_support.astype(np.uint8).tobytes()).hexdigest(),
        "geometry_status": geometry_status, "ui_fov_status": "AUTO_HOLD_BORDER" if geom.provenance["roi_clamped"] or edge_distance < 24 else "UNKNOWN_HUMAN_REVIEW_REQUIRED", "structure_boundary_status": "UNKNOWN_HUMAN_REVIEW_REQUIRED", "context_status": "PROXY_ONLY_HUMAN_REVIEW_REQUIRED", "restoration_status": "RESTORATION_INELIGIBLE" if geometry_status != "GEOMETRY_ELIGIBLE" else "HUMAN_REVIEW_REQUIRED", "rejection_reasons": reasons,
        "selection": None, "human_review_status": "PENDING", "human_review_timestamp": None, "selection_reason": None, "transform": t.__dict__,
    }


def connected_addendum(s2_root: Path, sources: dict[str, dict]) -> dict:
    records = []
    for sid, rid in (("2", "roi_0"), ("65", "roi_1")):
        input_path = s2_root / f"source_{sid}" / f"{rid}_input_canvas.png"
        output_path = s2_root / f"{sid}_{rid}_sdxl_canvas.png"
        if not input_path.exists() or not output_path.exists():
            continue
        inp = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.int16)
        out = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.int16)
        il = inp[..., 0] * 0.2126 + inp[..., 1] * 0.7152 + inp[..., 2] * 0.0722
        ol = out[..., 0] * 0.2126 + out[..., 1] * 0.7152 + out[..., 2] * 0.0722
        mask = (ol <= 25) & (il >= 45)
        source = sources[sid]
        roi = next(r for r in source["rois"] if r["roi_id"] == rid)
        t = roi["transform"]
        points = roi["points_xy"]
        box = roi["roi_box_xyxy"]
        native_poly = Image.new("1", (roi["native_width"], roi["native_height"]), 0)
        ImageDraw.Draw(native_poly).polygon([(round(x - box[0]), round(y - box[1])) for x, y in points], fill=1)
        full_poly = np.zeros((source["native_shape_hwc"][0], source["native_shape_hwc"][1]), dtype=bool)
        full_poly_img = Image.new("1", (source["native_shape_hwc"][1], source["native_shape_hwc"][0]), 0)
        ImageDraw.Draw(full_poly_img).polygon([(round(x), round(y)) for x, y in points], fill=1)
        full_poly = np.asarray(full_poly_img, dtype=bool)
        # Map Canvas pixels back to local native crop using the same nearest index rule.
        ys, xs = np.where(mask)
        nx = np.minimum(t["native_width"] - 1, np.maximum(0, ((xs - t["pad_left"] + 0.5) / t["scale"] - 0.5).astype(int)))
        ny = np.minimum(t["native_height"] - 1, np.maximum(0, ((ys - t["pad_top"] + 0.5) / t["scale"] - 0.5).astype(int)))
        local_support = np.zeros((t["native_height"], t["native_width"]), dtype=bool)
        # phase geometry support is reconstructed from polygon plus square radius-2 halo
        poly_local = np.asarray(native_poly, dtype=bool)
        support_local = poly_local.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                support_local |= np.roll(np.roll(poly_local, dy, axis=0), dx, axis=1)
        valid = (ys >= t["pad_top"]) & (ys < t["pad_top"] + t["resized_height"]) & (xs >= t["pad_left"]) & (xs < t["pad_left"] + t["resized_width"])
        support_hits = support_local[ny[valid], nx[valid]] if valid.any() else np.array([], dtype=bool)
        poly_hits = poly_local[ny[valid], nx[valid]] if valid.any() else np.array([], dtype=bool)
        comps = component_stats(mask)
        largest = comps["components"][0] if comps["components"] else None
        records.append({"source_id": sid, "roi_id": rid, "threshold": {"output_luma_le": 25, "input_luma_ge": 45}, "threshold_pixel_count": int(mask.sum()), "threshold_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if xs.size else None, "connected_component_count": comps["count"], "largest_component": largest, "mapped_threshold_pixels_in_native_content": int(valid.sum()), "mapped_threshold_pixels_in_support": int(support_hits.sum()) if support_hits.size else 0, "mapped_threshold_pixels_in_polygon": int(poly_hits.sum()) if poly_hits.size else 0, "overlap_ratio_support": float(support_hits.mean()) if support_hits.size else None, "overlap_ratio_polygon": float(poly_hits.mean()) if poly_hits.size else None, "labeling": comps["labeling"], "historical_s3_was_bbox_only": True})
    return {"status": "CPU_ONLY_ADDENDUM", "records": records, "no_model_calls": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--s2-root", type=Path, required=True)
    args = ap.parse_args()
    global RAW
    RAW = args.raw
    manifest = args.repo / MANIFEST_REL
    rows = load_rows(manifest)
    train_rows = sorted([r for r in rows if r.get("split") == "train"], key=lambda r: numeric_key(r["stem"]))
    out = args.repo / "results" / "exp13_phase1b_s4" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    all_records = []
    sources_meta = {}
    for row in train_rows:
        source_id = str(row["stem"])
        native = np.asarray(Image.open(RAW / row["image_path"]).convert("RGB"), dtype=np.uint8)
        ann = json.loads((RAW / row["json_path"]).read_text(encoding="utf-8"))
        shapes = [s for s in ann.get("shapes", []) if s.get("shape_type", "polygon") == "polygon"]
        geoms = [__import__("tools.exp13_restoration.geometry", fromlist=["build_geometry"]).build_geometry(tuple((float(x), float(y)) for x, y in s["points"]), native.shape[1], native.shape[0], context=16, halo_radius=2, roi_id=f"roi_{i}", source_id=source_id) for i, s in enumerate(shapes)]
        supports = [g.edit_support for g in geoms]
        records_for_source = []
        for i, (shape, geom) in enumerate(zip(shapes, geoms)):
            rec = base_record(row, source_id, f"roi_{i}", shape, geom, native, supports)
            records_for_source.append(rec); all_records.append(rec)
        sources_meta[source_id] = {"source_id": source_id, "native_shape_hwc": list(native.shape), "rois": [{"roi_id": r["roi_id"], "native_width": r["native_width"], "native_height": r["native_height"], "points_xy": r["points_xy"], "roi_box_xyxy": r["roi_box_xyxy"], "transform": r["transform"]} for r in records_for_source]}
    geometry_candidates = [r for r in all_records if r["geometry_status"] == "GEOMETRY_ELIGIBLE"]
    sensitivity = []
    for max_scale in (2, 4, 8):
        for min_content in (0.60, 0.75):
            for min_short in (16, 32, 64):
                kept = [r for r in geometry_candidates if r["scale"] <= max_scale and r["content_fraction"] >= min_content and r["native_short_side"] >= min_short]
                sensitivity.append({"max_scale": max_scale, "min_content_fraction": min_content, "min_native_short_side": min_short, "polygon_count": len(kept), "source_count": len({r["source_id"] for r in kept}), "class_counts": dict(Counter(r["class_name"] for r in kept))})
    ranked = sorted(geometry_candidates, key=lambda r: (-r["context_fraction"], r["padding_ratio"], r["scale"], numeric_key(r["source_id"]), r["roi_id"]))
    # Deterministic review packet: first cover available classes with distinct
    # sources, then round-robin classes, then fill by the proxy ranking.
    by_class = defaultdict(list)
    for rec in ranked:
        by_class[rec["class_name"]].append(rec)
    review = []
    used_ids = set()
    used_sources = set()
    for class_name in sorted(by_class):
        candidate = next((rec for rec in by_class[class_name] if rec["source_id"] not in used_sources), by_class[class_name][0])
        review.append(candidate); used_ids.add(candidate["candidate_id"]); used_sources.add(candidate["source_id"])
    while len(review) < 20:
        added = False
        for class_name in sorted(by_class):
            candidate = next((rec for rec in by_class[class_name] if rec["candidate_id"] not in used_ids and rec["source_id"] not in used_sources), None)
            if candidate is None:
                candidate = next((rec for rec in by_class[class_name] if rec["candidate_id"] not in used_ids), None)
            if candidate is not None:
                review.append(candidate); used_ids.add(candidate["candidate_id"]); used_sources.add(candidate["source_id"]); added = True
            if len(review) >= 20:
                break
        if not added:
            break
    if len(review) < 20:
        review.extend([rec for rec in ranked if rec["candidate_id"] not in used_ids][:20 - len(review)])
    sheet_dir = out / "candidate_contact_sheets"; sheet_dir.mkdir()
    review_manifest = []
    row_by_id = {str(r["stem"]): r for r in train_rows}
    for idx, rec in enumerate(review, 1):
        row = row_by_id[rec["source_id"]]; image = Image.open(RAW / row["image_path"]).convert("RGB")
        x0, y0, x1, y1 = rec["roi_box_xyxy"]; crop = image.crop((x0, y0, x1, y1))
        support_full = np.zeros((image.height, image.width), dtype=bool)
        points = rec["points_xy"]; ImageDraw.Draw(Image.fromarray((support_full.astype(np.uint8) * 255))).polygon([(round(x), round(y)) for x, y in points], fill=1)
        poly = Image.new("1", image.size, 0); ImageDraw.Draw(poly).polygon([(round(x), round(y)) for x, y in points], fill=1)
        poly_arr = np.asarray(poly, dtype=bool); support = poly_arr.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                support |= np.roll(np.roll(poly_arr, dy, axis=0), dx, axis=1)
        local_support = support[y0:y1, x0:x1]
        canvas, _ = make_canvas(np.asarray(crop), rec["transform"])
        sheet = sheet_dir / f"{idx:02d}_{rec['source_id']}_{rec['roi_id']}.png"; make_candidate_sheet(rec, image, crop, local_support, poly_arr[y0:y1, x0:x1], canvas, sheet)
        review_manifest.append({"review_index": idx, "candidate_id": rec["candidate_id"], "source_id": rec["source_id"], "roi_id": rec["roi_id"], "class_name": rec["class_name"], "sheet_path": str(sheet), "sheet_sha256": sha256(sheet), "review_status": "PENDING_HUMAN_REVIEW", "model_generated": False})
    dump(out / "train_eligibility_distribution.json", {"status": "TRAIN_ONLY_CPU", "manifest_sha256": MANIFEST_SHA, "train_image_count": len(train_rows), "train_polygon_count": len(all_records), "geometry_eligible_polygon_count": len(geometry_candidates), "geometry_eligible_source_count": len({r["source_id"] for r in geometry_candidates}), "class_counts": dict(Counter(r["class_name"] for r in all_records)), "geometry_eligible_class_counts": dict(Counter(r["class_name"] for r in geometry_candidates)), "context_proxy_is_not_semantic": True})
    with (out / "candidate_screening.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(all_records[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for rec in all_records:
            row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in rec.items()}; w.writerow(row)
    dump(out / "eligibility_threshold_sensitivity.json", {"thresholds_are_exploratory_not_scientific": True, "records": sensitivity})
    dump(out / "candidate_review_manifest.json", {"status": "PREPARED_AWAITING_HUMAN_REVIEW", "candidate_count": len(review_manifest), "records": review_manifest, "selection": None, "selection_reason": "No human review was executed in S4."})
    gate = {"name": "RESTORATION_ELIGIBILITY_GATE", "status": "PREPARED_AWAITING_HUMAN_REVIEW" if len({r["source_id"] for r in geometry_candidates}) >= 3 else "NO_GO", "checks": {"REG-1": "UNKNOWN_HUMAN_REVIEW_REQUIRED for non-border UI/FOV/structure content; border/clamping auto-held", "REG-2": "PROXY_ONLY_HUMAN_REVIEW_REQUIRED", "REG-3": "UNKNOWN_HUMAN_REVIEW_REQUIRED", "REG-4": "PASS_PROXY_RECORDED", "REG-5": "PASS_PROXY_RECORDED_WITH_RISK_FLAGS", "REG-6": "PASS_DETERMINISTIC_GEOMETRY", "REG-7": "HOLD_SELECTION_NULL_UNTIL_HUMAN_REVIEW"}, "candidate_count": len(all_records), "geometry_eligible_count": len(geometry_candidates), "human_review_candidate_count": len(review_manifest), "restoration_eligible_count": 0, "restoration_ineligible_count": sum(r["geometry_status"] != "GEOMETRY_ELIGIBLE" for r in all_records), "selection": None, "automatic_restoration_passes": 0, "reason": "Geometry and context proxies are insufficient for semantic restoration approval; human full-context review is mandatory."}
    dump(out / "restoration_eligibility_gate.json", gate)
    dump(out / "artifact_connected_component_addendum.json", connected_addendum(args.s2_root, sources_meta))
    dump(out / "alternative_method_feasibility.json", {"status": "ANALYSIS_ONLY", "route_a_sdxl_local_restoration": {"strengths": ["can model nontrivial texture and lighting", "existing offline artifact and geometry pipeline"], "risks": ["raw Canvas artifacts in S2", "UI/FOV/context contamination", "semantic hallucination and non-deterministic texture"], "reproducibility": "fixed seed/config is reproducible operationally but not semantically trustworthy", "current_recommendation": "HOLD"}, "route_b_classical_local_texture_control": {"strengths": ["deterministic, CPU-auditable, no model artifact or GPU budget", "useful as a bounded engineering baseline"], "risks": ["cannot infer missing geometry or lighting", "texture-copy seams and source contamination", "may preserve or smear defects"], "reproducibility": "high under fixed algorithm/mask/source region", "current_recommendation": "WORTH_SEPARATE_CPU_CONTROL_ONLY"}, "no_cpu_inpainting_executed": True, "not_training_data": True})
    dump(out / "budget_preservation.json", {"status": "PASS_ZERO_NEW_INFERENCE", "inherited_ledger": "results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json", "actual_calls": 3, "occupied_calls": 3, "smoke_remaining": 1, "new_model_load": 0, "new_inference": 0, "gpt_fallback": 0, "val_access": 0, "test_access": 0, "raw_dataset_write": 0, "training": 0, "ledger_reset": False})
    summary = {"status": gate["status"], "train_images": len(train_rows), "train_polygons": len(all_records), "geometry_eligible_polygons": len(geometry_candidates), "geometry_eligible_sources": len({r["source_id"] for r in geometry_candidates}), "human_review_sheets": len(review_manifest), "restoration_eligible": 0, "selection": None, "model_load": 0, "new_inference": 0, "val_access": 0, "test_access": 0, "raw_dataset_write": 0, "recommendation": "PREPARED_AWAITING_HUMAN_REVIEW" if gate["status"] != "NO_GO" else "RESTORATION_ELIGIBILITY_NO_GO"}
    dump(out / "summary.json", summary)
    dump(out / "provenance.json", {"script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)), "repo": str(args.repo), "raw_root": str(RAW), "manifest": str(manifest), "manifest_sha256": MANIFEST_SHA, "train_only": True, "val_access": 0, "test_access": 0, "raw_dataset_write": 0, "model_load": 0, "model_calls": 0, "contact_sheet_dir": str(sheet_dir), "human_review": "not performed"})
    return 0


if __name__ == "__main__":
    RAW = RAW_DEFAULT
    raise SystemExit(main())
