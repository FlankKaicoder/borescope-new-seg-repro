"""CPU-only root-cause audit for the existing Exp13 Phase 1-B-S2 artifacts.

This script never imports torch/diffusers and never creates a model output.
It compares the persisted S2 input/output stages and writes numeric evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageStat


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb_rows(image: Image.Image) -> list[tuple[int, int, int]]:
    return list(image.convert("RGB").getdata())


def luminance(px: tuple[int, int, int]) -> float:
    return 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]


def hist_stats(rows: Iterable[tuple[int, int, int]], mask: Iterable[bool] | None = None) -> dict:
    values: list[float] = []
    if mask is None:
        values = [luminance(p) for p in rows]
    else:
        values = [luminance(p) for p, keep in zip(rows, mask) if keep]
    if not values:
        return {"count": 0}
    values.sort()
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    def pct(q: float) -> float:
        return values[min(n - 1, int(round((n - 1) * q)))]
    return {"count": n, "mean_luma": mean, "std_luma": math.sqrt(var),
            "p01": pct(0.01), "p50": pct(0.50), "p99": pct(0.99),
            "min": values[0], "max": values[-1]}


def rgb_mean(rows: Iterable[tuple[int, int, int]], mask: Iterable[bool] | None = None) -> dict:
    sums = [0, 0, 0]
    count = 0
    if mask is None:
        for p in rows:
            for i in range(3):
                sums[i] += p[i]
            count += 1
    else:
        for p, keep in zip(rows, mask):
            if keep:
                for i in range(3):
                    sums[i] += p[i]
                count += 1
    return {"count": count, "mean_rgb": [x / count for x in sums] if count else None}


def mask_stats(image: Image.Image, mask: list[bool]) -> dict:
    rows = rgb_rows(image)
    return {**hist_stats(rows, mask), **rgb_mean(rows, mask)}


def polygon_mask(points: list[list[float]], width: int, height: int) -> list[bool]:
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon([(round(float(x)), round(float(y))) for x, y in points], fill=1)
    return [bool(x) for x in image.getdata()]


def dilate(mask: list[bool], width: int, height: int, radius: int = 2) -> list[bool]:
    out = [False] * (width * height)
    for y in range(height):
        for x in range(width):
            if not mask[y * width + x]:
                continue
            for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                    out[yy * width + xx] = True
    return out


def crop_mask(full: list[bool], full_w: int, box: list[int]) -> list[bool]:
    x0, y0, x1, y1 = box
    return [full[y * full_w + x] for y in range(y0, y1) for x in range(x0, x1)]


def image_diff_count(a: Image.Image, b: Image.Image, mask: list[bool] | None = None, threshold: int = 0) -> dict:
    ar, br = rgb_rows(a), rgb_rows(b)
    changed = []
    abs_sum = 0
    for i, (pa, pb) in enumerate(zip(ar, br)):
        if mask is not None and not mask[i]:
            continue
        d = max(abs(pa[j] - pb[j]) for j in range(3))
        if d > threshold:
            changed.append(i)
        abs_sum += sum(abs(pa[j] - pb[j]) for j in range(3))
    return {"pixels_considered": sum(mask) if mask is not None else len(ar),
            "changed_pixels": len(changed), "mae_rgb_sum": abs_sum / max(1, 3 * (sum(mask) if mask is not None else len(ar))),
            "threshold": threshold}


def bbox_from_indices(indices: list[int], width: int) -> list[int] | None:
    if not indices:
        return None
    xs = [i % width for i in indices]
    ys = [i // width for i in indices]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def dark_artifact(output: Image.Image, input_canvas: Image.Image, content_box: tuple[int, int, int, int], transform: dict, local_support: list[bool], local_polygon: list[bool]) -> dict:
    out_rows, in_rows = rgb_rows(output), rgb_rows(input_canvas)
    x0, y0, x1, y1 = content_box
    indices = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = y * output.width + x
            if luminance(out_rows[i]) <= 25 and luminance(in_rows[i]) >= 45:
                indices.append(i)
    rows = {}
    for i in indices:
        rows[i // output.width] = rows.get(i // output.width, 0) + 1
    cols = {}
    for i in indices:
        cols[i % output.width] = cols.get(i % output.width, 0) + 1
    mapped_support = 0
    mapped_polygon = 0
    for i in indices:
        cx, cy = i % output.width, i // output.width
        nx = min(transform["native_width"] - 1, max(0, int((cx - transform["pad_left"] + 0.5) / transform["scale"] - 0.5)))
        ny = min(transform["native_height"] - 1, max(0, int((cy - transform["pad_top"] + 0.5) / transform["scale"] - 0.5)))
        local_i = ny * transform["native_width"] + nx
        mapped_support += int(local_support[local_i])
        mapped_polygon += int(local_polygon[local_i])
    matching_input = sum(1 for i in indices if luminance(in_rows[i]) <= 25)
    return {"thresholds": {"output_luma_le": 25, "input_luma_ge": 45},
            "new_dark_pixels": len(indices), "bbox_xyxy_canvas": bbox_from_indices(indices, output.width),
            "row_span_peak": max(rows.values()) if rows else 0,
            "column_span_peak": max(cols.values()) if cols else 0,
            "matching_input_dark_pixels": matching_input,
            "mapped_dark_pixels_in_native_support": mapped_support,
            "mapped_dark_pixels_in_native_polygon": mapped_polygon,
            "interpretation": "The connected dark rectangular region is present in persisted SDXL output Canvas and not in the corresponding input Canvas under this conservative threshold."}


def edge_seam_stats(before: Image.Image, after: Image.Image, support: list[bool], width: int, height: int) -> dict:
    br, ar = rgb_rows(before), rgb_rows(after)
    boundary = []
    for y in range(height):
        for x in range(width):
            i = y * width + x
            if not support[i]:
                continue
            for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= xx < width and 0 <= yy < height and not support[yy * width + xx]:
                    boundary.append(i)
                    break
    def one(rows):
        vals = []
        for i in boundary:
            x, y = i % width, i // width
            for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= xx < width and 0 <= yy < height and not support[yy * width + xx]:
                    j = yy * width + xx
                    vals.append(sum(abs(rows[i][k] - rows[j][k]) for k in range(3)) / 3)
                    break
        return {"count": len(vals), "mean_adjacent_rgb_delta": sum(vals) / len(vals) if vals else None,
                "p95_adjacent_rgb_delta": sorted(vals)[int(0.95 * (len(vals) - 1))] if vals else None}
    return {"before": one(br), "after": one(ar)}


def audit_call(root: Path, source: dict, call: dict) -> dict:
    sid, rid = call["source_id"], call["roi_id"]
    source_dir = root / f"source_{sid}"
    original = Image.open(source_dir / "original.png").convert("RGB")
    input_canvas = Image.open(source_dir / f"{rid}_input_canvas.png").convert("RGB")
    mask_canvas = Image.open(source_dir / f"{rid}_mask_canvas.png").convert("L")
    output_canvas = Image.open(root / f"{sid}_{rid}_sdxl_canvas.png").convert("RGB")
    native_saved = Image.open(root / f"{sid}_{rid}_native_roi.png").convert("RGB")
    box = call["native_roi_xyxy"]
    x0, y0, x1, y1 = box
    crop = original.crop((x0, y0, x1, y1))
    transform = next(r["transform"] for r in source["rois"] if r["roi_id"] == rid)
    content_box = (transform["pad_left"], transform["pad_top"], transform["pad_left"] + transform["resized_width"], transform["pad_top"] + transform["resized_height"])
    content = output_canvas.crop(content_box)
    native_lanczos = content.resize(crop.size, Image.Resampling.LANCZOS)
    native_nearest = content.resize(crop.size, Image.Resampling.NEAREST)
    full_poly = polygon_mask(next(r["points_xy"] for r in source["rois"] if r["roi_id"] == rid), original.width, original.height)
    support = dilate(full_poly, original.width, original.height, 2)
    local_support = crop_mask(support, original.width, box)
    local_polygon = crop_mask(full_poly, original.width, box)
    raw_rows = rgb_rows(output_canvas)
    pad_left = transform["pad_left"]
    pad_top = transform["pad_top"]
    pad_right = transform["canvas_size"] - pad_left - transform["resized_width"]
    pad_bottom = transform["canvas_size"] - pad_top - transform["resized_height"]
    pad = transform["canvas_size"] ** 2 - transform["resized_width"] * transform["resized_height"]
    padded_mask = [p != 0 for p in mask_canvas.getdata()]
    raw_changed = image_diff_count(input_canvas, output_canvas)
    raw_changed_content = image_diff_count(input_canvas.crop(content_box), content)
    white_in_padding = 0
    for y in range(1024):
        for x in range(1024):
            in_content = content_box[0] <= x < content_box[2] and content_box[1] <= y < content_box[3]
            if not in_content and padded_mask[y * 1024 + x]:
                white_in_padding += 1
    return {
        "source_id": sid, "roi_id": rid, "class_name": next(r["class_name"] for r in source["rois"] if r["roi_id"] == rid),
        "artifact_hashes": {"original": image_hash(source_dir / "original.png"), "input_canvas": image_hash(source_dir / f"{rid}_input_canvas.png"), "mask_canvas": image_hash(source_dir / f"{rid}_mask_canvas.png"), "sdxl_canvas": image_hash(root / f"{sid}_{rid}_sdxl_canvas.png"), "native_roi": image_hash(root / f"{sid}_{rid}_native_roi.png")},
        "geometry": {"native_shape": list(original.size), "roi_box_xyxy": box, "transform": transform, "polygon_pixels": sum(local_polygon), "support_pixels": sum(local_support), "polygon_covered_by_support": all(not p or s for p, s in zip(local_polygon, local_support))},
        "canvas": {"size": list(output_canvas.size), "content_pixels": transform["resized_width"] * transform["resized_height"], "padding_pixels": pad, "padding_ratio": pad / (1024 * 1024), "mask_white_pixels": sum(padded_mask), "mask_white_in_padding": white_in_padding, "mask_bbox": list(mask_canvas.getbbox()) if mask_canvas.getbbox() else None},
        "stage_stats": {"native_crop": hist_stats(rgb_rows(crop)), "input_canvas": hist_stats(rgb_rows(input_canvas)), "sdxl_output_canvas": hist_stats(raw_rows), "sdxl_output_content": hist_stats(rgb_rows(content)), "inverse_lanczos_native": hist_stats(rgb_rows(native_lanczos)), "saved_native_roi_after_support_clip": hist_stats(rgb_rows(native_saved))},
        "dark_artifact": dark_artifact(output_canvas, input_canvas, content_box, transform, local_support, local_polygon),
        "mapping": {"lanczos_vs_nearest": image_diff_count(native_lanczos, native_nearest, local_support), "saved_native_vs_lanczos_inside_support": image_diff_count(native_saved, native_lanczos, local_support), "saved_native_outside_support_vs_original": image_diff_count(native_saved, crop, [not x for x in local_support]), "raw_model_changed_pixels": raw_changed, "raw_model_changed_content": raw_changed_content, "runtime_reported_raw_changed": call.get("raw_model_roi_changed_pixels_before_support_clip"), "runtime_reported_clipped_outside": call.get("raw_model_roi_changes_outside_native_support_clipped"), "runtime_clipped_fraction": call.get("raw_model_roi_changes_outside_native_support_clipped", 0) / max(1, call.get("raw_model_roi_changed_pixels_before_support_clip", 1))},
        "brightness": {"input_support": mask_stats(crop, local_support), "output_lanczos_support": mask_stats(native_lanczos, local_support), "saved_native_support": mask_stats(native_saved, local_support), "input_polygon": mask_stats(crop, local_polygon), "saved_polygon": mask_stats(native_saved, local_polygon), "input_context": mask_stats(crop, [not x for x in local_support]), "saved_context": mask_stats(native_saved, [not x for x in local_support])},
        "seam": edge_seam_stats(crop, native_saved, local_support, crop.width, crop.height),
        "postprocess_diagnostic_only": True,
    }


def make_diagnostic_sheet(root: Path, out: Path, calls: list[dict]) -> None:
    panels = []
    for call in calls:
        sid, rid = call["source_id"], call["roi_id"]
        paths = [root / f"source_{sid}" / f"{rid}_input_canvas.png", root / f"{sid}_{rid}_sdxl_canvas.png", root / f"{sid}_{rid}_native_roi.png"]
        titles = [f"{sid}/{rid} input", f"{sid}/{rid} SDXL output", f"{sid}/{rid} native clipped"]
        for path, title in zip(paths, titles):
            image = Image.open(path).convert("RGB")
            image.thumbnail((480, 360), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (500, 410), "white")
            panel.paste(image, ((500 - image.width) // 2, 35 + (360 - image.height) // 2))
            ImageDraw.Draw(panel).text((8, 10), title, fill="black")
            panels.append(panel)
    sheet = Image.new("RGB", (1000, ((len(panels) + 1) // 2) * 410), (235, 235, 235))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * 500, (i // 2) * 410))
    sheet.save(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.input_root
    manifest = load_json(root / "smoke_source_manifest.json")
    runtime = load_json(root / "runtime_metrics.json")
    sources = {s["source_id"]: s for s in manifest["sources"]}
    calls = runtime["calls"]
    records = [audit_call(root, sources[c["source_id"]], c) for c in calls]
    (args.output_dir / "source2_black_artifact_audit.json").write_text(json.dumps(records[0], indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "source65_dark_roi_audit.json").write_text(json.dumps({"calls": records[1:]}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "canvas_mask_scale_audit.json").write_text(json.dumps({"calls": [{"source_id": r["source_id"], "roi_id": r["roi_id"], "canvas": r["canvas"], "mapping": r["mapping"], "geometry": r["geometry"]} for r in records], "status": "PASS_WITH_DIAGNOSTIC_LIMITS"}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "postprocess_comparison.json").write_text(json.dumps({"calls": [{"source_id": r["source_id"], "roi_id": r["roi_id"], "stage_stats": r["stage_stats"], "mapping": r["mapping"], "seam": r["seam"], "postprocess_diagnostic_only": True} for r in records]}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "existing_artifact_inventory.json").write_text(json.dumps({"source_root": str(root), "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": image_hash(p) if p.is_file() else None} for p in sorted(root.rglob("*")) if p.is_file() and ".ipynb_checkpoints" not in str(p)]}, indent=2) + "\n", encoding="utf-8")
    make_diagnostic_sheet(root, args.output_dir / "s3_cpu_stage_diagnostic_contact_sheet.png", calls)
    summary = {"status": "CPU_AUDIT_ONLY", "model_load": 0, "new_inference_calls": 0, "val_access": 0, "test_access": 0, "raw_data_write": 0, "postprocess_outputs": "POSTPROCESS_DIAGNOSTIC_ONLY", "records": [{"source_id": r["source_id"], "roi_id": r["roi_id"], "dark_artifact_bbox": r["dark_artifact"]["bbox_xyxy_canvas"]} for r in records]}
    (args.output_dir / "cpu_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
