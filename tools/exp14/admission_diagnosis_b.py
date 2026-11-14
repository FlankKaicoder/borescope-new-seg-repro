"""Review fixed B pair evidence at the region scale. No transforms are fitted."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.exp13_cloud.g3.g3_batch_executor import digest, read_csv, require, write_exclusive, write_json_exclusive
from tools.exp14.prepare_large_scale_repair import hash_file, load_json
from tools.exp14.targeted_v2 import target_raster

CLASSES = ("Burn", "Crack", "Dent", "Material missing", "Tears", "corrosion")
PILOT_STATUS = "COUNTERFACTUAL_G1_PILOT_PENDING_CHATGPT_GATE"
TOLERANCE_ROI_DIAGONAL = 0.10
TOLERANCE_TARGET_SHORT_SIDE = 0.50


def region_metrics(mask: np.ndarray, valid: np.ndarray) -> dict:
    """Use the source mask and the saved valid mask in their existing grid."""
    require(mask.shape == valid.shape, "Source mask and saved valid mask differ in shape")
    ys, xs = np.nonzero(mask)
    require(len(xs) > 0, "Empty source Polygon raster")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    h, w = mask.shape
    width, height = right - left, bottom - top
    min_pad = max(12, round(0.055 * min(w, h)))
    xpad, ypad = max(min_pad, math.ceil(width * 0.45)), max(min_pad, math.ceil(height * 0.45))
    x0, y0, x1, y1 = max(0, left - xpad), max(0, top - ypad), min(w, right + xpad), min(h, bottom + ypad)
    roi_shape = y1 - y0, x1 - x0
    m, v = mask[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    context = ~m
    context_count = int(context.sum())
    available_context = int((context & v).sum())
    require(context_count > 0, "ROI lacks Polygon surroundings")
    halo = np.concatenate([v[0], v[-1], v[1:-1, 0], v[1:-1, -1]])
    sides = [left - x0, top - y0, x1 - right, y1 - bottom]
    return {
        "roi": [x0, y0, x1, y1],
        "target_bbox": [left, top, right, bottom],
        "saved_frame_size": [w, h],
        "target_pixels": int(m.sum()),
        "unsupported_target_pixels": int((m & ~v).sum()),
        "roi_pixels": v.size,
        "valid_roi_pixels": int(v.sum()),
        "valid_roi_fraction": round(float(v.mean()), 6),
        "context_pixels": context_count,
        "valid_context_pixels": available_context,
        "valid_context_fraction": round(available_context / context_count, 6),
        "valid_perimeter_fraction": round(float(halo.mean()), 6),
        "context_margin_pixels_ltrb": sides,
        "frame_edge_touch": x0 == 0 or y0 == 0 or x1 == w or y1 == h,
        "roi_short_side": min(roi_shape),
        "roi_diagonal": round(math.hypot(*roi_shape), 3),
        "target_short_side": min(width, height),
        "target_diagonal": round(math.hypot(width, height), 3),
    }


def eligible_pixel_support(metrics):
    return (metrics["unsupported_target_pixels"] == 0
            and metrics["valid_roi_pixels"] * 100 >= metrics["roi_pixels"] * 95
            and metrics["valid_context_pixels"] * 100 >= metrics["context_pixels"] * 90)


def pick_cases(rows):
    """Six per class, unique across source and near-duplicate group."""
    seen_sources, seen_groups, selected = set(), set(), []
    # Select rare classes first. Each list is frozen by target scale then Polygon ID.
    for klass in sorted(CLASSES, key=lambda name: sum(row["class"] == name for row in rows)):
        group = [r for r in rows if r["class"] == klass]
        group.sort(key=lambda r: (r["target_scale"], r["polygon_id"]))
        for slot in range(6):
            lo, hi = len(group) * slot // 6, len(group) * (slot + 1) // 6
            middle = (lo + hi - 1) / 2
            available = [i for i in range(len(group)) if group[i]["source_id"] not in seen_sources
                         and group[i]["group_id"] not in seen_groups]
            require(bool(available), f"Unable to draw {klass} with distinct sources and groups")
            index = min(available, key=lambda i: (not lo <= i < hi, abs(i - middle), group[i]["polygon_id"]))
            row = group[index]
            selected.append({**row, "size_stratum": slot + 1, "size_bucket_fallback": not lo <= index < hi})
            seen_sources.add(row["source_id"])
            seen_groups.add(row["group_id"])
    return sorted(selected, key=lambda r: (CLASSES.index(r["class"]), r["size_stratum"]))


def read_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def write_png(dest, image):
    data = io.BytesIO()
    image.save(data, "PNG")
    png = data.getvalue()
    if dest.exists():
        require(hash_file(dest) == digest(png), f"Existing evidence differs at {dest}")
    else:
        write_exclusive(dest, png)
    return digest(png)


def case_assets(row, output_dir, image_number, source, asset_index):
    chosen = row.pop("chosen")
    coords = json.loads(row["polygon_coordinates"])
    yolo = load_json(Path(row["annotation_path"]))
    shape_index = int(row["polygon_id"].rsplit("_", 1)[1])
    require(yolo["shapes"][shape_index]["points"] == coords
            and yolo["shapes"][shape_index]["label"] == row["class"], "Professional Polygon differs")
    original = read_rgb(row["original_path"])
    aligned = read_rgb(chosen["aligned"]["path"])
    saved_mask = read_rgb(chosen["valid_mask"]["path"])
    require(original.size == aligned.size == saved_mask.size, "Source-grid images differ in shape")
    valid = np.asarray(saved_mask)[:, :, 0] > 0
    asset_dir = Path(chosen["aligned"]["path"]).parent
    geometry_path = asset_dir / "geometry.json"
    expected_matrix = chosen["matrix_output_to_source"]
    if not chosen["geometry"]:
        # Some older rows carry the matrix but omit this link. Recover the
        # existing identity by requiring the original record and A9 asset index.
        file_digest = asset_index.get(geometry_path.as_posix())
        historical_record = load_json(asset_dir / "record.json")
        require(historical_record["review_id"] == chosen["review_id"] and
                historical_record["aligned_sha256"] == chosen["aligned"]["sha256"], "Legacy geometry path is unbound")
        require(historical_record["file_sha256"].get("geometry.json") == file_digest, "Legacy evidence hash registry differs")
        chosen["geometry"] = {"path": geometry_path.as_posix(), "sha256": file_digest or hash_file(geometry_path)}
        geometry_source = "RECORD_DIRECTORY_AND_A9_PINNED_MATRIX" if not file_digest else "A9_ASSET_INDEX_AND_REVIEW_RECORD"
    else:
        geometry_source = "A9_PAIR_MANIFEST"
    a9 = load_json(Path(chosen["geometry"]["path"]))
    matrix = np.asarray(expected_matrix, dtype=float)
    require(matrix.shape == (2, 3) and np.isfinite(matrix).all(), "Missing fixed historical matrix")
    fits = [a9.get("selected_fit") or {}, *a9.get("candidate_diagnostics", [])]
    require(any(fit.get("matrix_output_to_source") == expected_matrix for fit in fits), "Saved matrix differs from pinned manifest")
    source["size"] = list(original.size)
    source["all_polygon_ids"] = json.loads(source["all_gt_polygon_ids"])
    source["polygon_labels"] = {r: yolo["shapes"][i]["label"] for i, r in enumerate(source["all_polygon_ids"])}
    raster, _ = target_raster(source, yolo["shapes"], [row["polygon_id"]])
    mask = np.asarray(raster, dtype=bool)
    metrics = region_metrics(mask, valid)
    require(metrics["target_pixels"] == chosen["support"]["pixels"] and
            metrics["unsupported_target_pixels"] == chosen["support"]["unsupported_pixels"] == 0,
            "Recomputed target support differs from pinned manifest")
    for path, expected_hash in [(row["original_path"], row["original_sha256"]),
                                (row["annotation_path"], row["annotation_sha256"]),
                                *((chosen[k]["path"], chosen[k]["sha256"]) for k in ("aligned", "valid_mask", "geometry"))]:
        hash_file(Path(path), expected_hash)

    key = f"{image_number:02d}_{row['source_id']}_{row['polygon_id'].rsplit('_', 1)[1]}"
    orig_layer = original.copy()
    align_layer = aligned.copy()
    poly = [(int(x), int(y)) for x, y in coords]
    for image in (orig_layer, align_layer):
        draw = ImageDraw.Draw(image)
        draw.line([*poly, poly[0]], fill="#ffbd56", width=max(1, round(original.width / 530)))
        draw.rectangle(metrics["roi"], outline="#52d6ff", width=max(1, round(original.width / 530)))
    x0, y0, _, _ = metrics["roi"]
    crops = {}
    evidence = {
        "original_roi": original.crop(metrics["roi"]), "aligned_roi": aligned.crop(metrics["roi"]),
        "original_context": orig_layer, "aligned_context": align_layer,
        "valid_roi": saved_mask.crop(metrics["roi"]),
        "polygon_roi": original.crop(metrics["roi"]),
    }
    local_poly = [(x - x0, y - y0) for x, y in poly]
    ImageDraw.Draw(evidence["polygon_roi"]).line([*local_poly, local_poly[0]], fill="#ffbd56", width=2)
    for kind, image in evidence.items():
        dest = output_dir / f"{key}_{kind}.png"
        crops[kind] = {"path": dest.name, "sha256": write_png(dest, image)}
    return {**{k: row[k] for k in ("source_id", "polygon_id", "class", "group_id", "original_path",
                                  "original_sha256", "annotation_path", "annotation_sha256",
                                  "size_stratum", "size_bucket_fallback") if k in row},
            "case_id": key, "variant_id": chosen["variant_id"], "review_id": chosen["review_id"],
            "version": chosen["version"], "aligned_path": chosen["aligned"]["path"],
            "aligned_sha256": chosen["aligned"]["sha256"], "valid_mask_path": chosen["valid_mask"]["path"],
            "valid_mask_sha256": chosen["valid_mask"]["sha256"], "geometry_path": chosen["geometry"]["path"],
            "geometry_sha256": chosen["geometry"]["sha256"], "geometry_link_provenance": geometry_source,
            "pixel_asset_state": chosen["pixel_asset_state"],
            "risk_evidence": chosen["risk_evidence"], "polygon_state_basis": chosen["polygon_state_basis"],
            "sampling_target_scale": row["target_scale"], "polygon_coordinates": coords,
            "metrics": metrics, "evidence": crops, "g0": chosen["semantic_status"], "g2": "FORMAL_TRE_NOT_ESTABLISHED"}


def letterbox(canvas, image, x, y, width, height):
    thumb = ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)
    canvas.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))


def evidence_sheet(cases, folder, start):
    width, row_h = 1390, 254
    sheet = Image.new("RGB", (width, row_h * len(cases)), "#14202a")
    font = ImageFont.load_default(size=15)
    draw = ImageDraw.Draw(sheet)
    columns = (("Original + ROI", "original_context", 215), ("Aligned + ROI", "aligned_context", 215),
               ("Original ROI", "original_roi", 315), ("Aligned ROI", "aligned_roi", 315),
               ("Target", "polygon_roi", 145), ("Valid", "valid_roi", 145))
    for i, item in enumerate(cases):
        m = item["metrics"]
        y = i * row_h
        draw.text((5, y + 4), f"{item['case_id']}     {item['class']}   {item['variant_id']}    "
                              f"ROI {m['roi']}   valid={m['valid_roi_fraction']:.3f}   "
                              f"context={m['valid_context_fraction']:.3f}   target={m['target_pixels']} px",
                  font=font, fill="#f0f8ff")
        x = 5
        for label, key, w in columns:
            draw.text((x + 4, y + 25), label, font=font, fill="#b9d9ed")
            image = read_rgb(folder / item["evidence"][key]["path"])
            letterbox(sheet, image, x, y + 47, w - 5, 196)
            x += w
    path = folder / f"sheet_{start:02d}.png"
    write_png(path, sheet)


def prepare(a9, out):
    receipt = load_json(a9 / "handoff" / "delivery_receipt.json")
    paths = ["registry/b_pair_candidate_manifest.csv", "registry/source_master_registry.csv",
             "audits/alignment_support_audit.json", "registry/final_candidate_summary.json",
             "registry/candidate_asset_sha256.json"]
    for relative in paths:
        hash_file(a9 / relative, receipt["private_sha256"][relative])
    rows = read_csv(a9 / paths[0])[1]
    sources = {r["source_id"]: r for r in read_csv(a9 / paths[1])[1]}
    asset_index = load_json(a9 / paths[-1])["files"]
    require(len(rows) == len({r["polygon_id"] for r in rows}) == 545, "Wrong B cohort")
    for row in rows:
        chosen = json.loads(row["recommended"])
        require(chosen["semantic_status"] == "HUMAN_SEMANTIC_CANDIDATE" and
                chosen["geometry_status"] == "PAIR_GEOMETRY_PENDING_GATE", "Unexpected parent candidate status")
        source = sources[row["source_id"]]
        require("/images/train/" in Path(row["original_path"]).as_posix().lower()
                and "/annotations/train/" in Path(row["annotation_path"]).as_posix().lower(),
                "B pilot accepts TRAIN only")
        box = chosen["support"]["crop_box"]
        row.update(chosen=chosen, group_id=source["group_id"],
                   target_scale=round(chosen["support"]["pixels"] / ((box[2] - box[0]) * (box[3] - box[1])), 6))
    folder = out / "evidence" / "b_g1"
    folder.mkdir(parents=True, exist_ok=True)
    selected = pick_cases(rows)
    cases = [case_assets(row, folder, i, dict(sources[row["source_id"]]), asset_index)
             for i, row in enumerate(selected, 1)]
    for i in range(0, len(cases), 4):
        evidence_sheet(cases[i:i + 4], folder, i + 1)
    write_json_exclusive(folder / "prepared.json", {"cases": cases, "b_population": len(rows)})


def population_screen(a9, out):
    receipt = load_json(a9 / "handoff" / "delivery_receipt.json")
    b_path = "registry/b_pair_candidate_manifest.csv"
    source_path = "registry/source_master_registry.csv"
    for relative in (b_path, source_path):
        hash_file(a9 / relative, receipt["private_sha256"][relative])
    sources = {r["source_id"]: r for r in read_csv(a9 / source_path)[1]}
    rows = read_csv(a9 / b_path)[1]
    require(len(rows) == len({r["polygon_id"] for r in rows}) == 545, "Wrong B population")
    cases = []
    summary = defaultdict(Counter)
    mask_cache, annotation_cache = {}, {}
    for row in rows:
        chosen = json.loads(row["recommended"])
        source = {**sources[row["source_id"]]}
        source["all_polygon_ids"] = json.loads(source["all_gt_polygon_ids"])
        annotation_path = Path(row["annotation_path"])
        require("/annotations/train/" in annotation_path.as_posix().lower(), "B population accepts TRAIN only")
        if annotation_path not in annotation_cache:
            hash_file(annotation_path, row["annotation_sha256"])
            annotation_cache[annotation_path] = load_json(annotation_path)["shapes"]
        labels = annotation_cache[annotation_path]
        path = Path(chosen["valid_mask"]["path"])
        if path not in mask_cache:
            hash_file(path, chosen["valid_mask"]["sha256"])
            with Image.open(path) as saved:
                mask_cache[path] = np.asarray(saved.convert("L")) > 0
        valid = mask_cache[path]
        source["size"] = [valid.shape[1], valid.shape[0]]
        raster, _ = target_raster(source, labels, [row["polygon_id"]])
        metrics = region_metrics(np.asarray(raster, dtype=bool), valid)
        supported = eligible_pixel_support(metrics)
        cls = row["class"]
        for bucket in (summary["all"], summary[cls]):
            bucket["polygons"] += 1
            bucket["complete_target"] += metrics["unsupported_target_pixels"] == 0
            bucket["complete_roi"] += metrics["valid_roi_fraction"] == 1.0
            bucket["numeric_screen_pass"] += supported
            bucket["frame_edge_roi"] += metrics["frame_edge_touch"]
            bucket["geometry_link_omitted_in_a9"] += not chosen["geometry"]
        cases.append({"source_id": row["source_id"], "polygon_id": row["polygon_id"], "review_id": chosen["review_id"],
                      "variant_id": chosen["variant_id"], "class": cls, "valid_mask_sha256": chosen["valid_mask"]["sha256"],
                      "numeric_screen_pass": supported, **metrics})
    write_json_exclusive(out / "b_roi_integrity_population.json", {"status": PILOT_STATUS, "summary": dict(summary),
        "applicability": "EXISTING_SUPPORT_MASK_INTEGRITY_ONLY", "visually_adjudicated_population": False,
        "min_roi_valid_fraction": min(c["valid_roi_fraction"] for c in cases),
        "min_context_valid_fraction": min(c["valid_context_fraction"] for c in cases), "cases": cases})


def assess(case, observation, folder):
    same_part, target, context, structure, shortcut, note, stations = observation
    metrics = case["metrics"]
    width = metrics["roi"][2] - metrics["roi"][0]
    height = metrics["roi"][3] - metrics["roi"][1]
    tolerance = min(TOLERANCE_ROI_DIAGONAL * metrics["roi_diagonal"],
                    TOLERANCE_TARGET_SHORT_SIDE * metrics["target_short_side"])
    quantization_bound = math.hypot(width, height) * 0.03
    require(all(set(sample) == {"kind", "station_axis", "station_percent", "original_percent", "aligned_percent",
                                "sheet", "feature"} for sample in stations), "Invalid visual station record")
    require(len({(s["station_axis"], s["station_percent"], s["original_percent"]) for s in stations}) == len(stations),
            "Duplicate visual station coordinates")
    stations = [sample.copy() for sample in stations]
    observed_offsets = []
    for sample in stations:
        require(sample["station_axis"] in {"x", "y"} and all(
            0 <= sample[key] <= 100 for key in ("station_percent", "original_percent", "aligned_percent")),
            "Visual station outside ROI")
        axis_extent = height if sample["station_axis"] == "x" else width
        delta = abs(sample["original_percent"] - sample["aligned_percent"]) / 100 * axis_extent
        observed_offsets.append(delta)
        sample["observed_boundary_displacement_px"] = round(delta, 3)
        sample["conservative_upper_px"] = round(delta + quantization_bound, 3)
        sample["station_evidence_sha256"] = hash_file(folder / sample["sheet"])
    numeric = eligible_pixel_support(metrics)
    observed = max(observed_offsets, default=None)
    upper = observed + quantization_bound if observed is not None else None
    failed = not numeric or "NO" in (same_part, target, context) or structure == "ALTERED" or shortcut == "MATERIAL"
    if failed or (observed is not None and observed > tolerance):
        decision = "G1_PILOT_FAIL"
    elif (same_part, target, context, structure, shortcut) == ("YES", "YES", "YES", "CLEAR", "CLEAR") and len(stations) >= 2 and upper <= tolerance:
        decision = "G1_PILOT_FEASIBLE_PENDING_INDEPENDENT_REVIEW"
    elif "UNCERTAIN" in (same_part, target, context, structure, shortcut):
        decision = "G1_HOLD_VISUAL_AMBIGUITY"
    else:
        decision = "G1_HOLD_INDEPENDENT_BOUNDARY_EVIDENCE"
    return {**case, "status": PILOT_STATUS, "g1_result": decision, "numeric_integrity_screen": numeric,
            "same_mechanical_part": same_part, "target_inside_and_semantically_restored": target,
            "local_context": context, "structure": structure, "shortcut_risk": shortcut,
            "visual_note": note, "boundary_stations": stations,
            "estimated_upper_displacement_px": None if upper is None else round(upper, 3),
            "observed_max_displacement_px": None if observed is None else round(observed, 3),
            "g1_displacement_limit_px": round(tolerance, 3),
            "displacement_limit_roi_diagonal_fraction": round(tolerance / metrics["roi_diagonal"], 6),
            "displacement_limit_target_short_side_fraction": round(tolerance / metrics["target_short_side"], 6),
            "coarse_geometry_qualified_formally": False, "training_approved": False}


def publish(out, observations):
    folder = out / "evidence" / "b_g1"
    prepared = load_json(folder / "prepared.json")
    require({r["case_id"] for r in prepared["cases"]} == set(observations), "An image lacks a visual assessment")
    cases = [assess(r, observations[r["case_id"]], folder) for r in prepared["cases"]]
    fields = ["case_id", "source_id", "polygon_id", "class", "group_id", "version", "variant_id", "review_id",
              "sampling_target_scale", "size_stratum", "size_bucket_fallback", "status", "g0", "g1_result", "g2",
              "same_mechanical_part", "target_inside_and_semantically_restored", "local_context", "structure", "shortcut_risk",
              "numeric_integrity_screen", "observed_max_displacement_px", "estimated_upper_displacement_px",
              "g1_displacement_limit_px", "displacement_limit_roi_diagonal_fraction", "displacement_limit_target_short_side_fraction",
              "visual_note", "metrics", "boundary_stations", "evidence", "polygon_coordinates", "original_path", "original_sha256",
              "aligned_path", "aligned_sha256", "valid_mask_path", "valid_mask_sha256", "annotation_path", "annotation_sha256",
              "geometry_path", "geometry_sha256", "geometry_link_provenance", "pixel_asset_state",
              "coarse_geometry_qualified_formally", "training_approved"]
    with (out / "b_g1_pilot_manifest.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, bool)) else v
                             for k, v in case.items()})
    counts = Counter(r["g1_result"] for r in cases)
    by_class = {c: dict(Counter(r["g1_result"] for r in cases if r["class"] == c)) for c in CLASSES}
    summary = {"status": PILOT_STATUS, "b_population": prepared["b_population"], "sample_count": len(cases),
               "distinct_source_count": len({r["source_id"] for r in cases}), "by_class": by_class, "g1_result": dict(counts),
               "numeric_integrity_screen_pass": sum(r["numeric_integrity_screen"] for r in cases),
               "with_independent_visual_stations": sum(bool(r["boundary_stations"]) for r in cases),
               "geometry_link_provenance": dict(Counter(r["geometry_link_provenance"] for r in cases)),
               "formal_trusted_pair_count": 0, "new_alignment": 0, "new_api": 0, "new_training": 0}
    write_json_exclusive(folder / "cases.json", {"summary": summary, "cases": cases})
    write_json_exclusive(out / "b_g1_pilot_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "population"))
    parser.add_argument("--a9", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    if args.action in {"prepare", "population"}:
        require(args.a9 is not None, "Supply A9 for the read-only geometry action")
        a9, out = args.a9.resolve(), args.out.resolve()
        require(not out.is_relative_to(a9), "Choose an output directory outside the frozen A9 tree")
        run = prepare if args.action == "prepare" else population_screen
        run(a9, out)
    else:
        require(args.observations is not None, "Supply visual observations for the publish action")
        publish(args.out.resolve(), load_json(args.observations.resolve()))


if __name__ == "__main__":
    main()
