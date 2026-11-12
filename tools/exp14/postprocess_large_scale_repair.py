"""Register returned Exp14 variants and retain source-grid diagnostics for human review."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from tools.exp13_cloud.g3 import align_g3b2_batch as alignment
from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import mask_for_selection
from tools.exp14.prepare_large_scale_repair import (
    PHASE, PRIVATE_ROOT, canonical_digest, hash_file, load_json, target_mask, verify_mask,
)
from tools.exp14.run_large_scale_repair import frozen_plan, recover_responses


def candidates(root, scope, skip=frozenset(), running=False):
    entries = {variant: entry for variant, entry in scope["historical_variants"].items() if variant not in skip}
    for row in scope["requests"]:
        if row["variant_id"] in skip:
            continue
        folder = root / "live" / row["variant_id"]
        terminal = folder / "TERMINAL.json"
        if not terminal.is_file():
            continue
        try:
            state = load_json(terminal)
        except json.JSONDecodeError:
            if running:
                continue
            raise
        recovered = root / "response_recovery" / row["variant_id"] / "RECOVERED.json"
        if state["status"] != "RETURNED_IMAGE" and not recovered.is_file():
            continue
        lock = root / "locks" / f"{row['request_id']}.json"
        req = folder / "request_manifest.json"
        request = load_json(req)
        g3.require(load_json(lock)["request_sha256"] == canonical_digest(request) and
                   request["request_id"] == row["request_id"], "Returned image request lock mismatch")
        response_locations = [(row["variant_id"], folder, state.get("raw_sha256"), "")]
        if recovered.is_file():
            evidence = load_json(recovered)
            hash_file(terminal, evidence["original_terminal_sha256"])
            g3.require(evidence["request_id"] == row["request_id"], "Recovered response request identity mismatch")
            if evidence.get("outputs") == []:
                continue
            if evidence.get("provider_returned_positions", 1) > 1:
                response_locations = [
                    (f"{row['variant_id']}_OUT{out['position']}", recovered.parent / out["path"],
                     out["raw_sha256"], "PROVIDER_RETURNED_MULTIPLE_IMAGES")
                    for out in evidence["outputs"]
                ]
            else:
                response_locations = [(row["variant_id"], recovered.parent, evidence["raw_sha256"], "")]
        for variant, response_folder, raw_sha, response_hold in response_locations:
            if variant in skip:
                continue
            response = load_json(response_folder / "response_manifest.json")
            raw = g3.safe_file(response_folder, response["raw_filename"])
            entries[variant] = {
                "source_id": row["source_id"], "version": row["version"], "origin": "EXP14", "derived": {},
                "request_path": str(req), "request_sha256": hash_file(req), "lock_path": str(lock),
                "request_id": row["request_id"], "prompt_id": row["prompt_id"], "response_hold": response_hold,
                "prompt_wire_sha256": row["prompt_wire_sha256"], "mask": str(root / row["mask_path"]),
                "mask_sha256": row["mask_sha256"], "raw": str(raw), "raw_sha256": raw_sha,
                "model_metadata": response.get("model_metadata", "UNREPORTED" if response.get("response_model") is None else "MATCH"),
                "reported_model": response.get("response_model"),
            }
            g3.require(raw_sha == response["raw_output_sha256"], "Raw terminal/response hash mismatch")
    return entries


def write_image(path, image):
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image)
    if not success:
        raise OSError(f"Could not encode {path.name}")
    g3.write_exclusive(path, encoded.tobytes())


def copy_asset(source, expected_sha, destination):
    path = Path(source)
    hash_file(path, expected_sha)
    with path.open("rb") as handle, destination.open("xb") as output:
        shutil.copyfileobj(handle, output)
    return hash_file(destination, expected_sha)


def selection_overlay(original, shapes, indexes):
    overlay = original.copy()
    for i, shape in enumerate(shapes):
        pts = np.int32(np.round(shape["points"])).reshape((-1, 1, 2))
        color = (255, 80, 60) if i in indexes else (90, 200, 255)
        cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
    return overlay


def fit_variant(original, raw, target, folder, prefer_support=False):
    fits, selected, matches, delta = alignment.registration_candidates(original, raw, target)
    source_size = original.shape[1], original.shape[0]
    raw_size = raw.shape[1], raw.shape[0]
    if selected and prefer_support:
        def support_score(fit):
            valid = alignment.valid_support(np.float64(fit["matrix_output_to_source"]), raw_size, source_size)
            return (int(np.count_nonzero((target > 0) & (valid == 0))),
                    int(np.count_nonzero(valid == 0)), fit["median_reprojection_error_source_px"])

        selected = min((fit for fit in fits if fit["status"] == "GEOMETRY_CANDIDATE"), key=support_score)
    diagnostics = {
        "geometry_status": "GEOMETRY_CANDIDATE" if selected else "ALIGNMENT_UNVERIFIED",
        "mapping_method": selected["method"] if selected else "NONE",
        "match_count": matches, "aspect_ratio_difference": delta, "candidate_diagnostics": fits,
    }
    if selected is None:
        diagnostics["reason"] = "FEWER_THAN_12_MATCHES" if matches < 12 else "SIFT_GEOMETRY_SCREEN_FAILED"
        return None, None, diagnostics
    matrix = np.float64(selected["matrix_output_to_source"])
    aligned = cv2.warpAffine(raw, matrix, source_size, flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = alignment.valid_support(matrix, raw_size, source_size)
    target_valid, hard, feather = alignment.composites(original, aligned, valid, target)
    for filename, image in (("aligned.png", aligned), ("valid.png", valid), ("valid_target.png", target_valid),
                            ("hard.png", hard), ("feather.png", feather)):
        write_image(folder / filename, image)
    diagnostics["selected_fit"] = selected
    return aligned, valid, diagnostics


def build_candidate(root, variant, entry, source, parent=None):
    variant_root = root / "derived" / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    folder = variant_root / str(time.time_ns())
    folder.mkdir(exist_ok=False)
    image_path, annotation_path = Path(source["image"]), Path(source["annotation"])
    hash_file(image_path, source["image_sha256"])
    hash_file(annotation_path, source["annotation_sha256"])
    hash_file(Path(entry["raw"]), entry["raw_sha256"])
    shapes = load_json(annotation_path)["shapes"]
    target_image = target_mask(source, shapes)
    verify_mask(Path(entry["mask"]), target_image, entry["mask_sha256"])
    target = np.asarray(target_image)
    original, raw = alignment.read_rgb(image_path), alignment.read_rgb(entry["raw"])
    g3.require(list(original.shape[1::-1]) == source["size"], "Source image size changed")
    original_name = f"original{image_path.suffix.lower()}"
    raw_name = Path(entry["raw"]).name
    copy_asset(image_path, source["image_sha256"], folder / original_name)
    copy_asset(entry["raw"], entry["raw_sha256"], folder / raw_name)
    overlay = selection_overlay(original, shapes, source["indexes"])
    write_image(folder / "polygon_overlay.png", overlay)
    preserved = entry["derived"] if parent is None else {}
    if "aligned_output" in preserved:
        for key, name in (("aligned_output", "aligned.png"), ("valid_mask", "valid.png"),
                          ("hard_composite", "hard.png"), ("feather_composite", "feather.png")):
            asset = preserved[key]
            copy_asset(asset["path"], asset["sha256"], folder / name)
        aligned = alignment.read_rgb(folder / "aligned.png")
        valid = alignment.read_rgb(folder / "valid.png")[:, :, 0]
        write_image(folder / "valid_target.png", np.bitwise_and(target, valid))
        saved = preserved["geometry_diagnostic"]
        hash_file(Path(saved["path"]), saved["sha256"])
        diagnostics = load_json(Path(saved["path"]))
        diagnostics["reused_exact_aligned_sha256"] = preserved["aligned_output"]["sha256"]
    else:
        aligned, valid, diagnostics = fit_variant(original, raw, target, folder, prefer_support=parent is not None)

    polygons = []
    for pid, index in zip(source["selected_polygon_ids"], source["indexes"]):
        one, _ = mask_for_selection(tuple(source["size"]), shapes, {index})
        raster = np.asarray(one)
        left, top, width, height = cv2.boundingRect(raster)
        pad = max(8, round(min(source["size"]) * 0.04))
        box = (max(0, left - pad), max(0, top - pad), min(source["size"][0], left + width + pad),
               min(source["size"][1], top + height + pad))
        x0, y0, x1, y1 = box
        before = f"{pid}_before.png"
        after = f"{pid}_after.png" if aligned is not None else None
        write_image(folder / before, original[y0:y1, x0:x1])
        if after:
            write_image(folder / after, aligned[y0:y1, x0:x1])
        missing = int(np.count_nonzero((raster > 0) & (valid == 0))) if valid is not None else None
        polygons.append({"polygon_id": pid, "class": source["polygon_labels"][pid], "box": box,
                         "before": before, "after": after, "unsupported_target_pixels": missing})

    coverage = float(np.count_nonzero(valid) / valid.size) if valid is not None else None
    unsupported_bright = None
    if valid is not None:
        grayscale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        unsupported_bright = int(np.count_nonzero((valid == 0) & (grayscale > 12)))
    hashes = {path.name: hash_file(path) for path in sorted(folder.iterdir()) if path.is_file()}
    record = {
        "variant_id": variant, "source_id": source["source_id"], "version": entry["version"],
        "origin": entry["origin"], "classes": source["classes"], "polygons": polygons,
        "source_sha256": source["image_sha256"], "annotation_sha256": source["annotation_sha256"],
        "mask_sha256": entry["mask_sha256"], "request_sha256": entry["request_sha256"],
        "raw_sha256": entry["raw_sha256"], "aligned_sha256": hashes.get("aligned.png"),
        "original": original_name, "raw": raw_name, "asset_directory": folder.relative_to(root).as_posix(),
        "file_sha256": hashes, "geometry_status": diagnostics["geometry_status"],
        "geometry_reason": diagnostics.get("reason", ""), "mapping_method": diagnostics["mapping_method"],
        "match_count": diagnostics.get("match_count"), "valid_coverage": coverage,
        "unsupported_bright_original_pixels": unsupported_bright,
        "unsupported_target_pixels": int(np.count_nonzero((target > 0) & (valid == 0))) if valid is not None else None,
        "historical_visual": entry.get("historical_visual", ""), "a1_status": entry.get("a1_status", []),
        "model_metadata": entry.get("model_metadata", ""), "reported_model": entry.get("reported_model"),
        "a2_status": entry.get("a2_status", ""), "historical_support": entry.get("historical_support", ""),
        "ui": entry.get("historical_ui") or "待人工对比",
        "source_hold": source.get("hold") or entry.get("response_hold", ""),
    }
    if record["a2_status"] == "CORE_SOURCE_SCREEN_PASS":
        g3.require(record["aligned_sha256"] == preserved["aligned_output"]["sha256"],
                   "A2 review cannot be inherited across alignment images")
    record["review_id"] = canonical_digest({key: record[key] for key in (
        "variant_id", "source_sha256", "annotation_sha256", "mask_sha256", "request_sha256",
        "raw_sha256", "aligned_sha256",
    )})
    g3.write_json_exclusive(folder / "geometry.json", diagnostics)
    if parent:
        record["alignment_parent_review_id"] = parent["review_id"]
        record["generation_variant_id"] = parent.get("generation_variant_id", parent["variant_id"])
        if record["aligned_sha256"] == parent["aligned_sha256"]:
            g3.write_json_exclusive(folder / "UNCHANGED.json", record)
            return None
    g3.write_json_exclusive(folder / "record.json", record)
    return record


def finished_candidates(root):
    by_variant = {}
    for path in sorted((root / "derived").glob("*/*/record.json")):
        record = load_json(path)
        g3.require(path.parent == root / record["asset_directory"], "Candidate output path mismatch")
        by_variant[record["variant_id"]] = record
    return by_variant


def process(root, follow):
    cv2.setNumThreads(2)
    scope, _, _ = frozen_plan(root)
    done = set(finished_candidates(root))
    failed = set()
    inspected_responses = set()
    while True:
        recover_responses(root, scope, inspected_responses)
        for variant, entry in candidates(root, scope, done | failed, running=follow).items():
            try:
                record = build_candidate(root, variant, entry, scope["sources"][entry["source_id"]])
                print(f"{variant} {record['geometry_status']}", flush=True)
                done.add(variant)
            except (OSError, ValueError, cv2.error) as error:
                failed.add(variant)
                g3.write_json_exclusive(root / "logs" / f"alignment_error_{time.time_ns()}.json",
                                        {"variant_id": variant, "error": str(error), "type": type(error).__name__})
        launches = sorted((root / "logs").glob("start_*.json"))
        stops = sorted((root / "logs").glob("stop_*.json"))
        if not follow or (launches and stops and stops[-1].stat().st_mtime >= launches[-1].stat().st_mtime):
            if not candidates(root, scope, done | failed):
                break
        time.sleep(8)
    print(f"REVIEW_CANDIDATES={len(done)} POSTPROCESS_ERRORS={len(failed)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    process((PRIVATE_ROOT / PHASE).resolve(), args.follow)


if __name__ == "__main__":
    main()
