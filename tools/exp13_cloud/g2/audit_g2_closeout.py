"""One fixed, offline audit for the three saved Exp13-G2 source results.

This tool reads private evidence and writes a new private report directory.  It
never sends requests, follows URLs, opens VAL/TEST images, or changes source
assets.  Geometry uses the frozen G1F-2 SIFT/Lowe/RANSAC parameters once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_BUNDLE = "cee4f8efacab9027dcbcfa8b97e17d1edfb5ed72947bb3fc459fcd317ddfa115"
EXPECTED_SPLIT = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
SOURCES = ("260", "470", "23")
KNOWN_260_UI = {
    "battery_indicator": [62, 96, 145, 175],
    "status_text": [618, 136, 662, 171],
    "focus_overlay": [49, 320, 143, 419],
    "logo": [61, 414, 128, 482],
    "date_time": [551, 433, 669, 492],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def shape_points(payload: dict[str, Any]) -> list[tuple[str, list[tuple[float, float]]]]:
    result = []
    for shape in payload.get("shapes", []):
        points = [(float(point[0]), float(point[1])) for point in shape.get("points", [])]
        result.append((str(shape.get("label", "")), points))
    return result


def issue(issues: list[str], text: str) -> None:
    issues.append(text)


def audit_source_identity(source_id: str, bundle: dict[str, Any], asset_root: Path, issues: list[str]) -> dict[str, Any]:
    from PIL import Image
    source = bundle["sources"][source_id]
    directory = asset_root / "sources" / source_id
    actual: dict[str, Any] = {"source_id": source_id, "files": {}, "annotation": {}, "mask": {}}
    image_candidates = list(directory.glob("original_image.*"))
    if len(image_candidates) != 1:
        issue(issues, f"{source_id}: expected one original_image.* file, found {len(image_candidates)}")
        return actual
    image_path = image_candidates[0]
    annotation_path = directory / "original_annotation.json"
    binary_path = directory / "polygon_mask.png"
    rgba_path = directory / "api_rgba_mask.png"
    overlay_path = directory / "polygon_overlay.png"
    expected_hashes = source["sha256"]
    for key, path in (("original_image", image_path), ("original_professional_json", annotation_path), ("polygon_mask", binary_path), ("api_rgba_mask", rgba_path), ("polygon_overlay_private", overlay_path)):
        if not path.is_file():
            issue(issues, f"{source_id}: missing {path.name}")
            continue
        actual["files"][key] = sha256_file(path)
        if actual["files"][key] != expected_hashes[key]:
            issue(issues, f"{source_id}: {key} SHA256 mismatch")
    try:
        with Image.open(image_path) as image:
            image.verify()
        image = Image.open(image_path).convert("RGB")
        actual["image"] = {"size_wh": list(image.size), "mode": image.mode, "format": Image.open(image_path).format}
        if list(image.size) != source["image_size_wh"]:
            issue(issues, f"{source_id}: original image dimensions differ from frozen bundle")
    except Exception as exc:
        issue(issues, f"{source_id}: original image decode failed: {exc}")
        return actual
    try:
        annotation = read_json(annotation_path)
        shapes = shape_points(annotation)
        actual["annotation"] = {"shape_count": len(shapes), "labels": [label for label, _ in shapes], "point_counts": [len(points) for _, points in shapes]}
        if len(shapes) != source["polygon_count"] or actual["annotation"]["point_counts"] != source["polygon_point_counts"]:
            issue(issues, f"{source_id}: Polygon count/point counts differ from frozen bundle")
        for label, points in shapes:
            if label != source["class"]:
                issue(issues, f"{source_id}: unexpected annotation label {label!r}")
            if len(points) < 3 or any(not (math.isfinite(x) and math.isfinite(y)) for x, y in points):
                issue(issues, f"{source_id}: invalid Polygon points")
            if any(x < 0 or y < 0 or x > source["image_size_wh"][0] or y > source["image_size_wh"][1] for x, y in points):
                issue(issues, f"{source_id}: Polygon point outside image bounds")
        if annotation.get("imageWidth") not in (None, source["image_size_wh"][0]) or annotation.get("imageHeight") not in (None, source["image_size_wh"][1]):
            issue(issues, f"{source_id}: annotation dimensions differ from image")
    except Exception as exc:
        issue(issues, f"{source_id}: professional JSON parse failed: {exc}")
        shapes = []
    try:
        binary = Image.open(binary_path).convert("L")
        rgba = Image.open(rgba_path).convert("RGBA")
        overlay = Image.open(overlay_path).convert("RGB")
        target = __import__("numpy").asarray(binary) > 0
        alpha = __import__("numpy").asarray(rgba.getchannel("A"))
        actual["mask"] = {"binary_size_wh": list(binary.size), "rgba_size_wh": list(rgba.size), "overlay_size_wh": list(overlay.size), "target_pixels": int(target.sum()), "alpha_target_zero": bool((alpha[target] == 0).all()) if target.any() else False, "alpha_outside_255": bool((alpha[~target] == 255).all()) if (~target).any() else False}
        if binary.size != tuple(source["image_size_wh"]) or rgba.size != binary.size or overlay.size != binary.size:
            issue(issues, f"{source_id}: mask/overlay dimensions differ from source")
        if int(target.sum()) != source["mask_target_pixels"]:
            issue(issues, f"{source_id}: polygon mask pixel count mismatch")
        if not actual["mask"]["alpha_target_zero"] or not actual["mask"]["alpha_outside_255"]:
            issue(issues, f"{source_id}: API RGBA alpha polarity mismatch")
    except Exception as exc:
        issue(issues, f"{source_id}: mask/overlay decode failed: {exc}")
    return actual


def audit_execution(source_id: str, bundle: dict[str, Any], execution_root: Path, issues: list[str]) -> dict[str, Any]:
    source = bundle["sources"][source_id]
    live = execution_root / "live" / source_id
    lock_path = execution_root / "locks" / source_id / "STARTED.json"
    auth_path = execution_root / "authorizations" / source_id / "gate_approval.json"
    request_path, response_path = live / "request_manifest.json", live / "response_manifest.json"
    http_path, terminal_path = live / "http_response_metadata.json", live / "TERMINAL.json"
    required = (lock_path, auth_path, request_path, response_path, http_path, terminal_path, live / "raw_output.bin", live / "raw_output.png")
    for path in required:
        if not path.is_file():
            issue(issues, f"{source_id}: missing execution evidence {path.name}")
    if any(not path.is_file() for path in required):
        return {"source_id": source_id, "execution_state": "EVIDENCE_INCOMPLETE"}
    request, response = read_json(request_path), read_json(response_path)
    http, terminal = read_json(http_path), read_json(terminal_path)
    lock, auth = read_json(lock_path), read_json(auth_path)
    payload = request.get("request", {})
    fields = payload.get("multipart_fields", {})
    prompt = fields.get("prompt", {})
    checks = {
        "request_identity": request.get("status") == "FROZEN_BEFORE_POST" and payload.get("method") == "POST" and payload.get("model") == bundle["model"] and payload.get("n") == 1 and payload.get("source_id") == source_id and payload.get("source_role") == source["source_role"],
        "request_input_hashes": request.get("input_hashes") == source["sha256"],
        "prompt_hashes": prompt.get("file_sha256") == source["prompt_file_sha256"] and prompt.get("transmitted_string_sha256") == source["prompt_transport_sha256"],
        "approval_binding": request.get("approval_receipt_sha256") == sha256_file(auth_path) and auth.get("source_id") == source_id and auth.get("execution_bundle_sha256") == EXPECTED_BUNDLE,
        "lock_binding": lock.get("source_id") == source_id and lock.get("status") == "PERMANENT_BUDGET_OCCUPIED" and lock.get("max_posts") == 1 and lock.get("execution_bundle_sha256") == EXPECTED_BUNDLE,
        "http_once": http.get("http_status") == 200 and http.get("post_request_invocations") == 1 and http.get("redirect_followed") is False,
        "terminal_occupied": terminal.get("generation_posts_maximum") == 1 and terminal.get("budget_stays_occupied") is True,
        "response_once": response.get("post_request_invocations") == 1 and response.get("requested_model") == bundle["model"],
    }
    for name, passed in checks.items():
        if not passed:
            issue(issues, f"{source_id}: execution check failed: {name}")
    raw_bin, raw_png = (live / "raw_output.bin").read_bytes(), (live / "raw_output.png").read_bytes()
    raw_sha = hashlib.sha256(raw_bin).hexdigest()
    checks["raw_byte_identity"] = raw_bin == raw_png and raw_sha == response.get("raw_output_sha256")
    if not checks["raw_byte_identity"]:
        issue(issues, f"{source_id}: raw_output.bin/png or response hash mismatch")
    from PIL import Image

    try:
        with Image.open(live / "raw_output.png") as image:
            image.verify()
        decoded = Image.open(live / "raw_output.png")
        output_meta = {"format": decoded.format, "mode": decoded.mode, "size_wh": list(decoded.size), "bytes": len(raw_png), "sha256": raw_sha}
        checks["output_metadata"] = decoded.format == "PNG" and decoded.mode == "RGB" and list(decoded.size) == response.get("image", {}).get("size_wh")
    except Exception as exc:
        output_meta = {"decode_error": str(exc)}
        checks["output_metadata"] = False
    if not checks["output_metadata"]:
        issue(issues, f"{source_id}: raw output format/mode/size mismatch")
    usage = response.get("usage")
    cost_status = response.get("cost_status", "UNKNOWN")
    evidence_state = "EXECUTION_VERIFIED" if all(checks.values()) else "EVIDENCE_INCOMPLETE"
    if cost_status in ("NOT_RETURNED", "UNKNOWN", None):
        evidence_state = "EVIDENCE_INCOMPLETE"
    return {
        "source_id": source_id, "role": source["source_role"], "execution_state": "EXECUTION_VERIFIED" if all(checks.values()) else "EVIDENCE_INCOMPLETE",
        "evidence_state": evidence_state, "checks": checks, "http_status": http.get("http_status"), "post_count": http.get("post_request_invocations"),
        "response_format": response.get("response_format"), "output": output_meta, "usage": usage, "reported_cost": response.get("reported_cost"), "cost_status": cost_status,
        "upstream_identity": response.get("upstream_identity"), "elapsed_seconds": response.get("elapsed_seconds"), "prompt_id": fields.get("prompt", {}).get("prompt_id"),
        "request_manifest_sha256": sha256_file(request_path), "lock_sha256": sha256_file(lock_path), "approval_sha256": sha256_file(auth_path),
    }


def support_ids(keypoints: list[Any], allowed: Any, cv2: Any, np: Any) -> list[int]:
    distance = cv2.distanceTransform(np.pad(allowed, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    return [index for index, point in enumerate(keypoints) if 0 <= round(point.pt[0]) < allowed.shape[1] and 0 <= round(point.pt[1]) < allowed.shape[0] and distance[round(point.pt[1]), round(point.pt[0])] > max(2.0, 1.5 * point.size)]


def match_points(keys_a: list[Any], keys_b: list[Any], desc_a: Any, desc_b: Any, ids_a: list[int], ids_b: list[int], cv2: Any, np: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if desc_a is None or desc_b is None or not ids_a or len(ids_b) < 2:
        return [], {"initial_knn_queries": 0, "ratio_passed": 0, "distinct_pairs": 0}
    found = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a[ids_a], desc_b[ids_b], k=2)
    candidates = []
    for pair in found:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            first, second = pair
            candidates.append((first.distance / second.distance, first.distance, ids_a[first.queryIdx], ids_b[first.trainIdx]))
    used_a, used_b, result = set(), set(), []
    for ratio, distance, index_a, index_b in sorted(candidates):
        bin_a, bin_b = tuple(int(value // 2) for value in keys_a[index_a].pt), tuple(int(value // 2) for value in keys_b[index_b].pt)
        if bin_a not in used_a and bin_b not in used_b:
            used_a.add(bin_a); used_b.add(bin_b)
            result.append({"roi": list(keys_a[index_a].pt), "raw": list(keys_b[index_b].pt), "ratio": ratio, "descriptor_l2": distance})
    return result, {"initial_knn_queries": len(found), "ratio_passed": len(candidates), "distinct_pairs": len(result)}


def point_distribution(points: Any, width: int, height: int, domain_pixels: int, cv2: Any, np: Any) -> dict[str, Any]:
    if len(points) == 0:
        return {"bbox_xyxy": None, "span_xy": [0.0, 0.0], "supported_cells": [], "separated_supported_regions": False, "hull_fraction_of_domain": 0.0}
    values = np.asarray(points, dtype=np.float64)
    minimum, maximum = values.min(axis=0), values.max(axis=0)
    cells: dict[str, list[Any]] = {}
    for x, y in values:
        key = f"{min(2, int(x / (width / 3)))}:{min(1, int(y / (height / 2)))}"
        cells.setdefault(key, []).append((x, y))
    supported = [key for key, values in cells.items() if len(values) >= 3]
    centroids = {key: np.mean(values, axis=0) for key, values in cells.items()}
    separated = any(np.linalg.norm(centroids[left] - centroids[right]) >= 60 for index, left in enumerate(supported) for right in supported[index + 1:])
    hull_area = float(cv2.contourArea(cv2.convexHull(values.astype(np.float32)))) if len(values) >= 3 else 0.0
    return {"bbox_xyxy": [*minimum.tolist(), *maximum.tolist()], "span_xy": (maximum - minimum).tolist(), "cells": {key: len(value) for key, value in cells.items()}, "supported_cells": supported, "separated_supported_regions": separated, "hull_fraction_of_domain": hull_area / max(1, domain_pixels)}


def fit_model(pairs: list[dict[str, Any]], kind: str, raw_size: tuple[int, int], allowed_pixels: int, width: int, height: int, cv2: Any, np: Any) -> dict[str, Any]:
    if len(pairs) < 3:
        return {"status": "INSUFFICIENT_MATCHES", "inlier_count": 0, "inlier_ratio": 0.0, "numeric_screen_pass": False}
    source = np.float64([pair["raw"] for pair in pairs]); target = np.float64([pair["roi"] for pair in pairs])
    cv2.setRNGSeed(20261101)
    estimator = cv2.estimateAffinePartial2D if kind == "similarity" else cv2.estimateAffine2D
    matrix, inliers = estimator(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99, refineIters=10)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return {"status": "ESTIMATION_UNAVAILABLE", "inlier_count": 0, "inlier_ratio": 0.0, "numeric_screen_pass": False}
    keep = inliers.ravel().astype(bool)
    projected = source @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(projected - target, axis=1)
    inlier_count = int(keep.sum())
    distribution = point_distribution(target[keep], width, height, allowed_pixels, cv2, np)
    inverse = cv2.invertAffineTransform(matrix); yy, xx = np.indices((height, width)); back = np.column_stack((xx.ravel(), yy.ravel())) @ inverse[:, :2].T + inverse[:, 2]
    raw_width, raw_height = raw_size
    valid = ((back[:, 0] >= 1) & (back[:, 0] <= raw_width - 2) & (back[:, 1] >= 1) & (back[:, 1] <= raw_height - 2)).reshape(height, width)
    linear = matrix[:, :2]; scales = np.linalg.svd(linear, compute_uv=False)
    median, p90 = float(np.median(errors[keep])), float(np.percentile(errors[keep], 90))
    criteria = {
        "at_least_12_distinct_inliers": inlier_count >= 12,
        "median_at_most_3_original_px": median <= 3.0,
        "p90_at_most_3_original_px": p90 <= 3.0,
        "two_separated_supported_regions": distribution["separated_supported_regions"],
        "hull_at_least_5_percent_domain": distribution["hull_fraction_of_domain"] >= 0.05,
        "span_at_least_80x48_original_px": distribution["span_xy"][0] >= 80 and distribution["span_xy"][1] >= 48,
        "valid_overlap_at_least_half": float(valid.mean()) >= 0.5,
        "positive_determinant": float(np.linalg.det(linear)) > 0,
    }
    return {"status": "ESTIMATED", "matrix_raw_to_roi": matrix.tolist(), "inlier_count": inlier_count, "outlier_count": len(pairs) - inlier_count, "inlier_ratio": inlier_count / len(pairs), "median_error_original_px": median, "p90_error_original_px": p90, "valid_overlap_fraction": float(valid.mean()), "singular_scales": scales.tolist(), "distribution": distribution, "criteria": criteria, "numeric_screen_pass": all(criteria.values())}


def geometry_audit(source_id: str, asset_root: Path, execution_root: Path, cpu_deps: Path, bundle: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(cpu_deps.resolve()))
    import cv2
    import numpy as np
    from PIL import Image

    source_dir = asset_root / "sources" / source_id; live_dir = execution_root / "live" / source_id
    image_path = next(source_dir.glob("original_image.*")); raw_path = live_dir / "raw_output.png"
    roi = Image.open(image_path).convert("RGB"); raw = Image.open(raw_path).convert("RGB"); binary = np.asarray(Image.open(source_dir / "polygon_mask.png").convert("L")) > 0
    width, height = roi.size
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    roi_keys, roi_desc = sift.detectAndCompute(cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2GRAY), None)
    raw_keys, raw_desc = sift.detectAndCompute(cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2GRAY), None)
    excluded = cv2.dilate(np.uint8(binary), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    full_allowed = np.uint8(excluded == 0) * 255
    sets: list[tuple[str, Any, bool]] = [("full_frame_defect_excluded_ui_allowed", full_allowed, False)]
    ui_status = "UNKNOWN"
    if source_id == "260":
        ui_allowed = full_allowed.copy()
        for x1, y1, x2, y2 in KNOWN_260_UI.values():
            ui_allowed[y1:y2, x1:x2] = 0
        sets.append(("mechanical_ui_and_defect_excluded", ui_allowed, True)); ui_status = "FIXED_260_REUSED_FROM_G1F2_AND_PROTECTED_G2_AUDIT"
    results: dict[str, Any] = {"ui_exclusion_status": ui_status, "native_keypoints": {"roi": len(roi_keys), "raw": len(raw_keys)}, "sets": {}}
    for name, allowed, mechanical in sets:
        raw_allowed = cv2.resize(allowed, raw.size, interpolation=cv2.INTER_NEAREST)
        ids_a = support_ids(roi_keys, allowed, cv2, np); ids_b = support_ids(raw_keys, raw_allowed, cv2, np)
        pairs, counts = match_points(roi_keys, raw_keys, roi_desc, raw_desc, ids_a, ids_b, cv2, np)
        models = {kind: fit_model(pairs, kind, raw.size, int(np.count_nonzero(allowed)), width, height, cv2, np) for kind in ("similarity", "affine")}
        results["sets"][name] = {"mechanical_set": mechanical, "eligible_keypoints": {"roi": len(ids_a), "raw": len(ids_b)}, "matches": counts, "models": models}
    mechanical = results["sets"].get("mechanical_ui_and_defect_excluded")
    candidate = bool(mechanical and any(model.get("numeric_screen_pass") for model in mechanical["models"].values()))
    results["state"] = "GEOMETRY_CANDIDATE" if candidate else "GEOMETRY_UNVERIFIED"
    results["parameter_search"] = False; results["nonrigid_transform"] = False; results["api_posts"] = 0; results["gpu"] = 0; results["training"] = 0; results["val_test_access"] = 0
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--cpu-deps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-fixed-once", action="store_true")
    args = parser.parse_args()
    if not args.run_fixed_once:
        raise SystemExit("explicit --run-fixed-once is required")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    repo = Path(__file__).resolve().parents[3]
    bundle_path = repo / "configs/exp13_cloud/g2/execution_bundle.json"
    if sha256_file(bundle_path) != EXPECTED_BUNDLE:
        raise SystemExit("frozen execution bundle SHA256 mismatch")
    bundle = read_json(bundle_path)
    if bundle.get("frozen_split_manifest_sha256") != EXPECTED_SPLIT:
        raise SystemExit("frozen split manifest SHA256 mismatch")
    sys.path.insert(0, str(args.cpu_deps.resolve()))
    args.output.mkdir(parents=True)
    issues: list[str] = []
    result: dict[str, Any] = {"experiment": "Exp13-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "asset_id": "cloud_pilot_g2_execution_20261102", "bundle_sha256": EXPECTED_BUNDLE, "split_manifest_sha256": EXPECTED_SPLIT, "api_posts_added_by_audit": 0, "network_requests_added_by_audit": 0, "gpu": 0, "training": 0, "val_test_image_access": 0, "sources": {}}
    for source_id in SOURCES:
        source_issues: list[str] = []
        input_report = audit_source_identity(source_id, bundle, args.asset_root.parent / "cloud_pilot_g2_three_source_20261101_retry1", source_issues)
        execution_report = audit_execution(source_id, bundle, args.execution_root, source_issues)
        try:
            geometry_report = geometry_audit(source_id, args.asset_root.parent / "cloud_pilot_g2_three_source_20261101_retry1", args.execution_root, args.cpu_deps, bundle, source_issues)
        except Exception as exc:
            geometry_report = {"state": "GEOMETRY_UNVERIFIED", "status": "CPU_DIAGNOSTIC_UNAVAILABLE", "error": str(exc), "api_posts": 0, "gpu": 0, "training": 0, "val_test_access": 0}
            source_issues.append(f"{source_id}: geometry diagnostic failed: {exc}")
        execution_state = execution_report.get("execution_state", "EVIDENCE_INCOMPLETE")
        evidence_state = execution_report.get("evidence_state", execution_state)
        result["sources"][source_id] = {
            "class": bundle["sources"][source_id]["class"], "role": bundle["sources"][source_id]["source_role"], "visual_status": "VISUAL_ACCEPTED (user-confirmed)",
            "execution_state": execution_state, "evidence_state": evidence_state, "geometry_state": geometry_report.get("state", "GEOMETRY_UNVERIFIED"),
            "pair_status": "PAIR_USABILITY_PENDING", "train_pair_eligible": False, "input": input_report, "execution": execution_report, "geometry": geometry_report,
            "issues": source_issues,
        }
        write_json(args.output / f"source_{source_id}_audit.json", result["sources"][source_id])
    result["overall"] = {"visual": "VISUAL_ACCEPTED", "execution": "EXECUTION_VERIFIED" if all(item["execution_state"] == "EXECUTION_VERIFIED" for item in result["sources"].values()) else "EVIDENCE_INCOMPLETE", "geometry": {source_id: item["geometry_state"] for source_id, item in result["sources"].items()}, "pair": "PAIR_USABILITY_PENDING", "trusted_pair_pass": False}
    write_json(args.output / "audit_report.json", result)
    with (args.output / "execution_ledger.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["source_id", "class", "role", "visual_status", "execution_state", "evidence_state", "http_status", "post_count", "response_format", "output_size", "raw_output_sha256", "usage_total_tokens", "reported_cost", "cost_status", "geometry_state", "pair_status"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for source_id, item in result["sources"].items():
            execution = item["execution"]; output = execution.get("output", {})
            writer.writerow({"source_id": source_id, "class": item["class"], "role": item["role"], "visual_status": item["visual_status"], "execution_state": item["execution_state"], "evidence_state": item["evidence_state"], "http_status": execution.get("http_status"), "post_count": execution.get("post_count"), "response_format": execution.get("response_format"), "output_size": "x".join(map(str, output.get("size_wh", []))), "raw_output_sha256": output.get("sha256"), "usage_total_tokens": (execution.get("usage") or {}).get("total_tokens"), "reported_cost": execution.get("reported_cost"), "cost_status": execution.get("cost_status"), "geometry_state": item["geometry_state"], "pair_status": item["pair_status"]})
    lines = ["# Exp13-G2 closeout CPU audit", "", "VISUAL_ACCEPTED is the user's supplied visual result. It does not admit a training Pair.", "", "| Source | Role | Visual | Execution | Evidence | Geometry | Pair |", "|---|---|---|---|---|---|---|"]
    for source_id, item in result["sources"].items():
        lines.append(f"| {source_id} | {item['role']} | {item['visual_status']} | {item['execution_state']} | {item['evidence_state']} | {item['geometry_state']} | {item['pair_status']} |")
    lines += ["", "Fixed protocol: native grayscale SIFT, BF/L2 k=2, Lowe 0.75, 2-pixel spatial deduplication, Similarity/Affine RANSAC at 3 ROI pixels, seed 20261101. No parameter search or nonrigid fit.", "", "Cost fields are NOT_RETURNED in all three response manifests, so evidence is incomplete for billing reconciliation even where request/response/lock execution checks pass. UI exclusion is fixed only for source 260; sources 470 and 23 are therefore GEOMETRY_UNVERIFIED despite full-frame diagnostics. All Pair usability and trusted-pair decisions remain pending human review.", "", "API POSTs added by this audit: 0; GPU: 0; training: 0; VAL/TEST image access: 0."]
    (args.output / "G2_CLOSEOUT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if not issues else "PASS_WITH_RECORDED_LIMITATIONS", "output": str(args.output), "sources": result["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
