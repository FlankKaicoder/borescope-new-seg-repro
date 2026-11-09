"""Freeze existing transforms; collect independent human landmarks; evaluate, never fit.

Private build/evaluate outputs are exclusive-create. No cloud or training code.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from align_g3b2_batch import read_rgb, read_target_mask, sha256, write_rgb
from closeout_g3c1_review import EXPECTED_REVIEW_SHA256
from validate_alignment_g3d import target_neighborhood_coverage

HERE = Path(__file__).resolve().parent
SCHEMA = "g3d2-mechanical-landmarks-v1"
SELECTION = [
    ("558", "V1", "visual accepted / D1 holdout pass"),
    ("570", "V1", "multi-Polygon / original candidate / D1 holdout fail"),
    ("483", "V1", "visual accepted / D1 holdout pass"),
    ("469", "V1", "original candidate / D1 holdout fail"),
    ("861", "V1", "visual accepted / D1 holdout pass"),
    ("914", "V1", "original candidate / D1 holdout fail"),
    ("708", "V1", "visual accepted / D1 holdout pass"),
    ("636", "V1", "original candidate / D1 holdout fail"),
    ("805", "V1", "visual accepted / D1 holdout pass"),
    ("784", "V1", "original candidate / D1 holdout fail"),
    ("423", "V1", "multi-Polygon / visual accepted / D1 holdout pass"),
    ("195", "V1", "original candidate / D1 holdout fail"),
    ("1", "V1", "multi-Polygon / visual status versus Pair rejection note conflict"),
    ("271", "V1", "UI/Polygon overlap / visual retry / D1 holdout fail"),
    ("645", "V1", "geometry versus repair negative control / V1-V2 contrast"),
    ("645", "V2", "user visual acceptance / independent geometry still pending"),
    ("556", "V2", "low-feature unverified negative control"),
    ("640", "V2", "large feature-error tail / unverified negative control"),
]
QUALITY = ("visual_repair", "normal_structure", "mask_complete", "hard_seam",
           "feather_halo", "ui_non_target", "multi_polygon", "shortcut_risk")
POINT_STATES = {"PENDING", "CONFIRMED", "UNIDENTIFIABLE", "WRONG_MATCH",
                "STRUCTURE_CHANGED", "AMBIGUOUS"}
DECISIONS = {"PENDING", "PASS", "FAIL", "UNCERTAIN", "NOT_APPLICABLE"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def unique_csv(path, count):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, strict=True))
    by_id = {r["source_id"]: r for r in rows}
    if len(rows) != count or len(by_id) != count:
        raise ValueError(f"Wrong count/duplicate identities: {path}")
    return by_id


def saved_matrix(diagnostic):
    # Recover only the model actually used to write the historical aligned output.
    if diagnostic["geometry_status"] != "GEOMETRY_CANDIDATE":
        return None
    fits = [c for c in diagnostic["candidate_diagnostics"]
            if c["method"] == diagnostic["mapping_method"]
            and c["status"] == "GEOMETRY_CANDIDATE"]
    if len(fits) != 1:
        raise ValueError("Ambiguous historical selected transform")
    matrix = np.asarray(fits[0]["matrix_output_to_source"], np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Malformed historical transform")
    return matrix.tolist()


def coverage(matrix, raw_size, target):
    if matrix is None:
        return None
    valid = cv2.warpAffine(np.full((raw_size[1], raw_size[0]), 255, np.uint8),
                           np.asarray(matrix), (target.shape[1], target.shape[0]),
                           flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    radius = max(8, round(math.hypot(*target.shape) * .04))
    ring = cv2.dilate(target, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    ring[target > 0] = 0
    fraction = lambda region: float(np.mean(valid[region > 0] > 0)) if np.any(region) else None
    return {"full_target_coverage": fraction(target), "full_target_ring_coverage": fraction(ring),
            "ring_radius_source_px": radius, "basis": "FULL_FROZEN_REPAIR_MASK_NOT_CLIPPED"}


def initial_review(bundle):
    records = []
    for pair in bundle["pairs"]:
        records.append({"pair_id": pair["pair_id"], "reviewer": "", "notes": "",
                        "geometry_review": "PENDING", "pair_suitability": "PENDING",
                        "quality": {key: "PENDING" for key in QUALITY},
                        "points": [{"id": slot["id"], "status": "PENDING", "label": "",
                                    "source": None, "output": None, "uncertainty_px": None,
                                    "notes": ""} for slot in pair["slots"]]})
    return {"schema": SCHEMA, "bundle_id": bundle["bundle_id"], "records": records}


def validate_review(bundle, review):
    if review.get("schema") != SCHEMA or review.get("bundle_id") != bundle["bundle_id"]:
        raise ValueError("Wrong schema or frozen bundle identity")
    if set(review) != {"schema", "bundle_id", "records"}:
        raise ValueError("Unexpected review fields")
    records = review["records"]
    if not isinstance(records, list) or len(records) != len(bundle["pairs"]):
        raise ValueError("Exact pair set required")
    ids = [r["pair_id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != {p["pair_id"] for p in bundle["pairs"]}:
        raise ValueError("Unknown, duplicate or missing pair")
    by_id = {r["pair_id"]: r for r in records}
    for pair in bundle["pairs"]:
        r = by_id[pair["pair_id"]]
        if set(r) != {"pair_id", "reviewer", "notes", "geometry_review", "pair_suitability", "quality", "points"}:
            raise ValueError("Unexpected pair review fields")
        if any(not isinstance(r[key], str) for key in ("reviewer", "notes")):
            raise ValueError("Review text must be string")
        if r["geometry_review"] not in DECISIONS - {"NOT_APPLICABLE"}:
            raise ValueError("Invalid geometry decision")
        if r["pair_suitability"] not in {"PENDING", "REJECT", "CANDIDATE_FOR_GATE"}:
            raise ValueError("Pair cannot self-authorize trusted PASS")
        if set(r["quality"]) != set(QUALITY) or any(v not in DECISIONS for v in r["quality"].values()):
            raise ValueError("Invalid quality decisions")
        points = r["points"]
        if not isinstance(points, list):
            raise ValueError("Points must be array")
        pids = [p["id"] for p in points]
        if len(set(pids)) != len(pids) or set(pids) != {s["id"] for s in pair["slots"]}:
            raise ValueError("Exact point slot set required")
        for p in points:
            if set(p) != {"id", "status", "label", "source", "output", "uncertainty_px", "notes"}:
                raise ValueError("Unexpected point fields")
            if p["status"] not in POINT_STATES or any(not isinstance(p[k], str) for k in ("label", "notes")):
                raise ValueError("Invalid landmark state/text")
            for side in ("source", "output"):
                xy = p[side]
                if xy is not None:
                    size = pair[f"{side}_size"]
                    if (not isinstance(xy, list) or len(xy) != 2 or any(
                        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < size[i]
                        for i, v in enumerate(xy))):
                        raise ValueError("Landmark outside image or non-finite")
            u = p["uncertainty_px"]
            if u is not None and (type(u) not in (int, float) or not math.isfinite(u) or u < 0):
                raise ValueError("Invalid source-pixel uncertainty")
            if p["status"] == "CONFIRMED" and (
                not r["reviewer"].strip() or not p["label"].strip() or p["source"] is None
                or p["output"] is None or u is None):
                raise ValueError("Confirmed landmark requires reviewer, identity, both clicks and uncertainty")
    return by_id


def summaries(errors):
    return ({"n": len(errors), "median_px": float(np.median(errors)),
             "p90_px": float(np.percentile(errors, 90)), "max_px": float(max(errors))}
            if len(errors) else {"n": 0, "median_px": None, "p90_px": None, "max_px": None})


def audit_d1_masks(args, manifest, d1):
    counts = {}
    for version, rows in (("C0_V1_36", d1["pairs"][:36]), ("C2_V2_3", d1["pairs"][36:]),
                          ("ALL_39", d1["pairs"])):
        counts[version] = {det: dict(collections.Counter(
            row["methods"][det]["models"]["AFFINE"]["holdout"]["status"] for row in rows))
            for det in ("SIFT", "ORB")}
    checks = []
    for item in d1["pairs"][:36]:
        sid = item["source_id"]
        old = args.c0 / sid / "valid_target_mask.png"
        if not old.exists():
            continue
        row = manifest[sid]
        full = read_target_mask(args.batch / row["mask_path"],
                                (int(row["image_width"]), int(row["image_height"])))
        clipped = cv2.imread(str(old), cv2.IMREAD_GRAYSCALE)
        raw = read_rgb(args.c0 / sid / "raw_output.png")
        methods = {}
        for detector in ("SIFT", "ORB"):
            matrix = item["methods"][detector]["models"]["AFFINE"].get("matrix_output_to_source")
            if matrix is not None:
                matrix = np.asarray(matrix)
                methods[detector] = {
                    "D1_ring_from_clipped_target": target_neighborhood_coverage(matrix, raw.shape, clipped),
                    "same_31px_ring_from_full_target": target_neighborhood_coverage(matrix, raw.shape, full)}
        checks.append({"source_id": sid, "full_target_pixels": int(np.count_nonzero(full)),
                       "clipped_target_pixels": int(np.count_nonzero(clipped)),
                       "omitted_pixels": int(np.count_nonzero((full > 0) & (clipped == 0))), "methods": methods})
    return {"counts": counts, "clipped_mask_checks": checks,
            "interpretation": "D1_HOLDOUT_OF_REFITTED_MODELS_NOT_VALIDATION_OF_HISTORICAL_C0_TRANSFORM"}


def evaluate(bundle, review):
    records = validate_review(bundle, review)
    results = []
    for pair in bundle["pairs"]:
        r = records[pair["pair_id"]]
        slots = {s["id"]: s for s in pair["slots"]}
        accepted, exclusions = [], []
        target = np.asarray(pair["polygon_points"], dtype=object)
        radius = pair["near_radius_px"]
        for point in r["points"]:
            if point["status"] != "CONFIRMED":
                exclusions.append({"point_id": point["id"], "reason": point["status"]})
                continue
            xy = point["source"]
            distances = {pid: cv2.pointPolygonTest(np.float32(poly), tuple(xy), True)
                         for pid, poly in zip(pair["polygon_ids"], target)}
            slot = slots[point["id"]]
            normal = all(v < -2 for v in distances.values())
            near = slot["group"] == "NEAR"
            correct_group = (-radius <= distances[slot["polygon_id"]] < -2 if near
                             else all(v < -radius for v in distances.values()))
            if not normal or not correct_group or point["uncertainty_px"] > bundle["protocol"]["max_uncertainty_px"]:
                exclusions.append({"point_id": point["id"], "reason": "REGION_OR_UNCERTAINTY_INVALID"})
                continue
            if any(np.linalg.norm(np.asarray(point[side]) - p[side]) <= 4
                   for p in accepted for side in ("source", "output")):
                exclusions.append({"point_id": point["id"], "reason": "DUPLICATE_OR_NEAR_DUPLICATE_LANDMARK"})
                continue
            accepted.append(point)
        methods = {}
        for name, method in pair["methods"].items():
            matrix = method["matrix"]
            if matrix is None:
                methods[name] = {"status": "UNVERIFIED_NO_SAVED_TRANSFORM"}
                continue
            matrix = np.asarray(matrix)
            if accepted:
                src = np.asarray([p["source"] for p in accepted])
                out = np.asarray([p["output"] for p in accepted])
                errors = np.linalg.norm(out @ matrix[:, :2].T + matrix[:, 2] - src, axis=1)
            else:
                errors = np.array([])
            group_metrics = {g: summaries([e for e, p in zip(errors, accepted)
                                         if slots[p["id"]]["group"] == g]) for g in ("NEAR", "FAR")}
            per_polygon = {pid: summaries([e for e, p in zip(errors, accepted)
                                          if slots[p["id"]]["polygon_id"] == pid]) for pid in pair["polygon_ids"]}
            enough = group_metrics["FAR"]["n"] >= 3 and all(v["n"] >= 3 for v in per_polygon.values())
            limit = max(2.0, .003 * math.hypot(*pair["source_size"]))
            groups = [group_metrics["FAR"], *per_polygon.values()]
            within = enough and all(v["median_px"] <= limit and v["p90_px"] <= 2 * limit
                                    and v["max_px"] <= 3 * limit for v in groups)
            methods[name] = {"status": "TRE_WITHIN_PROTOCOL" if within else "TRE_EXCEEDS_PROTOCOL" if enough
                             else "PENDING_INDEPENDENT_POINTS", "global": summaries(errors),
                             "groups": group_metrics, "per_polygon": per_polygon,
                             "median_limit_px": limit,
                             "point_errors": [{"point_id": p["id"], "tre_px": float(e)} for p, e in zip(accepted, errors)]}
        results.append({"pair_id": pair["pair_id"], "review": r, "accepted_points": len(accepted),
                        "excluded_points": exclusions, "methods": methods,
                        "trusted_pair_status": "PENDING_CHATGPT_AND_EXPERT_GATE",
                        "candidate_status": "REJECT_RECORDED" if r["pair_suitability"] == "REJECT" else "PENDING"})
    return {"schema": SCHEMA, "bundle_id": bundle["bundle_id"], "pair_count": len(results),
            "confirmed_point_count": sum(r["accepted_points"] for r in results),
            "trusted_pair_pass": 0, "status": "PENDING_HUMAN_REVIEW" if not any(
                r["accepted_points"] for r in results) else "EVIDENCE_ONLY_PENDING_GATE", "results": results}


def build(args):
    # Read pinned TRAIN-only metadata and existing manifests; no dataset scan.
    manifest = unique_csv(args.batch / "expanded_batch_manifest.csv", 36)
    if sha256(args.batch / "expanded_batch_manifest.csv") != "2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa":
        raise ValueError("Frozen G3-B2 manifest changed")
    if sha256(args.review) != EXPECTED_REVIEW_SHA256:
        raise ValueError("Frozen G3-C1 review changed")
    visual = unique_csv(args.review, 36)
    v2_manifest = unique_csv(args.v2_batch / "expanded_batch_manifest.csv", 4)
    d1 = load(args.d1)
    if len(d1["pairs"]) != 39:
        raise ValueError("Expected 39 frozen D1 variants")
    # D1's variant field incorrectly repeats source ID. Bind by frozen ordinal AND hashes.
    d1_map = {(p["source_id"], "V1" if i < 36 else "V2"): p for i, p in enumerate(d1["pairs"])}
    if set(d1_map) != {(sid, "V1") for sid in manifest} | {(sid, "V2") for sid in ("556", "640", "645")}:
        raise ValueError("D1 pair variant crosswalk mismatch")
    bundle = {"schema": SCHEMA, "protocol": {"point_use": "EVALUATION_ONLY_NO_FITTING",
              "source_role": "INDEPENDENT_EVALUATION_ONLY", "max_uncertainty_px": 2,
              "duplicate_exclusion_radius_px": 4,
              "minimum_far_points": 3, "minimum_near_points_each_polygon": 3,
              "tre_limits": "median=max(2,.003*diagonal); P90=2x; max=3x; apply per Polygon and FAR",
              "trusted_pair_automatic_admission": False},
              "input_hashes": {"batch_manifest": sha256(args.batch / "expanded_batch_manifest.csv"),
                               "d1": sha256(args.d1), "c1_review": sha256(args.review),
                               "v2_manifest": sha256(args.v2_batch / "expanded_batch_manifest.csv")}, "pairs": []}
    args.output.mkdir(parents=True, exist_ok=False)
    save_new(args.output / "d1_interpretation_and_mask_audit.json", audit_d1_masks(args, manifest, d1))
    for sid, version, reason in SELECTION:
        row = manifest[sid]
        if row["split"].lower() != "train":
            raise ValueError("TRAIN only")
        frozen_image, annotation_path = args.train / row["image_path"], args.train / row["json_path"]
        if sha256(frozen_image) != row["source_image_sha256"] or sha256(annotation_path) != row["source_json_sha256"]:
            raise ValueError("TRAIN source/annotation identity mismatch")
        mask_row = row if version == "V1" else v2_manifest[sid]
        mask_path = (args.batch if version == "V1" else args.v2_batch) / mask_row["mask_path"]
        if sha256(mask_path) != mask_row["mask_sha256"] or mask_row["mask_sha256"] != row["mask_sha256"]:
            raise ValueError("Repair Mask identity differs from frozen selected Polygon mask")
        root = args.c0 if version == "V1" else args.c2
        folder = root / sid
        diag = load(folder / "geometry_diagnostics.json")
        raw_path = folder / "raw_output.png"
        provider_path = (args.generation if version == "V1" else args.v2_generation) / "live" / sid / "raw_output.png"
        if sha256(provider_path) != diag["raw_output_sha256"] or diag["source_image_sha256"] != row["source_image_sha256"]:
            raise ValueError("Saved raw/source identity mismatch")
        original, raw = read_rgb(folder / "original.png"), read_rgb(raw_path)
        if not np.array_equal(raw, read_rgb(provider_path)):
            raise ValueError("Raw review render differs from provider output pixels")
        if not np.array_equal(original, read_rgb(frozen_image)):
            raise ValueError("Source render pixel identity mismatch")
        record = d1_map[sid, version]
        if record["source_sha256"] != sha256(folder / "original.png") or record["output_sha256"] != sha256(raw_path):
            raise ValueError("D1 variant hash mismatch")
        size = [original.shape[1], original.shape[0]]
        target = read_target_mask(mask_path, tuple(size))
        annotation = load(annotation_path)
        if [annotation["imageWidth"], annotation["imageHeight"]] != size:
            raise ValueError("Annotation dimensions mismatch")
        polygon_ids = row["selected_polygon_ids"].split("|")
        polygons = [annotation["shapes"][int(pid.rsplit("_", 1)[1])]["points"] for pid in polygon_ids]
        selected = np.zeros(target.shape, np.uint8)
        for pid, points in zip(polygon_ids, polygons):
            shape = annotation["shapes"][int(pid.rsplit("_", 1)[1])]
            if shape["shape_type"] != "polygon" or not pid.startswith(sid + "_"):
                raise ValueError("Selected Polygon identity mismatch")
            cv2.fillPoly(selected, [np.int32(np.round(points))], 255)
        # Rasterizer conventions may differ by <=1 edge pixel; never replace frozen Mask.
        disagreement = int(np.count_nonzero((selected > 0) != (target > 0)))
        pair_id = f"{sid}_{version}"
        dest = args.output / pair_id
        dest.mkdir()
        for name in ("original.png", "raw_output.png", "aligned_output.png", "hard_composite.png", "feather_composite.png"):
            if (folder / name).exists():
                if name in diag.get("derived_sha256", {}) and sha256(folder / name) != diag["derived_sha256"][name]:
                    raise ValueError("Historical derived image changed")
                shutil.copyfile(folder / name, dest / name)
        cv2.imwrite(str(dest / "full_repair_mask.png"), target)
        overlay = original.copy()
        overlay[target > 0] = (.55 * overlay[target > 0] + .45 * np.array([255, 40, 90])).astype(np.uint8)
        write_rgb(dest / "mask_overlay.png", overlay)
        methods = {"SAVED_C0" if version == "V1" else "SAVED_C2": {
            "matrix": saved_matrix(diag), "origin": "HISTORICAL_SELECTED_TRANSFORM",
            "method": diag["mapping_method"], "diagnostic_sha256": sha256(folder / "geometry_diagnostics.json")}}
        for feature, value in record["methods"].items():
            for model, v in value["models"].items():
                methods[f"D1_{feature}_{model}"] = {"matrix": v.get("matrix_output_to_source"),
                    "origin": "D1_FULL_AUTOMATIC_MATCH_REFIT_NOT_HOLDOUT_FOLD_MATRIX", "holdout_diagnostic": v["holdout"]}
        for method in methods.values():
            method["coverage"] = coverage(method["matrix"], [raw.shape[1], raw.shape[0]], target)
        slots = [{"id": f"{pid}_N{i}", "group": "NEAR", "polygon_id": pid}
                 for pid in polygon_ids for i in range(1, 4)]
        slots += [{"id": f"F{i}", "group": "FAR", "polygon_id": ""} for i in range(1, 4)]
        clipped_path = folder / "valid_target_mask.png"
        clipped = cv2.imread(str(clipped_path), cv2.IMREAD_GRAYSCALE) if clipped_path.exists() else None
        pair = {"pair_id": pair_id, "source_id": sid, "version": version, "class": row["class"],
                "group_id": row["group_id"], "selection_reason": reason, "source_size": size,
                "output_size": [raw.shape[1], raw.shape[0]], "polygon_ids": polygon_ids,
                "polygon_points": polygons, "slots": slots, "near_radius_px": max(20, round(math.hypot(*size) * .075)),
                "source_image_sha256": row["source_image_sha256"], "source_json_sha256": row["source_json_sha256"],
                "raw_render_sha256": sha256(raw_path), "provider_raw_sha256": sha256(provider_path),
                "raw_pixel_identity_verified": True, "repair_mask_sha256": sha256(mask_path),
                "mask_polygon_raster_disagreement_pixels": disagreement,
                "target_pixels": int(np.count_nonzero(target)),
                "old_valid_target_omitted_pixels": int(np.count_nonzero((target > 0) & (clipped == 0))) if clipped is not None else None,
                "historical_geometry": diag["geometry_status"],
                "prior_visual": visual[sid] if version == "V1" else {
                    "user_conversation_acceptance": "ACCEPTED_IN_USER_CONTEXT", "formal_pair_review": "PENDING"},
                "methods": methods, "assets": {p.name: sha256(p) for p in dest.iterdir()}}
        # Preserve historical compositor evidence; report full-mask omissions separately.
        if (dest / "hard_composite.png").exists():
            pair["composite_invariant"] = {"outside_full_mask_changes": {
                name: int(np.count_nonzero(np.any(read_rgb(dest / name) != original, axis=2) & (target == 0)))
                for name in ("hard_composite.png", "feather_composite.png")},
                "interpretation": "COMPOSITOR_INVARIANT_ONLY_NOT_GEOMETRY_EVIDENCE"}
        bundle["pairs"].append(pair)
    bundle["bundle_id"] = canonical_hash(bundle)
    save_new(args.output / "benchmark.json", bundle)
    review = initial_review(bundle)
    save_new(args.output / "review_initial.json", review)
    save_new(args.output / "evaluation_pending.json", evaluate(bundle, review))
    template = (HERE / "g3d2_review.html").read_text(encoding="utf-8")
    content = json.dumps(bundle, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__BUNDLE_JSON__", content).replace("__REVIEW_SCRIPT__", (HERE / "g3d2_review.js").read_text(encoding="utf-8"))
    (args.output / "index.html").write_text(page, encoding="utf-8")
    log = {"status": "PENDING_HUMAN_LANDMARKS_AND_PAIR_REVIEW", "bundle_id": bundle["bundle_id"],
           "pair_count": len(bundle["pairs"]), "source_count": len({p["source_id"] for p in bundle["pairs"]}),
           "class_counts": dict(collections.Counter(p["class"] for p in bundle["pairs"])),
           "point_slots": sum(len(p["slots"]) for p in bundle["pairs"]), "human_points": 0,
           "trusted_pair_pass": 0, "generator_sha256": sha256(__file__),
           "review_js_sha256": sha256(HERE / "g3d2_review.js")}
    save_new(args.output / "build_log.json", log)
    print(json.dumps(log, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for name in ("train", "batch", "v2-batch", "generation", "v2-generation", "c0", "c2", "d1", "review", "output"):
        b.add_argument("--" + name, type=Path, required=True)
    e = sub.add_parser("evaluate")
    for name in ("benchmark", "review", "output"):
        e.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        bundle = load(args.benchmark)
        expected = bundle.pop("bundle_id")
        if canonical_hash(bundle) != expected:
            raise ValueError("Benchmark changed after freezing")
        bundle["bundle_id"] = expected
        save_new(args.output, evaluate(bundle, load(args.review)))
        print("Evaluation written; no model fit and no Trusted Pair admission")


if __name__ == "__main__":
    main()
