"""Read-only landmark CSV audit; exploratory fixed-transform TRE, no quality recovery.

CSV does not carry the bundle hash or Pair-quality decisions. Do not manufacture
a review JSON or regard structural compatibility as cryptographic export binding.
"""
import argparse
import collections
import csv
import hashlib
import io
import math
from pathlib import Path

import cv2
import numpy as np

from g3d2_benchmark import POINT_STATES, canonical_hash, load, save_new, summaries

BUNDLE_ID = "87112e0ea76b8f997f9be43b0e03480cedf4e8af45c38268174d9a566d7f7c69"
FIELDS = ["pair_id", "reviewer", "point_id", "group", "polygon_id", "status", "label",
          "source_x", "source_y", "output_x", "output_y", "uncertainty_px", "notes"]
CLOSED = "G3D2_CLOSED_BOUNDED_PARTIAL_HUMAN_EVIDENCE"


def parse_csv(raw, bundle):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    if reader.fieldnames != FIELDS:
        raise ValueError("CSV header does not match frozen D2 point export")
    expected = {(p["pair_id"], s["id"]): (p, s) for p in bundle["pairs"] for s in p["slots"]}
    result = {}
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed CSV row")
        key = row["pair_id"], row["point_id"]
        if key not in expected or key in result:
            raise ValueError("Unknown or duplicate pair/point identity")
        pair, slot = expected[key]
        if row["group"] != slot["group"] or row["polygon_id"] != slot["polygon_id"]:
            raise ValueError("Frozen point region/Polygon identity differs")
        if row["status"] not in POINT_STATES:
            raise ValueError("Unknown point status")
        parsed = dict(row)
        for side in ("source", "output"):
            values = [row[side + "_x"], row[side + "_y"]]
            if all(v == "" for v in values):
                parsed[side] = None
            else:
                if any(v == "" for v in values):
                    raise ValueError("Partial coordinate")
                xy = [float(v) for v in values]
                if any(not math.isfinite(v) or not 0 <= v < pair[side + "_size"][i]
                       for i, v in enumerate(xy)):
                    raise ValueError("Out-of-bounds or non-finite coordinate")
                parsed[side] = xy
        u = float(row["uncertainty_px"]) if row["uncertainty_px"] != "" else None
        if u is not None and (not math.isfinite(u) or u < 0):
            raise ValueError("Invalid uncertainty")
        parsed["uncertainty"] = u
        if row["status"] == "CONFIRMED" and (not row["reviewer"].strip() or not row["label"].strip()
                or parsed["source"] is None or parsed["output"] is None or u is None):
            raise ValueError("Incomplete confirmation fields")
        result[key] = parsed
    if set(result) != set(expected):
        raise ValueError("CSV must preserve all frozen Pair and point slots")
    return result


def audit(bundle, points, csv_sha):
    results = []
    for pair in bundle["pairs"]:
        accepted, exclusions, confirmed = [], [], 0
        slots = {s["id"]: s for s in pair["slots"]}
        for slot in pair["slots"]:
            p = points[pair["pair_id"], slot["id"]]
            if p["status"] != "CONFIRMED":
                continue
            confirmed += 1
            distances = {pid: cv2.pointPolygonTest(np.float32(poly), tuple(p["source"]), True)
                         for pid, poly in zip(pair["polygon_ids"], pair["polygon_points"])}
            normal = all(v < -2 for v in distances.values())
            region = (-pair["near_radius_px"] <= distances[slot["polygon_id"]] < -2
                      if slot["group"] == "NEAR" else
                      all(v < -pair["near_radius_px"] for v in distances.values()))
            reasons = []
            if not normal:
                reasons.append("NOT_OUTSIDE_ALL_SELECTED_POLYGONS_BY_2PX")
            if not region:
                reasons.append(slot["group"] + "_REGION_MISMATCH")
            if p["uncertainty"] > bundle["protocol"]["max_uncertainty_px"]:
                reasons.append("UNCERTAINTY_EXCEEDS_FROZEN_LIMIT")
            if not reasons and any(np.linalg.norm(np.asarray(p[side]) - q[side]) <= 4
                                   for q in accepted for side in ("source", "output")):
                reasons.append("DUPLICATE_OR_NEAR_DUPLICATE_LANDMARK")
            if reasons:
                exclusions.append({"point_id": slot["id"], "reasons": reasons,
                                   "source_polygon_signed_distances_px": distances})
            else:
                accepted.append(p)
        near = {pid: sum(p["polygon_id"] == pid and p["group"] == "NEAR" for p in accepted)
                for pid in pair["polygon_ids"]}
        far = sum(p["group"] == "FAR" for p in accepted)
        complete = far >= 3 and all(n >= 3 for n in near.values())
        methods = {}
        for name, value in pair["methods"].items():
            if value["matrix"] is None:
                methods[name] = {"status": "UNVERIFIED_NO_SAVED_TRANSFORM"}
                continue
            matrix = np.asarray(value["matrix"])
            errors = np.array([np.linalg.norm(matrix[:, :2] @ p["output"] + matrix[:, 2] - p["source"])
                               for p in accepted])
            methods[name] = {"status": "EXPLORATORY_PARTIAL_POINTS_ONLY" if len(errors)
                             else "NO_VALID_POINTS", "global": summaries(errors),
                             "groups": {g: summaries([e for e, p in zip(errors, accepted) if p["group"] == g])
                                        for g in ("NEAR", "FAR")},
                             "per_polygon": {pid: summaries([e for e, p in zip(errors, accepted)
                                                             if p["polygon_id"] == pid]) for pid in near}}
        results.append({"pair_id": pair["pair_id"], "planned_slots": len(slots),
                        "confirmed": confirmed, "rule_valid": len(accepted), "pending": sum(
                            points[pair["pair_id"], sid]["status"] == "PENDING" for sid in slots),
                        "valid_near_per_polygon": near, "valid_far": far,
                        "complete_point_coverage": complete, "exclusions": exclusions,
                        "methods": methods})
    return {"status": CLOSED, "formal_tre_gate": "NOT_ESTABLISHED", "trusted_pair_pass": 0,
            "mechanical_identity": "NOT_INDEPENDENTLY_EXPERT_VERIFIED", "bundle_id": bundle["bundle_id"],
            "csv_sha256": csv_sha, "csv_bundle_hash_embedded": False,
            "csv_binding_evidence": "EXACT_SLOT_REGION_SET_AND_COORDINATE_BOUNDS_ONLY",
            "pair_count": len(results), "planned_slots": len(points),
            "status_counts": dict(collections.Counter(p["status"] for p in points.values())),
            "confirmed_points": sum(r["confirmed"] for r in results),
            "rule_valid_points": sum(r["rule_valid"] for r in results),
            "confirmed_excluded": sum(len(r["exclusions"]) for r in results),
            "complete_evaluable_pairs": sum(r["complete_point_coverage"] for r in results),
            "pair_quality_decisions": "NOT_AVAILABLE_IN_CSV_NOT_RECONSTRUCTED", "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "csv", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    bundle = load(args.benchmark)
    claimed = bundle.pop("bundle_id")
    if claimed != BUNDLE_ID or canonical_hash(bundle) != claimed:
        raise ValueError("Frozen benchmark identity mismatch")
    bundle["bundle_id"] = claimed
    raw = args.csv.read_bytes()
    csv_sha = hashlib.sha256(raw).hexdigest()
    points = parse_csv(raw, bundle)
    evidence = audit(bundle, points, csv_sha)
    args.output.mkdir(parents=True, exist_ok=False)
    # Byte-identical archival copy; never change source CSV or any old result.
    with (args.output / "g3d2_landmarks_frozen.csv").open("xb") as stream:
        stream.write(raw)
    save_new(args.output / "partial_tre_audit_private.json", evidence)
    sanitized = {k: v for k, v in evidence.items() if k != "results"}
    sanitized["results"] = [{k: v for k, v in r.items() if k != "exclusions"} for r in evidence["results"]]
    save_new(args.output / "sanitized_closeout_summary.json", sanitized)
    if args.csv.read_bytes() != raw:
        raise RuntimeError("Source CSV changed during audit")
    print({k: v for k, v in evidence.items() if k != "results"})


if __name__ == "__main__":
    main()
