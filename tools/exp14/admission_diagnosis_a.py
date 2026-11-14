"""Diagnose A9 supervision admission losses without changing frozen assets."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.exp13_cloud.g3.g3_batch_executor import read_csv, require, write_json_exclusive
from tools.exp14.final_dataset import write_csv
from tools.exp14.final_dataset_identity import INPUTS
from tools.exp14.final_dataset_selection import a_proposals
from tools.exp14.prepare_large_scale_repair import hash_file, load_json
from tools.exp14.targeted_v2 import target_raster

COUNTERFACTUAL = "COUNTERFACTUAL_NOT_ADMITTED"
EDGE_FLAGS = frozenset({"VISIBLE_SOURCE_PIXELS_UNSUPPORTED", "NONFULL_FRAME_VALID_MASK"})
RAW_BYPASSES = EDGE_FLAGS | {"MASK_SUPPORT_INSUFFICIENT", "ALIGNMENT_UNVERIFIED", "STORED_MATRIX_UNAVAILABLE"}
AI_FLAGS = frozenset({"HUMAN_AI_WHOLE_DECISION_CONFLICT", "HUMAN_AI_POLYGON_DECISION_CONFLICT"})
MAIN_ORDER = (
    "IDENTITY_QUARANTINE", "ALIGNMENT_UNVERIFIED", "VISIBLE_SOURCE_PIXELS_UNSUPPORTED",
    "NONFULL_FRAME_VALID_MASK", "MASK_SUPPORT_INSUFFICIENT", "STORED_MATRIX_UNAVAILABLE",
    "UNSELECTED_ANNOTATED_DEFECT", "POLYGON_PASS_INCOMPLETE",
)
FOCUS = (
    *MAIN_ORDER, *sorted(AI_FLAGS), "CAMERA_UI_CHANGED", "MECHANICAL_STRUCTURE_CONFLICT",
    "MECHANICAL_STRUCTURE_OR_OCCLUSION", "MECHANICAL_STRUCTURE_ERROR", "NEW_VISUAL_ARTIFACT",
    "RESIDUAL_MARKS_NEED_SEVEN_CLASS_REVIEW", "VISIBLE_RESIDUAL_OR_GEOMETRY_EVIDENCE",
    "WHOLE_IMAGE_REVIEW_REQUESTED_IN_NOTES",
)


def risk_category(reason):
    if reason in RAW_BYPASSES:
        return "REGISTRATION_EDGE_OR_PIXEL_SUPPORT"
    if reason in {"UNSELECTED_ANNOTATED_DEFECT", "POLYGON_PASS_INCOMPLETE"}:
        return "LABEL_COMPLETENESS"
    if reason in AI_FLAGS:
        return "AI_HUMAN_DECISION_CONFLICT"
    if reason == "IDENTITY_QUARANTINE":
        return "IDENTITY"
    return "VISUAL_OR_MECHANICAL_QUALITY"


def assets_rgb(asset):
    path = Path(asset["path"])
    hash_file(path, asset["sha256"])
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def nearest_border_distance(h, w):
    x = np.arange(w, dtype=np.int32)
    y = np.arange(h, dtype=np.int32)
    return np.minimum(np.minimum(x, w - 1 - x)[None, :], np.minimum(y, h - 1 - y)[:, None])


def bounds(mask):
    if not np.any(mask):
        return None
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def crop_margin(invalid):
    """Find the largest retained rectangle after trimming only shallow margins."""
    h, w = invalid.shape
    allowance = max(1, int(np.floor(0.02 * min(w, h))))
    sums = np.pad(np.cumsum(invalid, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    best = None
    largest = -1
    for left in range(allowance + 1):
        for right in range(allowance + 1):
            contaminated = np.flatnonzero(sums[:, w - right] > sums[:, left])
            if np.any((contaminated >= allowance) & (contaminated < h - allowance)):
                continue
            top = max(contaminated[contaminated < allowance], default=-1) + 1
            bottom = min(contaminated[contaminated >= h - allowance], default=h)
            area = (w - right - left) * (bottom - top)
            if area > largest:
                largest = area
                best = [int(left), int(top), int(w - right), int(bottom)]
    return best


def edge_metrics(original, invalid, gt):
    """Measure mask loss and a proposed crop without creating a new image."""
    h, w = invalid.shape
    size = h * w
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    visible = gray > 12  # same brightness predicate as A9; a proxy, not a part mask
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    structure_proxy = cv2.magnitude(gx, gy) > 32
    span = nearest_border_distance(h, w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), connectivity=8)
    border_labels = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])) - {0}
    inner_components = [i for i in range(1, n) if i not in border_labels]
    box = crop_margin(invalid)
    removed = None
    if box:
        x0, y0, x1, y1 = box
        removed = np.ones((h, w), dtype=bool)
        removed[y0:y1, x0:x1] = False
    target_distance = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if np.any(gt) else None
    cropped_gt = int(np.count_nonzero(removed & gt)) if removed is not None else None
    cropped_visible = int(np.count_nonzero(removed & visible)) if removed is not None else None
    cropped_structure_proxy = int(np.count_nonzero(removed & structure_proxy)) if removed is not None else None
    hidden_gt = int(np.count_nonzero(invalid & gt))
    visible_invalid = int(np.count_nonzero(invalid & visible))
    invalid_count = int(np.count_nonzero(invalid))
    depth = int(span[invalid].max()) if invalid_count else 0
    min_clearance = float(target_distance[invalid].min()) if invalid_count and target_distance is not None else None
    peripheral_dark = (
        invalid_count > 0 and not inner_components and depth < 0.02 * min(w, h) and
        invalid_count <= 0.02 * size and hidden_gt == 0 and visible_invalid == 0 and
        min_clearance is not None and min_clearance >= 0.02 * min(w, h)
    )
    crop_possible = peripheral_dark and removed is not None and int(removed.sum()) <= 0.05 * size and cropped_gt == 0 and cropped_visible == 0
    return {
        "source_size_wh": [w, h], "invalid_pixels": invalid_count, "invalid_fraction": invalid_count / size,
        "invalid_bbox_xyxy": bounds(invalid), "visible_original_invalid_pixels": visible_invalid,
        "visible_invalid_bbox_xyxy": bounds(invalid & visible),
        "original_gradient_invalid_pixels": int(np.count_nonzero(invalid & structure_proxy)),
        "gt_overlap_pixels": hidden_gt, "min_invalid_to_gt_distance_px": min_clearance,
        "deepest_invalid_pixel_from_frame_px": depth, "inner_components": len(inner_components),
        "largest_inner_component_pixels": max([int(stats[i, cv2.CC_STAT_AREA]) for i in inner_components], default=0),
        "proposed_dark_margin_crop_xyxy": box,
        "crop_removed_pixels": int(removed.sum()) if removed is not None else None,
        "crop_removed_fraction": int(removed.sum()) / size if removed is not None else None,
        "crop_removed_gt_pixels": cropped_gt, "crop_removed_visible_pixels": cropped_visible,
        "crop_removed_gradient_proxy_pixels": cropped_structure_proxy,
        "crop_leaves_invalid_pixels": int(np.count_nonzero(invalid & ~removed)) if removed is not None else None,
        "a1_dark_margin_crop_screen_pass": bool(crop_possible),
        "a1_original_dark_edge_restore_screen_pass": peripheral_dark,
        "a1_dark_margin_screen_pass": peripheral_dark,
        "mechanical_foreground": "VISUAL_CONFIRMATION_REQUIRED",
    }


def mask_rgb_alpha(path, expected_sha):
    hash_file(path, expected_sha)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))[..., 3] == 0


def mask_rasters(source, shapes):
    return {pid: np.asarray(target_raster(source, shapes, [pid])[0]) > 0 for pid in source["all_polygon_ids"]}


def bounded_metrics(record, audit, original, aligned, valid, rasters, bounded):
    request_path = Path(audit["origin_paths"]["request_path"])
    hash_file(request_path, record["request_sha256"])
    request = load_json(request_path)
    geometry_asset = audit["assets"]["geometry.json"]
    geometry_path = Path(geometry_asset["path"])
    hash_file(geometry_path, geometry_asset["sha256"])
    geometry = load_json(geometry_path)["bounded_image"]
    history_root = next(p for p in request_path.parents
                        if p.name.startswith(("exp14_a6_targeted_v2_", "exp14_a7_targeted_v3_", "exp14_a8_final_v4_")))
    input_asset = {"path": str(history_root / request["batch_image_path"]), "sha256": request["source_image_sha256"]}
    base = assets_rgb(input_asset)
    require(base.shape == bounded.shape == aligned.shape == original.shape, "Bounded pixel grid differs")
    require(geometry["base_sha256"] == input_asset["sha256"], "Bounded base hash differs")
    editable = mask_rgb_alpha(Path(audit["origin_paths"]["mask"]), record["mask_sha256"])
    effective = editable & valid
    matches_replay = bool(np.array_equal(bounded[effective], aligned[effective]) and np.array_equal(bounded[~effective], base[~effective]))
    require(matches_replay and geometry["edited_pixels"] == int(effective.sum()), "Bounded disagrees with saved pixels")
    protected_ids = set(request.get("preserved_polygon_ids", []))
    gt_status = []
    for pid, mask in rasters.items():
        pixels = int(mask.sum())
        edited = int(np.count_nonzero(mask & effective))
        unchanged = int(np.count_nonzero(np.all(bounded == original, axis=2) & mask))
        preserved = pid in protected_ids and bool(request.get("parent_aligned_sha256") or request.get("pixel_sources"))
        gt_status.append({"polygon_id": pid, "pixels": pixels, "edit_supported_pixels": edited,
                          "pixel_equal_to_original": unchanged, "protected_prior_provenance": preserved,
                          "fully_accounted_by_edit_or_prior": pixels > 0 and (edited == pixels or preserved)})
    # Proxy for a local hard seam: luma jumps crossing the edit boundary which
    # were not present in the saved base. No automatic semantic clearance.
    gray_base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.int16)
    gray_final = cv2.cvtColor(bounded, cv2.COLOR_RGB2GRAY).astype(np.int16)
    jumps = []
    for axis in (0, 1):
        seam = np.abs(np.diff(effective.astype(np.int16), axis=axis)) > 0
        delta = np.maximum(np.abs(np.diff(gray_final, axis=axis)) - np.abs(np.diff(gray_base, axis=axis)), 0)
        jumps.extend(delta[seam].tolist())
    return {"bounded": audit["assets"]["bounded.png"], "base": input_asset,
            "preserves_base_outside_effective_edit": matches_replay, "edge_unsupported_in_aligned_pixels": int((~valid).sum()),
            "bounded_pixels_different_from_base_in_invalid_area": int(np.count_nonzero(np.any(bounded != base, axis=2) & ~valid)),
            "gt_coverage_by_polygon": gt_status,
            "all_gt_accounted": all(p["fully_accounted_by_edit_or_prior"] for p in gt_status),
            "seam_crossings": len(jumps), "seam_luma_jump_increment_p95": float(np.percentile(jumps, 95)) if jumps else 0,
            "seam_luma_jump_increment_gt20": sum(j > 20 for j in jumps),
            "mechanical_and_seam_status": "VISUAL_CONFIRMATION_REQUIRED"}


def prior_rules_order(row, source):
    reasons = set(row["reasons"])
    persisted_source_hold = source.get("hold")
    order = [*MAIN_ORDER, persisted_source_hold, *sorted(reasons - set(MAIN_ORDER) - {persisted_source_hold})]
    return [rule for rule in order if rule in reasons]


def removal_count(hold, rules):
    """Remove a rule counterfactually only within the same version."""
    return sum(any(set(v["reasons"]) <= rules for v in rows) for rows in hold.values())


def summarize_losses(hold, all_by_source, evidence_by_source, sources):
    reasons_by_source = {sid: set().union(*(set(v["reasons"]) for v in rows)) for sid, rows in hold.items()}
    all_reasons = sorted(set(FOCUS) | set().union(*reasons_by_source.values()))
    first, category, reason_sets = Counter(), Counter(), Counter()
    waterfall = []
    source_details = []
    for sid, rows in sorted(hold.items(), key=lambda pair: int(pair[0])):
        effective_version = min(rows, key=lambda row: (len(row["reasons"]), row["version"], row["variant_id"]))
        first_rule = prior_rules_order(effective_version, sources[sid])[0]
        first[first_rule] += 1
        category.update({risk_category(r) for r in reasons_by_source[sid]})
        reason_sets[tuple(sorted(reasons_by_source[sid]))] += 1
        variants = evidence_by_source[sid]
        pass_ids = sorted(item["record"]["variant_id"] for item in variants if item["human"]["decision"] == "PASS")
        other_pass = sorted(set(pass_ids) - {effective_version["variant_id"]})
        source_details.append({"source_id": sid, "diagnostic_variant": effective_version["variant_id"],
            "first_block": first_rule, "all_reasons": sorted(reasons_by_source[sid]), "version_reason_sets": [
                {"variant_id": row["variant_id"], "review_id": row["review_id"], "ordered_blockers": prior_rules_order(
                    row, sources[sid])} for row in rows],
            "other_human_overall_pass_variant_ids": other_pass,
            "other_full_a_eligible_variants": [r["variant_id"] for r in all_by_source[sid] if not r["reasons"]]})
    rules = []
    for r in all_reasons:
        alone = removal_count(hold, {r})
        rules.append({"rule": r, "category": risk_category(r), "source_count_any_version": sum(
            r in values for values in reasons_by_source.values()), "first_block_sources": first[r],
            "sources_lost_when_rule_used_alone": sum(all(r in v["reasons"] for v in rows) for rows in hold.values()),
            "source_only_cause_union_count": sum(values == {r} for values in reasons_by_source.values()),
            "recoverable_if_only_this_rule_removed": alone})
    remaining = {sid: rows for sid, rows in hold.items()}
    order = ["IDENTITY_QUARANTINE", "ALIGNMENT_UNVERIFIED", "VISIBLE_SOURCE_PIXELS_UNSUPPORTED",
             "NONFULL_FRAME_VALID_MASK", "MASK_SUPPORT_INSUFFICIENT", "STORED_MATRIX_UNAVAILABLE",
             "UNSELECTED_ANNOTATED_DEFECT", "POLYGON_PASS_INCOMPLETE"]
    order.extend(sorted(set(all_reasons) - set(order) - AI_FLAGS))
    order.extend(sorted(AI_FLAGS))
    # Version-based step elimination: a source is lost only when the last
    # version surviving all preceding rules is eliminated.
    active = {sid: list(rows) for sid, rows in all_by_source.items()}
    for step, r in enumerate(order, 1):
        incoming = len(active)
        active = {sid: [v for v in rows if r not in v["reasons"]] for sid, rows in active.items()}
        active = {sid: rows for sid, rows in active.items() if rows}
        waterfall.append({"step": step, "rule": r, "category": risk_category(r), "sources_before": incoming,
                          "sources_removed": incoming - len(active), "sources_after": len(active)})
    return {"rules": rules, "rule_category_incidence_nonexclusive": dict(category),
            "source_reason_combinations": [{"reasons": list(k), "source_count": n} for k, n in reason_sets.most_common()],
            "hold_sources_with_multiple_union_flags": sum(len(r) > 1 for r in reasons_by_source.values()),
            "hold_sources_with_no_single_rule_fix": sum(all(len(v["reasons"]) > 1 for v in rows) for rows in remaining.values()),
            "hold_sources_with_single_rule_fix": sum(any(len(v["reasons"]) == 1 for v in rows) for rows in remaining.values()),
            "hold_sources_with_other_human_overall_pass_variant": sum(bool(r["other_human_overall_pass_variant_ids"]) for r in source_details),
            "hold_sources_with_other_full_a_eligible_variant": sum(bool(r["other_full_a_eligible_variants"]) for r in source_details),
            "source_details": source_details,
            "count_semantics": {
                "first_block": "A9 evaluation order on the least-blocked human whole PASS variant (stable version/variant tiebreak).",
                "sources_lost_when_rule_used_alone": "All human whole-image PASS variants for a source are affected by this rule.",
                "source_only_cause_union_count": "Only this rule appears across any human whole-image PASS variant for the source.",
                "recoverable_if_only_this_rule_removed": "At least one same version becomes blocker-free if this single rule is removed."},
            "waterfall": waterfall}


def inspect_policies(hold, evidence_by_review, sources, source_info):
    detailed, queue = [], []
    pools = defaultdict(set)
    for sid, rows in hold.items():
        source = sources[sid]
        shapes = source_info[sid]["shapes"]
        rasters = mask_rasters(source, shapes)
        all_gt = np.logical_or.reduce(list(rasters.values()))
        for row in sorted(rows, key=lambda a: a["variant_id"]):
            item = evidence_by_review[row["review_id"]]
            record = item["record"]
            audit = item["audit"]
            assets = audit["assets"]
            reasons = set(row["reasons"])
            original = assets_rgb(assets[record["original"]])
            raw_ref = assets[record["raw"]]
            raw = assets_rgb(raw_ref)
            a1, a3 = None, None
            variants = {}
            if "valid.png" in assets:
                valid = assets_rgb(assets["valid.png"])[..., 0] > 0
                a1 = edge_metrics(original, ~valid, all_gt)
                require(a1["invalid_pixels"] == audit["invalid_pixels"] and
                        a1["visible_original_invalid_pixels"] == audit["visible_loss"], "Edge recount differs from verified audit")
                if "bounded.png" in assets:
                    aligned = assets_rgb(assets["aligned.png"])
                    bounded = assets_rgb(assets["bounded.png"])
                    a3 = bounded_metrics(record, audit, original, aligned, valid, rasters, bounded)
            policy_notes = {
                "A1": (reasons <= EDGE_FLAGS, a1 and a1["a1_dark_margin_screen_pass"] and reasons <= EDGE_FLAGS,
                       "Confirm dark margin, no clipped component, and a visually acceptable crop or original-dark-edge restoration."),
                "A2": (True, bool(reasons & RAW_BYPASSES and reasons <= RAW_BYPASSES),
                       "Re-review raw frame for all seven classes, geometry hallucination, scene content and UI/light/style shortcuts."),
                "A3": (a3 is not None, bool(a3 and a3["edge_unsupported_in_aligned_pixels"] and a3["all_gt_accounted"] and
                                            reasons <= EDGE_FLAGS),
                       "Review existing Bounded as independent evidence for all GT, real defect removal, geometry and seam quality."),
                "A4": (bool(reasons & AI_FLAGS), bool(reasons & AI_FLAGS and reasons <= AI_FLAGS),
                       "Arbitrate AI and human decisions on the same asset against the original with named visual evidence."),
            }
            for policy, (explorable, screened, confirm) in policy_notes.items():
                if not explorable:
                    continue
                if screened:
                    pools[policy].add(sid)
                variants[policy] = "AWAITING_ASSET_SPECIFIC_HUMAN_REVIEW" if screened else "RISK_EVIDENCE_ONLY"
                queue.append({"source_id": sid, "policy": policy, "variant_id": row["variant_id"], "review_id": row["review_id"],
                    "status": COUNTERFACTUAL, "screened_potential": bool(screened), "conditions_for_consideration": confirm,
                    "a9_blocking_reasons": sorted(reasons), "other_a9_rules_retained": sorted(reasons - (
                        RAW_BYPASSES if policy == "A2" else AI_FLAGS if policy == "A4" else EDGE_FLAGS)),
                    "aligned_asset": assets.get("aligned.png"), "raw_asset": raw_ref if policy == "A2" else None,
                    "bounded_asset": assets.get("bounded.png") if policy == "A3" else None,
                    "edge_metrics": a1 if policy in {"A1", "A3"} else None, "manual_review": "PENDING",
                    "formal_admission": False})
            detailed.append({"source_id": sid, "variant_id": row["variant_id"], "review_id": row["review_id"],
                "reasons": sorted(reasons), "status": COUNTERFACTUAL, "edge": a1, "policies": variants,
                "raw_risks": {"raw_asset": raw_ref, "raw_size_wh": [raw.shape[1], raw.shape[0]],
                              "source_size_wh": source["size"], "ai_reason": item.get("ai", {}).get("ai_reason"),
                              "seven_class_clearance": "NOT_VERIFIED_FOR_RAW_PIXEL_IDENTITY",
                              "structure_scene_ui_lighting_shortcuts": "VISUAL_CONFIRMATION_REQUIRED"},
                "bounded_metrics": a3})
    return detailed, queue, pools


def compare_policies(strict, hold, queue, pools):
    risks = {
        "A0": "Frozen full-frame Aligned and historical review rules.",
        "A1": "Technical candidates for audited original-dark-edge restoration or dark margin trim. No pixels created. Confirm retained structure, seams and labels on the independently reviewed result.",
        "A2": "Raw has its own unreviewed whole-image pixel identity; all classes, real structure, appearance and annotation coverage require inspection.",
        "A3": "A bit-exact saved composition is provenance evidence. Boundary seams, base-retained defects and content quality still need review.",
        "A4": "Remove only automatic AI label conflict, keeping independent evidence and a mandatory human tie-break.",
    }
    rows = []
    for policy in risks:
        explored = {r["source_id"] for r in queue if r["policy"] == policy}
        additions = len(pools[policy])
        ceiling = removal_count(hold, EDGE_FLAGS if policy in {"A1", "A3"} else
                                RAW_BYPASSES if policy == "A2" else AI_FLAGS) if policy != "A0" else 0
        if policy == "A3":
            ceiling = sum(any(r["bounded_alternative"] and set(r["reasons"]) <= EDGE_FLAGS for r in items)
                          for items in hold.values())
        rows.append({"policy": policy, "status": COUNTERFACTUAL, "existing_a9_candidate_sources": len(strict),
                     "exploratory_hold_source_pool": len(explored), "rule_only_increment_upper_bound": ceiling,
                     "technically_screened_increment_pending_review": additions,
                     "total_potential_candidates_including_unchanged_a0": len(strict) + additions,
                     "increment_requiring_human_asset_or_conflict_confirmation": additions,
                     "total_requiring_confirmation_including_a0_asset_selection": sum(
                         r["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION" for r in strict.values()) + additions,
                     "new_safely_confirmed_candidates": 0, "formal_admission_count": 0,
                     "evidence_boundary_and_risks": risks[policy]})
    return rows


def load_inputs(root):
    receipt = load_json(root / "handoff" / "delivery_receipt.json")
    relevant = ("audits/label_integrity_audit.json", "registry/a_candidate_manifest.csv",
                "registry/source_master_registry.csv", "inputs/input_hashes.json")
    for path in relevant:
        hash_file(root / path, receipt["private_sha256"][path])
    evidence = load_json(root / "audits" / "verified_evidence_reconciled.json")
    a9 = load_json(root / relevant[0])["records"]
    audits = {r["review_id"]: r for r in a9}
    current = {row["source_id"]: json.loads(row["recommended"]) for row in read_csv(root / relevant[1])[1]}
    source_index = {row["source_id"]: row for row in read_csv(root / relevant[2])[1]}
    by_review = {item["record"]["review_id"]: item for item in evidence["reviews"]}
    by_source = defaultdict(list)
    for item in evidence["reviews"]:
        by_source[item["record"]["source_id"]].append(item)
    frozen_reviews = {}
    for file, sha in INPUTS.items():
        path = root / "inputs" / file
        hash_file(path, sha)
        _, rows = read_csv(path)
        for row in rows:
            require(row["review_id"] not in frozen_reviews and row == by_review[row["review_id"]]["human"],
                    "Frozen review does not match A9 evidence cache")
            frozen_reviews[row["review_id"]] = row
    require(frozen_reviews.keys() == by_review.keys(), "Frozen review coverage differs")
    need_a = {rid for rid, row in frozen_reviews.items() if row["whole_image_normal"] == "YES" and row["decision"] == "PASS"}
    require(audits.keys() == need_a and len(audits) == len(a9), "A9 whole-image PASS rows differ from five confirmations")
    quarantined = {row["source_id"] for row in evidence["issues"]}
    all_by_source = {}
    for sid, source in evidence["sources"].items():
        rows = a_proposals(source, evidence["source_info"][sid], by_source[sid], sid in quarantined)
        if not rows:
            continue
        for row in rows:
            require({"source_id": sid, **row} == audits[row["review_id"]], "Recomputed policy blockers disagree with A9")
        all_by_source[sid] = [audits[r["review_id"]] for r in rows]
    hold = {sid: rows for sid, rows in all_by_source.items() if all(r["reasons"] for r in rows)}
    require(set(current) == all_by_source.keys() - hold.keys() and set(hold) == {
            sid for sid, row in source_index.items() if row["a_label_hold"] == "true"}, "Reproduced A9 source membership differs")
    return all_by_source, hold, by_source, by_review, current, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-selection", type=Path, required=True, help="Private JSON list of source/variant review IDs.")
    args = parser.parse_args()
    require(args.input.resolve() != args.output.resolve() and
            not args.output.resolve().is_relative_to(args.input.resolve()), "D0 output must not be in A9")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    all_by_source, hold, evidence_by_source, evidence_by_review, strict, evidence = load_inputs(args.input)
    report = summarize_losses(hold, all_by_source, evidence_by_source, evidence["sources"])
    details, queue, pools = inspect_policies(hold, evidence_by_review, evidence["sources"], evidence["source_info"])
    comparison = compare_policies(strict, hold, queue, pools)
    union = set().union(*pools.values())
    summary = {
        "human_whole_image_confirmed": len(all_by_source), "human_whole_image_confirmed_version_rows": sum(map(len, all_by_source.values())),
        "a0_current_candidates": len(strict), "a0_hold_sources": len(hold), "a0_hold_version_rows": sum(map(len, hold.values())),
        "a0_bound_aligned_candidates": sum(row["pixel_asset_state"] == "ALIGNED_REVIEW_ID_BOUND" for row in strict.values()),
        "a0_asset_selection_required": sum(row["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION" for row in strict.values()),
        "historical_a_rule_findings": {
            "local_ai_conflict_is_not_executed_by_a_rules": True,
            "local_ai_conflict_is_zero_as_a_saved_reason_but_exists_in_aligned_human_whole_pass_versions": sum(
                any(s in {"HOLD", "RETRY_REPAIR", "RETRY_ALIGNMENT"} for s in evidence_by_review[row["review_id"]]
                    .get("ai", {}).get("ai_polygon_decisions", {}).values()) for rows in hold.values() for row in rows),
            "a9_visible_unsupported_flag_implies_nonfull_frame_valid_mask": all(
                "NONFULL_FRAME_VALID_MASK" in r["reasons"] for rs in hold.values() for r in rs
                if "VISIBLE_SOURCE_PIXELS_UNSUPPORTED" in r["reasons"]),
            "whole_decision_only_blocker": next(r["recoverable_if_only_this_rule_removed"] for r in report["rules"]
                                                if r["rule"] == "HUMAN_AI_WHOLE_DECISION_CONFLICT")},
        "counterfactual_union_potential_increment": len(union),
        "counterfactual_union_also_baseline_candidates": 0,
        "safe_increment_verified_for_admission": 0,
        "counterfactual": comparison,
    }
    require(report["waterfall"][-1]["sources_after"] == len(strict), "Waterfall total disagrees with A0")
    selection = load_json(args.pilot_selection)["selection"]
    detail_index = {(r["source_id"], r["review_id"]): r for r in details}
    require(30 <= len(selection) <= 50 and len({r["source_id"] for r in selection}) == len(selection),
            "Pilot requires 30–50 distinct sources")
    pilot = [{**detail_index[(pick["source_id"], pick["review_id"])], "case_role": pick["case_role"]}
             for pick in selection]
    write_csv(args.output / "a_392_to_200_loss_waterfall.csv", report["waterfall"])
    write_csv(args.output / "a_policy_counterfactual_comparison.csv", comparison)
    write_csv(args.output / "a_recoverable_candidate_queue.csv", queue)
    write_json_exclusive(args.output / "a_192_root_cause_summary.json", {**summary, **report})
    write_json_exclusive(args.output / "a_hold_version_asset_geometry.json", {"status": COUNTERFACTUAL, "records": details})
    write_json_exclusive(args.output / "a_pilot_cases.json", {"status": COUNTERFACTUAL, "cases": pilot})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
