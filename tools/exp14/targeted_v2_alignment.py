"""Process A6 responses and attempt supported registration of previously returned Raw."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from tools.exp13_cloud.g3 import align_g3b2_batch as alignment
from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.postprocess_large_scale_repair import build_candidate, candidates, finished_candidates, write_image
from tools.exp14.prepare_large_scale_repair import hash_file, load_json
from tools.exp14.run_large_scale_repair import recover_responses
from tools.exp14.targeted_v2 import frozen_plan, target_raster


def enhanced_matches(original, raw, target):
    source_size = original.shape[1::-1]
    raw_size = raw.shape[1::-1]
    equalize = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(contrastThreshold=0.025)

    def features(image, mask):
        allowed = alignment.support_mask(image, mask)
        gray = equalize.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        return sift.detectAndCompute(gray, allowed)

    keys_a, desc_a = features(original, target)
    keys_b, desc_b = features(raw, cv2.resize(target, raw_size, interpolation=cv2.INTER_NEAREST))
    pairs = alignment.collect_matches(keys_a, desc_a, keys_b, desc_b, source_size, raw_size)
    fits = [fit for fit, _ in alignment.fit_models(pairs, raw_size, source_size)]
    if abs((source_size[0] / source_size[1]) / (raw_size[0] / raw_size[1]) - 1) <= 0.005:
        matrix = alignment.resize_candidate(source_size, raw_size)
        fits.append(alignment.validate_resize_candidate(pairs, matrix, source_size, raw_size))
    for fit in fits:
        fit["method"] = "CLAHE_" + fit["method"]
    return fits, len(pairs)


def fit_variant(original, raw, target, folder, prefer_support=False, *, old=None, old_sha=None):
    fits, selected, matches, aspect = alignment.registration_candidates(original, raw, target)
    extra_matches = 0
    if selected is None or old is not None:
        extra, extra_matches = enhanced_matches(original, raw, target)
        fits += extra
    source_size, raw_size = original.shape[1::-1], raw.shape[1::-1]
    bright = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) > 12

    def support(fit):
        matrix = np.asarray(fit["matrix_output_to_source"], np.float64)
        valid = alignment.valid_support(matrix, raw_size, source_size)
        return int(np.count_nonzero((valid == 0) & (target > 0))), int(np.count_nonzero((valid == 0) & bright))

    eligible = [fit for fit in fits if fit["status"] == "GEOMETRY_CANDIDATE"]
    for fit in eligible:
        fit["unsupported_target_pixels"], fit["unsupported_bright_pixels"] = support(fit)
    if old and old.get("selected_fit"):
        prior_support = support(old["selected_fit"])
        # Retain the old registration unless a validated map improves its usable support.
        eligible = [fit for fit in eligible if (fit["unsupported_target_pixels"], fit["unsupported_bright_pixels"])
                    < prior_support]
    picked = min(eligible, key=lambda fit: (fit["unsupported_target_pixels"], fit["unsupported_bright_pixels"],
                                            fit["median_reprojection_error_source_px"])) if eligible else None
    diagnostics = {"geometry_status": "GEOMETRY_CANDIDATE" if picked else "ALIGNMENT_UNVERIFIED",
                   "mapping_method": picked["method"] if picked else "NONE", "match_count": matches,
                   "enhanced_match_count": extra_matches, "aspect_ratio_difference": aspect,
                   "candidate_diagnostics": fits, "selected_fit": picked,
                   "parent_geometry_sha256": old_sha, "parent_geometry": old}
    if picked is None:
        diagnostics["reason"] = "NO_VERIFIED_SUPPORT_IMPROVEMENT" if old and old.get("selected_fit") else "GEOMETRY_SCREEN_FAILED"
        return None, None, diagnostics
    matrix = np.asarray(picked["matrix_output_to_source"], np.float64)
    aligned = cv2.warpAffine(raw, matrix, source_size, flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = alignment.valid_support(matrix, raw_size, source_size)
    target_valid, hard, feather = alignment.composites(original, aligned, valid, target)
    for name, image in (("aligned", aligned), ("valid", valid), ("valid_target", target_valid),
                        ("hard", hard), ("feather", feather)):
        write_image(folder / f"{name}.png", image)
    return aligned, valid, diagnostics


def complete_attempt(root, variant, entry, source, **kwargs):
    # Keep every attempt and give an incomplete attempt a durable terminal receipt.
    receipt = root / "postprocess_receipts" / f"{variant}.json"
    if receipt.exists():
        return
    try:
        record = build_candidate(root, variant, entry, source, **kwargs)
        status = record["geometry_status"] if record else "UNCHANGED"
        error = ""
    except (OSError, ValueError, cv2.error) as exc:
        record = None
        status, error = "LOCAL_POSTPROCESS_ERROR", str(exc)
    if record and kwargs.get("parent") and not record["aligned_sha256"]:
        status = "REGISTRATION_NOT_RESCUED"
    g3.write_json_exclusive(receipt, {"status": status, "record": record, "error": error})
    print(variant, status, flush=True)


def process(root, follow):
    cv2.setNumThreads(2)
    scope, _, _ = frozen_plan(root)
    parent_root = Path(scope["parent_root"])
    parents = finished_candidates(parent_root)
    parent_scope = load_json(parent_root / "scope.json")
    plans = {row["source_id"]: row for row in scope["routes"]}
    requests = {row["variant_id"]: row for row in scope["requests"]}
    inspected = set()
    while True:
        recover_responses(root, scope, inspected)
        done = {p.stem for p in (root / "postprocess_receipts").glob("*.json")}
        for variant, entry in candidates(root, {**scope, "historical_variants": {}}, done, running=follow).items():
            request = requests[variant.split("_OUT", 1)[0]]
            sid = entry["source_id"]
            source = scope["sources"][sid]
            shapes = load_json(Path(source["annotation"]))["shapes"]
            repair, _ = target_raster(source, shapes, plans[sid]["repair_targets"])
            base = g3.safe_file(root, request["batch_image_path"])
            hash_file(base, request["source_image_sha256"])
            complete_attempt(root, variant, entry, source, fit=fit_variant, repair_target=repair, base_path=base)
        # Complete first the three reusable V2 outputs, followed by ordinary CPU work.
        queued = sorted((p for p in plans.values() if p["alignment_targets"]),
                        key=lambda plan: not plan["existing_version_reuse"])
        for plan in queued:
            parent = parents[plan["parent_variant"]]
            sid = plan["source_id"]
            folder = parent_root / parent["asset_directory"]
            variant = f"{parent['variant_id']}_A6ALIGN"
            if variant in done:
                continue
            geometry = folder / "geometry.json"
            old_sha = hash_file(geometry, parent["file_sha256"].get("geometry.json"))
            old = load_json(geometry)
            source = scope["sources"][sid]

            history = parent_scope["historical_variants"].get(parent["variant_id"])
            mask = Path(history["mask"]) if history else parent_root / "masks" / f"{parent['variant_id'].split('_OUT',1)[0]}.png"
            entry = {**parent, "raw": str(folder / parent["raw"]), "mask": str(mask), "derived": {}}

            def rescue(original, raw, target, destination, prefer_support):
                return fit_variant(original, raw, target, destination, old=old, old_sha=old_sha)

            complete_attempt(root, variant, entry, source, parent=parent, fit=rescue)
        launched = list((root / "logs").glob("start_*.json"))
        stopped = list((root / "logs").glob("stop_*.json"))
        if not follow or (launched and len(launched) == len(stopped)):
            done = {p.stem for p in (root / "postprocess_receipts").glob("*.json")}
            if not candidates(root, {**scope, "historical_variants": {}}, done):
                break
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    process(args.root.resolve(), args.follow)


if __name__ == "__main__":
    main()
