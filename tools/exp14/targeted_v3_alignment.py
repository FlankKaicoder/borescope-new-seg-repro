"""Register new A7 images and test a scale-normalized fit on existing Raw."""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from tools.exp13_cloud.g3 import align_g3b2_batch as geometry
from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.postprocess_large_scale_repair import candidates
from tools.exp14.prepare_large_scale_repair import PHASE, hash_file, load_json
from tools.exp14.run_large_scale_repair import recover_responses
from tools.exp14.targeted_v2_alignment import complete_attempt, fit_variant
from tools.exp14.targeted_v3 import A6, frozen_plan, target_raster


def normalized_matches(original, raw, target):
    source_size, raw_size = original.shape[1::-1], raw.shape[1::-1]
    scale = min(1.0, max(source_size) / max(raw_size))
    scaled_size = tuple(max(1, round(n * scale)) for n in raw_size)
    resized = cv2.resize(raw, scaled_size, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(contrastThreshold=0.025)
    equalize = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract(rgb, mask):
        gray = equalize.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        return sift.detectAndCompute(gray, geometry.support_mask(rgb, mask))

    original_keys, original_descriptors = extract(original, target)
    resized_target = cv2.resize(target, scaled_size, interpolation=cv2.INTER_NEAREST)
    raw_keys, raw_descriptors = extract(resized, resized_target)
    matches = geometry.collect_matches(original_keys, original_descriptors, raw_keys, raw_descriptors,
                                       source_size, scaled_size)
    for pair in matches:
        pair["output"] = [(value + 0.5) * raw_n / scaled_n - 0.5
                          for value, raw_n, scaled_n in zip(pair["output"], raw_size, scaled_size, strict=True)]
    fits = [fit for fit, _ in geometry.fit_models(matches, raw_size, source_size)]
    if abs((source_size[0] / source_size[1]) / (raw_size[0] / raw_size[1]) - 1) <= 0.005:
        matrix = geometry.resize_candidate(source_size, raw_size)
        fits.append(geometry.validate_resize_candidate(matches, matrix, source_size, raw_size))
    for fit in fits:
        fit["method"] = "SCALE_NORMALIZED_CLAHE_" + fit["method"]
    return fits, len(matches)


def historical_record(root, scope, variant):
    identity = scope["historical_records"][variant]
    folder = root.parent / identity["root"] / identity["asset_directory"]
    record = load_json(folder / "record.json")
    g3.require(record["review_id"] == identity["review_id"] and record["variant_id"] == variant,
               "Historical candidate identity mismatch")
    return folder, record


def historical_mask(root, scope, parent, old_scopes):
    generation = parent.get("generation_variant_id", parent["variant_id"])
    for folder, plan in old_scopes.items():
        if generation in plan.get("historical_variants", {}):
            return Path(plan["historical_variants"][generation]["mask"]), None
        prefix = generation.split("_OUT", 1)[0]
        for row in plan["requests"]:
            if row["variant_id"] == prefix and row["mask_sha256"] == parent["mask_sha256"]:
                ids = row["selected_polygon_ids"].split("|") if folder == A6 else None
                return root.parent / folder / row["mask_path"], ids
    raise ValueError("Historical Raw mask identity not found")


def process(root, follow, rescue_only):
    cv2.setNumThreads(2)
    scope, _, _ = frozen_plan(root)
    old_scopes = {name: load_json(root.parent / name / "scope.json") for name in (PHASE, A6)}
    routes = {item["source_id"]: item for item in scope["routes"]}
    requests = {item["variant_id"]: item for item in scope["requests"]}
    for route in routes.values():
        source = scope["sources"][route["source_id"]]
        for job in route["alignments"]:
            parent_variant = job["parent_variant"]
            variant = parent_variant + "_A7ALIGN"
            if (root / "postprocess_receipts" / f"{variant}.json").is_file():
                continue
            folder, parent = historical_record(root, scope, parent_variant)
            g3.require(parent["raw_sha256"] == job["raw_sha256"], "A7 rescue Raw identity mismatch")
            geometry_path = folder / "geometry.json"
            old_sha = hash_file(geometry_path, parent["file_sha256"].get("geometry.json"))
            old = load_json(geometry_path)
            mask_path, ids = historical_mask(root, scope, parent, old_scopes)
            raster = None
            if ids:
                shapes = load_json(Path(source["annotation"]))["shapes"]
                raster, _ = target_raster(source, shapes, ids)
            entry = {**parent, "raw": str(folder / parent["raw"]), "mask": str(mask_path), "derived": {},
                     "a2_status": "", "historical_visual": ""}
            fit = partial(fit_variant, old=old, old_sha=old_sha, supplemental=normalized_matches)
            complete_attempt(root, variant, entry, source, parent=parent, fit=fit, repair_target=raster)
    if rescue_only:
        return
    inspected = set()
    while True:
        recover_responses(root, scope, inspected)
        done = {p.stem for p in (root / "postprocess_receipts").glob("*.json")}
        for variant, entry in candidates(root, {**scope, "historical_variants": {}}, done, running=follow).items():
            request = requests[variant.split("_OUT", 1)[0]]
            source = scope["sources"][entry["source_id"]]
            repair, _ = target_raster(source, load_json(Path(source["annotation"]))["shapes"],
                                      request["selected_polygon_ids"].split("|"))
            base = g3.safe_file(root, request["batch_image_path"])
            hash_file(base, request["source_image_sha256"])
            fit = partial(fit_variant, supplemental=normalized_matches)
            complete_attempt(root, variant, entry, source, fit=fit, repair_target=repair, base_path=base)
        launched = list((root / "logs").glob("start_*.json"))
        stopped = list((root / "logs").glob("stop_*.json"))
        if not follow or (launched and len(launched) == len(stopped)):
            done = {p.stem for p in (root / "postprocess_receipts").glob("*.json")}
            if not candidates(root, {**scope, "historical_variants": {}}, done):
                break
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--rescue-only", action="store_true")
    args = parser.parse_args()
    process(args.root.resolve(), args.follow, args.rescue_only)


if __name__ == "__main__":
    main()
