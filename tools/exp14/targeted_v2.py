"""Plan and execute Exp14 A6 from the pinned human export."""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import PROMPT_PATHS, mask_for_selection
from tools.exp13_cloud.g3.prepare_g3c2_v2 import PROMPTS
from tools.exp14.large_scale_review import validate_review
from tools.exp14.postprocess_large_scale_repair import finished_candidates
from tools.exp14.prepare_large_scale_repair import (
    NATIVE, PHASE, PRIVATE_ROOT, REPO, canonical_digest, hash_file, load_json, png_bytes,
    request_identity, verify_mask,
)
from tools.exp14.run_large_scale_repair import execute, frozen_plan as parent_plan, now

REVIEW_SHA = "b1dddcf664a8e69fb53062bf0a36b54e4fd372a92e643d9db5db7ddc81c5ff49"
REPAIR = "RETRY_REPAIR"
ALIGN = "RETRY_ALIGNMENT"
REASON_GUIDANCE = {
    "缺陷残留": "Remove the residual defect itself, including its dark core or raised bright lip. A brightness change is insufficient.",
    "缺材未补全": "Continue the intact edge through the damaged notch. Preserve the actual clearance behind the edge.",
    "机械结构错误": "Match the neighboring intact curvature and retain real openings, partitions, cooling holes and seams.",
    "新增异常内容": "Remove only the marked defect. Prevent new dark wedges, contours, fragments or repeating texture.",
}
DEFERRAL_REASONS = frozenset({
    "CONFIRM_RING_STRUCTURE", "CONFIRM_MATERIAL_BOUNDARY", "CONFIRM_REAL_HOLE",
    "CONFIRM_STEP_DESIGN", "CONFIRM_STRUCTURAL_SLOT", "CONFIRM_SEAM_AFTER_ALIGNMENT",
})


def target_raster(source, shapes, ids):
    return mask_for_selection(tuple(source["size"]), shapes,
                              {source["all_polygon_ids"].index(pid) for pid in ids})


def route_sources(accepted, records, sources, structural_deferrals):
    groups = defaultdict(list)
    for item in accepted:
        groups[item["decision"]["source_id"]].append(item)
    deferred_by_source = defaultdict(dict)
    for pid, reason in structural_deferrals.items():
        sid = pid.split("_", 1)[0]
        g3.require(sid in groups and reason in DEFERRAL_REASONS and
                   any(item["resolved_target_decisions"].get(pid) == REPAIR for item in groups[sid]),
                   "Structural deferral needs matching human repair evidence")
        deferred_by_source[sid][pid] = reason
    routes = []
    for sid, variants in groups.items():
        variants.sort(key=lambda item: item["decision"]["variant_id"])
        parent = variants[0]
        plan = {"source_id": sid, "human_evidence": [
            {"row": item["decision"], "targets": item["resolved_target_decisions"]} for item in variants],
            "repair_targets": [], "alignment_targets": [], "preserve_targets": [], "deferred_targets": {},
            "parent_variant": parent["decision"]["variant_id"], "reason": ""}
        if any(item["decision"]["version"] == "V2" for item in variants):
            parent = next(item for item in variants if item["decision"]["version"] == "V2")
            plan["parent_variant"] = parent["decision"]["variant_id"]
            plan["reason"] = "EXISTING_V2_TAKES_PRECEDENCE"
        state = parent["decision"]
        targets = parent["resolved_target_decisions"]
        plan["repair_targets"] = [pid for pid, decision in targets.items() if decision == REPAIR]
        plan["alignment_targets"] = [pid for pid, decision in targets.items() if decision == ALIGN]
        plan["preserve_targets"] = [pid for pid, decision in targets.items() if decision == "PASS"]
        if state["decision"] == ALIGN or (state["decision"] == "PASS" and not state["aligned_sha256"]
                                          and state["notes"].strip() == "重新配准"):
            plan["alignment_requested"] = True
            if not plan["alignment_targets"]:
                plan["alignment_targets"] = list(targets)
        if len(variants) > 1 and state["version"] != "V2":
            plan["reason"] = "SAME_POST_MULTIPLE_OUTPUTS_ONE_SOURCE"
            g3.require(all(item["resolved_target_decisions"] == targets for item in variants),
                       "Multiple output decisions require explicit reconciliation")
        if sources[sid].get("hold"):
            plan["deferred_targets"] = dict.fromkeys(plan["repair_targets"], sources[sid]["hold"])
            plan["repair_targets"] = []
        # These notes explicitly defer identifying the affected structure before editing it.
        for pid, reason in deferred_by_source[sid].items():
            if pid in plan["repair_targets"]:
                plan["repair_targets"].remove(pid)
                plan["deferred_targets"][pid] = reason
        if state["decision"] == REPAIR and not plan["repair_targets"] and not plan["alignment_targets"]:
            plan["reason"] = plan["reason"] or "NO_UNAMBIGUOUS_REPAIR_TARGET"
        if plan["preserve_targets"] and plan["repair_targets"]:
            parent_record = records[plan["parent_variant"]]
            passed = [p for p in parent_record["polygons"] if p["polygon_id"] in plan["preserve_targets"]]
            if not state["aligned_sha256"] or any(p["unsupported_target_pixels"] != 0 for p in passed):
                plan["deferred_targets"].update(dict.fromkeys(plan["repair_targets"], "PASS_TARGET_SUPPORT_UNVERIFIED"))
                plan["repair_targets"] = []
        if plan["repair_targets"]:
            action = "MIXED_ACTION" if plan["alignment_targets"] else "V2_REPAIR"
        elif plan["alignment_targets"]:
            action = "ALIGNMENT_ONLY"
        elif plan["deferred_targets"] or state["decision"] != "PASS":
            action = "HOLD_FOR_CLARIFICATION"
        else:
            action = "ALREADY_ACCEPTED"
        plan["action"] = action
        plan["existing_version_reuse"] = state["version"] == "V2" and action in {"ALIGNMENT_ONLY", "ALREADY_ACCEPTED"}
        routes.append(plan)
    return routes


def planned_input(root, parent_root, source, plan, parent):
    image_path, annotation = Path(source["image"]), Path(source["annotation"])
    hash_file(image_path, source["image_sha256"])
    hash_file(annotation, source["annotation_sha256"])
    shapes = load_json(annotation)["shapes"]
    repair, _ = target_raster(source, shapes, plan["repair_targets"])
    rgba = Image.new("RGBA", repair.size, (255, 255, 255, 255))
    rgba.putalpha(ImageOps.invert(repair))
    preserve_ids = plan["preserve_targets"]
    variant = f"{source['source_id']}_V2"
    base = image_path.read_bytes()
    input_mode = "ORIGINAL"
    if preserve_ids:
        from tools.exp13_cloud.g3.align_g3b2_batch import read_rgb

        keep, _ = target_raster(source, shapes, preserve_ids)
        protected = np.asarray(keep) > 0
        if np.any(protected & (np.asarray(repair) > 0)):
            return None
        prior = g3.safe_file(parent_root, parent["asset_directory"] + "/aligned.png")
        hash_file(prior, parent["aligned_sha256"])
        with Image.open(image_path) as original:
            pixels = np.array(original.convert("RGB"))
        pixels[protected] = read_rgb(prior)[protected]
        base = png_bytes(Image.fromarray(pixels))
        input_mode = "ORIGINAL_WITH_REVIEWED_ALIGNED_PASS_PIXELS"
    image = root / "inputs" / f"{variant}{'.png' if preserve_ids else image_path.suffix.lower()}"
    mask = root / "masks" / f"{variant}.png"
    prompt_path = root / "prompts" / f"{variant}.txt"
    labels = source["polygon_labels"]
    class_names = sorted({labels[pid] for pid in plan["repair_targets"]})
    class_prompts = []
    for label in class_names:
        path = REPO / (PROMPTS[label][1] if label in PROMPTS else PROMPT_PATHS[label])
        class_prompts.append(f"{label}:\n" + path.read_bytes().decode("utf-8").strip())
    hints = []
    for pid in plan["repair_targets"]:
        selected, _ = target_raster(source, shapes, [pid])
        x0, y0, x1, y1 = selected.getbbox()
        hints.append(f"{labels[pid]} within the selected mask around source pixels ({x0}, {y0})–({x1}, {y1}).")
    parent_rows = [entry["row"] for entry in plan["human_evidence"]]
    reason = next((row["reason"] for row in parent_rows if row["variant_id"] == parent["variant_id"]), "")
    context = (
        "Repair only the defects inside the transparent parts of this mask. "
        "Keep camera framing, surrounding metal, true holes, joints and all opaque areas in place. "
        "Some areas outside the mask can contain earlier successful repairs or unresolved defects. Preserve them as provided."
    )
    prompt = "\n\n".join(filter(None, (context, "\n".join(hints), REASON_GUIDANCE.get(reason, ""),
                                         "\n\n".join(class_prompts)))) + "\n"
    mask_bytes, prompt_bytes = png_bytes(rgba), prompt.encode("utf-8")
    g3.write_exclusive(image, base)
    g3.write_exclusive(mask, mask_bytes)
    g3.write_exclusive(prompt_path, prompt_bytes)
    return {
        "source_id": source["source_id"], "version": "V2", "variant_id": variant,
        "split": "train", "decision": "SELECT", "status": "PROPOSED",
        "manually_approved": "NO", "authorized_for_paid_post": "NO",
        "request_count_proposed": "1", "n": "1", "roi_policy": NATIVE,
        "batch_image_path": image.relative_to(root).as_posix(), "source_image_sha256": g3.digest(base),
        "input_mode": input_mode, "original_image_sha256": source["image_sha256"],
        "source_json_sha256": source["annotation_sha256"], "image_width": source["size"][0],
        "image_height": source["size"][1], "selected_polygon_ids": "|".join(plan["repair_targets"]),
        "preserved_polygon_ids": preserve_ids, "parent_review_id": parent["review_id"],
        "parent_aligned_sha256": parent["aligned_sha256"], "mask_path": mask.relative_to(root).as_posix(),
        "mask_sha256": g3.digest(mask_bytes), "target_alpha_sha256": g3.digest(repair.tobytes()),
        "prompt_file": prompt_path.relative_to(root).as_posix(), "prompt_id": "EXP14_A6_TARGETED_V2",
        "prompt_file_sha256": g3.digest(prompt_bytes), "prompt_wire_sha256": g3.digest(prompt_bytes[:-1]),
    }


def input_for_row(root, row, source):
    annotation = Path(source["annotation"])
    hash_file(annotation, source["annotation_sha256"])
    target, _ = target_raster(source, load_json(annotation)["shapes"], row["selected_polygon_ids"].split("|"))
    g3.require(g3.digest(target.tobytes()) == row["target_alpha_sha256"], "A6 target raster changed")
    verify_mask(g3.safe_file(root, row["mask_path"]), target, row["mask_sha256"])
    return g3.validate_input(row, root, root)


def build(root, parent_root, csv):
    scope, parent_auth, protocol_sha = parent_plan(parent_root)
    hash_file(csv, REVIEW_SHA)
    accepted = validate_review(parent_root, csv)
    records = finished_candidates(parent_root)
    g3.require(len(accepted) == len(records) == 530 and
               {item["decision"]["variant_id"] for item in accepted} == set(records), "A6 requires all 530 exact identities")
    validated = root / "input" / "validated_human_review.json"
    snapshot = {"csv_sha256": REVIEW_SHA, "records": accepted}
    if validated.is_file():
        g3.require(load_json(validated) == snapshot, "Validated review snapshot does not match the CSV")
    else:
        g3.write_json_exclusive(validated, snapshot)
    deferrals_path = root / "input" / "structural_deferrals.json"
    deferrals = load_json(deferrals_path)
    g3.require(deferrals["human_review_sha256"] == REVIEW_SHA and len(deferrals["polygons"]) == 6,
               "Expected the six structural deferrals tied to this review export")
    sources = scope["sources"]
    plans = route_sources(accepted, records, sources, deferrals["polygons"])
    g3.require(len(plans) == 523, "A6 source count changed")
    request = g3.load_protocol(REPO)[0]["request"]
    requests = []
    for plan in plans:
        if not plan["repair_targets"]:
            continue
        sid = plan["source_id"]
        g3.require(not any(row["row"]["version"] == "V2" for row in plan["human_evidence"]),
                   "An existing V2 requires independent version arbitration")
        row = planned_input(root, parent_root, sources[sid], plan, records[plan["parent_variant"]])
        if row is None:
            plan["deferred_targets"].update(dict.fromkeys(plan["repair_targets"], "REPAIR_OVERLAPS_APPROVED_REGION"))
            plan["repair_targets"] = []
            plan["action"] = "ALIGNMENT_ONLY" if plan["alignment_targets"] else "HOLD_FOR_CLARIFICATION"
            continue
        row.update(endpoint=request["base_url"] + request["endpoint"], model=request["model"])
        row["request_id"] = request_identity(row)
        input_for_row(root, row, sources[sid])
        plan["request_id"] = row["request_id"]
        requests.append(row)
    payload = {"started_utc": now(), "human_review_sha256": REVIEW_SHA, "human_review_file": csv.name,
               "structural_deferrals_sha256": hash_file(deferrals_path),
               "validated_human_review_sha256": hash_file(validated),
               "parent_root": str(parent_root), "parent_authorization_sha256": parent_auth,
               "protocol_sha256": protocol_sha, "sources": sources, "routes": plans, "requests": requests,
               "prior_review_asset_paths": {k: v["asset_directory"] for k, v in records.items()}}
    g3.write_json_exclusive(root / "scope.json", payload)
    g3.write_json_exclusive(root / "authorization.json", {
        "record_type": "EXP14_A6_EXPLICIT_AUTHORIZATION", "scope_sha256": hash_file(root / "scope.json"),
        "human_review_sha256": REVIEW_SHA, "max_posts": len(requests),
        "request_ids": [row["request_id"] for row in requests],
    })
    print(dict(Counter(plan["action"] for plan in plans)), flush=True)
    print(f"{len(requests)} exact requests / {sum(len(r['selected_polygon_ids'].split('|')) for r in requests)} editable Polygons", flush=True)


def frozen_plan(root):
    auth = load_json(root / "authorization.json")
    g3.require(auth["record_type"] == "EXP14_A6_EXPLICIT_AUTHORIZATION" and auth["human_review_sha256"] == REVIEW_SHA,
               "A6 authorization missing")
    hash_file(root / "scope.json", auth["scope_sha256"])
    scope = load_json(root / "scope.json")
    hash_file(root / "input" / scope["human_review_file"], REVIEW_SHA)
    # Earlier A6 plans carry the resolved structural deferrals inline.
    if "structural_deferrals_sha256" in scope:
        hash_file(root / "input" / "structural_deferrals.json", scope["structural_deferrals_sha256"])
    if "validated_human_review_sha256" in scope:
        hash_file(root / "input" / "validated_human_review.json", scope["validated_human_review_sha256"])
    parent_root = Path(scope["parent_root"])
    hash_file(parent_root / "authorization.json", scope["parent_authorization_sha256"])
    protocol, protocol_sha = g3.load_protocol(REPO)
    g3.require(scope["protocol_sha256"] == protocol_sha, "Protocol changed")
    ids = [row["request_id"] for row in scope["requests"]]
    g3.require(ids == auth["request_ids"] and len(ids) == len(set(ids)) == auth["max_posts"] and
               len(ids) == len({row["source_id"] for row in scope["requests"]}), "A6 request set mismatch")
    routes = {plan["source_id"]: plan for plan in scope["routes"]}
    request = protocol["request"]
    for row in scope["requests"]:
        source = scope["sources"][row["source_id"]]
        plan = routes[row["source_id"]]
        explicit = {pid for entry in plan["human_evidence"] for pid, state in entry["targets"].items()
                    if state == REPAIR}
        targets = row["selected_polygon_ids"].split("|")
        g3.require(row["request_id"] == request_identity(row) == plan["request_id"] and
                   row["endpoint"] == request["base_url"] + request["endpoint"] and row["model"] == request["model"] and
                   row["version"] == "V2" and row["variant_id"] == row["source_id"] + "_V2" and
                   row["original_image_sha256"] == source["image_sha256"] and
                   row["source_json_sha256"] == source["annotation_sha256"] and
                   source["split"] == "train" and source["decision"] == "SELECT" and not source.get("hold") and
                   targets == plan["repair_targets"] and set(targets) <= explicit and
                   set(targets).isdisjoint(row["preserved_polygon_ids"]) and
                   not (parent_root / "locks" / f"{row['request_id']}.json").exists() and
                   not (parent_root / "derived" / row["variant_id"]).exists(), "A6 request outside the frozen plan")
    return scope, hash_file(root / "authorization.json"), protocol_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        if not args.csv or args.csv.resolve().parent != root / "input":
            parser.error("Prepare requires the frozen CSV in the independent A6 input directory")
        build(root, PRIVATE_ROOT / PHASE, args.csv.resolve())
    elif args.action == "verify":
        scope, _, _ = frozen_plan(root)
        for row in scope["requests"]:
            inp = input_for_row(root, row, scope["sources"][row["source_id"]])
            g3.encode_multipart(row["source_id"], row["model"], inp["prompt_wire"], inp["image_path"].name,
                                inp["image_bytes"], inp["image_format"], inp["mask_bytes"])
        print(f"A6 preflight passed for {len(scope['requests'])} unique requests")
    else:
        execute(root, os.environ.pop("EXP14_IMAGE_API_KEY", ""), args.workers, plan_loader=frozen_plan,
                input_validator=input_for_row, historical_locks=PRIVATE_ROOT / PHASE / "locks")


if __name__ == "__main__":
    main()
