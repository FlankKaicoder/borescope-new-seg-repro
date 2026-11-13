"""Merge the two Exp14 human reviews and execute the frozen A7 request set."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from tools.exp13_cloud.g3 import align_g3b2_batch as alignment
from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import PROMPT_PATHS
from tools.exp13_cloud.g3.prepare_g3c2_v2 import PROMPTS
from tools.exp14.large_scale_review import validate_review
from tools.exp14.postprocess_large_scale_repair import finished_candidates
from tools.exp14.prepare_large_scale_repair import NATIVE, PHASE, PRIVATE_ROOT, REPO, canonical_digest, hash_file, load_json, png_bytes, request_identity
from tools.exp14.run_large_scale_repair import execute, now
from tools.exp14.targeted_v2 import ALIGN, REPAIR, REASON_GUIDANCE, frozen_plan as a6_plan, input_for_row, target_raster
from tools.exp14.targeted_v2_review import reviewable

A6 = "exp14_a6_targeted_v2_20261113"
A7 = "exp14_a7_targeted_v3"
INPUTS = (
    ("Exp14_review_AI_assisted_2026-11-13T07-37-09-066Z.csv", "b1dddcf664a8e69fb53062bf0a36b54e4fd372a92e643d9db5db7ddc81c5ff49"),
    ("Exp14_review_AI_assisted_2026-11-13T10-50-40-611Z.csv", "be536dc39a7cb8e3fb3671180e55699b8cdff4a909eb28ead5dd6d8694d58d38"),
)
ACTIONS = {"ACCEPTED_VERSION_PRESERVE", "PARTIAL_SUCCESS_PRESERVE", "V3_TARGETED_REPAIR", "ALIGNMENT_ONLY",
           "MIXED_REPAIR_AND_ALIGNMENT", "DELIVERY_RECOVERY", "HOLD_FOR_CLARIFICATION"}
CLASS_GUIDANCE = {
    "corrosion": "Remove the actual dark core, pale perimeter, and fine pitting. Retain surrounding grain and gradual illumination. Avoid a smooth or sharp-edged patch.",
    "Material missing": "Reconstruct the broken external lip from its intact segments, extending matching curvature. Retain the true cavity and gaps between parts. Rounding or brightening a remaining notch is insufficient.",
    "Crack": "Remove the fracture and its branching trace. Preserve a real seam with its original course and width. Restore continuous texture across the fracture.",
    "Dent": "Restore the continuous profile at the indentation. Match the original curvature and highlight on either side. Avoid raised ridges, a remaining depression, or a new corner.",
    "Burn": "Remove carbonized and delaminated material while preserving real lips, channels, and rows of cooling holes.",
    "Tears": "Reconstruct the original edge through the torn segment without closing the gap behind it or adding a step.",
}
VISUAL_CUES = {
    "dark core or band": "黑芯 深色 暗芯 黑凹 暗沟 暗线 黑斑",
    "pale rim or ring": "亮边 白环 亮环 白色 亮唇",
    "fine pits and roughness": "细坑 点蚀 粗糙 颗粒 斑驳",
    "remaining notch or stepped contour": "缺口 台阶 折点 豁口 凹口 折角",
    "boundary or lighting seam": "补片 拼接 色差 接痕",
    "delamination": "翘皮 剥落 剥蚀",
    "branching or hairline trace": "裂缝 分叉 裂线 细线",
}


def history(root):
    old = root.parent / PHASE
    second = root.parent / A6
    scope, _, _ = a6_plan(second)
    source_rows = load_json(old / "scope.json")
    collected = defaultdict(list)
    libraries = {}
    for number, (folder, records) in enumerate(((old, finished_candidates(old)), (second, reviewable(second))), 1):
        filename, sha = INPUTS[number - 1]
        path = root / "input" / filename
        hash_file(path, sha)
        rows = validate_review(folder, path, records=records.values())
        g3.require(len(rows) == len(records) == (530 if number == 1 else 288), "Review identity count changed")
        if number == 2:
            g3.require(Counter(x["decision"]["decision"] for x in rows) ==
                       {"PASS": 142, REPAIR: 108, ALIGN: 8, "HOLD": 30} and
                       Counter(s for x in rows for s in x["human_polygon_decisions"].values()) ==
                       {"PASS": 212, REPAIR: 182, ALIGN: 4, "HOLD": 55} and
                       sum(not x["human_polygon_decisions"] for x in rows) == 36, "Second review totals changed")
        for item in rows:
            row = item["decision"]
            record = records[row["variant_id"]]
            sid = row["source_id"]
            source = source_rows["sources"][sid]
            g3.require(source["split"] == "train" and source["decision"] == "SELECT" and
                       record["source_sha256"] == source["image_sha256"] and
                       record["annotation_sha256"] == source["annotation_sha256"] and
                       set(item["resolved_target_decisions"]) == set(source["selected_polygon_ids"]),
                       "Review record differs from frozen TRAIN")
            libraries[row["variant_id"]] = (folder, record)
            collected[sid].append({"round": number, "row": row, "targets": item["resolved_target_decisions"],
                                   "explicit": item["human_polygon_decisions"],
                                   "support": {p["polygon_id"]: p["unsupported_target_pixels"] for p in record["polygons"]}})
    g3.require(len(collected) == 523 and sum(any(x["round"] == 2 for x in g) for g in collected.values()) == 264,
               "Source counts changed")
    return collected, libraries, source_rows, scope


def delivery_audit(root, scope):
    parent = root.parent / A6
    receipts = {}
    for request in scope["requests"]:
        variant = request["variant_id"]
        folder = parent / "live" / variant
        terminal = load_json(folder / "TERMINAL.json")
        recovery = parent / "response_recovery" / variant / "RECOVERED.json"
        if terminal["status"] == "RETURNED_IMAGE" or recovery.is_file():
            continue
        http = load_json(folder / "http_response.json")
        body = folder / "response_body.bin"
        hash_file(body, http["response_body_sha256"])
        manifest = load_json(folder / "request_manifest.json")
        lock_path = parent / "locks" / f"{request['request_id']}.json"
        lock = load_json(lock_path)
        g3.require(manifest["request_id"] == request["request_id"] and lock["request_sha256"] == canonical_digest(manifest)
                   and terminal["post_invocations"] == 1 and terminal["http_status"] == http["http_status"],
                   "Historical delivery identity mismatch")
        payload = load_json(body) if http["http_status"] == 200 else {}
        retrievable = any(item.get("url") or item.get("b64_json") for item in payload.get("data", []) if isinstance(item, dict))
        g3.require(not retrievable, "Check saved image delivery before planning a new request")
        reason = ("UNCERTAIN_DELIVERY" if terminal["delivery_uncertain"] else
                  "PROVIDER_REVIEW_OR_FAILURE" if terminal["http_status"] == 477 else "CONFIRMED_NO_IMAGE_PAYLOAD")
        receipts[request["source_id"]] = {"variant": variant, "reason": reason, "request_id": request["request_id"],
                                         "terminal_sha256": hash_file(folder / "TERMINAL.json"),
                                         "response_sha256": http["response_body_sha256"], "lock_sha256": hash_file(lock_path)}
    return receipts


def evidence_priority(item):
    row = item["row"]
    return (row["whole_image_normal"] == "YES", row["b_region_review"] == "CONFIRMED",
            row["decision"] == "PASS", item["round"], row["version"], row["variant_id"])


def choose_routes(grouped, sources, a6, delivery):
    earlier_routes = {r["source_id"]: r for r in a6["routes"]}
    routes = []
    for sid, evidence in grouped.items():
        prior = earlier_routes[sid]
        source = sources[sid]
        latest = [e for e in evidence if e["round"] == 2]
        authoritative = latest or evidence
        result = {"source_id": sid, "human_evidence": evidence, "repair_targets": [], "repair_evidence": {},
                  "preserve_targets": [], "pixel_sources": {}, "alignment_targets": [], "alignments": [],
                  "deferred_targets": {}, "whole_confirmed_variants": [], "delivery": delivery.get(sid)}
        result["whole_confirmed_variants"] = [e["row"]["variant_id"] for e in evidence if
            e["row"]["decision"] == "PASS" and e["row"]["whole_image_normal"] == "YES"]
        result["prior_variants"] = {
            ver: [e["row"]["variant_id"] for e in sorted(evidence, key=evidence_priority, reverse=True)
                  if e["row"]["version"] == ver] for ver in ("V1", "V2")}
        alignment_jobs = {}
        for pid in source["selected_polygon_ids"]:
            passed = [e for e in evidence if e["targets"][pid] == "PASS" and
                      (not source.get("hold") or pid in e["explicit"])]
            supported = sorted((e for e in passed if e["support"][pid] == 0 and e["row"]["aligned_sha256"]),
                               key=evidence_priority, reverse=True)
            if passed:
                result["preserve_targets"].append(pid)
            if supported:
                result["pixel_sources"][pid] = supported[0]["row"]["variant_id"]
            alignment = [e for e in authoritative if e["targets"][pid] == ALIGN or
                         (e["row"]["decision"] == ALIGN and e["targets"][pid] == "PASS")]
            if not latest and pid in prior["alignment_targets"]:
                alignment += [e for e in evidence if e["row"]["variant_id"] == prior["parent_variant"]]
            if alignment and not any(e["row"]["decision"] == "PASS" and e["row"]["aligned_sha256"] and
                                     e["support"][pid] == 0 for e in passed):
                for e in alignment:
                    row = e["row"]
                    job = alignment_jobs.setdefault(row["raw_sha256"], {"parent_variant": row["variant_id"], "targets": set()})
                    job["targets"].add(pid)
            if passed or alignment:
                continue
            failures = [e for e in authoritative if e["targets"][pid] == REPAIR and
                        (pid in e["explicit"] or e["row"]["decision"] == REPAIR)]
            if not failures:
                continue
            competing_hold = any(e["targets"][pid] == "HOLD" for e in authoritative)
            reason = source.get("hold") or prior["deferred_targets"].get(pid)
            if competing_hold:
                reason = reason or "REPAIR_AND_HOLD_ACROSS_OUTPUTS"
            if any(e["row"]["reason"] == "机械结构错误" for e in failures):
                reason = reason or "STRUCTURAL_REFERENCE_REQUIRED"
            receipt = result["delivery"]
            if receipt and receipt["reason"] != "CONFIRMED_NO_IMAGE_PAYLOAD":
                reason = reason or receipt["reason"]
            if not latest and prior["action"] == "HOLD_FOR_CLARIFICATION":
                reason = reason or "PRIOR_REVIEW_REQUIRES_CLARIFICATION"
            if reason:
                result["deferred_targets"][pid] = reason
            else:
                result["repair_targets"].append(pid)
                result["repair_evidence"][pid] = [e["row"]["variant_id"] for e in failures]
        for raw, job in alignment_jobs.items():
            result["alignments"].append({**job, "targets": sorted(job["targets"]), "raw_sha256": raw})
        result["alignment_targets"] = sorted({pid for job in result["alignments"] for pid in job["targets"]})
        if result["repair_targets"] and set(result["preserve_targets"]) - result["pixel_sources"].keys():
            result["deferred_targets"].update(dict.fromkeys(result["repair_targets"], "APPROVED_REGION_HAS_NO_SOURCE_GRID_SUPPORT"))
            result["repair_targets"] = []
        routes.append(result)
    return routes


def prepare_request(root, source, route, libraries, protocol, occupied):
    sid = source["source_id"]
    version = f"V{max(occupied[sid], default=0) + 1}"
    g3.require(version in {"V2", "V3"}, "A7 version is outside the authorized range")
    variant = f"{sid}_{version}"
    original = Path(source["image"])
    hash_file(original, source["image_sha256"])
    annotation = Path(source["annotation"])
    hash_file(annotation, source["annotation_sha256"])
    shapes = load_json(annotation)["shapes"]
    selected, _ = target_raster(source, shapes, route["repair_targets"])
    edit = np.asarray(selected) > 0
    pixels = alignment.read_rgb(original)
    by_variant = defaultdict(list)
    for pid, prior in route["pixel_sources"].items():
        by_variant[prior].append(pid)
    protected = np.zeros(edit.shape, dtype=bool)
    entries = []
    for prior, ids in by_variant.items():
        mask, _ = target_raster(source, shapes, ids)
        region = np.asarray(mask) > 0
        if np.any(region & (protected | edit)):
            route["deferred_targets"].update(dict.fromkeys(route["repair_targets"], "PROTECTED_POLYGON_OVERLAP"))
            route["repair_targets"] = []
            return None
        previous_root, record = libraries[prior]
        folder = previous_root / record["asset_directory"]
        path = folder / "aligned.png"
        valid_path = folder / "valid.png"
        hash_file(path, record["aligned_sha256"])
        hash_file(valid_path, record["file_sha256"]["valid.png"])
        g3.require(np.all(alignment.read_rgb(valid_path)[region, 0] > 0), "Approved target lacks exact Valid support")
        pixels[region] = alignment.read_rgb(path)[region]
        protected |= region
        entries.append({"variant_id": prior, "version": record["version"], "review_id": record["review_id"],
                        "polygon_ids": ids, "aligned_sha256": record["aligned_sha256"],
                        "valid_sha256": record["file_sha256"]["valid.png"], "target_sha256": g3.digest(mask.tobytes())})
    base = png_bytes(Image.fromarray(pixels)) if entries else original.read_bytes()
    edit_image = Image.new("RGBA", selected.size, (255, 255, 255, 255))
    edit_image.putalpha(ImageOps.invert(selected))
    edit_bytes = png_bytes(edit_image)
    image = root / "inputs" / (variant + (".png" if entries else original.suffix.lower()))
    mask = root / "masks" / (variant + ".png")
    prompt = root / "prompts" / (variant + ".txt")
    labels = source["polygon_labels"]
    sections = ["Only repair the listed defects inside transparent mask pixels. Keep actual cooling holes, slots, seams, camera markings and all other structures in place. Match the surrounding gradient and texture across the mask boundary."]
    for pid in route["repair_targets"]:
        label = labels[pid]
        bounds = target_raster(source, shapes, [pid])[0].getbbox()
        sections.append(f"Target {pid}, {label}, native pixel box {bounds}. {CLASS_GUIDANCE[label]}")
        for round_no in (1, 2):
            historical = [e["row"] for e in route["human_evidence"] if e["round"] == round_no and e["targets"][pid] == REPAIR]
            cues = [name for name, words in VISUAL_CUES.items() if any(word in row["notes"]
                    for word in words.split() for row in historical)]
            guidance = list(dict.fromkeys(REASON_GUIDANCE[row["reason"]] for row in historical if row["reason"] in REASON_GUIDANCE))
            if cues or guidance:
                sections.append(f"Review round {round_no}, repair evidence for this target: " +
                                "; ".join(filter(None, [", ".join(cues), *guidance])) + ".")
    for label in sorted({labels[pid] for pid in route["repair_targets"]}):
        path = REPO / (PROMPTS[label][1] if label in PROMPTS else PROMPT_PATHS[label])
        sections.append(path.read_text(encoding="utf-8").strip())
    prompt_bytes = ("\n\n".join(sections) + "\n").encode("utf-8")
    for path, contents in ((image, base), (mask, edit_bytes), (prompt, prompt_bytes)):
        g3.write_exclusive(path, contents)
    versions = sorted({entry["version"] for entry in entries})
    row = {
        "source_id": sid, "version": version, "variant_id": variant, "execution_phase": A7,
        "split": "train", "decision": "SELECT", "status": "PROPOSED", "manually_approved": "NO",
        "authorized_for_paid_post": "NO", "request_count_proposed": "1", "n": "1", "roi_policy": NATIVE,
        "batch_image_path": image.relative_to(root).as_posix(), "source_image_sha256": g3.digest(base),
        "original_image_sha256": source["image_sha256"], "source_json_sha256": source["annotation_sha256"],
        "image_width": source["size"][0], "image_height": source["size"][1],
        "selected_polygon_ids": "|".join(route["repair_targets"]), "preserved_polygon_ids": route["preserve_targets"],
        "pixel_sources": entries, "input_mode": "Original" + "".join(f" + {v} PASS" for v in versions),
        "mask_path": mask.relative_to(root).as_posix(), "mask_sha256": g3.digest(edit_bytes),
        "target_alpha_sha256": g3.digest(selected.tobytes()), "prompt_file": prompt.relative_to(root).as_posix(),
        "prompt_id": "EXP14_A7_TARGETED", "prompt_file_sha256": g3.digest(prompt_bytes),
        "prompt_wire_sha256": g3.digest(prompt_bytes[:-1]), "model": protocol["model"],
        "endpoint": protocol["base_url"] + protocol["endpoint"],
    }
    row["request_id"] = request_identity(row)
    input_for_row(root, row, source)
    return row


def prepare(root):
    grouped, libraries, old_scope, second_scope = history(root)
    sources = old_scope["sources"]
    delivery = delivery_audit(root, second_scope)
    routes = choose_routes(grouped, sources, second_scope, delivery)
    protocol, protocol_sha = g3.load_protocol(REPO)
    occupied = defaultdict(set)
    historical_ids = set()
    for item in [*old_scope["requests"], *second_scope["requests"], *(record for _, record in libraries.values())]:
        occupied[item["source_id"]].add(int(item["version"][1:]))
        if item.get("request_id"):
            historical_ids.add(item["request_id"])
    requests = []
    for route in routes:
        sid = route["source_id"]
        row = prepare_request(root, sources[sid], route, libraries, protocol["request"], occupied) if route["repair_targets"] else None
        if row:
            g3.require(row["request_id"] not in historical_ids, "Historical request replay")
            route["request_id"] = row["request_id"]
            requests.append(row)
        if route["delivery"]:
            action = "DELIVERY_RECOVERY"
        elif row and route["alignments"]:
            action = "MIXED_REPAIR_AND_ALIGNMENT"
        elif row:
            action = "V3_TARGETED_REPAIR"
        elif route["alignments"]:
            action = "ALIGNMENT_ONLY"
        elif route["deferred_targets"]:
            action = "HOLD_FOR_CLARIFICATION"
        elif route["whole_confirmed_variants"] or (route["preserve_targets"] and
             len(route["preserve_targets"]) == len(sources[sid]["selected_polygon_ids"])):
            action = "ACCEPTED_VERSION_PRESERVE"
        elif route["preserve_targets"]:
            action = "PARTIAL_SUCCESS_PRESERVE"
        else:
            action = "HOLD_FOR_CLARIFICATION"
        route["action"] = action
    records = {variant: {"root": folder.name, "asset_directory": record["asset_directory"], "review_id": record["review_id"]}
               for variant, (folder, record) in libraries.items()}
    g3.write_json_exclusive(root / "scope.json", {"started_utc": now(), "protocol_sha256": protocol_sha,
        "historical_scopes": {name: hash_file(root.parent / name / "scope.json") for name in (PHASE, A6)},
        "inputs": dict(INPUTS), "sources": sources, "routes": routes, "requests": requests, "historical_records": records})
    g3.write_json_exclusive(root / "authorization.json", {"phase": A7, "scope_sha256": hash_file(root / "scope.json"),
        "request_ids": [row["request_id"] for row in requests]})
    print(dict(Counter(route["action"] for route in routes)))
    print(len(requests), "sources", sum(len(r["selected_polygon_ids"].split("|")) for r in requests), "Polygons",
          dict(Counter(r["version"] for r in requests)), "alignment jobs", sum(len(r["alignments"]) for r in routes))


def frozen_plan(root):
    auth = load_json(root / "authorization.json")
    hash_file(root / "scope.json", auth["scope_sha256"])
    scope = load_json(root / "scope.json")
    g3.require(auth["phase"] == A7 and scope["inputs"] == dict(INPUTS), "A7 authorization mismatch")
    for name, sha in INPUTS:
        hash_file(root / "input" / name, sha)
    for name, sha in scope["historical_scopes"].items():
        hash_file(root.parent / name / "scope.json", sha)
    protocol, protocol_sha = g3.load_protocol(REPO)
    g3.require(protocol_sha == scope["protocol_sha256"], "A7 protocol changed")
    routes = {r["source_id"]: r for r in scope["routes"]}
    ids = [r["request_id"] for r in scope["requests"]]
    g3.require(ids == auth["request_ids"] and len(ids) == len(set(ids)) == len({r["source_id"] for r in scope["requests"]}),
               "A7 request identities differ")
    for row in scope["requests"]:
        sid = row["source_id"]
        source, route = scope["sources"][sid], routes[sid]
        requested = row["selected_polygon_ids"].split("|")
        g3.require(row["request_id"] == request_identity(row) == route["request_id"] and row["n"] == "1"
                   and row["endpoint"] == protocol["request"]["base_url"] + protocol["request"]["endpoint"]
                   and row["model"] == protocol["request"]["model"] and source["split"] == "train"
                   and source["decision"] == "SELECT" and not source.get("hold")
                   and row["variant_id"] == f"{sid}_{row['version']}" and row["version"] in {"V2", "V3"}
                   and row["source_json_sha256"] == source["annotation_sha256"]
                   and row["original_image_sha256"] == source["image_sha256"]
                   and requested == route["repair_targets"] and set(requested) <= route["repair_evidence"].keys()
                   and set(requested).isdisjoint(route["preserve_targets"]) and set(requested).isdisjoint(route["alignment_targets"])
                   and all(not (root.parent / historical / "locks" / f"{row['request_id']}.json").exists()
                           and not (root.parent / historical / "live" / row["variant_id"]).exists()
                           for historical in (PHASE, A6))
                   and row["variant_id"] not in scope["historical_records"], "Request exceeds the A7 scope")
    return scope, hash_file(root / "authorization.json"), protocol_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    g3.require(root.parent == PRIVATE_ROOT.resolve() and root.name.startswith(A7 + "_"), "Use an independent private A7 directory")
    if args.action == "prepare":
        prepare(root)
    elif args.action == "live":
        execute(root, os.environ.pop("EXP14_IMAGE_API_KEY", ""), args.workers,
                plan_loader=frozen_plan, input_validator=input_for_row)
    else:
        scope, _, _ = frozen_plan(root)
        for row in scope["requests"]:
            inp = input_for_row(root, row, scope["sources"][row["source_id"]])
            g3.encode_multipart(row["source_id"], row["model"], inp["prompt_wire"], inp["image_path"].name,
                                inp["image_bytes"], inp["image_format"], inp["mask_bytes"])
        print("Verified A7 request scope and offline multipart", len(scope["requests"]))


if __name__ == "__main__":
    main()
