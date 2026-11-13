"""Verify the A7 asset, dispatch, and review ledgers and summarize independent candidates."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tools.exp13_cloud.g3 import align_g3b2_batch as geometry
from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.verify_train_export import EXPECTED_MANIFEST_SHA256
from tools.exp14.postprocess_large_scale_repair import candidates
from tools.exp14.prepare_large_scale_repair import canonical_digest, hash_file, load_json
from tools.exp14.targeted_v2 import input_for_row, target_raster
from tools.exp14.targeted_v2_review import reviewable, verified_files, waves
from tools.exp14.targeted_v3 import A7, frozen_plan, history
from tools.exp14.targeted_v3_alignment import historical_record


def audit_frozen_train(sources, historical_scope):
    manifest = Path(next(iter(sources.values()))["image"]).parents[2] / "frozen_split_manifest.csv"
    g3.require(historical_scope["provenance"]["split_sha256"] == EXPECTED_MANIFEST_SHA256,
               "Unexpected frozen TRAIN manifest identity")
    hash_file(manifest, EXPECTED_MANIFEST_SHA256)
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        train = {r["stem"]: r for r in csv.DictReader(source) if r["split"] == "train"}
    for sid, source in sources.items():
        row = train[sid]
        g3.require((source["group_id"], source["image_sha256"], source["annotation_sha256"]) ==
                   (row["group_id"], row["sha256"], row["json_sha256"]), "Frozen TRAIN identity differs")
        hash_file(Path(source["image"]), source["image_sha256"])
        hash_file(Path(source["annotation"]), source["annotation_sha256"])


def audit_requests(root, scope, auth_sha, protocol_sha):
    requests = {item["variant_id"]: item for item in scope["requests"]}
    routes = {item["source_id"]: item for item in scope["routes"]}
    locks = {path.stem for path in (root / "locks").glob("*.json")}
    folders = {path.name for path in (root / "live").glob("*") if path.is_dir()}
    g3.require(folders <= requests.keys() and locks == {requests[v]["request_id"] for v in folders},
               "Unexpected live folder or lock")
    http_counts, usage, cost, terminal_counts = Counter(), Counter(), Counter(), Counter()
    attempted_targets = 0
    for variant, row in requests.items():
        sid = row["source_id"]
        source = scope["sources"][sid]
        inp = input_for_row(root, row, source)
        with Image.open(root / row["mask_path"]) as mask:
            selected = np.asarray(mask.getchannel("A")) == 0
        shapes = load_json(Path(source["annotation"]))["shapes"]
        route = routes[sid]
        for pid in set(route["preserve_targets"]) | route["deferred_targets"].keys():
            protected, _ = target_raster(source, shapes, [pid])
            g3.require(not np.any(selected & (np.asarray(protected) > 0)), "Unapproved Polygon overlaps the edit")
        original = Path(source["image"])
        hash_file(original, row["original_image_sha256"])
        expected_pixels = geometry.read_rgb(original)
        for entry in row["pixel_sources"]:
            previous_folder, record = historical_record(root, scope, entry["variant_id"])
            g3.require(record["review_id"] == entry["review_id"], "Protected region review changed")
            image = previous_folder / "aligned.png"
            valid = previous_folder / "valid.png"
            hash_file(image, entry["aligned_sha256"])
            hash_file(valid, entry["valid_sha256"])
            mask, _ = target_raster(source, shapes, entry["polygon_ids"])
            g3.require(g3.digest(mask.tobytes()) == entry["target_sha256"], "Protected region changed")
            region = np.asarray(mask) > 0
            g3.require(np.all(geometry.read_rgb(valid)[region, 0] > 0), "Invalid protected region")
            expected_pixels[region] = geometry.read_rgb(image)[region]
        g3.require(np.array_equal(expected_pixels, geometry.read_rgb(inp["image_path"])), "Paid input changed unapproved pixels")
        if variant not in folders:
            continue
        attempted_targets += len(route["repair_targets"])
        body, ct = g3.encode_multipart(sid, row["model"], inp["prompt_wire"], inp["image_path"].name,
                                      inp["image_bytes"], inp["image_format"], inp["mask_bytes"])
        folder = root / "live" / variant
        request = load_json(folder / "request_manifest.json")
        expected = {**row, "method": "POST", "body_sha256": g3.digest(body), "body_bytes": len(body),
                    "content_type": ct, "authorization_sha256": auth_sha, "protocol_sha256": protocol_sha,
                    "automatic_retries": 0, "redirects": False}
        lock = load_json(root / "locks" / f"{row['request_id']}.json")
        terminal = load_json(folder / "TERMINAL.json")
        g3.require(request == expected and lock["request_sha256"] == canonical_digest(request) and
                   load_json(folder / "POST_STARTED.json")["post_invocations"] == terminal["post_invocations"] == 1,
                   "Request identity or POST receipt mismatch")
        http_counts[str(terminal["http_status"])] += 1
        terminal_counts[terminal["status"]] += 1
        if (folder / "http_response.json").exists():
            http = load_json(folder / "http_response.json")
            g3.require(http["http_status"] == terminal["http_status"], "Response status differs")
            hash_file(folder / "response_body.bin", http["response_body_sha256"])
        paths = [folder / "response_manifest.json"]
        recovered = root / "response_recovery" / variant
        if (recovered / "RECOVERED.json").is_file():
            receipt = load_json(recovered / "RECOVERED.json")
            g3.require(receipt["api_posts"] == 0, "Response recovery must be read-only")
            paths = [recovered / output["path"] / "response_manifest.json" for output in receipt["outputs"]]
        responses = [load_json(path) for path in paths if path.is_file()]
        if responses:
            first = responses[0]
            g3.require(all(r["usage"] == first["usage"] and r["reported_cost"] == first["reported_cost"] for r in responses),
                       "Multiple outputs have inconsistent billing metadata")
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = (first.get("usage") or {}).get(key)
                if type(value) is int:
                    usage[key] += value
            usage["requests_with_usage"] += bool(first.get("usage"))
            cost[first["cost_status"]] += 1
        else:
            cost[terminal["cost_status"]] += 1
    stop_reasons = [load_json(path)["stop_reason"] for path in sorted((root / "logs").glob("stop_*.json"))]
    return {"attempted": len(locks), "unattempted": len(requests) - len(locks), "http": dict(http_counts),
            "attempted_polygons": attempted_targets, "terminal_statuses": dict(terminal_counts),
            "usage": dict(usage), "cost": dict(cost), "stop_reasons": stop_reasons, "repeat_posts": 0}


def audit_candidates(root, scope):
    generated = candidates(root, {**scope, "historical_variants": {}})
    requests = {item["variant_id"]: item for item in scope["requests"]}
    rescues = {job["parent_variant"] + "_A7ALIGN": job for route in scope["routes"] for job in route["alignments"]}
    receipts = {p.stem: load_json(p) for p in (root / "postprocess_receipts").glob("*.json")}
    g3.require(receipts.keys() == generated.keys() | rescues.keys(), "Postprocess receipt set differs")
    for item in generated.values():
        hash_file(Path(item["raw"]), item["raw_sha256"])
    records = reviewable(root)
    bounded = 0
    for variant, record in records.items():
        folder = root / record["asset_directory"]
        verified_files(folder, record)
        g3.require(load_json(folder / "record.json") == record, "Stored record differs from receipt")
        if variant in rescues:
            _, previous = historical_record(root, scope, rescues[variant]["parent_variant"])
            g3.require(record["raw_sha256"] == previous["raw_sha256"] and record["aligned_sha256"] != previous["aligned_sha256"]
                       and record["alignment_parent_review_id"] == previous["review_id"], "Rescue identity changed")
        if "bounded.png" not in record["file_sha256"]:
            continue
        row = requests[variant.split("_OUT", 1)[0]]
        with Image.open(root / row["mask_path"]) as mask:
            selected = np.asarray(mask.getchannel("A")) == 0
        used = selected & (geometry.read_rgb(folder / "valid.png")[..., 0] > 0)
        actual = geometry.read_rgb(folder / "bounded.png")
        base = geometry.read_rgb(root / row["batch_image_path"])
        aligned = geometry.read_rgb(folder / "aligned.png")
        g3.require(np.array_equal(actual[~used], base[~used]) and np.array_equal(actual[used], aligned[used]),
                   "Bounded edited outside supported Repair Mask")
        bounded += 1
    return records, {
        "raw_count": len(generated), "sources_returned_images": len({r["source_id"] for r in generated.values()}),
        "raw_by_version": dict(Counter(r["version"] for r in generated.values())),
        "raw_aligned": sum(bool(records[v]["aligned_sha256"]) for v in generated.keys() & records.keys()),
        "old_raw_alignment_attempts": len(rescues), "rescues": dict(Counter(receipts[v]["status"] for v in rescues)),
        "bounded_pixel_checks": bounded, "local_errors": sum(x["status"] == "LOCAL_POSTPROCESS_ERROR" for x in receipts.values()),
    }


def candidate_overview(scope, records, observed):
    old_a, old_b = set(), set()
    old_b_targets = set()
    existing = defaultdict(set)
    old_a_versions, old_b_versions = defaultdict(set), defaultdict(set)
    pending_targets = defaultdict(set)
    unresolved_source_sets = defaultdict(set)
    new_by_source = defaultdict(list)
    for variant, result in observed.items():
        new_by_source[result["source_id"]].append((records[variant], result))
    merged = []
    final_targets, final_sources, final_versions = defaultdict(set), defaultdict(set), defaultdict(set)
    final_version_targets = defaultdict(set)
    for route in scope["routes"]:
        sid = route["source_id"]
        source = scope["sources"][sid]
        evidence = route["human_evidence"]
        latest = [e for e in evidence if e["round"] == 2] or evidence
        historical_passes = defaultdict(list)
        new_passes = defaultdict(list)
        follow_up = defaultdict(dict)
        source_holds = set()
        for record, result in new_by_source[sid]:
            for p in record["polygons"]:
                pid = p["polygon_id"]
                if result["ai_polygon_decisions"].get(pid) == "PASS" and p["unsupported_target_pixels"] == 0:
                    new_passes[pid].append(result["variant_id"])
        for e in route["human_evidence"]:
            row = e["row"]
            version = row["version"]
            if row["decision"] == "PASS":
                existing[version].add(sid)
            for pid, state in e["targets"].items():
                if state == "PASS":
                    historical_passes[pid].append(row["variant_id"])
            if row["whole_image_normal"] == "YES":
                old_a.add(sid)
                old_a_versions[version].add(sid)
            if row["b_region_review"] == "CONFIRMED":
                targets = {p for p, state in e["targets"].items() if state == "PASS" and e["support"][p] == 0}
                if targets:
                    old_b.add(sid)
                    old_b_versions[version].add(sid)
                    old_b_targets.update((sid, p) for p in targets)
        for key, values in (("RETRY_REPAIR", route["repair_targets"]), ("RETRY_ALIGNMENT", route["alignment_targets"]),
                            ("HOLD", route["deferred_targets"])):
            if values:
                unresolved_source_sets[key].add(sid)
                pending_targets[key].update((sid, p) for p in values)
        if route["action"] == "HOLD_FOR_CLARIFICATION":
            unresolved_source_sets["HOLD"].add(sid)
        if route["delivery"] and route["delivery"]["reason"] != "CONFIRMED_NO_IMAGE_PAYLOAD":
            source_holds.add(route["delivery"]["reason"])
        for pid in source["selected_polygon_ids"]:
            if pid in new_passes and pid not in route["deferred_targets"]:
                continue
            proposals = [(r["version"], r["ai_polygon_decisions"].get(pid)) for _, r in new_by_source[sid]]
            states = {state for _, state in proposals}
            old_states = {e["targets"][pid] for e in latest}
            if pid in route["deferred_targets"]:
                state, reason = "HOLD", route["deferred_targets"][pid]
            elif pid in route["alignment_targets"] or "RETRY_ALIGNMENT" in states:
                state, reason = "RETRY_ALIGNMENT", "VALID_CORRESPONDENCE_PENDING"
            elif pid in historical_passes:
                continue
            elif "HOLD" in states or ("HOLD" in old_states and pid not in route["repair_targets"]):
                state, reason = "HOLD", "HUMAN_CLARIFICATION_REQUIRED"
            elif pid in route["repair_targets"] or "RETRY_REPAIR" in states:
                state, reason = "RETRY_REPAIR", "REPAIR_PENDING"
            else:
                continue
            follow_up[state][pid] = reason
            versions = {version for version, proposed in proposals if proposed == state}
            versions.update(e["row"]["version"] for e in latest if e["targets"][pid] == state)
            if not versions:
                versions.update(e["row"]["version"] for e in latest if e["targets"][pid] != "PASS")
            final_versions[state].update((sid, version) for version in versions)
            final_version_targets[state].update((sid, pid, version) for version in versions)
        if any(result["ai_decision"] == "HOLD" for _, result in new_by_source[sid]):
            source_holds.add("NEW_WHOLE_IMAGE_AI_HOLD")
        if source_holds:
            final_sources["HOLD"].add(sid)
        for state, entries in follow_up.items():
            if entries:
                final_sources[state].add(sid)
                final_targets[state].update((sid, pid) for pid in entries)
        merged.append({"source_id": sid, "historical_pass_variants": [e["row"]["variant_id"] for e in evidence
                       if e["row"]["decision"] == "PASS"], "historical_pass_polygons": dict(historical_passes),
                       "new_ai_pass_polygons_with_valid_support": dict(new_passes),
                       "follow_up": dict(follow_up), "source_holds": sorted(source_holds)})

    new_a, new_b, new_b_targets, new_pass = set(), set(), set(), set()
    new_a_versions, new_b_versions, new_pass_versions = defaultdict(set), defaultdict(set), defaultdict(set)
    for variant, result in observed.items():
        sid, version = result["source_id"], result["version"]
        if result["ai_decision"] == "PASS":
            new_pass.add(sid)
            new_pass_versions[version].add(sid)
        if result["ai_a_candidate"] == "YES":
            new_a.add(sid)
            new_a_versions[version].add(sid)
        if result["ai_b_candidate"] == "YES":
            support = {p["polygon_id"]: p["unsupported_target_pixels"] for p in records[variant]["polygons"]}
            targets = {p for p, state in result["ai_polygon_decisions"].items() if state == "PASS" and support[p] == 0}
            if targets:
                new_b.add(sid)
                new_b_versions[version].add(sid)
                new_b_targets.update((sid, p) for p in targets)
    counts = lambda by_version: {version: len(values) for version, values in by_version.items()}
    summary = {"historical_pass_sources_by_version": counts(existing), "new_ai_pass_sources": len(new_pass),
            "historical_pass_unique_sources": len(set().union(*existing.values())),
            "new_ai_pass_sources_by_version": counts(new_pass_versions),
            "a_human_confirmed_sources": len(old_a), "a_human_confirmed_by_version": counts(old_a_versions),
            "a_new_ai_candidate_sources": len(new_a), "a_new_ai_by_version": counts(new_a_versions),
            "a_union_candidate_sources": len(old_a | new_a),
            "b_human_confirmed_sources": len(old_b), "b_human_confirmed_by_version": counts(old_b_versions),
            "b_new_ai_candidate_sources": len(new_b), "b_new_ai_by_version": counts(new_b_versions),
            "b_union_candidate_sources": len(old_b | new_b), "b_human_confirmed_polygons": len(old_b_targets),
            "b_new_ai_pass_polygons_increment": len(new_b_targets - old_b_targets),
            "b_union_candidate_polygons": len(old_b_targets | new_b_targets),
            "planned_follow_up_sources": counts(unresolved_source_sets), "planned_follow_up_polygons": counts(pending_targets),
            "remaining_sources": counts(final_sources), "remaining_polygons": counts(final_targets),
            "remaining_unique_sources": len(set().union(*final_sources.values())),
            "remaining_sources_by_version": {state: dict(Counter(version for _, version in values))
                                             for state, values in final_versions.items()},
            "remaining_polygons_by_version": {state: dict(Counter(version for _, _, version in values))
                                              for state, values in final_version_targets.items()}}
    return summary, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-overview", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scope, auth, protocol = frozen_plan(root)
    checked_history, _, original, _ = history(root)
    g3.require(scope["sources"] == original["sources"] and
               all(r["human_evidence"] == checked_history[r["source_id"]] for r in scope["routes"]),
               "Frozen TRAIN or historical review changed")
    audit_frozen_train(scope["sources"], original)
    dispatch = audit_requests(root, scope, auth, protocol)
    records, images = audit_candidates(root, scope)
    prepared, observed = waves(root)
    g3.require(prepared.keys() == observed.keys() == records.keys(), "Visual review incomplete")
    overview, merged = candidate_overview(scope, records, observed)
    by_phase = defaultdict(Counter)
    for variant, result in observed.items():
        kind = "A7ALIGN" if records[variant].get("alignment_parent_review_id") else result["version"]
        by_phase[kind][result["ai_decision"]] += 1
    payload = {
        "phase": A7, "started_utc": scope["started_utc"], "input_review_sha256": list(scope["inputs"].values()),
        "status": ("EXP14_A7_V3_REPAIR_COMPLETE_PENDING_HUMAN_REVIEW"
                   if not dispatch["unattempted"] and images["sources_returned_images"] == dispatch["attempted"]
                   else "EXP14_A7_V3_REPAIR_PARTIAL_PENDING_HUMAN_REVIEW"),
        "train_sources": len(scope["sources"]),
        "train_target_polygons": sum(len(s["selected_polygon_ids"]) for s in scope["sources"].values()),
        "frozen_split_sha256": EXPECTED_MANIFEST_SHA256,
        "routes": dict(Counter(x["action"] for x in scope["routes"])),
        "deferred_targets": dict(Counter(reason for x in scope["routes"] for reason in x["deferred_targets"].values())),
        "request_versions": dict(Counter(x["version"] for x in scope["requests"])),
        "repair_targets": sum(len(x["repair_targets"]) for x in scope["routes"]),
        "protected_inputs": sum(bool(x["pixel_sources"]) for x in scope["requests"]),
        "dispatch": dispatch, "images": images, "visual_reviewed": len(observed),
        "ai_decisions": dict(Counter(x["ai_decision"] for x in observed.values())),
        "ai_decisions_by_kind": {kind: dict(counts) for kind, counts in by_phase.items()},
        "ai_polygons": dict(Counter(state for x in observed.values() for state in x["ai_polygon_decisions"].values())),
        "overview": overview, "new_human_approvals": 0, "training_runs": 0,
    }
    g3.write_json_exclusive(args.private_overview.resolve(), {"phase": A7, "summary": overview, "sources": merged})
    g3.write_json_exclusive(args.output.resolve(), payload)


if __name__ == "__main__":
    main()
