"""Verify the private A6 execution and generate aggregate counts."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tools.exp13_cloud.g3 import align_g3b2_batch as alignment
from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.postprocess_large_scale_repair import candidates, finished_candidates
from tools.exp14.prepare_large_scale_repair import canonical_digest, hash_file, load_json
from tools.exp14.targeted_v2 import frozen_plan, input_for_row, target_raster
from tools.exp14.targeted_v2_review import reviewable, verified_files, waves


def audit(root):
    scope, auth_sha, protocol_sha = frozen_plan(root)
    requests = {row["variant_id"]: row for row in scope["requests"]}
    locks = {path.stem for path in (root / "locks").glob("*.json")}
    folders = {path.name for path in (root / "live").iterdir() if path.is_dir()}
    g3.require(folders == requests.keys() and locks == {r["request_id"] for r in requests.values()},
               "Live folders or permanent locks differ from the authorized scope")
    parent_root = Path(scope["parent_root"])
    parents = finished_candidates(parent_root)
    routes = {r["source_id"]: r for r in scope["routes"]}
    states, usage, costs = Counter(), Counter(), Counter()
    source_checks = set()
    for variant, row in requests.items():
        sid = row["source_id"]
        source = scope["sources"][sid]
        inp = input_for_row(root, row, source)
        body, content_type = g3.encode_multipart(sid, row["model"], inp["prompt_wire"], inp["image_path"].name,
                                                  inp["image_bytes"], inp["image_format"], inp["mask_bytes"])
        folder = root / "live" / variant
        request = load_json(folder / "request_manifest.json")
        expected = {**row, "method": "POST", "body_sha256": g3.digest(body), "body_bytes": len(body),
                    "content_type": content_type, "authorization_sha256": auth_sha, "protocol_sha256": protocol_sha,
                    "automatic_retries": 0, "redirects": False}
        lock = load_json(root / "locks" / f"{row['request_id']}.json")
        terminal = load_json(folder / "TERMINAL.json")
        g3.require(request == expected and lock["request_sha256"] == canonical_digest(request) and
                   load_json(folder / "POST_STARTED.json")["post_invocations"] == terminal["post_invocations"] == 1,
                   "Request body, permanent lock or POST receipt mismatch")
        if (folder / "http_response.json").exists():
            http = load_json(folder / "http_response.json")
            g3.require(terminal["http_status"] == http["http_status"], "HTTP receipt mismatch")
            hash_file(folder / "response_body.bin", http["response_body_sha256"])
        response_paths = [folder / "response_manifest.json"]
        recovered = root / "response_recovery" / variant
        recovery = recovered / "RECOVERED.json"
        if recovery.is_file():
            evidence = load_json(recovery)
            g3.require(evidence["api_posts"] == 0, "Recovery recorded unexpected paid POST")
            response_paths = [recovered / entry["path"] / "response_manifest.json" for entry in evidence["outputs"]]
        states[str(terminal["http_status"])] += 1
        responses = [load_json(path) for path in response_paths if path.is_file()]
        if responses:
            for item in responses[1:]:
                g3.require(item["usage"] == responses[0]["usage"] and item["reported_cost"] == responses[0]["reported_cost"],
                           "One provider response has conflicting usage or cost across image positions")
            metadata = responses[0]
            costs[metadata["cost_status"]] += 1
            fields = metadata.get("usage") or {}
            usage["requests_with_usage"] += bool(fields)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if type(fields.get(key)) is int:
                    usage[key] += fields[key]
        else:
            costs[terminal["cost_status"]] += 1
        original = Path(source["image"])
        hash_file(original, row["original_image_sha256"])
        base = alignment.read_rgb(inp["image_path"])
        original_pixels = alignment.read_rgb(original)
        if row["preserved_polygon_ids"]:
            keep, _ = target_raster(source, load_json(Path(source["annotation"]))["shapes"], row["preserved_polygon_ids"])
            keep = np.asarray(keep) > 0
            prior = parents[routes[sid]["parent_variant"]]
            previous = parent_root / prior["asset_directory"] / "aligned.png"
            hash_file(previous, row["parent_aligned_sha256"])
            g3.require(np.array_equal(base[keep], alignment.read_rgb(previous)[keep]) and
                       np.array_equal(base[~keep], original_pixels[~keep]), "Paid request altered unapproved base pixels")
            source_checks.add(sid)
        else:
            g3.require(row["source_image_sha256"] == row["original_image_sha256"], "Original input hash changed")

    generated = candidates(root, {**scope, "historical_variants": {}})
    all_receipts = {p.stem: load_json(p) for p in (root / "postprocess_receipts").glob("*.json")}
    expected_rescue = {f"{r['parent_variant']}_A6ALIGN": r for r in routes.values() if r["alignment_targets"]}
    g3.require(all_receipts.keys() == generated.keys() | expected_rescue.keys(), "Missing postprocess receipt")
    for variant, item in generated.items():
        hash_file(Path(item["raw"]), item["raw_sha256"])
    records = reviewable(root)
    bounded_count = 0
    for variant, record in records.items():
        folder = root / record["asset_directory"]
        g3.require(load_json(folder / "record.json") == record, "Candidate receipt does not match review identity")
        verified_files(folder, record)
        if variant in expected_rescue:
            parent = parents[expected_rescue[variant]["parent_variant"]]
            g3.require(record["alignment_parent_review_id"] == parent["review_id"] and
                       record["raw_sha256"] == parent["raw_sha256"] and
                       record["aligned_sha256"] != parent["aligned_sha256"], "Historical Raw or Aligned identity mismatch")
        if "bounded.png" not in record["file_sha256"]:
            continue
        request = requests[variant.split("_OUT", 1)[0]]
        base = alignment.read_rgb(root / request["batch_image_path"])
        edit = alignment.read_rgb(root / request["mask_path"])[..., 0]
        # Mask alpha defines the exact approved edit, while RGB is intentionally white.
        g3.require(np.all(edit == 255), "Unexpected generated mask RGB")
        with Image.open(root / request["mask_path"]) as mask:
            selected = np.asarray(mask.getchannel("A")) == 0
        valid = alignment.read_rgb(folder / "valid.png")[..., 0] > 0
        changed = selected & valid
        bounded = alignment.read_rgb(folder / "bounded.png")
        aligned = alignment.read_rgb(folder / "aligned.png")
        g3.require(np.array_equal(bounded[~changed], base[~changed]) and
                   np.array_equal(bounded[changed], aligned[changed]), "Bounded result changed an unapproved pixel")
        bounded_count += 1

    _, observed = waves(root)
    decisions_by_origin = {}
    for origin, variants in (("generation", generated.keys()), ("prior_raw_alignment", expected_rescue.keys())):
        decisions_by_origin[origin] = dict(Counter(observed[v]["ai_decision"] for v in observed.keys() & variants))
    generated_sources = {r["source_id"] for r in generated.values()}
    repair_ids = {r["source_id"] for r in requests.values()}
    return {
        "phase": "EXP14_A6_TARGETED_V2_REPAIR_AND_ALIGNMENT", "input_csv_sha256": scope["human_review_sha256"],
        "started_utc": scope["started_utc"], "routes": dict(Counter(r["action"] for r in routes.values())),
        "deferred_repair_polygons": dict(Counter(reason for r in routes.values() for reason in r["deferred_targets"].values())),
        "paid_targets": sum(len(r["selected_polygon_ids"].split("|")) for r in requests.values()),
        "paid_sources": len(requests), "paid_posts_verified": len(locks), "repeat_posts": 0,
        "prior_v2_reused": sum(r["existing_version_reuse"] for r in routes.values()),
        "http": dict(states), "cost": dict(costs), "usage": dict(usage),
        "image_response_sources": len(generated_sources), "image_response_missing_sources": len(repair_ids - generated_sources),
        "new_raw": len(generated), "new_raw_sha256_unique": len({r["raw_sha256"] for r in generated.values()}),
        "new_raw_aligned": sum(bool(records[v]["aligned_sha256"]) for v in generated.keys() & records.keys()),
        "local_processing_errors": sum(r["status"] == "LOCAL_POSTPROCESS_ERROR" for r in all_receipts.values()),
        "prior_raw_alignment_attempts": len(expected_rescue),
        "prior_raw_alignment_outcomes": dict(Counter(all_receipts[v]["status"] for v in expected_rescue)),
        "bounded_aligned_verified": bounded_count, "protected_request_inputs_verified": len(source_checks),
        "reviewable": len(records), "visual_reviewed": len(observed),
        "ai_decisions": dict(Counter(r["ai_decision"] for r in observed.values())),
        "ai_decisions_by_origin": decisions_by_origin,
        "ai_polygons": dict(Counter(state for r in observed.values() for state in r["ai_polygon_decisions"].values())),
        "human_approvals": 0, "training_runs": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    g3.write_json_exclusive(args.output.resolve(), audit(args.root.resolve()))


if __name__ == "__main__":
    main()
