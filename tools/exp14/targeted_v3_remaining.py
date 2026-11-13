"""Continue the unconsumed A7 request IDs and isolate the supplemental review."""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.audit_targeted_v3 import audit_candidates, audit_frozen_train, audit_requests
from tools.exp14.large_scale_review import validate_review
from tools.exp14.postprocess_large_scale_repair import candidates
from tools.exp14.prepare_large_scale_repair import PHASE, PRIVATE_ROOT, hash_file, load_json
from tools.exp14.run_large_scale_repair import execute, now
from tools.exp14.targeted_v2 import input_for_row
from tools.exp14 import targeted_v2_review as review
from tools.exp14.targeted_v3 import frozen_plan
from tools.exp14.targeted_v3_alignment import process
from tools.exp14.targeted_v3_review import context, history_for_page, render

BATCH = "a7b_completion"
PHASE_B = "exp14_a7b_v3_completion"
PAGE = "review_v3_remaining_ai_assisted"
FIRST_CSV = "Exp14_review_AI_assisted_2026-11-13T12-04-02-854Z.csv"
FIRST_SHA = "ae2465227161dc9c78cbfa8b4a1a74c254c5fa2edd0395683de4a929abfe5225"
BASE_STOP = "THREE_CONSECUTIVE_RELAY_OR_TRANSPORT_FAILURES"


def verify_first(root, csv, records):
    hash_file(csv, FIRST_SHA)
    prior = load_json(root / "ai_review_results.json")["records"]
    first = {r["variant_id"]: records[r["variant_id"]] for r in prior}
    g3.require(len(first) == len(prior) == 23, "Unexpected first-batch record count")
    rows = validate_review(root, csv, records=first.values())
    g3.require({r["decision"]["review_id"] for r in rows} == {r["review_id"] for r in first.values()},
               "First-batch human export identities differ")
    for record in first.values():
        review.verified_files(root / record["asset_directory"], record)
    return first, rows


def preflight(root, csv):
    batch = root / BATCH
    scope, auth, protocol = frozen_plan(root)
    delivery = load_json(root / "delivery_receipt.json")
    for path, sha in delivery["sha256"].items():
        hash_file(g3.safe_file(root, path), sha)
    audit_frozen_train(scope["sources"], load_json(root.parent / PHASE / "scope.json"))
    dispatch = audit_requests(root, scope, auth, protocol)
    requests = scope["requests"]
    locks = {p.stem for p in (root / "locks").glob("*.json")}
    existing = review.reviewable(root)
    first, rows = verify_first(root, csv, existing)
    stopped = sorted((root / "logs").glob("stop_*.json"))
    started = sorted((root / "logs").glob("start_*.json"))
    first_stop = load_json(stopped[0])
    g3.require(len(stopped) == len(started) and first_stop["stop_reason"] == BASE_STOP and
               first_stop["counts_this_process"] == {"FAILED_NO_RETRY": 5, "RETURNED_IMAGE": 7},
               "Historical A7 stop receipt differs")
    initial = []
    for row in requests:
        if row["request_id"] in locks:
            terminal = load_json(root / "live" / row["variant_id"] / "TERMINAL.json")
            if terminal["finished_utc"] <= first_stop["finished_utc"]:
                initial.append(row)
    g3.require(len(initial) == 12 and Counter(r["version"] for r in requests) == {"V3": 93, "V2": 1},
               "Original A7 execution count differs")
    blocked_sources = {r["source_id"] for r in first.values()}
    pending = [r for r in requests if r["request_id"] not in locks]
    existing_sources = {r["source_id"] for r in existing.values()} | blocked_sources
    g3.require(len(pending) <= 82 and all(r["source_id"] not in existing_sources for r in pending),
               "A candidate or first-batch review already exists for a pending source")
    historical = [root / "delivery_receipt.json", *started, *stopped]
    for name in ("ai_review_results.csv", "ai_review_results.json", "authorization.json", "scope.json"):
        historical.append(root / name)
    historical.extend(p for p in (root / "review_v3_ai_assisted").rglob("*") if p.is_file())
    historical.extend(p for p in (root / "visual_waves").rglob("*") if p.is_file())
    for row in requests:
        if row["request_id"] in locks:
            historical.append(root / "locks" / (row["request_id"] + ".json"))
            historical.extend(p for p in (root / "live" / row["variant_id"]).rglob("*") if p.is_file())
    historical.extend((root / "postprocess_receipts").glob("*.json"))
    for record in first.values():
        folder = root / record["asset_directory"]
        historical.extend([folder / "record.json", *(folder / name for name in record["file_sha256"])])
    batch.mkdir()
    (batch / "logs").mkdir()
    (batch / "input").mkdir()
    g3.write_exclusive(batch / "input" / FIRST_CSV, csv.read_bytes())
    hash_file(batch / "input" / FIRST_CSV, FIRST_SHA)
    g3.write_json_exclusive(batch / "baseline.json", {
        "phase": PHASE_B, "started_utc": now(), "first_csv_path": str(csv.resolve()), "first_csv_sha256": FIRST_SHA,
        "authorization_sha256": auth, "protocol_sha256": protocol,
        "pending": [r["request_id"] for r in pending], "consumed": sorted(locks),
        "first_dispatch": [r["variant_id"] for r in initial],
        "first_reviews": {variant: r["review_id"] for variant, r in first.items()},
        "protected": {p.relative_to(root).as_posix(): hash_file(p) for p in sorted(set(historical))},
    })
    print("A7B preflight", dispatch["attempted"], "consumed,", len(pending), "pending",
          dict(Counter(r["version"] for r in pending)), "and", len(rows), "protected human rows", flush=True)


def batch_plan(root):
    baseline = load_json(root / BATCH / "baseline.json")
    scope, auth, protocol = frozen_plan(root)
    all_ids = {r["request_id"] for r in scope["requests"]}
    pending, consumed = set(baseline["pending"]), set(baseline["consumed"])
    g3.require(baseline["phase"] == PHASE_B and auth == baseline["authorization_sha256"] and
               protocol == baseline["protocol_sha256"] and not pending & consumed and
               pending | consumed == all_ids and len(pending) <= 82, "Supplemental scope changed")
    return {**scope, "requests": [r for r in scope["requests"] if r["request_id"] in pending]}, auth, protocol


def verify_protected(root):
    baseline = load_json(root / BATCH / "baseline.json")
    for path, sha in baseline["protected"].items():
        hash_file(g3.safe_file(root, path), sha)
    hash_file(Path(baseline["first_csv_path"]), FIRST_SHA)
    hash_file(root / BATCH / "input" / FIRST_CSV, FIRST_SHA)
    return baseline


def run(root, workers):
    logs = root / BATCH / "logs"
    g3.require(not list(logs.glob("start_*.json")), "A7B dispatch already started")
    verify_protected(root)
    scope, _, _ = batch_plan(root)
    g3.require(all(not (root / "locks" / (r["request_id"] + ".json")).exists() and
                   not (root / "live" / r["variant_id"]).exists() for r in scope["requests"]),
               "A pending request now has dispatch evidence")
    g3.reserve_lock(logs / "dispatch_started.json", {"started_utc": now(), "phase": PHASE_B})
    execute(root, os.environ.pop("EXP14_IMAGE_API_KEY", ""), workers, plan_loader=batch_plan,
            input_validator=input_for_row, log_dir=logs, require_first_image=True)


def incremental_records(root, *, require_complete=False):
    scope, _, _ = batch_plan(root)
    generated = candidates(root, {**scope, "historical_variants": {}}, running=not require_complete)
    records = {v: r for v, r in review.reviewable(root).items() if v in generated}
    baseline = load_json(root / BATCH / "baseline.json")
    first = {v: r for v, r in review.reviewable(root).items() if v in baseline["first_reviews"]}
    g3.require({v: r["review_id"] for v, r in first.items()} == baseline["first_reviews"] and
               not records.keys() & first.keys() and
               {r["review_id"] for r in records.values()}.isdisjoint(baseline["first_reviews"].values()) and
               {(r["raw_sha256"], r["aligned_sha256"]) for r in records.values()}.isdisjoint(
                   (r["raw_sha256"], r["aligned_sha256"]) for r in first.values()),
               "Supplemental review intersects the first batch")
    if require_complete:
        g3.require(records.keys() == generated.keys(), "Some new Raw images lack reviewable evidence")
        for variant, record in records.items():
            entry = generated[variant]
            source = scope["sources"][entry["source_id"]]
            g3.require(record["raw_sha256"] == entry["raw_sha256"] and
                       record["request_sha256"] == entry["request_sha256"] and
                       record["mask_sha256"] == entry["mask_sha256"] and
                       record["source_sha256"] == source["image_sha256"] and
                       record["annotation_sha256"] == source["annotation_sha256"] and
                       not record.get("alignment_parent_review_id"), "New review does not bind its frozen request and source")
    return records


def audit(root, output):
    batch = root / BATCH
    baseline = verify_protected(root)
    scope, auth, protocol = frozen_plan(root)
    dispatch = audit_requests(root, scope, auth, protocol)
    records, _ = audit_candidates(root, scope)
    first, human = verify_first(root, batch / "input" / FIRST_CSV, records)
    new = incremental_records(root, require_complete=True)
    prepared, observed = review.waves(root, batch / "visual_waves")
    published = load_json(batch / "ai_review_results.json")
    g3.require(new.keys() == prepared.keys() == observed.keys() and published["phase"] == PHASE_B and
               {r["variant_id"]: r for r in published["records"]} == observed,
               "Supplemental visual review is incomplete or differs from publication")
    for variant, result in observed.items():
        g3.require(result["ai_notes"].strip() and set(result["ai_polygon_decisions"]) ==
                   {p["polygon_id"] for p in new[variant]["polygons"]}, "Incomplete supplemental visual review")
    requests = {r["request_id"]: r for r in scope["requests"]}
    present = {p.stem for p in (root / "locks").glob("*.json")}
    added = present - set(baseline["consumed"])
    g3.require(added <= set(baseline["pending"]) and set(baseline["consumed"]) <= present,
               "Historical locks changed or a POST exceeded the A7B plan")
    stop_files = list((batch / "logs").glob("stop_*.json"))
    g3.require(len(stop_files) == len(list((batch / "logs").glob("start_*.json"))) == 1,
               "Expected exactly one supplemental dispatch")
    stopped = load_json(stop_files[0])
    delivering = {v.split("_OUT", 1)[0] for v in new}
    new_status = Counter()
    exceptions = []
    for rid in sorted(present):
        row = requests[rid]
        variant = row["variant_id"]
        folder = root / "live" / variant
        terminal = load_json(folder / "TERMINAL.json")
        if rid in added:
            new_status[str(terminal["http_status"])] += 1
        if terminal["status"] != "RETURNED_IMAGE" and variant not in delivering:
            exceptions.append({"batch": "A7B" if rid in added else "A7", "variant_id": variant,
                               "request_id": rid, "terminal": terminal})
    g3.require(sum(stopped["counts_this_process"].values()) == len(added), "New terminal receipts differ from executor counts")
    new_variants = {requests[r]["variant_id"] for r in added}
    g3.require(delivering <= new_variants, "Supplemental image does not bind a new POST")
    unsent = set(baseline["pending"]) - added
    labels = Counter(r["ai_decision"] for r in observed.values())
    payload = {
        "experiment": "EXP14_A7B_V3_COMPLETION",
        "status": ("EXP14_A7B_V3_COMPLETION_PARTIAL_PENDING_HUMAN_REVIEW" if unsent
                   else "EXP14_A7B_V3_COMPLETION_PENDING_HUMAN_REVIEW"),
        "started_utc": baseline["started_utc"], "dispatch": "PARTIAL" if unsent else "DISPATCH_COMPLETE",
        "delivery": "IMAGE_DELIVERY_PARTIAL" if exceptions or unsent else "IMAGE_DELIVERY_COMPLETE",
        "review": "HUMAN_REVIEW_PENDING" if new else "NO_NEW_REVIEW_RECORDS",
        "first_human_csv_sha256": FIRST_SHA, "first_human_records": len(human),
        "original_frozen_requests": len(requests), "original_a7_posts": len(baseline["first_dispatch"]),
        "previous_posts": len(baseline["consumed"]), "available_at_start": len(baseline["pending"]),
        "new_posts": len(added), "cumulative_posts": dispatch["attempted"], "still_unsent": len(unsent),
        "new_posts_by_version": dict(Counter(requests[r]["version"] for r in added)),
        "new_http": dict(new_status), "new_delivering_requests": len(delivering),
        "new_failed_delivery": len(added) - len(delivering), "cumulative_failed_delivery": len(exceptions),
        "stop_reason": stopped["stop_reason"], "repeat_posts": 0, "new_raw": len(new),
        "new_aligned": sum(bool(r["aligned_sha256"]) for r in new.values()),
        "new_bounded": sum("bounded.png" in r["file_sha256"] for r in new.values()),
        "new_review_ids": len({r["review_id"] for r in new.values()}),
        "first_review_id_intersection": len({r["review_id"] for r in new.values()} & {r["review_id"] for r in first.values()}),
        "ai_decisions": {key: labels[key] for key in ("PASS", "RETRY_REPAIR", "RETRY_ALIGNMENT", "HOLD")},
        "ai_polygon_decisions": dict(Counter(v for r in observed.values() for v in r["ai_polygon_decisions"].values())),
        "new_human_approvals": 0, "v4_requests": 0, "gpu_training_runs": 0,
    }
    g3.write_json_exclusive(batch / "delivery_exceptions.json", {
        "phase": PHASE_B, "attempted_without_image": exceptions,
        "unsent": [{k: requests[rid][k] for k in ("variant_id", "request_id", "version")} for rid in sorted(unsent)],
    })
    g3.write_json_exclusive(output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "live", "align", "prepare", "publish", "verify-export", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    g3.require(root.parent == PRIVATE_ROOT.resolve(), "A7B needs the private A7 run")
    batch = root / BATCH
    wave_dir = batch / "visual_waves"
    if args.action == "preflight":
        if args.csv is None:
            parser.error("preflight requires --csv")
        preflight(root, args.csv)
    elif args.action == "live":
        run(root, args.workers)
    elif args.action == "align":
        process(root, True, False, plan_loader=batch_plan, log_dir=batch / "logs", rescue=False)
    elif args.action == "prepare":
        review.prepare(root, args.limit, plan_loader=batch_plan, build_context=context, render=render,
                       records=incremental_records(root), wave_dir=wave_dir)
    elif args.action == "publish":
        verify_protected(root)
        g3.require(len(list((batch / "logs").glob("stop_*.json"))) == 1, "Dispatch is still active")
        scope, _, _ = batch_plan(root)
        review.publish(root, phase=PHASE_B, name=PAGE, title="Exp14-A7B 第三轮补充复核",
                       history=history_for_page(scope), records=incremental_records(root, require_complete=True),
                       wave_dir=wave_dir, results_dir=batch,
                       empty_message="本批未取得可供复核的图像。发送结果见 A7B Handoff。")
    elif args.action == "verify-export":
        if args.export is None:
            parser.error("verify-export requires --export")
        records = incremental_records(root, require_complete=True)
        rows = validate_review(root, args.export, phase=PHASE_B, records=records.values())
        g3.require(len(rows) == len(records), "Supplemental export does not cover this batch")
        print("Verified A7B export", len(rows))
    else:
        audit(root, args.output or batch / "summary.json")


if __name__ == "__main__":
    main()
