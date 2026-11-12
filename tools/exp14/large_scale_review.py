"""Publish an offline review bundle and validate exported decisions for Exp14 follow-up."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import _csv_bytes
from tools.exp13_cloud.g3.prepare_g3c2_v2 import PROMPTS, prompt_entry
from tools.exp13_cloud.g3.verify_train_export import CLASSES
from tools.exp14.postprocess_large_scale_repair import build_candidate, candidates, finished_candidates
from tools.exp14.prepare_large_scale_repair import PHASE, PRIVATE_ROOT, REPO, canonical_digest, hash_file, load_json
from tools.exp14.run_large_scale_repair import frozen_plan

IDENTITY = (
    "review_id", "variant_id", "source_id", "version", "source_sha256", "annotation_sha256",
    "mask_sha256", "request_sha256", "raw_sha256", "aligned_sha256",
)
FIELDS = (*IDENTITY, "decision", "reason", "notes", "whole_image_normal", "b_region_review", "polygon_decisions")
STATES = {"UNREVIEWED", "PASS", "RETRY_REPAIR", "RETRY_ALIGNMENT", "HOLD"}
REASONS = {"", "缺陷残留", "机械结构错误", "缺材未补全", "新增异常内容", "配准错位", "黑边截断", "相机 UI 变化", "其他"}


def initial_row(record):
    historical = record["a2_status"] == "CORE_SOURCE_SCREEN_PASS"
    hold = record["source_hold"] or record["a2_status"] == "TRAINING_HOLD"
    failed = record["historical_visual"] == "NEEDS_RETRY"
    decision = "PASS" if historical else "HOLD" if hold else "RETRY_REPAIR" if failed else "UNREVIEWED"
    return {
        **{key: record[key] or "" for key in IDENTITY},
        "decision": decision, "reason": "", "notes": "",
        "whole_image_normal": "YES" if historical else "",
        "b_region_review": "CONFIRMED" if historical else "",
        "polygon_decisions": "{}",
    }


def publish(root, name):
    frozen_plan(root)
    records = list(finished_candidates(root).values())
    records.sort(key=lambda item: (int(item["source_id"]), item["version"], item["variant_id"]))
    g3.require(records, "No usable review records")
    g3.require(re.fullmatch(r"review(?:_[a-zA-Z0-9_-]+)?", name) is not None, "Unsafe review release name")
    destination = root / name
    data = {"phase": PHASE, "class_names": CLASSES, "identity": IDENTITY, "fields": FIELDS,
            "states": sorted(STATES), "reasons": sorted(REASONS), "records": []}
    for record in records:
        folder = root / record["asset_directory"]
        for asset, sha in record["file_sha256"].items():
            hash_file(g3.safe_file(folder, asset), sha)
        public = {key: record[key] for key in (
            "variant_id", "source_id", "version", "origin", "classes", "polygons", "original", "raw",
            "geometry_status", "geometry_reason", "mapping_method", "match_count", "valid_coverage",
            "unsupported_target_pixels", "unsupported_bright_original_pixels", "historical_visual",
            "a1_status", "a2_status", "historical_support", "ui", "source_hold",
        )}
        public["model_metadata"] = record.get("model_metadata", "")
        public["reported_model"] = record.get("reported_model")
        data["records"].append({**public, "assets": "../" + record["asset_directory"] + "/",
                                "aligned": bool(record["aligned_sha256"]), "initial": initial_row(record)})
    destination.mkdir(exist_ok=False)
    (destination / "exports").mkdir()
    template = (REPO / "tools/exp14/large_scale_review.html").read_text(encoding="utf-8")
    script = (REPO / "tools/exp14/large_scale_review.js").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__BUNDLE_JSON__", payload).replace("__REVIEW_SCRIPT__", script)
    g3.write_exclusive(destination / "index.html", page.encode("utf-8"))
    initial = [item["initial"] for item in data["records"]]
    g3.write_exclusive(destination / "review_template.csv", _csv_bytes(initial, FIELDS))
    g3.write_json_exclusive(destination / "review_template.json", {"phase": PHASE, "records": initial})
    instructions = (
        "双击上一级 index.html 开始。页面可离线使用。审核完成后点击导出 CSV 或备份 JSON。\n"
        "浏览器通常下载到 Downloads。请把导出文件复制到本 exports 目录，并保留文件名和历史版本。\n"
        "重新打开页面，点击导入 CSV / JSON 选择已导出的文件。继续审核后另存一份。\n"
        "把导出的文件交给下一次 Codex。程序将按请求、Raw、配准图和 Polygon 身份验证决定。\n"
        "整图正常选项要求人工确认画面中不存在全部七类缺陷。局部 PASS 自身不构成该确认。\n"
    )
    g3.write_exclusive(destination / "exports" / "保存与恢复.txt", instructions.encode("utf-8"))
    print(f"OFFLINE_REVIEW_READY {len(records)} variants from {len({r['source_id'] for r in records})} sources")


def validate_review(root, path):
    current = {record["review_id"]: record for record in finished_candidates(root).values()}
    if path.suffix.lower() == ".json":
        data = load_json(path)
        g3.require(data["phase"] == PHASE, "Review JSON belongs to a different phase")
        rows = data["records"]
    else:
        header, rows = g3.read_csv(path)
        g3.require(tuple(header) == FIELDS, "Unexpected review CSV schema")
    seen = set()
    accepted = []
    for row in rows:
        g3.require(set(row) == set(FIELDS) and all(isinstance(value, str) for value in row.values()),
                   "Invalid review row schema")
        record = current.get(row["review_id"])
        g3.require(record is not None and row["review_id"] not in seen, "Unknown, stale or duplicate review identity")
        seen.add(row["review_id"])
        g3.require(all(row[key] == (record[key] or "") for key in IDENTITY), "Review file identity mismatch")
        g3.require(row["decision"] in STATES and row["reason"] in REASONS, "Invalid decision or reason")
        g3.require(row["whole_image_normal"] in {"", "YES"} and row["b_region_review"] in {"", "CONFIRMED"},
                   "Invalid human confirmation field")
        polygon = json.loads(row["polygon_decisions"])
        g3.require(isinstance(polygon, dict) and set(polygon) <= {p["polygon_id"] for p in record["polygons"]} and
                   all(isinstance(value, str) and value in STATES - {"UNREVIEWED"} for value in polygon.values()),
                   "Unknown Polygon ID or decision")
        g3.require(not row["whole_image_normal"] or
                   (row["decision"] == "PASS" and all(state == "PASS" for state in polygon.values())),
                   "Whole-image normal confirmation conflicts with a target decision")
        g3.require(not row["b_region_review"] or
                   (record["aligned_sha256"] and (row["decision"] == "PASS" or "PASS" in polygon.values())),
                   "Regional confirmation requires aligned output and a passed target")
        # Check the actual selected artifact as well as the stored review identity.
        if record["aligned_sha256"]:
            hash_file(root / record["asset_directory"] / "aligned.png", record["aligned_sha256"])
        hash_file(root / record["asset_directory"] / record["raw"], record["raw_sha256"])
        decisions_by_target = {p["polygon_id"]: polygon.get(p["polygon_id"], row["decision"])
                               for p in record["polygons"]}
        routes = set(decisions_by_target.values())
        retry_targets = [p for p in record["polygons"] if decisions_by_target[p["polygon_id"]] == "RETRY_REPAIR"]
        prompt_options = {
            label: prompt_entry(REPO, label)
            for label in sorted({p["class"] for p in retry_targets})
            if record["version"] == "V1" and label in PROMPTS
        }
        next_steps = []
        if "RETRY_ALIGNMENT" in routes:
            next_steps.append("REPROCESS_EXISTING_RAW")
        if "RETRY_REPAIR" in routes:
            next_steps.append("PROPOSE_INDEPENDENT_PROMPT_VERSION_FROM_HUMAN_EVIDENCE")
        if "HOLD" in routes:
            next_steps.append("RETAIN_HOLD")
        if "PASS" in routes:
            next_steps.append("PRESERVE_PASSED_IMAGE_OR_REGION")
        accepted.append({"decision": row, "next_steps": next_steps, "repair_failure_reason": row["reason"],
                         "raw_path": str(root / record["asset_directory"] / record["raw"]),
                         "human_polygon_decisions": polygon, "resolved_target_decisions": decisions_by_target,
                         "prepared_prompt_options": prompt_options,
                         "prompt_selection_requires_explicit_version_request": True})
    g3.require(accepted, "Review export has no rows")
    return accepted


def import_review(root, path):
    sha = hash_file(path)
    accepted = validate_review(root, path)
    hash_file(path, sha)
    directory = root / "review_imports" / f"{time.time_ns()}_{sha[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    g3.write_exclusive(directory / ("review" + path.suffix.lower()), path.read_bytes())
    g3.write_json_exclusive(directory / "follow_up_plan.json", {
        "phase": PHASE, "review_sha256": sha, "review_filename": path.name, "api_posts": 0, "records": accepted,
        "decisions": dict(Counter(entry["decision"]["decision"] for entry in accepted)),
        "repair_failure_reasons": dict(Counter(entry["repair_failure_reason"] for entry in accepted
            if "PROPOSE_INDEPENDENT_PROMPT_VERSION_FROM_HUMAN_EVIDENCE" in entry["next_steps"])),
    })
    print(f"REVIEW_IMPORT_PASS {len(accepted)} rows; follow-up plan {directory.name}")


def reprocess_alignment(root, path):
    import cv2

    cv2.setNumThreads(2)
    accepted = validate_review(root, path)
    scope, _, _ = frozen_plan(root)
    entries = candidates(root, scope)
    records = {record["review_id"]: record for record in finished_candidates(root).values()}
    outcomes = []
    for item in accepted:
        if "REPROCESS_EXISTING_RAW" not in item["next_steps"]:
            continue
        parent = records[item["decision"]["review_id"]]
        generation_variant = parent.get("generation_variant_id", parent["variant_id"])
        entry = {**entries[generation_variant], "a1_status": [], "a2_status": "",
                 "historical_visual": "", "historical_support": ""}
        revision = f"{generation_variant}_ALIGN_{parent['review_id'][:12]}"
        if any(r["variant_id"] == revision for r in records.values()):
            continue
        result = build_candidate(root, revision, entry, scope["sources"][parent["source_id"]], parent)
        outcomes.append({"parent_review_id": parent["review_id"], "revision": revision,
                         "status": "NEW_REVIEW_CANDIDATE" if result else "UNCHANGED"})
    g3.write_json_exclusive(root / "logs" / f"alignment_followup_{time.time_ns()}.json",
                            {"review_sha256": hash_file(path), "api_posts": 0, "outcomes": outcomes})
    print(f"REPROCESS_ALIGNMENT {len(outcomes)} reviewed variants; POST=0")


def summarize(root):
    scope, _, _ = frozen_plan(root)
    sources = scope["sources"]
    historical = scope["historical_variants"]
    variants = candidates(root, scope)
    records = {key: record for key, record in finished_candidates(root).items() if key in variants}
    stops = sorted((root / "logs").glob("stop_*.json"))
    starts = sorted((root / "logs").glob("start_*.json"))
    execution_terminal = bool(stops and starts and stops[-1].stat().st_mtime >= starts[-1].stat().st_mtime)
    stop_reason = load_json(stops[-1])["stop_reason"] if execution_terminal else None
    lock_ids = {path.stem for path in (root / "locks").glob("*.json")}
    request_ids = {row["request_id"] for row in scope["requests"]}
    g3.require(lock_ids <= request_ids, "Live lock outside authorized request set")
    live_ids = {path.name for path in (root / "live").iterdir() if path.is_dir()} if (root / "live").exists() else set()
    g3.require(live_ids <= {row["variant_id"] for row in scope["requests"]},
               "Live variant outside authorized request set")
    request_counts, failures, usage, costs, model_metadata = (
        Counter(), Counter(), Counter(), Counter(), Counter()
    )
    confirmed_http = 0
    per_source = defaultdict(list)
    for variant, entry in variants.items():
        per_source[entry["source_id"]].append(variant)
    request_states = {}
    request_by_source = {}
    dispatched = set()
    for row in scope["requests"]:
        variant = row["variant_id"]
        directory = root / "live" / variant
        terminal = directory / "TERMINAL.json"
        locked = (root / "locks" / f"{row['request_id']}.json").is_file()
        started = (directory / "POST_STARTED.json").is_file()
        request_counts["permanent_locks"] += int(locked)
        request_counts["post_invocations_recorded"] += int(started)
        confirmed_http += int((directory / "http_response.json").is_file())
        status = "NOT_STARTED"
        if terminal.is_file():
            result = load_json(terminal)
            request = load_json(directory / "request_manifest.json")
            permanent = load_json(root / "locks" / f"{row['request_id']}.json")
            g3.require(started and load_json(directory / "POST_STARTED.json")["post_invocations"] == 1 and
                       request["request_id"] == row["request_id"] and row["request_id"] not in dispatched and
                       permanent["request_sha256"] == canonical_digest(request),
                       "POST receipt does not bind a unique authorized request and lock")
            dispatched.add(row["request_id"])
            status = result["status"]
            request_counts["uncertain_delivery"] += int(result["delivery_uncertain"])
            recovery = root / "response_recovery" / variant
            if (recovery / "RECOVERED.json").is_file():
                recovered = load_json(recovery / "RECOVERED.json")
                outputs = recovered.get("outputs", [{"path": "."}])
                request_counts["multi_image_responses"] += int(recovered.get("provider_returned_positions", 1) > 1)
                if outputs:
                    status = "RETURNED_IMAGE"
                    directory = recovery / outputs[0]["path"]
                    request_counts["images_recovered_without_post"] += len(outputs)
            if status != "RETURNED_IMAGE":
                code = str(result["http_status"] or result.get("error_type") or "UNKNOWN")
                failures[code] += 1
            else:
                response = load_json(directory / "response_manifest.json")
                model_metadata[response.get("model_metadata",
                                           "UNREPORTED" if response["response_model"] is None else "MATCH")] += 1
                if isinstance(response["usage"], dict):
                    request_counts["usage_returned"] += 1
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        value = response["usage"].get(key)
                        if isinstance(value, int):
                            usage[key] += value
                costs[response["cost_status"]] += 1
        elif locked or directory.exists():
            status = "LOCKED_NO_TERMINAL"
            request_counts["uncertain_delivery"] += 1
        request_states[variant] = status
        request_by_source[row["source_id"]] = variant
        request_counts[status] += 1

    version_history = Counter(entry["version"] for entry in historical.values())
    new_entries = [entry for entry in variants.values() if entry["origin"] == "EXP14"]
    new_requests_with_images = {entry["request_id"]: entry["version"] for entry in new_entries}
    new_records = [record for record in records.values() if record["origin"] == "EXP14"]
    geometry = Counter(record["geometry_status"] for record in records.values())
    aligned = [record for record in records.values() if record["aligned_sha256"]]
    preexisting = [record for record in aligned if historical.get(record["variant_id"], {}).get("derived", {}).get("aligned_output")]
    unverified = Counter(record["geometry_reason"] or "HISTORICAL_GEOMETRY_UNVERIFIED" for record in records.values()
                         if record["geometry_status"] == "ALIGNMENT_UNVERIFIED")
    class_counts = {}
    original_class_counts = Counter()
    for source in sources.values():
        classes = source["classes"]
        if len(source["all_polygon_ids"]) != len(source["selected_polygon_ids"]):
            annotation = Path(source["annotation"])
            hash_file(annotation, source["annotation_sha256"])
            classes = {shape["label"] for shape in load_json(annotation)["shapes"]}
        original_class_counts.update(classes)
    for label in CLASSES:
        class_counts[label] = {"sources_with_selected_targets": sum(label in s["classes"] for s in sources.values()),
                               "sources_with_class_present": original_class_counts[label],
                               "polygons": sum(list(s["polygon_labels"].values()).count(label) for s in sources.values())}
    source_rows = [
        {"source_id": sid, "variants": "|".join(sorted(per_source[sid])),
         "new_request": request_by_source.get(sid, ""),
         "hold": source.get("hold", ""), "has_native_raw": bool(per_source[sid]),
         "geometry_candidate_variants": "|".join(v for v in per_source[sid] if records.get(v, {}).get("aligned_sha256"))}
        for sid, source in sources.items()
    ]
    for row in source_rows:
        row["new_request_status"] = request_states.get(row["new_request"], "")
    summary = {
        "phase": "EXP14_LARGE_SCALE_REPAIR_EXECUTION",
        "status": "EXP14_LARGE_SCALE_REPAIR_COMPLETED_PENDING_HUMAN_REVIEW" if execution_terminal else
                  "EXP14_LARGE_SCALE_REPAIR_IN_PROGRESS",
        "relay_stop_reason": stop_reason,
        "selected_sources": len(sources), "class_counts": class_counts,
        "selected_polygons": sum(v["polygons"] for v in class_counts.values()),
        "frozen_decision_counts": scope["provenance"]["decisions"],
        "select_earlier_backup_decisions": dict(Counter(s["earlier_backup_decision"] for s in sources.values())),
        "historical_native_sources": len({entry["source_id"] for entry in historical.values()}),
        "historical_native_variants_by_version": dict(version_history),
        "earlier_roi_exposure_sources": sum(bool(s.get("prior_roi")) for s in sources.values()),
        "planned_request_versions": dict(Counter(r["version"] for r in scope["requests"])),
        "request_counts": dict(request_counts), "http_responses_confirmed": confirmed_http,
        "terminal_post_receipts_verified": len(dispatched),
        "request_failure_categories": dict(failures), "reported_usage_totals": dict(usage),
        "returned_image_cost_states": dict(costs), "response_model_metadata": dict(model_metadata),
        "automatic_retry_posts": 0,
        "new_returned_sources": len({entry["source_id"] for entry in new_entries}),
        "new_request_variants_with_images_by_version": dict(Counter(new_requests_with_images.values())),
        "new_returned_images_by_version": dict(Counter(entry["version"] for entry in new_entries)),
        "total_sources_with_native_raw": sum(bool(items) for items in per_source.values()),
        "sources_without_native_raw": len(sources) - len(per_source),
        "hold_reasons": dict(Counter(s["hold"] for s in sources.values() if s.get("hold"))),
        "review_variants": len(records), "geometry_status_counts": dict(geometry),
        "alignment_unverified_reasons": dict(unverified),
        "raw_variants_missing_postprocess": len(variants) - len(records),
        "aligned_sources": len({r["source_id"] for r in aligned}), "aligned_variants": len(aligned),
        "aligned_historical_exact_reuse": len(preexisting),
        "aligned_historical_raw_reprocessed": sum(r["origin"] != "EXP14" for r in aligned) - len(preexisting),
        "new_aligned_variants": sum(bool(r["aligned_sha256"]) for r in new_records),
        "new_aligned_sources": len({r["source_id"] for r in new_records if r["aligned_sha256"]}),
        "sources_without_aligned": len(sources) - len({r["source_id"] for r in aligned}),
        "aligned_with_invalid_frame_pixels": sum(r["valid_coverage"] < 1 for r in aligned),
        "aligned_with_unsupported_target_pixels": sum(r["unsupported_target_pixels"] > 0 for r in aligned),
        "aligned_with_unsupported_visible_original_pixels": sum(r["unsupported_bright_original_pixels"] > 0 for r in aligned),
        "historical_a2_core_pass": sum(r["a2_status"] == "CORE_SOURCE_SCREEN_PASS" for r in aligned),
        "historical_a2_support_hold": sum(r["a2_status"] == "TRAINING_HOLD" for r in aligned),
        "new_human_approvals": 0, "new_trusted_pair_pass": 0,
    }
    if not execution_terminal and (root / "locks").exists():
        summary["status"] = "EXP14_LARGE_SCALE_REPAIR_IN_PROGRESS"
    return summary, source_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "import", "reprocess-alignment", "summary"))
    parser.add_argument("--bundle", default="review")
    parser.add_argument("--review-export", type=Path)
    args = parser.parse_args()
    root = (PRIVATE_ROOT / PHASE).resolve()
    if args.action == "publish":
        publish(root, args.bundle)
    elif args.action in {"import", "reprocess-alignment"}:
        if not args.review_export:
            parser.error("--review-export is required for this action")
        operation = import_review if args.action == "import" else reprocess_alignment
        operation(root, args.review_export)
    else:
        summary, rows = summarize(root)
        tag = str(time.time_ns())
        g3.write_json_exclusive(root / "logs" / f"summary_{tag}.json", summary)
        g3.write_exclusive(root / "logs" / f"source_status_{tag}.csv", _csv_bytes(rows, list(rows[0])))
        print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
