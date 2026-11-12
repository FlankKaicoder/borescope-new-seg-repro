"""Publish immutable AI suggestions beside the original offline human-review workflow."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from tools.exp13_cloud.g3.g3_batch_executor import write_exclusive, write_json_exclusive
from tools.exp13_cloud.g3.prepare_g3b2a_batch import _csv_bytes
from tools.exp13_cloud.g3.verify_train_export import CLASSES
from tools.exp14.ai_visual_review import MANIFEST, PARTIAL, READY, RESULT_FIELDS, load_observations, now, require
from tools.exp14.export_ai_review_pilot import frozen_bundle
from tools.exp14.large_scale_review import FIELDS, PHASE
from tools.exp14.prepare_large_scale_repair import REPO, hash_file, load_json


def relpath(path, start):
    return Path(os.path.relpath(path, start)).as_posix()


def publish_page(output, root, records, suggestions, destination):
    original = root / "review/index.html"
    text = original.read_text(encoding="utf-8")
    left = '<script id="bundle" type="application/json">'
    begin = text.index(left) + len(left)
    bundle = json.loads(text[begin:text.index("</script>", begin)])
    require([row["variant_id"] for row in bundle["records"]] == list(records), "Frozen page order changed")
    for page in bundle["records"]:
        item = records[page["variant_id"]]
        page["assets"] = relpath(root / item["asset_directory"], destination) + "/"
        if page["variant_id"] in suggestions:
            result = suggestions[page["variant_id"]]
            page["ai"] = {key: value for key, value in result.items()
                          if key.startswith("ai_") or key == "needs_human_attention"}
            page["ai"]["sheet_url"] = relpath(output / item["visual_sheet"], destination)
    bundle["ai_review"] = {"completed": len(suggestions), "total": len(records)}
    payload = json.dumps(bundle, ensure_ascii=False).replace("<", "\\u003c")
    template = (REPO / "tools/exp14/large_scale_review.html").read_text(encoding="utf-8")
    script = (REPO / "tools/exp14/large_scale_review.js").read_text(encoding="utf-8")
    destination.mkdir(exist_ok=False)
    write_exclusive(destination / "index.html", template.replace("__BUNDLE_JSON__", payload)
                    .replace("__REVIEW_SCRIPT__", script).encode("utf-8"))
    write_json_exclusive(destination / "review_template.json", {"phase": PHASE,
                         "records": [page["initial"] for page in bundle["records"]]})
    write_exclusive(destination / "review_template.csv", _csv_bytes([page["initial"] for page in bundle["records"]], FIELDS))
    write_exclusive(destination / "exports/保存与恢复.txt", (
        "双击 index.html。AI_DRAFT 是独立建议。点击采纳可提交当前未审核项，也可载入建议、修改后点确认人工决定。\n"
        "原审核状态保留。原页面若有未导出的浏览器进度，请先从原页面导出，再导入本页。\n"
        "正式 CSV 导出只含已有人工记录和本页已确认值。JSON 备份也保存未确认草稿、确认来源和重点标记。\n"
        "整图正常和区域确认由你分别勾选并确认。保存下载文件即可从新会话恢复。\n"
    ).encode("utf-8"))


def missing_result(item):
    return {**item["identity"], "class_names": item["class_names"],
            "polygon_ids": [p["polygon_id"] for p in item["polygons"]],
            "geometry_status": item["geometry_status"], "ai_decision": "NOT_REVIEWED",
            "ai_reason": "", "ai_notes": "", "ai_confidence": "",
            "ai_a_candidate": "", "ai_b_candidate": "", "ai_polygon_decisions": {},
            "visual_evidence_paths": {}, "visual_review_completed": False, "needs_human_attention": False}


def historical_comparison(suggestions, controls):
    rows = []
    for row in load_json(controls)["records"]:
        result = suggestions.get(row["variant_id"])
        if not row["reference_consistency_eligible"] or result is None:
            continue
        require(result["review_id"] == row["review_id"], "Historical comparison refers to a different review version")
        rows.append({"variant_id": row["variant_id"], "review_id": row["review_id"],
                     "historical": row["frozen_review"]["decision"], "ai": result["ai_decision"],
                     "notes": result["ai_notes"]})
    agree = sum(row["historical"] == row["ai"] for row in rows)
    return {"comparisons": len(rows), "exact_agreement": agree, "different": len(rows) - agree, "rows": rows,
            "interpretation": "Historical HOLD includes framing/support constraints. This comparison is not blinded or a general accuracy estimate."}


def summarize(results, comparison):
    reviewed = [row for row in results if row["visual_review_completed"]]
    a = {r["source_id"] for r in reviewed if r["ai_a_candidate"] == "YES"}
    b = {r["source_id"] for r in reviewed if r["ai_b_candidate"] == "YES"}
    reasons = Counter(row["ai_reason"] or "其他" for row in reviewed if row["ai_decision"] != "PASS")
    return {
        "status": READY if len(reviewed) == len(results) else PARTIAL,
        "records": len(results), "sources": len({row["source_id"] for row in results}), "reviewed": len(reviewed),
        "reviewed_sources": len({row["source_id"] for row in reviewed}), "not_reviewed": len(results) - len(reviewed),
        "ai_decisions": dict(Counter(row["ai_decision"] for row in results)),
        "reviewed_geometry": dict(Counter(row["geometry_status"] for row in reviewed)),
        "class_counts": {label: {
            "records": sum(label in r["class_names"] for r in results),
            "reviewed_records": sum(label in r["class_names"] for r in reviewed),
            "reviewed_sources": len({r["source_id"] for r in reviewed if label in r["class_names"]}),
            "ai_decisions": dict(Counter(r["ai_decision"] for r in reviewed if label in r["class_names"])),
        } for label in CLASSES},
        "failure_reasons": dict(reasons),
        "a_candidate_sources": len(a), "b_candidate_sources": len(b), "a_and_b_candidate_sources": len(a & b),
        "human_priority_sources": len({r["source_id"] for r in reviewed if r["needs_human_attention"]}),
        "historical_comparison": {key: comparison[key] for key in ("comparisons", "exact_agreement", "different")},
        "reviewed_polygons": sum(len(r["ai_polygon_decisions"]) for r in reviewed),
        "api_post": 0, "new_alignment": 0, "gpu_training": 0, "human_approvals_written": 0,
    }


def publish(output, controls, preview):
    manifest = load_json(output / MANIFEST)
    root = Path(manifest["repair_root"])
    pages, _, frozen_hashes = frozen_bundle(root)
    require(frozen_hashes == manifest["frozen_hashes"], "Frozen human review source changed")
    unordered = {item["variant_id"]: item for item in manifest["records"]}
    require(len(unordered) == len(manifest["records"]) == len(pages) == 530 and set(unordered) == set(pages),
            "Review identity exact-set changed")
    records = {variant: unordered[variant] for variant in pages}
    suggestions = load_observations(output, manifest)
    for variant, item in records.items():
        require(item["identity"]["review_id"] == pages[variant]["initial"]["review_id"], "Review identity changed")
        for file, sha in item["file_sha256"].items():
            hash_file(root / item["asset_directory"] / file, sha)
    for sheet in manifest["sheets"]:
        hash_file(output / sheet["sheet"], sheet["sha256"])
    if preview:
        publish_page(output, root, records, suggestions, output / "review_ai_assisted_qa")
        return
    results = [suggestions.get(variant) or missing_result(item) for variant, item in records.items()]
    comparison = historical_comparison(suggestions, controls)
    summary = summarize(results, comparison)
    write_json_exclusive(output / "ai_review_results.json", {"records": results, "summary": summary})
    serial = [{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, bool)) else value)
               for key, value in row.items()} for row in results]
    write_exclusive(output / "ai_review_results.csv", _csv_bytes(serial, RESULT_FIELDS))
    write_json_exclusive(output / "ai_review_summary.json", summary)
    write_json_exclusive(output / "logs/historical_comparison.json", comparison)
    routes = {"V2_REPAIR": [], "ALIGNMENT_REPAIR": [], "HUMAN_REVIEW_PRIORITY": []}
    for row in suggestions.values():
        entry = {key: row[key] for key in ("review_id", "variant_id", "source_id", "ai_reason", "ai_notes")}
        for state, route in (("RETRY_REPAIR", "V2_REPAIR"), ("RETRY_ALIGNMENT", "ALIGNMENT_REPAIR")):
            targets = [key for key, decision in row["ai_polygon_decisions"].items() if decision == state]
            if row["ai_decision"] == state or targets:
                routes[route].append({**entry, "polygon_ids": targets})
        if row["needs_human_attention"]:
            routes["HUMAN_REVIEW_PRIORITY"].append(entry)
    write_json_exclusive(output / "follow_up_candidates.json", {
        "status": "HUMAN_CONFIRMATION_REQUIRED", "created_utc": now(), "routes": routes,
        "existing_versions_must_be_reconciled_before_repair": True,
    })
    publish_page(output, root, records, suggestions, output / "review_ai_assisted")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    publish(args.output.resolve(), args.controls, args.preview)


if __name__ == "__main__":
    main()
