"""Build the read-only Exp14 A9 candidate package for independent data review."""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import _csv_bytes
from tools.exp13_cloud.g3.verify_train_export import EXPECTED_MANIFEST_SHA256
from tools.exp14.final_dataset_identity import A8_ROOT, INPUTS, archive, load_reviews, verify_existing_matrix_support
from tools.exp14.final_dataset_selection import CANDIDATE, consolidate, exclusion_reasons
from tools.exp14.prepare_large_scale_repair import PRIVATE_ROOT, hash_file, load_json
from tools.exp14.run_large_scale_repair import now

EXPERIMENT = "EXP14_A9_FINAL_DATASET_CONSOLIDATION"
STATUS = "EXP14_A9_FINAL_DATASET_CONSOLIDATED_PENDING_CHATGPT_DATA_GATE"
PARTIAL = "EXP14_A9_FINAL_DATASET_PARTIAL_PENDING_CHATGPT_DATA_GATE"


def collect(root):
    sources, info, _, reviews, issues, scopes, prior, split = load_reviews(root)
    for item in reviews:
        item["base"] = str(item["base"])
    evidence = {"started_utc": now(), "sources": sources, "source_info": info, "reviews": reviews, "issues": issues,
                "split": dict(split), "prior_registry": prior, "a8_routes": scopes[A8_ROOT]["routes"]}
    g3.write_json_exclusive(root / "audits" / "verified_evidence.json", evidence)
    print("A9 verified", len(reviews), "review rows;", len(issues), "identity conflicts", flush=True)


def write_csv(path, rows):
    fields = list(rows[0])
    g3.require(all(set(row) == set(fields) for row in rows), f"CSV schema differs: {path.name}")
    serial = [{k: (json.dumps(v, ensure_ascii=False, separators=(",", ":")) if isinstance(v, (list, dict, bool))
                   else "" if v is None else str(v)) for k, v in row.items()} for row in rows]
    g3.write_exclusive(path, _csv_bytes(serial, fields))
    g3.require(g3.read_csv(path) == (fields, serial), f"CSV round-trip mismatch: {path.name}")


def verify_asset_index(result, views, sources, source_info):
    by_path = {}

    def visit(value):
        if isinstance(value, dict):
            if value.keys() >= {"path", "sha256"}:
                path, sha = value["path"], value["sha256"]
                g3.require(by_path.get(path, sha) == sha, "Candidate path has conflicting hashes")
                by_path[path] = sha
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for name in ("a_candidates", "b_candidates"):
        visit(result[name])
    for sid in source_info:
        source = sources[sid]
        by_path[Path(source["image"]).as_posix()] = source["image_sha256"]
        by_path[Path(source["annotation"]).as_posix()] = source["annotation_sha256"]
    for item in views:
        if "audit" not in item:
            continue
        # Include all indexed views so the private preview can be independently verified.
        visit(item["audit"]["assets"])
    for path, sha in by_path.items():
        hash_file(Path(path), sha)
    return by_path


def variant_rows(views, sources):
    rows = []
    for item in views:
        rec, human = item["record"], item["human"]
        audit = item.get("audit", {})
        data = {"source_id": rec["source_id"], "variant_id": rec["variant_id"], "version": rec["version"],
                "review_id": rec["review_id"], "review_input": item["input_name"],
                "generation_variant_id": rec.get("generation_variant_id", rec["variant_id"]),
                "alignment_parent_review_id": rec.get("alignment_parent_review_id", ""),
                "original_sha256": rec["source_sha256"], "annotation_sha256": rec["annotation_sha256"],
                "request_sha256": rec["request_sha256"], "mask_sha256": rec["mask_sha256"],
                "raw_sha256": rec["raw_sha256"], "aligned_sha256": rec["aligned_sha256"],
                "geometry_status": rec["geometry_status"], "matrix_output_to_source": audit.get("matrix"),
                "valid_coverage": rec["valid_coverage"], "valid_support": audit.get("support", {}),
                "human_decision": human["decision"], "human_reason": human["reason"], "human_notes": human["notes"],
                "human_polygon_decisions": item.get("explicit", {}), "resolved_polygon_decisions": item.get("resolved", {}),
                "whole_image_normal": human["whole_image_normal"], "b_region_review": human["b_region_review"],
                "final_version_choice": human.get("final_version_choice", ""), "ai_decision": item.get("ai", {}).get("ai_decision"),
                "aligned": audit.get("assets", {}).get("aligned.png"), "bounded": audit.get("assets", {}).get("bounded.png"),
                "raw": audit.get("assets", {}).get(rec["raw"]), "origin_paths": audit.get("origin_paths", {}),
                "identity_status": "VERIFIED" if audit and "resolved" in item else "QUARANTINED",
                "human_exclusion_reasons": exclusion_reasons(item, sources[rec["source_id"]])}
        rows.append(data)
    return rows


def dataset_summary(evidence, result, variants, assets):
    counts = result["counts"]
    by_version = {}
    for version in ("V1", "V2", "V3", "V4"):
        rows = [r for r in variants if r["version"] == version]
        by_version[version] = {
            "review_rows": len(rows), "unique_sources": len({r["source_id"] for r in rows}),
            "human_pass_sources": len({r["source_id"] for r in rows if r["human_decision"] == "PASS"}),
            "a_selected_proposals": result["a_coverage"]["versions"][version],
            "b_selected_polygons": result["b_coverage"]["versions"][version],
            "b_selected_sources": len({r["source_id"] for r in result["b_candidates"] if r["recommended"]["version"] == version}),
        }
    exclusions = defaultdict(set)
    for row in result["excluded"]:
        for reason in row["reasons"]:
            exclusions[reason].add(row["source_id"])
    for row in result["a_audit"]:
        for reason in row["reasons"]:
            exclusions[reason].add(row["source_id"])
    for row in result["b_audit"]:
        for candidate in row["semantic_candidates"]:
            for reason in candidate["reasons"]:
                exclusions[reason].add(row["source_id"])
    return {"experiment": EXPERIMENT, "status": PARTIAL if evidence["issues"] else STATUS,
        "started_utc": evidence["started_utc"],
        "input_review_sha256": INPUTS, "frozen_split_sha256": EXPECTED_MANIFEST_SHA256, "split_totals": evidence["split"],
        "sources": len(result["sources"]), "polygons": len(result["polygons"]), "variant_review_rows": len(variants),
        "distinct_generation_identities": len({r["generation_variant_id"] for r in variants}),
        "human_decisions": dict(Counter(r["human_decision"] for r in variants)),
        "human_confirmed_variant_rows": sum(r["human_decision"] != "UNREVIEWED" for r in variants),
        "unreviewed_variant_rows": sum(r["human_decision"] == "UNREVIEWED" for r in variants),
        "v4_human_decisions": dict(Counter(r["human_decision"] for r in variants if r["version"] == "V4")),
        "audit_verified_assets": len(assets), "identity_issue_count": len(evidence["issues"]),
        "a_original_class_coverage": result["a_coverage"]["classes"], "b_polygon_class_coverage": result["b_coverage"]["classes"],
        "candidate_distribution_by_actual_version": by_version, "counts": counts,
        "unique_source_counts_by_risk": {r: len(s) for r, s in exclusions.items()},
        "formal_tre": "NOT_ESTABLISHED", "formal_training_admissions": 0, "trusted_training_pairs": 0,
        "new_api_posts": 0, "new_alignments": 0, "new_training_runs": 0, "new_generated_images": 0}


def publish(root, evidence):
    views = evidence["reviews"]
    result = consolidate(evidence)
    variants = variant_rows(views, evidence["sources"])
    source_ids = {r["source_id"] for r in result["sources"]}
    polygon_ids = {r["polygon_id"] for r in result["polygons"]}
    g3.require(len(result["sources"]) == len(source_ids) == 523 and len(result["polygons"]) == len(polygon_ids) == 901 and
               len(variants) == len({r["review_id"] for r in variants}) == 956, "Candidate index count or identity collision")
    a = result["a_candidates"]
    b = result["b_candidates"]
    g3.require(len(a) == len({r["source_id"] for r in a}) and len(b) == len({(r["source_id"], r["polygon_id"]) for r in b}),
               "Candidate proposals duplicate a source or Polygon")
    assets = verify_asset_index(result, views, evidence["sources"], evidence["source_info"])
    summary = dataset_summary(evidence, result, variants, assets)
    registry = root / "registry"
    for name, rows in (("source_master_registry", result["sources"]), ("variant_master_registry", variants),
                       ("polygon_master_registry", result["polygons"]), ("a_candidate_manifest", a),
                       ("b_pair_candidate_manifest", b), ("final_version_selection_queue", result["queue"]),
                       ("unresolved_and_excluded", result["excluded"])):
        write_csv(registry / (name + ".csv"), rows)
    g3.write_json_exclusive(registry / "candidate_asset_sha256.json", {"status": CANDIDATE, "files": assets})
    g3.write_json_exclusive(registry / "final_candidate_summary.json", summary)
    audit = root / "audits"
    g3.write_json_exclusive(audit / "identity_audit.json", {"status": "PARTIAL" if evidence["issues"] else "PASS",
        "frozen_split_sha256": EXPECTED_MANIFEST_SHA256, "sources": len(source_ids), "polygons": len(polygon_ids),
        "review_identity_count": len(variants), "issues": evidence["issues"]})
    g3.write_json_exclusive(audit / "review_reconciliation.json", {"input_hashes": INPUTS,
        "review_rows": [{k: row[k] for k in ("source_id", "variant_id", "version", "review_id", "review_input", "human_decision",
            "human_reason", "human_notes", "whole_image_normal", "b_region_review", "human_polygon_decisions", "resolved_polygon_decisions",
            "final_version_choice")} for row in variants],
        "historical_context": "A8 registry independently reconciled; V4 human CSV supersedes V4 AI only as the human record"})
    g3.write_json_exclusive(audit / "label_integrity_audit.json", {"status": CANDIDATE, "records": result["a_audit"],
        "selection_rule": "All original GT targets PASS on the same Aligned review identity, with full Valid Mask and no source, visual or label conflict"})
    g3.write_json_exclusive(audit / "alignment_support_audit.json", {"formal_tre": "NOT_ESTABLISHED", "trusted_pair_pass": 0,
        "records": result["b_audit"],
        "selection_rule": "Human CONFIRMED on this version, PASS on this Polygon, audited mapping and full Polygon support; independent geometry Gate still pending"})
    g3.write_json_exclusive(audit / "quality_gate_precheck.json", {"identity": "PARTIAL" if evidence["issues"] else "PASS",
        "csv_roundtrip": "PASS", "file_hashes": "PASS", "a_label_audit": "COMPLETE",
        "b_support_audit": "COMPLETE", "formal_tre": "NOT_ESTABLISHED", "independent_data_gate": "PENDING",
        "training_approved": False, "split_selection": "TRAIN_ONLY", "new_api_posts": 0, "new_alignments": 0,
        "new_training_runs": 0, "bound_asset_requires_confirmation": summary["counts"]["a_pending_asset_choice_sources"]})
    preview(root, result, views)
    print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)


def preview(root, result, views):
    directory = root / "previews"
    source_lookup = {r["source_id"]: r for r in result["sources"]}
    rows = defaultdict(list)
    a = {r["source_id"]: r for r in result["a_candidates"]}
    b = defaultdict(list)
    for row in result["b_candidates"]:
        b[row["source_id"]].append(row)
    for item in views:
        rec = item["record"]
        human = item["human"]
        asset = item.get("audit", {}).get("assets", {})
        files = {label: {**asset[name], "url": Path(os.path.relpath(asset[name]["path"], directory)).as_posix()}
                 for label, name in {"Original + Polygon": "polygon_overlay.png", "Raw": rec["raw"], "Aligned": "aligned.png",
                                     "Bounded": "bounded.png", "Valid Mask": "valid.png"}.items() if name in asset}
        polygon_files = {
            pid: {label: {**p[key], "url": Path(os.path.relpath(p[key]["path"], directory)).as_posix()}
                  for label, key in (("原图局部", "before"), ("Aligned 局部", "after")) if p.get(key)}
            for pid, p in item.get("audit", {}).get("support", {}).items()
        }
        rows[rec["source_id"]].append({"variant": rec["variant_id"], "version": rec["version"], "human": human["decision"],
            "notes": human["notes"], "files": files, "polygons": item.get("resolved", {}),
            "polygon_files": polygon_files, "review_id": rec["review_id"], "geometry": rec["geometry_status"]})
    records = []
    for sid, source in source_lookup.items():
        options = b[sid]
        b_targets = [{"id": r["polygon_id"], "class": r["class"], "variant": r["recommended"]["variant_id"],
                      "sha256": r["recommended"]["aligned"]["sha256"], "risks": r["recommended"]["risk_evidence"]}
                     for r in options]
        original_url = Path(os.path.relpath(source["original_path"], directory)).as_posix() if any(
            v["files"] for v in rows[sid]) else ""
        records.append({"id": sid, "source": source, "variants": rows[sid], "original_url": original_url,
                        "b": b_targets, "a": a[sid]["recommended"] if sid in a else None})
    template = Path(__file__).with_name("final_dataset_preview.html").read_text(encoding="utf-8")
    data = json.dumps({"records": records, "counts": result["counts"]}, ensure_ascii=False).replace("<", "\\u003c")
    g3.write_exclusive(directory / "index.html", template.replace("__DATA__", data).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "publish", "matrix-check", "preview", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    g3.require(root.parent == PRIVATE_ROOT.resolve() and root.name.startswith("exp14_a9_final_dataset_"), "Expected independent A9 root")
    if args.action == "run":
        archive(root, args.csv, args.audit)
    if args.action in {"collect", "run"}:
        collect(root)
    if args.action in {"publish", "matrix-check", "preview", "run"}:
        evidence = load_json(args.evidence or root / "audits" / "verified_evidence.json")
        if args.action == "matrix-check":
            result = verify_existing_matrix_support(evidence)
            g3.write_json_exclusive(root / "audits" / "stored_matrix_support_crosscheck.json", result)
        elif args.action == "preview":
            preview(root, consolidate(evidence), evidence["reviews"])
        else:
            publish(root, evidence)


if __name__ == "__main__":
    main()
