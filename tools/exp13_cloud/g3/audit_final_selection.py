"""Audit a restored G3-B1 source selection against its backup and TRAIN export."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.exp13_cloud.g3.build_candidate_review import HISTORY
from tools.exp13_cloud.g3.build_source_review import CSV_FIELDS
from tools.exp13_cloud.g3.review_decisions import read_csv, restore_backup_rows
from tools.exp13_cloud.g3.verify_train_export import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_TRAIN_INSTANCES,
    read_rows,
    sha256_file,
    validate_export_data,
)

USER_FIELDS = {"selected_polygon_ids", "repair_priority", "user_decision", "user_notes"}


def audit(backup_path: Path, baseline_path: Path, final_csv: Path, train_root: Path, report_path: Path) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {report_path}")
    frozen = train_root / "frozen_split_manifest.csv"
    export = train_root / "export_manifest.csv"
    mapping = train_root / "class_mapping.json"
    gate = validate_export_data(frozen, export, train_root, mapping)
    if gate["status"] != "PASS" or gate["issues"]:
        raise ValueError("G3-A TRAIN export did not pass current integrity revalidation")
    if gate["frozen_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("frozen TRAIN Manifest identity mismatch")

    backup = json.loads(backup_path.read_text(encoding="utf-8-sig"))
    baseline = read_csv(baseline_path)
    expected_rows = restore_backup_rows(backup, baseline)
    final_rows = read_csv(final_csv)
    if not final_rows or list(final_rows[0]) != CSV_FIELDS:
        raise ValueError("final selection CSV does not have the exact 14-field schema")
    if len(final_rows) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(f"expected {EXPECTED_TRAIN_IMAGES} final source rows, got {len(final_rows)}")
    if len({row["source_id"] for row in final_rows}) != len(final_rows):
        raise ValueError("final selection contains duplicate source IDs")

    expected_by_id = {row["source_id"]: row for row in expected_rows}
    final_by_id = {row["source_id"]: row for row in final_rows}
    if set(final_by_id) != set(expected_by_id):
        raise ValueError("final selection source_id set differs from the strict backup restore")
    export_rows = read_rows(export)
    export_by_id = {row["source_id"]: row for row in export_rows}
    if set(export_by_id) != set(final_by_id):
        raise ValueError("final selection source_id set differs from frozen TRAIN export")

    decisions: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    history: Counter[str] = Counter()
    selected_polygons = 0
    note_count = 0
    for source_id, row in final_by_id.items():
        expected = expected_by_id[source_id]
        source = export_by_id[source_id]
        for field in CSV_FIELDS:
            if field in USER_FIELDS:
                if row[field] != expected[field]:
                    raise ValueError(f"{source_id}: restored review field {field} differs from latest backup")
            elif row[field] != expected[field]:
                raise ValueError(f"{source_id}: frozen review metadata field {field} changed")
        if row["source_image_sha256"] != source["source_image_sha256"]:
            raise ValueError(f"{source_id}: image hash does not match G3-A export")
        if row["group_id"] != source["group_id"]:
            raise ValueError(f"{source_id}: group_id does not match G3-A export")
        if row["image_width"] != source["image_width"] or row["image_height"] != source["image_height"]:
            raise ValueError(f"{source_id}: image dimensions do not match G3-A export")

        annotation = json.loads((train_root / source["export_json_path"]).read_text(encoding="utf-8-sig"))
        polygon_ids = [f"{source_id}_{index:03d}" for index, _ in enumerate(annotation.get("shapes", []))]
        listed_ids = [value for value in row["polygon_ids"].split("|") if value]
        if listed_ids != polygon_ids or len(listed_ids) != len(set(listed_ids)):
            raise ValueError(f"{source_id}: Polygon exact-set/order differs from the G3-A annotation")
        selected = [value for value in row["selected_polygon_ids"].split("|") if value]
        if any(value not in set(polygon_ids) for value in selected):
            raise ValueError(f"{source_id}: selected Polygon is not owned by this source")
        if row["user_decision"] == "SELECT" and not selected:
            raise ValueError(f"{source_id}: SELECT has no target Polygon")
        if row["user_decision"] == "REJECT" and selected:
            raise ValueError(f"{source_id}: REJECT retains target Polygons")
        if row["user_decision"] == "UNREVIEWED":
            raise ValueError(f"{source_id}: final selection contains an unreviewed source")

        decisions[row["user_decision"]] += 1
        selected_polygons += len(selected)
        note_count += bool(row["user_notes"])
        for label in row["classes_present"].split("|"):
            if label:
                classes[label] += 1
        exposure = row["historical_cloud_exposure"]
        expected_exposure = HISTORY.get(source_id, "NONE_RECORDED_IN_G1_G2_AUDIT_MAP")
        if exposure != expected_exposure:
            raise ValueError(f"{source_id}: historical G1/G2 exposure marker mismatch")
        if exposure != "NONE_RECORDED_IN_G1_G2_AUDIT_MAP":
            history[exposure] += 1

    expected_decisions = {"SELECT": 523, "MAYBE": 23, "REJECT": 122}
    if dict(decisions) != expected_decisions:
        raise ValueError(f"decision totals differ from reported completed review: {dict(decisions)}")
    report = {
        "status": "G3B1_FINAL_SELECTION_AUDIT_PASS",
        "source_rows": len(final_rows),
        "schema_fields": len(CSV_FIELDS),
        "decision_counts": dict(sorted(decisions.items())),
        "selected_polygon_count": selected_polygons,
        "nonempty_user_notes": note_count,
        "class_source_counts": dict(sorted(classes.items())),
        "historical_cloud_exposure_sources": dict(sorted(history.items())),
        "source_exact_set_matches_latest_backup": True,
        "polygon_exact_set_matches_train_annotations": True,
        "review_fields_match_latest_backup": True,
        "frozen_manifest_sha256": gate["frozen_manifest_sha256"],
        "export_manifest_sha256": gate["export_manifest_sha256"],
        "train_images": gate["checked_train_images"],
        "train_instances": gate["checked_train_instances"],
        "train_integrity_issues": len(gate["issues"]),
        "backup_sha256": sha256_file(backup_path),
        "baseline_csv_sha256": sha256_file(baseline_path),
        "final_csv_sha256": sha256_file(final_csv),
        "api_posts": 0,
        "gpu_training": 0,
        "test_image_access": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-json", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.backup_json, args.baseline_csv, args.final_csv, args.train_root, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
