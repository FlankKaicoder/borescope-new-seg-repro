#!/usr/bin/env python3
"""Audit annotation consistency inside perceptual near-duplicate groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    labels: dict[str, Counter[str] | None] = {}
    with args.groups_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_path = Path(row["image_path"])
        json_path = args.data_root / image_path.with_suffix(".json")
        if not json_path.exists():
            labels[row["image_path"]] = None
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        labels[row["image_path"]] = Counter(str(shape.get("label", "")) for shape in payload.get("shapes", []) if isinstance(shape, dict))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    output_rows = []
    for group_id, members in sorted(groups.items()):
        signatures = set()
        label_sets = set()
        missing = []
        for member in members:
            value = labels[member["image_path"]]
            if value is None:
                missing.append(member["image_path"])
                signature = "<NO_JSON>"
            else:
                signature = "|".join(f"{key}:{value[key]}" for key in sorted(value))
                label_sets.add("|".join(sorted(value)))
            signatures.add(signature)
            output_rows.append({
                "group_id": group_id,
                "group_size": len(members),
                "image_path": member["image_path"],
                "annotation_signature": signature,
                "group_signature_count": len(signatures),
            })
        final_count = len(signatures)
        for row in output_rows[-len(members):]:
            row["group_signature_count"] = final_count
            row["group_has_annotation_mismatch"] = final_count > 1
            row["group_has_label_set_mismatch"] = len(label_sets) > 1
            row["group_has_instance_count_only_mismatch"] = final_count > 1 and len(label_sets) == 1
            row["group_has_missing_json"] = bool(missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group_id", "group_size", "image_path", "annotation_signature", "group_signature_count", "group_has_annotation_mismatch", "group_has_label_set_mismatch", "group_has_instance_count_only_mismatch", "group_has_missing_json"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    mismatch_groups = {row["group_id"] for row in output_rows if row["group_has_annotation_mismatch"]}
    label_set_mismatch_groups = {row["group_id"] for row in output_rows if row["group_has_label_set_mismatch"]}
    instance_count_only_groups = {row["group_id"] for row in output_rows if row["group_has_instance_count_only_mismatch"]}
    missing_groups = {row["group_id"] for row in output_rows if row["group_has_missing_json"]}
    summary = {"group_count": len(groups), "annotation_mismatch_group_count": len(mismatch_groups), "label_set_mismatch_group_count": len(label_set_mismatch_groups), "instance_count_only_mismatch_group_count": len(instance_count_only_groups), "groups_with_missing_json_count": len(missing_groups), "mismatch_groups": sorted(mismatch_groups), "label_set_mismatch_groups": sorted(label_set_mismatch_groups), "instance_count_only_mismatch_groups": sorted(instance_count_only_groups), "groups_with_missing_json": sorted(missing_groups)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
