#!/usr/bin/env python3
"""Summarize Phase C1 manual-review candidates without changing review labels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CANDIDATE_TYPES = ["WEAK_CHANGE_CANDIDATE", "NOISE_PATTERN_CANDIDATE"]
FAMILIES = [
    "A_small_low_contrast",
    "B_diffuse_texture",
    "C_elongated_ribbon",
    "D_irregular_boundary",
]
CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-review-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manual_review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))

    required = {
        "edit_id", "source_id", "family", "class", "candidate_type", "review_label"
    }
    if not records or set(records[0]) != required | {
        "changed_ratio", "component_count"
    }:
        raise RuntimeError("UNEXPECTED_MANUAL_REVIEW_SCHEMA")

    output_rows = []
    for candidate_type in CANDIDATE_TYPES:
        for family in [*FAMILIES, "ALL"]:
            for defect_class in [*CLASSES, "ALL"]:
                selected = [
                    row for row in records
                    if row["candidate_type"] == candidate_type
                    and (family == "ALL" or row["family"] == family)
                    and (
                        defect_class == "ALL"
                        or defect_class in {
                            item.strip() for item in row["class"].split("|") if item.strip()
                        }
                    )
                ]
                output_rows.append({
                    "candidate_type": candidate_type,
                    "morphology_family": family,
                    "defect_class": defect_class,
                    "outlier_record_count": len(selected),
                    "unique_edit_count": len({(row["edit_id"], row["source_id"]) for row in selected}),
                    "review_label_empty_count": sum(not row["review_label"].strip() for row in selected),
                    "review_label_nonempty_count": sum(bool(row["review_label"].strip()) for row in selected),
                })

    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
