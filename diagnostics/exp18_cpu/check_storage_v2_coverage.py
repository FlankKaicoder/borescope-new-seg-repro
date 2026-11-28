"""Cross-check V2 tables against previous independently accepted coverage margins."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from tools.exp18.common import require, sha256, write_json_exclusive


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate(output, old_cross):
    receipt = json.loads((output / "coverage_receipt.json").read_text(encoding="utf-8"))
    for file, digest in receipt["csv_sha256"].items():
        require(Path(file).name == file and sha256(output / file) == digest, "Coverage member SHA changed")
    cross = rows(output / "coverage_class_version_p3_cross.csv")
    fine = [r for r in cross if r["granularity"] == "class_version_p3" and int(r["K_exposures"])]
    previous = rows(old_cross)
    keys = ("class_id", "repair_family", "P3_bucket")
    counts = ("K_exposures", "S_exposures", "K_unique_polygons", "S_unique_polygons",
              "K_unique_groups", "S_unique_groups")
    def project(values):
        return {tuple(row[k] for k in keys): tuple(int(row[k]) for k in counts) for row in values}
    require(len(fine) == len(previous) == 52 and project(fine) == project(previous),
            "Cross margins differ from impartial CPU coverage")
    anchors = rows(output / "partner_diversity_by_anchor.csv")
    edges = rows(output / "partner_edges.csv")
    sums = Counter()
    for edge in edges:
        sums[edge["A_polygon_id"]] += int(edge["exposures"])
    require(all(int(row["aux_exposures"]) == sums[row["polygon_id"]] for row in anchors), "Edge/anchor exposures differ")
    require(len({row["group_id"] for row in anchors}) == 327
            and len({row["group_id"] for row in anchors if int(row["aux_exposures"])}) == 316, "Group population differs")
    return {"status": "PASS", "positive_cross_rows_equal_previous": len(fine), "K_unique_groups": 327,
            "S_unique_groups": 316, "edge_rows_verified": len(edges),
            "old_cross_sha256": sha256(old_cross), "verified_coverage_receipt_sha256": sha256(output / "coverage_receipt.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-cross", type=Path, required=True)
    args = parser.parse_args()
    write_json_exclusive(args.output / "reference_reconciliation.json", validate(args.output, args.old_cross))


if __name__ == "__main__":
    main()
