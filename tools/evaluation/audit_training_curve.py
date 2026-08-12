#!/usr/bin/env python3
"""Audit an Ultralytics segmentation results.csv without changing the run."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results_csv", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
with a.results_csv.open(encoding="utf-8", newline="") as handle:
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
numeric = [{k: float(v) for k, v in row.items()} for row in rows]
nan_inf = []
for row in numeric:
    bad = [k for k, v in row.items() if not math.isfinite(v)]
    if bad:
        nan_inf.append({"epoch": int(row["epoch"]), "fields": bad})
for row in numeric:
    row["ultralytics_segment_fitness"] = row["metrics/mAP50-95(B)"] + row["metrics/mAP50-95(M)"]
best = max(numeric, key=lambda x: x["ultralytics_segment_fitness"])
best_mask = max(numeric, key=lambda x: x["metrics/mAP50-95(M)"])
payload = {
    "rows": len(rows), "epochs_complete": [int(x["epoch"]) for x in numeric] == list(range(1, 101)),
    "best_epoch_by_ultralytics_segment_fitness": int(best["epoch"]),
    "best_epoch_fitness": best["ultralytics_segment_fitness"],
    "best_epoch_box_map50_95": best["metrics/mAP50-95(B)"],
    "best_epoch_mask_map50_95": best["metrics/mAP50-95(M)"],
    "best_mask_map50_95_epoch": int(best_mask["epoch"]),
    "best_mask_map50_95": best_mask["metrics/mAP50-95(M)"],
    "non_finite_rows": nan_inf,
    "training_metric_non_finite_count": sum(k.startswith("train/") for row in nan_inf for k in row["fields"]),
    "validation_metric_non_finite_count": sum(k.startswith("val/") for row in nan_inf for k in row["fields"]),
    "final_epoch_all_finite": all(math.isfinite(v) for v in numeric[-1].values()),
}
a.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
