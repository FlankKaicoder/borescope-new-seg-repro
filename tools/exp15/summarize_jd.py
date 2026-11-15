"""Reconcile exactly four JD native receipts and write publication-safe paired summaries."""

import argparse
import csv
from pathlib import Path

from tools.exp14 import a11_full_eval as native
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.label_plan import CLASSES
from tools.exp15 import jd
from tools.exp15.jd_diagnostics import summarize_trace


def table(path, rows):
    with path.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def delta(left, right):
    return right - left if left is not None and right is not None else None


def side(key, left, right):
    return {f"{key}_j0": left, f"{key}_j1": right, f"{key}_delta": delta(left, right)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    root = a.evaluation
    receipt_path = root / "paired_receipt.json"
    r = jd.json_load(receipt_path)
    require(r["status"] == jd.COMPLETE and r["completed_passes"] == 4
            and r["checkpoint_bytes_rechecked"] and r["val_bytes_rechecked"]
            and r["training_updates"] == r["test_image_reads"] == r["test_label_reads"] == r["test_inference"] == 0,
            "Passed paired evaluation or scope metadata missing")
    require(set(p.stem for p in (root / "claims").glob("*.json")) == set(jd.PASSES)
            and not (root / "run_hard_gate.json").exists(), "Run pass attempt ledger differs")
    for name in jd.PASSES:
        arm, mode = name.split("_")
        p = root / name
        t = r["passes"][name]
        claim = jd.json_load(root / "claims" / f"{name}.json")
        require(claim["file_sha256"] == jd.CHECKPOINTS[arm][0]
                and claim["native_master_sha256"] == jd.CHECKPOINTS[arm][1]
                and claim["pass"] == name and claim["epoch"] == 100
                and claim["confidence"] == (native.STANDARD_CONF if mode == "standard" else native.FIXED_CONF),
                "Unbound evaluation result")
        require(t == jd.json_load(p / "completion.json") and t["metrics"] == jd.json_load(p / "evaluation.json")
                and t["metrics_sha256"] == sha256_file(p / "evaluation.json")
                and t["diagnostic_trace_sha256"] == sha256_file(p / "diagnostic_trace.jsonl"),
                "Evaluation readback differs")
        require(summarize_trace(p / "diagnostic_trace.jsonl", t["metrics"], claim["confidence"]) == t["diagnostics"],
                "Independent diagnostic aggregation differs")
    require(all(x["pass"] for x in r["j0_vs_historical_e0"].values())
            and set(r["j0_vs_historical_e0"]) == {"standard", "fixed"}, "Historical consistency Gate missing")

    names = ("j0", "j1")
    standard = [r["passes"][f"{k}_standard"]["metrics"] for k in names]
    fixed = [r["passes"][f"{k}_fixed"]["metrics"] for k in names]
    overall = [{"metric": key, "j0": standard[0]["overall"][key], "j1": standard[1]["overall"][key],
                "j1_minus_j0": standard[1]["overall"][key] - standard[0]["overall"][key]}
               for typ in ("M", "B") for metric in ("mAP50-95", "mAP50", "precision", "recall")
               for key in (f"metrics/{metric}({typ})",)]
    classes, counts, sizes, diag = [], [], [], []
    for n, label in enumerate(CLASSES):
        left, right = (t["per_class"][n] for t in standard)
        require(left["class"] == right["class"] == label and left["gt_instances"] == right["gt_instances"],
                "Paired class identity differs")
        row = {"class_id": n, "class": label, "gt_instances": left["gt_instances"]}
        for key in left:
            if key.startswith(("mask_", "box_")):
                row.update(side(key, left[key], right[key]))
        classes.append(row)
    for label, n in [("overall", None), *[(label, i) for i, label in enumerate(CLASSES)]]:
        left, right = (t["fixed_counts"]["overall"] if n is None else t["fixed_counts"]["per_class"][n] for t in fixed)
        require(left["gt"] == right["gt"], "Paired fixed GT differs")
        row = {"class": label, "gt": left["gt"]}
        for key in ("detections", "mask_tp", "mask_fp", "mask_fn", "box_tp", "box_fp", "box_fn"):
            row.update(side(key, left[key], right[key]))
        counts.append(row)
        for size in native.SIZE_NAMES:
            x, y = (t["size_recall"]["overall"][size] if n is None else t["size_recall"]["per_class"][label][size] for t in fixed)
            require(x["gt"] == y["gt"], "Paired size GT differs")
            row = {"class": label, "size": size, "gt": x["gt"]}
            for key in ("mask_tp", "mask_fn", "mask_recall"):
                row.update(side(key, x[key], y[key]))
            sizes.append(row)
    for mode in ("standard", "fixed"):
        x, y = (r["passes"][f"{k}_{mode}"]["diagnostics"] for k in names)
        require(x.keys() == y.keys(), "Diagnostics returned different definitions")
        diag += [{"mode": mode, "diagnostic": label, "j0": x[label], "j1": y[label],
                  "j1_minus_j0": delta(x[label], y[label])} for label in sorted(x)]
    a.output.mkdir()
    for file, rows in (("overall_metrics.csv", overall), ("class_metrics.csv", classes), ("fixed_counts.csv", counts),
                       ("size_recall.csv", sizes), ("diagnostics.csv", diag)):
        table(a.output / file, rows)
    jd.write_json_exclusive(a.output / "summary.json", {
        "status": jd.COMPLETE, "seed": 42, "primary_metric": overall[0], "overall_metrics": overall,
        "val_images": native.VAL_IMAGES, "val_instances": native.VAL_INSTANCES,
        "j0_vs_historical_e0": r["j0_vs_historical_e0"], "preflight": r["preflight"],
        "evaluation_pass_receipts": {name: {k: t[k] for k in ("metrics_sha256", "diagnostic_trace_sha256", "completed_at")}
                                    for name, t in r["passes"].items()},
        "fixed_counts": counts, "size_recall_overall": sizes[:4], "diagnostics": diag,
        "validation_passes": 4, "test_inference": 0, "training_updates": 0,
        "authorization_sha256": r["authorization_sha256"], "paired_receipt_sha256": sha256_file(receipt_path),
    })


if __name__ == "__main__":
    main()
