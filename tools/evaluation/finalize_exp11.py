#!/usr/bin/env python3
"""Materialize Exp11 final tables and figures from frozen, already-produced results.

This script performs no model loading, inference, training, or parameter selection.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EXP11 = ROOT / "results/final_test/exp11_retry1"
FINAL = ROOT / "results/final"
FIGURES = FINAL / "figures"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_bar(path: Path, labels, values, title, ylabel, color="#4472C4", ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}" if isinstance(value, float) else str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((EXP11 / "summary.json").read_text(encoding="utf-8"))
    overall = read_csv(EXP11 / "overall_metrics.csv")
    per_class = read_csv(EXP11 / "per_class_metrics.csv")
    fixed = read_csv(EXP11 / "fixed_threshold_metrics.csv")
    sizes = read_csv(EXP11 / "size_metrics.csv")
    errors = read_csv(EXP11 / "error_instances.csv")
    three_seed = read_csv(ROOT / "results/final_verify/exp10_three_seed_summary.csv")
    paired = read_csv(ROOT / "results/final_verify/exp10_paired_deltas.csv")
    roi = read_csv(ROOT / "results/project_review/roi_representation_results.csv")
    dataset = read_csv(ROOT / "results/project_review/dataset_summary_for_report.csv")

    methods = [
        ("Baseline", "FINAL_SELECTED_METHOD", "FINAL_TEST", "seed44 frozen checkpoint; only final TEST evaluation"),
        ("Low-confidence diagnostic", "POSITIVE_DIAGNOSTIC", "VAL_DIAGNOSTIC", "useful error analysis; not a final model"),
        ("One-class diagnostic", "NO_CLEAR_GAIN", "VAL_DIAGNOSTIC", "no clear reproducible gain"),
        ("Hard Mining", "NOT_CONFIRMED", "VAL_MULTI_SEED", "mean delta -0.005543; 1/3 positive seeds"),
        ("ROI ResNet18 CE", "COMPLETE_DIAGNOSTIC", "ROI_VAL", "classification diagnostic only"),
        ("ROI ResNet18 CE + SupCon", "POSITIVE_ROI_REPRESENTATION", "ROI_VAL", "representation result; not segmentation method"),
        ("Stage2", "NEGATIVE", "VAL", "negative downstream result"),
        ("Knowledge Distillation", "SKIPPED_BY_ENGINEERING_GATE", "NOT_EVALUATED", "not evaluated"),
        ("SimSiam", "INVALID_BY_BACKBONE_NO_UPDATE", "NOT_EVALUATED", "invalid; not evaluated"),
        ("Uniform continued training", "CONTROL_ONLY", "VAL_MULTI_SEED", "control only; not final candidate"),
        ("Resolution study", "NOT_FORMALLY_ANSWERED", "FUTURE_WORK_ONLY", "deferred by evidence"),
    ]
    method_rows = [dict(method=m, status=s, evidence_scope=e, final_role=n) for m, s, e, n in methods]
    write_csv(FINAL / "final_method_status_matrix.csv", method_rows[0].keys(), method_rows)
    write_csv(ROOT / "results/fast_repro/method_status_matrix.csv", method_rows[0].keys(), method_rows)
    write_csv(FINAL / "paper_method_status.csv", method_rows[0].keys(), method_rows)

    main_rows = []
    for row in overall:
        main_rows.append({
            "section": "FINAL_TEST", "result_scope": "ONE_FINAL_FROZEN_EVALUATION", "method": "Baseline seed44",
            "domain": row["domain"], "P": row["P"], "R": row["R"], "mAP50": row["mAP50"], "mAP50_95": row["mAP50_95"],
            "notes": "147 images; 285 instances; no post-test selection",
        })
    for row in three_seed:
        if row["model_type"] in {"baseline100", "uniform_control30", "hard_treatment30"}:
            main_rows.append({
                "section": "VAL_MULTI_SEED", "result_scope": "EXP10_UNIFIED_VAL", "method": f'{row["model_type"]} seed{row["seed"]}',
                "domain": "Mask", "P": row["Mask_P"], "R": row["Mask_R"], "mAP50": row["Mask_mAP50"], "mAP50_95": row["Mask_mAP50_95"],
                "notes": "VAL only; must not be compared as TEST",
            })
    main_rows.append({
        "section": "SYSTEM_DIAGNOSTIC", "result_scope": "FIXED_THRESHOLD_TEST", "method": "Baseline seed44 conf0.25",
        "domain": "Mask matched IoU0.50", "P": summary["fixed_threshold"]["Precision"], "R": summary["fixed_threshold"]["Recall"],
        "mAP50": "", "mAP50_95": "", "notes": f'TP={summary["fixed_threshold"]["TP"]}; FP={summary["fixed_threshold"]["FP"]}; FN={summary["fixed_threshold"]["FN"]}; F1={summary["fixed_threshold"]["F1"]}',
    })
    write_csv(FINAL / "final_main_results.csv", main_rows[0].keys(), main_rows)
    write_csv(FINAL / "paper_main_table.csv", main_rows[0].keys(), main_rows[:2])
    write_csv(FINAL / "paper_per_class_test.csv", per_class[0].keys(), per_class)
    write_csv(FINAL / "paper_three_seed_table.csv", three_seed[0].keys(), three_seed)
    write_csv(FINAL / "paper_hard_mining_ablation.csv", paired[0].keys(), paired)
    write_csv(FINAL / "final_roi_representation_results.csv", roi[0].keys(), roi)
    write_csv(FINAL / "paper_roi_supcon_table.csv", roi[0].keys(), roi)

    global_classes = [r for r in dataset if r["Category"] == "Class" and r["Scope"] == "Global"]
    global_sizes = [r for r in dataset if r["Category"] == "Size" and r["Scope"] == "Global"]
    save_bar(FIGURES / "dataset_class_distribution.png", [r["Class_or_bin"] for r in global_classes], [int(r["Value"]) for r in global_classes], "Dataset class distribution", "Instances", "#5B9BD5")
    save_bar(FIGURES / "dataset_size_distribution.png", [r["Class_or_bin"] for r in global_sizes], [int(r["Value"]) for r in global_sizes], "Dataset size distribution", "Instances", "#70AD47")
    save_bar(FIGURES / "baseline_test_size_recall.png", [r["size_bin"] for r in sizes], [float(r["Recall"]) for r in sizes], "Frozen Baseline TEST recall by size", "Recall", "#ED7D31", (0, 0.65))

    labels = [r["class_name"] for r in per_class]
    ap = [float(r["Mask_AP50_95"]) for r in per_class]
    recall = [float(r["Mask_R"]) for r in per_class]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar([v - .2 for v in x], ap, .4, label="Mask AP50-95", color="#4472C4")
    ax.bar([v + .2 for v in x], recall, .4, label="Mask Recall", color="#A5A5A5")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Frozen Baseline TEST per-class metrics"); ax.grid(axis="y", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "final_test_per_class_metrics.png", dpi=180); plt.close(fig)

    counts = Counter(r["category"] for r in errors)
    order = ["TP", "FP", "NO_RESPONSE", "LOCALIZATION_FAILURE", "POOR_MASK_BOUNDARY", "WRONG_CLASS"]
    save_bar(FIGURES / "final_test_error_taxonomy.png", order, [counts[k] for k in order], "Fixed-threshold TEST error taxonomy", "Count", "#FFC000")

    environment = json.loads((EXP11 / "environment.json").read_text(encoding="utf-8"))
    (EXP11 / "environment.txt").write_text("\n".join(f"{k}: {v}" for k, v in environment.items()) + "\n", encoding="utf-8")
    pointer = {
        "status": "PASS", "canonical_result_dir": "results/final_test/exp11_retry1",
        "initial_attempt": "INTERRUPTED_PREMETRIC_PRESERVED", "retry": "ONE_EXPLICITLY_AUTHORIZED_RETRY",
        "candidate_freeze_commit": "9991fcfcb9cf6c0ab8920ad7deadeed579ce5585",
        "retry_authorization_commit": "4fd982f49fc949fd99f4dd121b77791d3060c649",
        "checkpoint": summary["checkpoint"], "checkpoint_sha256": summary["checkpoint_sha256"],
        "seed": 44, "test_accessed": True, "model_selection_closed": True,
        "post_test_training": False, "post_test_selection": False,
        "Mask_mAP50_95": summary["Mask"]["mAP50_95"], "Box_mAP50_95": summary["Box"]["mAP50_95"],
    }
    dump_json(ROOT / "results/final_test/exp11_final_result.json", pointer)

    required = list(EXP11.rglob("*")) + list(FINAL.rglob("*"))
    required_files = [p for p in required if p.is_file()]
    parsed_csv = 0; parsed_json = 0; decoded_png = 0
    for path in required_files:
        assert path.stat().st_size > 0, path
        if path.suffix == ".csv": read_csv(path); parsed_csv += 1
        elif path.suffix == ".json": json.loads(path.read_text(encoding="utf-8")); parsed_json += 1
        elif path.suffix == ".png":
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", path; decoded_png += 1
    verification = {
        "status": "PASS", "canonical_result_dir": "results/final_test/exp11_retry1",
        "checked_nonempty_files": len(required_files), "parsed_csv": parsed_csv, "parsed_json": parsed_json,
        "decoded_final_png": decoded_png, "qualitative_image_count": len(list((EXP11 / "qualitative").glob("*.jpg"))),
        "test_accessed": True, "model_selection_closed": True, "post_test_training": False,
        "checkpoint_sha256": summary["checkpoint_sha256"],
    }
    dump_json(ROOT / "results/final_test/exp11_artifact_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
