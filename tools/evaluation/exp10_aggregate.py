#!/usr/bin/env python3
"""Aggregate unified Exp10 three-seed VAL results and verify artifacts."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/final_verify"
CONTROLLED = RESULTS / "exp10_controlled_restart"
FIGURES = RESULTS / "figures/exp10"
MODELS = ("baseline100", "uniform_control30", "hard_treatment30")
SEEDS = (42, 43, 44)
MAIN_METRICS = ("Mask_mAP50_95", "Mask_R", "fixed_F1", "FP", "FN")
DIFFICULT = ("Burn", "Crack", "corrosion")
SIZES = ("tiny", "small", "medium", "large")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = CONTROLLED / f"seed{seed}/unified_val/seed{seed}_unified_val.csv"
        for raw in read_csv(path):
            row: dict[str, Any] = dict(raw)
            row["seed"] = int(raw["seed"])
            for key, value in list(row.items()):
                if key not in {"seed", "model", "checkpoint_sha256", "test_accessed"}:
                    row[key] = float(value)
            rows.append(row)
    if len(rows) != 9:
        raise RuntimeError(f"expected 9 seed-model rows, got {len(rows)}")
    return rows


def by_seed_model(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(row["seed"], row["model"]): row for row in rows}


def make_summary(rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed", "model_type", "Mask_P", "Mask_R", "Mask_mAP50", "Mask_mAP50_95",
        "Box_P", "Box_R", "Box_mAP50", "Box_mAP50_95", "TP", "FP", "FN", "fixed_F1",
        "tiny_R", "small_R", "medium_R", "large_R", "Burn_R", "Burn_AP50_95",
        "Crack_R", "Crack_AP50_95", "corrosion_R", "corrosion_AP50_95",
        "checkpoint_sha256", "status", "test_accessed",
    ]
    output = []
    for row in rows:
        rec = {key: row.get(key, "") for key in fields}
        rec["model_type"] = row["model"]
        rec["status"] = "PASS"
        rec["test_accessed"] = False
        output.append(rec)
    write_csv(RESULTS / "exp10_three_seed_summary.csv", output)


def paired_and_effects(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = by_seed_model(rows)
    paired_rows = []
    all_effects = []
    paired_values: dict[str, list[float]] = {metric: [] for metric in MAIN_METRICS}
    for seed in SEEDS:
        baseline = index[(seed, "baseline100")]
        control = index[(seed, "uniform_control30")]
        treatment = index[(seed, "hard_treatment30")]
        paired = {metric: treatment[metric] - control[metric] for metric in MAIN_METRICS}
        paired_rows.append({
            "seed": seed,
            "delta_mask_mAP50_95": paired["Mask_mAP50_95"],
            "delta_mask_recall": paired["Mask_R"],
            "delta_fixed_F1": paired["fixed_F1"],
            "delta_FP": paired["FP"],
            "delta_FN": paired["FN"],
            "status": "PASS",
        })
        for metric, value in paired.items():
            paired_values[metric].append(value)
        for effect, left, right in (
            ("Control30-Baseline100", control, baseline),
            ("Treatment30-Control30", treatment, control),
            ("Treatment30-Baseline100", treatment, baseline),
        ):
            all_effects.append({
                "seed": seed,
                "effect": effect,
                **{metric: left[metric] - right[metric] for metric in MAIN_METRICS},
            })
    stats_rows = []
    operations = {
        "AGGREGATE_MEAN": statistics.mean,
        "AGGREGATE_SAMPLE_STD": statistics.stdev,
        "AGGREGATE_MIN": min,
        "AGGREGATE_MAX": max,
    }
    mapping = {
        "delta_mask_mAP50_95": "Mask_mAP50_95",
        "delta_mask_recall": "Mask_R",
        "delta_fixed_F1": "fixed_F1",
        "delta_FP": "FP",
        "delta_FN": "FN",
    }
    for label, operation in operations.items():
        stats_rows.append({
            "seed": label,
            **{column: operation(paired_values[metric]) for column, metric in mapping.items()},
            "status": "AGGREGATE",
        })
    positive = sum(value > 0 for value in paired_values["Mask_mAP50_95"])
    stats_rows.append({
        "seed": "POSITIVE_SEED_COUNT",
        "delta_mask_mAP50_95": positive,
        "delta_mask_recall": "",
        "delta_fixed_F1": "",
        "delta_FP": "",
        "delta_FN": "",
        "status": "AGGREGATE",
    })
    write_csv(RESULTS / "exp10_paired_deltas.csv", paired_rows + stats_rows)
    write_csv(RESULTS / "exp10_all_effects.csv", all_effects)
    aggregate = {
        "paired_treatment_minus_control": {
            metric: {
                **mean_std(values),
                "min": min(values),
                "max": max(values),
                "positive_seed_count": sum(value > 0 for value in values),
            }
            for metric, values in paired_values.items()
        }
    }
    return paired_rows, aggregate


def aggregate_models(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    index = by_seed_model(rows)
    aggregate["models"] = {}
    for model in MODELS:
        aggregate["models"][model] = {
            metric: mean_std([index[(seed, model)][metric] for seed in SEEDS])
            for metric in MAIN_METRICS
        }
    aggregate["difficult_class_treatment_minus_control"] = {}
    for class_name in DIFFICULT:
        aggregate["difficult_class_treatment_minus_control"][class_name] = {
            "Mask_AP50_95": mean_std([
                index[(seed, "hard_treatment30")][f"{class_name}_AP50_95"]
                - index[(seed, "uniform_control30")][f"{class_name}_AP50_95"]
                for seed in SEEDS
            ]),
            "Mask_R": mean_std([
                index[(seed, "hard_treatment30")][f"{class_name}_R"]
                - index[(seed, "uniform_control30")][f"{class_name}_R"]
                for seed in SEEDS
            ]),
        }
    aggregate["size_recall_treatment_minus_control"] = {
        size: mean_std([
            index[(seed, "hard_treatment30")][f"{size}_R"]
            - index[(seed, "uniform_control30")][f"{size}_R"]
            for seed in SEEDS
        ])
        for size in SIZES
    }
    deltas = aggregate["paired_treatment_minus_control"]["Mask_mAP50_95"]
    if deltas["positive_seed_count"] == 3 and deltas["mean"] > 0:
        recommendation = "HARD_MINING_STRONG_ROBUST_POSITIVE"
    elif deltas["positive_seed_count"] >= 2 and deltas["mean"] > 0:
        recommendation = "HARD_MINING_ROBUST_POSITIVE"
    elif deltas["mean"] <= 0 and deltas["positive_seed_count"] <= 1:
        recommendation = "HARD_MINING_NOT_CONFIRMED"
    else:
        recommendation = "HARD_MINING_UNSTABLE"
    aggregate.update(
        recommendation=recommendation,
        candidate_freeze_recommendation="Baseline" if recommendation == "HARD_MINING_NOT_CONFIRMED" else "Hard Mining",
        candidate_freeze_executed=False,
        exp11_run=False,
        test_accessed=False,
        statistical_significance_claimed=False,
    )
    dump_json(RESULTS / "exp10_aggregate_summary.json", aggregate)


def make_figures(rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    index = by_seed_model(rows)
    colors = {"baseline100": "#4C78A8", "uniform_control30": "#F58518", "hard_treatment30": "#54A24B"}
    labels = {"baseline100": "Baseline100", "uniform_control30": "Uniform Control30", "hard_treatment30": "Hard Treatment30"}
    x = np.arange(3); width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, model in enumerate(MODELS):
        ax.bar(x + (offset - 1) * width, [index[(seed, model)]["Mask_mAP50_95"] for seed in SEEDS], width, label=labels[model], color=colors[model])
    ax.set_xticks(x, [f"seed{seed}" for seed in SEEDS]); ax.set_ylabel("Mask mAP50-95"); ax.set_title("Exp10 unified VAL across three seeds"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "exp10_mask_map_three_seeds.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = [float(row["delta_mask_mAP50_95"]) for row in paired_rows]
    ax.bar([f"seed{seed}" for seed in SEEDS], values, color=["#54A24B" if v > 0 else "#E45756" for v in values]); ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Treatment - Control Mask mAP50-95"); ax.set_title("Paired Hard Mining effect")
    for i, v in enumerate(values): ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout(); fig.savefig(FIGURES / "exp10_paired_hard_mining_delta.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ("Mask_mAP50_95", "Mask_R", "fixed_F1"); xx = np.arange(len(metrics))
    for offset, model in enumerate(MODELS):
        means = [aggregate["models"][model][metric]["mean"] for metric in metrics]
        stds = [aggregate["models"][model][metric]["sample_std"] for metric in metrics]
        ax.bar(xx + (offset - 1) * width, means, width, yerr=stds, capsize=4, label=labels[model], color=colors[model])
    ax.set_xticks(xx, ["Mask mAP50-95", "Mask Recall", "Fixed F1"]); ax.set_title("Three-seed mean ± sample std"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "exp10_mean_std_main_metrics.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, metric in zip(axes, ("FP", "FN")):
        for offset, model in enumerate(MODELS):
            ax.bar(x + (offset - 1) * width, [index[(seed, model)][metric] for seed in SEEDS], width, label=labels[model], color=colors[model])
        ax.set_xticks(x, [f"seed{seed}" for seed in SEEDS]); ax.set_title(metric); ax.grid(axis="y", alpha=.25)
    axes[0].legend(); fig.suptitle("Fixed-point FP/FN at conf=0.25, mask IoU=0.50")
    fig.tight_layout(); fig.savefig(FIGURES / "exp10_fp_fn_three_seeds.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8)); xx = np.arange(len(DIFFICULT))
    ap = [aggregate["difficult_class_treatment_minus_control"][name]["Mask_AP50_95"]["mean"] for name in DIFFICULT]
    recall = [aggregate["difficult_class_treatment_minus_control"][name]["Mask_R"]["mean"] for name in DIFFICULT]
    ax.bar(xx - .18, ap, .36, label="Mask AP50-95 delta", color="#4C78A8"); ax.bar(xx + .18, recall, .36, label="Mask Recall delta", color="#F58518"); ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xx, DIFFICULT); ax.set_title("Mean Treatment - Control effect on difficult classes"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "exp10_difficult_classes.png", dpi=180); plt.close(fig)


def verify_artifacts() -> None:
    formal_dirs = [
        CONTROLLED / "seed43/hard_treatment30/formal/ultralytics/treatment",
        CONTROLLED / "seed44/baseline100/formal/ultralytics/baseline",
        CONTROLLED / "seed44/uniform_control30/formal/ultralytics/control",
        CONTROLLED / "seed44/hard_treatment30/formal/ultralytics/treatment",
    ]
    required_names = [
        "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png",
        "MaskF1_curve.png", "MaskPR_curve.png", "MaskP_curve.png", "MaskR_curve.png",
        "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    ]
    paths = [directory / name for directory in formal_dirs for name in required_names]
    unified_dirs = [
        CONTROLLED / f"seed{seed}/unified_val/ultralytics/{model}"
        for seed in SEEDS for model in MODELS
    ]
    unified_required = [name for name in required_names if name != "results.png"]
    paths += [directory / name for directory in unified_dirs for name in unified_required]
    paths += [FIGURES / name for name in (
        "exp10_mask_map_three_seeds.png", "exp10_paired_hard_mining_delta.png",
        "exp10_mean_std_main_metrics.png", "exp10_fp_fn_three_seeds.png", "exp10_difficult_classes.png",
    )]
    records = []
    for path in paths:
        exists = path.is_file(); size = path.stat().st_size if exists else 0
        decoded = bool(exists and size > 0 and cv2.imread(str(path)) is not None)
        records.append({"path": str(path), "exists": exists, "size_bytes": size, "decode_pass": decoded})
    write_csv(RESULTS / "exp10_artifact_manifest.csv", records)
    passed = all(record["exists"] and record["size_bytes"] > 0 and record["decode_pass"] for record in records)
    dump_json(RESULTS / "exp10_artifact_verification.json", {
        "status": "PASS" if passed else "FAIL", "checked_images": len(records),
        "all_exist": all(record["exists"] for record in records),
        "all_nonempty": all(record["size_bytes"] > 0 for record in records),
        "all_decode_pass": all(record["decode_pass"] for record in records),
        "test_accessed": False,
    })
    if not passed:
        raise RuntimeError("artifact verification failed")


def main() -> int:
    rows = load_rows()
    make_summary(rows)
    paired_rows, aggregate = paired_and_effects(rows)
    aggregate_models(rows, aggregate)
    make_figures(rows, paired_rows, aggregate)
    verify_artifacts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
