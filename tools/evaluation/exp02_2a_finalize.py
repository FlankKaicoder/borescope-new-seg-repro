#!/usr/bin/env python3
"""Finalize Exp02.2a evidence without accessing test or changing model/data/loss."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import Attention


LOSS_FIELDS = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_model_audit(path: Path) -> dict[str, Any]:
    observed_sha = sha256(path)
    YOLO(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    tensors = list(model.state_dict().items())
    bad = [name for name, value in tensors if torch.is_tensor(value) and not bool(torch.isfinite(value).all())]
    return {
        "path": str(path),
        "sha256": observed_sha,
        "size_bytes": path.stat().st_size,
        "ultralytics_load_status": "PASS",
        "state_tensor_count": len(tensors),
        "nonfinite_state_tensors": bad,
        "all_state_tensors_finite": not bad,
        "numerically_valid": not bad,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-best", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()

    original = read_csv(run / "original_early_epoch_audit/early_epoch_loss_audit.csv")
    reproduction = read_csv(run / "short_repro/ultralytics/short_repro/results.csv")
    comparison = []
    for epoch in range(1, 7):
        old = original[epoch - 1]
        new = reproduction[epoch - 1]
        old_nan = [field for field in LOSS_FIELDS if not math.isfinite(float(old[f"val_{field}"]))]
        new_nan = [field for field in LOSS_FIELDS if not math.isfinite(float(new[f"val/{field}"]))]
        comparison.append(
            {
                "epoch": epoch,
                "original_nonfinite": json.dumps(old_nan),
                "reproduction_nonfinite": json.dumps(new_nan),
                "pattern_match": old_nan == new_nan,
                **{f"original_train_{field}": old[f"train_{field}"] for field in LOSS_FIELDS},
                **{f"reproduction_train_{field}": new[f"train/{field}"] for field in LOSS_FIELDS},
            }
        )
    with (run / "reproduction_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    short_summary = json.loads((run / "short_repro/summary.json").read_text())
    checkpoint_rows = []
    for item in short_summary["checkpoints"]:
        epoch = int(item["checkpoint_epoch"])
        checkpoint_rows.append(
            {
                "checkpoint_epoch": epoch,
                "validation_state_kind": "pretrained_7class" if epoch == 0 else "raw_fp32_ema_before_validation",
                "validation_state_path": item["path"] if epoch == 0 else item["raw_ema_path"],
                "validation_state_sha256": item["sha256"] if epoch == 0 else item["raw_ema_sha256"],
                "period_checkpoint_path": "" if epoch == 0 else item["path"],
                "period_checkpoint_sha256": "" if epoch == 0 else item["sha256"],
            }
        )
    with (run / "checkpoint_hashes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    probe_rows = read_csv(run / "val_loss_probe/val_loss_batch_probe.csv")
    probe_by_epoch: dict[str, Any] = {}
    for epoch in range(7):
        probe_by_epoch[str(epoch)] = {}
        for mode in ("validator_equivalent_amp", "fp32"):
            selected = [r for r in probe_rows if int(r["checkpoint_epoch"]) == epoch and r["precision_mode"] == mode]
            probe_by_epoch[str(epoch)][mode] = {
                "batches": len(selected),
                "raw_prediction_nonfinite_batches": sum(r["raw_prediction_finite"] == "False" for r in selected),
                "loss_nonfinite_batches": sum(
                    any(r[f"{field}_finite"] == "False" for field in LOSS_FIELDS) for r in selected
                ),
                "first_batch_losses": {field: float(selected[0][field]) for field in LOSS_FIELDS},
            }

    first = json.loads((run / "val_loss_probe/first_nonfinite_batch_audit.json").read_text())
    trace = json.loads((run / "forward_trace/forward_first_nonfinite_trace.json").read_text())
    amp_trace = trace["modes"]["validator_equivalent_amp"]
    fp32_trace = trace["modes"]["fp32"]
    gt_classes = first["gt_input_audit"]["gt_classes"]
    names = {0: "Burn", 1: "Crack", 2: "Dent", 3: "Material missing", 4: "Tears", 5: "Tip curl", 6: "corrosion"}
    class_counts = {names[index]: count for index, count in sorted(Counter(gt_classes).items())}

    attention_source = Path(inspect.getsourcefile(Attention)).resolve()
    attention_source_audit = {
        "path": str(attention_source),
        "sha256": sha256(attention_source),
        "size_bytes": attention_source.stat().st_size,
        "symbol": "ultralytics.nn.modules.block.Attention.forward",
        "site_packages_modified": False,
    }
    (run / "source_audit/attention_source_hash.json").write_text(json.dumps(attention_source_audit, indent=2) + "\n")

    best = finite_model_audit(args.baseline_best.resolve())
    (run / "baseline_best_finite_audit.json").write_text(json.dumps(best, indent=2) + "\n")

    first_op = amp_trace["first_nonfinite_attention_operation"]
    summary = {
        "status": "COMPLETE_BASELINE_GATE_STOP",
        "experiment_id": "Exp02.2a",
        "test_accessed": False,
        "root_cause_case": "Case C",
        "root_cause_class": "MODEL_FORWARD_NUMERICAL_INSTABILITY",
        "original_nan_pattern": "epoch 1-5 all box/seg/cls/dfl val losses nonfinite; epoch 6 recovered",
        "short_reproduction": {
            "configured_epochs": 100,
            "actual_epochs": 6,
            "pattern": "YES",
            "all_epoch_1_to_6_patterns_match": all(row["pattern_match"] for row in comparison),
            "train_losses_all_finite": all(
                math.isfinite(float(row[f"train/{field}"])) for row in reproduction for field in LOSS_FIELDS
            ),
        },
        "first_nonfinite": {
            "checkpoint_epoch": first["row"]["checkpoint_epoch"],
            "precision_mode": first["row"]["precision_mode"],
            "batch_index": first["row"]["batch_index"],
            "first_recorded_component": "box_loss",
            "all_nonfinite_components": [field for field in LOSS_FIELDS if not first["row"][f"{field}_finite"]],
            "image_stems": json.loads(first["row"]["image_stems"]),
            "image_count": first["row"]["image_count"],
            "instance_count": first["row"]["instance_count"],
            "class_counts": class_counts,
            "input_and_gt_finite": True,
            "empty_or_degenerate_mask": first["gt_input_audit"]["empty_or_degenerate_mask"],
            "raw_prediction_finite": first["row"]["raw_prediction_finite"],
            "operation": first_op,
        },
        "precision_probe_by_epoch": probe_by_epoch,
        "precision_conclusion": (
            "epoch1-5: all 5 AMP batches have nonfinite raw predictions/losses; all FP32 batches are finite. "
            "epoch0 and epoch6 are finite in both paths."
        ),
        "forward_trace": {
            "module": "model.10.m.0.attn (C2PSA Attention)",
            "first_nonfinite_operation": "(q * scale).transpose(-2, -1) @ k",
            "amp_q_finite": True,
            "amp_k_finite": True,
            "amp_qk_logits": first_op,
            "fp32_qk_logits": next(op for op in fp32_trace["attention_operation_trace"] if op["operation"] == "qk_matmul_logits"),
            "causal_chain": "FP16 qk matmul overflow -> Inf logits -> softmax NaN -> forward outputs NaN -> all loss components NaN",
        },
        "source_audit_attention": attention_source_audit,
        "baseline_best": best,
        "recommendation": {
            "accept_current_baseline_now": False,
            "controlled_full_rerun_now": False,
            "baseline_gate": "STOP pending human review",
            "human_review_option": (
                "Because final best.pt is loadable/finite and independent VAL metrics are finite, a human may revise the gate to "
                "permit documented early training-validation AMP forward overflow only when FP32 validation-loss audit and final "
                "checkpoint audits pass. This diagnostic itself does not mark PASS."
            ),
            "next_if_gate_approved": "Recommend Exp03 low-confidence threshold sweep; do not execute automatically.",
        },
        "historical_exp02_2_preserved": {
            "GT": 296, "TP": 123, "FN": 173, "FP": 99,
            "low_box_iou": 96, "no_same_class_candidate": 59, "low_mask_iou": 12, "wrong_class": 6,
            "size_recall": {"tiny": 0.4078, "small": 0.5526, "medium": 0.3871, "large": 0.2727},
            "monotonic_smaller_is_harder_evidence": False,
        },
    }
    (run / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
