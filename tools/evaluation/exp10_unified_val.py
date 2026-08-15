#!/usr/bin/env python3
"""Unified VAL-only evaluator for Exp10 checkpoints; TEST is never accessed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=/absolute/path/to/best.pt")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("model name is empty")
    return name, Path(raw_path).resolve()


def scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def metric_record(metric, index: int | None = None) -> dict[str, float]:
    if index is None:
        return {
            "P": scalar(metric.mp),
            "R": scalar(metric.mr),
            "mAP50": scalar(metric.map50),
            "mAP50_95": scalar(metric.map),
        }
    return {
        "P": scalar(metric.p[index]),
        "R": scalar(metric.r[index]),
        "mAP50": scalar(metric.ap50[index]),
        "mAP50_95": scalar(metric.maps[index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    rows = []
    details = []
    for model_name, checkpoint in args.model:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        val_project = args.output / "ultralytics"
        metrics = YOLO(str(checkpoint)).val(
            data=str(args.data.resolve()),
            split="val",
            imgsz=640,
            batch=32,
            device=0,
            plots=True,
            project=str(val_project),
            name=model_name,
            exist_ok=False,
            verbose=True,
        )
        fixed_path = args.output / f"{model_name}_fixed_size.json"
        subprocess.run(
            [
                sys.executable,
                str(repo / "tools/evaluation/exp10_fixed_size.py"),
                "--weights", str(checkpoint),
                "--data-root", str(args.data_root.resolve()),
                "--manifest", str(args.manifest.resolve()),
                "--output", str(fixed_path),
            ],
            check=True,
        )
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
        names = metrics.names
        per_class = {}
        for class_id in range(len(names)):
            class_name = names[class_id]
            per_class[class_name] = {
                "class_id": class_id,
                "Box": metric_record(metrics.box, class_id),
                "Mask": metric_record(metrics.seg, class_id),
            }
        detail = {
            "seed": args.seed,
            "model": model_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "split": "val",
            "test_accessed": False,
            "Box": metric_record(metrics.box),
            "Mask": metric_record(metrics.seg),
            "per_class": per_class,
            "fixed_point": fixed,
            "ultralytics_save_dir": str(metrics.save_dir),
        }
        atomic_json(args.output / f"{model_name}_metrics.json", detail)
        details.append(detail)
        row = {
            "seed": args.seed,
            "model": model_name,
            "Box_P": detail["Box"]["P"],
            "Box_R": detail["Box"]["R"],
            "Box_mAP50": detail["Box"]["mAP50"],
            "Box_mAP50_95": detail["Box"]["mAP50_95"],
            "Mask_P": detail["Mask"]["P"],
            "Mask_R": detail["Mask"]["R"],
            "Mask_mAP50": detail["Mask"]["mAP50"],
            "Mask_mAP50_95": detail["Mask"]["mAP50_95"],
            "TP": fixed["TP"],
            "FP": fixed["FP"],
            "FN": fixed["FN"],
            "fixed_P": fixed["precision"],
            "fixed_R": fixed["recall"],
            "fixed_F1": fixed["F1"],
            "tiny_R": fixed["size_recall"]["tiny"],
            "small_R": fixed["size_recall"]["small"],
            "medium_R": fixed["size_recall"]["medium"],
            "large_R": fixed["size_recall"]["large"],
            "Burn_R": per_class["Burn"]["Mask"]["R"],
            "Burn_AP50": per_class["Burn"]["Mask"]["mAP50"],
            "Burn_AP50_95": per_class["Burn"]["Mask"]["mAP50_95"],
            "Crack_R": per_class["Crack"]["Mask"]["R"],
            "Crack_AP50": per_class["Crack"]["Mask"]["mAP50"],
            "Crack_AP50_95": per_class["Crack"]["Mask"]["mAP50_95"],
            "corrosion_R": per_class["corrosion"]["Mask"]["R"],
            "corrosion_AP50": per_class["corrosion"]["Mask"]["mAP50"],
            "corrosion_AP50_95": per_class["corrosion"]["Mask"]["mAP50_95"],
            "checkpoint_sha256": detail["checkpoint_sha256"],
            "test_accessed": False,
        }
        rows.append(row)
    csv_path = args.output / f"seed{args.seed}_unified_val.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.output / "summary.json", {
        "status": "PASS",
        "seed": args.seed,
        "models": [name for name, _ in args.model],
        "evaluator": "Ultralytics 8.4.117 VAL + fixed conf=0.25 mask-IoU=0.50",
        "size_thresholds": [0.0020196759259259256, 0.007364969135802469, 0.029209280303030303],
        "test_accessed": False,
        "details": details,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
