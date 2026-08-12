#!/usr/bin/env python3
"""Reproducible Exp02 YOLO11n-seg probe, smoke, and baseline runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
import ultralytics
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def environment() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": props.name, "gpu_total_bytes": props.total_memory,
    }


def common_args(args: argparse.Namespace, epochs: int, val: bool, save: bool, fraction: float) -> dict[str, object]:
    return {
        "data": str(args.data), "imgsz": 640, "epochs": epochs, "batch": args.batch,
        "seed": 42, "deterministic": True, "amp": True, "optimizer": "AdamW", "device": 0,
        "project": str(args.output / "ultralytics"), "name": args.mode, "exist_ok": False,
        "workers": args.workers, "cache": False, "val": val, "plots": val, "save": save,
        "fraction": fraction, "verbose": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "smoke", "baseline"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=0.03)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one CUDA device, got {torch.cuda.device_count()}")
    if "test" in str(args.data).lower():
        raise RuntimeError("Training data argument unexpectedly references test")

    start = datetime.now(timezone.utc); wall = time.monotonic()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    status = "FAIL"; error = ""; save_dir = None
    try:
        model = YOLO(str(args.model))
        if args.mode == "probe":
            resolved = common_args(args, epochs=1, val=False, save=False, fraction=args.fraction)
        elif args.mode == "smoke":
            resolved = common_args(args, epochs=1, val=True, save=True, fraction=1.0)
        else:
            resolved = common_args(args, epochs=args.epochs, val=True, save=True, fraction=1.0)
        dump(args.output / "requested_args.json", resolved)
        model.train(**resolved)
        save_dir = Path(model.trainer.save_dir)
        shutil.copy2(save_dir / "args.yaml", args.output / "resolved_args.yaml")
        status = "PASS"
    except torch.cuda.OutOfMemoryError as exc:
        status = "OOM"; error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status = "FAIL"; error = f"{type(exc).__name__}: {exc}"
    peak_alloc = torch.cuda.max_memory_allocated(0); peak_reserved = torch.cuda.max_memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    checkpoints: dict[str, object] = {}
    if save_dir:
        for name in ("best.pt", "last.pt"):
            path = save_dir / "weights" / name
            if path.is_file():
                checkpoints[name] = {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        if args.mode == "smoke" and checkpoints:
            reload_path = Path(checkpoints.get("best.pt", checkpoints["last.pt"])["path"])
            reloaded = YOLO(str(reload_path))
            reload_metrics = reloaded.val(data=str(args.data), split="val", imgsz=640, batch=args.batch, device=0, plots=False, verbose=False)
            checkpoints["reload_val_results"] = dict(reload_metrics.results_dict)
    summary = {
        "status": status, "mode": args.mode, "batch": args.batch,
        "start_utc": start.isoformat(), "end_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - wall, "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved, "gpu_total_bytes": total,
        "reserved_headroom_fraction": max(0.0, (total - peak_reserved) / total),
        "error": error, "environment": environment(), "model_path": str(args.model),
        "model_sha256": sha256(args.model), "data": str(args.data), "save_dir": str(save_dir) if save_dir else None,
        "checkpoints": checkpoints,
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else (42 if status == "OOM" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
