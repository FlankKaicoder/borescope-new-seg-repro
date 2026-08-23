#!/usr/bin/env python3
"""Exp12.3 controlled downstream A/B comparison after the Exp12.2 gate.

Route A starts from the official COCO YOLO11n-seg checkpoint. Route B uses the
same model construction and replaces only model layers 0..10 with the final
Exp12.1 SimSiam backbone. Every route is executed in a fresh Python process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer


REPO = Path("/root/autodl-tmp/borescope-new-seg-repro")
DATA_ROOT = Path("/root/autodl-tmp/borescope-new-seg-data/v1")
OFFICIAL = REPO / "weights/yolo11n-seg.pt"
SSL_BACKBONE = (
    REPO
    / "results/exp12_simsiam_basic/formal_20260822T071127Z/adapted_backbone_final.pt"
)
BACKBONE_END = 10
EXPECTED_TRAIN_IMAGES = 668
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Exp12Gate(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in state.items()}


def finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if (value.is_floating_point() or value.is_complex()) else True
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return True


def compare_states(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> dict[str, Any]:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    shape_mismatch = sorted(
        name for name in set(expected) & set(actual) if expected[name].shape != actual[name].shape
    )
    comparable = sorted(set(expected) & set(actual) - set(shape_mismatch))
    mismatched = [
        name
        for name in comparable
        if not torch.equal(expected[name].to(actual[name].dtype), actual[name].detach().cpu())
    ]
    return {
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "exact_match_count": len(comparable) - len(mismatched),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "value_mismatch": mismatched,
        "pass": not (missing or unexpected or shape_mismatch or mismatched),
    }


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def split_modules(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    model = unwrap(model)
    return model.model[: BACKBONE_END + 1], model.model[BACKBONE_END + 1 :]


def checkpoint_audit(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return report
    report["sha256"] = file_sha256(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = payload.get("ema") or payload.get("model")
        report["state_finite"] = model is not None and finite(model.state_dict())
        report["torch_load_pass"] = True
    except Exception as exc:  # pragma: no cover - retained in the artifact
        report.update(torch_load_pass=False, load_error=repr(exc), state_finite=False)
    try:
        YOLO(str(path))
        report["yolo_reload_pass"] = True
    except Exception as exc:  # pragma: no cover - retained in the artifact
        report.update(yolo_reload_pass=False, yolo_reload_error=repr(exc))
    return report


class Exp12SegmentationTrainer(SegmentationTrainer):
    route = "a"
    audit_dir = Path(".")
    ssl_backbone_path = SSL_BACKBONE
    instance: "Exp12SegmentationTrainer | None" = None

    def __init__(self, *args, **kwargs):
        type(self).instance = self
        self.expected_backbone: dict[str, torch.Tensor] = {}
        self.expected_tail: dict[str, torch.Tensor] = {}
        self.train_images_seen = 0
        self.first_train_batch: dict[str, Any] | None = None
        self.optimizer_steps = 0
        self.loader_paths: list[dict[str, str]] = []
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        backbone, tail = split_modules(model)
        official_backbone = cpu_state(backbone)
        tail_before = cpu_state(tail)
        parameter_names = set(dict(backbone.named_parameters()))

        source_payload = torch.load(self.ssl_backbone_path, map_location="cpu", weights_only=False)
        source = source_payload["backbone"]
        source_structure = compare_states(source, official_backbone)
        source_structure['value_difference_expected'] = True
        source_structure['structural_pass'] = not any(
            source_structure[key] for key in ('missing', 'unexpected', 'shape_mismatch')
        )
        if not source_structure['structural_pass']:
            raise Exp12Gate('SSL backbone keys or shapes do not strictly match YOLO layers 0..10')

        if self.route == "b":
            backbone.load_state_dict(source, strict=True)
        immediate_backbone = cpu_state(backbone)
        tail_after = cpu_state(tail)
        expected = source if self.route == "b" else official_backbone
        expected_match = compare_states(expected, immediate_backbone)
        tail_unchanged = compare_states(tail_before, tail_after)
        changed_names = [
            name
            for name in official_backbone
            if not torch.equal(official_backbone[name], immediate_backbone[name])
        ]
        changed_parameters = [name for name in changed_names if name in parameter_names]
        changed_buffers = [name for name in changed_names if name not in parameter_names]

        self.expected_backbone = cpu_state(backbone)
        self.expected_tail = cpu_state(tail)
        status = "PASS"
        if not expected_match["pass"] or not tail_unchanged["pass"]:
            status = "HARD_GATE"
        if self.route == "b" and not changed_parameters:
            status = "INVALID_BY_BACKBONE_NO_UPDATE"
        report = {
            "status": status,
            "route": self.route,
            "injection_boundary": [0, BACKBONE_END],
            "official_weights": str(OFFICIAL),
            "official_weights_sha256": file_sha256(OFFICIAL),
            "ssl_backbone": str(self.ssl_backbone_path),
            "ssl_backbone_sha256": file_sha256(self.ssl_backbone_path),
            "ssl_scope": source_payload.get("scope"),
            "ssl_encoder_layers": source_payload.get("encoder_layers"),
            "source_structure": source_structure,
            "route_expected_backbone_match": expected_match,
            "tail_unchanged_by_injection": tail_unchanged,
            "backbone_state_tensor_count": len(immediate_backbone),
            "backbone_parameter_tensor_count": len(parameter_names),
            "changed_from_official_tensor_count": len(changed_names),
            "changed_from_official_parameter_count": len(changed_parameters),
            "changed_from_official_buffer_count": len(changed_buffers),
            "backbone_hashes": state_hashes(immediate_backbone),
            "tail_hashes": state_hashes(tail_after),
            "val_accessed": False,
            "test_accessed": False,
        }
        atomic_json(self.audit_dir / "injection_audit.json", report)
        if status != "PASS":
            raise Exp12Gate(status)
        return model

    def _setup_train(self):
        super()._setup_train()
        backbone, tail = split_modules(self.model)
        actual_backbone = cpu_state(backbone)
        actual_tail = cpu_state(tail)
        backbone_match = compare_states(self.expected_backbone, actual_backbone)
        tail_match = compare_states(self.expected_tail, actual_tail)
        trainable_backbone = sum(parameter.requires_grad for parameter in backbone.parameters())
        total_backbone = sum(1 for _ in backbone.parameters())
        report = {
            "status": "PASS" if backbone_match["pass"] and tail_match["pass"] and trainable_backbone > 0 else "HARD_GATE",
            "route": self.route,
            "audit_point": "AFTER_TRAINER_SETUP_BEFORE_FIRST_TRAIN_BATCH",
            "backbone_match_to_route_expected": backbone_match,
            "tail_match_to_pre_setup": tail_match,
            "backbone_trainable_parameter_count": trainable_backbone,
            "backbone_total_parameter_count": total_backbone,
            "backbone_hashes": state_hashes(actual_backbone),
            "tail_hashes": state_hashes(actual_tail),
            "requested_amp": bool(self.args.amp),
            "effective_amp": bool(self.amp),
            "device": str(self.device),
            "val_accessed": False,
            "test_accessed": False,
        }
        atomic_json(self.audit_dir / "trainer_initialization_audit.json", report)
        if report["status"] != "PASS":
            raise Exp12Gate("Trainer initialization changed the injected state or froze the backbone")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        resolved = str(dataset_path)
        self.loader_paths.append({"mode": str(mode), "path": resolved})
        normalized = resolved.replace("\\", "/").lower()
        if "/images/test" in normalized or normalized.endswith("/test"):
            raise Exp12Gate("TEST loader construction is forbidden")
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def preprocess_batch(self, batch):
        processed = super().preprocess_batch(batch)
        stems = [Path(path).stem for path in batch["im_file"]]
        self.train_images_seen += len(stems)
        if self.first_train_batch is None:
            self.first_train_batch = {
                "stems": stems,
                "input_shape": list(processed["img"].shape),
                "input_dtype": str(processed["img"].dtype),
                "input_sha256": tensor_sha256(processed["img"]),
                "input_finite": finite(processed["img"]),
                "targets_finite": finite({key: value for key, value in processed.items() if key != "img"}),
            }
            atomic_json(self.audit_dir / "first_train_batch.json", self.first_train_batch)
            if not self.first_train_batch["input_finite"] or not self.first_train_batch["targets_finite"]:
                raise Exp12Gate("First TRAIN batch is non-finite")
        return processed

    def optimizer_step(self):
        if not finite(self.loss):
            raise Exp12Gate("Non-finite TRAIN loss reached optimizer_step")
        super().optimizer_step()
        self.optimizer_steps += 1

    def _handle_nan_recovery(self, epoch):
        if self.tloss is not None and not finite(self.tloss):
            raise Exp12Gate(f"Non-finite aggregate TRAIN loss at epoch {int(epoch) + 1}")
        return False


def write_train_val_yaml(path: Path) -> None:
    text = """# Exp12.3 frozen supervised TRAIN/VAL configuration. TEST is intentionally absent.
path: /root/autodl-tmp/borescope-new-seg-data/v1
train: images/train
val: images/val
names:
  0: Burn
  1: Crack
  2: Dent
  3: Material missing
  4: Tears
  5: Tip curl
  6: corrosion
"""
    path.write_text(text, encoding="utf-8")


def image_count(directory: Path) -> int:
    return sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir())


def requested_args(data_yaml: Path, output: Path, epochs: int, phase: str) -> dict[str, Any]:
    return {
        "data": str(data_yaml),
        "imgsz": 640,
        "batch": 32,
        "epochs": epochs,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "optimizer": "AdamW",
        "device": 0,
        "workers": 4,
        "cache": False,
        "val": True,
        "plots": phase == "formal",
        "save": True,
        "patience": 100,
        "project": str(output / "ultralytics"),
        "name": "train",
        "exist_ok": False,
        "verbose": True,
    }


def results_audit(path: Path) -> tuple[bool, int, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False, 0, {}
    train_loss_keys = [key for key in rows[0] if key.startswith("train/") and key.endswith("loss")]
    losses_finite = all(math.isfinite(float(row[key])) for row in rows for key in train_loss_keys)
    final_metrics = {}
    for key, value in rows[-1].items():
        try:
            final_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return losses_finite, len(rows), final_metrics


def run_route(args) -> int:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    Exp12SegmentationTrainer.route = args.route
    Exp12SegmentationTrainer.audit_dir = args.output
    Exp12SegmentationTrainer.ssl_backbone_path = args.ssl_backbone
    Exp12SegmentationTrainer.instance = None
    requested = requested_args(args.data, args.output, args.epochs, args.phase)
    atomic_json(args.output / "requested_args.json", requested)
    environment = {
        "route": args.route,
        "phase": args.phase,
        "official_weights": str(OFFICIAL),
        "official_weights_sha256": file_sha256(OFFICIAL),
        "ssl_backbone": str(args.ssl_backbone),
        "ssl_backbone_sha256": file_sha256(args.ssl_backbone),
        "data_yaml": str(args.data),
        "data_yaml_sha256": file_sha256(args.data),
        "train_image_count": image_count(DATA_ROOT / "images/train"),
        "val_image_count": image_count(DATA_ROOT / "images/val"),
        "test_path_present_in_yaml": "test:" in args.data.read_text(encoding="utf-8").lower(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    atomic_json(args.output / "environment.json", environment)
    if environment["train_image_count"] != EXPECTED_TRAIN_IMAGES or environment["test_path_present_in_yaml"]:
        raise Exp12Gate("Frozen data protocol failed")

    started = time.monotonic()
    model = YOLO(str(OFFICIAL))
    model.train(**requested, trainer=Exp12SegmentationTrainer)
    trainer = Exp12SegmentationTrainer.instance
    if trainer is None:
        raise Exp12Gate("Controlled trainer instance unavailable")
    save_dir = Path(trainer.save_dir)
    losses_finite, epochs_completed, final_metrics = results_audit(save_dir / "results.csv")
    checkpoints = {
        name: checkpoint_audit(save_dir / "weights" / name) for name in ("best.pt", "last.pt")
    }
    expected_images = EXPECTED_TRAIN_IMAGES * args.epochs
    checks = {
        "epochs_complete": epochs_completed == args.epochs,
        "train_losses_finite": losses_finite,
        "train_images_seen_exact": trainer.train_images_seen == expected_images,
        "optimizer_steps_positive": trainer.optimizer_steps > 0,
        "checkpoints_valid": all(
            item.get("torch_load_pass") and item.get("state_finite") and item.get("yolo_reload_pass")
            for item in checkpoints.values()
        ),
        "no_test_loader": all("test" not in item["path"].replace("\\", "/").lower() for item in trainer.loader_paths),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "HARD_GATE",
        "route": args.route,
        "phase": args.phase,
        "checks": checks,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "train_images_seen": trainer.train_images_seen,
        "expected_train_images_seen": expected_images,
        "optimizer_steps": trainer.optimizer_steps,
        "first_train_batch": trainer.first_train_batch,
        "loader_paths": trainer.loader_paths,
        "final_epoch_metrics": final_metrics,
        "checkpoints": checkpoints,
        "save_dir": str(save_dir),
        "val_accessed": True,
        "test_accessed": False,
        "retry_performed": False,
        "hyperparameters_modified": False,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "PASS" else 3


def route_artifacts(route_dir: Path) -> dict[str, Any]:
    return {
        "summary": json.loads((route_dir / "summary.json").read_text(encoding="utf-8")),
        "injection": json.loads((route_dir / "injection_audit.json").read_text(encoding="utf-8")),
        "trainer": json.loads((route_dir / "trainer_initialization_audit.json").read_text(encoding="utf-8")),
        "requested": json.loads((route_dir / "requested_args.json").read_text(encoding="utf-8")),
        "first_batch": json.loads((route_dir / "first_train_batch.json").read_text(encoding="utf-8")),
    }


def fair_args(requested: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in requested.items() if key not in {"project"}}


def pair_gate(output: Path, phase: str) -> dict[str, Any]:
    a = route_artifacts(output / "route_a")
    b = route_artifacts(output / "route_b")
    checks = {
        "route_a_pass": a["summary"]["status"] == "PASS",
        "route_b_pass": b["summary"]["status"] == "PASS",
        "route_a_official_backbone_exact_after_trainer_setup": a["trainer"]["backbone_match_to_route_expected"]["pass"],
        "route_b_ssl_backbone_exact_after_trainer_setup": b["trainer"]["backbone_match_to_route_expected"]["pass"],
        "route_b_backbone_really_differs_from_official": b["injection"]["changed_from_official_parameter_count"] > 0,
        "route_a_injection_changed_zero_parameters": a["injection"]["changed_from_official_parameter_count"] == 0,
        "neck_head_identical_after_model_construction": a["injection"]["tail_hashes"] == b["injection"]["tail_hashes"],
        "neck_head_identical_after_trainer_setup": a["trainer"]["tail_hashes"] == b["trainer"]["tail_hashes"],
        "first_train_batch_stems_identical": a["first_batch"]["stems"] == b["first_batch"]["stems"],
        "first_train_batch_tensor_identical": a["first_batch"]["input_sha256"] == b["first_batch"]["input_sha256"],
        "training_arguments_identical": fair_args(a["requested"]) == fair_args(b["requested"]),
        "test_access_zero": not a["summary"]["test_accessed"] and not b["summary"]["test_accessed"],
    }
    report = {
        "status": "PASS" if all(checks.values()) else "HARD_GATE",
        "phase": phase,
        "checks": checks,
        "route_b_changed_parameter_count": b["injection"]["changed_from_official_parameter_count"],
        "route_b_total_parameter_count": b["injection"]["backbone_parameter_tensor_count"],
        "route_b_changed_ratio": b["injection"]["changed_from_official_parameter_count"] / b["injection"]["backbone_parameter_tensor_count"],
        "official_weights_sha256": a["injection"]["official_weights_sha256"],
        "ssl_backbone_sha256": b["injection"]["ssl_backbone_sha256"],
        "test_accessed": False,
    }
    atomic_json(output / f"{phase}_gate.json", report)
    return report


def run_validation(route: str, best: Path, data: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    model = YOLO(str(best))
    metrics = model.val(
        data=str(data), split="val", imgsz=640, batch=32, device=0, workers=4,
        plots=True, project=str(output / "frozen_val"), name=f"route_{route}",
        exist_ok=False, verbose=True,
    )
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    report = {
        "status": "PASS" if results and all(math.isfinite(value) for value in results.values()) else "HARD_GATE",
        "route": route,
        "checkpoint": str(best),
        "checkpoint_sha256": file_sha256(best),
        "selection_policy": "ULTRALYTICS_BEST_BY_FROZEN_VAL_FITNESS",
        "evaluation_split": "val",
        "metrics": results,
        "test_accessed": False,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(output / f"route_{route}_frozen_val.json", report)
    return report


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        key: {"route_a": a[key], "route_b": b[key], "b_minus_a": b[key] - a[key]}
        for key in sorted(set(a) & set(b))
    }


def run_pair(args) -> int:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    data_yaml = args.output / "exp12_train_val_only.yaml"
    write_train_val_yaml(data_yaml)
    if args.phase == "formal":
        if args.smoke_gate is None:
            raise Exp12Gate("Formal run requires --smoke-gate")
        smoke = json.loads(args.smoke_gate.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS":
            raise Exp12Gate("Exp12.3-S smoke Gate did not PASS")
        if smoke.get("official_weights_sha256") != file_sha256(OFFICIAL):
            raise Exp12Gate("Official checkpoint changed after smoke")
        if smoke.get("ssl_backbone_sha256") != file_sha256(args.ssl_backbone):
            raise Exp12Gate("SSL backbone changed after smoke")
    epochs = 1 if args.phase == "smoke" else 100
    protocol = {
        "scope": "EXP12.3_SIMSIAM_DOWNSTREAM_AB",
        "phase": args.phase,
        "route_a": "official COCO YOLO11n-seg -> supervised fine-tuning",
        "route_b": "official COCO YOLO11n-seg -> strict layers 0..10 SSL injection -> supervised fine-tuning",
        "epochs": epochs,
        "selection_policy": "ULTRALYTICS_BEST_BY_FROZEN_VAL_FITNESS",
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": file_sha256(data_yaml),
        "test_in_yaml": False,
        "test_accessed": False,
    }
    atomic_json(args.output / "protocol.json", protocol)
    script = Path(__file__).resolve()
    for route in ("a", "b"):
        command = [
            sys.executable, str(script), "route", "--route", route, "--phase", args.phase,
            "--epochs", str(epochs), "--data", str(data_yaml), "--ssl-backbone",
            str(args.ssl_backbone), "--output", str(args.output / f"route_{route}"),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            atomic_json(args.output / f"{args.phase}_gate.json", {
                "status": "HARD_GATE", "failed_route": route,
                "returncode": completed.returncode, "test_accessed": False,
            })
            return completed.returncode
    gate = pair_gate(args.output, args.phase)
    if gate["status"] != "PASS":
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
        return 3
    if args.phase == "formal":
        route_reports = {}
        for route in ("a", "b"):
            summary = json.loads((args.output / f"route_{route}/summary.json").read_text(encoding="utf-8"))
            best = Path(summary["checkpoints"]["best.pt"]["path"])
            route_reports[route] = run_validation(route, best, data_yaml, args.output)
        comparison = {
            "status": "PASS" if all(item["status"] == "PASS" for item in route_reports.values()) else "HARD_GATE",
            "primary_metric": "metrics/mAP50-95(M)",
            "selection_policy": "ULTRALYTICS_BEST_BY_FROZEN_VAL_FITNESS",
            "evaluation_split": "val",
            "metrics": metric_delta(route_reports["a"]["metrics"], route_reports["b"]["metrics"]),
            "test_accessed": False,
            "conclusion_scope": "INDEPENDENT_EXP12_RESEARCH_EXTENSION_ONLY",
            "exp00_exp11_conclusions_modified": False,
        }
        atomic_json(args.output / "frozen_val_comparison.json", comparison)
        gate["frozen_val_comparison_status"] = comparison["status"]
        gate["status"] = "PASS" if gate["status"] == comparison["status"] == "PASS" else "HARD_GATE"
        atomic_json(args.output / "formal_gate.json", gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    return 0 if gate["status"] == "PASS" else 3


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    route = subparsers.add_parser("route")
    route.add_argument("--route", choices=["a", "b"], required=True)
    route.add_argument("--phase", choices=["smoke", "formal"], required=True)
    route.add_argument("--epochs", type=int, required=True)
    route.add_argument("--data", type=Path, required=True)
    route.add_argument("--ssl-backbone", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--phase", choices=["smoke", "formal"], required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--ssl-backbone", type=Path, default=SSL_BACKBONE)
    pair.add_argument("--smoke-gate", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "route":
        return run_route(args)
    return run_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
