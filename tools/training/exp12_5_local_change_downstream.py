#!/usr/bin/env python3
"""Exp12.5 controlled Route C downstream comparison.

Route C injects the fixed-final-epoch Exp12.4 local-change backbone into the
same official YOLO11n-seg model used by Exp12.3. Locked Exp12.3 Route A/B
artifacts are reused as references; only Route C is newly trained. TEST is
absent and never accessed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from tools.training import exp12_3_downstream as base


SMOKE_REFERENCE = base.REPO / "results/exp12_downstream_ab/smoke_20260822T085916Z"
FORMAL_REFERENCE = base.REPO / "results/exp12_downstream_ab/formal_20260822T090202Z"
LOCAL_BACKBONE = (
    base.REPO
    / "results/exp12_local_change/formal_20260823T064923Z/adapted_backbone_final.pt"
)
LOCAL_FORMAL_SUMMARY = (
    base.REPO / "results/exp12_local_change/formal_20260823T064923Z/summary.json"
)


class Exp12LocalChangeTrainer(base.Exp12SegmentationTrainer):
    route = "c"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super(base.Exp12SegmentationTrainer, self).get_model(
            cfg=cfg, weights=weights, verbose=verbose
        )
        backbone, tail = base.split_modules(model)
        official_backbone = base.cpu_state(backbone)
        tail_before = base.cpu_state(tail)
        parameter_names = set(dict(backbone.named_parameters()))

        source_payload = torch.load(self.ssl_backbone_path, map_location="cpu", weights_only=False)
        source = source_payload["backbone"]
        source_structure = base.compare_states(source, official_backbone)
        source_structure["value_difference_expected"] = True
        source_structure["structural_pass"] = not any(
            source_structure[key] for key in ("missing", "unexpected", "shape_mismatch")
        )
        if not source_structure["structural_pass"]:
            raise base.Exp12Gate("Local-change backbone keys/shapes do not match layers 0..10")

        backbone.load_state_dict(source, strict=True)
        immediate_backbone = base.cpu_state(backbone)
        tail_after = base.cpu_state(tail)
        expected_match = base.compare_states(source, immediate_backbone)
        tail_unchanged = base.compare_states(tail_before, tail_after)
        changed_names = [
            name for name in official_backbone
            if not torch.equal(official_backbone[name], immediate_backbone[name])
        ]
        changed_parameters = [name for name in changed_names if name in parameter_names]
        changed_buffers = [name for name in changed_names if name not in parameter_names]

        self.expected_backbone = base.cpu_state(backbone)
        self.expected_tail = base.cpu_state(tail)
        status = "PASS"
        if not expected_match["pass"] or not tail_unchanged["pass"]:
            status = "HARD_GATE"
        if not changed_parameters:
            status = "INVALID_BY_BACKBONE_NO_UPDATE"
        report = {
            "status": status,
            "route": "c",
            "route_description": "official COCO -> Exp12.4 local-change layers 0..10 -> supervised fine-tuning",
            "injection_boundary": [0, base.BACKBONE_END],
            "official_weights": str(base.OFFICIAL),
            "official_weights_sha256": base.file_sha256(base.OFFICIAL),
            "ssl_backbone": str(self.ssl_backbone_path),
            "ssl_backbone_sha256": base.file_sha256(self.ssl_backbone_path),
            "ssl_scope": source_payload.get("scope"),
            "ssl_encoder_layers": source_payload.get("encoder_layers"),
            "ssl_tap_layers": source_payload.get("tap_layers"),
            "source_structure": source_structure,
            "route_expected_backbone_match": expected_match,
            "tail_unchanged_by_injection": tail_unchanged,
            "backbone_state_tensor_count": len(immediate_backbone),
            "backbone_parameter_tensor_count": len(parameter_names),
            "changed_from_official_tensor_count": len(changed_names),
            "changed_from_official_parameter_count": len(changed_parameters),
            "changed_from_official_buffer_count": len(changed_buffers),
            "backbone_hashes": base.state_hashes(immediate_backbone),
            "tail_hashes": base.state_hashes(tail_after),
            "val_accessed": False,
            "test_accessed": False,
        }
        base.atomic_json(self.audit_dir / "injection_audit.json", report)
        if status != "PASS":
            raise base.Exp12Gate(status)
        return model


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_artifacts(route_dir: Path) -> dict[str, Any]:
    return {
        "summary": load_json(route_dir / "summary.json"),
        "injection": load_json(route_dir / "injection_audit.json"),
        "trainer": load_json(route_dir / "trainer_initialization_audit.json"),
        "requested": load_json(route_dir / "requested_args.json"),
        "first_batch": load_json(route_dir / "first_train_batch.json"),
    }


def run_route_c(output: Path, phase: str, epochs: int, data_yaml: Path,
                ssl_backbone: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    Exp12LocalChangeTrainer.route = "c"
    Exp12LocalChangeTrainer.audit_dir = output
    Exp12LocalChangeTrainer.ssl_backbone_path = ssl_backbone
    Exp12LocalChangeTrainer.instance = None
    requested = base.requested_args(data_yaml, output, epochs, phase)
    base.atomic_json(output / "requested_args.json", requested)
    environment = {
        "route": "c",
        "phase": phase,
        "official_weights": str(base.OFFICIAL),
        "official_weights_sha256": base.file_sha256(base.OFFICIAL),
        "ssl_backbone": str(ssl_backbone),
        "ssl_backbone_sha256": base.file_sha256(ssl_backbone),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": base.file_sha256(data_yaml),
        "train_image_count": base.image_count(base.DATA_ROOT / "images/train"),
        "val_image_count": base.image_count(base.DATA_ROOT / "images/val"),
        "test_path_present_in_yaml": "test:" in data_yaml.read_text(encoding="utf-8").lower(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    base.atomic_json(output / "environment.json", environment)
    if environment["train_image_count"] != base.EXPECTED_TRAIN_IMAGES or environment["test_path_present_in_yaml"]:
        raise base.Exp12Gate("Frozen data protocol failed")

    started = time.monotonic()
    model = YOLO(str(base.OFFICIAL))
    model.train(**requested, trainer=Exp12LocalChangeTrainer)
    trainer = Exp12LocalChangeTrainer.instance
    if trainer is None:
        raise base.Exp12Gate("Controlled Route C trainer instance unavailable")
    save_dir = Path(trainer.save_dir)
    losses_finite, epochs_completed, final_metrics = base.results_audit(save_dir / "results.csv")
    checkpoints = {
        name: base.checkpoint_audit(save_dir / "weights" / name)
        for name in ("best.pt", "last.pt")
    }
    checks = {
        "epochs_complete": epochs_completed == epochs,
        "train_losses_finite": losses_finite,
        "train_images_seen_exact": trainer.train_images_seen == base.EXPECTED_TRAIN_IMAGES * epochs,
        "optimizer_steps_positive": trainer.optimizer_steps > 0,
        "checkpoints_valid": all(
            item.get("torch_load_pass") and item.get("state_finite") and item.get("yolo_reload_pass")
            for item in checkpoints.values()
        ),
        "no_test_loader": all(
            "test" not in item["path"].replace("\\", "/").lower()
            for item in trainer.loader_paths
        ),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "HARD_GATE",
        "route": "c",
        "phase": phase,
        "checks": checks,
        "epochs_requested": epochs,
        "epochs_completed": epochs_completed,
        "train_images_seen": trainer.train_images_seen,
        "expected_train_images_seen": base.EXPECTED_TRAIN_IMAGES * epochs,
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
    base.atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "PASS" else 3


def fair_args(requested: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in requested.items() if key != "project"}


def comparison_gate(output: Path, reference: Path, phase: str,
                    local_backbone: Path) -> dict[str, Any]:
    a = route_artifacts(reference / "route_a")
    b = route_artifacts(reference / "route_b")
    c = route_artifacts(output / "route_c")
    checks = {
        "locked_route_a_pass": a["summary"]["status"] == "PASS",
        "locked_route_b_pass": b["summary"]["status"] == "PASS",
        "new_route_c_pass": c["summary"]["status"] == "PASS",
        "route_c_local_backbone_exact_after_trainer_setup": c["trainer"]["backbone_match_to_route_expected"]["pass"],
        "route_c_backbone_really_differs_from_official": c["injection"]["changed_from_official_parameter_count"] > 0,
        "neck_head_identical_after_model_construction": (
            a["injection"]["tail_hashes"] == b["injection"]["tail_hashes"] == c["injection"]["tail_hashes"]
        ),
        "neck_head_identical_after_trainer_setup": (
            a["trainer"]["tail_hashes"] == b["trainer"]["tail_hashes"] == c["trainer"]["tail_hashes"]
        ),
        "first_train_batch_stems_identical": (
            a["first_batch"]["stems"] == b["first_batch"]["stems"] == c["first_batch"]["stems"]
        ),
        "first_train_batch_tensor_identical": (
            a["first_batch"]["input_sha256"] == b["first_batch"]["input_sha256"] == c["first_batch"]["input_sha256"]
        ),
        "training_arguments_identical": (
            fair_args(a["requested"]) == fair_args(b["requested"]) == fair_args(c["requested"])
        ),
        "local_backbone_hash_exact": c["injection"]["ssl_backbone_sha256"] == base.file_sha256(local_backbone),
        "test_access_zero": not any(item["summary"]["test_accessed"] for item in (a, b, c)),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "HARD_GATE",
        "phase": phase,
        "checks": checks,
        "reference_exp12_3": str(reference),
        "reference_route_a_checkpoint_sha256": a["summary"]["checkpoints"]["best.pt"]["sha256"],
        "reference_route_b_checkpoint_sha256": b["summary"]["checkpoints"]["best.pt"]["sha256"],
        "route_c_changed_parameter_count": c["injection"]["changed_from_official_parameter_count"],
        "route_c_total_parameter_count": c["injection"]["backbone_parameter_tensor_count"],
        "route_c_changed_ratio": (
            c["injection"]["changed_from_official_parameter_count"]
            / c["injection"]["backbone_parameter_tensor_count"]
        ),
        "local_backbone_sha256": base.file_sha256(local_backbone),
        "test_accessed": False,
    }
    base.atomic_json(output / f"{phase}_gate.json", report)
    return report


def metric_delta(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    return {
        key: {
            "reference": reference[key],
            "candidate": candidate[key],
            "candidate_minus_reference": candidate[key] - reference[key],
        }
        for key in sorted(set(reference) & set(candidate))
    }


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference = SMOKE_REFERENCE if args.phase == "smoke" else FORMAL_REFERENCE
    for required in (
        reference / "route_a/summary.json",
        reference / "route_b/summary.json",
        reference / "exp12_train_val_only.yaml",
        args.local_backbone,
        args.local_formal_summary,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    upstream = load_json(args.local_formal_summary)
    if upstream.get("status") != "PASS" or upstream.get("changed_ratio", 0.0) <= 0:
        raise base.Exp12Gate("Exp12.4 formal backbone update Gate did not PASS")
    if upstream.get("backbone_export_sha256") != base.file_sha256(args.local_backbone):
        raise base.Exp12Gate("Exp12.4 backbone hash differs from formal summary")
    if args.phase == "formal":
        if args.smoke_gate is None or load_json(args.smoke_gate).get("status") != "PASS":
            raise base.Exp12Gate("Exp12.5 formal requires Route C smoke PASS")

    output.mkdir(parents=True)
    epochs = 1 if args.phase == "smoke" else 100
    data_yaml = reference / "exp12_train_val_only.yaml"
    protocol = {
        "scope": "EXP12.5_LOCAL_CHANGE_DOWNSTREAM_ABC",
        "phase": args.phase,
        "route_a": "locked Exp12.3 official COCO reference",
        "route_b": "locked Exp12.3 SimSiam reference",
        "route_c": "new Exp12.4 local-change backbone supervised fine-tuning",
        "reference_retrained": False,
        "epochs": epochs,
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": base.file_sha256(data_yaml),
        "selection_policy": "ULTRALYTICS_BEST_BY_FROZEN_VAL_FITNESS",
        "test_in_yaml": False,
        "test_accessed": False,
    }
    base.atomic_json(output / "protocol.json", protocol)
    result = run_route_c(output / "route_c", args.phase, epochs, data_yaml, args.local_backbone)
    if result != 0:
        return result
    gate = comparison_gate(output, reference, args.phase, args.local_backbone)
    if gate["status"] != "PASS":
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
        return 3

    if args.phase == "formal":
        route_c = load_json(output / "route_c/summary.json")
        c_report = base.run_validation(
            "c", Path(route_c["checkpoints"]["best.pt"]["path"]), data_yaml, output
        )
        a_report = load_json(reference / "route_a_frozen_val.json")
        b_report = load_json(reference / "route_b_frozen_val.json")
        reports = {"a": a_report, "b": b_report, "c": c_report}
        comparison = {
            "status": "PASS" if all(report["status"] == "PASS" for report in reports.values()) else "HARD_GATE",
            "primary_metric": "metrics/mAP50-95(M)",
            "selection_policy": "ULTRALYTICS_BEST_BY_FROZEN_VAL_FITNESS",
            "evaluation_split": "val",
            "route_a_vs_c": metric_delta(a_report["metrics"], c_report["metrics"]),
            "route_b_vs_c": metric_delta(b_report["metrics"], c_report["metrics"]),
            "route_metrics": {route: report["metrics"] for route, report in reports.items()},
            "test_accessed": False,
            "conclusion_scope": "INDEPENDENT_EXP12_RESEARCH_EXTENSION_ONLY",
            "exp00_exp11_conclusions_modified": False,
        }
        base.atomic_json(output / "frozen_val_comparison.json", comparison)
        gate["frozen_val_comparison_status"] = comparison["status"]
        gate["status"] = "PASS" if comparison["status"] == "PASS" else "HARD_GATE"
        base.atomic_json(output / "formal_gate.json", gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    return 0 if gate["status"] == "PASS" else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-backbone", type=Path, default=LOCAL_BACKBONE)
    parser.add_argument("--local-formal-summary", type=Path, default=LOCAL_FORMAL_SUMMARY)
    parser.add_argument("--smoke-gate", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
