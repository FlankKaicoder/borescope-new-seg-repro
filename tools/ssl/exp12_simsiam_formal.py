#!/usr/bin/env python3
"""Exp12.1 formal TRAIN-only SimSiam and Exp12.2 parameter audit.

The run is fixed-budget and uses only the frozen TRAIN split. It does not use
VAL/TEST, early stopping, checkpoint selection, or downstream fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.ssl.exp12_simsiam_smoke import (
    DEFAULT_DATA,
    DEFAULT_EXP12_0,
    DEFAULT_ROOT,
    DEFAULT_SPLIT,
    EMBEDDING_VARIANCE_THRESHOLD,
    EXPECTED_PARAMETER_ELEMENTS,
    EXPECTED_PARAMETER_TENSORS,
    EXPECTED_TRAIN_IMAGES,
    FEATURE_STD_THRESHOLD,
    SimSiamSmoke,
    TwoViewTrainDataset,
    batch_metrics,
    build_augmentation,
    compare_buffers,
    compare_parameter_tensors,
    file_sha256,
    gradient_report,
    load_exp12_0_snapshot,
    negative_cosine_similarity,
    optimizer_identity_report,
    parameter_snapshot,
    set_seed,
    tensor_sha256,
    validate_train_only,
    write_json,
    write_jsonl,
)


COLLAPSE_PATIENCE = 5


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: SimSiamSmoke,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    run_config: dict,
    epoch_rows: list[dict],
) -> None:
    torch.save(
        {
            "scope": "EXP12.1_FORMAL_TRAIN_ONLY_SIMSIAM",
            "epoch": epoch,
            "model": cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "run_config": run_config,
            "epoch_metrics": epoch_rows,
            "test_accessed": False,
            "val_images_seen": 0,
            "test_images_seen": 0,
        },
        path,
    )


def write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_alignment(before_rows: list[dict], exp12_0_snapshot: Path) -> dict:
    previous_rows = load_exp12_0_snapshot(exp12_0_snapshot)
    rows = []
    for row in before_rows:
        previous = previous_rows.get(row["name"])
        rows.append(
            {
                "name": row["name"],
                "name_present": previous is not None,
                "shape_match": previous is not None and previous["shape"] == row["shape"],
                "hash_match": previous is not None and previous["sha256_raw_dtype"] == row["sha256_raw_dtype"],
            }
        )
    exact_match = (
        len(previous_rows) == EXPECTED_PARAMETER_TENSORS
        and len(rows) == EXPECTED_PARAMETER_TENSORS
        and all(row["name_present"] and row["shape_match"] and row["hash_match"] for row in rows)
    )
    return {"exact_match": exact_match, "rows": rows}


def formal_run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    weights = args.weights.resolve()
    split_manifest = args.split_manifest.resolve()
    exp12_0_snapshot = args.exp12_0_snapshot.resolve()
    smoke_summary_path = args.smoke_summary.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Exp12.1 formal output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP12_CUDA_REQUIRED")
    for required in (weights, split_manifest, exp12_0_snapshot, smoke_summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    smoke_summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    smoke_gate = (
        smoke_summary.get("status") == "PASS"
        and all(smoke_summary.get("gates", {}).values())
        and smoke_summary.get("changed_ratio", 0.0) > 0
        and smoke_summary.get("val_images_seen") == 0
        and smoke_summary.get("test_images_seen") == 0
    )
    if not smoke_gate:
        raise RuntimeError("EXP12_FORMAL_REQUIRES_SMOKE_PASS")

    train_paths, data_report = validate_train_only(data_root, split_manifest)
    output.mkdir(parents=True)
    started = time.monotonic()
    event_log = output / "run.log"

    def log(event: str, **payload: object) -> None:
        row = {"time_monotonic": time.monotonic(), "event": event, **payload}
        with event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    set_seed(args.seed)
    run_config = {
        "scope": "EXP12.1_FORMAL_TRAIN_ONLY_SIMSIAM",
        "selection_policy": "FIXED_FINAL_EPOCH_ONLY_NO_EARLY_STOP_NO_BEST_CHECKPOINT_SELECTION",
        "project_root": str(project_root),
        "data_root": str(data_root),
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
        "exp12_0_snapshot": str(exp12_0_snapshot),
        "exp12_0_snapshot_sha256": file_sha256(exp12_0_snapshot),
        "smoke_summary": str(smoke_summary_path),
        "smoke_summary_sha256": file_sha256(smoke_summary_path),
        "output": str(output),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "epochs": args.epochs,
        "seed": args.seed,
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": args.epochs,
        "scheduler_eta_min": 0.0,
        "checkpoint_interval": args.checkpoint_interval,
        "amp": False,
        "drop_last": True,
        "feature_std_threshold": FEATURE_STD_THRESHOLD,
        "embedding_variance_threshold": EMBEDDING_VARIANCE_THRESHOLD,
        "collapse_patience_epochs": COLLAPSE_PATIENCE,
        "formal_training": True,
        "downstream_training": False,
        **data_report,
    }
    write_json(output / "run_config.json", run_config)
    log("FORMAL_DATA_AND_SMOKE_GATE_PASS", train_images=len(train_paths))

    dataset = TwoViewTrainDataset(train_paths, build_augmentation(args.imgsz))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=True,
        generator=generator,
    )
    if len(loader) != EXPECTED_TRAIN_IMAGES // args.batch:
        raise RuntimeError(f"EXP12_FORMAL_STEP_COUNT_GATE: {len(loader)}")

    model = SimSiamSmoke(weights)
    requires_grad_after_unfreeze = sum(parameter.requires_grad for parameter in model.encoder.parameters())
    before_rows, before_tensors = parameter_snapshot(model.encoder)
    before_buffers = {
        name: value.detach().cpu().clone() for name, value in model.encoder.named_buffers()
    }
    write_jsonl(output / "parameter_snapshot_before.jsonl", before_rows)
    alignment = baseline_alignment(before_rows, exp12_0_snapshot)
    write_json(output / "baseline_alignment_report.json", alignment)
    unfreeze_gates = {
        "requires_grad_true_at_framework_load_is_0": model.requires_grad_true_at_framework_load == 0,
        "requires_grad_true_after_explicit_unfreeze_is_120": requires_grad_after_unfreeze == EXPECTED_PARAMETER_TENSORS,
        "parameter_tensor_count_120": len(before_rows) == EXPECTED_PARAMETER_TENSORS,
        "parameter_element_count_1365472": sum(row["numel"] for row in before_rows) == EXPECTED_PARAMETER_ELEMENTS,
        "all_before_parameters_finite": all(row["finite"] for row in before_rows),
        "exp12_0_parameter_snapshot_exact_match": alignment["exact_match"],
    }
    write_json(output / "unfreeze_gate_report.json", unfreeze_gates)
    if not all(unfreeze_gates.values()):
        write_json(output / "summary.json", {"status": "HARD_GATE", "failed_stage": "FORMAL_UNFREEZE_OR_BASELINE_ALIGNMENT"})
        return 3

    model.cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    optimizer_report = optimizer_identity_report(model, optimizer)
    optimizer_gate = (
        optimizer_report["encoder_parameters_present_exactly_once"] == EXPECTED_PARAMETER_TENSORS
        and optimizer_report["all_model_parameters_present_exactly_once"]
        and optimizer_report["no_optimizer_duplicates"]
    )
    optimizer_report["status"] = "PASS" if optimizer_gate else "HARD_GATE"
    write_json(output / "optimizer_identity_report.json", optimizer_report)
    if not optimizer_gate:
        write_json(output / "summary.json", {"status": "HARD_GATE", "failed_stage": "FORMAL_OPTIMIZER_IDENTITY"})
        return 3
    log(
        "FORMAL_UNFREEZE_AND_OPTIMIZER_PASS",
        framework_load=model.requires_grad_true_at_framework_load,
        after_unfreeze=requires_grad_after_unfreeze,
    )

    epoch_rows: list[dict] = []
    periodic_checkpoints: list[dict] = []
    collapse_consecutive_epochs = 0
    single_step_report: dict | None = None
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        losses: list[float] = []
        feature_means: list[float] = []
        feature_stds: list[float] = []
        embedding_variances: list[float] = []
        processed_samples = 0
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        for step, (view1, view2) in enumerate(loader, start=1):
            view1 = view1.cuda(non_blocking=True)
            view2 = view2.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            feature1, embedding1, prediction1 = model.branch(view1)
            feature2, embedding2, prediction2 = model.branch(view2)
            loss = 0.5 * (
                negative_cosine_similarity(prediction1, embedding2)
                + negative_cosine_similarity(prediction2, embedding1)
            )
            if not bool(torch.isfinite(loss)) or loss.grad_fn is None:
                raise RuntimeError(f"EXP12_FORMAL_NONFINITE_LOSS epoch={epoch} step={step}")
            metrics = batch_metrics(feature1, feature2, embedding1, embedding2)
            loss.backward()

            if epoch == 1 and step == 1:
                gradients = gradient_report(model.encoder)
                single_before = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.encoder.named_parameters()
                }
                optimizer.step()
                single_rows, single_update = compare_parameter_tensors(single_before, model.encoder)
                gradient_gate = (
                    gradients["gradient_present_tensor_count"] == EXPECTED_PARAMETER_TENSORS
                    and gradients["gradient_finite_tensor_count"] == EXPECTED_PARAMETER_TENSORS
                    and gradients["gradient_nonzero_tensor_count"] > 0
                    and gradients["gradient_nonzero_element_count"] > 0
                )
                update_gate = single_update["changed_ratio"] > 0
                single_step_report = {
                    "status": "PASS" if gradient_gate and update_gate else (
                        "INVALID_BY_BACKBONE_NO_UPDATE" if not update_gate else "HARD_GATE"
                    ),
                    "loss": float(loss.detach()),
                    "loss_finite": True,
                    "loss_has_grad_fn": True,
                    "gradients": gradients,
                    "update": single_update,
                    "parameters": single_rows,
                }
                write_json(output / "formal_single_step_update_report.json", single_step_report)
                if not gradient_gate or not update_gate:
                    status = single_step_report["status"]
                    write_json(
                        output / "summary.json",
                        {
                            "status": status,
                            "failed_stage": "FORMAL_SINGLE_STEP_UPDATE_GATE",
                            "val_images_seen": 0,
                            "test_images_seen": 0,
                            "test_accessed": False,
                        },
                    )
                    return 12 if status == "INVALID_BY_BACKBONE_NO_UPDATE" else 3
                log(
                    "FORMAL_SINGLE_STEP_UPDATE_PASS",
                    changed_parameters=single_update["changed_parameter_count"],
                    changed_ratio=single_update["changed_ratio"],
                )
            else:
                optimizer.step()

            values = [
                float(loss.detach()),
                metrics["feature_mean"],
                metrics["feature_std"],
                metrics["embedding_variance"],
            ]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"EXP12_FORMAL_NONFINITE_METRIC epoch={epoch} step={step}")
            losses.append(values[0])
            feature_means.append(values[1])
            feature_stds.append(values[2])
            embedding_variances.append(values[3])
            processed_samples += int(view1.shape[0])

        epoch_row = {
            "epoch": epoch,
            "steps": len(losses),
            "processed_samples": processed_samples,
            "loss": float(np.mean(losses)),
            "feature_mean": float(np.mean(feature_means)),
            "feature_std": float(np.mean(feature_stds)),
            "embedding_variance": float(np.mean(embedding_variances)),
            "learning_rate": epoch_lr,
            "all_metrics_finite": True,
            "epoch_wall_seconds": time.monotonic() - epoch_started,
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        low_variance = (
            epoch_row["feature_std"] < FEATURE_STD_THRESHOLD
            or epoch_row["embedding_variance"] < EMBEDDING_VARIANCE_THRESHOLD
        )
        collapse_consecutive_epochs = collapse_consecutive_epochs + 1 if low_variance else 0
        epoch_row["collapse_threshold_breached"] = low_variance
        epoch_row["collapse_consecutive_epochs"] = collapse_consecutive_epochs
        epoch_rows.append(epoch_row)
        write_metrics(output / "epoch_metrics.csv", epoch_rows)
        scheduler.step()
        log("EPOCH_COMPLETE", **epoch_row)

        if collapse_consecutive_epochs >= COLLAPSE_PATIENCE:
            stopped_checkpoint = output / f"stopped_collapse_epoch_{epoch:03d}.pt"
            save_checkpoint(stopped_checkpoint, model, optimizer, scheduler, epoch, run_config, epoch_rows)
            write_json(
                output / "summary.json",
                {
                    "status": "COLLAPSE_SUSPECTED",
                    "failed_stage": "FORMAL_COLLAPSE_GATE",
                    "stopped_epoch": epoch,
                    "val_images_seen": 0,
                    "test_images_seen": 0,
                    "test_accessed": False,
                },
            )
            return 4

        if epoch % args.checkpoint_interval == 0 and epoch < args.epochs:
            checkpoint = output / f"epoch_{epoch:03d}.pt"
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, run_config, epoch_rows)
            periodic_checkpoints.append(
                {"epoch": epoch, "path": str(checkpoint), "sha256": file_sha256(checkpoint)}
            )
            write_json(output / "periodic_checkpoints.json", periodic_checkpoints)
            log("PERIODIC_CHECKPOINT_SAVED", epoch=epoch, sha256=periodic_checkpoints[-1]["sha256"])

    final_checkpoint = output / "last.pt"
    save_checkpoint(
        final_checkpoint, model, optimizer, scheduler, args.epochs, run_config, epoch_rows
    )
    after_rows, _ = parameter_snapshot(model.encoder)
    write_jsonl(output / "parameter_snapshot_after.jsonl", after_rows)
    update_rows, update_summary = compare_parameter_tensors(before_tensors, model.encoder)
    update_summary.update(compare_buffers(before_buffers, model.encoder))
    update_summary["status"] = (
        "PASS" if update_summary["changed_ratio"] > 0 and update_summary["all_after_finite"]
        else "INVALID_BY_BACKBONE_NO_UPDATE"
    )
    update_summary["parameters"] = update_rows
    write_json(output / "parameter_update_report.json", update_summary)
    if update_summary["status"] != "PASS":
        write_json(
            output / "summary.json",
            {"status": update_summary["status"], "failed_stage": "FORMAL_FINAL_PARAMETER_UPDATE"},
        )
        return 12

    backbone_export = output / "adapted_backbone_final.pt"
    backbone_state = cpu_state_dict(model.encoder)
    torch.save(
        {
            "scope": "EXP12.1_FORMAL_FINAL_BACKBONE",
            "selection_policy": "FIXED_FINAL_EPOCH_ONLY",
            "backbone": backbone_state,
            "encoder_layers": [0, 10],
            "source_weights_sha256": file_sha256(weights),
            "ssl_train_images": EXPECTED_TRAIN_IMAGES,
            "epochs": args.epochs,
            "seed": args.seed,
            "val_images_seen": 0,
            "test_images_seen": 0,
            "test_accessed": False,
        },
        backbone_export,
    )

    in_memory_hashes = {
        name: tensor_sha256(parameter) for name, parameter in model.encoder.named_parameters()
    }
    checkpoint_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    reloaded_model = SimSiamSmoke(weights).cpu()
    reloaded_model.load_state_dict(checkpoint_payload["model"], strict=True)
    checkpoint_hashes = {
        name: tensor_sha256(parameter)
        for name, parameter in reloaded_model.encoder.named_parameters()
    }
    export_payload = torch.load(backbone_export, map_location="cpu", weights_only=False)
    export_hashes = {
        name: tensor_sha256(value)
        for name, value in export_payload["backbone"].items()
        if name in in_memory_hashes
    }
    reload_report = {
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "backbone_export": str(backbone_export),
        "backbone_export_sha256": file_sha256(backbone_export),
        "in_memory_parameter_count": len(in_memory_hashes),
        "checkpoint_reload_match_count": sum(
            checkpoint_hashes.get(name) == digest for name, digest in in_memory_hashes.items()
        ),
        "export_reload_match_count": sum(
            export_hashes.get(name) == digest for name, digest in in_memory_hashes.items()
        ),
    }
    reload_report["status"] = "PASS" if (
        reload_report["checkpoint_reload_match_count"] == EXPECTED_PARAMETER_TENSORS
        and reload_report["export_reload_match_count"] == EXPECTED_PARAMETER_TENSORS
    ) else "HARD_GATE"
    write_json(output / "checkpoint_reload_report.json", reload_report)

    gates = {
        "smoke_gate_pass": smoke_gate,
        "data_train_only": data_report["train_manifest_exact_set_668"] and data_report["train_symlink_targets_frozen_read_only_source"],
        "explicit_unfreeze_and_baseline_alignment": all(unfreeze_gates.values()),
        "optimizer_identity": optimizer_gate,
        "formal_single_step_gradient_and_update": single_step_report is not None and single_step_report["status"] == "PASS",
        "fixed_100_epochs_complete": len(epoch_rows) == args.epochs == 100,
        "all_epoch_metrics_finite": all(row["all_metrics_finite"] for row in epoch_rows),
        "collapse_patience_not_triggered": max(
            row["collapse_consecutive_epochs"] for row in epoch_rows
        ) < COLLAPSE_PATIENCE,
        "formal_parameter_update": update_summary["status"] == "PASS",
        "checkpoint_and_export_reload": reload_report["status"] == "PASS",
        "val_images_seen_0": data_report["val_images_seen"] == 0,
        "test_images_seen_0": data_report["test_images_seen"] == 0,
    }
    status = "PASS" if all(gates.values()) else "HARD_GATE"
    summary = {
        "status": status,
        "scope": "EXP12.1_FORMAL_SSL_AND_EXP12.2_UPDATE_AUDIT",
        "gates": gates,
        "formal_training_epochs": len(epoch_rows),
        "downstream_training_performed": False,
        "test_accessed": False,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "ssl_train_images": EXPECTED_TRAIN_IMAGES,
        "processed_samples_per_epoch": epoch_rows[-1]["processed_samples"],
        "first_epoch_metrics": epoch_rows[0],
        "final_epoch_metrics": epoch_rows[-1],
        "minimum_feature_std": min(row["feature_std"] for row in epoch_rows),
        "minimum_embedding_variance": min(row["embedding_variance"] for row in epoch_rows),
        "minimum_train_loss_diagnostic_only_not_selected": min(row["loss"] for row in epoch_rows),
        "changed_parameter_count": update_summary["changed_parameter_count"],
        "total_parameter_count": update_summary["total_parameter_count"],
        "changed_ratio": update_summary["changed_ratio"],
        "changed_parameter_element_count": update_summary["changed_parameter_element_count"],
        "total_parameter_element_count": update_summary["total_parameter_element_count"],
        "periodic_checkpoint_count": len(periodic_checkpoints),
        "final_checkpoint_sha256": reload_report["final_checkpoint_sha256"],
        "backbone_export_sha256": reload_report["backbone_export_sha256"],
        "wall_seconds": time.monotonic() - started,
        "next_gate": "WAITING_SEPARATE_AUTHORIZATION_BEFORE_EXP12.3_DOWNSTREAM",
    }
    write_json(output / "summary.json", summary)
    log("EXP12_FORMAL_AND_AUDIT_COMPLETE", status=status, changed_ratio=update_summary["changed_ratio"])
    return 0 if status == "PASS" else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--exp12-0-snapshot", type=Path)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.00625)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.weights = args.weights or args.project_root / "weights/yolo11n-seg.pt"
    args.split_manifest = args.split_manifest or args.project_root / DEFAULT_SPLIT
    args.exp12_0_snapshot = args.exp12_0_snapshot or args.project_root / DEFAULT_EXP12_0
    args.smoke_summary = args.smoke_summary or (
        args.project_root / "results/exp12_simsiam_basic/smoke_20260822T063541Z/summary.json"
    )
    if args.epochs != 100:
        raise ValueError("EXP12_FORMAL_EPOCH_BUDGET_MUST_BE_100")
    if args.batch != 32:
        raise ValueError("EXP12_FORMAL_BATCH_MUST_BE_32")
    return args


def main() -> int:
    args = parse_args()
    try:
        return formal_run(args)
    except Exception as error:
        output = args.output.resolve()
        if output.exists() and output.is_dir():
            write_json(
                output / "fatal_error.json",
                {
                    "status": "HARD_GATE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "val_images_seen": 0,
                    "test_images_seen": 0,
                    "test_accessed": False,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
