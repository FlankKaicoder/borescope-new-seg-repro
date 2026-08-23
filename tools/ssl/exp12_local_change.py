#!/usr/bin/env python3
"""Exp12.4 TRAIN-only multi-scale local-change SSL with update auditing.

The proxy task receives an image and a synthetically changed, spatially
aligned copy. A shared YOLO11n-seg backbone exposes P3/P4/P5 features; a small
head predicts the synthetic change mask from absolute feature differences.
VAL and TEST are never opened. Smoke must PASS before the fixed 30-epoch run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from tools.ssl.exp12_simsiam_smoke import (
    DEFAULT_DATA,
    DEFAULT_EXP12_0,
    DEFAULT_ROOT,
    DEFAULT_SPLIT,
    EXPECTED_PARAMETER_ELEMENTS,
    EXPECTED_PARAMETER_TENSORS,
    EXPECTED_TRAIN_IMAGES,
    compare_buffers,
    compare_parameter_tensors,
    file_sha256,
    gradient_report,
    load_exp12_0_snapshot,
    parameter_snapshot,
    set_seed,
    tensor_sha256,
    validate_train_only,
    write_json,
    write_jsonl,
)


BACKBONE_END = 10
TAP_LAYERS = (4, 6, 10)
FEATURE_STD_THRESHOLD = 1e-4
LOGIT_STD_THRESHOLD = 1e-5
COLLAPSE_PATIENCE = 5


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


class LocalChangeDataset(Dataset):
    """Create aligned image pairs and masks from frozen TRAIN images only."""

    def __init__(self, paths: list[Path], imgsz: int) -> None:
        self.paths = paths
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def ellipse_mask(height: int, width: int) -> torch.Tensor:
        yy = (torch.arange(height, dtype=torch.float32) + 0.5 - height / 2) / max(height / 2, 1)
        xx = (torch.arange(width, dtype=torch.float32) + 0.5 - width / 2) / max(width / 2, 1)
        return yy[:, None].square() + xx[None, :].square() <= 1.0

    def __getitem__(self, index: int):
        path = self.paths[index]
        if path.parent.name != "train" or path.parent.parent.name != "images":
            raise RuntimeError(f"EXP12_RUNTIME_TRAIN_ONLY_GATE: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
            original = TF.pil_to_tensor(image).float().div_(255.0)
        if random.random() < 0.5:
            original = torch.flip(original, dims=(2,))

        changed = original.clone()
        mask = torch.zeros((1, self.imgsz, self.imgsz), dtype=torch.float32)
        operation_counts = torch.zeros(3, dtype=torch.int64)
        for _ in range(random.randint(1, 3)):
            height = random.randint(max(12, self.imgsz // 42), max(20, self.imgsz // 13))
            width = random.randint(max(12, self.imgsz // 42), max(20, self.imgsz // 13))
            y0 = random.randint(0, self.imgsz - height)
            x0 = random.randint(0, self.imgsz - width)
            region = self.ellipse_mask(height, width)
            operation = random.randrange(3)
            operation_counts[operation] += 1
            target = changed[:, y0:y0 + height, x0:x0 + width]

            if operation == 0:  # same-image texture transplant
                sy = random.randint(0, self.imgsz - height)
                sx = random.randint(0, self.imgsz - width)
                replacement = original[:, sy:sy + height, sx:sx + width].clone()
            elif operation == 1:  # localized photometric/contrast change
                factor = random.uniform(0.55, 1.45)
                offset = random.choice((-1.0, 1.0)) * random.uniform(0.08, 0.28)
                replacement = (target * factor + offset).clamp(0.0, 1.0)
            else:  # localized blur
                kernel = random.choice((5, 7, 9, 11))
                replacement = TF.gaussian_blur(target, [kernel, kernel], [0.8, 2.2])

            region3 = region.unsqueeze(0).expand_as(target)
            changed[:, y0:y0 + height, x0:x0 + width] = torch.where(region3, replacement, target)
            mask[0, y0:y0 + height, x0:x0 + width] = torch.maximum(
                mask[0, y0:y0 + height, x0:x0 + width], region.float()
            )

        if not bool(mask.any()):
            raise RuntimeError("EXP12_LOCAL_CHANGE_EMPTY_MASK")
        return original, changed, mask, operation_counts


class MultiScaleLocalChange(nn.Module):
    def __init__(self, weights: Path, probe_size: int = 64) -> None:
        super().__init__()
        source = YOLO(str(weights))
        source_layers = source.model.model[: BACKBONE_END + 1]
        if len(source_layers) != BACKBONE_END + 1:
            raise RuntimeError("EXP12_BACKBONE_LAYER_COUNT_GATE")
        self.encoder = nn.Sequential(*[copy.deepcopy(layer) for layer in source_layers])
        self.requires_grad_true_at_framework_load = sum(
            parameter.requires_grad for parameter in self.encoder.parameters()
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            channels = [feature.shape[1] for feature in self.encode(torch.zeros(1, 3, probe_size, probe_size))]
        self.encoder.train(was_training)
        self.tap_channels = channels
        projection_channels = 64
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, projection_channels, 1, bias=False),
                    nn.GroupNorm(8, projection_channels),
                    nn.SiLU(inplace=True),
                )
                for channel in channels
            ]
        )
        self.change_head = nn.Sequential(
            nn.Conv2d(projection_channels * len(channels), 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )
        del source

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        value = image
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in TAP_LAYERS:
                features.append(value)
        if len(features) != len(TAP_LAYERS):
            raise RuntimeError("EXP12_MULTISCALE_TAP_GATE")
        return features

    def forward(self, original: torch.Tensor, changed: torch.Tensor):
        original_features = self.encode(original)
        changed_features = self.encode(changed)
        target_size = original_features[0].shape[-2:]
        projected = []
        for projection, left, right in zip(
            self.projections, original_features, changed_features, strict=True
        ):
            value = projection(torch.abs(left - right))
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            projected.append(value)
        logits = self.change_head(torch.cat(projected, dim=1))
        return logits, original_features, changed_features


def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = logits.sigmoid()
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha = torch.where(target > 0.5, 0.75, 0.25)
    focal = (alpha * (1.0 - pt).square() * bce).mean()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) /
                  (probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)).mean()
    return focal + dice, {"focal_loss": float(focal.detach()), "dice_loss": float(dice.detach())}


def batch_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    left: list[torch.Tensor],
    right: list[torch.Tensor],
) -> dict:
    probability = logits.detach().sigmoid().float()
    prediction = probability >= 0.5
    truth = target.detach() >= 0.5
    intersection = (prediction & truth).sum().item()
    union = (prediction | truth).sum().item()
    true_positive = intersection
    predicted_positive = prediction.sum().item()
    actual_positive = truth.sum().item()
    tap_stds = []
    tap_means = []
    for a, b in zip(left, right, strict=True):
        feature = torch.cat((a.detach(), b.detach()), dim=0).float()
        tap_means.append(float(feature.mean()))
        tap_stds.append(float(feature.std(dim=(0, 2, 3), unbiased=False).mean()))
    return {
        "feature_mean": float(np.mean(tap_means)),
        "feature_std": float(np.mean(tap_stds)),
        "p3_feature_std": tap_stds[0],
        "p4_feature_std": tap_stds[1],
        "p5_feature_std": tap_stds[2],
        "logit_std": float(logits.detach().float().std(unbiased=False)),
        "probability_mean": float(probability.mean()),
        "prediction_iou": intersection / max(union, 1),
        "prediction_precision": true_positive / max(predicted_positive, 1),
        "prediction_recall": true_positive / max(actual_positive, 1),
        "target_positive_ratio": float(target.detach().float().mean()),
    }


def optimizer_identity_report(model: MultiScaleLocalChange, optimizer: torch.optim.Optimizer) -> dict:
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    counts = Counter(id(parameter) for parameter in optimizer_parameters)
    encoder = list(model.encoder.parameters())
    all_parameters = list(model.parameters())
    return {
        "optimizer_parameter_object_count": len(optimizer_parameters),
        "optimizer_unique_parameter_object_count": len(counts),
        "optimizer_duplicate_object_count": len(optimizer_parameters) - len(counts),
        "model_parameter_object_count": len(all_parameters),
        "encoder_parameter_tensor_count": len(encoder),
        "encoder_parameter_element_count": sum(parameter.numel() for parameter in encoder),
        "encoder_parameters_present_exactly_once": sum(counts[id(parameter)] == 1 for parameter in encoder),
        "all_model_parameters_present_exactly_once": all(counts[id(parameter)] == 1 for parameter in all_parameters),
        "no_optimizer_duplicates": len(optimizer_parameters) == len(counts),
    }


def baseline_alignment(before_rows: list[dict], exp12_0_snapshot: Path) -> dict:
    previous_rows = load_exp12_0_snapshot(exp12_0_snapshot)
    rows = []
    for row in before_rows:
        previous = previous_rows.get(row["name"])
        rows.append({
            "name": row["name"],
            "name_present": previous is not None,
            "shape_match": previous is not None and previous["shape"] == row["shape"],
            "hash_match": previous is not None and previous["sha256_raw_dtype"] == row["sha256_raw_dtype"],
        })
    exact = (
        len(previous_rows) == EXPECTED_PARAMETER_TENSORS
        and len(rows) == EXPECTED_PARAMETER_TENSORS
        and all(row["name_present"] and row["shape_match"] and row["hash_match"] for row in rows)
    )
    return {"exact_match": exact, "rows": rows}


def write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: MultiScaleLocalChange, optimizer, scheduler,
                    epoch: int, run_config: dict, rows: list[dict]) -> None:
    torch.save({
        "scope": "EXP12.4_TRAIN_ONLY_MULTISCALE_LOCAL_CHANGE",
        "epoch": epoch,
        "model": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "run_config": run_config,
        "epoch_metrics": rows,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }, path)


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    weights = args.weights.resolve()
    split_manifest = args.split_manifest.resolve()
    exp12_0_snapshot = args.exp12_0_snapshot.resolve()
    output = args.output.resolve()
    epochs = 1 if args.phase == "smoke" else 30
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Exp12.4 output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP12_CUDA_REQUIRED")
    for required in (weights, split_manifest, exp12_0_snapshot):
        if not required.is_file():
            raise FileNotFoundError(required)
    smoke_gate = None
    if args.phase == "formal":
        if args.smoke_summary is None or not args.smoke_summary.is_file():
            raise RuntimeError("EXP12_4_FORMAL_REQUIRES_SMOKE_SUMMARY")
        smoke_gate = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
        if (
            smoke_gate.get("status") != "PASS"
            or smoke_gate.get("changed_ratio", 0.0) <= 0
            or smoke_gate.get("val_images_seen") != 0
            or smoke_gate.get("test_images_seen") != 0
        ):
            raise RuntimeError("EXP12_4_FORMAL_REQUIRES_SMOKE_PASS")

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
        "scope": "EXP12.4_TRAIN_ONLY_MULTISCALE_LOCAL_CHANGE",
        "phase": args.phase,
        "selection_policy": "FIXED_FINAL_EPOCH_ONLY_NO_VAL_NO_EARLY_STOP",
        "project_root": str(project_root),
        "data_root": str(data_root),
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
        "exp12_0_snapshot": str(exp12_0_snapshot),
        "exp12_0_snapshot_sha256": file_sha256(exp12_0_snapshot),
        "smoke_summary": str(args.smoke_summary.resolve()) if args.smoke_summary else None,
        "smoke_summary_sha256": file_sha256(args.smoke_summary) if args.smoke_summary else None,
        "output": str(output),
        "proxy_task": "aligned synthetic local change mask localization",
        "change_types": ["same_image_texture_transplant", "localized_photometric", "localized_blur"],
        "tap_layers": list(TAP_LAYERS),
        "loss": "focal_bce_plus_soft_dice",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "epochs": epochs,
        "seed": args.seed,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": epochs,
        "amp": False,
        "drop_last": True,
        "formal_training": args.phase == "formal",
        "downstream_training": False,
        **data_report,
    }
    write_json(output / "run_config.json", run_config)
    log("DATA_GATE_PASS", phase=args.phase, train_images=len(train_paths))

    dataset = LocalChangeDataset(train_paths, args.imgsz)
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
    expected_steps = EXPECTED_TRAIN_IMAGES // args.batch
    if len(loader) != expected_steps:
        raise RuntimeError(f"EXP12_4_STEP_COUNT_GATE: {len(loader)} != {expected_steps}")

    model = MultiScaleLocalChange(weights)
    before_rows, before_tensors = parameter_snapshot(model.encoder)
    before_buffers = {name: value.detach().cpu().clone() for name, value in model.encoder.named_buffers()}
    write_jsonl(output / "parameter_snapshot_before.jsonl", before_rows)
    alignment = baseline_alignment(before_rows, exp12_0_snapshot)
    write_json(output / "baseline_alignment_report.json", alignment)
    trainable = sum(parameter.requires_grad for parameter in model.encoder.parameters())
    structure_gates = {
        "requires_grad_true_at_framework_load_is_0": model.requires_grad_true_at_framework_load == 0,
        "requires_grad_true_after_explicit_unfreeze_is_120": trainable == EXPECTED_PARAMETER_TENSORS,
        "parameter_tensor_count_120": len(before_rows) == EXPECTED_PARAMETER_TENSORS,
        "parameter_element_count_1365472": sum(row["numel"] for row in before_rows) == EXPECTED_PARAMETER_ELEMENTS,
        "all_before_parameters_finite": all(row["finite"] for row in before_rows),
        "exp12_0_parameter_snapshot_exact_match": alignment["exact_match"],
        "tap_layer_count_3": len(model.tap_channels) == 3,
    }
    write_json(output / "structure_gate_report.json", {
        "status": "PASS" if all(structure_gates.values()) else "HARD_GATE",
        "gates": structure_gates,
        "tap_layers": list(TAP_LAYERS),
        "tap_channels": model.tap_channels,
    })
    if not all(structure_gates.values()):
        write_json(output / "summary.json", {"status": "HARD_GATE", "failed_stage": "STRUCTURE"})
        return 3

    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    identity = optimizer_identity_report(model, optimizer)
    identity_gate = (
        identity["encoder_parameters_present_exactly_once"] == EXPECTED_PARAMETER_TENSORS
        and identity["all_model_parameters_present_exactly_once"]
        and identity["no_optimizer_duplicates"]
    )
    identity["status"] = "PASS" if identity_gate else "HARD_GATE"
    write_json(output / "optimizer_identity_report.json", identity)
    if not identity_gate:
        write_json(output / "summary.json", {"status": "HARD_GATE", "failed_stage": "OPTIMIZER_IDENTITY"})
        return 3

    rows: list[dict] = []
    single_step_report = None
    collapse_consecutive = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        accum: dict[str, list[float]] = {}
        processed = 0
        operation_counts = np.zeros(3, dtype=np.int64)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        for step, (original, changed, mask, operations) in enumerate(loader, start=1):
            operation_counts += operations.sum(dim=0).numpy()
            original = original.cuda(non_blocking=True)
            changed = changed.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, left, right = model(original, changed)
            target = F.adaptive_max_pool2d(mask, logits.shape[-2:])
            loss, components = focal_dice_loss(logits, target)
            metrics = batch_metrics(logits, target, left, right)
            if not bool(torch.isfinite(loss)) or loss.grad_fn is None:
                raise RuntimeError(f"EXP12_4_NONFINITE_LOSS epoch={epoch} step={step}")
            loss.backward()

            if epoch == 1 and step == 1:
                gradients = gradient_report(model.encoder)
                single_before = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.encoder.named_parameters()
                }
                optimizer.step()
                parameter_rows, update = compare_parameter_tensors(single_before, model.encoder)
                gradient_gate = (
                    gradients["gradient_present_tensor_count"] == EXPECTED_PARAMETER_TENSORS
                    and gradients["gradient_finite_tensor_count"] == EXPECTED_PARAMETER_TENSORS
                    and gradients["gradient_nonzero_tensor_count"] > 0
                    and gradients["gradient_nonzero_element_count"] > 0
                )
                update_gate = update["changed_ratio"] > 0
                single_step_report = {
                    "status": "PASS" if gradient_gate and update_gate else (
                        "INVALID_BY_BACKBONE_NO_UPDATE" if not update_gate else "HARD_GATE"
                    ),
                    "loss": float(loss.detach()),
                    "gradients": gradients,
                    "update": update,
                    "parameters": parameter_rows,
                }
                write_json(output / "single_step_update_report.json", single_step_report)
                if single_step_report["status"] != "PASS":
                    write_json(output / "summary.json", {
                        "status": single_step_report["status"],
                        "failed_stage": "SINGLE_STEP_BACKBONE_UPDATE",
                        "val_images_seen": 0,
                        "test_images_seen": 0,
                        "test_accessed": False,
                    })
                    return 12 if not update_gate else 3
            else:
                optimizer.step()

            values = {"loss": float(loss.detach()), **components, **metrics}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"EXP12_4_NONFINITE_METRIC epoch={epoch} step={step}")
            for key, value in values.items():
                accum.setdefault(key, []).append(value)
            processed += int(original.shape[0])

        row = {
            "epoch": epoch,
            "steps": len(loader),
            "processed_samples": processed,
            **{key: float(np.mean(value)) for key, value in accum.items()},
            "learning_rate": epoch_lr,
            "texture_transplant_count": int(operation_counts[0]),
            "photometric_count": int(operation_counts[1]),
            "blur_count": int(operation_counts[2]),
            "all_metrics_finite": True,
            "epoch_wall_seconds": time.monotonic() - epoch_started,
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        low_variance = row["feature_std"] < FEATURE_STD_THRESHOLD or row["logit_std"] < LOGIT_STD_THRESHOLD
        collapse_consecutive = collapse_consecutive + 1 if low_variance else 0
        row["collapse_threshold_breached"] = low_variance
        row["collapse_consecutive_epochs"] = collapse_consecutive
        rows.append(row)
        write_metrics(output / "epoch_metrics.csv", rows)
        scheduler.step()
        log("EPOCH_COMPLETE", **row)
        if collapse_consecutive >= COLLAPSE_PATIENCE:
            save_checkpoint(output / f"stopped_collapse_epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, run_config, rows)
            write_json(output / "summary.json", {
                "status": "COLLAPSE_SUSPECTED",
                "failed_stage": "COLLAPSE_GATE",
                "stopped_epoch": epoch,
                "val_images_seen": 0,
                "test_images_seen": 0,
                "test_accessed": False,
            })
            return 4

    final_checkpoint = output / "last.pt"
    save_checkpoint(final_checkpoint, model, optimizer, scheduler, epochs, run_config, rows)
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
        write_json(output / "summary.json", {
            "status": "INVALID_BY_BACKBONE_NO_UPDATE",
            "failed_stage": "FINAL_BACKBONE_UPDATE",
            "val_images_seen": 0,
            "test_images_seen": 0,
            "test_accessed": False,
        })
        return 12

    backbone_export = output / "adapted_backbone_final.pt"
    torch.save({
        "scope": "EXP12.4_FINAL_MULTISCALE_LOCAL_CHANGE_BACKBONE",
        "selection_policy": "FIXED_FINAL_EPOCH_ONLY",
        "backbone": cpu_state_dict(model.encoder),
        "encoder_layers": [0, 10],
        "tap_layers": list(TAP_LAYERS),
        "source_weights_sha256": file_sha256(weights),
        "ssl_train_images": EXPECTED_TRAIN_IMAGES,
        "epochs": epochs,
        "seed": args.seed,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }, backbone_export)

    memory_hashes = {name: tensor_sha256(value) for name, value in model.encoder.named_parameters()}
    checkpoint_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    reloaded = MultiScaleLocalChange(weights).cpu()
    reloaded.load_state_dict(checkpoint_payload["model"], strict=True)
    checkpoint_hashes = {name: tensor_sha256(value) for name, value in reloaded.encoder.named_parameters()}
    export_payload = torch.load(backbone_export, map_location="cpu", weights_only=False)
    export_hashes = {
        name: tensor_sha256(value)
        for name, value in export_payload["backbone"].items()
        if name in memory_hashes
    }
    reload_report = {
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "backbone_export": str(backbone_export),
        "backbone_export_sha256": file_sha256(backbone_export),
        "checkpoint_reload_match_count": sum(checkpoint_hashes.get(name) == digest for name, digest in memory_hashes.items()),
        "export_reload_match_count": sum(export_hashes.get(name) == digest for name, digest in memory_hashes.items()),
    }
    reload_report["status"] = "PASS" if (
        reload_report["checkpoint_reload_match_count"] == EXPECTED_PARAMETER_TENSORS
        and reload_report["export_reload_match_count"] == EXPECTED_PARAMETER_TENSORS
    ) else "HARD_GATE"
    write_json(output / "checkpoint_reload_report.json", reload_report)

    gates = {
        "formal_requires_smoke_pass": args.phase == "smoke" or smoke_gate is not None,
        "data_train_only": data_report["train_manifest_exact_set_668"] and data_report["train_symlink_targets_frozen_read_only_source"],
        "structure_and_explicit_unfreeze": all(structure_gates.values()),
        "optimizer_identity": identity_gate,
        "single_step_gradient_and_backbone_update": single_step_report is not None and single_step_report["status"] == "PASS",
        "fixed_epoch_budget_complete": len(rows) == epochs,
        "all_epoch_metrics_finite": all(row["all_metrics_finite"] for row in rows),
        "collapse_patience_not_triggered": max(row["collapse_consecutive_epochs"] for row in rows) < COLLAPSE_PATIENCE,
        "final_parameter_update": update_summary["status"] == "PASS",
        "checkpoint_and_export_reload": reload_report["status"] == "PASS",
        "val_images_seen_0": data_report["val_images_seen"] == 0,
        "test_images_seen_0": data_report["test_images_seen"] == 0,
    }
    status = "PASS" if all(gates.values()) else "HARD_GATE"
    summary = {
        "status": status,
        "scope": "EXP12.4_MULTISCALE_LOCAL_CHANGE_SMOKE" if args.phase == "smoke" else "EXP12.4_MULTISCALE_LOCAL_CHANGE_FORMAL",
        "phase": args.phase,
        "gates": gates,
        "epochs_completed": len(rows),
        "ssl_train_images": EXPECTED_TRAIN_IMAGES,
        "processed_samples_per_epoch": rows[-1]["processed_samples"],
        "first_epoch_metrics": rows[0],
        "final_epoch_metrics": rows[-1],
        "changed_parameter_count": update_summary["changed_parameter_count"],
        "total_parameter_count": update_summary["total_parameter_count"],
        "changed_ratio": update_summary["changed_ratio"],
        "changed_parameter_element_count": update_summary["changed_parameter_element_count"],
        "total_parameter_element_count": update_summary["total_parameter_element_count"],
        "minimum_feature_std": min(row["feature_std"] for row in rows),
        "minimum_logit_std": min(row["logit_std"] for row in rows),
        "final_checkpoint_sha256": reload_report["final_checkpoint_sha256"],
        "backbone_export_sha256": reload_report["backbone_export_sha256"],
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
        "wall_seconds": time.monotonic() - started,
        "next_gate": "EXP12.4_FORMAL" if args.phase == "smoke" else "EXP12.5_DOWNSTREAM",
    }
    write_json(output / "summary.json", summary)
    log("EXP12_4_COMPLETE", status=status, phase=args.phase, changed_ratio=summary["changed_ratio"])
    return 0 if status == "PASS" else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--exp12-0-snapshot", type=Path)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.weights = args.weights or args.project_root / "weights/yolo11n-seg.pt"
    args.split_manifest = args.split_manifest or args.project_root / DEFAULT_SPLIT
    args.exp12_0_snapshot = args.exp12_0_snapshot or args.project_root / DEFAULT_EXP12_0
    if args.batch != 16:
        raise ValueError("EXP12_4_BATCH_MUST_BE_16")
    return args


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as error:
        output = args.output.resolve()
        if output.exists() and output.is_dir():
            write_json(output / "fatal_error.json", {
                "status": "HARD_GATE",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "val_images_seen": 0,
                "test_images_seen": 0,
                "test_accessed": False,
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
