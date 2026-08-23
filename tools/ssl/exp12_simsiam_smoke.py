#!/usr/bin/env python3
"""Exp12.1-S TRAIN-only SimSiam update and one-epoch smoke audit.

This is intentionally a smoke-only program. It proves that the YOLO11n-seg
encoder parameters are trainable, receive gradients, change after one real
optimizer step, and survive checkpoint/export reload. It never reads VAL or
TEST and it does not implement downstream fine-tuning.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from torchvision import transforms
from ultralytics import YOLO


DEFAULT_ROOT = Path("/root/autodl-tmp/borescope-new-seg-repro")
DEFAULT_DATA = Path("/root/autodl-tmp/borescope-new-seg-data/v1")
DEFAULT_SPLIT = Path("results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv")
DEFAULT_EXP12_0 = Path("results/ssl/exp12_0_encoder_analysis/parameter_snapshot_before.jsonl")
READ_ONLY_SOURCE = Path("/root/autodl-tmp/损伤训练数据集")
BACKBONE_END = 10
EXPECTED_TRAIN_IMAGES = 668
EXPECTED_PARAMETER_TENSORS = 120
EXPECTED_PARAMETER_ELEMENTS = 1_365_472
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FEATURE_STD_THRESHOLD = 1e-4
EMBEDDING_VARIANCE_THRESHOLD = 1e-6


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_train_only(data_root: Path, split_manifest: Path) -> tuple[list[Path], dict]:
    train_root = (data_root / "images/train").resolve()
    read_only_source = READ_ONLY_SOURCE.resolve()
    if not train_root.is_dir() or train_root.name != "train" or train_root.parent.name != "images":
        raise RuntimeError(f"EXP12_TRAIN_ONLY_PATH_GATE: {train_root}")
    logical_paths = sorted(
        path for path in train_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    resolved_targets = [path.resolve() for path in logical_paths]
    symlink_gate = (
        len(logical_paths) == EXPECTED_TRAIN_IMAGES
        and all(path.is_symlink() for path in logical_paths)
        and all(target.is_file() and read_only_source in target.parents for target in resolved_targets)
        and all(path.name == target.name for path, target in zip(logical_paths, resolved_targets, strict=True))
    )
    if not symlink_gate:
        raise RuntimeError("EXP12_TRAIN_SYMLINK_TARGET_GATE")
    with split_manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_train = sorted(
        train_root / f"{row['stem']}{row['image_suffix']}"
        for row in manifest_rows if row["split"] == "train"
    )
    exact_set_gate = (
        len(manifest_train) == EXPECTED_TRAIN_IMAGES
        and len(set(logical_paths)) == EXPECTED_TRAIN_IMAGES
        and set(logical_paths) == set(manifest_train)
    )
    if not exact_set_gate:
        raise RuntimeError("EXP12_TRAIN_MANIFEST_EXACT_SET_GATE")
    report = {
        "ssl_input_root": str(train_root),
        "ssl_train_image_count": len(logical_paths),
        "ssl_train_symlink_count": sum(path.is_symlink() for path in logical_paths),
        "read_only_source_root": str(read_only_source),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
        "train_manifest_exact_set_668": exact_set_gate,
        "train_symlink_targets_frozen_read_only_source": symlink_gate,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    return logical_paths, report


class TwoViewTrainDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        if path.parent.name != "train" or path.parent.parent.name != "images":
            raise RuntimeError(f"EXP12_RUNTIME_TRAIN_ONLY_GATE: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return self.transform(rgb), self.transform(rgb)


def build_augmentation(imgsz: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imgsz, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.ToTensor(),
        ]
    )


class SimSiamSmoke(nn.Module):
    def __init__(self, weights: Path) -> None:
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
        self.projector = nn.Sequential(
            nn.Linear(256, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
        )
        del source

    def branch(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.encoder(image)
        feature = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        embedding = self.projector(feature)
        prediction = self.predictor(embedding)
        return feature, embedding, prediction


def negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -F.cosine_similarity(prediction, target.detach(), dim=1).mean()


def parameter_snapshot(module: nn.Module) -> tuple[list[dict], dict[str, torch.Tensor]]:
    rows: list[dict] = []
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        raw = parameter.detach().cpu().contiguous().clone()
        numeric = raw.to(torch.float64)
        tensors[name] = raw
        rows.append(
            {
                "name": name,
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                "numel": raw.numel(),
                "requires_grad": bool(parameter.requires_grad),
                "sha256_raw_dtype": tensor_sha256(raw),
                "mean_float64": float(numeric.mean()) if numeric.numel() else 0.0,
                "std_float64_population": float(numeric.std(unbiased=False)) if numeric.numel() else 0.0,
                "finite": bool(torch.isfinite(numeric).all()),
            }
        )
    return rows, tensors


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare_parameter_tensors(before: dict[str, torch.Tensor], module: nn.Module) -> tuple[list[dict], dict]:
    after_named = dict(module.named_parameters())
    if set(before) != set(after_named):
        raise RuntimeError("EXP12_PARAMETER_KEY_SET_CHANGED")
    rows: list[dict] = []
    changed_elements_total = 0
    total_elements = 0
    for name, before_raw in before.items():
        after_raw = after_named[name].detach().cpu().contiguous()
        if before_raw.shape != after_raw.shape:
            raise RuntimeError(f"EXP12_PARAMETER_SHAPE_CHANGED: {name}")
        before64 = before_raw.to(torch.float64)
        after64 = after_raw.to(torch.float64)
        delta = after64 - before64
        changed_mask = after_raw.ne(before_raw)
        changed_elements = int(changed_mask.sum())
        changed_elements_total += changed_elements
        total_elements += before_raw.numel()
        l2 = float(torch.linalg.vector_norm(delta))
        base_l2 = float(torch.linalg.vector_norm(before64))
        rows.append(
            {
                "name": name,
                "shape": list(before_raw.shape),
                "numel": before_raw.numel(),
                "before_sha256": tensor_sha256(before_raw),
                "after_sha256": tensor_sha256(after_raw),
                "exact_equal": bool(torch.equal(before_raw, after_raw)),
                "changed_element_count": changed_elements,
                "max_abs_delta": float(delta.abs().max()) if delta.numel() else 0.0,
                "mean_abs_delta": float(delta.abs().mean()) if delta.numel() else 0.0,
                "l2_delta": l2,
                "relative_l2_delta": l2 / max(base_l2, 1e-30),
                "after_finite": bool(torch.isfinite(after64).all()),
            }
        )
    changed_parameters = sum(not row["exact_equal"] for row in rows)
    summary = {
        "changed_parameter_count": changed_parameters,
        "total_parameter_count": len(rows),
        "changed_ratio": changed_parameters / len(rows) if rows else 0.0,
        "changed_parameter_element_count": changed_elements_total,
        "total_parameter_element_count": total_elements,
        "changed_element_ratio": changed_elements_total / total_elements if total_elements else 0.0,
        "all_after_finite": all(row["after_finite"] for row in rows),
    }
    return rows, summary


def compare_buffers(before: dict[str, torch.Tensor], module: nn.Module) -> dict:
    after = dict(module.named_buffers())
    if set(before) != set(after):
        raise RuntimeError("EXP12_BUFFER_KEY_SET_CHANGED")
    changed = [name for name in before if not torch.equal(before[name], after[name].detach().cpu())]
    return {
        "buffer_changed_count": len(changed),
        "buffer_total_count": len(before),
        "changed_buffer_names": changed,
    }


def load_exp12_0_snapshot(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["name"]: row for row in rows}


def optimizer_identity_report(model: SimSiamSmoke, optimizer: torch.optim.Optimizer) -> dict:
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    identity_counts = Counter(id(parameter) for parameter in optimizer_parameters)
    encoder_parameters = list(model.encoder.parameters())
    projector_parameters = list(model.projector.parameters())
    predictor_parameters = list(model.predictor.parameters())
    encoder_exact_once = sum(identity_counts[id(parameter)] == 1 for parameter in encoder_parameters)
    all_model_parameters = list(model.parameters())
    return {
        "optimizer_parameter_object_count": len(optimizer_parameters),
        "optimizer_unique_parameter_object_count": len(identity_counts),
        "optimizer_duplicate_object_count": len(optimizer_parameters) - len(identity_counts),
        "model_parameter_object_count": len(all_model_parameters),
        "encoder_parameter_tensor_count": len(encoder_parameters),
        "encoder_parameter_element_count": sum(parameter.numel() for parameter in encoder_parameters),
        "encoder_parameters_present_exactly_once": encoder_exact_once,
        "projector_parameter_tensor_count": len(projector_parameters),
        "predictor_parameter_tensor_count": len(predictor_parameters),
        "all_model_parameters_present_exactly_once": all(
            identity_counts[id(parameter)] == 1 for parameter in all_model_parameters
        ),
        "no_optimizer_duplicates": len(optimizer_parameters) == len(identity_counts),
    }


def gradient_report(module: nn.Module) -> dict:
    parameters = list(module.parameters())
    present = [parameter for parameter in parameters if parameter.grad is not None]
    finite = [parameter for parameter in present if bool(torch.isfinite(parameter.grad).all())]
    nonzero = [parameter for parameter in present if bool(parameter.grad.ne(0).any())]
    return {
        "parameter_tensor_count": len(parameters),
        "gradient_present_tensor_count": len(present),
        "gradient_finite_tensor_count": len(finite),
        "gradient_nonzero_tensor_count": len(nonzero),
        "gradient_nonzero_element_count": sum(int(parameter.grad.ne(0).sum()) for parameter in present),
        "gradient_l2_norm": math.sqrt(
            sum(float(parameter.grad.detach().float().pow(2).sum()) for parameter in present)
        ),
    }


def batch_metrics(feature1: torch.Tensor, feature2: torch.Tensor,
                  embedding1: torch.Tensor, embedding2: torch.Tensor) -> dict:
    features = torch.cat([feature1.detach(), feature2.detach()], dim=0).float()
    embeddings = torch.cat([embedding1.detach(), embedding2.detach()], dim=0).float()
    return {
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std(dim=0, unbiased=False).mean()),
        "embedding_variance": float(embeddings.var(dim=0, unbiased=False).mean()),
    }


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    weights = args.weights.resolve()
    split_manifest = args.split_manifest.resolve()
    exp12_0_snapshot_path = args.exp12_0_snapshot.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Exp12.1-S output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP12_CUDA_REQUIRED")
    for required in (weights, split_manifest, exp12_0_snapshot_path):
        if not required.is_file():
            raise FileNotFoundError(required)

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
        "scope": "EXP12.1-S_SINGLE_STEP_AND_ONE_EPOCH_SMOKE",
        "project_root": str(project_root),
        "data_root": str(data_root),
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "exp12_0_snapshot": str(exp12_0_snapshot_path),
        "exp12_0_snapshot_sha256": file_sha256(exp12_0_snapshot_path),
        "output": str(output),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "epochs": 1,
        "seed": args.seed,
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "amp": False,
        "drop_last": True,
        "feature_std_threshold": FEATURE_STD_THRESHOLD,
        "embedding_variance_threshold": EMBEDDING_VARIANCE_THRESHOLD,
        **data_report,
    }
    write_json(output / "run_config.json", run_config)
    log("DATA_GATE_PASS", train_images=len(train_paths), split_sha256=data_report["split_manifest_sha256"])

    augmentation = build_augmentation(args.imgsz)
    dataset = TwoViewTrainDataset(train_paths, augmentation)
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
    if len(loader) < 2:
        raise RuntimeError("EXP12_SMOKE_DATALOADER_TOO_SHORT")

    model = SimSiamSmoke(weights)
    requires_grad_after_unfreeze = sum(parameter.requires_grad for parameter in model.encoder.parameters())
    before_rows, before_tensors = parameter_snapshot(model.encoder)
    before_buffers = {
        name: buffer.detach().cpu().clone() for name, buffer in model.encoder.named_buffers()
    }
    write_jsonl(output / "parameter_snapshot_before.jsonl", before_rows)

    exp12_0_rows = load_exp12_0_snapshot(exp12_0_snapshot_path)
    baseline_match_rows = []
    for row in before_rows:
        previous = exp12_0_rows.get(row["name"])
        baseline_match_rows.append(
            {
                "name": row["name"],
                "name_present": previous is not None,
                "shape_match": previous is not None and previous["shape"] == row["shape"],
                "hash_match": previous is not None and previous["sha256_raw_dtype"] == row["sha256_raw_dtype"],
            }
        )
    baseline_exact_match = (
        len(exp12_0_rows) == EXPECTED_PARAMETER_TENSORS
        and len(baseline_match_rows) == EXPECTED_PARAMETER_TENSORS
        and all(row["name_present"] and row["shape_match"] and row["hash_match"] for row in baseline_match_rows)
    )
    write_json(
        output / "baseline_alignment_report.json",
        {"exact_match": baseline_exact_match, "rows": baseline_match_rows},
    )
    structure_gate = {
        "requires_grad_true_at_framework_load_is_0": model.requires_grad_true_at_framework_load == 0,
        "requires_grad_true_after_explicit_unfreeze_is_120": requires_grad_after_unfreeze == EXPECTED_PARAMETER_TENSORS,
        "parameter_tensor_count_120": len(before_rows) == EXPECTED_PARAMETER_TENSORS,
        "parameter_element_count_1365472": sum(row["numel"] for row in before_rows) == EXPECTED_PARAMETER_ELEMENTS,
        "all_before_parameters_finite": all(row["finite"] for row in before_rows),
        "exp12_0_parameter_snapshot_exact_match": baseline_exact_match,
    }
    write_json(output / "unfreeze_gate_report.json", structure_gate)
    if not all(structure_gate.values()):
        write_json(output / "summary.json", {"status": "HARD_GATE", "failed_stage": "UNFREEZE_OR_BASELINE_ALIGNMENT", "gates": structure_gate})
        return 3
    log(
        "EXPLICIT_UNFREEZE_PASS",
        framework_load=model.requires_grad_true_at_framework_load,
        after_unfreeze=requires_grad_after_unfreeze,
    )

    model.cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
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
        write_json(output / "summary.json", {"status": "HARD_GATE", "failed_stage": "OPTIMIZER_IDENTITY"})
        return 3
    log("OPTIMIZER_IDENTITY_PASS", encoder_parameters=EXPECTED_PARAMETER_TENSORS)

    epoch_losses: list[float] = []
    feature_means: list[float] = []
    feature_stds: list[float] = []
    embedding_variances: list[float] = []
    single_step_report: dict | None = None
    processed_samples = 0

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
            raise RuntimeError("EXP12_SIMSIAM_LOSS_GATE")
        metrics = batch_metrics(feature1, feature2, embedding1, embedding2)
        loss.backward()

        if step == 1:
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
                "real_train_batch": True,
                "batch_size": int(view1.shape[0]),
                "loss": float(loss.detach()),
                "loss_finite": bool(torch.isfinite(loss)),
                "loss_has_grad_fn": loss.grad_fn is not None,
                "gradients": gradients,
                "update": single_update,
                "parameter_rows": single_rows,
            }
            write_json(output / "single_step_update_report.json", single_step_report)
            if not gradient_gate or not update_gate:
                status = single_step_report["status"]
                write_json(
                    output / "summary.json",
                    {
                        "status": status,
                        "failed_stage": "SINGLE_REAL_BATCH_UPDATE_GATE",
                        "test_accessed": False,
                        "val_images_seen": 0,
                        "test_images_seen": 0,
                    },
                )
                log("SINGLE_STEP_GATE_STOP", status=status)
                return 12 if status == "INVALID_BY_BACKBONE_NO_UPDATE" else 3
            log(
                "SINGLE_STEP_UPDATE_PASS",
                changed_parameter_count=single_update["changed_parameter_count"],
                changed_ratio=single_update["changed_ratio"],
                encoder_nonzero_gradient_tensors=gradients["gradient_nonzero_tensor_count"],
            )
        else:
            optimizer.step()

        epoch_losses.append(float(loss.detach()))
        feature_means.append(metrics["feature_mean"])
        feature_stds.append(metrics["feature_std"])
        embedding_variances.append(metrics["embedding_variance"])
        processed_samples += int(view1.shape[0])

    epoch_row = {
        "epoch": 1,
        "steps": len(epoch_losses),
        "processed_samples": processed_samples,
        "loss": float(np.mean(epoch_losses)),
        "feature_mean": float(np.mean(feature_means)),
        "feature_std": float(np.mean(feature_stds)),
        "embedding_variance": float(np.mean(embedding_variances)),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "all_metrics_finite": all(
            math.isfinite(value)
            for value in [
                *epoch_losses,
                *feature_means,
                *feature_stds,
                *embedding_variances,
            ]
        ),
    }
    with (output / "epoch_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_row))
        writer.writeheader()
        writer.writerow(epoch_row)

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
        write_json(output / "summary.json", {"status": update_summary["status"], "failed_stage": "ONE_EPOCH_PARAMETER_UPDATE"})
        return 12

    checkpoint = output / "last_smoke.pt"
    checkpoint_payload = {
        "scope": "EXP12.1-S_SMOKE_ONLY_NOT_FORMAL",
        "epoch": 1,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "run_config": run_config,
        "epoch_metrics": epoch_row,
    }
    torch.save(checkpoint_payload, checkpoint)
    exported_backbone = output / "adapted_backbone_smoke_only.pt"
    backbone_state = {
        name: value.detach().cpu() for name, value in model.encoder.state_dict().items()
    }
    torch.save(
        {
            "scope": "EXP12.1-S_SMOKE_ONLY_NOT_FORMAL",
            "backbone": backbone_state,
            "encoder_layers": [0, BACKBONE_END],
            "source_weights_sha256": file_sha256(weights),
            "ssl_train_images": EXPECTED_TRAIN_IMAGES,
            "test_accessed": False,
        },
        exported_backbone,
    )

    in_memory_hashes = {
        name: tensor_sha256(parameter) for name, parameter in model.encoder.named_parameters()
    }
    loaded_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded = SimSiamSmoke(weights).cpu()
    reloaded.load_state_dict(loaded_checkpoint["model"], strict=True)
    reload_hashes = {
        name: tensor_sha256(parameter) for name, parameter in reloaded.encoder.named_parameters()
    }
    loaded_export = torch.load(exported_backbone, map_location="cpu", weights_only=False)
    export_hashes = {
        name: tensor_sha256(value)
        for name, value in loaded_export["backbone"].items()
        if name in in_memory_hashes
    }
    reload_report = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "exported_backbone_sha256": file_sha256(exported_backbone),
        "in_memory_parameter_count": len(in_memory_hashes),
        "checkpoint_reload_match_count": sum(
            reload_hashes.get(name) == digest for name, digest in in_memory_hashes.items()
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

    collapse_suspected = (
        epoch_row["feature_std"] < FEATURE_STD_THRESHOLD
        or epoch_row["embedding_variance"] < EMBEDDING_VARIANCE_THRESHOLD
    )
    gates = {
        "data_train_only": data_report["train_manifest_exact_set_668"] and data_report["train_symlink_targets_frozen_read_only_source"],
        "explicit_unfreeze": all(structure_gate.values()),
        "optimizer_identity": optimizer_gate,
        "single_step_gradient_and_update": single_step_report is not None and single_step_report["status"] == "PASS",
        "one_epoch_parameter_update": update_summary["status"] == "PASS",
        "metrics_finite": epoch_row["all_metrics_finite"],
        "collapse_not_suspected": not collapse_suspected,
        "checkpoint_and_export_reload": reload_report["status"] == "PASS",
        "val_images_seen_0": data_report["val_images_seen"] == 0,
        "test_images_seen_0": data_report["test_images_seen"] == 0,
    }
    status = "PASS" if all(gates.values()) else "HARD_GATE"
    summary = {
        "status": status,
        "scope": "EXP12.1-S_SINGLE_STEP_AND_ONE_EPOCH_SMOKE",
        "formal_training_performed": False,
        "smoke_training_epochs": 1,
        "gates": gates,
        "test_accessed": False,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "ssl_train_images": EXPECTED_TRAIN_IMAGES,
        "epoch_metrics": epoch_row,
        "single_step_changed_parameter_count": single_step_report["update"]["changed_parameter_count"],
        "single_step_changed_ratio": single_step_report["update"]["changed_ratio"],
        "changed_parameter_count": update_summary["changed_parameter_count"],
        "total_parameter_count": update_summary["total_parameter_count"],
        "changed_ratio": update_summary["changed_ratio"],
        "checkpoint_sha256": reload_report["checkpoint_sha256"],
        "exported_backbone_sha256": reload_report["exported_backbone_sha256"],
        "wall_seconds": time.monotonic() - started,
        "next_gate": "WAITING_SEPARATE_AUTHORIZATION_BEFORE_FORMAL_SSL",
    }
    write_json(output / "summary.json", summary)
    log("EXP12_1S_COMPLETE", status=status, changed_ratio=update_summary["changed_ratio"])
    return 0 if status == "PASS" else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--exp12-0-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.00625)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.weights = args.weights or args.project_root / "weights/yolo11n-seg.pt"
    args.split_manifest = args.split_manifest or args.project_root / DEFAULT_SPLIT
    args.exp12_0_snapshot = args.exp12_0_snapshot or args.project_root / DEFAULT_EXP12_0
    return args


def main() -> int:
    args = parse_args()
    try:
        return run(args)
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
                    "test_accessed": False,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
