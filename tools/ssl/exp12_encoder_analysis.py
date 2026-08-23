#!/usr/bin/env python3
"""Exp12.0 no-training audit for the YOLO11n-seg SimSiam encoder boundary."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO


DEFAULT_ROOT = Path("/root/autodl-tmp/borescope-new-seg-repro")
DEFAULT_DATA = Path("/root/autodl-tmp/borescope-new-seg-data/v1")
DEFAULT_SPLIT = Path("results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv")
READ_ONLY_SOURCE = Path("/root/autodl-tmp/损伤训练数据集")
BACKBONE_END = 10
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_parameters(module: torch.nn.Module) -> list[dict]:
    rows = []
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().to(torch.float64)
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "requires_grad_at_load": bool(parameter.requires_grad),
                "sha256_raw_dtype": tensor_sha256(parameter),
                "mean_float64": float(value.mean()) if value.numel() else 0.0,
                "std_float64_population": float(value.std(unbiased=False)) if value.numel() else 0.0,
                "finite": bool(torch.isfinite(value).all()),
            }
        )
    return rows


def forward_shapes(layers: torch.nn.Sequential, imgsz: int) -> tuple[list[dict], list[int]]:
    x = torch.zeros(2, 3, imgsz, imgsz)
    rows = []
    layers.eval()
    with torch.inference_mode():
        for index, layer in enumerate(layers):
            input_shape = list(x.shape)
            x = layer(x)
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"EXP12_ENCODER_NON_TENSOR_OUTPUT layer={index} type={type(x)!r}")
            rows.append(
                {
                    "index": index,
                    "class": layer.__class__.__name__,
                    "from": getattr(layer, "f", None),
                    "input_shape": input_shape,
                    "output_shape": list(x.shape),
                }
            )
    return rows, list(x.shape)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    weights = (args.weights or project_root / "weights/yolo11n-seg.pt").resolve()
    split_manifest = (args.split_manifest or project_root / DEFAULT_SPLIT).resolve()
    output = (args.output or project_root / "results/ssl/exp12_0_encoder_analysis").resolve()
    train_images = (data_root / "images/train").resolve()
    read_only_source = READ_ONLY_SOURCE.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Exp12.0 output: {output}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not split_manifest.is_file():
        raise FileNotFoundError(split_manifest)
    if not train_images.is_dir() or train_images.name != "train" or train_images.parent.name != "images":
        raise RuntimeError(f"EXP12_TRAIN_ONLY_PATH_GATE: {train_images}")

    images = sorted(
        path
        for path in train_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images or any(path.parent != train_images for path in images):
        raise RuntimeError("EXP12_TRAIN_ONLY_RESOLVED_PATH_GATE")
    resolved_targets = [path.resolve() for path in images]
    frozen_source_gate = (
        len(images) == 668
        and all(path.is_symlink() for path in images)
        and all(target.is_file() and read_only_source in target.parents for target in resolved_targets)
        and all(path.name == target.name for path, target in zip(images, resolved_targets, strict=True))
    )
    if not frozen_source_gate:
        raise RuntimeError("EXP12_TRAIN_SYMLINK_TARGET_GATE")
    with split_manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_train = sorted(
        train_images / f"{row['stem']}{row['image_suffix']}"
        for row in rows
        if row["split"] == "train"
    )
    train_set_gate = (
        len(images) == 668
        and len(manifest_train) == 668
        and len(set(images)) == 668
        and set(images) == set(manifest_train)
    )
    if not train_set_gate:
        raise RuntimeError("EXP12_TRAIN_MANIFEST_EXACT_SET_GATE")

    yolo = YOLO(str(weights))
    source_layers = yolo.model.model[: BACKBONE_END + 1]
    if len(source_layers) != BACKBONE_END + 1:
        raise RuntimeError("EXP12_BACKBONE_LAYER_COUNT_GATE")
    encoder = torch.nn.Sequential(*[copy.deepcopy(layer) for layer in source_layers]).cpu()
    structure, encoder_output_shape = forward_shapes(encoder, args.imgsz)
    parameter_rows = snapshot_parameters(encoder)
    state_tensor_count = len(encoder.state_dict())
    expected_classes = ["Conv", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2", "SPPF", "C2PSA"]
    gates = {
        "train_manifest_exact_set_668": train_set_gate,
        "train_symlink_targets_frozen_read_only_source": frozen_source_gate,
        "encoder_layer_count_11": len(source_layers) == 11,
        "encoder_class_sequence": [row["class"] for row in structure] == expected_classes,
        "encoder_output_shape": encoder_output_shape == [2, 256, args.imgsz // 32, args.imgsz // 32],
        "parameter_tensor_count_120": len(parameter_rows) == 120,
        "parameter_element_count_1365472": sum(row["numel"] for row in parameter_rows) == 1_365_472,
        "state_tensor_count_240": state_tensor_count == 240,
        "all_parameters_finite": all(row["finite"] for row in parameter_rows),
    }
    status = "PASS" if all(gates.values()) else "HARD_GATE"

    output.mkdir(parents=True)
    with (output / "parameter_snapshot_before.jsonl").open("w", encoding="utf-8") as handle:
        for row in parameter_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    structure_report = {
        "status": status,
        "gates": gates,
        "training_performed": False,
        "optimizer_created": False,
        "backward_called": False,
        "test_accessed": False,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "data_root": str(data_root),
        "ssl_input_root": str(train_images),
        "ssl_train_image_count": len(images),
        "ssl_train_symlink_count": sum(path.is_symlink() for path in images),
        "read_only_source_root": str(read_only_source),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
        "official_weights": str(weights),
        "official_weights_sha256": file_sha256(weights),
        "encoder_layer_range": [0, BACKBONE_END],
        "encoder_layer_count": len(source_layers),
        "input_shape": [2, 3, args.imgsz, args.imgsz],
        "encoder_output_shape": encoder_output_shape,
        "pooled_feature_dimension": encoder_output_shape[1],
        "parameter_tensor_count": len(parameter_rows),
        "parameter_element_count": sum(row["numel"] for row in parameter_rows),
        "state_tensor_count": state_tensor_count,
        "all_parameters_finite": all(row["finite"] for row in parameter_rows),
        "requires_grad_true_at_framework_load": sum(row["requires_grad_at_load"] for row in parameter_rows),
        "layers": structure,
    }
    write_json(output / "encoder_structure.json", structure_report)
    write_json(
        output / "summary.json",
        {
            "status": structure_report["status"],
            "scope": "EXP12.0_NO_TRAINING_ENCODER_ANALYSIS",
            "training_performed": False,
            "test_accessed": False,
            "next_gate": "WAITING_USER_CONFIRMATION_BEFORE_ANY_SMOKE_OPTIMIZER_STEP",
            "artifacts": ["encoder_structure.json", "parameter_snapshot_before.jsonl", "run.log"],
        },
    )
    structure_text = json.dumps(structure_report, ensure_ascii=False, indent=2)
    (output / "run.log").write_text(structure_text + "\n", encoding="utf-8")
    print(structure_text)
    return 0 if structure_report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
