#!/usr/bin/env python3
"""Repair Exp09 transfer verification without rerunning SimSiam SSL."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path("/root/autodl-tmp/borescope-new-seg-repro")
OUT = ROOT / "results/fast_repro/exp09_transfer_repair"
OFFICIAL = ROOT / "weights/yolo11n-seg.pt"
SOURCE = ROOT / "results/fast_repro/exp09_simsiam/ssl/adapted_backbone.pt"
OLD_NATIVE = ROOT / "results/fast_repro/exp09_simsiam/simsiam_init_yolo11n_seg.pt"
BACKBONE_END = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def category(name: str) -> str:
    if name.endswith("running_mean"):
        return "bn_running_mean"
    if name.endswith("running_var"):
        return "bn_running_var"
    if name.endswith("num_batches_tracked"):
        return "bn_num_batches_tracked"
    if name.endswith(".conv.weight") or (name.endswith(".weight") and ".bn." not in name):
        return "conv_weight"
    if name.endswith(".bn.weight"):
        return "bn_weight"
    if name.endswith(".bn.bias"):
        return "bn_bias"
    return "other"


def kind(name: str, named_parameters: set[str]) -> str:
    return "parameter" if name in named_parameters else "buffer"


def differences(source: torch.Tensor, destination: torch.Tensor) -> tuple[bool, float, float]:
    source_cast = source.detach().cpu().to(destination.dtype)
    destination = destination.detach().cpu()
    exact = torch.equal(source_cast, destination)
    delta = (source_cast.to(torch.float64) - destination.to(torch.float64)).abs()
    return exact, float(delta.max()) if delta.numel() else 0.0, float(delta.mean()) if delta.numel() else 0.0


def save_behavior() -> None:
    package = ROOT / ".venv/lib/python3.12/site-packages/ultralytics"
    records = [
        (package / "engine/model.py", "Model.save", 364, '"model": deepcopy(self.model).half()'),
        (package / "engine/trainer.py", "Trainer.save_model", 725, "ema = deepcopy(ema).half()"),
        (package / "utils/torch_utils.py", "strip_optimizer", 828, 'x["model"].half()'),
    ]
    lines = ["Ultralytics version: 8.4.117", "Site-packages modified: no", ""]
    for path, function, line, behavior in records:
        actual = path.read_text(encoding="utf-8").splitlines()[line - 1].strip()
        if behavior not in actual:
            raise RuntimeError(f"LOCAL_SOURCE_MISMATCH: {path}:{line}: {actual}")
        lines.extend([f"source file: {path}", f"function: {function}", f"actual code location: line {line}", f"checkpoint model dtype behavior: {actual}", ""])
    (OUT / "local_save_behavior.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    save_behavior()
    source_package = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source = source_package["backbone"]
    official_yolo = YOLO(str(OFFICIAL))
    official_backbone = official_yolo.model.model[: BACKBONE_END + 1]
    expected = official_backbone.state_dict()
    parameter_names = set(dict(official_backbone.named_parameters()))
    missing = sorted(set(expected) - set(source))
    unexpected = sorted(set(source) - set(expected))
    shape_mismatch = sorted(name for name in set(expected) & set(source) if expected[name].shape != source[name].shape)

    old_reload = YOLO(str(OLD_NATIVE)).model.model[: BACKBONE_END + 1].state_dict()
    old_rows = []
    for name in expected:
        exact, max_diff, mean_diff = differences(source[name], old_reload[name])
        old_rows.append({
            "name": name,
            "shape": "x".join(map(str, source[name].shape)),
            "dtype_source": str(source[name].dtype),
            "dtype_reload": str(old_reload[name].dtype),
            "parameter_or_buffer": kind(name, parameter_names),
            "tensor_category": category(name),
            "old_byte_equal": str(hashlib.sha256(source[name].numpy().tobytes()).hexdigest() == hashlib.sha256(old_reload[name].numpy().tobytes()).hexdigest()).lower(),
            "exact_after_dtype_cast": str(exact).lower(),
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
        })
    write_csv(OUT / "old_mismatch_audit.csv", old_rows)

    fresh = YOLO(str(OFFICIAL))
    destination_module = fresh.model.model[: BACKBONE_END + 1]
    destination_module.load_state_dict(source, strict=True)
    fresh.model.eval()
    immediate = destination_module.state_dict()
    immediate_rows = []
    for name in expected:
        exact, max_diff, _ = differences(source[name], immediate[name])
        immediate_rows.append({
            "name": name,
            "parameter_or_buffer": kind(name, parameter_names),
            "source_dtype": str(source[name].dtype),
            "destination_dtype": str(immediate[name].dtype),
            "exact_after_dtype_cast": str(exact).lower(),
            "max_abs_diff": max_diff,
            "pass": str(exact).lower(),
        })
    write_csv(OUT / "immediate_transfer_audit.csv", immediate_rows)

    bn_buffer_categories = {"bn_running_mean", "bn_running_var", "bn_num_batches_tracked"}
    before_eval = {name: tensor.detach().cpu().clone() for name, tensor in immediate.items() if category(name) in bn_buffer_categories}
    with torch.no_grad():
        fresh.model(torch.zeros(1, 3, 640, 640))
    after_eval = destination_module.state_dict()
    diagnostic = deepcopy(fresh.model)
    diagnostic.train()
    diagnostic_backbone = diagnostic.model[: BACKBONE_END + 1]
    before_train = {name: tensor.detach().cpu().clone() for name, tensor in diagnostic_backbone.state_dict().items() if category(name) in bn_buffer_categories}
    with torch.no_grad():
        diagnostic(torch.randn(2, 3, 640, 640))
    after_train = diagnostic_backbone.state_dict()
    bn_rows = []
    for name, before in before_eval.items():
        bn_rows.append({
            "name": name,
            "tensor_category": category(name),
            "eval_forward_changed": str(not torch.equal(before, after_eval[name].cpu())).lower(),
            "train_forward_changed": str(not torch.equal(before_train[name], after_train[name].cpu())).lower(),
        })
    write_csv(OUT / "bn_buffer_behavior.csv", bn_rows)

    fp32_path = OUT / "fp32_backbone_state_dict.pt"
    torch.save({"backbone": source}, fp32_path)
    control_source = torch.load(fp32_path, map_location="cpu", weights_only=False)["backbone"]
    control = YOLO(str(OFFICIAL)).model.model[: BACKBONE_END + 1]
    control.load_state_dict(control_source, strict=True)
    fp32_exact = sum(differences(source[name], control.state_dict()[name])[0] for name in expected)

    native_path = OUT / "native_roundtrip.pt"
    fresh.save(native_path)
    native = YOLO(str(native_path)).model.model[: BACKBONE_END + 1].state_dict()
    native_rows = []
    for name in expected:
        exact, max_diff, mean_diff = differences(source[name], native[name])
        source_half_float = source[name].half().to(native[name].dtype) if source[name].is_floating_point() else source[name].to(native[name].dtype)
        quantized_equal = torch.equal(source_half_float, native[name].cpu())
        native_rows.append({
            "name": name,
            "parameter_or_buffer": kind(name, parameter_names),
            "tensor_category": category(name),
            "source_dtype": str(source[name].dtype),
            "reload_dtype": str(native[name].dtype),
            "exact_source_cast": str(exact).lower(),
            "exact_after_fp16_quantization": str(quantized_equal).lower(),
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "allclose_fp16_tolerance": str(torch.allclose(source[name].to(torch.float32), native[name].cpu().to(torch.float32), atol=5e-4, rtol=5e-3)).lower(),
        })
    write_csv(OUT / "native_roundtrip_audit.csv", native_rows)

    changed_from_coco = [name for name in expected if not torch.equal(source[name].to(expected[name].dtype), expected[name].cpu())]
    changed_trainable = [name for name in changed_from_coco if name in parameter_names]
    witness = changed_trainable[0] if changed_trainable else None
    witness_pass = bool(witness and torch.equal(source[witness].to(immediate[witness].dtype), immediate[witness].cpu()))
    immediate_parameter_failures = [row["name"] for row in immediate_rows if row["parameter_or_buffer"] == "parameter" and row["pass"] != "true"]
    native_unexplained = [row["name"] for row in native_rows if row["exact_after_fp16_quantization"] != "true"]
    eval_bn_changes = [row["name"] for row in bn_rows if row["eval_forward_changed"] == "true"]
    old_mismatches = [row for row in old_rows if row["old_byte_equal"] != "true"]
    real_failure = bool(missing or unexpected or shape_mismatch or immediate_parameter_failures or fp32_exact != len(expected) or native_unexplained or not witness_pass or eval_bn_changes)
    report = {
        "status": "REAL_TRANSFER_FAILURE" if real_failure else "PASS_REVISED",
        "test_accessed": False,
        "source_sha256": sha256(SOURCE),
        "expected": len(expected),
        "loaded": len(immediate),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
        "trainable_parameter_count": len(parameter_names),
        "immediate_trainable_parameter_failures": len(immediate_parameter_failures),
        "fp32_roundtrip_exact": fp32_exact,
        "native_roundtrip_fp16_explained": len(expected) - len(native_unexplained),
        "native_roundtrip_unexplained": len(native_unexplained),
        "old_byte_mismatch_count": len(old_mismatches),
        "old_mismatch_by_category": dict(sorted(Counter(row["tensor_category"] for row in old_mismatches).items())),
        "old_mismatch_by_parameter_or_buffer": dict(sorted(Counter(row["parameter_or_buffer"] for row in old_mismatches).items())),
        "eval_bn_changed_count": len(eval_bn_changes),
        "train_bn_changed_count": sum(row["train_forward_changed"] == "true" for row in bn_rows),
        "coco_changed_tensor_count": len(changed_from_coco),
        "coco_changed_trainable_parameter_count": len(changed_trainable),
        "witness_parameter": witness,
        "witness_pass": witness_pass,
        "native_checkpoint_sha256": sha256(native_path),
        "fp32_control_sha256": sha256(fp32_path),
    }
    (OUT / "transfer_gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS_REVISED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
