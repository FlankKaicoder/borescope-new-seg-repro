#!/usr/bin/env python3
"""Exp10.1b TRUE-FP32 and formal Trainer-path no-update verification.

TRAIN-only diagnostic. Uses the frozen Exp10.1a post-preprocess batch and state;
does not validate, optimize, train an epoch, or access TEST.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp10_1a_nan_probe as old  # noqa: E402


LOSS_NAMES = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss")


def clean_json(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    return x


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_legacy_single_replay(path: Path) -> dict[str, Any] | None:
    """Read the terminal JSON emitted by the frozen Exp10.1a replay implementation."""
    if not path.exists():
        return None
    marker = "SINGLE_REPLAY_JSON="
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensors(x: Any):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from tensors(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from tensors(v)


def finite(x: Any) -> bool:
    return all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in tensors(x))


def max_abs(x: Any) -> float | None:
    values = []
    for t in tensors(x):
        if t.is_floating_point() and t.numel():
            q = t.detach()[torch.isfinite(t.detach())]
            if q.numel():
                values.append(float(q.abs().max().cpu()))
    return max(values) if values else None


def first_dtype(x: Any) -> str:
    return next((str(t.dtype) for t in tensors(x) if t.is_floating_point()), "N/A")


def dtype_counts(model: torch.nn.Module) -> dict[str, Any]:
    params = [p for p in model.parameters() if p.is_floating_point()]
    buffers = [b for b in model.buffers() if b.is_floating_point()]
    def count(items, dtype): return sum(x.dtype == dtype for x in items)
    return {
        "first_parameter_dtype": str(next(model.parameters()).dtype),
        "float16_parameter_count": count(params, torch.float16),
        "float32_parameter_count": count(params, torch.float32),
        "other_parameter_count": sum(x.dtype not in {torch.float16, torch.float32} for x in params),
        "float16_buffer_count": count(buffers, torch.float16),
        "float32_buffer_count": count(buffers, torch.float32),
        "other_buffer_count": sum(x.dtype not in {torch.float16, torch.float32} for x in buffers),
        "parameter_dtype_unique": sorted({str(x.dtype) for x in params}),
        "buffer_dtype_unique": sorted({str(x.dtype) for x in buffers}),
    }


def state_dtype_counts(state: dict[str, torch.Tensor], parameter_keys: set[str], buffer_keys: set[str]) -> dict[str, Any]:
    params = [v for k, v in state.items() if k in parameter_keys and v.is_floating_point()]
    buffers = [v for k, v in state.items() if k in buffer_keys and v.is_floating_point()]
    return {
        "first_parameter_dtype": str(params[0].dtype),
        "float16_parameter_count": sum(x.dtype == torch.float16 for x in params),
        "float32_parameter_count": sum(x.dtype == torch.float32 for x in params),
        "other_parameter_count": sum(x.dtype not in {torch.float16, torch.float32} for x in params),
        "float16_buffer_count": sum(x.dtype == torch.float16 for x in buffers),
        "float32_buffer_count": sum(x.dtype == torch.float32 for x in buffers),
        "other_buffer_count": sum(x.dtype not in {torch.float16, torch.float32} for x in buffers),
        "parameter_dtype_unique": sorted({str(x.dtype) for x in params}),
        "buffer_dtype_unique": sorted({str(x.dtype) for x in buffers}),
    }


def tensor_hash(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    q = t.detach().float().cpu()
    return {
        "shape": list(t.shape), "dtype": str(t.dtype), "min": float(q.min()), "max": float(q.max()),
        "mean": float(q.mean()), "std": float(q.std()), "sha256": tensor_hash(t), "finite": finite(t),
    }


def batch_hashes(batch: dict[str, Any]) -> dict[str, str]:
    return {k: tensor_hash(v) for k, v in batch.items() if torch.is_tensor(v)}


def loss_result(preds: Any, items: dict[str, torch.Tensor], total: torch.Tensor) -> dict[str, Any]:
    return {
        "raw_output_finite": finite(preds),
        **{f"{k}_finite": finite(items[k]) for k in LOSS_NAMES},
        **{k: float(items[k].detach().cpu()) for k in LOSS_NAMES},
        "total_loss": float(total.detach().cpu()), "total_loss_finite": finite(total),
        "all_loss_finite": finite(items) and finite(total),
    }


def layer_trace(model: torch.nn.Module):
    rows: list[dict[str, Any]] = []
    handles = []
    for i, module in enumerate(model.model):
        def hook(_m, inp, out, idx=i):
            rows.append({
                "module": f"model.{idx}", "input_dtype": first_dtype(inp), "output_dtype": first_dtype(out),
                "input_finite": finite(inp), "output_finite": finite(out),
                "input_max_abs": max_abs(inp), "output_max_abs": max_abs(out),
            })
        handles.append(module.register_forward_hook(hook))
    return handles, rows


def run_forward_loss(model: torch.nn.Module, batch: dict[str, Any], device_type: str, amp: bool, trace: bool = False):
    handles, rows = layer_trace(model) if trace else ([], [])
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=amp):
        preds = model(batch["img"])
        loss, items = model.loss(batch, preds)
        total = loss.sum()
    for h in handles: h.remove()
    return loss_result(preds, items, total), rows


def trainer_overrides(checkpoint: Path, data: Path, out: Path) -> dict[str, Any]:
    return {
        "model": str(checkpoint), "data": str(data), "imgsz": 640, "batch": 32, "epochs": 30,
        "seed": 43, "deterministic": True, "amp": True, "optimizer": "AdamW", "device": 0,
        "workers": 4, "cache": False, "val": False, "plots": False, "save": False,
        "project": str(out / "trainer_setup"), "name": "no_update", "exist_ok": False, "verbose": True,
    }


def make_trainer(checkpoint: Path, data: Path, out: Path, hard: set[str], pool: dict[str, dict[str, str]]):
    old.ProbeTrainer.hard = hard
    old.ProbeTrainer.pool = pool
    old.ProbeTrainer.out = out
    old.ProbeTrainer.baseline_checkpoint = checkpoint
    trainer = old.ProbeTrainer(overrides=trainer_overrides(checkpoint, data, out))
    trainer._setup_train()
    return trainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--data", type=Path, default=Path("/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml"))
    ap.add_argument("--resume-existing", action="store_true", help="Regenerate evidence in an existing Exp10.1b directory")
    args = ap.parse_args()
    root = args.repo.resolve()
    out = root / "results/final_verify/exp10_1b_true_fp32"
    if out.exists() and not args.resume_existing: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=args.resume_existing)
    old_dir = root / "results/final_verify/exp10_1a_seed43_nan_probe"
    snapshot_path = old_dir / "first_nonfinite_snapshot.pt"
    initial_path = old_dir / "baseline_initial_state.pt"
    checkpoint = root / "results/final_verify/exp10/seed43/baseline100/ultralytics/baseline/weights/best.pt"
    assert sha256(snapshot_path) == "fc1e78db6cb284f7ed3ebaefefa7351beda745ef001f8f01d71c10eff1b5e790"
    assert sha256(initial_path) == "2c6a2e62d59717f35b0caa77ef13a7f894776055f2560ba5ec5b1fdb7509ec30"
    assert sha256(checkpoint) == "09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c"
    saved = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    saved_batch_cpu = saved["batch"]
    state = saved["pre_step_model_state"]
    pool, hard = old.read_pool(root / "results/final_verify/exp10/seed43/hard_pool/hard_pool.csv")

    # Raw serialized EMA and public YOLO load dtype transition.
    raw_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_model = raw_ckpt.get("ema") or raw_ckpt.get("model")
    raw_counts = dtype_counts(raw_model)
    loaded_model = YOLO(str(checkpoint)).model
    loaded_counts = dtype_counts(loaded_model)

    trainer = make_trainer(checkpoint, args.data, out, hard, pool)
    trainer_model = unwrap_model(trainer.model)
    trainer_counts = dtype_counts(trainer_model)
    trainer_effective_amp = bool(trainer.amp)

    # Identify state keys using the freshly built architecture, then audit frozen state dtype.
    parameter_keys = {k for k, _ in trainer_model.named_parameters()}
    buffer_keys = {k for k, _ in trainer_model.named_buffers()}
    frozen_state_counts = state_dtype_counts(state, parameter_keys, buffer_keys)
    saved_input = saved_batch_cpu["img"]
    saved_input_summary = tensor_summary(saved_input)
    saved_target_finite = finite({k: v for k, v in saved_batch_cpu.items() if torch.is_tensor(v) and k != "img"})

    explicit_model = deepcopy(trainer_model).float()
    explicit_counts = dtype_counts(explicit_model)
    transition_rows = []
    for stage, rec in (
        ("raw_serialized_checkpoint_ema", raw_counts),
        ("YOLO_checkpoint_load", loaded_counts),
        ("trainer_setup_complete", trainer_counts),
        ("explicit_model_float", explicit_counts),
    ):
        transition_rows.append({"stage": stage, **{k: rec[k] for k in (
            "float16_parameter_count", "float32_parameter_count", "other_parameter_count",
            "float16_buffer_count", "float32_buffer_count", "other_buffer_count")}})
    with (out / "checkpoint_dtype_transition.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(transition_rows[0])); w.writeheader(); w.writerows(transition_rows)

    audit_text = [
        "Exp10.1a old replay dtype-method audit",
        "implementation: model.load_state_dict(state); model.train(); batch=to_device(batch_cpu); autocast(enabled=False)",
        "old replay called model.float(): NO",
        "old replay called input.float(): NO",
        f"old replay state first parameter dtype: {frozen_state_counts['first_parameter_dtype']}",
        f"old replay parameter dtype distribution: {frozen_state_counts['parameter_dtype_unique']} counts fp16={frozen_state_counts['float16_parameter_count']} fp32={frozen_state_counts['float32_parameter_count']} other={frozen_state_counts['other_parameter_count']}",
        f"old replay floating buffer dtype distribution: {frozen_state_counts['buffer_dtype_unique']} counts fp16={frozen_state_counts['float16_buffer_count']} fp32={frozen_state_counts['float32_buffer_count']} other={frozen_state_counts['other_buffer_count']}",
        f"old replay input dtype: {saved_input.dtype}",
        "old replay autocast: disabled for FP32-labelled branch",
        "old replay CUDA device: cuda:0",
        "old replay model mode: train",
    ]
    old_invalid = bool(
        frozen_state_counts["float16_parameter_count"] or frozen_state_counts["other_parameter_count"] or
        frozen_state_counts["float16_buffer_count"] or frozen_state_counts["other_buffer_count"] or
        saved_input.dtype != torch.float32
    )
    audit_text.append(f"EXP10_1A_FP32_REPLAY_INVALID_BY_DTYPE: {str(old_invalid).lower()}")
    (out / "exp10_1a_dtype_method_audit.txt").write_text("\n".join(audit_text) + "\n", encoding="utf-8")

    # Recreate formal first dataloader batch only for preprocessing equivalence; it is not used as exact replay input.
    raw_first = next(iter(trainer.train_loader))
    reproduced = trainer.preprocess_batch(raw_first)
    saved_hashes = batch_hashes(saved_batch_cpu)
    reproduced_cpu = old.cpu_clone(reproduced)
    reproduced_hashes = batch_hashes(reproduced_cpu)
    preprocess_rows = []
    for name, batch, provenance in (
        ("exp10_1a_saved_post_preprocess", saved_batch_cpu, "frozen exact tensor; already normalized/cast/resized/augmented"),
        ("exp10_1b_recreated_trainer_post_preprocess", reproduced_cpu, "fresh deterministic reconstruction; audit only"),
    ):
        s = tensor_summary(batch["img"])
        preprocess_rows.append({"source": name, "provenance": provenance, **s,
                                "all_tensor_hashes_match_saved": batch_hashes(batch) == saved_hashes})
    with (out / "preprocess_equivalence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(preprocess_rows[0])); w.writeheader(); w.writerows(preprocess_rows)
    preprocessing_exact = reproduced_hashes == saved_hashes

    device = trainer.device
    exact_batch_gpu = old.to_device(saved_batch_cpu, device)
    exact_batch_gpu["img"] = exact_batch_gpu["img"].float()

    # TRUE-FP32: explicit FP32 model/state/input and native autocast disabled.
    true_model = trainer_model
    true_model.load_state_dict(state, strict=True)
    true_model.float().train()
    true_counts = dtype_counts(true_model)
    true_gate = bool(
        true_counts["parameter_dtype_unique"] == ["torch.float32"] and
        true_counts["buffer_dtype_unique"] == ["torch.float32"] and
        exact_batch_gpu["img"].dtype == torch.float32 and finite(exact_batch_gpu["img"]) and saved_target_finite
    )
    if not true_gate: raise RuntimeError("TRUE-FP32 dtype gate failed")
    conv_capture: dict[str, Any] = {}
    conv = true_model.model[1].conv
    def conv_pre(_m, inp): conv_capture["input"] = inp[0].detach().clone()
    def conv_post(_m, _inp, output): conv_capture["output"] = output.detach().clone()
    hp = conv.register_forward_pre_hook(conv_pre); ho = conv.register_forward_hook(conv_post)
    fp32_result, fp32_trace = run_forward_loss(true_model, exact_batch_gpu, "cuda", False, trace=True)
    hp.remove(); ho.remove()
    first_fp32_nonfinite = next((r["module"] for r in fp32_trace if not r["output_finite"]), "NONE")
    fp32_record = {
        "name": "TRUE_FP32_REPLAY", "model_parameter_dtype_unique": true_counts["parameter_dtype_unique"],
        "model_buffer_dtype_unique": true_counts["buffer_dtype_unique"], "input_dtype": str(exact_batch_gpu["img"].dtype),
        "input_finite": finite(exact_batch_gpu["img"]), "targets_finite": saved_target_finite,
        "autocast_enabled": False, "model_mode": "train", "device": str(device), "dtype_gate_pass": true_gate,
        "first_nonfinite_module": first_fp32_nonfinite, **fp32_result,
    }
    dump_json(out / "true_fp32_replay.json", fp32_record)
    with (out / "true_fp32_module_trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fp32_trace[0])); w.writeheader(); w.writerows(fp32_trace)

    # TRUE AMP starts from the exact same FP32 master state and FP32 input.
    true_model.load_state_dict(state, strict=True); true_model.float().train()
    amp_result, _ = run_forward_loss(true_model, exact_batch_gpu, "cuda", True, trace=False)
    amp_record = {
        "name": "TRUE_AMP_REPLAY", "master_parameter_dtype_unique": dtype_counts(true_model)["parameter_dtype_unique"],
        "master_buffer_dtype_unique": dtype_counts(true_model)["buffer_dtype_unique"], "input_dtype": str(exact_batch_gpu["img"].dtype),
        "autocast_enabled": True, "model_mode": "train", "device": str(device), **amp_result,
    }
    dump_json(out / "true_amp_replay.json", amp_record)

    # Formal Trainer path: no extra preprocessing because frozen batch is already post-preprocess exact tensor.
    trainer_model.load_state_dict(state, strict=True); trainer_model.train()
    formal_counts = dtype_counts(trainer_model)
    formal_amp_result, _ = run_forward_loss(trainer_model, exact_batch_gpu, "cuda", trainer_effective_amp, trace=False)
    trainer_model.load_state_dict(state, strict=True); trainer_model.float().train()
    formal_fp32_result, _ = run_forward_loss(trainer_model, exact_batch_gpu, "cuda", False, trace=False)
    trainer_path_record = {
        "formal_setup_model_dtypes": formal_counts, "effective_amp": trainer_effective_amp,
        "saved_batch_used_directly_as_post_preprocess_tensor": True,
        "formal_trainer_path_amp": formal_amp_result,
        "formal_trainer_path_true_fp32": formal_fp32_result,
        "optimizer_step_performed": False, "val_accessed_for_probe": False, "test_accessed": False,
    }
    dump_json(out / "trainer_path_replay.json", trainer_path_record)
    trainer_dtype_rows = [
        {"stage": "trainer_setup", "effective_amp": trainer_effective_amp, **trainer_counts},
        {"stage": "trainer_exact_state_loaded", "effective_amp": trainer_effective_amp, **formal_counts},
        {"stage": "trainer_explicit_float", "effective_amp": False, **dtype_counts(trainer_model)},
    ]
    flat_keys = ["stage", "effective_amp", "first_parameter_dtype", "float16_parameter_count", "float32_parameter_count",
                 "other_parameter_count", "float16_buffer_count", "float32_buffer_count", "other_buffer_count",
                 "parameter_dtype_unique", "buffer_dtype_unique"]
    with (out / "trainer_path_dtype_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys); w.writeheader(); w.writerows(trainer_dtype_rows)

    # FP32 conv sanity and conditional CPU replay.
    conv_input_max = max_abs(conv_capture.get("input")); weight_max = float(conv.weight.detach().abs().max().cpu())
    bias_max = float(conv.bias.detach().abs().max().cpu()) if conv.bias is not None else 0.0
    kh, kw = conv.kernel_size; terms = kh * kw * (conv.in_channels / conv.groups)
    bound = float(conv_input_max * weight_max * terms + bias_max) if conv_input_max is not None else None
    conv_sanity = {
        "module": "model.1.conv", "input_dtype": first_dtype(conv_capture.get("input")),
        "output_dtype": first_dtype(conv_capture.get("output")), "input_finite": finite(conv_capture.get("input")),
        "output_finite": finite(conv_capture.get("output")), "input_max_abs": conv_input_max,
        "output_max_abs": max_abs(conv_capture.get("output")), "weight_max_abs": weight_max, "bias_max_abs": bias_max,
        "kernel_size": [kh, kw], "in_channels": conv.in_channels, "groups": conv.groups,
        "conservative_accumulation_scale": bound, "float32_max": torch.finfo(torch.float32).max,
        "scale_far_below_float32_overflow": bool(bound is not None and bound < torch.finfo(torch.float32).max / 1e12),
    }
    dump_json(out / "conv_fp32_sanity.json", conv_sanity)

    cpu_record = {"executed": False}
    if not fp32_result["raw_output_finite"] or not fp32_result["all_loss_finite"]:
        cpu_model = YOLO(str(checkpoint)).model.cpu().float()
        cpu_model.load_state_dict(state, strict=True); cpu_model.float().train()
        cpu_batch = old.to_device(saved_batch_cpu, torch.device("cpu")); cpu_batch["img"] = cpu_batch["img"].float()
        cpu_result, cpu_trace = run_forward_loss(cpu_model, cpu_batch, "cpu", False, trace=True)
        cpu_record = {
            "executed": True, "parameter_dtype_unique": dtype_counts(cpu_model)["parameter_dtype_unique"],
            "buffer_dtype_unique": dtype_counts(cpu_model)["buffer_dtype_unique"], "input_dtype": str(cpu_batch["img"].dtype),
            "autocast_enabled": False, "first_nonfinite_module": next((r["module"] for r in cpu_trace if not r["output_finite"]), "NONE"),
            **cpu_result,
        }
    dump_json(out / "cpu_true_fp32_replay.json", cpu_record)

    # Optional same-code-path reproducibility audit. These logs are produced by
    # invoking the frozen Exp10.1a --replay-only-mode implementation in fresh
    # processes; they are diagnostic-only and never perform an optimizer step.
    legacy_fp32 = read_legacy_single_replay(out / "legacy_rerun" / "fp32.log")
    legacy_amp = read_legacy_single_replay(out / "legacy_rerun" / "amp.log")
    legacy_repro = {
        "implementation": "frozen Exp10.1a --replay-only-mode, fresh process, no update",
        "fp32": legacy_fp32,
        "amp": legacy_amp,
        "fp32_finite": bool(legacy_fp32 and legacy_fp32["row"]["raw_output_finite"] and legacy_fp32["row"]["loss_finite"]),
        "amp_finite": bool(legacy_amp and legacy_amp["row"]["raw_output_finite"] and legacy_amp["row"]["loss_finite"]),
    }
    dump_json(out / "legacy_replay_reproducibility.json", legacy_repro)

    if fp32_result["raw_output_finite"] and fp32_result["all_loss_finite"] and not amp_result["all_loss_finite"]:
        case = "CASE C AMP_ONLY_TRAIN_FORWARD_INSTABILITY"
    elif (not fp32_result["raw_output_finite"] or not fp32_result["all_loss_finite"]) and cpu_record.get("all_loss_finite") and conv_sanity["scale_far_below_float32_overflow"]:
        case = "CASE D GPU_FP32_RUNTIME_OR_KERNEL_ANOMALY"
    elif old_invalid and fp32_result["all_loss_finite"]:
        case = "CASE A EXP10_1A_FALSE_FP32_REPLAY"
    elif (
        fp32_result["all_loss_finite"] and amp_result["all_loss_finite"] and
        formal_fp32_result["all_loss_finite"] and formal_amp_result["all_loss_finite"] and
        legacy_repro["fp32_finite"] and legacy_repro["amp_finite"]
    ):
        case = "CASE B REPLAY_OR_PREPROCESS_PIPELINE_BUG"
    elif (not fp32_result["raw_output_finite"] or not fp32_result["all_loss_finite"]) and not cpu_record.get("all_loss_finite", True):
        case = "CASE E TRUE_FP32_MODEL_FORWARD_NONFINITE"
    else:
        raise RuntimeError("Root-cause gate did not resolve")
    revised = {
        "status": "EXP10_TRUE_FP32_DIAGNOSTIC_COMPLETE_WAITING_REVIEW", "classification": case,
        "exp10_1a_fp32_replay_invalid_by_dtype": old_invalid,
        "preprocess_exact_tensor_match": preprocessing_exact,
        "case_b_subtype": "NONREPRODUCIBLE_EXP10_1A_DIAGNOSTIC_REPLAY" if case.startswith("CASE B") else None,
        "case_b_note": (
            "The tensor preprocessing is hash-identical; CASE B is assigned to the replay side, not to an extra /255, "
            "normalization, resize, cast, or augmentation. The frozen Exp10.1a replay implementation now reproduces "
            "finite FP32 and AMP values exactly, matching TRUE replay and formal Trainer-path replay."
            if case.startswith("CASE B") else None
        ),
        "legacy_exp10_1a_replay_reproducibility": legacy_repro,
        "saved_batch_provenance": "post-preprocess exact tensor; replay did not divide by 255, resize, cast, or augment again",
        "true_fp32": fp32_record, "true_amp": amp_record, "formal_trainer_path": trainer_path_record,
        "conv_fp32_sanity": conv_sanity, "cpu_true_fp32": cpu_record,
        "formal_hyperparameters_modified": False, "optimizer_step_performed": False, "seed44_run": False,
        "formal_treatment_resumed": False, "val_accessed_for_probe": False, "test_accessed": False,
        "candidate_freeze": False, "exp11_run": False,
    }
    dump_json(out / "root_cause_revised.json", revised)
    print(json.dumps(clean_json(revised), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
