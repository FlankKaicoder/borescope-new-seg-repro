#!/usr/bin/env python3
"""Trace the first non-finite leaf-module output for one validation batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics.utils.torch_utils import autocast, select_device

from exp02_2a_val_loss_probe import (
    flatten_tensors,
    load_model,
    make_validator,
    tensor_stats,
)


def compact_stats(obj: Any) -> list[dict[str, Any]]:
    return [{"name": name, **tensor_stats(value)} for name, value in flatten_tensors(obj)]


def all_finite(obj: Any) -> bool:
    tensors = flatten_tensors(obj)
    return all(bool(torch.isfinite(value).all()) for _, value in tensors)


def parameter_stats(module: torch.nn.Module) -> dict[str, Any]:
    result = {}
    for name, value in module.named_parameters(recurse=False):
        result[name] = tensor_stats(value)
    for name, value in module.named_buffers(recurse=False):
        if torch.is_tensor(value):
            result[f"buffer:{name}"] = tensor_stats(value)
    return result


def trace(model: torch.nn.Module, batch: dict[str, Any], amp_on: bool) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    handles = []
    names = {module: name for name, module in model.named_modules()}
    attention_input: list[torch.Tensor] = []

    def pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        module.__exp02_input_stats = compact_stats(inputs)  # type: ignore[attr-defined]

    def post_hook(module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        output_finite = all_finite(output)
        events.append(
            {
                "sequence": len(events),
                "module": names[module],
                "module_class": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "input_stats": getattr(module, "__exp02_input_stats", []),
                "output_stats": compact_stats(output),
                "output_finite": output_finite,
                "parameter_stats": parameter_stats(module),
            }
        )

    for module in model.modules():
        if not any(module.children()):
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
    attention = dict(model.named_modules())["model.10.m.0.attn"]
    handles.append(attention.register_forward_pre_hook(lambda _module, inputs: attention_input.append(inputs[0].detach().clone())))
    try:
        with torch.inference_mode(), autocast(amp_on, device=batch["img"].device.type):
            output = model(batch["img"])
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode(), autocast(amp_on, device=batch["img"].device.type):
        x = attention_input[0]
        b, c, h, w = x.shape
        n = h * w
        qkv = attention.qkv(x)
        q, k, v = qkv.view(b, attention.num_heads, attention.key_dim * 2 + attention.head_dim, n).split(
            [attention.key_dim, attention.key_dim, attention.head_dim], dim=2
        )
        scaled_q = q * attention.scale
        logits = scaled_q.transpose(-2, -1) @ k
        weights = logits.softmax(dim=-1)
        attended = v @ weights.transpose(-2, -1)
        positional = attention.pe(v.reshape(b, c, h, w))
        added = attended.view(b, c, h, w) + positional
        projected = attention.proj(added)
    attention_ops = []
    for name, value in (
        ("attention_input", x),
        ("qkv_conv", qkv),
        ("q", q),
        ("k", k),
        ("v", v),
        ("scaled_q", scaled_q),
        ("qk_matmul_logits", logits),
        ("softmax", weights),
        ("v_attention_matmul", attended),
        ("positional_encoding", positional),
        ("attention_plus_positional", added),
        ("projection", projected),
    ):
        stats = tensor_stats(value)
        attention_ops.append({"operation": name, "finite": stats["finite_ratio"] == 1.0, **stats})
    first_attention_nonfinite = next((op for op in attention_ops if not op["finite"]), None)
    first_index = next((i for i, event in enumerate(events) if not event["output_finite"]), None)
    lo = max(0, first_index - 3) if first_index is not None else max(0, len(events) - 3)
    hi = min(len(events), first_index + 4) if first_index is not None else len(events)
    return {
        "amp": amp_on,
        "image_stems": [Path(x).stem for x in batch["im_file"]],
        "input_stats": tensor_stats(batch["img"]),
        "leaf_module_events": len(events),
        "first_nonfinite_event_index": first_index,
        "first_nonfinite_event": events[first_index] if first_index is not None else None,
        "event_window": events[lo:hi],
        "raw_output_finite": all_finite(output),
        "raw_output_stats": compact_stats(output),
        "attention_operation_trace": attention_ops,
        "first_nonfinite_attention_operation": first_attention_nonfinite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = select_device("0")
    result = {"test_accessed": False, "checkpoint": str(args.checkpoint), "modes": {}}
    for mode, amp_on in (("validator_equivalent_amp", True), ("fp32", False)):
        model = load_model(args.checkpoint, device)
        validator = make_validator(args.data, device, amp_on)
        batch = validator.preprocess(next(iter(validator.dataloader)))
        result["modes"][mode] = trace(model, batch, amp_on)
        del model, validator
        torch.cuda.empty_cache()
    (args.output / "forward_first_nonfinite_trace.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "status": "PASS",
        "test_accessed": False,
        "amp_first_nonfinite_module": result["modes"]["validator_equivalent_amp"]["first_nonfinite_event"],
        "fp32_first_nonfinite_module": result["modes"]["fp32"]["first_nonfinite_event"],
    }
    (args.output / "forward_trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
