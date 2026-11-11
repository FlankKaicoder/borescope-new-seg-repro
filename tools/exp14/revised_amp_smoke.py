"""Exp14-A3-D2 matched two-epoch Smoke using native GradScaler with initial scale 1."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import ultralytics

from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.diagnose_amp_first_batch import parameters_sha
from tools.exp14.freeze_core import COCO_SHA256
from tools.exp14.smoke_trainer import SmokeTrainer, finite, verify_payload, write_json


BASE_GIT = "78ff69dfe03adb3111f5e67505cc95cc6c081336"
GATE = "EXP14_A3_D2_PENDING_CHATGPT_FORMAL_TRAIN_GATE"
FROZEN_PROTOCOL_SHA256 = "fc280bf2afa0c1e9cd04c1cff193c67fcf7cb298570dee8633ec11c9d669ffca"
FROZEN_PAYLOAD_MANIFEST_SHA256 = "c6eba4d4d81b059f6ae72bc4059c327da2ce9ede5c731fbe792a78de53ec94e4"
NATIVE_DEFAULTS = {"growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000}


def gradient_population(parameters):
    grads = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    require(bool(grads), "No parameter gradients after backward")
    require(all(g.dtype == torch.float32 for g in grads), "Master gradients are not FP32")
    flat = torch.cat(grads)
    return {"tensors": len(grads), "elements": flat.numel(),
            "nonfinite": int((~torch.isfinite(flat)).sum()),
            "zero": flat.numel() - int(torch.count_nonzero(flat))}


def unscale_and_inspect(scaler, optimizer, parameters):
    scaled = gradient_population(parameters)
    scaler.unscale_(optimizer)
    checks = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    require(bool(checks), "GradScaler did not record native overflow detection")
    overflow = any(bool(check) for check in checks.values())
    unscaled = gradient_population(parameters)
    return scaled, unscaled, overflow


def clip_native_gradients(parameters):
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0, foreach=False, error_if_nonfinite=True)
    return float(norm)


def active_groups(optimizer):
    return [(group["param_group"], [p for p in group["params"] if p.requires_grad])
            for group in optimizer.param_groups if group["lr"] > 0]


def parameter_vector(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


class RevisedAmpSmokeTrainer(SmokeTrainer):
    def __init__(self, payload, weight, run, arm):
        self.native_step_calls = 0
        self.skipped_updates = 0
        self.completed_epochs = 0
        self.scale_history = []
        self.native_overflow = None
        super().__init__(payload, weight, run, arm, "train")

    def _setup_train(self):
        super()._setup_train()
        stock_state = self.scaler.state_dict()
        require(all(stock_state[k] == v for k, v in NATIVE_DEFAULTS.items()), "Native scale strategy differs")
        self.scaler = torch.amp.GradScaler("cuda", init_scale=1.0, enabled=self.amp)
        actual = self.scaler.state_dict()
        require(actual == {**stock_state, "scale": 1.0}, "Scale revision changed another scaler setting")
        require(type(self.optimizer) is torch.optim.AdamW
                and not getattr(self.optimizer, "_step_supports_amp_scaling", False),
                "Expected ordinary native AdamW step semantics")
        self.initial_parameters_sha = parameters_sha(self.model)
        self.event("amp_protocol", scaler_state=actual, old_initial_scale=stock_state["scale"],
                   initial_parameters_sha256=self.initial_parameters_sha,
                   model_optimizer_native_calls=self.native_step_calls)

    def run_callbacks(self, name):
        super().run_callbacks(name)
        if name == "on_train_epoch_end":
            self.completed_epochs += 1

    def before_step(self, optimizer, args, kwargs):
        require(self.native_overflow is False, "Native optimizer attempted to step on a detected overflow")
        self.native_step_calls += 1
        self.active_parameter_snapshots = [
            (name, params, parameter_vector(params))
            for name, params in active_groups(optimizer) if params
        ]
        require(bool(self.active_parameter_snapshots), "No trainable parameter group with nonzero LR")

    def after_step(self, optimizer, args, kwargs):
        parameters = list(self.model.parameters())
        flat = parameter_vector(parameters)
        require(flat.dtype == torch.float32 and finite(flat), "Master parameters changed dtype or became non-finite")
        changes = []
        for name, params, previous in self.active_parameter_snapshots:
            current = parameter_vector(params)
            delta = (current - previous).abs()
            changes.append({"group": name, "elements": delta.numel(), "changed": int(torch.count_nonzero(delta)),
                            "max_abs_delta": float(delta.max())})
        self.active_parameter_snapshots = []
        require(any(group["changed"] for group in changes), "Native AdamW step changed no active parameters")
        self.step_changes = changes
        self.update_count += 1

    def optimizer_step(self):
        require(finite(self.loss) and finite(self.loss_items) and self.accumulate == 1,
                "Non-finite loss or changed accumulation")
        self.optimizer_calls += 1
        parameters = list(self.model.parameters())
        step_before, update_before = self.native_step_calls, self.update_count
        scale_before = self.scaler.get_scale()
        scaled, unscaled, self.native_overflow = unscale_and_inspect(self.scaler, self.optimizer, parameters)
        self.event("backward_complete", epoch=self.epoch + 1, batch=self.current_event["batch"],
                   scaled_gradients=scaled, unscaled_gradients=unscaled,
                   native_overflow=self.native_overflow, scale_before=scale_before)
        norm = None
        if not self.native_overflow:
            require(not scaled["nonfinite"] and not unscaled["nonfinite"], "Unhandled non-finite gradient")
            norm = clip_native_gradients(parameters)
            require(math.isfinite(norm) and norm > 0, "Gradient norm is non-finite or zero")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        called = self.native_step_calls == step_before + 1
        updated = self.update_count == update_before + 1
        self.scale_history.append([scale_before, scale_after])
        self.event("native_step_result", epoch=self.epoch + 1, batch=self.current_event["batch"],
                   overflow=self.native_overflow, native_step_called=called, parameter_changed=updated,
                   scale_before=scale_before, scale_after=scale_after)
        if self.native_overflow:
            require(not called and not updated and scale_after == scale_before * NATIVE_DEFAULTS["backoff_factor"],
                    "Native overflow handling differed")
            self.skipped_updates += 1
            self.optimizer.zero_grad()
            raise ValueError("Native GradScaler skipped an update; matched Smoke acceptance stopped")
        require(called and updated, "Finite gradients did not produce an effective optimizer update")
        self.event("optimizer_update", update=self.update_count, epoch=self.epoch + 1,
                   batch=self.current_event["batch"], gradient_norm=norm, parameter_changed=True,
                   parameter_groups=self.step_changes, amp=self.amp, scale=scale_after,
                   accumulate=self.accumulate, lrs=[g["lr"] for g in self.optimizer.param_groups])
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


def run(args):
    payload = args.payload.resolve(strict=True)
    output = args.output.resolve(strict=True)
    require(ultralytics.__version__ == "8.4.117" and torch.__version__ == "2.8.0+cu128"
            and torch.cuda.is_available(), "Frozen training environment unavailable")
    require(sha256_file(args.weight) == COCO_SHA256, "Frozen COCO weight mismatch")
    require(sha256_file(payload / "training_protocol.json") == FROZEN_PROTOCOL_SHA256
            and sha256_file(payload / "transport_manifest.json") == FROZEN_PAYLOAD_MANIFEST_SHA256,
            "Frozen protocol or transport identity differs")
    actual_git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(actual_git == BASE_GIT, "Server checkout differs from approved D1 baseline")
    manifest = verify_payload(payload)
    preflights = [json.loads((args.a3 / f"preflight_{arm}/summary.json").read_text()) for arm in ("e0", "e1")]
    require(all(row["status"] == "PASS_DIAGNOSTIC_ONLY" for row in preflights)
            and preflights[0]["initial_state_sha256"] == preflights[1]["initial_state_sha256"],
            "Historical preflight initialization differs")
    if args.arm == "e1":
        e0 = json.loads((output / "train_e0/summary.json").read_text())
        require(e0["status"] == "PASS_DIAGNOSTIC_ONLY" and e0["effective_updates"] == 44,
                "E0 Smoke has not passed")
    run_dir = output / f"train_{args.arm}"
    run_dir.mkdir()
    environment = {
        "diagnostic_only": True, "git_head": actual_git, "a2_source_git": manifest["source_git"],
        "d2_adapter_sha256": sha256_file(Path(__file__)), "python": sys.version.split()[0],
        "torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "disk_free_gib": round(shutil.disk_usage(output).free / 2**30, 2),
        "driver_gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            text=True).strip(),
        "weight_sha256": COCO_SHA256, "payload_files": len(manifest["files"]),
        "payload_manifest_sha256": FROZEN_PAYLOAD_MANIFEST_SHA256, "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "ui_risk_flags": manifest["ui_risks"],
    }
    write_json(run_dir / "environment.json", environment)
    trainer = None
    error = None
    try:
        trainer = RevisedAmpSmokeTrainer(payload, args.weight, run_dir, args.arm)
        trainer.expected_init_sha = preflights[0]["initial_state_sha256"]
        trainer.train()
        require(len(trainer.observed) == trainer.optimizer_calls == trainer.native_step_calls == trainer.update_count == 44
                and trainer.completed_epochs == 2 and trainer.skipped_updates == 0, "Smoke completed with a budget mismatch")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    verify_payload(payload)
    observed = trainer.observed if trainer else []
    summary = {
        "status": "STOP_DIAGNOSTIC_ONLY" if error else "PASS_DIAGNOSTIC_ONLY",
        "reason": error, "diagnostic_only": True, "gate": GATE, "phase": "train", "arm": args.arm,
        "completed_epochs": getattr(trainer, "completed_epochs", 0), "batches": len(observed),
        "samples": sum(e["samples"] for e in observed), "epoch_counts": dict(Counter(e["epoch"] for e in observed)),
        "optimizer_calls": getattr(trainer, "native_step_calls", 0),
        "optimizer_attempts": getattr(trainer, "optimizer_calls", 0),
        "effective_updates": getattr(trainer, "update_count", 0), "scaler_skips": getattr(trainer, "skipped_updates", 0),
        "scale_history": getattr(trainer, "scale_history", []), "forward_checks": getattr(trainer, "forward_count", 0),
        "amp": getattr(trainer, "amp", None) if isinstance(getattr(trainer, "amp", None), bool) else None,
        "master_dtype": "float32" if hasattr(trainer, "initial_state_sha") else None,
        "initial_state_sha256": getattr(trainer, "initial_state_sha", None),
        "initial_parameters_sha256": getattr(trainer, "initial_parameters_sha", None),
        "final_parameters_sha256": parameters_sha(trainer.model) if hasattr(trainer, "initial_parameters_sha") else None,
        "actual_validation_runs": 0, "peak_memory_allocated": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved": torch.cuda.max_memory_reserved(),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--a3", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("e0", "e1"), required=True)
    run(parser.parse_args())
