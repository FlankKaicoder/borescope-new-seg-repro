"""Frozen VAL policy and isolated callbacks for a separately authorized J2 run.

Importing this module performs no data access or model computation. The real
adapter reuses the historical evaluator unchanged, and cannot run without an
explicit protocol-, path- and budget-bound validation scope.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

VAL_EPOCHS = tuple(range(5, 101, 5))
ROLES = ("master", "ema")
PASSES = ("standard", "fixed")
MAX_PASSES = 80  # 20 events x 2 separately reported model states x 2 native passes.
VAL_FILES_SHA256 = "50e4561534750edc5f1d739c4c8c24591edb6e75a9d70f8d1a640e2bd139c467"
METRIC_KEYS = tuple(f"metrics/{name}({kind})" for kind in ("B", "M")
                    for name in ("precision", "recall", "mAP50", "mAP50-95"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_epochs():
    return VAL_EPOCHS


def due(epoch, completed_updates, batch_position=21):
    require(type(epoch) is int and 1 <= epoch <= 100, "Invalid completed epoch")
    require(type(completed_updates) is int and completed_updates == epoch * 21
            and batch_position == 21, "VAL may run only at a complete epoch boundary")
    return epoch in VAL_EPOCHS


def native_fitness(standard):
    """Ultralytics 8.4.117: seg AP50:95 + box AP50:95, never Mask AP alone.

    Nonfinite metrics are rejected instead of relying on native nan_to_num;
    the experiment's numerical hard gate is stricter than that native fallback.
    """
    metrics = standard["overall"]
    require(all(k in metrics and math.isfinite(float(metrics[k])) for k in METRIC_KEYS),
            "Missing or nonfinite native standard metrics")
    value = float(metrics["metrics/mAP50-95(M)"]) + float(metrics["metrics/mAP50-95(B)"])
    if "fitness" in metrics:
        require(math.isfinite(float(metrics["fitness"]))
                and math.isclose(float(metrics["fitness"]), value, rel_tol=0, abs_tol=1e-12),
                "Native fitness is not pinned segmentation-plus-box AP50:95")
    return value


def best_update(previous, *, epoch, role, master_sha256, ema_sha256, evaluation):
    """Choose latest tied native fitness; keep master/EMA selections independent.

    Native validate updates its numeric best only on strict improvement, while
    save_model writes best on equality. Thus a tied checkpoint is the latest.
    """
    require(epoch in VAL_EPOCHS and role in ROLES, "Best selection outside frozen VAL cadence/role")
    require(all(isinstance(s, str) and len(s) == 64 and set(s) <= set("0123456789abcdef")
                for s in (master_sha256, ema_sha256)), "Invalid bound checkpoint state hashes")
    fitness = native_fitness(evaluation["standard"])
    if previous is not None:
        require(previous["role"] == role and previous["epoch"] < epoch
                and math.isfinite(previous["fitness"]), "Best state order or role changed")
    selected = previous is None or fitness >= previous["fitness"]
    record = {"role": role, "epoch": epoch, "fitness": fitness,
              "mask_mAP50_95": float(evaluation["standard"]["overall"]["metrics/mAP50-95(M)"]),
              "master_state_sha256": master_sha256, "ema_state_sha256": ema_sha256,
              "evaluation": copy.deepcopy(evaluation), "rule": "LATEST_EQUAL_NATIVE_FITNESS"}
    return (record if selected else copy.deepcopy(previous)), selected


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state().clone(),
            "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _backend_state():
    return {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def _restore_backend(state):
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]


def _snapshot(obj):
    """Explicit state-dict protocol, plus model flags/gradients and ModelEMA count."""
    if isinstance(obj, torch.nn.Module):
        return {"kind": "model", "state": copy.deepcopy(obj.state_dict()),
                "modes": [m.training for m in obj.modules()],
                "gradients": [None if p.grad is None else _clone(p.grad) for p in obj.parameters()]}
    if hasattr(obj, "ema") and isinstance(obj.ema, torch.nn.Module):
        return {"kind": "ema", "model": _snapshot(obj.ema),
                "attributes": copy.deepcopy({k: v for k, v in vars(obj).items() if k != "ema"})}
    if isinstance(obj, torch.amp.GradScaler):
        # state_dict omits per-optimizer READY/UNSCALED/STEPPED stage and lazy
        # tensor initialization. Those states affect the next real update too.
        return {"kind": "gradscaler", "attributes": copy.deepcopy(vars(obj))}
    if isinstance(obj, torch.optim.Optimizer):
        return {"kind": "optimizer", "state": copy.deepcopy(obj.state_dict()),
                "defaults": copy.deepcopy(obj.defaults)}
    require(hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict"),
            "Protected live component must expose strict state_dict/load_state_dict")
    return {"kind": "state_dict", "state": copy.deepcopy(obj.state_dict())}


def _restore(obj, state):
    if state["kind"] == "ema":
        _restore(obj.ema, state["model"])
        for key in list(vars(obj)):
            if key != "ema":
                delattr(obj, key)
        vars(obj).update(copy.deepcopy(state["attributes"]))
    elif state["kind"] == "gradscaler":
        vars(obj).clear()
        vars(obj).update(copy.deepcopy(state["attributes"]))
    elif state["kind"] == "optimizer":
        obj.load_state_dict(state["state"])
        obj.defaults = copy.deepcopy(state["defaults"])
    elif state["kind"] == "model":
        obj.load_state_dict(state["state"], strict=True)
        for module, training in zip(obj.modules(), state["modes"], strict=True):
            module.training = training
        for parameter, gradient in zip(obj.parameters(), state["gradients"], strict=True):
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    else:
        obj.load_state_dict(state["state"])


class IsolationViolation(RuntimeError):
    """Unexpected mutation is restored where possible, and remains a hard failure."""


def run_isolated(callback, source_model, *, model, ema=None, optimizer=None,
                 scheduler=None, scaler=None, sampler=None, allow_grad=False, extras=None):
    """Invoke callback(clone) without changing the protected training trajectory.

    A clone may change BN/modes/gradients during the callback. Live state must
    never change; any mutation causes a hard gate even after restoration. Global
    callback RNG and backend settings are restored intentionally, including on
    exceptional exit. N's separate no-RNG-consumption rule remains unchanged.
    """
    objects = {"model": model, "ema": ema, "optimizer": optimizer, "scheduler": scheduler,
               "scaler": scaler, "sampler": sampler, **(extras or {})}
    objects = {k: v for k, v in objects.items() if v is not None}
    states = {k: _snapshot(v) for k, v in objects.items()}
    rng, backend = _rng(), _backend_state()
    hook = None
    try:
        if optimizer is not None:
            def reject(*_args):
                raise IsolationViolation("A validation/diagnostic callback attempted a live optimizer step")
            hook = optimizer.register_step_pre_hook(reject)
        clone = copy.deepcopy(source_model)
        clone.eval() if not allow_grad else clone.train()
        with torch.set_grad_enabled(allow_grad):
            result = callback(clone)
    finally:
        if hook is not None:
            hook.remove()
        changed = []
        try:
            changed = [k for k, obj in objects.items() if not _equal(_snapshot(obj), states[k])]
            for key in changed:
                _restore(objects[key], states[key])
        finally:
            _restore_rng(rng)
            _restore_backend(backend)
        require(_equal(_rng(), rng), "Failed to restore callback RNG")
        if changed:
            raise IsolationViolation("Validation/diagnostic changed live training state: " + ", ".join(changed))
    return result


def validate_metadata(path, expected_sha256=VAL_FILES_SHA256):
    """Metadata-only preflight; never opens an image or a segmentation label."""
    path = Path(path)
    require(sha256_file(path) == expected_sha256, "Frozen VAL metadata SHA256 differs")
    supplied = json.loads(path.read_text(encoding="utf-8"))
    entries = supplied["entries"]
    require(len(entries) == 154, "Frozen VAL metadata must contain 154 entries")
    images = set()
    for entry in entries:
        for kind, first in (("image", "images"), ("label", "labels")):
            item = entry[kind]
            name = Path(item["path"])
            require(not name.is_absolute() and name.parts[:2] == (first, "val") and len(name.parts) == 3
                    and ".." not in name.parts and len(item["sha256"]) == 64,
                    "VAL metadata contains another split or an invalid member")
        images.add(entry["image"]["path"])
    require(len(images) == 154, "Duplicate frozen VAL metadata member")
    return {"images": 154, "expected_instances": 296, "val_files_sha256": expected_sha256}


class FrozenValAdapter:
    """Lazy, explicitly authorized wrapper around the unchanged historical evaluator.

    A future formal authorization must include this exact VAL scope. The shared
    ledger_root is outside per-epoch outputs; duplicate or failed pass claims
    cannot be retried. G1 only tests refusal and synthetic callbacks.
    """
    authorization_stage = "EXP17_G1_FROZEN_VAL_AUTHORIZATION"

    def __init__(self, *, authorization, protocol_sha256, val_copy, val_root, output, ledger_root):
        self.protocol_sha256 = protocol_sha256
        self.val_copy, self.val_root = Path(val_copy).resolve(), Path(val_root).resolve()
        self.output, self.ledger_root = Path(output).resolve(), Path(ledger_root).resolve()
        expected = {"stage": self.authorization_stage, "authorized": True,
                    "protocol_sha256": protocol_sha256, "val_files_sha256": VAL_FILES_SHA256,
                    "validation_epochs": list(VAL_EPOCHS), "roles": list(ROLES), "passes": list(PASSES),
                    "maximum_native_passes": MAX_PASSES, "output": str(self.output),
                    "ledger_root": str(self.ledger_root), "val_copy": str(self.val_copy), "val_root": str(self.val_root)}
        require(authorization == expected and authorization.get("authorized") is True,
                "Explicit path-, protocol- and 80-pass-bound VAL authorization required")
        require(not any(self.output.is_relative_to(p) or self.ledger_root.is_relative_to(p)
                        for p in (self.val_copy, self.val_root)), "VAL output/ledger intersects frozen data")
        self.authorization = copy.deepcopy(authorization)

    def _claim(self, epoch, role, mode):
        require(epoch in VAL_EPOCHS and role in ROLES and mode in PASSES, "Unexpected native VAL pass")
        self.ledger_root.mkdir(parents=True, exist_ok=True)
        claim = self.ledger_root / f"epoch{epoch:03d}_{role}_{mode}.json"
        record = {"epoch": epoch, "role": role, "mode": mode, "protocol_sha256": self.protocol_sha256,
                  "state": "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"}
        with claim.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())

    def evaluate(self, native_model, *, epoch, role):
        """Future runtime only; each role uses fresh master/EMA-native copies per pass."""
        require(epoch in VAL_EPOCHS and role in ROLES, "VAL cadence/role differs")
        import ultralytics
        from tools.exp14 import a11_full_eval as native
        from tools.exp15.jd import read_val_copy
        require(ultralytics.__version__ == "8.4.117", "VAL framework version differs")
        require(not hasattr(native_model, "region_projection")
                and native_model.model[-1].nc == 7, "VAL requires strictly stripped native seven-class model")
        require(all(not v.is_floating_point() or (v.dtype == torch.float32 and bool(torch.isfinite(v).all()))
                    for v in native_model.state_dict().values()), "VAL source is not a finite FP32 master/EMA state")
        validate_metadata(self.val_copy / "val_files.json")
        config, val_dir, labels = read_val_copy(SimpleNamespace(output=self.val_copy, val=self.val_root))
        results = {}
        self.output.mkdir(parents=True, exist_ok=True)
        for mode, cutoff in (("standard", native.STANDARD_CONF), ("fixed", native.FIXED_CONF)):
            self._claim(epoch, role, mode)
            pass_model = copy.deepcopy(native_model).float().eval()
            path = self.output / f"epoch{epoch:03d}_{role}_{mode}"
            results[mode] = native.evaluate_master(pass_model, config, val_dir, labels, path, cutoff)
        read_val_copy(SimpleNamespace(output=self.val_copy, val=self.val_root))
        results["fitness"] = native_fitness(results["standard"])
        results["role"] = role
        return results
