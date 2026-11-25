"""Owned synthetic CPU recovery commits, immutable and append-only.

No historical checkpoint is opened. Every load requires a matching owner marker,
completed commit receipt, exact file SHA and EXP18 synthetic payload binding.
Native model checkpoint loading remains a separately gated future capability.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from .bn import cpu_rng_state, restore_cpu_rng

SCHEMA = "EXP18_FULL_RECOVERY_V1"


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def state_hash(value):
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            require(item.device.type == "cpu", "Current checkpoint stage is CPU-only")
            item = item.detach().contiguous()
            result.update(str(item.dtype).encode())
            result.update(str(tuple(item.shape)).encode())
            result.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str(item.dtype).encode()); result.update(str(item.shape).encode()); result.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update((type(item).__name__ + ":" + repr(item)).encode())
    visit(value)
    return result.hexdigest()


def _tree(value, *, native=False):
    if isinstance(value, torch.Tensor):
        require(native or value.device.type == "cpu", "Synthetic checkpoint accepts only CPU tensors")
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor")
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _tree(v, native=native) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree(v, native=native) for v in value)
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite checkpoint scalar")
    return copy.deepcopy(value)


def _rng_for_safe_pickle(*, native=False):
    rng = cpu_rng_state()
    np_state = rng["numpy"]
    rng["numpy"] = (np_state[0], torch.tensor(np_state[1].astype(np.int64)), *np_state[2:])
    if native:
        # Reached only after the separate native protocol gate, never CPU stage.
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _restore_rng(record):
    rng = copy.deepcopy(record)
    state = rng["numpy"]
    rng["numpy"] = (state[0], state[1].numpy().astype(np.uint32), *state[2:])
    restore_cpu_rng(rng)
    if "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])


def backend_state():
    """Read scalar backend policies; this does not discover/initialize CUDA."""
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_allow_fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "matmul_allow_bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def restore_backend(state):
    torch.use_deterministic_algorithms(state["deterministic_algorithms"], warn_only=state["deterministic_warn_only"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(state["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_allow_tf32"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = state["matmul_allow_fp16_reduction"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = state["matmul_allow_bf16_reduction"]


def optimizer_names(model, optimizer):
    by_id = {id(p): name for name, p in model.named_parameters()}
    names = [[by_id[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    flat = [name for group in names for name in group]
    require(len(flat) == len(set(flat)) and set(flat) >= {n for n, p in model.named_parameters() if p.requires_grad},
            "Optimizer trainable names missing/repeated")
    return names


def capture_training_state(model, optimizer, scheduler, ema, *, bindings, position,
                           sampler_state, head_active_updates, scaler=None, counters=None, extra=None,
                           model_origin="synthetic", native_protocol=None):
    require(model_origin in ("synthetic", "native"), "Unknown model origin")
    native = model_origin == "native"
    if native:
        from .protocol import require_operation
        require_operation("train", native_protocol)
    else:
        require(getattr(model, "exp18_synthetic", False) is True, "Synthetic capture requires an explicitly synthetic model")
    require(all(p.grad is None for p in model.parameters()), "Checkpoint must follow gradient clearing")
    record = {"schema": SCHEMA, "artifact_kind": "native_full" if native else "synthetic_cpu", "bindings": _tree(bindings),
              "position": _tree(position), "sampler": _tree(sampler_state),
              "head_active_updates": head_active_updates,
              "model_config": _tree(getattr(model, "yaml", None)),
              "class_names": _tree(getattr(model, "names", None)),
              "state": _tree(model.state_dict(), native=native), "ema": _tree(ema.state_dict(), native=native),
              "optimizer": _tree(optimizer.state_dict(), native=native), "optimizer_names": optimizer_names(model, optimizer),
              "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
              "scheduler": _tree(scheduler.state_dict()), "scaler": None if scaler is None else _tree(scaler.state_dict()),
              "rng": _rng_for_safe_pickle(native=native), "counters": _tree(counters or {}), "extra": _tree(extra or {}),
              "backend": backend_state(),
              "training_modes": {name: module.training for name, module in model.named_modules()}}
    validate_payload(record, bindings, native_protocol=native_protocol)
    return record


def validate_payload(record, bindings, *, native_protocol=None):
    require(record.get("schema") == SCHEMA and record.get("artifact_kind") in ("synthetic_cpu", "native_full"), "Recovery schema rejected")
    if record["artifact_kind"] == "native_full":
        from .protocol import require_operation
        require_operation("train", native_protocol)
    require(record.get("bindings") == bindings, "Recovery code/protocol/matching binding changed")
    for key in ("code_sha256", "protocol_sha256", "matching_sha256"):
        require(isinstance(bindings.get(key), str) and len(bindings[key]) == 64, "Missing recovery binding: " + key)
    update = record["position"]["global_update"]
    head = record["head_active_updates"]
    require(type(update) is int and type(head) is int and 0 <= head <= update, "Invalid recovery clocks")
    require(record["ema"]["updates"] == update and record["ema"]["head_active_updates"] == head, "EMA clocks disagree")
    position, sampler = record["position"], record["sampler"]
    if "epoch" in position and "batch_position" in position:
        epoch, batch = position["epoch"], position["batch_position"]
        require(type(epoch) is int and type(batch) is int and epoch >= 0 and 0 <= batch <= 21
                and epoch * 21 + batch == update, "Recovery epoch/batch position disagrees with global update")
        require(sampler.get("next_batch") == batch and sampler.get("epoch", epoch) == epoch,
                "Recovery sampler cursor disagrees with position")
    counters = record.get("counters", {})
    require(counters.get("optimizer_attempts", update) == update and counters.get("amp_skips", 0) == 0,
            "Recovery optimizer attempt/AMP counters disagree")
    parameter_states = record["optimizer"]["state"]
    trainable = set(record["trainable_parameter_names"])
    flat_names = [name for group in record["optimizer_names"] for name in group]
    require(len(flat_names) == len(set(flat_names)) and trainable <= set(flat_names), "Recovery optimizer trainable names omitted/repeated")
    for group, names in zip(record["optimizer"]["param_groups"], record["optimizer_names"], strict=True):
        for identifier, name in zip(group["params"], names, strict=True):
            count = head if name.startswith("state_head.") else update
            slot = parameter_states.get(identifier)
            if name not in trainable:
                require(name in record["state"] and slot is None, "Frozen optimizer member has unexpected state")
                continue
            if count == 0:
                require(slot is None, "Unused optimizer parameter has a populated slot")
            else:
                require(slot is not None and float(slot["step"]) == count, "AdamW parameter clock differs: " + name)
                require(set(slot) >= {"step", "exp_avg", "exp_avg_sq"}, "Incomplete AdamW slot")
                require(name in record["state"] and slot["exp_avg"].shape == slot["exp_avg_sq"].shape == record["state"][name].shape,
                        "AdamW slot shape differs from master parameter")
    _tree(record)
    return True


def restore_training_state(model, optimizer, scheduler, ema, record, *, bindings, scaler=None, native_protocol=None):
    if record.get("artifact_kind") != "native_full":
        require(getattr(model, "exp18_synthetic", False) is True, "Synthetic recovery requires an explicitly synthetic model")
    validate_payload(record, bindings, native_protocol=native_protocol)
    require(optimizer_names(model, optimizer) == record["optimizer_names"], "Optimizer names/order changed")
    require([name for name, parameter in model.named_parameters() if parameter.requires_grad] == record["trainable_parameter_names"],
            "Recovery trainability changed")
    require(state_hash(getattr(model, "yaml", None)) == state_hash(record["model_config"])
            and state_hash(getattr(model, "names", None)) == state_hash(record["class_names"]),
            "Recovery architecture/class metadata changed")
    require((scaler is None) == (record["scaler"] is None), "Scaler presence changed")
    require(set(dict(model.named_modules())) == set(record["training_modes"]), "Recovery model modules changed")
    model.load_state_dict(record["state"], strict=True)
    optimizer.load_state_dict(record["optimizer"])
    scheduler.load_state_dict(record["scheduler"])
    ema.load_state_dict(record["ema"])
    if scaler is not None:
        scaler.load_state_dict(record["scaler"])
    for name, module in model.named_modules():
        module.training = record["training_modes"][name]
    _restore_rng(record["rng"])
    restore_backend(record["backend"])
    return {"position": copy.deepcopy(record["position"]), "sampler": copy.deepcopy(record["sampler"]),
            "head_active_updates": record["head_active_updates"], "counters": copy.deepcopy(record["counters"]),
            "extra": copy.deepcopy(record["extra"])}


def _exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


class ImmutableCheckpointStore:
    def __init__(self, directory, bindings, *, create=False, native_protocol=None, max_bytes=None):
        self.directory = Path(directory).absolute()
        require(not self.directory.is_symlink(), "Checkpoint root is a symlink")
        self.bindings = copy.deepcopy(bindings)
        self.native_protocol = copy.deepcopy(native_protocol)
        require(max_bytes is None or type(max_bytes) is int and max_bytes > 0, "Invalid measured checkpoint byte limit")
        self.max_bytes = max_bytes
        marker = self.directory / "owner.json"
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
            _exclusive_json(marker, {"schema": SCHEMA, "owner_id": uuid.uuid4().hex,
                                     "bindings_sha256": state_hash(bindings)})
        require(marker.is_file() and not marker.is_symlink(), "Owned synthetic checkpoint marker missing")
        self.owner = json.loads(marker.read_text(encoding="utf-8"))
        require(self.owner["schema"] == SCHEMA and self.owner["bindings_sha256"] == state_hash(bindings), "Checkpoint owner binding mismatch")

    def save(self, record):
        validate_payload(record, self.bindings, native_protocol=self.native_protocol)
        update = record["position"]["global_update"]
        revision = record["position"].get("revision", 0)
        require(type(revision) is int and revision in (0, 1), "Only update/epoch-event immutable revisions are supported")
        path = self.directory / f"update_{update:06d}_r{revision}.pt"
        receipt_path = path.with_suffix(".commit.json")
        require(not path.exists() and not receipt_path.exists(), "Immutable commit already exists")
        with path.open("xb") as stream:
            torch.save(record, stream); stream.flush(); os.fsync(stream.fileno())
        require(self.max_bytes is None or path.stat().st_size <= self.max_bytes,
                "Checkpoint exceeds measured storage gate; preserve uncommitted file")
        digest = file_sha(path)
        from .guards import active_guard
        guard = active_guard()
        if guard is not None:
            require(record["artifact_kind"] == "synthetic_cpu", "CPU guard rejects native checkpoint verification")
            guard.allow_cpu_load(path, digest, kind="synthetic_CPU_checkpoint")
        # Read every serialized tensor before publishing the commit receipt.
        check = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(check, self.bindings, native_protocol=self.native_protocol)
        require(state_hash(check) == state_hash(record), "Serialized checkpoint failed complete readback; preserve orphan")
        receipt = {"schema": SCHEMA, "owner_id": self.owner["owner_id"], "file": path.name,
                   "bytes": path.stat().st_size, "sha256": digest, "state_sha256": state_hash(record),
                   "global_update": update, "revision": revision, "artifact_kind": record["artifact_kind"],
                   "bindings_sha256": state_hash(self.bindings)}
        _exclusive_json(receipt_path, receipt)
        return receipt

    def load(self, receipt):
        require(receipt.get("schema") == SCHEMA and receipt.get("owner_id") == self.owner["owner_id"]
                and receipt.get("bindings_sha256") == state_hash(self.bindings), "Unowned checkpoint receipt")
        expected_name = f"update_{receipt['global_update']:06d}_r{receipt['revision']}.pt"
        require(receipt["file"] == expected_name, "Unsafe checkpoint member path")
        path = self.directory / expected_name
        require(not path.is_symlink() and path.is_file(), "Owned checkpoint path changed")
        commit = path.with_suffix(".commit.json")
        require(commit.is_file() and not commit.is_symlink() and json.loads(commit.read_text(encoding="utf-8")) == receipt,
                "Completed immutable commit receipt missing/changed")
        require(path.stat().st_size == receipt["bytes"] and file_sha(path) == receipt["sha256"], "Synthetic checkpoint bytes changed")
        if receipt["artifact_kind"] == "native_full":
            from .protocol import require_operation
            require_operation("train", self.native_protocol)
        else:
            require(receipt["artifact_kind"] == "synthetic_cpu", "Unknown checkpoint artifact kind")
        from .guards import active_guard
        guard = active_guard()
        if guard is not None:
            require(receipt["artifact_kind"] == "synthetic_cpu", "CPU guard rejects native model checkpoint")
            guard.allow_cpu_load(path, receipt["sha256"], kind="synthetic_CPU_checkpoint")
        # No caller-controlled path reaches torch.load before ownership+SHA checks.
        record = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(record, self.bindings, native_protocol=self.native_protocol)
        require(state_hash(record) == receipt["state_sha256"], "Synthetic checkpoint state changed")
        return record
