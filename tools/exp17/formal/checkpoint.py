"""Full-state recovery and atomic two-slot storage for a separately gated run.

An optimizer update becomes recoverable only after the latest pointer is
durably committed. The execution ledger must hard-stop on an attempted but
uncommitted update: physical process death inside that window is ambiguous,
and must never be hidden by replaying an optimizer update automatically.
"""
import copy
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

import torch

from tools.exp17.formal.rng import capture_rng, restore_rng, rng_equal

SCHEMA = "EXP17_G1_FULL_RECOVERY_V1"
POINTER_SCHEMA = "EXP17_G1_ATOMIC_RECOVERY_POINTER_V1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    require(value is None or isinstance(value, (str, int, float, bool)), "Recovery metadata must be plain portable values")
    return copy.deepcopy(value)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def state_hash(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str):
                h.update(repr(key).encode()); visit(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for item in v: visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def optimizer_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    result = [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    flat = [name for group in result for name in group]
    require(len(flat) == len(set(flat)), "Optimizer repeats a parameter")
    require(set(flat) >= {n for n, p in model.named_parameters() if p.requires_grad}, "Trainable optimizer parameter is missing")
    return result


def _ema_model(trainer):
    return trainer.ema.ema


def capture_training_state(trainer, *, bindings, position, sampler_state, extra=None):
    """Capture after zero_grad, at a committed-update boundary; no RNG draws."""
    require(set(position) >= {"epoch", "batch_position", "global_update"}, "Missing update position")
    require(all(isinstance(position[k], int) and position[k] >= 0 for k in ("epoch", "batch_position", "global_update")), "Invalid update position")
    require(all(p.grad is None for p in trainer.model.parameters()), "Checkpoint must follow gradient clearing")
    state = cpu_tree(trainer.model.state_dict())
    require(all(v.dtype == torch.float32 for v in state.values() if v.is_floating_point()), "Master is not FP32")
    audit = {k: int(getattr(trainer.audit, k)) for k in ("optimizer_attempts", "native_steps", "effective_updates", "amp_skips")}
    update = position["global_update"]
    require(audit["optimizer_attempts"] == audit["native_steps"] == audit["effective_updates"] == update
            and audit["amp_skips"] == 0 and trainer.ema.updates == update, "Optimizer/EMA update counters disagree")
    record = {"schema": SCHEMA, "bindings": cpu_tree(bindings), "position": cpu_tree(position),
              "model_config": cpu_tree(getattr(trainer.model,"yaml",None)),
              "class_names": cpu_tree(getattr(trainer.model,"names",None)),
              "state": state, "ema": cpu_tree(_ema_model(trainer).state_dict()), "ema_updates": trainer.ema.updates,
              "optimizer": cpu_tree(trainer.optimizer.state_dict()), "optimizer_names": optimizer_names(trainer.model, trainer.optimizer),
              "trainable_parameter_names": [n for n,p in trainer.model.named_parameters() if p.requires_grad],
              "scaler": cpu_tree(trainer.scaler.state_dict()), "scheduler": cpu_tree(trainer.scheduler.state_dict()),
              "rng": capture_rng(), "sampler": cpu_tree(sampler_state), "audit": audit,
              "counters": cpu_tree(dict(trainer.counters)), "training_modes": {n: m.training for n, m in trainer.model.named_modules()},
              "bn_state_sha256": state_hash({n: v for n, v in state.items() if n.endswith(("running_mean", "running_var", "num_batches_tracked"))}),
              "extra": cpu_tree(extra or {})}
    record["master_state_sha256"] = state_hash(record["state"])
    record["ema_state_sha256"] = state_hash(record["ema"])
    require(finite_tree(record), "Nonfinite full recovery state")
    return record


def validate_payload(record, bindings):
    require(record.get("schema") == SCHEMA and record.get("bindings") == bindings, "Recovery protocol/code/manifest/environment binding mismatch")
    require(finite_tree(record), "Nonfinite recovery state")
    require(state_hash(record["state"]) == record["master_state_sha256"]
            and state_hash(record["ema"]) == record["ema_state_sha256"], "Recovery model identity mismatch")
    require(all(v.dtype == torch.float32 for v in record["state"].values() if v.is_floating_point()), "Recovery master is not FP32")
    update = record["position"]["global_update"]
    require(all(isinstance(record["position"][k],int) and not isinstance(record["position"][k],bool)
                and record["position"][k] >= 0 for k in ("epoch","batch_position","global_update")), "Invalid recovery position")
    require(all(record["audit"][k] == update for k in ("optimizer_attempts", "native_steps", "effective_updates"))
            and record["audit"]["amp_skips"] == 0 and record["ema_updates"] == update, "Recovery counters disagree")
    named_ids = {name: key for group,names in zip(record["optimizer"]["param_groups"],record["optimizer_names"],strict=True)
                 for key,name in zip(group["params"],names,strict=True)}
    require(set(record["trainable_parameter_names"]) <= named_ids.keys(), "Recovery optimizer omitted a trainable parameter")
    for name in record["trainable_parameter_names"]:
        state = record["optimizer"]["state"].get(named_ids[name])
        if update == 0:
            require(state is None, "Unstarted optimizer has populated state")
        else:
            require(state is not None and float(state["step"]) == update, "AdamW step state differs from recovery counter")
    bn = {n:v for n,v in record["state"].items() if n.endswith(("running_mean","running_var","num_batches_tracked"))}
    require(state_hash(bn) == record["bn_state_sha256"], "Recovery BN identity differs")
    sampler=record["sampler"]
    if sampler.get("schema") == "EXP17_G1_SEALED_BATCH_CURSOR_V1":
        require(sampler["epoch"] == record["position"]["epoch"]
                and sampler["next_batch"] == record["position"]["batch_position"]
                and sampler["epoch"] * sampler["num_batches"] + sampler["next_batch"] == update,
                "Recovery position disagrees with sealed sampler cursor")


def restore_training_state(trainer, record, *, bindings):
    validate_payload(record, bindings)
    require(state_hash(getattr(trainer.model, "yaml", None)) == state_hash(record["model_config"])
            and state_hash(getattr(trainer.model, "names", None)) == state_hash(record["class_names"]),
            "Model architecture metadata or class mapping changed at recovery")
    require(optimizer_names(trainer.model, trainer.optimizer) == record["optimizer_names"], "Optimizer groups changed at recovery")
    trainer.model.load_state_dict(record["state"], strict=True)
    _ema_model(trainer).load_state_dict(record["ema"], strict=True)
    trainer.ema.updates = record["ema_updates"]
    trainer.optimizer.load_state_dict(record["optimizer"])
    trainer.scaler.load_state_dict(record["scaler"])
    trainer.scheduler.load_state_dict(record["scheduler"])
    for key, value in record["audit"].items(): setattr(trainer.audit, key, value)
    trainer.counters.clear(); trainer.counters.update(record["counters"])
    if hasattr(trainer, "completed_epochs"):
        trainer.completed_epochs = record["position"]["global_update"] // 21
    modules = dict(trainer.model.named_modules())
    require(modules.keys() == record["training_modes"].keys(), "Module structure changed at recovery")
    for name, mode in record["training_modes"].items(): modules[name].training = mode
    require(state_hash(trainer.model.state_dict()) == record["master_state_sha256"], "Master did not restore exactly")
    require(state_hash(_ema_model(trainer).state_dict()) == record["ema_state_sha256"], "EMA did not restore exactly")
    require(state_hash(trainer.optimizer.state_dict()) == state_hash(record["optimizer"])
            and state_hash(trainer.scheduler.state_dict()) == state_hash(record["scheduler"])
            and state_hash(trainer.scaler.state_dict()) == state_hash(record["scaler"]), "Optimizer/scheduler/scaler did not restore exactly")
    restore_rng(record["rng"])
    return copy.deepcopy(record["position"]), copy.deepcopy(record["sampler"]), copy.deepcopy(record["extra"])


def fsync_directory(directory):
    """POSIX directory fsync; Windows lacks this interface, so report it."""
    if os.name == "nt": return False
    fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)
    return True


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    require(not path.is_symlink(), "Atomic JSON target is a symlink")
    if not replace: require(not path.exists(), "Immutable JSON target already exists")
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


class AtomicCheckpointStore:
    """Two recovery slots, plus immutable periodic/Best/final artifacts.

    The outer executor must hold its persistent run lock. A store-level
    pointer compare protects accidental stale writers; it is not a run lock.
    Failed temporary/orphan files are preserved for investigation.
    """
    def __init__(self, directory, bindings, *, failpoint=None):
        self.directory = Path(directory)
        require(not self.directory.is_symlink(), "Recovery directory is a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bindings = copy.deepcopy(bindings)
        self.failpoint = failpoint

    def _fail(self, stage):
        if self.failpoint: self.failpoint(stage)

    def _write(self, path, payload):
        require(not path.is_symlink(), "Checkpoint target is a symlink")
        temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
        with temporary.open("xb") as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        self._fail("after_fsync")
        digest = sha256_file(temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=True)
        require(state_hash(check) == state_hash(payload), "Serialized recovery state did not reload exactly")
        self._fail("before_blob_rename")
        os.replace(temporary, path); fsync_directory(path.parent)
        self._fail("after_blob_rename")
        return {"schema": POINTER_SCHEMA, "file": path.name, "sha256": digest, "bytes": path.stat().st_size,
                "position": copy.deepcopy(payload["position"]), "bindings_sha256": state_hash(self.bindings),
                "directory_fsync_supported": os.name != "nt"}

    def save_latest(self, payload, *, allow_same_update=False):
        validate_payload(payload, self.bindings)
        pointer = self.directory / "latest.json"
        previous = pointer.read_bytes() if pointer.exists() else None
        old = None
        if previous is not None:
            old_payload, old = self.load_latest()
            old_step = old_payload["position"]["global_update"]
            step = payload["position"]["global_update"]
            require(step == old_step + 1 or (allow_same_update and step == old_step), "Recovery update skipped or repeated")
            if step == old_step:
                require(all(state_hash(payload[k]) == state_hash(old_payload[k]) for k in
                    ("state", "ema", "optimizer", "scaler", "scheduler", "rng", "audit", "counters", "sampler", "training_modes")),
                    "Same-update save changed training state")
        slot = 0 if old is None else 1 - int(old["file"].split("_")[-1].split(".")[0])
        receipt = self._write(self.directory / f"recovery_slot_{slot}.pt", payload)
        require((pointer.read_bytes() if pointer.exists() else None) == previous, "Stale recovery writer; latest pointer changed")
        receipt["role"] = "last_recovery"
        self._fail("before_pointer_commit")
        atomic_json(pointer, receipt, replace=previous is not None)
        self._fail("after_pointer_commit")
        return receipt

    def _load(self, path, receipt):
        require(path.parent.resolve() == self.directory.resolve() and not path.is_symlink() and path.is_file(), "Recovery path changed")
        require(receipt["schema"] == POINTER_SCHEMA and receipt["bindings_sha256"] == state_hash(self.bindings), "Recovery receipt binding mismatch")
        require(path.stat().st_size == receipt["bytes"] and sha256_file(path) == receipt["sha256"], "Checkpoint corruption/hash conflict")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(payload, self.bindings)
        require(payload["position"] == receipt["position"], "Checkpoint position differs from receipt")
        return payload, receipt

    def load_latest(self, *, expected_sha256=None):
        pointer = self.directory / "latest.json"
        require(pointer.is_file() and not pointer.is_symlink(), "Missing trusted Last pointer")
        receipt = json.loads(pointer.read_text(encoding="utf-8"))
        require(receipt.get("role") == "last_recovery" and receipt.get("file") in
                ("recovery_slot_0.pt", "recovery_slot_1.pt"), "Malformed Last pointer")
        if expected_sha256 is not None: require(receipt["sha256"] == expected_sha256, "Requested checkpoint hash conflicts with Last")
        return self._load(self.directory / receipt["file"], receipt)

    def save_artifact(self, name, payload, *, role):
        require(role in {"periodic", "best", "final", "epoch_last", "ema"}, "Unknown checkpoint artifact role")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.pt", name) and ".." not in name, "Unsafe artifact filename")
        validate_payload(payload, self.bindings)
        path = self.directory / name
        require(not path.exists() and not path.with_suffix(path.suffix + ".json").exists(), "Immutable checkpoint already exists")
        receipt = self._write(path, payload); receipt["role"] = role
        atomic_json(path.with_suffix(path.suffix + ".json"), receipt)
        return receipt

    def load_artifact(self, name, *, expected_sha256=None):
        require(Path(name).name == name, "Unsafe checkpoint artifact path")
        receipt = json.loads((self.directory / (name + ".json")).read_text(encoding="utf-8"))
        require(receipt["file"] == name and receipt.get("role") in {"periodic", "best", "final", "epoch_last", "ema"}, "Artifact role mismatch")
        if expected_sha256 is not None: require(receipt["sha256"] == expected_sha256, "Artifact hash conflicts")
        return self._load(self.directory / name, receipt)
