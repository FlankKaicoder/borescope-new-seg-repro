"""Numerical and checkpoint primitives for the gated A11 full experiment."""

import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tools.exp14.a11_smoke_numerics import (
    NumericalAudit, NumericalFailure, exact_tree, finite_tree, model_fp32_finite, optimizer_group_names,
)
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.diagnose_amp_first_batch import parameters_sha
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp14.revised_amp_smoke import NATIVE_DEFAULTS


CHECKPOINT_SCHEMA = "exp14_a11_full_epoch_v1"
TARGET_EPOCHS = 100
ARM_EPOCH_BATCHES = {"e0": 21, "e1": 33}


def expected_scaler_state(updates):
    require(type(updates) is int and 0 <= updates <= max(ARM_EPOCH_BATCHES.values()) * TARGET_EPOCHS,
            "Invalid formal AMP update count")
    growths, tracker = divmod(updates, NATIVE_DEFAULTS["growth_interval"])
    return {**NATIVE_DEFAULTS, "scale": NATIVE_DEFAULTS["growth_factor"] ** growths, "_growth_tracker": tracker}


def validate_scaler_trajectory(state, updates):
    expected = expected_scaler_state(updates)
    if state != expected:
        raise NumericalFailure("Native GradScaler differs from the finite-update trajectory",
                               {"updates": updates, "expected_scaler_state": expected,
                                "observed_scaler_state": state if finite_tree(state) else None})
    return expected


class FormalNumericalAudit(NumericalAudit):
    def __init__(self, trainer, max_updates):
        require(trainer.arm in ARM_EPOCH_BATCHES and max_updates == ARM_EPOCH_BATCHES[trainer.arm] * TARGET_EPOCHS,
                "Formal numerical budget differs from the arm's planned update count")
        super().__init__(trainer, update_limit=max_updates)

    def resume_at(self, completed_updates):
        require(type(completed_updates) is int and 0 < completed_updates <= self.update_limit
                and completed_updates % ARM_EPOCH_BATCHES[self.trainer.arm] == 0
                and self.optimizer_attempts == self.native_steps == self.effective_updates == self.amp_skips == 0,
                "Resume requires untouched counters and a valid completed epoch")
        validate_scaler_trajectory(self.scaler.state_dict(), completed_updates)
        self.optimizer_attempts = self.native_steps = self.effective_updates = completed_updates

    def optimizer_step(self):
        before = validate_scaler_trajectory(self.scaler.state_dict(), self.effective_updates)
        report = super().optimizer_step()
        try:
            after = validate_scaler_trajectory(report["scaler_state"], self.effective_updates)
        except NumericalFailure as error:
            error.evidence = {**report, **error.evidence}
            raise
        report["native_scale_growth"] = after["scale"] > before["scale"]
        report["ema_model_fp32_finite"] = model_fp32_finite(self.trainer.ema.ema)
        report["ema_updates"] = self.trainer.ema.updates
        if not report["ema_model_fp32_finite"] or report["ema_updates"] != self.effective_updates:
            raise NumericalFailure("Native EMA state or update count differs", report)
        return report


def capture_rng_state():
    name, state, position, has_gauss, cached_gauss = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (name, state.tolist(), position, has_gauss, cached_gauss),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def numpy_rng_state(state):
    name, words, position, has_gauss, cached_gauss = state
    require(name == "MT19937" and len(words) == 624 and all(type(v) is int and 0 <= v < 2**32 for v in words)
            and type(position) is int and 0 <= position <= 624 and type(has_gauss) is int and has_gauss in (0, 1)
            and math.isfinite(cached_gauss), "Checkpoint NumPy RNG state differs")
    return name, np.asarray(words, dtype=np.uint32), position, has_gauss, cached_gauss


def validate_rng_state(state):
    require(set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"} and len(state["torch_cuda"]) == 1
            and all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() > 0
                    for v in [state["torch_cpu"], *state["torch_cuda"]]), "Checkpoint RNG state differs")
    random.Random(0).setstate(state["python"])
    np.random.RandomState(0).set_state(numpy_rng_state(state["numpy"]))
    torch.Generator(device="cpu").set_state(state["torch_cpu"])


def restore_rng_state(state):
    validate_rng_state(state)
    require(torch.cuda.device_count() == len(state["torch_cuda"]), "Checkpoint CUDA RNG device count differs")
    random.setstate(state["python"])
    np.random.set_state(numpy_rng_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def adam_step_counts(state, updates):
    values = [t["step"].item() for t in state.values()]
    require(values and all(math.isfinite(v) and v == int(v) and 0 < v <= updates for v in values)
            and max(values) == updates, "AdamW slots do not corroborate the completed update count")
    return Counter(int(v) for v in values)


def validate_epoch_metadata(arm, metadata):
    require(arm in ARM_EPOCH_BATCHES and metadata["epoch_batches"] == ARM_EPOCH_BATCHES[arm]
            and type(metadata["completed_epochs"]) is int and 1 <= metadata["completed_epochs"] <= TARGET_EPOCHS
            and metadata["target_epochs"] == TARGET_EPOCHS
            and metadata["optimizer_updates"] == metadata["completed_epochs"] * ARM_EPOCH_BATCHES[arm],
            "Checkpoint metadata is outside the arm's completed-epoch budget")
    return metadata["optimizer_updates"]


def json_plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def validate_model_states(master, shadow):
    require(master and master.keys() == shadow.keys()
            and all(isinstance(value, torch.Tensor) and isinstance(shadow[name], torch.Tensor)
                    and value.dtype == shadow[name].dtype and value.shape == shadow[name].shape
                    and (not value.is_floating_point() or value.dtype == torch.float32)
                    for name, value in master.items()), "Formal checkpoint master and EMA schemas or dtypes differ")


def read_epoch_checkpoint(path, expected_identity=None, expected_arm=None):
    """Read a formal checkpoint on CPU, with no model construction or optimizer calls."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    require(record["schema"] == CHECKPOINT_SCHEMA and record["class_names"] == dict(enumerate(CLASSES))
            and (expected_arm is None or record["arm"] == expected_arm)
            and (expected_identity is None or record["identity"] == json_plain(expected_identity)),
            "Formal checkpoint schema, arm, data/model identity, or class mapping differs")
    updates = validate_epoch_metadata(record["arm"], record["metadata"])
    require(record["optimizer_attempts"] == record["native_steps"] == record["effective_updates"] == updates
            and record["amp_skips"] == 0 and record["ema_updates"] == updates
            and record["initial_parameters_sha256"] != record["parameters_sha256"]
            and record["scheduler"].get("last_epoch") == record["metadata"]["completed_epochs"] - 1,
            "Formal checkpoint update, EMA, scheduler, or parameter trajectory differs")
    validate_model_states(record["model_state"], record["ema_state"])
    require(finite_tree([record["model_state"], record["ema_state"], record["optimizer"], record["scheduler"]])
            and record["model_state_sha256"] == hash_state(record["model_state"])
            and record["ema_state_sha256"] == hash_state(record["ema_state"]),
            "Formal checkpoint contains invalid tensors or mismatched model hashes")
    validate_scaler_trajectory(record["scaler"], updates)
    adam_step_counts(record["optimizer"]["state"], updates)
    validate_rng_state(record["rng"])
    return record


def verify_terminal_reload(record):
    """Reconstruct the formal final architecture and its recorded CPU optimizer slots."""
    from ultralytics.nn.tasks import SegmentationModel

    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(record["model_config"], ch=3, nc=len(CLASSES), verbose=False)
    model.names = record["class_names"]
    for state, state_hash in (("model_state", "model_state_sha256"), ("ema_state", "ema_state_sha256")):
        model.load_state_dict(record[state], strict=True)
        require(model_fp32_finite(model) and hash_state(model.state_dict()) == record[state_hash],
                "Strict final seven-class model reload differs")
        if state == "model_state":
            require(parameters_sha(model) == record["parameters_sha256"], "Final parameter hash differs after strict reload")
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [named[name] for name in group]} for group in record["optimizer_group_names"]])
    optimizer.load_state_dict(record["optimizer"])
    scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
    scaler.load_state_dict(record["scaler"])
    require(exact_tree(optimizer.state_dict(), record["optimizer"]) and scaler.state_dict() == record["scaler"],
            "Strict final optimizer or scaler reload differs")


def save_epoch_checkpoint(trainer, path, metadata):
    path = Path(path)
    model, audit, ema = trainer.model, trainer.audit, trainer.ema
    updates = validate_epoch_metadata(trainer.arm, metadata)
    require(audit.optimizer_attempts == audit.native_steps == audit.effective_updates == ema.updates == updates
            and audit.amp_skips == 0 and model_fp32_finite(model) and model_fp32_finite(ema.ema),
            "Only completed finite epochs may publish a resumable checkpoint")
    master, shadow = cpu_state(model.state_dict()), cpu_state(ema.ema.state_dict())
    record = {"schema": CHECKPOINT_SCHEMA, "arm": trainer.arm, "identity": json_plain(trainer.checkpoint_identity),
              "metadata": json_plain(metadata), "model_config": json_plain(model.yaml), "class_names": dict(model.names),
              "model_state": master, "model_state_sha256": hash_state(master),
              "ema_state": shadow, "ema_state_sha256": hash_state(shadow), "ema_updates": ema.updates,
              "parameters_sha256": parameters_sha(model), "initial_parameters_sha256": audit.initial_parameters_sha256,
              "initial_state_sha256": audit.initial_state_sha256, "optimizer": cpu_state(trainer.optimizer.state_dict()),
              "optimizer_group_names": optimizer_group_names(model, trainer.optimizer),
              "scheduler": cpu_state(trainer.scheduler.state_dict()), "scaler": trainer.scaler.state_dict(),
              "optimizer_attempts": audit.optimizer_attempts, "native_steps": audit.native_steps,
              "effective_updates": updates, "amp_skips": audit.amp_skips, "rng": capture_rng_state()}
    with TemporaryDirectory(prefix=".a11_checkpoint_", dir=path.parent) as temp:
        staged = Path(temp) / "epoch.pt"
        with staged.open("xb") as stream:
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded = read_epoch_checkpoint(staged, trainer.checkpoint_identity, trainer.arm)
        require(exact_tree(reloaded, record), "Formal checkpoint CPU readback changed serialized state")
        terminal = metadata["completed_epochs"] == TARGET_EPOCHS
        if terminal:
            verify_terminal_reload(reloaded)
        os.link(staged, path)
        if os.name == "posix":
            folder = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
    return {"path": str(path), "sha256": sha256_file(path), "completed_epochs": metadata["completed_epochs"],
            "optimizer_updates": updates, "model_state_sha256": record["model_state_sha256"],
            "ema_state_sha256": record["ema_state_sha256"], "parameters_sha256": record["parameters_sha256"],
            "cpu_readback_exact": True, "strict_terminal_reload": terminal}


def load_epoch_checkpoint(trainer, path, expected_identity):
    """Restore an attested completed epoch into a fresh trainer exactly once."""
    record = read_epoch_checkpoint(path, expected_identity, trainer.arm)
    model, audit, ema, optimizer = trainer.model, trainer.audit, trainer.ema, trainer.optimizer
    require(not optimizer.state and ema.updates == 0
            and audit.optimizer_attempts == audit.native_steps == audit.effective_updates == audit.amp_skips == 0
            and record["initial_parameters_sha256"] == audit.initial_parameters_sha256
            and record["initial_state_sha256"] == audit.initial_state_sha256
            and record["model_config"] == json_plain(model.yaml)
            and record["class_names"] == dict(model.names)
            and record["optimizer_group_names"] == optimizer_group_names(model, optimizer),
            "Resume trainer is not freshly initialized with the same architecture, inputs and optimizer groups")
    model.load_state_dict(record["model_state"], strict=True)
    ema.ema.load_state_dict(record["ema_state"], strict=True)
    ema.updates = record["ema_updates"]
    optimizer.load_state_dict(record["optimizer"])
    trainer.scheduler.load_state_dict(record["scheduler"])
    trainer.scaler.load_state_dict(record["scaler"])
    require(hash_state(model.state_dict()) == record["model_state_sha256"]
            and parameters_sha(model) == record["parameters_sha256"] and model_fp32_finite(model)
            and hash_state(ema.ema.state_dict()) == record["ema_state_sha256"] and model_fp32_finite(ema.ema)
            and exact_tree(cpu_state(optimizer.state_dict()), record["optimizer"])
            and exact_tree(cpu_state(trainer.scheduler.state_dict()), record["scheduler"]),
            "Strict resumed model, EMA, AdamW or scheduler differs from the checkpoint")
    audit.resume_at(record["effective_updates"])
    restore_rng_state(record["rng"])
    return record["metadata"]
