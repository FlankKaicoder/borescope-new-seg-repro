"""Bounded Best slots/history and immutable epoch100 native FP32 endpoints.

These are model-selection/evaluation artifacts. Full optimizer, scheduler,
GradScaler, RNG and sampler recovery state remains in the recovery checkpoint.
No artifact is created until a future authorized executor calls these helpers.
"""
import copy
import json
import os
import uuid
from pathlib import Path

import torch

from tools.exp14.formal_e0_e1 import hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp17.formal.checkpoint import (atomic_json, cpu_tree, finite_tree,
    fsync_directory, require, sha256_file, state_hash)
from tools.exp17.formal.rng import capture_rng, restore_rng, rng_equal


def _fp32(state):
    require(state and all(isinstance(v, torch.Tensor) and (not v.is_floating_point()
            or v.dtype == torch.float32) for v in state.values()) and finite_tree(state),
            "Export state must be finite FP32")


def _native_state(state):
    return {k: v for k, v in state.items() if not k.startswith("region_projection.")}


def _atomic_export(path, payload, validator, *, weights_only, replace=False, write_receipt=True, failpoint=None):
    path = Path(path)
    receipt_path = path.with_suffix(path.suffix + ".json")
    require(not path.parent.is_symlink(), "Export directory is a symlink")
    require(not path.is_symlink() and (replace or not path.exists())
            and not receipt_path.exists() and not receipt_path.is_symlink(), "Immutable export already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve .pt for the pinned Ultralytics load_checkpoint suffix check.
    temporary = path.with_name(path.stem + ".tmp." + uuid.uuid4().hex + path.suffix)
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if failpoint: failpoint("after_fsync")
    digest = sha256_file(temporary)
    # A module checkpoint is read only from the just-created trusted temporary.
    loaded = torch.load(temporary, map_location="cpu", weights_only=weights_only)
    extra = validator(loaded, temporary)
    require(sha256_file(temporary) == digest, "Export bytes changed during strict reload")
    require((replace or not path.exists()) and not receipt_path.exists(), "Concurrent immutable export writer")
    if failpoint: failpoint("before_blob_rename")
    os.replace(temporary, path)
    fsync_directory(path.parent)
    if failpoint: failpoint("after_blob_rename")
    receipt = {"file": path.name, "sha256": digest, "bytes": path.stat().st_size,
               "schema": payload["schema"], "bindings_sha256": state_hash(payload["bindings"]),
               "directory_fsync_supported": os.name != "nt", **extra}
    if write_receipt:
        atomic_json(receipt_path, receipt)
    return receipt


def save_best_bundle(output, payload, selection_record, *, failpoint=None):
    """Two atomic slots per role, with immutable lightweight selection history.

    Only the inactive slot is replaced. The prior valid Best remains readable
    until the new pointer commits. Older selection receipts retain identities
    and metrics; they do not claim their rotated slot bytes remain archived.
    The outer executor must hold the experiment process lease.
    """
    selection = cpu_tree(selection_record)
    role, epoch = selection["role"], selection["epoch"]
    require(role in {"master", "ema"} and type(epoch) is int and epoch in range(5, 101, 5),
            "Invalid Best role/epoch")
    require(payload["position"]["global_update"] == epoch * 21, "Best epoch/update identity differs")
    state, ema = cpu_tree(payload["state"]), cpu_tree(payload["ema"])
    _fp32(state); _fp32(ema)
    require(state_hash(state) == payload["master_state_sha256"]
            and state_hash(ema) == payload["ema_state_sha256"], "Best recovery state identity differs")
    # Selection records use canonical native-state hashes, permitting the same
    # identity vocabulary as the fixed historical J0/J1 endpoint evaluations.
    master_native, ema_native = hash_state(_native_state(state)), hash_state(_native_state(ema))
    require(selection["master_state_sha256"] == master_native
            and selection["ema_state_sha256"] == ema_native, "Best selection binds other model states")
    bundle = {"schema": "EXP17_G1_BEST_FP32_TENSOR_BUNDLE_V1", "bindings": cpu_tree(payload["bindings"]),
              "position": cpu_tree(payload["position"]), "state": state, "ema": ema,
              "model_config": cpu_tree(payload["model_config"]), "class_names": cpu_tree(payload["class_names"]),
              "ema_updates": payload["ema_updates"], "selection": selection,
              "master_recovery_state_sha256": state_hash(state), "ema_recovery_state_sha256": state_hash(ema),
              "master_native_state_sha256": master_native, "ema_native_state_sha256": ema_native,
              "evaluation_role": role, "supplementary_VAL_selected": True,
              "primary_fixed_budget_endpoint": False, "full_recovery_checkpoint": False}
    require(finite_tree(bundle), "Nonfinite Best metadata/state")
    def validate(loaded, _temporary):
        require(state_hash(loaded) == state_hash(bundle), "Best tensor bundle did not strictly reload")
        return {"role": "best_" + role, "epoch": epoch, "fitness": selection["fitness"],
                "strict_tensor_reload": True, "master_native_state_sha256": master_native,
                "ema_native_state_sha256": ema_native, "optimizer_included": False}
    output = Path(output)
    require(not output.is_symlink(), "Best directory is a symlink")
    output.mkdir(parents=True, exist_ok=True)
    pointer = output / f"best_{role}.json"
    require(not pointer.is_symlink(), "Best pointer is a symlink")
    before = pointer.read_bytes() if pointer.exists() else None
    if before is None:
        slot = 0
    else:
        previous = json.loads(before)
        require(previous["role"] == "best_" + role and previous["epoch"] < epoch
                and previous["bindings_sha256"] == state_hash(payload["bindings"])
                and previous["file"] in {f"best_{role}_slot0.pt", f"best_{role}_slot1.pt"},
                "Best pointer role/order/bindings differ")
        old = output / previous["file"]
        require(not old.is_symlink() and old.is_file() and sha256_file(old) == previous["sha256"],
                "Current Best is corrupt or missing; refuse rotation")
        slot = 1 - int(old.stem[-1])
    history = output / f"best_{role}_epoch{epoch:03d}.selection.json"
    require(not history.exists() and not history.is_symlink(), "Best selection epoch already recorded")
    receipt = _atomic_export(output / f"best_{role}_slot{slot}.pt", bundle, validate,
        weights_only=True, replace=before is not None, write_receipt=False, failpoint=failpoint)
    receipt["blob_retention"] = "TWO_ROTATING_SLOTS_PER_ROLE_CURRENT_AND_PREVIOUS"
    receipt["historical_selection_record_is_weight_archive"] = False
    require((pointer.read_bytes() if pointer.exists() else None) == before, "Stale Best pointer writer")
    if failpoint: failpoint("before_pointer_commit")
    atomic_json(pointer, receipt, replace=before is not None)
    if failpoint: failpoint("after_pointer_commit")
    atomic_json(history, receipt)
    return receipt


def _reuse_completed_final(path, *, bindings, role, expected, names, synthetic_exporter):
    """Read an already committed endpoint during separately authorized resume.

    A blob without its final receipt is an interrupted save, never an implicit
    completed export. This helper does not repair or rewrite either file.
    """
    path = Path(path)
    receipt_path = path.with_suffix(path.suffix + ".json")
    if not path.exists() and not receipt_path.exists():
        return None
    require(path.is_file() and receipt_path.is_file() and not path.is_symlink()
            and not receipt_path.is_symlink(), "Partial final export without committed blob/receipt; preserve evidence")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == "EXP17_G1_EPOCH100_NATIVE_FP32_V1"
            and receipt.get("file") == path.name and receipt.get("role") == "epoch100_" + role
            and receipt.get("bindings_sha256") == state_hash(bindings)
            and receipt.get("native_state_sha256") == expected
            and receipt.get("epoch") == 100 and receipt.get("effective_updates") == 2100
            and receipt.get("primary_evaluation_input") is (role == "master")
            and receipt.get("strict_native_state_reload") is True,
            "Existing final endpoint receipt role/binding/state identity mismatch")
    require(path.stat().st_size == receipt.get("bytes") and sha256_file(path) == receipt.get("sha256"),
            "Existing final endpoint file SHA256/size conflict")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    require(loaded.get("schema") == receipt["schema"] and loaded.get("bindings") == bindings
            and loaded.get("evaluation_role") == role and loaded.get("epoch") == 99
            and loaded.get("effective_updates") == 2100 and loaded.get("ema") is None
            and loaded.get("primary_evaluation_input") is (role == "master")
            and loaded.get("native_state_sha256") == expected, "Existing final checkpoint metadata mismatch")
    model = loaded["model"]
    _fp32(model.state_dict())
    require(not hasattr(model, "region_projection") and model.model[-1].nc == 7
            and model.names == names and hash_state(model.state_dict()) == expected,
            "Existing final checkpoint is not the selected seven-class FP32 native state")
    if not synthetic_exporter:
        from ultralytics.nn.tasks import SegmentationModel
        require(isinstance(model, SegmentationModel) and receipt.get("native_ultralytics_loader_checked") is True,
                "Existing production endpoint lacks its native model/loader verification")
    return copy.deepcopy(receipt)


def write_final_native(trainer, output, bindings, role="master", *, exporter=None):
    """Export explicit FP32 MASTER/EMA as model, never implicit EMA substitution.

    exporter injection is for tiny CPU protocol tests. Production uses the
    unchanged seven-class projection-stripping helper and verifies Ultralytics
    YOLO can load the trusted, just-written native checkpoint without inference.
    """
    require(role in {"master", "ema"}, "Unknown final endpoint role")
    require(trainer.epoch == 99 and trainer.audit.effective_updates == trainer.audit.native_steps
            == trainer.audit.optimizer_attempts == trainer.ema.updates == 2100
            and trainer.audit.amp_skips == 0, "Final export requires complete epoch100/2100 updates")
    require(dict(trainer.model.names) == dict(enumerate(CLASSES)), "Final class mapping differs")
    source = trainer.model if role == "master" else trainer.ema.ema
    source_before, rng = state_hash(source.state_dict()), capture_rng()
    source_modes = [m.training for m in source.modules()]
    try:
        _fp32(source.state_dict())
        expected = hash_state(_native_state(source.state_dict()))
        name = "epoch100_last_master_fp32.pt" if role == "master" else "epoch100_ema_fp32.pt"
        existing = _reuse_completed_final(Path(output) / name, bindings=bindings, role=role,
            expected=expected, names=trainer.model.names, synthetic_exporter=exporter is not None)
        if existing is not None:
            return existing
        if exporter is None:
            from tools.exp17.engineering.model import export_native_model
            exporter_fn = export_native_model
        else:
            exporter_fn = exporter
        native = copy.deepcopy(exporter_fn(source)).cpu().float().eval()
        native.names = copy.deepcopy(trainer.model.names)
        require(not hasattr(native, "region_projection") and native.model[-1].nc == 7,
                "Final native structure still has auxiliary projection or wrong class count")
        _fp32(native.state_dict())
        require(hash_state(native.state_dict()) == expected, "Final native model differs from selected FP32 role")
        payload = {"schema": "EXP17_G1_EPOCH100_NATIVE_FP32_V1", "bindings": cpu_tree(bindings),
                   "model": native, "ema": None, "epoch": 99, "effective_updates": 2100,
                   "train_args": {"task": "segment", "imgsz": 640, "single_cls": False},
                   "evaluation_role": role, "primary_evaluation_input": role == "master",
                   "native_state_sha256": expected, "full_recovery_checkpoint": False}
        def validate(loaded, temporary):
            require(loaded["schema"] == payload["schema"] and loaded["bindings"] == bindings
                    and loaded["evaluation_role"] == role and loaded["ema"] is None,
                    "Final checkpoint role/binding differs")
            model = loaded["model"]
            _fp32(model.state_dict())
            require(type(model) is type(native) and model.names == native.names
                    and hash_state(model.state_dict()) == expected, "Final native checkpoint state failed reload")
            skeleton = copy.deepcopy(native)
            skeleton.load_state_dict(model.state_dict(), strict=True)
            checked = False
            if exporter is None:
                from ultralytics.nn.tasks import load_checkpoint
                loaded_native, _ = load_checkpoint(str(temporary), device="cpu", inplace=False, fuse=False)
                require(hash_state(loaded_native.state_dict()) == expected
                        and not hasattr(loaded_native, "region_projection"), "Native YOLO loader changed selected state")
                checked = True
            return {"role": "epoch100_" + role, "epoch": 100, "effective_updates": 2100,
                    "native_state_sha256": expected, "strict_native_state_reload": True,
                    "native_ultralytics_loader_checked": checked,
                    "primary_evaluation_input": role == "master", "optimizer_included": False}
        return _atomic_export(Path(output) / name, payload, validate, weights_only=False)
    finally:
        restore_rng(rng)
        require(rng_equal(capture_rng(), rng) and state_hash(source.state_dict()) == source_before
                and [m.training for m in source.modules()] == source_modes,
                "Final export changed live source state/modes or RNG")
