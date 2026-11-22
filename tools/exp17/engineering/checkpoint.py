"""Engineering-only FP32 master/EMA/AdamW/GradScaler serialization and reload."""
from pathlib import Path

import torch
from ultralytics.nn.tasks import SegmentationModel

from tools.exp14.a11_full_numerics import adam_step_counts, validate_scaler_trajectory
from tools.exp14.a11_smoke_numerics import exact_tree, finite_tree, optimizer_group_names
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp15.regions import add_projection
from tools.exp17.engineering.model import export_native_model

SCHEMA = "exp17_j2_engineering_smoke_only_v1"


def save_engineering_checkpoint(trainer, path, configuration, diagnostics):
    """A smoke snapshot can never be used as a formal initialization/resume."""
    require(not configuration.get("formal_training_authorized", False), "Formal state is outside this entry point")
    updates = trainer.audit.effective_updates
    require(0 < updates <= 50 and trainer.audit.native_steps == updates == trainer.ema.updates,
            "Engineering update/EMA counters differ")
    state, ema = cpu_state(trainer.model.state_dict()), cpu_state(trainer.ema.ema.state_dict())
    actual_groups = optimizer_group_names(trainer.model, trainer.optimizer)
    require(actual_groups == trainer.optimizer_names, "Actual AdamW parameter groups differ from their initial identity")
    require(all(v.dtype == torch.float32 for name, v in state.items() if v.is_floating_point()),
            "Engineering master contains reduced-precision parameters or buffers")
    record = {"schema": SCHEMA, "formal_model": False, "formal_resume_allowed": False,
              "effective_updates": updates, "model_config": trainer.model.yaml,
              "class_names": dict(trainer.model.names), "configuration": configuration,
              "diagnostics": diagnostics, "optimizer_names": actual_groups,
              "state": state, "ema": ema, "ema_updates": trainer.ema.updates,
              "optimizer": cpu_state(trainer.optimizer.state_dict()),
              "scaler": trainer.scaler.state_dict(), "scheduler": trainer.scheduler.state_dict(),
              "model_state_sha256": hash_state(state), "ema_state_sha256": hash_state(ema),
              "native_state_sha256": hash_state({k: v for k, v in state.items() if not k.startswith("region_projection.")})}
    require(finite_tree(record), "Nonfinite engineering checkpoint")
    with Path(path).open("xb") as stream:
        torch.save(record, stream)
    check = torch.load(path, map_location="cpu", weights_only=True)
    require(exact_tree(record, check) and finite_tree(check), "Engineering checkpoint bytes did not reload exactly")
    validate_scaler_trajectory(check["scaler"], updates)
    adam_step_counts(check["optimizer"]["state"], updates)
    with torch.random.fork_rng(devices=[]):
        loaded = SegmentationModel(check["model_config"], ch=3, nc=7, verbose=False)
        add_projection(loaded)
    for field in ("state", "ema"):
        loaded.load_state_dict(check[field], strict=True)
        require(exact_tree(dict(loaded.state_dict()), check[field]), "Strict master/EMA model reload failed")
    parameters = dict(loaded.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [parameters[n] for n in names]} for names in check["optimizer_names"]])
    optimizer.load_state_dict(check["optimizer"])
    scaler = torch.amp.GradScaler("cpu")
    scaler.load_state_dict(check["scaler"])
    require(exact_tree(optimizer.state_dict(), check["optimizer"]) and scaler.state_dict() == check["scaler"],
            "Strict AdamW/GradScaler reload failed")
    loaded.load_state_dict(check["state"], strict=True)
    loaded.names = check["class_names"]
    native = export_native_model(loaded)
    native.names = check["class_names"]
    require(hash_state(native.state_dict()) == check["native_state_sha256"], "Stripped FP32 master differs")
    require(check["model_state_sha256"] != check["ema_state_sha256"], "Master was confused with EMA")
    native.eval()
    with torch.no_grad():
        prediction = native(torch.zeros(1, 3, 64, 64))
    require(finite_tree(prediction), "Native seven-class synthetic structural forward is nonfinite")
    return {"sha256": sha256_file(path), "schema": SCHEMA, "effective_updates": updates,
            "strict_master_reload": True, "strict_ema_reload": True,
            "strict_adamw_reload": True, "strict_gradscaler_reload": True,
            "master_and_ema_differ": check["model_state_sha256"] != check["ema_state_sha256"],
            "native_state_sha256": check["native_state_sha256"],
            "native_seven_class_synthetic_forward": True, "formal_checkpoint": False}
