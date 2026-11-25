"""Strict future native exports; real constructors are gated before imports."""
from .common import require, sha256, write_json_exclusive
from .protocol import require_operation


def export_native(adapter, *, role):
    require_operation("export", adapter.protocol)
    require(role in ("MASTER", "EMA"), "Unknown export role")
    import copy
    import os
    from pathlib import Path
    import torch
    from tools.exp17.engineering.model import _native_activation_scope
    from ultralytics.nn.tasks import SegmentationModel
    from .model import native_state_dict
    from .checkpoint import state_hash, _tree
    horizon = adapter.protocol["train"]["epochs"]
    require(adapter.ema.updates == 21 * horizon, "Native export requires the complete common horizon")
    source = adapter.model if role == "MASTER" else adapter.ema.ema
    source_hash = state_hash(_tree(source.state_dict(), native=True))
    state = {name: tensor.cpu() for name, tensor in native_state_dict(source).items()}
    require(all(not x.is_floating_point() or x.dtype == torch.float32 for x in state.values()), "Export source is not FP32")
    output = Path(adapter.output) / f"last_{role.lower()}_h{horizon}.pt"
    receipt_path = output.with_suffix(".receipt.json")
    if output.exists() or receipt_path.exists():
        import json
        require(output.is_file() and receipt_path.is_file() and not output.is_symlink() and not receipt_path.is_symlink(),
                "Partial/orphan native export preserved; independent recovery required")
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(existing["file"] == output.name and existing["role"] == role and existing["horizon"] == horizon
                and existing["bindings"] == adapter.bindings and existing["native_state_sha256"] == state_hash(state)
                and existing["sha256"] == sha256(output) and existing["bytes"] == output.stat().st_size,
                "Existing export conflicts with recovered endpoint")
        return {"path": str(receipt_path), "sha256": sha256(receipt_path), "reused_verified": True}
    with torch.random.fork_rng(devices=[]), _native_activation_scope():
        model = SegmentationModel(copy.deepcopy(source.yaml), ch=3, nc=7, verbose=False)
    model.load_state_dict(state, strict=True)
    model.names = copy.deepcopy(source.names)
    model.args = copy.deepcopy(source.args)
    require(not hasattr(model, "state_head") and not hasattr(model, "region_projection"), "Training head leaked into native graph")
    require(state_hash(model.state_dict()) == state_hash(state) and state_hash(_tree(source.state_dict(), native=True)) == source_hash,
            "Native strict reload or protected source changed")
    with output.open("xb") as stream:
        torch.save({"model": model.cpu().float(), "ema": None, "epoch": horizon - 1,
                    "exp18": {"role": role, "bindings": adapter.bindings, "state_sha256": state_hash(state)}}, stream)
        stream.flush(); os.fsync(stream.fileno())
    return write_json_exclusive(output.with_suffix(".receipt.json"),
        {"file": output.name, "sha256": sha256(output), "bytes": output.stat().st_size,
         "native_state_sha256": state_hash(state), "training_heads_removed": True,
         "inference_executed": False, "role": role, "horizon": horizon, "bindings": adapter.bindings})
