"""Resolve a versioned CPU runtime profile without importing Torch."""
import json
from pathlib import Path

from .common import digest, require, sha256
from .protocol import validate_protocol


def load_cpu_profile(path, repository):
    path, repository = Path(path).resolve(strict=True), Path(repository).resolve(strict=True)
    runtime = json.loads(path.read_text(encoding="utf-8"))
    require(runtime["schema"] == "EXP18_CPU_RUNTIME_V1" and runtime["device"] == "cpu", "Not a CPU runtime profile")
    require(runtime["lambda_state"] is None and runtime["formal_training_authorized"] is False,
            "Runtime cannot grant training or determine lambda")
    require(all(type(runtime[k]) is int and runtime[k] == 0 for k in
                ("GPU_calls", "real_models", "real_checkpoint_tensor_loads", "VAL_passes", "TEST_accesses")),
            "Runtime exceeds CPU authorization")
    protocol_path = (repository / runtime["protocol"]).resolve(strict=True)
    require(protocol_path.is_relative_to(repository / "configs/exp18"), "Protocol escaped EXP18 config directory")
    require(sha256(protocol_path) == runtime["protocol_file_sha256"], "Protocol file SHA changed")
    protocol = validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")))
    require(digest(protocol) == runtime["protocol_sha256"] and protocol["train"]["epochs"] == runtime["horizon"],
            "Runtime/protocol horizon binding differs")
    from .data import RUNTIME_CONFIG_SHA256, RUNTIME_CONFIG
    require(runtime["historical_runtime_config"] == RUNTIME_CONFIG
            and runtime["historical_runtime_sha256"] == RUNTIME_CONFIG_SHA256, "Historical TRAIN bindings differ")
    return runtime, protocol
