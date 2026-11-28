"""CPU-only V2 storage overlay for the sealed EXP18 scientific contract."""
from pathlib import Path
import json

from .common import digest, require, sha256
from .protocol import validate_protocol

SCHEMA = "EXP18_STORAGE_V2"
BATCHES = 21
INTERVAL = 5
ROLLING_SLOTS = 2
MAX_REPLAY_UPDATES = INTERVAL * BATCHES
VAL_JOBS = ("MASTER_STANDARD", "MASTER_FIXED", "EMA_STANDARD", "EMA_FIXED")


def policy(horizon):
    require(type(horizon) is int and horizon in (100, 200), "Independent H100/H200 profiles required")
    return {
        "schema": SCHEMA,
        "checkpoint_interval_epochs": INTERVAL,
        "permanent_epochs": [0, horizon // 2, horizon],
        "recovery_slots": ROLLING_SLOTS,
        "cache_epochs": 2,
        "max_replay_updates_per_incident": MAX_REPLAY_UPDATES,
        "native_replay_updates_authorized": 0,
        "formal_storage_gate": "PENDING",
    }


def load_storage_profile(path, repository):
    repository = Path(repository).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    require(path.parent == repository / "configs/exp18", "V2 profile escaped EXP18 config directory")
    runtime = json.loads(path.read_text(encoding="utf-8"))
    horizon = runtime.get("horizon")
    storage = policy(horizon)
    old_path = repository / f"configs/exp18/runtime_h{horizon}.json"
    protocol_path = repository / f"configs/exp18/protocol_h{horizon}.json"
    protocol = validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")))
    old = json.loads(old_path.read_text(encoding="utf-8"))
    require(old["protocol_sha256"] == digest(protocol) and old["protocol_file_sha256"] == sha256(protocol_path)
            and horizon == protocol["train"]["epochs"], "Sealed CPU profile identity changed")
    expected = {
        "schema": "EXP18_CPU_STORAGE_RUNTIME_V2", "device": "cpu", "horizon": horizon,
        "base_runtime_file_sha256": sha256(old_path), "base_protocol_sha256": digest(protocol),
        "storage": storage, "storage_sha256": digest(storage),
        "lambda_state": None, "formal_training_authorized": False,
        "budgets": {"CUDA": 0, "real_model": 0, "real_tensor_load": 0, "real_forward": 0,
                    "real_backward": 0, "real_updates": 0, "VAL": 0, "TEST": 0},
    }
    require(runtime == expected, "V2 CPU authorization or storage identity differs")
    return runtime, protocol


def estimate_capacity(*, checkpoint_bytes, best_bytes, last_bytes, journal_bytes,
                      diagnostic_bytes, val_pass_bytes, cache_peak_bytes, extra_bytes,
                      horizon=200, interval=5, permanent_only=False):
    """Byte accounting for three arms. Supply verified upper bounds for formal use."""
    policy(horizon)
    sizes = (checkpoint_bytes, best_bytes, last_bytes, journal_bytes, diagnostic_bytes,
             val_pass_bytes, cache_peak_bytes, extra_bytes)
    require(all(type(size) is int and size >= 0 for size in sizes), "Invalid byte accounting")
    require(interval in (5, 10, 20, 25, 50) and horizon % interval == 0, "Unsupported comparison cadence")
    events = horizon // INTERVAL
    counts = 3 * (horizon // interval + 1) if permanent_only else 15
    rows = {
        "checkpoints": counts * checkpoint_bytes,
        "checkpoint_publish_headroom": checkpoint_bytes,
        "best": 3 * best_bytes, "best_publish_headroom": best_bytes,
        "last": 6 * last_bytes, "journal": 3 * journal_bytes,
        "diagnostics": 3 * events * diagnostic_bytes, "VAL_predictions": 12 * events * val_pass_bytes,
        "epoch_cache_peak": cache_peak_bytes, "recovery_and_evidence": extra_bytes,
    }
    peak = sum(rows.values())
    reserve = max(8 * 1024**3, (peak + 4) // 5)
    return {"checkpoint_count": counts, "max_replay_updates": interval * BATCHES,
            "categories": rows, "peak_bytes": peak, "reserve_bytes": reserve,
            "required_free_bytes": peak + reserve}
