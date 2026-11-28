"""Build a status-labelled storage estimate from sealed/CPU size receipts."""
import argparse
import csv
import json
from pathlib import Path

from tools.exp18.common import require, sha256, write_json_exclusive
from tools.exp18.storage_policy import estimate_capacity

GIB = 1024**3


def build(coverage, synthetic, historical):
    coverage_path, synthetic_path, historical_path = map(Path, (coverage, synthetic, historical))
    c = json.loads(coverage_path.read_text(encoding="utf-8"))
    s = json.loads(synthetic_path.read_text(encoding="utf-8"))
    require(c["status"] == "PASS_4200_SEALED_METADATA_STEPS" and s["status"] == "PASS",
            "Coverage and synthetic CPU measurement receipts must PASS")
    with historical_path.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    full = [r for r in history if r["role"] == "periodic"]
    best = [r for r in history if r["role"] in ("best_master", "best_ema")]
    require(full and best, "Missing historical size receipts")
    checkpoint = max(int(r["bytes"]) for r in full)
    best_pack = max(int(r["bytes"]) for r in best)  # historical bundles already have both role states
    a_epoch = 668 * (3 * 640**2 + 160**2 + 4 * 160**2)
    n_slot = 4 * 3 * 640**2 + 160**2
    adj2, adj3, max_n = (c["max_N_slots_two_adjacent_epochs"], c["max_N_slots_three_adjacent_epochs"], c["max_N_slots_single_epoch"])
    cache_two = 6 * a_epoch + 2 * adj2 * n_slot
    cache_build = 7 * a_epoch + (adj2 + adj3) * n_slot
    cache_recovery = 7 * a_epoch + 5 * max_n * n_slot  # builder + 2 per arm; non-adjacent during rewind
    opts = {"checkpoint_bytes": checkpoint, "best_bytes": best_pack,
            "last_bytes": (best_pack + 1) // 2, "journal_bytes": s["synthetic_sizes_bytes"]["journal_total"],
            "diagnostic_bytes": s["synthetic_sizes_bytes"]["diagnostic_maximum"],
            "val_pass_bytes": 0, "cache_peak_bytes": cache_recovery, "extra_bytes": 0}
    subtotal = estimate_capacity(**opts)
    theoretical = {
        "old_full_checkpoint_parameter_and_moment_floor_GiB": 564.351479404,
        "old_all_epoch_image_tensor_floor_GiB": 646.087646484375,
        "old_combined_floor_GiB": 1210.439125888,
    }
    return {
        "status": "FORMAL_STORAGE_GATE_PENDING", "historical_server_free_estimate_bytes": 45 * GIB,
        "old_architecture": theoretical, "synthetic_real_serialized_sizes": s["synthetic_sizes_bytes"],
        "historical_full_state_max_bytes": checkpoint, "historical_best_bundle_max_bytes": best_pack,
        "checkpoint_examples_using_historical_max_bytes": {"9": 9 * checkpoint, "15": 15 * checkpoint,
                                                            "16_including_publication": 16 * checkpoint},
        "permanent_periodic_comparison_using_historical_max": [
            {"interval_epochs": i, "checkpoints_per_arm": 200 // i + 1, "checkpoints_all_arms": 3 * (200 // i + 1),
             "checkpoint_bytes_all_arms": 3 * (200 // i + 1) * checkpoint, "max_replay_updates": 21 * i}
            for i in (5, 10, 20, 25, 50)],
        "cache_tensor_bytes_excluding_serialization": {
            "A_per_epoch": a_epoch, "N_per_image": n_slot, "three_arms_two_epochs": cache_two,
            "single_builder_adjacent_epochs": cache_build, "single_builder_recovery_nonadjacent_epochs": cache_recovery,
        },
        "partial_subtotal_using_historical_and_synthetic_proxies": subtotal,
        "excluded_from_subtotal": ["Real VAL prediction bound, 480 passes",
            "Real checkpoint and model-only output sizes", "Real journal/diagnostic row and attempt upper bounds",
            "Cache serialization/masks/metadata beyond counted tensors", "Retention and failure-evidence working space",
            "Real current filesystem free bytes, durability and inode headroom"],
        "sources": {p.name: sha256(p) for p in (coverage_path, synthetic_path, historical_path)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coverage", required=True)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--historical", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    write_json_exclusive(args.output, build(args.coverage, args.synthetic, args.historical))


if __name__ == "__main__":
    main()
