"""Create the public Exp15 J-A result from private TRAIN-only run ledgers."""

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def events(path, kind, limit):
    with path.open(encoding="utf-8") as stream:
        rows = [row for line in stream if (row := json.loads(line))["event"] == kind]
    return rows[:limit]


def select(obj, keys):
    return {k: obj[k] for k in keys}


def compare(left, right, keys):
    return len(left) == len(right) and all(select(a, keys) == select(b, keys) for a, b in zip(left, right, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "preflight", "j0", "j1", "historical-e0", "cpu-replay", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    gate = load(args.root / args.preflight / "summary.json")
    j0, j1 = (load(args.root / arm / "summary.json") for arm in (args.j0, args.j1))
    total = j0["steps"]
    keys = ("batch", "epoch", "source_paths", "raw_targets", "augmented_targets", "lrs")
    native_batches, region_batches = (events(args.root / arm / "events.jsonl", "batch_read", total)
                                     for arm in (args.j0, args.j1))
    native_pair = compare(native_batches, region_batches, keys)
    reference_batches = events(args.historical_e0, "batch_read", total)
    reference_loss = events(args.historical_e0, "forward", total)
    reference_init = events(args.historical_e0, "setup", 1)[0]
    audited_events = [events(args.root / arm / "events.jsonl", "update", total) for arm in (args.j0, args.j1)]
    for result, updates in zip((j0, j1), audited_events, strict=True):
        ledger = args.root / "budget" / f'{result["variant"]}.attempts'
        durable = ledger.read_text().splitlines()
        assert len(durable) == total and len(updates) == total
        assert all(row["step"] == i == row["optimizer"]["effective_update"] == int(ticket)
                   and row["optimizer"]["native_step_called"] and not row["optimizer"]["amp_skipped"]
                   for i, (row, ticket) in enumerate(zip(updates, durable, strict=True), 1))
    assert gate["identity"] == j0["identity"] == j1["identity"]
    assert (native_pair and j0["initial_native_state_sha256"] == j1["initial_native_state_sha256"]
            and j0["counts"] == j1["counts"] and j0["native_parameter_groups"] == j1["native_parameter_groups"]
            and compare(j0["trace"], j1["trace"], ("step", "batch_sha256", "lrs")))
    paired = {"matched_batch_records": total, "identical_augmented_tensor_sha256_count": total,
              "same_native_optimizer_group_membership_and_lrs": True, "same_coco_initialized_native_state": True,
              "historical_e0_same_initial_model_state": reference_init["initial_state_sha256"] == j0["initial_native_state_sha256"],
              "historical_e0_same_batch_ledgers": compare(native_batches, reference_batches, keys),
              "historical_e0_exact_native_loss_items_count": sum(a["native_loss_items"] == b["loss_items"]
                                                                 for a, b in zip(j0["trace"], reference_loss, strict=True)),
              "historical_e0_exact_native_total_loss_count": sum(a["native_loss"] == b["loss"]
                                                                 for a, b in zip(j0["trace"], reference_loss, strict=True))}
    keys = ("steps", "after_warmup", "arm_attempt_ceiling", "warmup_iterations", "amp_skips",
            "initial_bn_sha256", "final_bn_sha256", "backbone_parameter_sha256_before", "backbone_parameter_sha256_after",
            "projection_parameter_sha256_before", "projection_parameter_sha256_after", "optimizer_unique_parameters",
            "counts", "checkpoints", "peak_allocated_bytes", "peak_reserved_bytes",
            "median_postwarmup_forward_backward_seconds", "median_postwarmup_audited_step_seconds",
            "lambda_s")
    cpu = load(args.cpu_replay)
    assert cpu["status"] == "PASS_REAL_TRAIN_CPU_FORWARD_BACKWARD" and cpu["effective_updates"] == 0
    arms = {}
    for arm, result, updates in zip(("j0", "j1"), (j0, j1), audited_events, strict=True):
        data = arms[arm] = select(result, keys)
        data["native_optimizer_calls"] = len(updates)
        data["checkpoint_ema_updates"] = updates[-1]["optimizer"]["effective_update"]
        data["last_native_loss"] = result["trace"][-1]["native_loss"]
        finite = [row["supcon"] for row in result["trace"] if row["supcon"] is not None]
        if finite:
            data["supcon_loss_range"] = [min(finite), max(finite)]
    preflight_public = select(gate, ("disabled_equivalence", "coverage", "gradients", "weight_selection", "private_visuals"))
    active_group_counts = list(zip(*(r["eligible_groups_by_class"] for r in gate["per_batch_regions"]), strict=True))
    preflight_public["coverage"]["median_active_source_groups_per_eligible_batch_by_class"] = [
        statistics.median(count for count in cls if count) for cls in active_group_counts
    ]
    report = {"status": "READY_FOR_J0_J1_FORMAL_REVIEW", "scope": "ENGINEERING_SMOKE", "runtime": gate["runtime"],
              "identity": gate["identity"], "arms": arms, "pair_comparison": paired, "preflight": preflight_public, "cpu_replay": cpu,
              "heldout_inference": {"val": 0, "test": 0}, "j2_updates": 0}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

