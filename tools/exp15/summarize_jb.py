"""Reconcile private J-B ledgers and write public evidence and the formal protocol."""

import argparse
import math
import statistics
from pathlib import Path

from tools.exp14.audit_admission import require, sha256_file
from tools.exp15 import ja, jb
from tools.exp15.summarize import compare, events


def bounds(values):
    vals = [v for v in values if v is not None]
    return [min(vals), statistics.median(vals), max(vals)] if vals else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ja-root", "calibration", "rejected", "smoke", "budget", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    cp, sp = (folder / "summary.json" for folder in (args.calibration, args.smoke))
    cal, smoke = map(jb.load, (cp, sp))
    rejected = jb.load(args.rejected / "summary.json")
    require(rejected["lambda_s_candidate"] == 10 and rejected["status"] == jb.HOLD
            and rejected["optimizer_attempts"] == 0 and rejected["training_batches"] == cal["training_batches"]
            and rejected["binding"] == cal["binding"] and rejected["prespecified_screen"] == cal["prespecified_screen"],
            "Rejected lambda=10 candidate evidence differs")
    limit = jb.SMOKE_STEPS
    require(cal["status"] == "PASS_TRAIN_CALIBRATION" and cal["optimizer_attempts"] == 0
            and smoke["status"] == "PASS_JB_BOUNDED_SMOKE" and smoke["identity"] == cal["identity"]
            and smoke["effective_updates"] == smoke["optimizer_attempts"] == smoke["native_optimizer_calls"] == limit
            and smoke["amp_skips"] == 0 and smoke["lambda_s"] == jb.WEIGHT, "J-B receipts did not pass")
    s_events = args.smoke / "events.jsonl"
    j0 = args.ja_root / "j0_r5"
    keys = ("batch", "epoch", "source_paths", "raw_targets", "augmented_targets", "lrs")
    prior, a_batches = events(j0 / "events.jsonl", "batch_read", limit), events(s_events, "batch_read", limit)
    j0_summary = jb.load(j0 / "summary.json")
    b_updates = events(s_events, "jb_update", limit)
    old_grad = events(j0 / "events.jsonl", "update", limit)
    tickets = (args.budget / "j1.attempts").read_text().splitlines()
    require(len(tickets) <= jb.SMOKE_CAP and len(b_updates) == len(old_grad) == limit and len(tickets) >= limit
            and tickets == [str(i) for i in range(1, len(tickets) + 1)], "J-B independent ledger differs")
    require(compare(prior, a_batches, keys) and smoke["native_groups"] == j0_summary["native_parameter_groups"]
            and smoke["counts"] == j0_summary["counts"] and smoke["initial_native_sha256"] == j0_summary["initial_native_state_sha256"]
            and len(a_batches) == limit, "J-B altered the matched native TRAIN behavior")
    for i, (update, ticket) in enumerate(zip(b_updates, tickets[-limit:], strict=True), 1):
        opt = update["optimizer"]
        require(i == update["step"] == opt["effective_update"] and int(ticket) == update["ticket"]
                and opt["native_step_called"] and not opt["amp_skipped"], "Native update or budget differs")
        require(math.isclose(update["total_loss"], update["native_loss"] + jb.WEIGHT * update["supcon_loss"],
                             rel_tol=3e-7, abs_tol=2e-5), "Actual loss did not bind the calibrated weight")
    require(all(x["strict_reload"] and x["stripped_native_forward"] for x in smoke["checkpoints"].values())
            and set(smoke["checkpoints"]) == {"1", str(limit)}, "Strict checkpoint or stripped native forward missing")

    probes = cal["probes"]
    layers = {}
    for layer in probes[0]["layers"]:
        records = [p["layers"][layer] for p in probes]
        layers[layer] = {k: bounds([r[k] for r in records]) for k in (
            "weighted_ratio", "cosine", "native_joint_cosine", "nonzero_auxiliary_tensors")}
    snapshots = {}
    for name in cal["snapshot_weighted_medians"]:
        records = [p for p in probes if p["snapshot"] == name]
        snapshots[name] = {
            "backbone_weighted_ratio": bounds([p["backbone"]["weighted_ratio"] for p in records]),
            "backbone_cosine": bounds([p["backbone"]["cosine"] for p in records]),
            "native_joint_cosine": bounds([p["backbone"]["native_joint_cosine"] for p in records]),
            "native_loss": bounds([p["native_loss"] for p in records]),
            "auxiliary_loss": bounds([p["auxiliary_loss"] for p in records]),
            "joint_loss": bounds([p["joint_loss"] for p in records]),
            "clip_coefficient_native": bounds([p["clip_coef"]["native"] for p in records]),
            "clip_coefficient_joint": bounds([p["clip_coef"]["joint"] for p in records]),
            "clip_coefficient_relative_joint_vs_native": bounds([
                p["clip_coef"]["joint"] / p["clip_coef"]["native"] for p in records
            ]),
        }
    old_norms = [p["optimizer"]["gradient_norm"] for p in old_grad]
    clip0 = [min(1.0, 10.0 / (n + 1e-6)) for n in old_norms]
    clip1 = [p["clip_coef"] for p in b_updates]
    requested = ("identity", "runtime", "native_initial_sha256", "training_batches", "coverage", "binding",
                 "featured_binding_cases", "prespecified_screen", "optimizer_attempts")
    summary = {
        "status": jb.PASS, "lambda_s": jb.WEIGHT, "calibration": {k: cal[k] for k in requested},
        "snapshot_metrics": snapshots, "layer_metrics": layers,
        "rejected_lambda_10": {
            "snapshot_medians": rejected["snapshot_weighted_medians"],
            "max_backbone_ratio": rejected["max_backbone_ratio"],
            "min_native_joint_cosine": rejected["min_native_joint_cosine"],
            "optimizer_attempts": 0, "runtime_source_identity": rejected["identity"],
        },
        "metric_triplets": "min/median/max",
        "smoke": {k: smoke[k] for k in (
            "effective_updates", "optimizer_attempts", "native_optimizer_calls", "amp_skips", "post_warmup_steps",
            "counts", "peak_allocated_bytes", "peak_reserved_bytes", "elapsed_seconds",
            "coverage", "checkpoints", "backbone_before", "backbone_after", "projection_before", "projection_after",
        )},
        "loss_equation_verified_steps": limit, "native_equivalent_train_batches": limit,
        "smoke_budget_cap": jb.SMOKE_CAP, "smoke_budget_consumed": len(tickets),
        "smoke_native_bn_only_checked_steps": [p["step"] for p in b_updates if p["bn_native_only_check"]],
        "smoke_native_loss": bounds([p["native_loss"] for p in b_updates]),
        "smoke_supcon_loss": bounds([p["supcon_loss"] for p in b_updates]),
        "smoke_joint_loss": bounds([p["total_loss"] for p in b_updates]),
        "smoke_native_gradient_norm": bounds(old_norms),
        "smoke_joint_gradient_norm": bounds([p["global_gradient_norm"] for p in b_updates]),
        "smoke_native_clip_coef": bounds(clip0), "smoke_joint_clip_coef": bounds(clip1),
        "smoke_steps_clipped_j0_reference": sum(r < 1 for r in clip0),
        "smoke_steps_clipped_j1": sum(r < 1 for r in clip1),
        "smoke_median_postwarmup_audited_seconds": statistics.median(
            p["audited_seconds"] for p in b_updates if p["step"] > smoke["warmup_iterations"]),
        "heldout_inference": {"val": 0, "test": 0}, "j2_updates": 0, "formal_updates": 0,
    }
    args.output.mkdir()
    ja.record_json(args.output / "summary.json", summary)
    ja.record_json(args.output / "calibration_probes.json", {"lambda_s": jb.WEIGHT, "probes": probes})
    ja.record_json(args.output / "formal_protocol.json", jb.formal_protocol(smoke["identity"], sha256_file(cp), sha256_file(sp)))


if __name__ == "__main__":
    main()
