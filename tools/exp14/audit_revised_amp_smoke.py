"""Audit D2 paired Smoke records without a model, optimizer or GPU."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from tools.exp14.audit_admission import read_rows, require, sha256_file
from tools.exp14.summarize_smoke import audit_run, read_json, read_run, verify_payload


GATE = "EXP14_A3_D2_PENDING_CHATGPT_FORMAL_TRAIN_GATE"
CONFIG = (
    "seed", "task", "imgsz", "epochs", "optimizer", "batch", "nbs", "deterministic", "amp", "lr0", "lrf",
    "cos_lr", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "momentum", "weight_decay", "mosaic",
    "close_mosaic", "mixup", "cutmix", "copy_paste", "rect", "multi_scale", "degrees", "translate", "scale",
    "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "bgr",
)


def point(event):
    return event["epoch"], event["batch"]


def values_range(values):
    return [min(values), max(values)] if values else None


def audit_numerics(events, summary, detail, loaded, protocol):
    backwards = [e for e in events if e["kind"] == "backward_complete"]
    steps = [e for e in events if e["kind"] == "native_step_result"]
    updates = [e for e in events if e["kind"] == "optimizer_update"]
    forwards = [e for e in events if e["kind"] == "finite_forward"]
    scaler = [e for e in events if e["kind"] == "amp_protocol"]
    setups = [e for e in events if e["kind"] == "setup"]
    if not events:
        return {}, {}
    require(len(scaler) <= 1 and len(setups) <= 1, "Duplicate numeric setup")
    state = scaler[0]["scaler_state"] if scaler else None
    config = {k: setups[0]["effective_args"][k] for k in CONFIG} if setups else {}
    require(not config or config == {k: protocol[k] for k in CONFIG}, "Effective protocol differs")
    if state:
        require(state["scale"] == 1 and state["growth_factor"] == 2
                and state["growth_interval"] == 2000 and state["backoff_factor"] == 0.5,
                "Effective native scaler settings differ")
    require([point(e) for e in backwards] == [point(e) for e in loaded[:len(backwards)]]
            and [point(e) for e in steps] == [point(e) for e in backwards[:len(steps)]],
            "Numeric evidence is not aligned to the consumed batches")
    require([point(e) for e in updates] == [point(e) for e in steps if e["parameter_changed"]],
            "Parameter update records differ from native outcomes")
    require(sum(e["native_step_called"] for e in steps) == detail["optimizer_calls"]
            and len(updates) == detail["effective_updates"]
            and sum(e["overflow"] for e in steps) == summary["scaler_skips"],
            "Native call, skip or parameter-change counters disagree")
    pairs = [[e["scale_before"], e["scale_after"]] for e in steps]
    require(pairs == summary["scale_history"] and all(b[0] == a[1] for a, b in zip(pairs, pairs[1:])),
            "Scaler trace differs")
    require(all(math.isfinite(e["loss"]) and all(map(math.isfinite, e["components"].values()))
                and e["amp_autocast"] for e in forwards), "Saved loss or autocast evidence differs")
    for b, s in zip(backwards, steps):
        require(b["native_overflow"] == s["overflow"] and b["scale_before"] == s["scale_before"],
                "Native overflow flag or input scale differs")
        if s["overflow"]:
            require(not s["native_step_called"] and not s["parameter_changed"]
                    and s["scale_after"] == 0.5 * s["scale_before"], "Overflow did not skip and back off")
        else:
            require(s["native_step_called"] and s["parameter_changed"]
                    and not b["scaled_gradients"]["nonfinite"] and not b["unscaled_gradients"]["nonfinite"],
                    "Finite-step evidence differs")
    groups = {}
    for row in updates:
        require(math.isfinite(row["gradient_norm"]) and row["gradient_norm"] > 0
                and any(g["changed"] > 0 for g in row["parameter_groups"]), "Ineffective update")
        for g in row["parameter_groups"]:
            require(0 <= g["changed"] <= g["elements"] and math.isfinite(g["max_abs_delta"]),
                    "Invalid active-group parameter delta")
            out = groups.setdefault(g["group"], {"observed_steps": 0, "steps_with_changes": 0, "max_abs_delta": 0.0})
            out["observed_steps"] += 1
            out["steps_with_changes"] += g["changed"] > 0
            out["max_abs_delta"] = max(out["max_abs_delta"], g["max_abs_delta"])
    loss_range = values_range([e["loss"] for e in forwards])
    zero_fractions = [e["unscaled_gradients"]["zero"] / e["unscaled_gradients"]["elements"] for e in backwards]
    gradient_norm = values_range([e["gradient_norm"] for e in updates])
    if summary["status"] == "PASS_DIAGNOSTIC_ONLY":
        require(summary["completed_epochs"] == 2 and len(backwards) == len(steps) == len(updates) == len(forwards) == 44
                and summary["optimizer_attempts"] == 44 and summary["scaler_skips"] == 0 and state is not None
                and summary["initial_parameters_sha256"] != summary["final_parameters_sha256"],
                "Passing run is missing numerical evidence")
        require(detail["worker_epochs_disjoint"] and detail["suppressed_validation_events"] == 2,
                "Epoch worker isolation or validation isolation differs")
    return {
        "completed_epochs": summary["completed_epochs"], "backwards": len(backwards),
        "optimizer_attempts": summary["optimizer_attempts"], "native_step_calls": detail["optimizer_calls"],
        "scaler_skips": summary["scaler_skips"], "effective_updates": len(updates),
        "scaler_initial_state": state, "scale_history": pairs,
        "scaled_nonfinite_batches": sum(e["scaled_gradients"]["nonfinite"] > 0 for e in backwards),
        "unscaled_nonfinite_batches": sum(e["unscaled_gradients"]["nonfinite"] > 0 for e in backwards),
        "unscaled_zero_fraction_range": values_range(zero_fractions),
        "unscaled_gradient_norm_before_clipping_range": gradient_norm,
        "loss_range": loss_range, "active_parameter_groups": groups,
        "first_batch_unscaled_gradients": backwards[0]["unscaled_gradients"] if backwards else None,
        "first_forward_loss_components": forwards[0]["components"] if forwards else None,
        "effective_config": config,
    }, {"first_loss": forwards[0]["loss"] if forwards else None,
        "first_parts": forwards[0]["components"] if forwards else None}


def summarize(runtime, payload, d1):
    manifest = verify_payload(payload)
    protocol = read_json(payload / "training_protocol.json")
    slots = {row["slot_id"]: row for row in read_rows(payload / "matched_slots.csv")}
    plan = read_json(payload / "epoch_slot_order.json")
    require(len(slots) == 684 and len(plan) == 100 and plan[0]["batches"] != plan[1]["batches"],
            "Frozen plan identity differs")
    d1_identity = read_json(d1 / "first_batch_identity.json")
    d1_summary = read_json(d1 / "numerical_summary.json")
    require(d1_summary["classification"] == "AMP_SCALE_OVERFLOW_LIKELY"
            and d1_summary["completed_cases"] == 5 and d1_summary["model_optimizer_updates"] == 0,
            "D1 evidence differs")
    report, loaded_by_arm, summaries, environments, firsts, exposure = {}, {}, {}, {}, {}, []
    for arm in ("e0", "e1"):
        summary, events = read_run(runtime, "train", arm)
        detail, draws, loaded = audit_run(events, summary, "train", arm, slots, plan)
        numerics, firsts[arm] = audit_numerics(events, summary, detail, loaded, protocol)
        report[arm] = {**detail, "numerics": numerics}
        summaries[arm], loaded_by_arm[arm] = summary, loaded
        exposure.extend((arm, *key, count) for key, count in sorted(draws.items()))
        p = runtime / f"train_{arm}/environment.json"
        if p.is_file():
            env = read_json(p)
            require(env["payload_files"] == len(manifest["files"])
                    and env["protocol_sha256"] == sha256_file(payload / "training_protocol.json")
                    and env["payload_manifest_sha256"] == sha256_file(payload / "transport_manifest.json"),
                    "Runtime transport or protocol evidence differs")
            environments[arm] = env
    passed = all(s["status"] == "PASS_DIAGNOSTIC_ONLY" for s in summaries.values())
    if passed:
        require(summaries["e0"]["initial_state_sha256"] == summaries["e1"]["initial_state_sha256"]
                == d1_identity["initial_state_sha256"]
                and summaries["e0"]["initial_parameters_sha256"] == summaries["e1"]["initial_parameters_sha256"],
                "Initial models differ")
        require(all([r["slot_id"] for r in a["slots"]] == [r["slot_id"] for r in b["slots"]]
                    and a["lrs"] == b["lrs"] for a, b in zip(loaded_by_arm["e0"], loaded_by_arm["e1"], strict=True)),
                "Matched slot or LR trajectory differs")
        require(all(environments["e0"][k] == environments["e1"][k]
                    for k in ("weight_sha256", "d2_adapter_sha256", "git_head", "ultralytics", "torch")),
                "Arm environment identity differs")
    first_match = (bool(firsts["e0"]) and firsts["e0"]["first_loss"] == d1_identity["historical_loss"]
                   and firsts["e0"]["first_parts"] == d1_identity["historical_components"])
    with (runtime / "actual_exposure.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("arm", "epoch", "source_id", "group_id", "image_kind", "draws"))
        writer.writerows(exposure)
    result = {
        "stage": "Exp14-A3-D2", "gate": GATE, "diagnostic_only": True,
        "status": "PASS_BOUNDED_SMOKE" if passed else "STOP_BOUNDED_SMOKE",
        "paired_effective_updates_match": passed, "paired_slots_and_lr_match": passed,
        "paired_initial_state_matches_d1": passed, "d1_first_loss_and_parts_exactly_match": first_match,
        "payload_files_verified": len(manifest["files"]), "ui_risk_flags_retained": len(manifest["ui_risks"]),
        "environment": {arm: {k: v for k, v in env.items() if k != "ui_risk_flags"}
                        for arm, env in environments.items()},
        "runs": report, "actual_validation_runs": 0, "formal_training_started": False,
    }
    with (runtime / "revised_smoke_summary.json").open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    args = parser.parse_args()
    summarize(*(p.resolve(strict=True) for p in (args.runtime, args.payload, args.d1)))
