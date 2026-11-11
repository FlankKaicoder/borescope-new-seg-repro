"""Independently audit A4 batch budgets, numeric traces, exposures, and final states."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from tools.exp14.audit_admission import read_rows, require, sha256_file
from tools.exp14.formal_e0_e1 import BASE_GIT, BATCHES, BATCHES_PER_EPOCH, EPOCHS, RUN_IDENTITIES, validate_checkpoint
from tools.exp14.revised_amp_smoke import FROZEN_PAYLOAD_MANIFEST_SHA256, FROZEN_PROTOCOL_SHA256
from tools.exp14.smoke_trainer import verify_payload, write_json


EVENTS = ("batch_loaded", "finite_forward", "backward_complete", "native_step_result", "optimizer_update")


def coordinates(position):
    epoch, index = divmod(position, BATCHES_PER_EPOCH)
    return epoch + 1, index + 1


def actual_point(record):
    return record["epoch"], record["batch"]


def expected_lr(position, protocol):
    epoch, _ = coordinates(position)
    base = protocol["lr0"] * ((1 - (epoch - 1) / EPOCHS) * (1 - protocol["lrf"]) + protocol["lrf"])
    warmup = round(protocol["warmup_epochs"] * BATCHES_PER_EPOCH)
    if position >= warmup:
        return [base, base, base]
    t = position / warmup
    return [base * t, base * t, protocol["warmup_bias_lr"] + (base - protocol["warmup_bias_lr"]) * t]


def audit_arm(root, arm, slots, plan, protocol, exposure, base_git=BASE_GIT):
    folder = root / f"train_{arm}"
    if not folder.is_dir():
        return {"status": "NOT_STARTED"}, {}
    summary_file = folder / "summary.json"
    if not summary_file.is_file():
        return {"status": "NO_FINAL_SUMMARY", "has_events": (folder / "events.jsonl").is_file()}, {}
    summary = json.loads(summary_file.read_text())
    env = json.loads((folder / "environment.json").read_text())
    require(env["git_head"] == base_git and env["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
            and env["payload_manifest_sha256"] == FROZEN_PAYLOAD_MANIFEST_SHA256,
            "Formal runtime identity differs")
    traces = {kind: [] for kind in EVENTS}
    epochs, workers, counts, probes, anomalies, configs = [], {}, Counter(), {}, [], []
    with (folder / "events.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            require(event["arm"] == arm and event["phase"] == "train", "Trace arm differs")
            kind = event["kind"]
            if kind == "batch_loaded":
                i = len(traces[kind])
                ep, batch = coordinates(i)
                require(i < BATCHES and actual_point(event) == (ep, batch)
                        and event["accumulate"] == 1, "Unexpected batch position or accumulation")
                expected = plan[ep - 1]["batches"][batch - 1]
                records = event["slots"]
                require([r["slot_id"] for r in records] == expected and len(records) == (12 if batch == 22 else 32),
                        "Consumed slot sequence or batch size differs")
                require(all(math.isclose(x, y, abs_tol=1e-14, rel_tol=1e-12)
                            for x, y in zip(event["lrs"], expected_lr(i, protocol), strict=True)),
                        "Observed LR differs from frozen schedule")
                for row in records:
                    slot = slots[row["slot_id"]]
                    require(row["epoch"] == ep and row["im_file"].endswith("/dataset/" + slot[f"{arm}_image_path"])
                            and row["raw_instances"] == int(slot[f"{arm}_instance_count"])
                            and row["actual_reads"] == [row["dataset_index"]] and row["worker_id"] in {0, 1},
                            "Actual worker, source, target or secondary-read identity differs")
                    kind_name = "aligned" if arm == "e1" and slot["slot_kind"] == "EXTRA" else "original"
                    counts[(ep, kind_name)] += 1
                    counts[(ep, "instances")] += row["raw_instances"]
                    counts[(ep, "empty")] += not row["raw_instances"]
                    exposure[(arm, ep, slot["source_id"], slot["group_id"], kind_name)] += 1
                    workers.setdefault(ep, set()).add(row["worker_pid"])
                traces[kind].append({k: event[k] for k in ("epoch", "batch", "samples", "lrs")})
            elif kind in traces:
                require(actual_point(event) == coordinates(len(traces[kind])), "Numerical event position differs")
                traces[kind].append(event)
            elif kind == "epoch_complete":
                epochs.append(event)
            elif kind == "gradient_anomaly":
                anomalies.append(event)
            elif kind == "setup":
                configs.append(event)
            elif kind == "augmentation_probe":
                ep, batch = actual_point(event)
                keys = plan[ep - 1]["batches"][batch - 1]
                require(batch in {1, 22} and (ep, batch) not in probes
                        and len(keys) == len(event["image_tensor_sha256"]), "Probe identity differs")
                probes[(ep, batch)] = dict(zip(keys, event["image_tensor_sha256"], strict=True))
    loaded, forwards, backwards, steps, updates = (traces[kind] for kind in EVENTS)
    require(len(loaded) == summary["batches"] and len(forwards) == summary["forward_checks"]
            and len(updates) == summary["effective_updates"]
            and sum(s["overflow"] for s in steps) == summary["scaler_skips"], "Summary counters differ")
    require(all(math.isfinite(r["loss"]) and r["amp_autocast"]
                and all(map(math.isfinite, r["components"].values())) for r in forwards), "Invalid forward evidence")
    require(all(math.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0
                and r["parameter_changed"] and any(g["changed"] > 0 for g in r["parameter_groups"])
                and all(math.isfinite(g["max_abs_delta"]) for g in r["parameter_groups"]) for r in updates),
            "Invalid effective update evidence")
    for b, s in zip(backwards, steps):
        require(actual_point(b) == actual_point(s) and b["native_overflow"] == s["overflow"]
                and b["scale_before"] == s["scale_before"], "Overflow or scale evidence differs")
        if s["overflow"]:
            require(not s["native_step_called"] and not s["parameter_changed"]
                    and s["scale_after"] == s["scale_before"] * 0.5, "Overflow did not skip and back off")
        else:
            require(s["native_step_called"] and s["parameter_changed"]
                    and not b["scaled_gradients"]["nonfinite"] and not b["unscaled_gradients"]["nonfinite"],
                    "Unhandled non-finite or ineffective update")
    passed = summary["status"] == "PASS_FORMAL_ARM"
    if passed:
        require(len(loaded) == len(forwards) == len(backwards) == len(steps) == len(updates)
                == summary["optimizer_attempts"] == summary["native_step_calls"] == BATCHES
                and len(epochs) == summary["completed_epochs"] == EPOCHS and summary["scaler_skips"] == 0,
                "Complete arm lacks exact budget evidence")
        require(len(probes) == EPOCHS * 2 and len(configs) == 1
                and all(configs[0]["effective_args"][k] == protocol[k] for k in (
                    "seed", "optimizer", "batch", "nbs", "imgsz", "epochs", "lr0", "lrf", "warmup_epochs",
                    "amp", "mosaic", "mixup", "cutmix", "copy_paste", "rect", "multi_scale")),
                "Effective runtime config or probe coverage differs")
        for ep in range(1, EPOCHS + 1):
            require(counts[ep, "original"] == (684 if arm == "e0" else 668)
                    and counts[ep, "aligned"] == (0 if arm == "e0" else 16)
                    and counts[ep, "instances"] == (1284 if arm == "e0" else 1266)
                    and counts[ep, "empty"] == (0 if arm == "e0" else 16)
                    and len(workers[ep]) == 2
                    and (ep == 1 or workers[ep].isdisjoint(workers[ep - 1])),
                    "Exposure totals or epoch worker isolation differ")
        require(all(s["scale_before"] == (1 if i < 2000 else 2)
                    and s["scale_after"] == (1 if i < 1999 else 2)
                    for i, s in enumerate(steps)), "Actual scaler growth differs")
        path = folder / "last.pt"
        require(sha256_file(path) == summary["checkpoint"]["sha256"], "Checkpoint hash differs")
        validate_checkpoint(path, arm)
    zero = [b["unscaled_gradients"]["zero"] / b["unscaled_gradients"]["elements"] for b in backwards]
    maxima = sorted(range(len(zero)), key=lambda i: zero[i], reverse=True)[:5]
    result = {
        **summary, "environment": env, "events_sha256": sha256_file(folder / "events.jsonl"),
        "batch_sequence_verified": True, "native_warmup_and_lr_verified": True,
        "original_draws": sum(v for (_, kind), v in counts.items() if kind == "original"),
        "aligned_draws": sum(v for (_, kind), v in counts.items() if kind == "aligned"),
        "raw_instances": sum(v for (_, kind), v in counts.items() if kind == "instances"),
        "explicit_empty_draws": sum(v for (_, kind), v in counts.items() if kind == "empty"),
        "training_epoch_seconds_total": sum(e["elapsed_seconds"] for e in epochs),
        "batches_with_zero_fraction_ge_1pct": sum(x >= 0.01 for x in zero),
        "highest_zero_fractions": [{"epoch": coordinates(i)[0], "batch": coordinates(i)[1], "fraction": zero[i]}
                                   for i in maxima],
        "growth_window": [{"update": i + 1, "epoch": coordinates(i)[0], "batch": coordinates(i)[1],
                           "scale_before": steps[i]["scale_before"], "scale_after": steps[i]["scale_after"],
                           "loss": forwards[i]["loss"], "zero_fraction": zero[i],
                           "scaled_nonfinite": backwards[i]["scaled_gradients"]["nonfinite"],
                           "unscaled_nonfinite": backwards[i]["unscaled_gradients"]["nonfinite"],
                           "gradient_norm": updates[i]["gradient_norm"] if i < len(updates) else None,
                           "effective_update": steps[i]["parameter_changed"]}
                          for i in range(1994, min(2005, len(steps)))],
    }
    require(len(anomalies) == len(summary["anomalies"]), "Anomaly summary differs")
    if base_git != BASE_GIT:
        strides = ("8", "16", "32")
        expected_points = [actual_point(b) for b, fraction in zip(backwards, zero, strict=True)
                           if fraction >= env["numerical_policy"]["zero_diagnostics_fraction"] or b["native_overflow"]]
        require(env["numerical_policy"]["zero_fraction_action"] == "observe"
                and [actual_point(row) for row in anomalies] == expected_points
                and all(set(row["foreground"]) == set(strides)
                        and all(type(n) is int and n >= 0 for n in row["foreground"].values()) for row in anomalies),
                "R1 foreground evidence does not cover all flagged batches")
        result["foreground_at_flagged_batches"] = {
            stride: {"zero": sum(row["foreground"][stride] == 0 for row in anomalies),
                     "positive": sum(row["foreground"][stride] > 0 for row in anomalies)}
            for stride in strides
        }
        result["flagged_empty_supervised_branches"] = {
            f"{head}.{i}": sum(row["foreground"][stride] > 0 and (
                f"model.23.{head}.{i}" not in row["layers"] or
                row["layers"][f"model.23.{head}.{i}"]["zero"] == row["layers"][f"model.23.{head}.{i}"]["elements"]
            ) for row in anomalies)
            for head in ("cv2", "cv4") for i, stride in enumerate(strides)
        }
    return result, probes


def run(runtime, payload, revision="a4"):
    base_git, gate = RUN_IDENTITIES[revision]
    require(sha256_file(payload / "transport_manifest.json") == FROZEN_PAYLOAD_MANIFEST_SHA256
            and sha256_file(payload / "training_protocol.json") == FROZEN_PROTOCOL_SHA256, "Frozen inputs differ")
    verify_payload(payload)
    slots = {row["slot_id"]: row for row in read_rows(payload / "matched_slots.csv")}
    plan = json.loads((payload / "epoch_slot_order.json").read_text())
    protocol = json.loads((payload / "training_protocol.json").read_text())
    require(len(plan) == EPOCHS and len(slots) == 684, "Unexpected frozen plan")
    exposures = Counter()
    arms, probes = {}, {}
    for arm in ("e0", "e1"):
        arms[arm], probes[arm] = audit_arm(runtime, arm, slots, plan, protocol, exposures, base_git)
    passed = all(row["status"] == "PASS_FORMAL_ARM" for row in arms.values())
    compared, equal, different_inputs = 0, 0, 0
    if passed:
        e0, e1 = arms["e0"], arms["e1"]
        require(e0["initial_state_sha256"] == e1["initial_state_sha256"]
                and all(e0["environment"][k] == e1["environment"][k] for k in
                        ("weight_sha256", "trainer_sha256", "protocol_sha256", "payload_manifest_sha256")),
                "Paired training identity differs")
        if revision == "r1":
            require(all(e0["environment"][k] == e1["environment"][k] for k in
                        ("diagnosis_sha256", "foreground_audit_sha256")), "Paired foreground audit identity differs")
        require(probes["e0"].keys() == probes["e1"].keys(), "Probe positions differ")
        for point, left in probes["e0"].items():
            right = probes["e1"][point]
            require(left.keys() == right.keys(), "Probe slots differ")
            for key, digest in left.items():
                if slots[key]["e0_image_sha256"] == slots[key]["e1_image_sha256"]:
                    compared += 1
                    equal += digest == right[key]
                else:
                    different_inputs += 1
    with (runtime / "actual_exposure.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("arm", "epoch", "source_id", "group_id", "image_kind", "draws"))
        writer.writerows((*key, count) for key, count in sorted(exposures.items()))
    result = {"status": "PASS_PAIRED_FORMAL_BUDGET" if passed else "STOP_FORMAL_PAIR",
              "gate": gate, "base_git": base_git, "arms": arms,
              "augmentation_probe": {"positions": "first and tail batch of each epoch",
                                     "same_input_pairs": compared, "identical_image_tensor_pairs": equal,
                                     "different_input_pairs": different_inputs, "labels_compared": False}}
    write_json(runtime / "training_audit.json", result)
    print(json.dumps({"status": result["status"], "arms": {a: r["status"] for a, r in arms.items()},
                      "augmentation_probe": result["augmentation_probe"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--revision", choices=RUN_IDENTITIES, default="a4")
    args = parser.parse_args()
    run(args.runtime.resolve(strict=True), args.payload.resolve(strict=True), args.revision)
