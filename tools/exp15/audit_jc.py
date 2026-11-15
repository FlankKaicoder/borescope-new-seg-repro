"""Independently audit J-C training and publish only aggregate results and batch digests."""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from ultralytics.nn.tasks import SegmentationModel

from tools.exp14.a11_full_numerics import adam_step_counts, expected_scaler_state, validate_model_states
from tools.exp14.a11_smoke_numerics import exact_tree, finite_tree
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp15 import ja, jb
from tools.exp15.preflight_jc import APPROVED_HEAD, APPROVED_PROTOCOL_SHA
from tools.exp15.regions import add_projection

STEPS = 2100
BATCHES = 21
COMPLETE = "EXP15_J0_J1_FORMAL_TRAIN_COMPLETE_PENDING_VAL_GATE"
HARD_GATE = "EXP15_FORMAL_TRAIN_HARD_GATE"
FIELDS = ("batch", "epoch", "source_paths", "raw_targets", "augmented_targets", "lrs")


def by_event(path, live=False):
    result = collections.defaultdict(list)
    # The writer opens/appends/closes one row at a time. Ignore incomplete live tails only.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if live and not line.endswith("\n"):
                break
            row = json.loads(line)
            result[row["event"]].append(row)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def expected_lr(step, config):
    epoch = (step - 1) // BATCHES
    dest = config["lr0"] * ((1 - epoch / config["epochs"]) * (1 - config["lrf"]) + config["lrf"])
    if step > 63:
        return [dest] * 3
    initial = [0, 0, config["warmup_bias_lr"]]
    return [lr + (dest - lr) * (step - 1) / 63 for lr in initial]


def compare_batches(left, right, limit):
    for i in range(limit):
        require(all(left[i][k] == right[i][k] for k in FIELDS), f"Paired native batch differs at step {i + 1}")


def compare_prefix(root):
    j0 = by_event(root / "j0" / "events.jsonl")
    j1 = by_event(root / "j1" / "events.jsonl", live=True)
    n = len(j1["batch_read"])
    compare_batches(j0["batch_read"], j1["batch_read"], n)
    for a, b in zip(j0["jb_update"], j1["jb_update"]):
        require(a["step"] == b["step"] and a["batch_sha256"] == b["batch_sha256"] and a["lr"] == b["lr"],
                f"Paired augmented tensors or schedule differs at step {b['step']}")
    return [n, len(j1["jb_update"])]


def check_terminal(record):
    require(record["class_names"] == dict(enumerate(CLASSES)), "Checkpoint class map differs")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(record["model_config"], ch=3, nc=7, verbose=False)
        if record["arm"] == "j1":
            add_projection(model)
    for key in ("ema", "state"):
        model.load_state_dict(record[key], strict=True)
        require(exact_tree(dict(model.state_dict()), record[key]), "Checkpoint strict CPU weight reload differs")
    names = dict(model.named_parameters())
    optim = torch.optim.AdamW([{"params": [names[n] for n in group]} for group in record["optimizer_names"]])
    optim.load_state_dict(record["optimizer"])
    scaler = torch.amp.GradScaler("cpu")
    scaler.load_state_dict(record["scaler"])
    require(exact_tree(optim.state_dict(), record["optimizer"]) and scaler.state_dict() == record["scaler"],
            "Checkpoint strict CPU AdamW or GradScaler reload differs")
    if record["arm"] == "j1":
        del model.region_projection
    native_master = {n: v for n, v in record["state"].items() if not n.startswith(ja.HEAD_PREFIX)}
    model.load_state_dict(native_master, strict=True)
    require(hash_state(model.state_dict()) == record["native_state_sha256"], "Stripped original architecture differs")
    model.names = record["class_names"]
    model.eval()
    with torch.no_grad():
        observation, _ = model(torch.zeros(1, 3, 64, 64))
    require(finite_tree(observation), "Native model synthetic output is nonfinite")
    return record["native_state_sha256"]


def check_checkpoints(run, summary, arm):
    expected = [1, *range(105, STEPS + 1, 105)]
    recorded = summary["checkpoints"]
    require(set(map(int, recorded)) == set(expected), "Missing or extra checkpoint summary steps")
    names = {("last.pt" if step == STEPS else f"step_{step:04}.pt") for step in expected}
    require({p.name for p in run.glob("*.pt")} == names, "Checkpoint files disagree with saved schedule")
    history = []
    for step in expected:
        path = run / ("last.pt" if step == STEPS else f"step_{step:04}.pt")
        declared = recorded[str(step)]
        sha = sha256_file(path)
        require(declared["sha256"] == sha and declared["strict_reload"] and declared["stripped_native_forward"],
                "Checkpoint readback SHA or training verification differs")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(saved["schema"] == ja.SEAL and saved["arm"] == arm and saved["identity"] == summary["identity"]
                and saved["effective_updates"] == saved["ema_updates"] == step,
                "Checkpoint lineage or update counter differs")
        validate_model_states(saved["state"], saved["ema"])
        require(finite_tree([saved["state"], saved["ema"], saved["optimizer"], saved["scheduler"]]),
                "Checkpoint contains nonfinite state")
        master, ema = hash_state(saved["state"]), hash_state(saved["ema"])
        require(declared["state_sha256"] == saved["model_state_sha256"] == master != ema == saved["ema_state_sha256"],
                "Checkpoint FP32 master and EMA identities differ from the record")
        require(saved["scaler"] == expected_scaler_state(step) and saved["scheduler"]["last_epoch"] == (step - 1) // BATCHES,
                "Checkpoint scheduler or AMP state differs")
        slots = adam_step_counts(saved["optimizer"]["state"], step)
        require(slots == {step: 290 + (4 if arm == "j1" else 0)}, "Checkpoint AdamW slot trajectory differs")
        groups = saved["optimizer_names"]
        native = [[n for n in group if not n.startswith(ja.HEAD_PREFIX)] for group in groups]
        require(native == summary["native_groups"] and sum(map(len, groups)) == summary["optimizer_unique_parameters"],
                "Checkpoint optimizer parameter groups differ")
        history.append({"step": step, "sha256": sha, "master_state_sha256": master, "ema_state_sha256": ema,
                        "master_native_sha256": saved["native_state_sha256"]})
        if step == STEPS:
            require(check_terminal(saved) == declared["native_state_sha256"], "Terminal native state reload failed")
    return history


def check_arm(root, arm, p, preflight, expected):
    run = root / arm
    exit_row = (root / f"{arm}.launch.exit").read_text().strip().split("\t")
    require(len(exit_row) == 2 and exit_row[1] == "0" and not (run / "failed.json").exists(),
            f"{arm} launcher or run terminated with a hard failure")
    summary = jb.load(run / "summary.json")
    require(summary["status"] == "PASS_FORMAL_ARM" and summary["variant"] == arm
            and summary["identity"] == preflight["identity"] and summary["runtime"] == preflight["runtime"]
            and summary["initial_native_sha256"] == preflight["native_initial_sha256"]
            and summary["lambda_s"] == p["lambda_s"][arm] and summary["backbone_before"] != summary["backbone_after"],
            f"{arm} formal readback or initialization differs")
    require(summary["effective_updates"] == summary["optimizer_attempts"] == summary["native_optimizer_calls"] == STEPS
            and summary["amp_skips"] == 0 and summary["attempt_cap"] == STEPS
            and summary["warmup_iterations"] == 63 and summary["post_warmup_steps"] == STEPS - 63,
            f"{arm} recorded budget differs")
    events = by_event(run / "events.jsonl")
    require(set(events) == {"setup", "batch_read", "jb_update"} and len(events["setup"]) == 1,
            f"{arm} has unexpected events")
    setup, batches, updates = events["setup"][0], events["batch_read"], events["jb_update"]
    require(setup["train_images"] == 668 and setup["scaler"] == expected_scaler_state(0)
            and setup["variant"] == arm and len(batches) == len(updates) == len(summary["updates"]) == STEPS,
            f"{arm} setup or event count differs")
    tickets = (root / "formal_budget" / f"{arm}.attempts").read_text().splitlines()
    require(tickets == [str(n) for n in range(1, STEPS + 1)], f"{arm} formal ticket ledger differs")
    scopes = []
    stats = collections.Counter()
    seen = set()
    rows = []
    scale_changes, max_lr_error = [], 0.0
    weight = p["lambda_s"][arm]
    for i, (batch, up, declared) in enumerate(zip(batches, updates, summary["updates"], strict=True), 1):
        epoch = (i - 1) // BATCHES + 1
        tail = i % BATCHES == 0
        paths, raw, augmented = (batch[k] for k in ("source_paths", "raw_targets", "augmented_targets"))
        require(batch["epoch"] == epoch and batch["batch"] == up["step"] == up["ticket"] == i
                and batch["batch_type"] == "all_original" and batch["repaired_images"] == 0
                and len(paths) == len(raw) == len(augmented) == batch["original_images"] == (28 if tail else 32)
                and len(set(paths)) == len(paths) and not seen.intersection(paths)
                and all(path in expected and expected[path] == n and 0 <= m <= n
                        for path, n, m in zip(paths, raw, augmented, strict=True)),
                f"{arm} TRAIN membership or annotations differ at step {i}")
        seen.update(paths)
        if tail:
            require(seen == expected.keys(), f"{arm} full epoch TRAIN set differs at epoch {epoch}")
            seen.clear()
        error = max(abs(a - b) for a, b in zip(up["lr"], expected_lr(i, p["train"]), strict=True))
        max_lr_error = max(max_lr_error, error)
        require(batch["lrs"] == up["lr"] and error <= 1e-14 and all(up[k] == v for k, v in declared.items()),
                f"{arm} learning rate or update readback differs at step {i}")
        opt = up["optimizer"]
        require(opt["effective_update"] == i and opt["optimizer_attempted"] and opt["native_step_called"]
                and opt["parameter_changed"] and opt["model_state_fp32_finite"] and opt["optimizer_state_finite"]
                and not opt["overflow"] and not opt["amp_skipped"]
                and opt["native_found_inf"] == {"cuda:0": 0.0}
                and all(opt[k]["nonfinite"] == 0 and opt[k]["master_dtype_fp32"] for k in ("scaled_gradients", "unscaled_gradients"))
                and opt["scaler_state"] == expected_scaler_state(i) and opt["scale_after"] == expected_scaler_state(i)["scale"]
                and opt["scale_before"] == expected_scaler_state(i - 1)["scale"], f"{arm} AMP or numeric gate differs at step {i}")
        if opt["scale_after"] != opt["scale_before"]:
            scale_changes.append(i)
        native = up["native_loss"]
        auxiliary = up["supcon_loss"]
        total = up["total_loss"]
        require(finite_tree([native, auxiliary, total, up["global_gradient_norm"], up["audited_seconds"]])
                and up["global_gradient_norm"] == opt["gradient_norm"] > 0
                and math.isclose(total, native + weight * (auxiliary or 0), rel_tol=3e-7, abs_tol=2e-5),
                f"{arm} nonfinite loss or loss equation differs at step {i}")
        stats.update(images=len(paths), raw=sum(raw), augmented=sum(augmented), bn=up["bn_native_only_check"],
                     clipped=up["clip_coef"] < 1)
        coverage = up["coverage"]
        if arm == "j1":
            require(auxiliary is not None and up["projection_nonzero_gradients"] == 4 and coverage["eligible_anchors"] > 0
                    and coverage["targets"] == sum(augmented), f"{arm} auxiliary coverage or gradient differs at step {i}")
            scopes.append(coverage)
        else:
            require(auxiliary is None and up["projection_nonzero_gradients"] == 0 and coverage is None,
                    f"{arm} unexpectedly used auxiliary loss at step {i}")
        rows.append((i, epoch, len(paths), sum(raw), sum(augmented), digest(paths), up["batch_sha256"], *up["lr"]))
    require(stats["images"] == 66800 and stats["raw"] == 126600 and scale_changes == [2000]
            and summary["counts"] == {"all_original": STEPS, "augmented_instances": stats["augmented"],
                                     "batches_read": STEPS, "original_exposures": stats["images"],
                                     "raw_instance_exposures": stats["raw"], "repair_exposures": 0},
            f"{arm} exposure or scaler totals differ")
    if arm == "j1":
        require(ja.summarize_regions(scopes) == summary["coverage"]
                and all(summary["coverage"]["eligible_anchors_by_class"])
                and summary["projection_before"] != summary["projection_after"],
                f"{arm} auxiliary exposure or projection identity differs")
    checkpoints = check_checkpoints(run, summary, arm)
    names = ("effective_updates", "optimizer_attempts", "native_optimizer_calls", "amp_skips", "initial_native_sha256",
             "lambda_s", "peak_allocated_bytes", "peak_reserved_bytes", "elapsed_seconds", "optimizer_unique_parameters",
             "backbone_before", "backbone_after", "projection_before", "projection_after", "coverage")
    numbers = {k: [min(values), max(values)] for k in ("native_loss", "supcon_loss", "total_loss", "global_gradient_norm",
                                                       "clip_coef")
               if (values := [u[k] for u in updates if u[k] is not None])}
    arm_result = {**{k: summary[k] for k in names}, "exposure": dict(stats), "max_lr_absolute_error": max_lr_error,
                  "amp_growth_after_steps": scale_changes, "final_scaler": expected_scaler_state(STEPS),
                  "loss_and_gradient_ranges": numbers, "checkpoints": checkpoints, "terminal_strict_cpu_reload": True,
                  "stripped_native_synthetic_prediction": True, "exit_timestamp": exit_row[0],
                  "events_sha256": sha256_file(run / "events.jsonl"), "summary_sha256": sha256_file(run / "summary.json"),
                  "tickets_sha256": sha256_file(root / "formal_budget" / f"{arm}.attempts")}
    return batches, updates, rows, arm_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("j0", "prefix", "pair"))
    for name in ("root", "release", "destination", "historical-e0"):
        parser.add_argument(f"--{name}", type=Path)
    a = parser.parse_args()
    require(a.root is not None, "Run root is required")
    if a.phase == "prefix":
        print(compare_prefix(a.root))
        return
    require(a.release and a.destination, "Release and destination are required")
    require(sha256_file(a.root / "private" / "formal_protocol.json") == APPROVED_PROTOCOL_SHA,
            "Audited formal protocol SHA differs")
    p = jb.load(a.root / "private" / "formal_protocol.json")
    preflight = jb.load(a.root / "private" / "preflight.json")
    require(preflight["approved_head"] == APPROVED_HEAD and preflight["identity"] == p["identity"],
            "Preflight deployment identity differs")
    now = ja.checked_train(a.release.resolve(strict=True))
    require(len(now) == 668 and len(set(now.values())) == preflight["train_groups"], "Frozen TRAIN postflight differs")
    inv = jb.load(a.release / "configs" / "train_entry_files.json")["baseline"]
    expected = {row["image_path"]: row["instance_count"] for row in inv}
    torch.set_num_threads(4)
    left = check_arm(a.root, "j0", p, preflight, expected)
    if a.phase == "j0":
        if a.historical_e0:
            old = by_event(a.historical_e0)
            compare_batches(left[0], old["batch_read"], STEPS)
        ja.record_json(a.destination, {"status": "PASS_JC_J0_PRE_J1", "arm": left[3], "protocol_sha256": APPROVED_PROTOCOL_SHA})
        return
    accepted_path = a.root / "private" / "j0_accepted.json"
    accepted = jb.load(accepted_path)
    require(accepted["status"] == "PASS_JC_J0_PRE_J1" and accepted["arm"] == left[3],
            "J0 acceptance changed since J1 launch")
    right = check_arm(a.root, "j1", p, preflight, expected)
    compare_batches(left[0], right[0], STEPS)
    require(left[2] == right[2], "Full augmented tensor/order/learning-rate traces differ")
    native_groups = [jb.load(a.root / arm / "summary.json")["native_groups"] for arm in ("j0", "j1")]
    require(native_groups[0] == native_groups[1]
            and left[3]["initial_native_sha256"] == right[3]["initial_native_sha256"]
            and left[3]["optimizer_unique_parameters"] + 4 == right[3]["optimizer_unique_parameters"],
            "Native optimizer group or initialization alignment differs")
    a.destination.mkdir()
    trace = a.destination / "matched_batch_ledger.csv"
    with trace.open("x", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("step", "epoch", "images", "raw_targets", "augmented_targets", "ordered_sources_sha256",
                         "augmented_tensors_sha256", "lr_no_decay", "lr_decay", "lr_bias"))
        writer.writerows(left[2])
    report = {"status": COMPLETE, "approved_head": APPROVED_HEAD, "protocol_sha256": APPROVED_PROTOCOL_SHA,
              "identity": p["identity"], "runtime": preflight["runtime"], "input_preflight": preflight,
              "matched_epochs": 100, "matched_batches": STEPS, "matched_initial_native_and_optimizer_groups": True,
              "matched_train_members_order_augmentation_exposure_and_lr": True, "matched_amp_schedule": True,
              "train_files_hash_verified_pre_and_post": 1336, "synthetic_native_model_reload_only": True,
              "val_model_inference": 0, "test_model_inference": 0, "j2_updates": 0,
              "matched_batch_ledger_sha256": sha256_file(trace), "j0_acceptance_before_j1_sha256": sha256_file(accepted_path)}
    for name, state in (("j0", left[3]), ("j1", right[3])):
        ja.record_json(a.destination / f"{name}_summary.json", state)
    ja.record_json(a.destination / "pair_summary.json", report)


if __name__ == "__main__":
    main()
