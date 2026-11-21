"""Independent J3 terminal/numerical audit against both immutable Exp15 arms."""

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.tasks import SegmentationModel

from tools.exp14.a11_full_numerics import adam_step_counts, expected_scaler_state, validate_model_states
from tools.exp14.a11_smoke_numerics import exact_tree, finite_tree
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp15 import audit_jc, ja

STEPS = 2100
SCHEMA = "exp16_j3_region_ce_v1"
PASS = "PASS_EXP16_J3_FORMAL_ARM"
PREFIX = "region_projection."


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def check_ce_coverage(geometry, supervised, images):
    """Verify CE uses every nonempty pooled polygon, including non-anchors."""
    require(set(supervised) == {"valid_regions", "ce_regions_by_class", "ce_groups_by_class"},
            "J3 CE supervision fields differ")
    counts, groups = supervised["ce_regions_by_class"], supervised["ce_groups_by_class"]
    expected = [n - missing for n, missing in zip(
        geometry["targets_by_class"], geometry["raster_missing_by_class"], strict=True)]
    require(len(counts) == len(groups) == len(expected) == 7
            and type(supervised["valid_regions"]) is int
            and all(type(n) is int and n >= 0 for n in counts + groups)
            and counts == expected and sum(counts) == supervised["valid_regions"]
            == geometry["targets"] - geometry["missing_native_raster"] > 0,
            "J3 CE did not supervise all valid polygon regions")
    require(all((group == 0 if count == 0 else 1 <= group <= min(count, images))
                and group >= eligible for count, group, eligible in zip(
                    counts, groups, geometry["eligible_groups_by_class"], strict=True)),
            "J3 CE source-group support bounds differ")


def summarize_ce_coverage(rows):
    return {"batches": len(rows), "valid_regions": sum(r["valid_regions"] for r in rows),
            "zero_valid_batches": sum(r["valid_regions"] == 0 for r in rows),
            "ce_regions_by_class": [sum(r["ce_regions_by_class"][c] for r in rows) for c in range(7)],
            "ce_group_exposures_by_class": [sum(r["ce_groups_by_class"][c] for r in rows) for c in range(7)],
            "batches_with_regions_by_class": [sum(r["ce_regions_by_class"][c] > 0 for r in rows) for c in range(7)]}


def strict_terminal(record):
    """Construct the declared architecture independently of the training loader."""
    require(record["class_names"] == dict(enumerate(CLASSES)), "J3 class mapping differs")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(record["model_config"], ch=3, nc=7, verbose=False)
        model.add_module("region_projection", nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.Linear(128, 7)))
    for key in ("ema", "state"):
        model.load_state_dict(record[key], strict=True)
        require(exact_tree(dict(model.state_dict()), record[key]), "J3 strict model reload differs")
    params = dict(model.named_parameters())
    optim = torch.optim.AdamW([{"params": [params[n] for n in group]} for group in record["optimizer_names"]])
    optim.load_state_dict(record["optimizer"])
    scaler = torch.amp.GradScaler("cpu")
    scaler.load_state_dict(record["scaler"])
    require(exact_tree(optim.state_dict(), record["optimizer"]) and scaler.state_dict() == record["scaler"],
            "J3 strict AdamW or GradScaler reload differs")
    del model.region_projection
    native = {n: v for n, v in record["state"].items() if not n.startswith(PREFIX)}
    model.load_state_dict(native, strict=True)
    require(hash_state(model.state_dict()) == record["native_state_sha256"], "J3 stripped native state differs")
    model.eval()
    with torch.no_grad():
        result, _ = model(torch.zeros(1, 3, 64, 64))
    require(finite_tree(result), "J3 stripped synthetic native output is nonfinite")


def checkpoints(run, summary):
    expected = [1, *range(105, STEPS + 1, 105)]
    saved = summary["checkpoints"]
    require(set(map(int, saved)) == set(expected), "J3 checkpoint schedule differs")
    names = {"last.pt" if s == STEPS else f"step_{s:04}.pt" for s in expected}
    require({p.name for p in run.glob("*.pt")} == names, "J3 checkpoint files differ")
    history = []
    for step in expected:
        path = run / ("last.pt" if step == STEPS else f"step_{step:04}.pt")
        declared, sha = saved[str(step)], sha256_file(path)
        require(declared["sha256"] == sha and declared["strict_reload"] and declared["stripped_native_forward"],
                "J3 checkpoint byte identity or training readback differs")
        r = torch.load(path, map_location="cpu", weights_only=True)
        require(r["schema"] == SCHEMA and r["arm"] == "j3" and r["identity"] == summary["identity"]
                and r["effective_updates"] == r["ema_updates"] == step, "J3 checkpoint lineage differs")
        validate_model_states(r["state"], r["ema"])
        require(finite_tree([r["state"], r["ema"], r["optimizer"], r["scheduler"]]), "J3 checkpoint is nonfinite")
        master, ema = hash_state(r["state"]), hash_state(r["ema"])
        require(declared["state_sha256"] == r["model_state_sha256"] == master != ema == r["ema_state_sha256"]
                and declared["native_state_sha256"] == r["native_state_sha256"], "J3 master/EMA identity differs")
        native = {n: v for n, v in r["state"].items() if not n.startswith(PREFIX)}
        require(hash_state(native) == r["native_state_sha256"]
                and r["scaler"] == expected_scaler_state(step)
                and r["scheduler"]["last_epoch"] == (step - 1) // 21, "J3 native/AMP/scheduler identity differs")
        require(adam_step_counts(r["optimizer"]["state"], step) == {step: 296}, "J3 AdamW slot trajectory differs")
        groups = r["optimizer_names"]
        flattened = [n for group in groups for n in group]
        require([[n for n in g if not n.startswith(PREFIX)] for g in groups] == summary["native_groups"]
                and sum(map(len, groups)) == summary["optimizer_unique_parameters"]
                and len(flattened) == len(set(flattened))
                and sum(n.startswith(PREFIX) for n in flattened) == 6, "J3 optimizer groups differ")
        if step == STEPS:
            strict_terminal(r)
        history.append({"step": step, "sha256": sha, "master_state_sha256": master,
                        "ema_state_sha256": ema, "master_native_sha256": r["native_state_sha256"]})
    return history


def audit_arm(root, seed, baseline_root, protocol, expected_identity=None):
    """root and baseline_root are seed folders; this function performs no updates."""
    require(seed in (42, 142, 242), "J3 seed outside the authorized set")
    root, baseline_root = Path(root), Path(baseline_root)
    run = root / "j3"
    summary = load(run / "summary.json")
    old = {arm: load(baseline_root / arm / "summary.json") for arm in ("j0", "j1")}
    exit_row = (root / "j3.launch.exit").read_text().strip().split("\t")
    require(len(exit_row) == 2 and exit_row[1] == "0" and not (run / "failed.json").exists()
            and not (run / "failure.log").exists(), "J3 launcher or training failed")
    weight = protocol["lambda_ce"]
    training = protocol.get("train_common", protocol.get("train"))
    require(summary["status"] == "PASS_FORMAL_ARM" and summary["variant"] == "j3"
            and summary["lambda_ce"] == weight > 0 and summary["backbone_before"] != summary["backbone_after"]
            and summary["projection_before"] != summary["projection_after"], "J3 summary or parameter update differs")
    if expected_identity is not None:
        require(summary["identity"] == expected_identity, "J3 frozen protocol identity differs")
    require(summary["initial_native_sha256"] == old["j0"]["initial_native_sha256"] == old["j1"]["initial_native_sha256"]
            and summary["runtime"] == old["j0"]["runtime"] == old["j1"]["runtime"]
            and summary["native_groups"] == old["j0"]["native_groups"] == old["j1"]["native_groups"]
            and summary["optimizer_unique_parameters"] == old["j0"]["optimizer_unique_parameters"] + 6,
            "J3 native initialization/BN, runtime or optimizer grouping differs from Exp15")
    require(summary["effective_updates"] == summary["optimizer_attempts"] == summary["native_optimizer_calls"]
            == summary["attempt_cap"] == STEPS and summary["amp_skips"] == 0
            and summary["warmup_iterations"] == 63 and summary["post_warmup_steps"] == STEPS - 63,
            "J3 update budget differs")
    events = audit_jc.by_event(run / "events.jsonl")
    require(set(events) == {"setup", "batch_read", "jb_update"} and len(events["setup"]) == 1,
            "J3 unexpected event types")
    setup, batches, updates = events["setup"][0], events["batch_read"], events["jb_update"]
    require(setup["config"]["seed"] == seed and setup["train_images"] == 668
            and setup["scaler"] == expected_scaler_state(0) and setup["variant"] == "j3"
            and len(batches) == len(updates) == len(summary["updates"]) == STEPS, "J3 setup or batch count differs")
    tickets = root / "formal_budget/j3.attempts"
    require(tickets.read_text().splitlines() == [str(n) for n in range(1, STEPS + 1)], "J3 persistent attempts differ")
    references = {arm: audit_jc.by_event(baseline_root / arm / "events.jsonl") for arm in ("j0", "j1")}
    for arm, ref in references.items():
        require(len(ref["batch_read"]) == len(ref["jb_update"]) == STEPS, "Historical complete trajectory missing")
        audit_jc.compare_batches(batches, ref["batch_read"], STEPS)
    coverage, ce_coverage, rows, stats, seen = [], [], [], collections.Counter(), set()
    max_lr_error, scale_changes = 0.0, []
    for i, (batch, up, declared) in enumerate(zip(batches, updates, summary["updates"], strict=True), 1):
        epoch, tail = (i - 1) // 21 + 1, i % 21 == 0
        paths, raw, augmented = (batch[k] for k in ("source_paths", "raw_targets", "augmented_targets"))
        require(batch["epoch"] == epoch and batch["batch"] == up["step"] == up["ticket"] == i
                and batch["batch_type"] == "all_original" and batch["repaired_images"] == 0
                and len(paths) == len(raw) == len(augmented) == batch["original_images"] == (28 if tail else 32)
                and len(set(paths)) == len(paths) and not seen.intersection(paths)
                and all(0 <= a <= r for a, r in zip(augmented, raw, strict=True)), "J3 exposure or instance binding differs")
        seen.update(paths)
        if tail:
            require(len(seen) == 668, "J3 full TRAIN epoch membership differs")
            seen.clear()
        error = max(abs(a-b) for a,b in zip(up["lr"], audit_jc.expected_lr(i, training), strict=True))
        max_lr_error = max(max_lr_error, error)
        require(batch["lrs"] == up["lr"] and error <= 1e-14
                and all(up[k] == v for k,v in declared.items()), "J3 LR or summary readback differs")
        for arm, ref in references.items():
            prior = ref["jb_update"][i-1]
            require(up["batch_sha256"] == prior["batch_sha256"] and up["lr"] == prior["lr"],
                    f"J3 augmented input/schedule differs from {arm} at {i}")
        require(up["coverage"] == references["j1"]["jb_update"][i-1]["coverage"],
                "J3 polygon geometry or pooled-region availability differs from J1")
        check_ce_coverage(up["coverage"], up["ce_coverage"], len(paths))
        opt = up["optimizer"]
        require(opt["effective_update"] == i and opt["optimizer_attempted"] and opt["native_step_called"]
                and opt["parameter_changed"] and opt["model_state_fp32_finite"] and opt["optimizer_state_finite"]
                and not opt["overflow"] and not opt["amp_skipped"] and opt["native_found_inf"] == {"cuda:0": 0.0}
                and all(opt[k]["nonfinite"] == 0 and opt[k]["master_dtype_fp32"] for k in ("scaled_gradients", "unscaled_gradients"))
                and opt["scaler_state"] == expected_scaler_state(i)
                and opt["scale_after"] == expected_scaler_state(i)["scale"]
                and opt["scale_before"] == expected_scaler_state(i-1)["scale"], "J3 numeric or AMP gate differs")
        if opt["scale_after"] != opt["scale_before"]:
            scale_changes.append(i)
        require(finite_tree([up["native_loss"], up["ce_loss"], up["total_loss"], up["global_gradient_norm"]])
                and up["global_gradient_norm"] == opt["gradient_norm"] > 0
                and math.isclose(up["total_loss"], up["native_loss"]+weight*up["ce_loss"], rel_tol=3e-7, abs_tol=2e-5)
                and math.isclose(up["clip_coef"], min(1.0, 10.0/(opt["gradient_norm"]+1e-6)), abs_tol=1e-14)
                and up["projection_nonzero_gradients"] == 6
                and up["coverage"]["targets"] == sum(augmented)
                and up["lambda_ce"] == weight
                and up["bn_native_only_check"] == (i in (1, 64, 70) or (i-1) % 21 == 0),
                "J3 CE equation/gradient/coverage/BN audit differs")
        coverage.append(up["coverage"])
        ce_coverage.append(up["ce_coverage"])
        stats.update(images=len(paths), raw=sum(raw), augmented=sum(augmented), bn=up["bn_native_only_check"],
                     clipped=up["clip_coef"] < 1)
        rows.append((i, epoch, len(paths), sum(raw), sum(augmented), audit_jc.digest(paths), up["batch_sha256"], *up["lr"]))
    require(stats["images"] == 66800 and stats["raw"] == 126600 and scale_changes == [2000]
            and summary["coverage"] == ja.summarize_regions(coverage)
            and summary["ce_coverage"] == summarize_ce_coverage(ce_coverage)
            and all(summary["ce_coverage"]["ce_regions_by_class"])
            and summary["counts"] == {"all_original": STEPS, "augmented_instances": stats["augmented"],
                "batches_read": STEPS, "original_exposures": stats["images"], "raw_instance_exposures": stats["raw"],
                "repair_exposures": 0}, "J3 total exposure/coverage/scaler trajectory differs")
    history = checkpoints(run, summary)
    result = {"status": PASS, "seed": seed, "arm": "j3", "matched_native_batches": STEPS,
              "matched_historical_arms": ["j0", "j1"], "matched_initial_native_optimizer_and_bn": True,
              "matched_main_sampling_augmentation_exposure_lr_and_regions": True,
              "region_comparison": "SAME_POLYGON_GEOMETRY_AND_VALID_POOLED_REGION_AVAILABILITY",
              "ce_supervision": "ALL_VALID_POOLED_REGIONS_CLASS_BALANCED_MEAN",
              "identical_per_region_gradient_support_claimed": False,
              "effective_updates": STEPS, "optimizer_attempts": STEPS, "amp_skips": 0,
              "initial_native_sha256": summary["initial_native_sha256"], "lambda_ce": weight,
              "exposure": dict(stats), "max_lr_absolute_error": max_lr_error, "amp_growth_after_steps": scale_changes,
              "terminal_strict_cpu_reload": True, "stripped_native_synthetic_prediction": True,
              "checkpoints": history, "endpoint": history[-1], "runtime": summary["runtime"],
              "coverage": summary["coverage"], "ce_coverage": summary["ce_coverage"],
              "peak_allocated_bytes": summary["peak_allocated_bytes"],
              "peak_reserved_bytes": summary["peak_reserved_bytes"], "elapsed_seconds": summary.get("elapsed_seconds"),
              "events_sha256": sha256_file(run / "events.jsonl"), "summary_sha256": sha256_file(run / "summary.json"),
              "tickets_sha256": sha256_file(tickets), "exit_timestamp": exit_row[0], "val_inference": 0, "test_inference": 0}
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "baseline-root", "protocol", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    result, rows = audit_arm(a.root, a.seed, a.baseline_root, load(a.protocol))
    with a.output.with_suffix(".csv").open("x", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("step", "epoch", "images", "raw_targets", "augmented_targets", "ordered_sources_sha256",
                         "augmented_tensors_sha256", "lr_no_decay", "lr_decay", "lr_bias"))
        writer.writerows(rows)
    result["matched_batch_ledger_sha256"] = sha256_file(a.output.with_suffix(".csv"))
    write(a.output, result)


if __name__ == "__main__":
    main()
