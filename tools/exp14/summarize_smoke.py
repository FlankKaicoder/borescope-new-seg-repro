"""Recheck bounded Exp14-A3 evidence and export private exposure and public counts."""

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from tools.exp14.audit_admission import read_rows, require, sha256_file
from tools.exp14.freeze_core import COCO_SHA256


def read_json(path):
    return json.loads(path.read_text())


def read_run(runtime, phase, arm):
    path = runtime / f"{phase}_{arm}"
    if not path.is_dir():
        return {"status": "NOT_STARTED"}, []
    summary = read_json(path / "summary.json")
    with (path / "events.jsonl").open() as stream:
        events = [json.loads(line) for line in stream]
    require(all(e["diagnostic_only"] and e["phase"] == phase and e["arm"] == arm for e in events),
            "Run identity mismatch")
    return summary, events


def verify_payload(payload):
    manifest = read_json(payload / "transport_manifest.json")
    listed = {r["path"]: r for r in manifest["files"]}
    require(manifest["scope"] == "TRAIN_ONLY" and manifest["test_files"] == 0
            and len(listed) == len(manifest["files"]) == 1373, "Payload scope or count mismatch")
    require({p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
            == set(listed) | {"transport_manifest.json"}, "Payload file set changed")
    for key, row in listed.items():
        path = (payload / key).resolve(strict=True)
        require(path.is_relative_to(payload) and path.stat().st_size == row["bytes"]
                and sha256_file(path) == row["sha256"], "Payload bytes changed")
    return manifest


def audit_run(events, summary, phase, arm, slots, plan):
    exposure = Counter()
    loaded = [e for e in events if e["kind"] == "batch_loaded"]
    forwards = [e for e in events if e["kind"] == "finite_forward"]
    updates = [e for e in events if e["kind"] == "optimizer_update"]
    setup = [e for e in events if e["kind"] == "setup"]
    require(len(loaded) <= (22 if phase == "preflight" else 44), "Batch cap exceeded")
    require(all(math.isfinite(e["loss"]) and all(map(math.isfinite, e["components"].values()))
                and e["amp_autocast"] for e in forwards), "Saved finite-forward evidence differs")
    require(len(updates) == summary.get("effective_updates", 0)
            and len(forwards) == summary.get("forward_checks", 0), "Run counters disagree")
    require(not updates if phase == "preflight" else len(updates) <= len(loaded), "Update phase mismatch")
    epoch_summary = {}
    classes = set()
    worker_by_epoch = {}
    for offset, batch in enumerate(loaded):
        epoch, index = divmod(offset, 22)
        require((batch["epoch"], batch["batch"]) == (epoch + 1, index + 1)
                and batch["accumulate"] == 1, "Batch identity or accumulation differs")
        records = batch["slots"]
        require([r["slot_id"] for r in records] == plan[epoch]["batches"][index],
                "Saved slot order differs")
        out = epoch_summary.setdefault(epoch + 1, {"batches": 0, "samples": 0,
                                                   "full_batches": 0, "tail_batches": 0,
                                                   "empty_samples": 0, "updates": 0})
        out["batches"] += 1
        out["samples"] += len(records)
        require(len(records) == (12 if index == 21 else 32), "Observed batch size differs")
        out["tail_batches" if index == 21 else "full_batches"] += 1
        for row in records:
            slot = slots[row["slot_id"]]
            expected = slot[f"{arm}_image_path"]
            require(row["im_file"].endswith("/dataset/" + expected) and row["epoch"] == epoch + 1
                    and row["actual_reads"] == [row["dataset_index"]] and row["worker_id"] in {0, 1},
                    "Worker, read count, path or epoch differs")
            worker_by_epoch.setdefault(epoch + 1, set()).add(row["worker_pid"])
            category = "aligned" if arm == "e1" and slot["slot_kind"] == "EXTRA" else "original"
            exposure[(epoch + 1, slot["source_id"], slot["group_id"], category)] += 1
            out["empty_samples"] += row["raw_instances"] == 0
            classes.update(row["raw_classes"])
    for offset, row in enumerate(updates):
        require(row["update"] == offset + 1 and row["parameter_changed"] and row["amp"]
                and row["accumulate"] == 1 and row["gradient_norm"] >= 0, "Optimizer update evidence differs")
        require((row["epoch"], row["batch"]) == (loaded[offset]["epoch"], loaded[offset]["batch"]),
                "Optimizer update is not tied to the loaded batch")
        epoch_summary[row["epoch"]]["updates"] += 1
    if len(worker_by_epoch) == 2:
        require(worker_by_epoch[1].isdisjoint(worker_by_epoch[2]), "Workers persisted across epochs")
    if summary["status"] == "PASS_DIAGNOSTIC_ONLY":
        require(len(setup) == 1 and setup[0]["amp"] and setup[0]["master_dtype"] == "float32"
                and setup[0]["workers"] == 2 and setup[0]["prefetch_factor"] == 2
                and setup[0]["persistent_workers"] is False, "Effective AMP/loader configuration differs")
        expected = 22 if phase == "preflight" else 44
        require(len(loaded) == expected and len(updates) == (0 if phase == "preflight" else 44),
                "PASS run is incomplete")
        require(all(e["samples"] == 684 and e["empty_samples"] == (16 if arm == "e1" else 0)
                    and e["full_batches"] == 21 and e["tail_batches"] == 1 for e in epoch_summary.values()),
                "Actual exposures or tail batches differ")
        require(classes == set(range(7)), "Not all original classes observed")
    return {
        "status": summary["status"], "epoch_details": epoch_summary, "classes_seen": sorted(classes),
        "forward_checks": len(forwards), "optimizer_calls": summary.get("optimizer_calls", 0),
        "effective_updates": len(updates), "amp": summary.get("amp"), "master_dtype": setup[0]["master_dtype"] if setup else None,
        "peak_memory_allocated_bytes": summary.get("peak_memory_allocated"),
        "peak_memory_reserved_bytes": summary.get("peak_memory_reserved"),
        "suppressed_validation_events": sum(e["kind"] in {"validation_suppressed", "final_evaluation_suppressed"}
                                            for e in events),
        "learning_rates_first": loaded[0]["lrs"] if loaded else [],
        "learning_rates_last": loaded[-1]["lrs"] if loaded else [],
        "no_secondary_reads": bool(loaded),
        "worker_epochs_disjoint": len(worker_by_epoch) == 2,
        "stop_reason": summary.get("reason"),
        "first_finite_loss": forwards[0]["loss"] if forwards else None,
    }, exposure, loaded


def summarize(root):
    payload, runtime = root / "payload", root / "runtime"
    manifest = verify_payload(payload)
    slots = {row["slot_id"]: row for row in read_rows(payload / "matched_slots.csv")}
    plan = read_json(payload / "epoch_slot_order.json")
    require(len(slots) == 684 and len(plan) == 100 and plan[0]["batches"] != plan[1]["batches"],
            "Frozen slots or order differ")
    report, paths, exposure = {}, {}, []
    for phase in ("preflight", "train"):
        for arm in ("e0", "e1"):
            key = f"{phase}_{arm}"
            summary, events = read_run(runtime, phase, arm)
            detail, draws, loaded = audit_run(events, summary, phase, arm, slots, plan)
            report[key] = detail
            paths[key] = loaded
            exposure.extend((phase, arm, *key, count) for key, count in sorted(draws.items()))
    preflight_init = [read_json(runtime / f"preflight_{arm}/summary.json").get("initial_state_sha256")
                      for arm in ("e0", "e1") if (runtime / f"preflight_{arm}/summary.json").is_file()]
    matched_init = len(preflight_init) == 2 and bool(preflight_init[0]) and preflight_init[0] == preflight_init[1]
    passed = matched_init and all(r["status"] == "PASS_DIAGNOSTIC_ONLY" for r in report.values())
    if passed:
        require(all(a["lrs"] == b["lrs"] for a, b in zip(paths["train_e0"], paths["train_e1"], strict=True)),
                "Matched optimizer learning rate trajectories differ")
    with (runtime / "actual_exposure.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("phase", "arm", "epoch", "source_id", "group_id", "image_kind", "draws"))
        writer.writerows(exposure)
    env = read_json(runtime / "preflight_e0/environment.json")
    safe_environment = {k: env[k] for k in ("python", "torch", "torchvision", "ultralytics", "numpy",
                                            "pillow", "gpu", "cuda_runtime", "driver_gpu")}
    require(env["weight_sha256"] == COCO_SHA256 and env["actual_git"] == manifest["source_git"],
            "Remote environment identity differs")
    safe_environment["disk_free_gib_at_audit"] = round(shutil.disk_usage(root).free / 2**30, 2)
    result = {
        "stage": "Exp14-A3", "diagnostic_only": True,
        "status": "SMOKE_PASS_PENDING_CHATGPT_TRAIN_GATE" if passed else "SMOKE_STOP_PENDING_CHATGPT_REVIEW",
        "base_git_sha": manifest["source_git"], "environment": safe_environment, "payload_files_verified": len(manifest["files"]),
        "weight_sha256": COCO_SHA256, "ui_risk_flags_retained": len(manifest["ui_risks"]),
        "preflight_initial_state_match": matched_init, "runs": report,
        "actual_validation_runs": 0, "formal_training_started": False,
    }
    with (runtime / "smoke_summary.json").open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root.resolve(strict=True))
