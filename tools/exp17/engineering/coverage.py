"""Read-only, full native J1 trajectory audit and strict augmented Pair coverage.

This command performs no image/model forward, backward or optimizer update.
The original J1 loader, epoch seeds, warmup and native six-tensor digest remain
the authority. Valid masks are replayed; repaired N images are never decoded.
All outputs are new, exclusive files. Any first mismatch stops the traversal.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.exp17.engineering.augmentation import TRACE_KEY, array_sha
from tools.exp17.engineering.data import CLASSES, digest, load, require, sha, write

STEPS = 2100
EPOCHS = 100
BATCHES_PER_EPOCH = 21
FIXED_BATCHES = (1, 7, 11, 21)
PASS = "EXP17_J2_READONLY_NATIVE_TRAJECTORY_AND_COVERAGE_PASS"
STOP = "EXP17_J2_READONLY_NATIVE_TRAJECTORY_AND_COVERAGE_HARD_GATE"
REFERENCE_FIELDS = ("step", "epoch", "images", "raw_targets", "augmented_targets",
                    "ordered_sources_sha256", "augmented_tensors_sha256",
                    "lr_no_decay", "lr_decay", "lr_bias")


def fingerprint(value):
    """A bounded metadata/tensor fingerprint; never pickle arbitrary objects."""
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": array_sha(value), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.ndarray):
        return {"array_sha256": array_sha(value), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    require(value is None or isinstance(value, (str, bool, int, float)), "Unsupported state fingerprint type")
    return value


def rng_identity():
    """Observe, never restore, RNG. Avoid initializing a previously unused CUDA."""
    state = {"python": fingerprint(random.getstate()), "numpy": fingerprint(np.random.get_state()),
             "torch_cpu": fingerprint(torch.get_rng_state()), "cuda_initialized": torch.cuda.is_initialized()}
    if state["cuda_initialized"]:
        state["torch_cuda_all"] = fingerprint(torch.cuda.get_rng_state_all())
    return digest(state)


def model_optimizer_identity(trainer):
    from tools.exp14.a11_smoke_numerics import optimizer_group_names
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp15.ja import bn_fingerprint

    optimizer = trainer.optimizer.state_dict()
    # Scheduling intentionally changes lr (and SGD-style momentum, if present).
    # Everything else, including AdamW slot tensors/step counts, is immutable.
    fixed_groups = [{k: v for k, v in group.items() if k not in {"params", "lr", "momentum"}}
                    for group in optimizer["param_groups"]]
    result = {"model_state_sha256": hash_state(trainer.model.state_dict()),
              "bn_state_sha256": bn_fingerprint(trainer.model),
              "gradients_sha256": digest(fingerprint({n: p.grad for n, p in trainer.model.named_parameters()})),
              "optimizer_state_sha256": digest(fingerprint(optimizer["state"])),
              "optimizer_fixed_groups_sha256": digest(fingerprint(fixed_groups)),
              "optimizer_parameter_names_sha256": digest(optimizer_group_names(trainer.model, trainer.optimizer)),
              "scaler_sha256": digest(fingerprint(trainer.scaler.state_dict()))}
    ema = getattr(trainer, "ema", None)
    if ema is not None:
        result["ema_state_sha256"] = hash_state(ema.ema.state_dict())
        result["ema_updates"] = int(ema.updates)
    return result


class NoForwardOrUpdate:
    """Hard guards apply after native setup, including direct child-module calls."""
    def __init__(self, model, optimizer):
        self.forward_attempts = 0
        self.optimizer_attempts = 0
        self.handles = [m.register_forward_pre_hook(self.block_forward) for m in model.modules()]
        self.handles.append(optimizer.register_step_pre_hook(self.block_step))

    def block_forward(self, *_args, **_kwargs):
        self.forward_attempts += 1
        raise RuntimeError("Read-only coverage cannot execute any model/module forward")

    def block_step(self, *_args, **_kwargs):
        self.optimizer_attempts += 1
        raise RuntimeError("Read-only coverage cannot execute an optimizer step")

    def close(self):
        for handle in self.handles:
            handle.remove()


def read_reference(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == REFERENCE_FIELDS, "Frozen J-C native ledger schema differs")
        rows = list(reader)
    require(len(rows) == STEPS, "Frozen J-C native ledger is not exactly 2100 batches")
    for step, row in enumerate(rows, 1):
        for key in ("step", "epoch", "images", "raw_targets", "augmented_targets"):
            row[key] = int(row[key])
        for key in ("lr_no_decay", "lr_decay", "lr_bias"):
            row[key] = float(row[key])
            require(math.isfinite(row[key]), "Frozen ledger has nonfinite LR")
        require(row["step"] == step and row["epoch"] == (step - 1) // BATCHES_PER_EPOCH + 1,
                "Frozen ledger batch/epoch order differs")
    require(sum(r["images"] for r in rows) == 66800 and sum(r["raw_targets"] for r in rows) == 126600,
            "Frozen J-C exposure budget differs")
    return rows


def native_evidence(trainer, batch, step):
    from tools.exp15 import ja
    from tools.exp15.audit_jc import digest as ordered_digest

    full = [Path(name).resolve(strict=True) for name in batch["im_file"]]
    paths = [p.relative_to(trainer.dataset_root).as_posix() for p in full]
    raw = [len(trainer.expected[p][0]) for p in full]
    return {"step": step, "epoch": trainer.epoch + 1, "images": len(paths), "raw_targets": sum(raw),
            "augmented_targets": int(batch["cls"].numel()), "ordered_sources_sha256": ordered_digest(paths),
            "augmented_tensors_sha256": ja.batch_digest(batch),
            **dict(zip(("lr_no_decay", "lr_decay", "lr_bias"),
                       [float(g["lr"]) for g in trainer.optimizer.param_groups], strict=True))}


def compare_reference(actual, reference, protocol):
    from tools.exp15.audit_jc import expected_lr

    require(actual.keys() == reference.keys(), "Native evidence field domain differs from frozen ledger")
    for key in REFERENCE_FIELDS:
        require(actual[key] == reference[key], f"Frozen J1 native trajectory differs at step {actual['step']}: {key}")
    expected = expected_lr(actual["step"], protocol)
    actual_lr = [actual[k] for k in ("lr_no_decay", "lr_decay", "lr_bias")]
    require(all(math.isfinite(a) and abs(a - b) <= 1e-14 for a, b in zip(actual_lr, expected, strict=True)),
            "Independent native linear LR/warmup formula differs")


class CoverageCounts:
    """Exposure frequencies stay separate from independent group cardinalities."""
    def __init__(self):
        self.totals = Counter()
        self.by_epoch = defaultdict(Counter)
        self.by_class = {str(c): Counter() for c in range(7)}
        self.source_exposures = Counter()
        self.source_group_exposures = Counter()
        self.raw_polygon_exposures = Counter()
        self.native_group_exposures = Counter()
        self.anchor_group_exposures = Counter()
        self.anchor_polygon_exposures = Counter()
        self.b_group_batch_participation = Counter()
        self.class_groups = {str(c): set() for c in range(7)}
        self.batch_histograms = {k: Counter() for k in ("actual_valid_ABN", "valid_N_image_slots", "active_classes", "active_A_groups")}

    def add(self, step, batch, prepared, index):
        epoch = (step - 1) // BATCHES_PER_EPOCH + 1
        coverage = prepared.coverage
        plan = prepared.plan
        valid_rows = list(coverage["valid_A_rows"])
        valid_slots = list(coverage["valid_N_slots"])
        require(valid_rows == [a.a_row for a in plan.anchors] and len(set(valid_rows)) == len(valid_rows),
                "Coverage plan did not retain precisely its support-complete native A rows")
        require(valid_slots == sorted({a.a_slot for a in plan.anchors})
                and coverage["actual_valid_ABN"] == len(valid_rows), "Actual valid Pair/slot coverage differs")
        require(coverage["N_images_decoded_this_call"] == 0 and prepared.normal_images is None
                and not prepared.bindings, "Read-only coverage decoded/exposed N or created model bindings")
        counts = Counter({"batches": 1, "images": len(batch["im_file"]),
                          "augmented_targets": int(batch["cls"].numel()),
                          "native_regions_with_cells": len(plan.native_rows),
                          "native_regions_without_cells": int((prepared.a_regions.cells == 0).sum()),
                          "admitted_A": coverage["admitted_A"],
                          "pre_valid_support_ABN": coverage["pre_valid_support_planned_ABN"],
                          "actual_valid_ABN": len(valid_rows), "valid_N_image_slots": len(valid_slots),
                          "batches_without_valid_ABN": int(not valid_rows)})
        for key, value in coverage["skip"].items():
            counts["skip_" + key] += value
        for path in batch["im_file"]:
            sid = index.source_for_image(path)
            self.source_exposures[sid] += 1
            labels = index.train[sid]["native_labels"]
            self.source_group_exposures[index.train[sid]["group_id"]] += 1
            counts["raw_targets"] += len(labels)
            for row in labels:
                self.raw_polygon_exposures[row["polygon_id"]] += 1
                self.by_class[str(row["class_id"])]["raw_target_exposures"] += 1
        for cls, value in Counter(int(c) for c in batch["cls"].reshape(-1).tolist()).items():
            self.by_class[str(cls)]["augmented_target_exposures"] += value
        for cls in prepared.a_regions.classes[prepared.a_regions.cells == 0]:
            self.by_class[str(int(cls))]["augmented_native_raster_missing_exposures"] += 1
        for row in plan.native_rows:
            self.native_group_exposures[row["group_id"]] += 1
        groups = set()
        b_groups = set()
        active_classes = set()
        for anchor in plan.anchors:
            self.anchor_group_exposures[anchor.group] += 1
            self.anchor_polygon_exposures[anchor.polygon_id] += 1
            self.class_groups[str(anchor.class_id)].add(anchor.group)
            groups.add(anchor.group)
            active_classes.add(anchor.class_id)
            b_groups.update(anchor.b_rows_by_group)
        self.b_group_batch_participation.update(b_groups)
        for cls, values in coverage["by_class"].items():
            self.by_class[cls].update(values)
        counts["active_class_batch_exposures"] = len(active_classes)
        counts["active_A_group_batch_exposures"] = len(groups)
        self.totals.update(counts)
        self.by_epoch[epoch].update(counts)
        hist = {"actual_valid_ABN": len(valid_rows), "valid_N_image_slots": len(valid_slots),
                "active_classes": len(active_classes), "active_A_groups": len(groups)}
        for key, value in hist.items():
            self.batch_histograms[key][value] += 1
        return {"counts": dict(counts), "active_classes": sorted(active_classes),
                "active_A_groups": sorted(groups), "B_groups": sorted(b_groups),
                "coverage": coverage}

    def report(self):
        return {"totals": dict(self.totals), "by_epoch": {str(k): dict(v) for k, v in sorted(self.by_epoch.items())},
                "by_class": {str(c): {"class_name": CLASSES[c], **dict(self.by_class[str(c)]),
                                     "independent_valid_A_groups": len(self.class_groups[str(c)])} for c in range(7)},
                "independent_counts": {"original_sources": len(self.source_exposures),
                    "original_source_groups": len(self.source_group_exposures),
                    "original_polygons": len(self.raw_polygon_exposures), "native_region_groups": len(self.native_group_exposures),
                    "valid_A_groups": len(self.anchor_group_exposures), "valid_A_polygons": len(self.anchor_polygon_exposures),
                    "participating_B_groups": len(self.b_group_batch_participation)},
                "batch_histograms": {k: {str(n): v for n, v in sorted(counter.items())} for k, counter in self.batch_histograms.items()},
                "original_source_exposures": dict(sorted(self.source_exposures.items())),
                "original_source_group_image_exposures": dict(sorted(self.source_group_exposures.items())),
                "raw_polygon_exposures": dict(sorted(self.raw_polygon_exposures.items())),
                "native_group_region_exposures": dict(sorted(self.native_group_exposures.items())),
                "valid_A_group_anchor_exposures": dict(sorted(self.anchor_group_exposures.items())),
                "valid_A_polygon_exposures": dict(sorted(self.anchor_polygon_exposures.items())),
                "B_group_batch_participation": dict(sorted(self.b_group_batch_participation.items())),
                "group_frequencies_are_exposures_not_independent_samples": True,
                "coverage_is_support_eligible_not_actual_N_model_exposure": True}


def append_json(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def assert_cpu(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "Calibration cache must contain the original CPU native batch")
    elif isinstance(value, dict):
        for item in value.values():
            assert_cpu(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_cpu(item)


def run(config, output, reference_path):
    from tools.exp14.formal_e0_e1 import cpu_state
    from tools.exp15.jb import matched_batches
    from tools.exp17.engineering.runtime import DEFAULT_ASSET_CACHE, prepare_batch, setup_trainer

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"schema": "EXP17_J2_READONLY_COVERAGE_V1", "status": STOP,
               "reference_ledger_sha256": sha(reference_path), "config_sha256": digest(config),
               "source_sha256": sha(__file__), "completed_batches": 0, "fixed_batches": [],
               "model_forwards": 0, "optimizer_updates": 0, "N_image_decodes": 0,
               "sampler_modified": False, "formal_training_authorized": False,
               "RNG_note": "Native data loading legitimately advances RNG; each added metadata/Valid replay helper must consume zero RNG."}
    trainer = guard = None
    counts = CoverageCounts()
    ledger_path = output / "native_trajectory_and_pair_coverage.jsonl"
    traces_path = output / "compact_explicit_augmentation_traces.jsonl"
    try:
        references = read_reference(reference_path)
        trainer = setup_trainer(config, output / "native_trainer")
        summary["data_identity"] = {"trusted_manifest_sha256": trainer.index.manifest_sha256,
            "native_train_identity_sha256": sha(config["native_identity"]),
            "original_train_sources": len(trainer.index.train),
            "original_train_groups": len({r["group_id"] for r in trainer.index.train.values()}),
            "trusted_polygons": len(trainer.index.records)}
        guard = NoForwardOrUpdate(trainer.model, trainer.optimizer)
        summary["before"] = model_optimizer_identity(trainer)
        summary["native_RNG_before"] = rng_identity()
        initial_N_decodes = DEFAULT_ASSET_CACHE.decodes["N"]
        cache = output / "fixed_cpu_batches"
        cache.mkdir()
        with ledger_path.open("x", encoding="utf8", newline="\n") as ledger, traces_path.open("x", encoding="utf8", newline="\n") as traces_stream:
            for step, batch in matched_batches(trainer, STEPS):
                actual = native_evidence(trainer, batch, step)
                # Persist a mismatch before stopping, never silently discard it.
                append_json(ledger, {"event": "native_observation", **actual})
                compare_reference(actual, references[step - 1], trainer.formal)
                added_rng_before = rng_identity()
                prepared = prepare_batch(batch, trainer.index, trainer.groups, decode_N=False)
                require(rng_identity() == added_rng_before, "Added Pair support diagnostics consumed native RNG")
                require(native_evidence(trainer, batch, step) == actual, "Coverage preparation modified native batch tensors/order")
                detail = counts.add(step, batch, prepared, trainer.index)
                append_json(ledger, {"event": "matched_native_and_pair_coverage", "step": step,
                    "epoch": trainer.epoch + 1, "warmup_position": step - 1 if step <= 63 else None,
                    "reference_all_fields_exact": True, **detail})
                compact = [trace.compact_record() for trace in batch[TRACE_KEY]]
                append_json(traces_stream, {"step": step, "epoch": trainer.epoch + 1,
                    "ordered_traces_sha256": digest(compact), "traces": compact})
                if step in FIXED_BATCHES:
                    assert_cpu(batch)
                    path = cache / f"epoch000_batch{step:03d}_native_cpu.pt"
                    payload = {"schema": "EXP17_J2_NATIVE_CPU_BATCH_CACHE_V1", "step": step,
                               "epoch": 0, "batch_position": step, "batch": cpu_state(batch)}
                    with path.open("xb") as stream:
                        torch.save(payload, stream)
                    summary["fixed_batches"].append({"batch_position": step, "step": step, "epoch": 0,
                        "path": str(path), "sha256": sha(path), "native_six_tensor_sha256": actual["augmented_tensors_sha256"]})
                summary["completed_batches"] = step
                if step % BATCHES_PER_EPOCH == 0:
                    print(json.dumps({"event": "readonly_epoch_complete", "epoch": trainer.epoch + 1,
                        "matched_batches": step, "valid_ABN_exposures": counts.totals["actual_valid_ABN"]}), flush=True)
        summary["after"] = model_optimizer_identity(trainer)
        summary["native_RNG_after"] = rng_identity()
        require(summary["before"] == summary["after"], "Read-only traversal changed model/BN/optimizer slots/scaler/EMA")
        audit = trainer.audit
        require(all(getattr(audit, key) == 0 for key in ("optimizer_attempts", "native_steps", "effective_updates", "amp_skips"))
                and guard.forward_attempts == guard.optimizer_attempts == 0,
                "Read-only traversal attempted forward or optimizer execution")
        require(DEFAULT_ASSET_CACHE.decodes["N"] == initial_N_decodes, "Coverage decoded a repaired N image")
        require(counts.totals["batches"] == STEPS and counts.totals["images"] == 66800
                and counts.totals["raw_targets"] == 126600 and len(counts.source_exposures) == 668
                and len(counts.raw_polygon_exposures) == 1266
                and set(counts.source_exposures.values()) == {100}
                and set(counts.raw_polygon_exposures.values()) == {100}, "Full native exposure budget differs")
        require(len(summary["fixed_batches"]) == len(FIXED_BATCHES), "Fixed calibration CPU batch caches incomplete")
        summary.update(status=PASS, all_2100_native_batches_exact=True, warmup_positions=63,
                       N_image_decodes=0, numerical_audit_counts={k: getattr(audit, k) for k in
                        ("optimizer_attempts", "native_steps", "effective_updates", "amp_skips")})
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if trainer is not None:
            try:
                summary["after_or_failure"] = model_optimizer_identity(trainer)
            except Exception as snapshot_error:
                summary["failure_snapshot_error"] = str(snapshot_error)
    finally:
        if guard is not None:
            summary["guard_attempts"] = {"model_forward": guard.forward_attempts, "optimizer_step": guard.optimizer_attempts}
            guard.close()
        if trainer is not None:
            trainer.close()
        summary["coverage"] = counts.report()
        summary["elapsed_seconds"] = time.monotonic() - started
        for name, path in (("coverage_ledger", ledger_path), ("explicit_trace_ledger", traces_path)):
            if path.exists():
                summary[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-ledger", type=Path,
                        default=Path(__file__).resolve().parents[3] / "results/exp15_jc_formal_seed42/matched_batch_ledger.csv")
    args = parser.parse_args()
    result = run(load(args.config), args.output, args.reference_ledger)
    print(json.dumps({"status": result["status"], "completed_batches": result["completed_batches"]}), flush=True)
    raise SystemExit(0 if result["status"] == PASS else 2)


if __name__ == "__main__":
    main()
