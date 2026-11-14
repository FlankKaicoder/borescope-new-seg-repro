"""Probe two explicit E1 TRAIN batch compositions with AMP backward and zero optimizer steps."""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from time import monotonic

from tools.exp13_cloud.g3.g3_batch_executor import write_json_exclusive
from tools.exp14.audit_admission import require


CASES = ("original_repair_mixed", "all_background")
PICK_SEED = 1101


def index_inventory(dataset_root, inventory):
    """Identify TRAIN members by frozen inventory, including their raw labels."""
    rows = {}
    for origin in ("baseline", "derived"):
        for row in inventory[origin]:
            path = (dataset_root / row["image_path"]).resolve(strict=True)
            require(row["split"] == "train" and path.is_relative_to(dataset_root)
                    and path not in rows and (row["instance_count"] > 0) == (origin == "baseline"),
                    "Unexpected TRAIN identity or label semantics in diagnostic inventory")
            rows[path] = origin, row
    return rows


def case_indices(dataset_paths, inventory, case, batch_size):
    require(case in CASES and batch_size >= 2, "Invalid diagnostic composition")
    require(len(dataset_paths) == len(inventory) and set(dataset_paths) == inventory.keys(),
            "Native diagnostic dataset differs from the frozen E1 membership")
    buckets = {"baseline": [], "derived": []}
    for index, path in enumerate(dataset_paths):
        buckets[inventory[path][0]].append(index)
    original_count = batch_size // 2 if case == "original_repair_mixed" else 0
    require(original_count <= len(buckets["baseline"]) and batch_size - original_count <= len(buckets["derived"]),
            "Insufficient frozen E1 images for the diagnostic batch")
    rng = random.Random(PICK_SEED + CASES.index(case))
    indices = rng.sample(buckets["baseline"], original_count) + rng.sample(buckets["derived"], batch_size - original_count)
    rng.shuffle(indices)
    return indices


def validate_label_ownership(batch, paths, inventory):
    require([Path(p).resolve() for p in batch["im_file"]] == paths, "Native diagnostic collate changed image order")
    indexes = batch["batch_idx"].reshape(-1).tolist()
    require(len(indexes) == len(batch["cls"]) and all(0 <= i < len(paths) and i == int(i) for i in indexes),
            "Diagnostic collate returned invalid instance indices")
    targets = Counter(indexes)
    for slot, path in enumerate(paths):
        origin, row = inventory[path]
        require(targets[slot] <= row["instance_count"], "Diagnostic augmentation added instances")
        if origin == "derived":
            require(targets[slot] == 0 and not bool(batch["masks"][slot].any()) and not bool(batch["sem_masks"][slot].any()),
                    "An explicitly empty repair sample acquired labels or foreground masks")
    return len(indexes)


def validate_native_loss(case, components, foreground, branches):
    foreground_total = sum(foreground.values())
    cls = [value for key, value in branches.items() if key.startswith("model.23.cv3.")]
    require(components["cls_loss"] > 0 and cls
            and any(row["elements"] != row["zero"] and not row["nonfinite"] for row in cls),
            "No finite nonzero classification signal in the diagnostic batch")
    if case == "all_background":
        require(foreground_total == 0 and all(value == 0 for key, value in components.items() if key != "cls_loss"),
                "Pure empty-label batch created a foreground-dependent loss")
        foreground_branches = [value for key, value in branches.items()
                               if key.startswith(("model.23.cv2.", "model.23.cv4.", "model.23.proto"))]
        require(foreground_branches and all(row["elements"] == row["zero"] and not row["nonfinite"]
                                           for row in foreground_branches),
                "Pure empty-label batch created a foreground-dependent head gradient")
    else:
        require(foreground_total > 0 and components["seg_loss"] > 0,
                "Controlled mixed batch did not exercise positive segmentation supervision")


def probe(trainer, inventory, case, batch_size, evidence):
    import torch

    from tools.exp14.diagnose_amp_first_batch import parameters_sha
    from tools.exp14.a11_smoke_numerics import model_fp32_finite
    from tools.exp14.foreground_audit import ForegroundAudit
    from tools.exp14.formal_e0_e1 import layer_gradient_statistics

    now = monotonic()
    trainer.setup()
    model, optimizer, scaler = trainer.model, trainer.optimizer, trainer.scaler
    require(model.training and scaler.is_enabled() and scaler.get_scale() == 1 and not optimizer.state
            and trainer.train_loader.num_workers == 0,
            "Diagnostic setup did not start from fresh training-mode AMP state")
    before = parameters_sha(model)
    evidence.update(initial_parameter_sha256=before, optimizer_state_entries_before=len(optimizer.state),
                    amp_scale_before=scaler.get_scale())

    def reject_step(_optimizer, _args, _kwargs):
        evidence["optimizer_step_attempts"] += 1
        raise RuntimeError("Optimizer step is forbidden in the isolated backward diagnostic")

    handle = optimizer.register_step_pre_hook(reject_step)
    try:
        foreground_audit = getattr(trainer, "foreground", None) or ForegroundAudit(model)
        dataset = trainer.train_loader.dataset
        dataset_paths = [Path(p).resolve() for p in dataset.im_files]
        indexes = case_indices(dataset_paths, inventory, case, batch_size)
        paths = [dataset_paths[i] for i in indexes]
        counts = Counter(inventory[path][0] for path in paths)
        evidence.update(original_images=counts["baseline"], repair_images=counts["derived"],
                        explicit_empty_labels=counts["derived"],
                        raw_instances=sum(inventory[path][1]["instance_count"] for path in paths),
                        input_identity_sha256=hashlib.sha256(
                            "\n".join(inventory[path][1]["image_sha256"] for path in paths).encode("ascii")
                        ).hexdigest())
        batch = dataset.collate_fn([dataset[i] for i in indexes])
        evidence["batches_loaded"] += 1
        targets = evidence["instances_after_augmentation"] = validate_label_ownership(batch, paths, inventory)
        require((targets == 0) == (case == "all_background"), "Native transforms did not preserve diagnostic composition")
        batch = trainer.preprocess_batch(batch)
        torch.cuda.reset_peak_memory_stats(trainer.device)
        evidence["forward_attempts"] += 1
        with torch.autocast("cuda", enabled=True):
            raw_loss, raw_parts = model(batch)
            loss = raw_loss.sum()
        evidence["forward_calls"] += 1
        evidence["loss"] = trainer.audit.check_forward(loss, raw_parts)
        evidence["foreground_by_stride"] = foreground_audit.read()
        evidence["backward_attempts"] += 1
        scaler.scale(loss).backward()
        evidence["backward_calls"] += 1
        evidence["gradients"] = trainer.audit.diagnostic_backward()
        branches = layer_gradient_statistics(model, by_scale=True)
        evidence["head_gradients"] = {name: row for name, row in branches.items() if name.startswith("model.23.")}
        values = {key: float(value) for key, value in raw_parts.items()}
        evidence["loss_components"] = values
        validate_native_loss(case, values, evidence["foreground_by_stride"], evidence["head_gradients"])
        after = parameters_sha(model)
        evidence.update(final_parameter_sha256=after, optimizer_state_entries_after=len(optimizer.state),
                        model_state_finite=model_fp32_finite(model),
                        amp_scale_after=scaler.get_scale(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(trainer.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(trainer.device))
        require(after == before and evidence["model_state_finite"]
                and not optimizer.state and evidence["optimizer_step_attempts"] == 0
                and trainer.audit.optimizer_attempts == trainer.audit.native_steps == trainer.audit.effective_updates == 0
                and scaler.get_scale() == 1, "Diagnostic changed parameters, optimizer state or Scale")
    finally:
        handle.remove()
        evidence["elapsed_seconds"] = monotonic() - now


def run(release, weights, report, batch_size):
    require(not report.exists() and not report.is_relative_to(release),
            "Diagnostic report already exists or points into the frozen release")
    cases_root = report.parent / (report.stem + "_cases")
    cases_root.mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(cases_root / "settings"), MPLCONFIGDIR=str(cases_root / "matplotlib"),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")

    import torch

    from tools.exp14.a11_smoke_train import BoundedSmokeTrainer

    inventory = json.loads((release / "configs" / "train_entry_files.json").read_text(encoding="utf-8"))
    dataset_root = (release / "dataset").resolve(strict=True)
    membership = index_inventory(dataset_root, inventory)
    require(batch_size >= 2 and torch.cuda.is_available(), "Paired GPU diagnostics require CUDA and a mixed batch size")
    summary = {"stage": "EXP14_A11_1_ISOLATED_E1_BACKWARD_DIAGNOSTICS", "status": "PASS", "batch_size": batch_size,
               "case_pick_seed": PICK_SEED, "loader_workers": 0, "optimizer_updates": 0, "optimizer_step_attempts": 0,
               "separate_from_bounded_smoke": True, "checkpoint_written": False, "cases": {}}
    try:
        for case in CASES:
            output = cases_root / case
            output.mkdir()
            evidence = {"status": "PASS", "batches_loaded": 0, "optimizer_step_attempts": 0, "optimizer_updates": 0,
                        "forward_attempts": 0, "forward_calls": 0, "backward_attempts": 0, "backward_calls": 0}
            summary["cases"][case] = evidence
            trainer = None
            try:
                trainer = BoundedSmokeTrainer(release, weights, output, "e1", batch_size)
                trainer.args.workers = 0
                probe(trainer, membership, case, batch_size, evidence)
            except Exception as error:
                evidence.update(status="HARD_GATE", failure={"type": type(error).__name__, "message": str(error),
                                                              "evidence": getattr(error, "evidence", {})})
                raise
            finally:
                if trainer is not None:
                    trainer.close()
                    del trainer
                torch.cuda.empty_cache()
                write_json_exclusive(output / "case.json", evidence)
        require(len({r["initial_parameter_sha256"] for r in summary["cases"].values()}) == 1,
                "Independent E1 diagnostic cases did not use the same initial parameter state")
    except Exception:
        summary["status"] = "HARD_GATE"
        raise
    finally:
        summary["optimizer_step_attempts"] = sum(r["optimizer_step_attempts"] for r in summary["cases"].values())
        write_json_exclusive(report, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", "--batch", type=int, default=32)
    args = parser.parse_args()
    result = run(args.release.resolve(strict=True), args.weights.resolve(strict=True), args.report.resolve(), args.batch_size)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
