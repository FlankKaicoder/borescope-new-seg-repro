"""Run the Exp15 J-A TRAIN-only preflight and the capped J0/J1 engineering smoke."""

import argparse
import json
import os
import statistics
import time
import traceback
import warnings
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a10_dataset_entries import load_frozen_split, verify_inventory
from tools.exp14.a11_full_numerics import adam_step_counts, validate_scaler_trajectory
from tools.exp14.a11_full_protocol import protocol_spec, source_sha256
from tools.exp14.a11_full_train import build_formal_trainer, check_epoch_counts
from tools.exp14.a11_smoke_numerics import NumericalAudit, exact_tree, finite_tree, optimizer_group_names
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp15.regions import P3_LAYER, add_projection, forward, inspect_regions

MAX_ATTEMPTS = 120
STEPS = 70
BACKBONE_END = 10
HEAD_PREFIX = "region_projection."
SEAL = "exp15_ja_train_smoke_v1"
FILES = ("ja.py", "regions.py", "test_ja.py")
FormalTrainer = build_formal_trainer()


def code_identity():
    return {name: source_sha256(Path(__file__).with_name(name)) for name in FILES}


def checked_train(release):
    configs = release / "configs"
    split = load_frozen_split(configs / "frozen_split_manifest.csv")
    inventory = json.loads((configs / "train_entry_files.json").read_text())
    verify_inventory(inventory, split)
    paths = {}
    dataset = release / "dataset"
    for row in inventory["baseline"]:
        image, label = (dataset / row[key] for key in ("image_path", "label_path"))
        require(image.resolve() == image and label.resolve() == label and image.is_relative_to(dataset)
                and label.is_relative_to(dataset), "TRAIN image or label resolves outside the release")
        require(sha256_file(image) == row["image_sha256"] and sha256_file(label) == row["label_sha256"],
                "Frozen original TRAIN pixels or labels changed")
        paths[str(image)] = row["group_id"]
    require(len(paths) == 668 and len(set(paths.values())) > 1, "Frozen TRAIN identity count differs")
    return paths


def native_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith(HEAD_PREFIX)}


def bn_fingerprint(model):
    names = ("running_mean", "running_var", "num_batches_tracked")
    return hash_state({k: v for k, v in model.named_buffers() if k.endswith(names)})


def named_parameter_ids(model, optimizer):
    names = optimizer_group_names(model, optimizer)
    flat = [n for group in names for n in group]
    required = {n for n, p in model.named_parameters() if p.requires_grad}
    require(len(flat) == len(set(flat)) and set(flat) >= required, "Optimized parameters are repeated or missing")
    return names


def probe_gradients(model, native, auxiliary):
    shared = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith("model.")
              and int(n.split(".")[1]) <= BACKBONE_END]
    all_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    wy = torch.autograd.grad(native, [p for _, p in shared], retain_graph=True, allow_unused=True)
    ws = torch.autograd.grad(auxiliary, [p for _, p in all_trainable], allow_unused=True)
    require(finite_tree((wy, ws)), "Nonfinite independent backbone gradients")
    by_name = dict(zip((n for n, _ in all_trainable), ws, strict=True))
    norm_by_layer = Counter()
    reached = Counter()
    for n, _ in all_trainable:
        g = by_name[n]
        if g is not None:
            name = n.split(".")[1] if n.startswith("model.") else "projection"
            square = float(g.float().square().sum())
            norm_by_layer[name] += square
            reached[name] += int(square > 0)
    dot = sy = ss = 0.0
    for (name, _), y in zip(shared, wy, strict=True):
        s = by_name[name]
        if y is not None:
            sy += float(y.float().square().sum())
        if s is not None:
            ss += float(s.float().square().sum())
        if y is not None and s is not None:
            dot += float((y.float() * s.float()).sum())
    require(sy > 0 and ss > 0 and reached["projection"] > 0 and not any(
        reached[str(layer)] for layer in range(P3_LAYER + 1, len(model.model))), "Auxiliary gradient endpoints differ")
    return {"native_backbone_norm": sy**0.5, "supcon_backbone_norm": ss**0.5,
            "gradient_cosine": dot / (sy * ss)**0.5, "norm_ratio": (ss / sy)**0.5,
            "supcon_norm_by_layer": {k: v**0.5 for k, v in norm_by_layer.items()},
            "nonzero_aux_tensors_by_layer": dict(reached)}


class JATrainer(FormalTrainer):
    def __init__(self, release, weight, output, variant, groups, identity):
        self.variant, self.groups = variant, groups
        super().__init__(release, weight, output, "e0", protocol_spec(), identity)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if self.variant == "j1":
            add_projection(model)
        return model

    def event(self, kind, **values):
        super().event(kind, variant=self.variant, **values)

    def setup(self):
        super().setup()
        self.audit.close()
        self.audit = NumericalAudit(self, update_limit=STEPS)
        self.initial_native = hash_state(native_state(self.model))
        self.initial_bn = bn_fingerprint(self.model)
        self.optimizer_names = named_parameter_ids(self.model, self.optimizer)

    def run_formal(self, resume):
        raise RuntimeError("This entry point is limited to engineering smoke")

    def close(self):
        super().close()
        if hasattr(self, "train_loader") and hasattr(self.train_loader, "close"):
            self.train_loader.close()

    def batches(self):
        dataset = self.train_loader.dataset
        consumed = 0
        for self.epoch in range(4):
            self._model_train()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.scheduler.step()
            delta = self.counters.copy()
            batches, tail, seen = 0, 0, set()
            for batch in islice(self.new_epoch_loader(dataset, self.epoch), STEPS - consumed):
                self.warmup(consumed, 63)
                paths = self.audit_batch(batch)
                require(not seen.intersection(paths), "Within-epoch source repetition")
                seen.update(paths)
                batches += 1
                tail = len(paths)
                consumed += 1
                yield consumed, batch
            if batches == len(self.train_loader):
                check_epoch_counts(self.counters - delta, self.expected_arm, batches, tail, self.expected.keys(), seen)
            if consumed == STEPS:
                break


def batch_digest(batch):
    keys = ("img", "cls", "bboxes", "masks", "sem_masks", "batch_idx")
    return hash_state({key: batch[key] for key in keys})


def probe_disabled(trainer, batch):
    """Check attachment with lambda=0 against the native model and loss on the same input."""
    before = hash_state(native_state(trainer.model))
    with torch.random.fork_rng(devices=[]):
        bare = SegmentationModel(trainer.model.yaml, ch=3, nc=7, verbose=False).to(trainer.device)
    bare.load_state_dict(native_state(trainer.model), strict=True)
    bare.args = trainer.model.args
    bare.train()
    attached = dict(trainer.model.named_parameters())
    reference, observed = [], []
    for name, p in bare.named_parameters():
        p.requires_grad_(attached[name].requires_grad)
        if p.requires_grad:
            reference.append(p)
            observed.append(attached[name])
    native = trainer.preprocess_batch(dict(batch))
    with autocast(trainer.amp, device=trainer.device.type):
        expected, expected_parts = bare(native)
    ga = torch.autograd.grad(expected.sum(), reference, allow_unused=True)
    expected_loss, expected_parts = expected.detach().sum(), cpu_state(expected_parts)
    expected_hash = hash_state(bare.state_dict())
    del expected, bare
    with autocast(trainer.amp, device=trainer.device.type):
        actual, actual_parts, disabled = forward(trainer.model, native)
    require(disabled is None and torch.equal(actual, expected_loss) and exact_tree(cpu_state(actual_parts), expected_parts)
            and expected_hash == hash_state(native_state(trainer.model)),
            "Attached model at zero auxiliary weight differs from native forward or BN")
    gb = torch.autograd.grad(actual, observed, allow_unused=True)
    require(exact_tree(ga, gb), "Attached model at zero auxiliary weight differs from native backward")
    return {"native_initial_sha256": before, "compared_native_trainable_tensors": len(reference),
            "native_loss_forward_backward_and_bn_bitwise_equal": True}


def save_checkpoint(trainer, file):
    model = trainer.model
    audit = trainer.audit
    updates = audit.effective_updates
    state = cpu_state(model.state_dict())
    ema = cpu_state(trainer.ema.ema.state_dict())
    record = {"schema": SEAL, "arm": trainer.variant, "effective_updates": updates, "model_config": model.yaml,
              "class_names": dict(enumerate(CLASSES)), "weight_transfer": trainer.weight_transfer,
              "identity": trainer.identity, "optimizer_names": trainer.optimizer_names,
              "native_state_sha256": hash_state(native_state(model)), "model_state_sha256": hash_state(state),
              "ema_state_sha256": hash_state(ema), "ema_updates": trainer.ema.updates,
              "state": state, "ema": ema, "optimizer": cpu_state(trainer.optimizer.state_dict()),
              "scaler": trainer.scaler.state_dict(), "scheduler": trainer.scheduler.state_dict()}
    with file.open("xb") as stream:
        torch.save(record, stream)
    check = torch.load(file, map_location="cpu", weights_only=True)
    require(exact_tree(record, check) and finite_tree(check) and audit.native_steps == updates == check["ema_updates"],
            "Engineering checkpoint serialization or step count changed")
    validate_scaler_trajectory(check["scaler"], updates)
    adam_step_counts(check["optimizer"]["state"], updates)
    with torch.random.fork_rng(devices=[]):
        reloaded = SegmentationModel(check["model_config"], ch=3, nc=7, verbose=False)
        if trainer.variant == "j1":
            add_projection(reloaded)
    for key in ("ema", "state"):
        reloaded.load_state_dict(check[key], strict=True)
        require(exact_tree(dict(reloaded.state_dict()), check[key]), "Strict checkpoint model state differs")
    params = dict(reloaded.named_parameters())
    optim = torch.optim.AdamW([{"params": [params[name] for name in names]} for names in check["optimizer_names"]])
    optim.load_state_dict(check["optimizer"])
    scaler = torch.amp.GradScaler("cpu")
    scaler.load_state_dict(check["scaler"])
    require(exact_tree(optim.state_dict(), check["optimizer"]) and scaler.state_dict() == check["scaler"],
            "Strict checkpoint optimizer or scaler state differs")
    if trainer.variant == "j1":
        del reloaded.region_projection
    require(exact_tree(dict(reloaded.state_dict()), native_state(reloaded))
            and hash_state(reloaded.state_dict()) == check["native_state_sha256"],
            "Removing the projection did not preserve the strict native state")
    reloaded.names = check["class_names"]
    # CPU prediction uses a synthetic image, independent of held-out data.
    reloaded.eval()
    with torch.no_grad():
        prediction, _ = reloaded(torch.zeros(1, 3, 64, 64))
    require(finite_tree(prediction), "Stripped native model produced nonfinite output")
    return {"sha256": sha256_file(file), "strict_reload": True, "stripped_native_forward": True,
            "state_sha256": check["model_state_sha256"], "native_state_sha256": check["native_state_sha256"]}


def record_json(path, result):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def visualize_private(output, batch, regions):
    from PIL import Image, ImageDraw

    directory = output / "private_roi_examples"
    directory.mkdir()
    files = []
    for slot in list(dict.fromkeys(regions.slots[(regions.cells > 0) & (regions.cells <= 4)]))[:2]:
        idx = int(np.flatnonzero((regions.slots == slot) & (regions.cells > 0) & (regions.cells <= 4))[0])
        image = batch["img"][slot].permute(1, 2, 0).numpy()
        mask = batch["masks"][slot].numpy() == regions.ordinals[idx] + 1
        y, x = np.nonzero(mask)
        cy, cx = int(y.mean()) * 4, int(x.mean()) * 4
        left, top = max(0, cx - 80), max(0, cy - 80)
        right, bottom = min(image.shape[1], cx + 80), min(image.shape[0], cy + 80)
        colored = image.copy()
        color_mask = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)
        colored[color_mask] = colored[color_mask] // 2 + np.array([0, 127, 64], dtype=np.uint8)
        panel = Image.fromarray(colored).crop((left, top, right, bottom))
        draw = ImageDraw.Draw(panel)
        for xx, yy in set(zip(x // 2, y // 2)):
            draw.rectangle((8 * xx - left, 8 * yy - top, 8 * xx + 7 - left, 8 * yy + 7 - top), outline="#ffff00")
        file = directory / f"tiny_{len(files) + 1}.png"
        panel.resize((panel.width * 3, panel.height * 3)).save(file)
        files.append(file.name)
    return files


def summarize_regions(trace):
    return {"batches": len(trace), "zero_eligible_batches": sum(r["eligible_anchors"] == 0 for r in trace),
            "min_positive_pairs": min(r["positives"] for r in trace),
            "median_positive_pairs": statistics.median(r["positives"] for r in trace),
            "min_negative_pairs": min(r["negatives"] for r in trace),
            "min_eligible_anchors": min(r["eligible_anchors"] for r in trace),
            "median_eligible_anchors": statistics.median(r["eligible_anchors"] for r in trace),
            **{k: sum(r[k] for r in trace) for k in (
                "targets", "missing_native_raster", "p3_cells_1", "p3_cells_2_to_4", "p3_cells_over_4")},
            **{k: np.sum([r[k] for r in trace], axis=0).tolist() for k in (
                "targets_by_class", "raster_missing_by_class", "eligible_anchors_by_class")},
            "batches_with_anchors_by_class": np.count_nonzero([r["eligible_anchors_by_class"] for r in trace], axis=0).tolist(),
            "median_batch_box_fill": statistics.median(r["box_fraction_q10_q50_q90"][1] for r in trace)}


def preflight(trainer):
    trainer.setup()
    require(len(trainer.train_loader) == 21 and trainer._get_warmup_iterations(21) == 63, "Unexpected native schedule")
    coverage, probes, digests, examples = [], [], [], []
    zero = None
    diagnostic_steps = (1, 11, 21)
    for step, batch in trainer.batches():
        region = inspect_regions(batch, trainer.groups)
        coverage.append(region.report())
        digests.append(batch_digest(batch))
        if not examples and ((0 < region.cells) & (region.cells <= 4)).any():
            example_dir = trainer.output / "private_roi_examples"
            if not example_dir.exists():
                examples = visualize_private(trainer.output, batch, region)
        if step not in diagnostic_steps:
            continue
        before = bn_fingerprint(trainer.model)
        require(trainer.audit.native_steps == 0, "Preflight performed an optimizer update")
        if step == 1:
            pristine = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
            zero = probe_disabled(trainer, batch)
            trainer.model.load_state_dict(pristine, strict=True)
            require(bn_fingerprint(trainer.model) == before, "Could not restore BN before diagnostic")
            del pristine
        with autocast(trainer.amp, device="cuda"):
            native = trainer.preprocess_batch(dict(batch))
            y, parts, s = forward(trainer.model, native, region, lambda: bn_fingerprint(trainer.model))
        advanced = bn_fingerprint(trainer.model) != before
        require(advanced and finite_tree((y, s, parts)), "BN did not advance or preflight loss was nonfinite")
        report = probe_gradients(trainer.model, y, s)
        probes.append({"batch": step, "native_loss": float(y.detach()), "supcon_loss": float(s.detach()),
                       "native_bn_advanced": advanced, "aux_bn_unchanged": True, **report})
    stats = summarize_regions(coverage)
    require(len(coverage) == STEPS and stats["zero_eligible_batches"] == 0
            and all(stats["eligible_anchors_by_class"]), "Insufficient real cross-group SupCon coverage")
    peak_ratio = max(p["norm_ratio"] for p in probes)
    weight = min(1.0, 0.05 / peak_ratio)
    return {"status": "PASS_TRAIN_PREFLIGHT", "scope": SEAL, "identity": trainer.identity,
            "native_initial_state_sha256": trainer.initial_native, "initial_bn_sha256": trainer.initial_bn,
            "runtime": trainer.runtime, "disabled_equivalence": zero, "coverage": stats,
            "per_batch_regions": coverage, "batch_tensor_sha256": digests, "gradients": probes,
            "weight_selection": {"lambda_s": weight, "target_max_weighted_backbone_norm_ratio": 0.05, "ceiling": 1.0,
                                 "uses_yolo_batch_sum_and_supcon_anchor_class_mean": True},
            "private_visuals": len(examples), "effective_updates": 0}


def reserve_attempt(budget, variant):
    ledger = budget / f"{variant}.attempts"
    with ledger.open("a+", encoding="ascii") as stream:
        stream.seek(0)
        previous = stream.readlines()
        require(len(previous) < MAX_ATTEMPTS, "Cumulative arm optimizer-attempt ceiling reached")
        stream.write(f"{len(previous) + 1}\n")
        stream.flush()
        os.fsync(stream.fileno())


def smoke(trainer, gate, budget):
    trainer.setup()
    require(trainer.initial_native == gate["native_initial_state_sha256"]
            and trainer.initial_bn == gate["initial_bn_sha256"] and len(trainer.train_loader) == 21
            and trainer._get_warmup_iterations(21) == 63, "Smoke initial model, BN or schedule changed")
    variant = trainer.variant
    params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    backbone = {n: p for n, p in params.items() if n.startswith("model.") and int(n.split(".")[1]) <= BACKBONE_END}
    head = {n: p for n, p in params.items() if n.startswith(HEAD_PREFIX)}
    initial_backbone = hash_state(backbone)
    initial_head = hash_state(head)
    trace = []
    weight = gate["weight_selection"]["lambda_s"] if variant == "j1" else 0.0
    require((not head and variant == "j0") or (head and variant == "j1" and 0 < weight <= 1), "Unexpected arm projection")
    checkpoints = {}
    torch.cuda.reset_peak_memory_stats()
    for step, batch in trainer.batches():
        digest = batch_digest(batch)
        require(digest == gate["batch_tensor_sha256"][step - 1], "Actual TRAIN batch/order/augmentation differs")
        region = inspect_regions(batch, trainer.groups) if weight else None
        if region:
            require(region.report() == gate["per_batch_regions"][step - 1], "Actual auxiliary ROI sampler differs")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast(trainer.amp, device="cuda"):
            y, parts, s = forward(trainer.model, trainer.preprocess_batch(batch), region)
            trainer.loss = y if s is None else y + weight * s
            evidence = trainer.audit.check_forward(trainer.loss, parts)
        trainer.scaler.scale(trainer.loss).backward()
        if head:
            require(all(p.grad is not None for p in head.values())
                    and any(bool(p.grad.count_nonzero()) for p in head.values()), "Projection gradient is absent or zero")
        torch.cuda.synchronize()
        compute_seconds = time.perf_counter() - started
        reserve_attempt(budget, variant)
        numerical = trainer.audit.optimizer_step()
        torch.cuda.synchronize()
        require(trainer.audit.effective_updates == trainer.audit.native_steps == trainer.ema.updates == step,
                "Expected one native optimizer/EMA update per main batch")
        validate_scaler_trajectory(numerical["scaler_state"], step)
        row = {"step": step, "batch_sha256": digest, "native_loss": float(y.detach()),
               "native_loss_items": evidence["loss_items"], "total_loss": evidence["loss"],
               "supcon": None if s is None else float(s.detach()), "compute_seconds": compute_seconds,
               "audited_step_seconds": time.perf_counter() - started,
               "lrs": [g["lr"] for g in trainer.optimizer.param_groups]}
        trace.append(row)
        trainer.event("update", **row, optimizer=numerical)
        if step in (1, STEPS):
            require(hash_state(backbone) != initial_backbone
                    and (not head or hash_state(head) != initial_head),
                    "Backbone or projection parameters did not change")
            checkpoints[str(step)] = save_checkpoint(trainer, trainer.output / f"engineering_step_{step:03}.pt")
    require(len(trace) == STEPS and trainer.audit.amp_skips == 0 and not trainer.counters["repair_exposures"],
            "Smoke terminal budget or sample accounting differs")
    measured = [r for r in trace if r["step"] > 63]
    return {"status": "PASS_BOUNDED_SMOKE_ARM", "variant": variant, "scope": SEAL, "identity": trainer.identity,
            "runtime": trainer.runtime, "lambda_s": weight, "arm_attempt_ceiling": MAX_ATTEMPTS,
            "steps": len(trace), "after_warmup": len(measured), "warmup_iterations": 63, "amp_skips": 0,
            "initial_native_state_sha256": trainer.initial_native, "initial_bn_sha256": trainer.initial_bn,
            "final_bn_sha256": bn_fingerprint(trainer.model), "backbone_parameter_sha256_before": initial_backbone,
            "backbone_parameter_sha256_after": hash_state(backbone), "projection_parameter_sha256_before": initial_head if head else None,
            "projection_parameter_sha256_after": hash_state(head) if head else None,
            "optimizer_unique_parameters": sum(map(len, trainer.optimizer_names)),
            "native_parameter_groups": [[n for n in group if not n.startswith(HEAD_PREFIX)] for group in trainer.optimizer_names],
            "counts": dict(trainer.counters), "trace": trace, "checkpoints": checkpoints,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "median_postwarmup_forward_backward_seconds": statistics.median(r["compute_seconds"] for r in measured),
            "median_postwarmup_audited_step_seconds": statistics.median(r["audited_step_seconds"] for r in measured)}


def run(args):
    import fcntl  # This launcher targets the audited Linux AutoDL runtime.

    release = args.release.resolve(strict=True)
    output = args.output.resolve()
    backbone_roots = (release.parent, release.parent.parent / "损伤训练数据集",
                      release.parent.parent / "borescope-new-seg-data")
    for target in (output, Path(os.environ["YOLO_CONFIG_DIR"]).resolve(), args.budget.resolve()):
        require(not any(target.is_relative_to(p) for p in backbone_roots), "Use a separate writable experiment directory")
    output.mkdir()
    trainer = None
    try:
        identity = {"code": code_identity(), "coco_sha256": sha256_file(args.weight),
                    "train_split_sha256": sha256_file(release / "configs/frozen_split_manifest.csv")}
        groups = checked_train(release)
        variant = args.arm if args.phase == "smoke" else "j1"
        trainer = JATrainer(release, args.weight.resolve(strict=True), output, variant, groups, identity)
        if args.phase == "preflight":
            result = preflight(trainer)
        else:
            gate = json.loads(args.preflight.read_text())
            require(gate["status"] == "PASS_TRAIN_PREFLIGHT" and gate["identity"] == identity
                    and len(gate["batch_tensor_sha256"]) == STEPS and gate["effective_updates"] == 0,
                    "Smoke requires a successful preflight of this code and frozen input")
            with (args.budget / "lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = smoke(trainer, gate, args.budget)
        require(checked_train(release) == groups, "After-run TRAIN identity differs")
        record_json(output / "summary.json", result)
    except Exception as failure:
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        audit = getattr(trainer, "audit", None)
        record_json(output / "failed.json", {"phase": args.phase, "variant": getattr(trainer, "variant", None),
                    "attempts": getattr(audit, "optimizer_attempts", 0), "native_steps": getattr(audit, "native_steps", 0),
                    "effective_updates": getattr(audit, "effective_updates", 0),
                    "error_type": type(failure).__name__, "numerical_evidence": getattr(failure, "evidence", {})})
        raise
    finally:
        if trainer:
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "smoke"))
    for name in ("release", "weight", "output", "budget"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=("j0", "j1"))
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.phase == "smoke" and (not args.arm or not args.preflight):
        parser.error("smoke requires --arm and --preflight")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()

