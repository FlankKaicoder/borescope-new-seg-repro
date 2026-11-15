"""TRAIN-only Exp15 J-B strength calibration, smoke, and separately gated formal runner."""

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
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a11_full_numerics import validate_scaler_trajectory
from tools.exp14.a11_full_protocol import protocol_spec, source_sha256
from tools.exp14.a11_full_train import check_epoch_counts
from tools.exp14.a11_smoke_numerics import NumericalAudit, finite_tree
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import hash_state
from tools.exp15 import ja
from tools.exp15.jb_binding import check_pool_tags, install_format_audit
from tools.exp15.regions import P3_LAYER, forward, inspect_regions

WEIGHT = 8.0
SMOKE_STEPS = 70
SMOKE_CAP = 80
PROBES = (1, 7, 11, 21, 32, 42, 56, 70)
PASS = "EXP15_JB_PASS_READY_FOR_FORMAL_GATE"
HOLD = "EXP15_JB_HOLD_WITH_BLOCKERS"
SCHEMA = "exp15_jb_train_only_v1"
FORMAL_STEPS = 2100
AUDIT_STEPS = (1, 64, 70)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_identity():
    folder = Path(__file__).parent
    return {name: source_sha256(folder / name) for name in ("jb.py", "jb_binding.py", "ja.py", "regions.py")}


def native_layer(name):
    return name.split(".")[1] if name.startswith("model.") else "projection"


def grad_norm(vectors):
    return sum(float(x.float().square().sum()) for x in vectors if x is not None) ** 0.5


def dot(left, right):
    return sum(float((a.float() * b.float()).sum()) for a, b in zip(left, right, strict=True)
               if a is not None and b is not None)


def metrics(native, auxiliary, joint):
    n, s, j = (grad_norm(g) for g in (native, auxiliary, joint))
    return {"native_norm": n, "auxiliary_norm": s, "joint_norm": j,
            "weighted_ratio": WEIGHT * s / n if n else None,
            "cosine": dot(native, auxiliary) / (n * s) if n and s else None,
            "native_joint_cosine": dot(native, joint) / (n * j) if n and j else None}


def gradient_probe(model, native, auxiliary):
    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad], strict=False)
    y = torch.autograd.grad(native, params, retain_graph=True, allow_unused=True)
    s = torch.autograd.grad(auxiliary, params, retain_graph=True, allow_unused=True)
    joint = torch.autograd.grad(native + WEIGHT * auxiliary, params, allow_unused=True)
    require(finite_tree((y, s, joint)), "Independent or joint gradient is nonfinite")
    layers = {}
    for layer in dict.fromkeys(map(native_layer, names)):
        idx = [i for i, n in enumerate(names) if native_layer(n) == layer]
        layers[layer] = metrics(*[[g[i] for i in idx] for g in (y, s, joint)])
        layers[layer]["nonzero_auxiliary_tensors"] = sum(s[i] is not None and bool(s[i].count_nonzero()) for i in idx)
    backbone = [i for i, n in enumerate(names) if n.startswith("model.") and int(native_layer(n)) <= ja.BACKBONE_END]
    shared = metrics(*[[g[i] for i in backbone] for g in (y, s, joint)])
    require(len(backbone) == 120 and sum(layers[str(i)]["nonzero_auxiliary_tensors"] for i in range(11)) == 120
            and not any(row["nonzero_auxiliary_tensors"] for layer, row in layers.items()
                        if layer != "projection" and int(layer) > P3_LAYER), "Unexpected auxiliary gradient endpoints")
    norms = {label: grad_norm(grad) for label, grad in zip(("native", "joint"), (y, joint), strict=True)}
    return {"backbone": shared, "layers": layers, "global": norms,
            "clip_coef": {label: min(1.0, 10.0 / (norm + 1e-6)) for label, norm in norms.items()}}


def read_reference(root, public):
    replay = load(root / "preflight_r5" / "summary.json")
    require(replay["identity"] == public["identity"] and replay["status"] == "PASS_TRAIN_PREFLIGHT"
            and len(replay["batch_tensor_sha256"]) == SMOKE_STEPS, "J-A audited preflight identity differs")
    checkpoints = {}
    for variant in ("j0", "j1"):
        path = root / f"{variant}_r5" / "engineering_step_070.pt"
        expected = public["arms"][variant]["checkpoints"]["70"]["sha256"]
        require(sha256_file(path) == expected, "J-A audited checkpoint hash differs")
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(state["schema"] == ja.SEAL and state["identity"] == public["identity"]
                and state["effective_updates"] == 70 and state["arm"] == variant, "Unexpected J-A engineering checkpoint")
        checkpoints[variant] = state["state"]
    return replay, checkpoints


def calibrate(trainer, prior, states):
    trainer.setup()
    require(trainer.initial_native == prior["native_initial_state_sha256"], "Native initialization differs from J-A")
    require(len(trainer.train_loader) == 21 and trainer._get_warmup_iterations(21) == 63, "Native schedule differs")
    model = trainer.model
    initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    # J0 never had a projection. Use the same deterministic initial projection for that control.
    snapshots = {"coco_init": initial, "ja_j0_step70_initial_projection":
                 {**states["j0"], **{n: v for n, v in initial.items() if n.startswith(ja.HEAD_PREFIX)}},
                 "ja_j1_step70": states["j1"]}
    install_format_audit(trainer.train_loader.dataset)
    private = trainer.output / "private_binding"
    private.mkdir()
    counts, featured, probes, coverage = Counter(), set(), [], []
    for step, batch in trainer.batches():
        digest = ja.batch_digest(batch)
        require(digest == prior["batch_tensor_sha256"][step - 1], "Binding instrumentation changed native augmentation")
        regions = inspect_regions(batch, trainer.groups)
        detail = regions.report()
        require(detail == prior["per_batch_regions"][step - 1], "J-A region coverage changed")
        coverage.append(detail)
        counts.update(check_pool_tags(batch, trainer.groups, private, featured))
        if step not in PROBES:
            continue
        payload = trainer.preprocess_batch({k: v for k, v in batch.items() if k != "jb_binding"})
        for label, state in snapshots.items():
            model.load_state_dict(state, strict=True)
            before = ja.bn_fingerprint(model)
            with autocast(trainer.amp, device="cuda"):
                native, parts, auxiliary = forward(model, payload, regions, lambda: ja.bn_fingerprint(model))
            require(ja.bn_fingerprint(model) != before and finite_tree((native, parts, auxiliary)),
                    "Nonfinite calibration loss or native BN did not advance")
            record = gradient_probe(model, native, auxiliary)
            record.update(batch=step, snapshot=label, native_loss=float(native.detach()),
                          auxiliary_loss=float(auxiliary.detach()), joint_loss=float((native + WEIGHT * auxiliary).detach()))
            probes.append(record)
    require(trainer.audit.optimizer_attempts == trainer.audit.native_steps == 0 and len(coverage) == SMOKE_STEPS
            and len(probes) == len(PROBES) * len(snapshots), "Calibration step count differs")
    require(featured == {"high_overlap", "small_polygon", "thin_crack", "mixed_class"},
            "Targeted TRAIN binding checks lack one of the requested case types")
    medians = {label: statistics.median(p["backbone"]["weighted_ratio"] for p in probes if p["snapshot"] == label)
               for label in snapshots}
    peak = max(p["backbone"]["weighted_ratio"] for p in probes)
    cosine_min = min(p["backbone"]["native_joint_cosine"] for p in probes)
    # Prespecified conservative engineering screen, before any optimizer update.
    accepted = all(0.01 <= x <= 0.10 for x in medians.values()) and peak <= 0.20 and cosine_min >= 0.98
    return {"status": "PASS_TRAIN_CALIBRATION" if accepted else HOLD, "schema": SCHEMA,
            "identity": trainer.identity, "runtime": trainer.runtime, "lambda_s_candidate": WEIGHT,
            "native_initial_sha256": trainer.initial_native, "initial_bn_sha256": trainer.initial_bn,
            "training_batches": len(coverage), "coverage": ja.summarize_regions(coverage), "binding": dict(counts),
            "featured_binding_cases": sorted(featured), "optimizer_attempts": 0,
            "prespecified_screen": {"snapshot_median_range": [0.01, 0.10], "max_backbone_ratio": 0.20,
                                    "min_native_joint_cosine": 0.98},
            "snapshot_weighted_medians": medians, "max_backbone_ratio": peak, "min_native_joint_cosine": cosine_min,
            "probes": probes}


def reserve_attempt(budget, cap, variant):
    # Protected by the caller's exclusive lock. A crash after writing still consumes the attempt.
    with (budget / f"{variant}.attempts").open("a+", encoding="ascii") as stream:
        stream.seek(0)
        tickets = stream.read().splitlines()
        require(len(tickets) < cap and tickets == [str(n) for n in range(1, len(tickets) + 1)],
                "Independent J-B attempt ledger exhausted or invalid")
        stream.write(f"{len(tickets) + 1}\n")
        stream.flush()
        os.fsync(stream.fileno())
        return len(tickets) + 1


def matched_batches(trainer, limit):
    dataset = trainer.train_loader.dataset
    step = 0
    for trainer.epoch in range(100):
        trainer._model_train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer.scheduler.step()
        before, seen = trainer.counters.copy(), set()
        tail, batches = 0, 0
        for batch in islice(trainer.new_epoch_loader(dataset, trainer.epoch), limit - step):
            trainer.warmup(step, 63)
            current = trainer.audit_batch(batch)
            require(not seen.intersection(current), "TRAIN image repeated within formal epoch")
            seen.update(current)
            step += 1
            batches += 1
            tail = len(current)
            yield step, batch
        if batches == len(trainer.train_loader):
            check_epoch_counts(trainer.counters - before, trainer.expected_arm, batches, tail, trainer.expected.keys(), seen)
        if step == limit:
            break
    require(step == limit, "Matched steps differ from the frozen budget")


def run_arm(trainer, prior, gate, budget, formal=False):
    trainer.setup()
    trainer.audit.close()
    limit = FORMAL_STEPS if formal else SMOKE_STEPS
    cap = FORMAL_STEPS if formal else SMOKE_CAP
    trainer.audit = NumericalAudit(trainer, update_limit=limit)
    require(trainer.initial_native == gate["native_initial_sha256"]
            and trainer.initial_bn == gate["initial_bn_sha256"] and len(trainer.train_loader) == 21
            and trainer._get_warmup_iterations(21) == 63, "Run initialization or schedule differs")
    model = trainer.model
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    backbone = {n: p for n, p in params.items() if n.startswith("model.") and int(native_layer(n)) <= ja.BACKBONE_END}
    projection = {n: p for n, p in params.items() if n.startswith(ja.HEAD_PREFIX)}
    weight = WEIGHT if trainer.variant == "j1" else 0.0
    require(len(backbone) == 120 and len(projection) == (4 if weight else 0), "Unexpected trainable parameter identity")
    before = hash_state(backbone), hash_state(projection)
    reports, coverage, checkpoints = [], [], {}
    batches = matched_batches(trainer, limit)
    torch.cuda.reset_peak_memory_stats()
    for step, batch in batches:
        digest = ja.batch_digest(batch)
        if step <= SMOKE_STEPS:
            require(digest == prior["batch_tensor_sha256"][step - 1], "Main task TRAIN tensor or order differs from J-A")
        region = inspect_regions(batch, trainer.groups) if weight else None
        audit_roi = region.report() if region is not None else None
        if weight and step <= SMOKE_STEPS:
            require(audit_roi == prior["per_batch_regions"][step - 1], "Main task Polygon/anchor coverage differs from J-A")
        timed = time.perf_counter()
        bn_watch = step in AUDIT_STEPS or (formal and (step - 1) % 21 == 0)
        bn = (lambda: ja.bn_fingerprint(model)) if bn_watch else None
        old_bn = bn() if bn else None
        with autocast(trainer.amp, device="cuda"):
            native, parts, auxiliary = forward(model, trainer.preprocess_batch(batch), region, bn)
            trainer.loss = native + weight * auxiliary if weight else native
            if weight:
                require(finite_tree(auxiliary), "SupCon loss is not finite")
            forward_audit = trainer.audit.check_forward(trainer.loss, parts)
        if bn:
            require(bn() != old_bn, "Native forward did not update BatchNorm")
        trainer.scaler.scale(trainer.loss).backward()
        projection_nonzero = sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in projection.values())
        require(not weight or projection_nonzero == len(projection), "Projection gradient missing or zero")
        ticket = reserve_attempt(budget, cap, trainer.variant)
        try:
            audit = trainer.audit.optimizer_step()
        except Exception as error:
            trainer.event("optimizer_stopped", step=step, ticket=ticket, **getattr(error, "evidence", {}))
            raise
        require(trainer.audit.native_steps == trainer.audit.effective_updates == step == trainer.ema.updates,
                "Native AdamW or EMA count differs")
        validate_scaler_trajectory(audit["scaler_state"], step)
        report = {"step": step, "ticket": ticket, "batch_sha256": digest, "native_loss": float(native.detach()),
                  "supcon_loss": float(auxiliary.detach()) if weight else None, "total_loss": forward_audit["loss"],
                  "global_gradient_norm": audit["gradient_norm"], "clip_coef": min(1.0, 10.0 / (audit["gradient_norm"] + 1e-6)),
                  "projection_nonzero_gradients": projection_nonzero, "bn_native_only_check": bool(bn),
                  "lr": [g["lr"] for g in trainer.optimizer.param_groups], "audited_seconds": time.perf_counter() - timed}
        reports.append(report)
        if audit_roi:
            coverage.append(audit_roi)
        trainer.event("jb_update", **report, coverage=audit_roi, optimizer=audit)
        needs_checkpoint = step in (1, limit) or (formal and step % 105 == 0)
        if needs_checkpoint:
            require(hash_state(backbone) != before[0] and (not weight or hash_state(projection) != before[1]),
                    "Backbone or projection parameter update absent")
            name = "last.pt" if formal and step == limit else f"step_{step:04}.pt"
            checkpoints[str(step)] = ja.save_checkpoint(trainer, trainer.output / name)
    require(len(reports) == limit and not trainer.audit.amp_skips and not trainer.counters["repair_exposures"],
            "Run budget, AMP or TRAIN accounting changed")
    return {"status": "PASS_FORMAL_ARM" if formal else "PASS_JB_BOUNDED_SMOKE", "schema": SCHEMA,
            "identity": trainer.identity, "variant": trainer.variant, "lambda_s": weight, "runtime": trainer.runtime,
            "initial_native_sha256": trainer.initial_native, "counts": dict(trainer.counters), "attempt_cap": cap,
            "native_optimizer_calls": trainer.audit.native_steps, "effective_updates": trainer.audit.effective_updates,
            "optimizer_attempts": trainer.audit.optimizer_attempts, "amp_skips": trainer.audit.amp_skips,
            "warmup_iterations": 63, "post_warmup_steps": limit - 63, "native_groups": [
                [n for n in group if not n.startswith(ja.HEAD_PREFIX)] for group in trainer.optimizer_names],
            "optimizer_unique_parameters": sum(map(len, trainer.optimizer_names)), "checkpoints": checkpoints,
            "backbone_before": before[0], "backbone_after": hash_state(backbone),
            "projection_before": before[1], "projection_after": hash_state(projection),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "coverage": ja.summarize_regions(coverage) if weight else None, "updates": reports}


def formal_protocol(identity, calibration_sha, smoke_sha):
    copied = protocol_spec()
    return {"schema": "exp15_j0_j1_formal_v1", "status": PASS, "authorization": "PENDING_INDEPENDENT_GATE",
            "identity": identity, "calibration_report_sha256": calibration_sha, "smoke_report_sha256": smoke_sha,
            "lambda_s": {"j0": 0.0, "j1": WEIGHT}, "temperature": 0.1,
            "train": copied["training"], "evaluation": copied["evaluation"], "runtime": copied["runtime"],
            "epochs": 100, "effective_updates_per_arm": FORMAL_STEPS, "original_images_per_epoch": 668,
            "raw_instances_per_epoch": 1266, "batches_per_epoch": 21, "warmup_iterations": 63,
            "initialization": "FRESH_FROZEN_COCO_TRANSFER_AND_SHARED_NATIVE_STATE_SHA",
            "checkpoint": "FP32_MASTER_LAST_AT_EPOCH100", "heldout_access_during_training": False,
            "hard_gate": "STOP_PRESERVE_EVIDENCE_NO_AUTOMATIC_RETRY_OR_HYPERPARAMETER_CHANGE"}


def verify_authorization(protocol, authorization, identity):
    require(protocol.is_file() and authorization.is_file(), "Formal protocol and separate authorization are required")
    spec = load(protocol)
    require(spec == formal_protocol(identity, spec["calibration_report_sha256"], spec["smoke_report_sha256"]),
            "Formal protocol or runtime source identity differs")
    require(load(authorization) == {"stage": "EXP15_J0_J1_FORMAL_AUTHORIZATION", "decision": "PASS",
                                    "protocol_sha256": sha256_file(protocol), "authorized_arms": ["j0", "j1"]},
            "Separate formal Gate is not valid for this exact protocol")
    return spec


def run(args):
    import fcntl

    output, release = args.output.resolve(), args.release.resolve(strict=True)
    budget = args.budget.resolve(strict=True)
    history = args.ja_root.resolve(strict=True)
    protected = (release.parent, history, release.parent.parent / "损伤训练数据集",
                 release.parent.parent / "borescope-new-seg-data")
    for target in (output, budget, Path(os.environ["YOLO_CONFIG_DIR"]).resolve()):
        require(not any(target.is_relative_to(folder) for folder in protected), "Use an independent J-B run directory")
    identity = {"files": source_identity(), "frozen_split_sha256": sha256_file(release / "configs/frozen_split_manifest.csv"),
                "coco_sha256": sha256_file(args.weight), "ja_audited_summary_sha256": sha256_file(args.ja_public)}
    formal = args.phase == "formal"
    if formal:
        spec = verify_authorization(args.protocol, args.authorize, identity)
        require(sha256_file(args.calibration) == spec["calibration_report_sha256"]
                and sha256_file(args.smoke) == spec["smoke_report_sha256"], "Formal evidence binding differs")
        smoke = load(args.smoke)
        require(smoke["status"] == "PASS_JB_BOUNDED_SMOKE" and smoke["identity"] == identity
                and smoke["lambda_s"] == WEIGHT and smoke["effective_updates"] == SMOKE_STEPS
                and smoke["amp_skips"] == 0 and smoke["variant"] == "j1", "The bound smoke did not pass")
        require(budget.name == "formal_budget", "Use the independent formal budget")
    else:
        require(budget.name == "jb_budget", "Use the independent J-B budget")
    output.mkdir()
    trainer = None
    started = time.perf_counter()
    try:
        public = load(args.ja_public)
        prior, states = read_reference(history, public)
        groups = ja.checked_train(release)
        trainer = ja.JATrainer(release, args.weight.resolve(strict=True), output, args.arm or "j1", groups, identity)
        if args.phase == "calibrate":
            result = calibrate(trainer, prior, states)
        else:
            gate = load(args.calibration)
            require(gate["status"] == "PASS_TRAIN_CALIBRATION" and gate["identity"] == identity
                    and gate["lambda_s_candidate"] == WEIGHT, "Calibration did not pass for this exact runtime identity")
            with (budget / "lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if formal:
                    require(not (budget / f"{trainer.variant}.attempts").exists(), "Formal arm already started")
                result = run_arm(trainer, prior, gate, budget, formal=formal)
        require(ja.checked_train(release) == groups, "Original TRAIN identity changed")
        result["elapsed_seconds"] = time.perf_counter() - started
        ja.record_json(output / "summary.json", result)
        require(result["status"] != HOLD, "TRAIN calibrations failed the predeclared engineering screen")
    except Exception as failure:
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        audit = getattr(trainer, "audit", None)
        ja.record_json(output / "failed.json", {"status": HOLD, "phase": args.phase, "error": type(failure).__name__,
                      "attempts": getattr(audit, "optimizer_attempts", 0),
                      "native_steps": getattr(audit, "native_steps", 0),
                      "effective_updates": getattr(audit, "effective_updates", 0),
                      "numerics": getattr(failure, "evidence", {})})
        raise
    finally:
        if trainer:
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "smoke", "formal"))
    for name in ("release", "weight", "output", "budget", "ja-root", "ja-public"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("calibration", "protocol", "authorize", "smoke"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--arm", choices=("j0", "j1"))
    args = parser.parse_args()
    if args.phase == "smoke" and (not args.calibration or args.arm not in (None, "j1")):
        parser.error("Only a calibrated J1 is authorized by the bounded smoke entry")
    if args.phase == "formal" and not all((args.arm, args.calibration, args.protocol, args.authorize, args.smoke)):
        parser.error("Formal requires the exact protocol, evidence, arm and a separate PASS authorization")
    if args.phase == "calibrate" and args.arm not in (None, "j1"):
        parser.error("Calibration needs the J1 architecture")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
