"""Minimal J3 adapter of the audited J-B epoch/update loop; no resume path."""

import time
from unittest.mock import patch

import torch
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import autocast

from tools.exp14 import a11_smoke_train
from tools.exp14.a11_full_numerics import adam_step_counts, validate_scaler_trajectory
from tools.exp14.a11_smoke_numerics import NumericalAudit, exact_tree, finite_tree
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp15 import ja, jb
from tools.exp15.regions import inspect_regions
from tools.exp16 import ce

SEEDS = (42, 142, 242)
SEAL = "exp16_j3_region_ce_v1"
SMOKE_STEPS, SMOKE_CAP, FORMAL_STEPS = 1, 5, 2100


class J3Trainer(ja.JATrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ja.FormalTrainer.get_model(self, cfg, weights, verbose)
        ce.add_head(model)
        return model


def make_trainer(release, weight, output, groups, identity, seed):
    require(type(seed) is int and seed in SEEDS, "Only the three frozen J3 seeds are permitted")
    base_protocol, base_training = ja.protocol_spec, a11_smoke_train.training_protocol

    def protocol():
        spec = base_protocol()
        return {**spec, "training": {**spec["training"], "seed": seed}}

    with patch.object(ja, "protocol_spec", protocol), patch.object(
        a11_smoke_train, "training_protocol", lambda *a: {**base_training(*a), "seed": seed}
    ):
        trainer = J3Trainer(release, weight, output, "j3", groups, identity)
    require(trainer.args.seed == trainer.formal["seed"] == seed,
            "Native constructor and epoch-loader seed differ")
    return trainer


def strict_checkpoint(path, expected_step=FORMAL_STEPS, expected_identity=None):
    """Read-only strict FP32 master/EMA/AdamW/scaler/native validation on CPU."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved["schema"] == SEAL and saved["arm"] == "j3"
            and saved["effective_updates"] == saved["ema_updates"] == expected_step
            and saved["class_names"] == dict(enumerate(CLASSES))
            and (expected_identity is None or saved["identity"] == expected_identity),
            "J3 checkpoint lineage, seed, class map or update budget differs")
    require(finite_tree(saved), "J3 checkpoint contains nonfinite state")
    for label, key in (("state", "model_state_sha256"), ("ema", "ema_state_sha256")):
        require(hash_state(saved[label]) == saved[key], "J3 checkpoint state digest differs")
        require(all(value.dtype == torch.float32 for value in saved[label].values() if value.is_floating_point()),
                "J3 checkpoint master/EMA is not FP32")
    require(saved["model_state_sha256"] != saved["ema_state_sha256"], "FP32 master and EMA unexpectedly identical")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(saved["model_config"], ch=3, nc=7, verbose=False)
        ce.add_head(model)
    for key in ("ema", "state"):
        model.load_state_dict(saved[key], strict=True)
        require(exact_tree(dict(model.state_dict()), saved[key]), "J3 strict model state replay differs")
    named = dict(model.named_parameters())
    require(len([name for name in named if name.startswith(ce.HEAD_PREFIX)]) == ce.HEAD_TENSORS,
            "J3 head tensor count differs")
    optimizer = torch.optim.AdamW([{"params": [named[n] for n in names]} for names in saved["optimizer_names"]])
    optimizer.load_state_dict(saved["optimizer"])
    scaler = torch.amp.GradScaler("cpu")
    scaler.load_state_dict(saved["scaler"])
    require(exact_tree(optimizer.state_dict(), saved["optimizer"])
            and scaler.state_dict() == saved["scaler"], "J3 strict AdamW/scaler reload differs")
    require(adam_step_counts(saved["optimizer"]["state"], expected_step) == {expected_step: 296},
            "J3 AdamW slots or step counts differ")
    validate_scaler_trajectory(saved["scaler"], expected_step)
    require(saved["scheduler"]["last_epoch"] == (expected_step - 1) // 21,
            "J3 checkpoint scheduler position differs")
    native = {name: value for name, value in saved["state"].items() if not name.startswith(ce.HEAD_PREFIX)}
    del model.region_projection
    model.load_state_dict(native, strict=True)
    require(hash_state(model.state_dict()) == saved["native_state_sha256"],
            "J3 head removal did not preserve the strict native master")
    model.names = saved["class_names"]
    model.eval()
    with torch.no_grad():
        observation, _ = model(torch.zeros(1, 3, 64, 64))
    require(finite_tree(observation), "J3 stripped native synthetic output is nonfinite")
    return {"sha256": sha256_file(path), "strict_reload": True, "stripped_native_forward": True,
            "state_sha256": saved["model_state_sha256"], "native_state_sha256": saved["native_state_sha256"],
            "ema_state_sha256": saved["ema_state_sha256"], "adam_slots": 296, "head_tensors_removed": 6}


def save_checkpoint(trainer, file):
    model = trainer.model
    step = trainer.audit.effective_updates
    require(trainer.audit.native_steps == trainer.ema.updates == step, "J3 checkpoint step count differs")
    state, ema = cpu_state(model.state_dict()), cpu_state(trainer.ema.ema.state_dict())
    record = {"schema": SEAL, "arm": "j3", "effective_updates": step, "model_config": model.yaml,
              "class_names": dict(enumerate(CLASSES)), "weight_transfer": trainer.weight_transfer,
              "identity": trainer.identity, "optimizer_names": trainer.optimizer_names,
              "native_state_sha256": hash_state(ja.native_state(model)), "model_state_sha256": hash_state(state),
              "ema_state_sha256": hash_state(ema), "ema_updates": trainer.ema.updates,
              "state": state, "ema": ema, "optimizer": cpu_state(trainer.optimizer.state_dict()),
              "scaler": trainer.scaler.state_dict(), "scheduler": trainer.scheduler.state_dict()}
    with file.open("xb") as stream:
        torch.save(record, stream)
    require(exact_tree(record, torch.load(file, map_location="cpu", weights_only=True)),
            "J3 serialized checkpoint differs from the live state")
    return strict_checkpoint(file, step, trainer.identity)


def run_arm(trainer, prior, gate, budget, lambda_ce, formal=False):
    """Exactly one joint AdamW call per native batch, with durable bounded tickets."""
    require(type(lambda_ce) in (float, int) and 0 < lambda_ce < float("inf"), "Invalid fixed CE weight")
    limit, cap = (FORMAL_STEPS, FORMAL_STEPS) if formal else (SMOKE_STEPS, SMOKE_CAP)
    require(len(prior["batch_tensor_sha256"]) >= limit, "Missing historical tensor trajectory")
    require(not (budget / "j3.attempts").exists(), "This J3 arm already started; automatic restart is forbidden")
    started = time.perf_counter()
    trainer.setup()
    trainer.audit.close()
    trainer.audit = NumericalAudit(trainer, update_limit=limit)
    require(trainer.variant == "j3" and trainer.initial_native == gate["native_initial_sha256"]
            and trainer.initial_bn == gate["initial_bn_sha256"] and len(trainer.train_loader) == 21
            and trainer._get_warmup_iterations(21) == 63 and trainer.amp and trainer.scaler.is_enabled()
            and trainer.test_loader is None, "J3 initialization, schedule or TRAIN-only scope differs")
    params = {name: value for name, value in trainer.model.named_parameters() if value.requires_grad}
    backbone = {n: p for n, p in params.items() if n.startswith("model.") and int(jb.native_layer(n)) <= ja.BACKBONE_END}
    head = {n: p for n, p in params.items() if n.startswith(ce.HEAD_PREFIX)}
    require(len(backbone) == 120 and len(head) == 6, "J3 parameter identity differs")
    native_groups = [[n for n in group if not n.startswith(ce.HEAD_PREFIX)] for group in trainer.optimizer_names]
    if "native_groups" in gate:
        require(native_groups == gate["native_groups"], "J3 native optimizer groups differ from frozen J0")
    before = hash_state(backbone), hash_state(head)
    reports, coverage, checkpoints = [], [], {}
    torch.cuda.reset_peak_memory_stats()
    for step, batch in jb.matched_batches(trainer, limit):
        digest = ja.batch_digest(batch)
        require(digest == prior["batch_tensor_sha256"][step - 1],
                f"J3 main-task tensor/order trajectory differs at step {step}")
        if "lr" in prior:
            require([g["lr"] for g in trainer.optimizer.param_groups] == prior["lr"][step - 1],
                    f"J3 native LR differs at step {step}")
        regions = inspect_regions(batch, trainer.groups)
        roi = regions.report()
        if step <= len(prior.get("per_batch_regions", [])):
            require(roi == prior["per_batch_regions"][step - 1], "J3 region exposure differs from frozen J1")
        timed = time.perf_counter()
        bn_watch = step in jb.AUDIT_STEPS or (formal and (step - 1) % 21 == 0)
        bn = (lambda: ja.bn_fingerprint(trainer.model)) if bn_watch else None
        old_bn = bn() if bn else None
        with autocast(trainer.amp, device="cuda"):
            native, parts, auxiliary = ce.forward(trainer.model, trainer.preprocess_batch(batch), regions, bn)
            trainer.loss = native + lambda_ce * auxiliary
            require(finite_tree(auxiliary), "J3 CE loss is nonfinite")
            forward_audit = trainer.audit.check_forward(trainer.loss, parts)
        if bn:
            require(bn() != old_bn, "J3 native forward did not update BatchNorm")
        trainer.scaler.scale(trainer.loss).backward()
        head_nonzero = sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in head.values())
        require(head_nonzero == 6, "J3 head gradient is missing or zero")
        ticket = jb.reserve_attempt(budget, cap, "j3")
        try:
            numeric = trainer.audit.optimizer_step()
        except Exception as failure:
            trainer.event("optimizer_stopped", step=step, ticket=ticket, **getattr(failure, "evidence", {}))
            raise
        require(trainer.audit.native_steps == trainer.audit.effective_updates == step == trainer.ema.updates,
                "J3 single native AdamW or EMA count differs")
        validate_scaler_trajectory(numeric["scaler_state"], step)
        report = {"step": step, "ticket": ticket, "batch_sha256": digest,
                  "native_loss": float(native.detach()), "ce_loss": float(auxiliary.detach()),
                  "lambda_ce": lambda_ce, "total_loss": forward_audit["loss"],
                  "global_gradient_norm": numeric["gradient_norm"],
                  "clip_coef": min(1.0, 10.0 / (numeric["gradient_norm"] + 1e-6)),
                  "projection_nonzero_gradients": head_nonzero, "bn_native_only_check": bool(bn),
                  "lr": [g["lr"] for g in trainer.optimizer.param_groups],
                  "audited_seconds": time.perf_counter() - timed}
        reports.append(report)
        coverage.append(roi)
        trainer.event("jb_update", **report, coverage=roi, optimizer=numeric)
        if step in (1, limit) or (formal and step % 105 == 0):
            require(hash_state(backbone) != before[0] and hash_state(head) != before[1],
                    "J3 backbone or auxiliary head did not update")
            name = "last.pt" if formal and step == limit else f"step_{step:04}.pt"
            checkpoints[str(step)] = save_checkpoint(trainer, trainer.output / name)
    require(len(reports) == limit and not trainer.audit.amp_skips and not trainer.counters["repair_exposures"],
            "J3 terminal budget, AMP or sample accounting differs")
    return {"status": "PASS_FORMAL_ARM" if formal else "PASS_J3_BOUNDED_SMOKE", "schema": SEAL,
            "identity": trainer.identity, "variant": "j3", "lambda_ce": lambda_ce, "runtime": trainer.runtime,
            "initial_native_sha256": trainer.initial_native, "initial_bn_sha256": trainer.initial_bn,
            "counts": dict(trainer.counters), "attempt_cap": cap, "native_groups": native_groups,
            "native_optimizer_calls": trainer.audit.native_steps, "effective_updates": trainer.audit.effective_updates,
            "optimizer_attempts": trainer.audit.optimizer_attempts, "amp_skips": trainer.audit.amp_skips,
            "warmup_iterations": 63, "post_warmup_steps": max(0, limit - 63),
            "optimizer_unique_parameters": sum(map(len, trainer.optimizer_names)), "checkpoints": checkpoints,
            "backbone_before": before[0], "backbone_after": hash_state(backbone),
            "projection_before": before[1], "projection_after": hash_state(head),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "coverage": ja.summarize_regions(coverage),
            "updates": reports, "elapsed_seconds": time.perf_counter() - started}
