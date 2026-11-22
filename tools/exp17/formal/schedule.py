"""Audited J1-compatible schedule, with a separate mathematical/code audit.

No real model, dataset, optimizer update or CUDA operation is needed by this
module. The code-reference audit executes the unchanged native schedule methods
on a tiny CPU optimizer, whose ``step`` method is never called.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np


GROUP_ORDER = ("weight", "bn", "bias")
FROZEN_J1_PROTOCOL_SHA256 = "b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca"


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ScheduleProfile:
    epochs: int = 100
    batches_per_epoch: int = 21
    train_images: int = 668
    batch: int = 32
    nbs: int = 32
    lr0: float = 0.01
    lrf: float = 0.01
    warmup_iterations: int = 63
    warmup_bias_lr: float = 0.1
    warmup_momentum: float = 0.8
    momentum: float = 0.937
    beta2: float = 0.999
    weight_decay: float = 0.0005

    def __post_init__(self):
        require(asdict(self) == asdict_frozen(), "Schedule differs from frozen J1 seed42 plan")

    @classmethod
    def from_protocol(cls, protocol):
        """Accept only complete J1-compatible schedule fields, never defaults for missing keys."""
        t = protocol.get("train", protocol.get("training"))
        require(isinstance(t, dict), "Missing training schedule")
        expected = {"epochs": 100, "batch": 32, "nbs": 32, "lr0": .01, "lrf": .01,
                    "warmup_epochs": 3., "warmup_bias_lr": .1, "warmup_momentum": .8,
                    "momentum": .937, "weight_decay": .0005, "optimizer": "AdamW", "cos_lr": False,
                    "accumulate": 1, "drop_last": False}
        require(all(k in t and t[k] == v for k, v in expected.items()), "Training fields differ from frozen J1 schedule")
        return cls()

    @property
    def updates(self):
        return self.epochs * self.batches_per_epoch

    def point(self, update):
        """Candidate floating-point schedule, one-based update, zero-based warmup index.

        Match native operation ordering (including np.interp) for bit-exact LR
        comparisons. ``mathematical_point`` independently checks exact decimal
        mathematics without reusing this implementation or np.interp.
        """
        require(type(update) is int and 1 <= update <= self.updates, "Update outside frozen budget")
        epoch0, batch0 = divmod(update - 1, self.batches_per_epoch)
        factor = max(1 - epoch0 / self.epochs, 0) * (1 - self.lrf) + self.lrf
        base = self.lr0 * factor
        warm = update - 1 < self.warmup_iterations
        groups = []
        for name in GROUP_ORDER:
            lr = base
            if warm:
                lr = float(np.interp(update - 1, [0, self.warmup_iterations],
                                     [self.warmup_bias_lr if name == "bias" else 0., base]))
            groups.append({"param_group": name, "lr": lr, "initial_lr": self.lr0,
                           "weight_decay": self.weight_decay if name == "weight" else 0.,
                           "betas": [self.momentum, self.beta2], "momentum_key_present": False})
        return {"update": update, "epoch": epoch0 + 1, "batch_index": batch0,
                "images": self.batch if batch0 < 20 else 28, "warmup": warm,
                "warmup_iteration_zero_based": update - 1 if warm else None,
                "accumulate": 1, "groups": groups,
                "scheduler": {"base_lrs": [self.lr0] * 3, "last_epoch": epoch0,
                              "_step_count": epoch0 + 2, "_is_initial": False,
                              "_get_lr_called_within_step": False, "_last_lr": [base] * 3,
                              "lr_lambdas": [None] * 3}}

    def mathematical_point(self, update):
        """Independent decimal derivation; checks math, not C/Python rounding order."""
        require(type(update) is int and 1 <= update <= 2100, "Invalid mathematical update")
        epoch0 = (update - 1) // 21
        with localcontext() as context:
            context.prec = 60
            base = Decimal("0.01") * (Decimal(1) - Decimal("0.99") * Decimal(epoch0) / Decimal(100))
            if update <= 63:
                t = Decimal(update - 1) / Decimal(63)
                weight = base * t
                bias = Decimal("0.1") * (Decimal(1) - t) + base * t
            else:
                weight = bias = base
            return {"lr_weight": float(weight), "lr_bn": float(weight), "lr_bias": float(bias),
                    "beta1": .937, "beta2": .999, "weight_decay": [.0005, 0., 0.]}


def asdict_frozen():
    # Separate literal avoids constructing a second dataclass in __post_init__.
    return {"epochs": 100, "batches_per_epoch": 21, "train_images": 668, "batch": 32, "nbs": 32,
            "lr0": .01, "lrf": .01, "warmup_iterations": 63, "warmup_bias_lr": .1,
            "warmup_momentum": .8, "momentum": .937, "beta2": .999, "weight_decay": .0005}


def verify_optimizer_point(optimizer, scheduler, update, profile=None):
    """Hard-gate actual native groups and complete native LambdaLR state pre-update."""
    expected = (profile or ScheduleProfile()).point(update)
    actual = optimizer.param_groups
    require(len(actual) == 3, "Native AdamW parameter group count differs")
    for group, ref in zip(actual, expected["groups"]):
        for field in ("param_group", "lr", "initial_lr", "weight_decay"):
            require(field in group and group[field] == ref[field], f"Native schedule {field} differs at update {update}")
        require(list(group.get("betas", [])) == ref["betas"] and "momentum" not in group,
                "AdamW beta or momentum-key behavior differs")
        require(group.get("eps") == 1e-8 and group.get("amsgrad") is False,
                "Native AdamW numerical settings differ")
    require(scheduler.state_dict() == expected["scheduler"], f"Scheduler state differs at update {update}")
    return expected


def _read_warmup_method(repo):
    """Compile the exact unchanged function AST; do not instantiate its GPU trainer."""
    source = repo / "tools/exp14/a11_smoke_train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BoundedSmokeTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "warmup")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["warmup"]


def native_code_trace(repo):
    """Run real native optimizer grouping/scheduler/warmup on CPU, zero optimizer steps."""
    import torch
    import ultralytics
    from ultralytics.engine.trainer import BaseTrainer
    require(ultralytics.__version__ == "8.4.117", "Native source version differs")
    warmup = _read_warmup_method(Path(repo))
    args = SimpleNamespace(cos_lr=False, lrf=.01, warmup_epochs=3., warmup_bias_lr=.1,
                           warmup_momentum=.8, momentum=.937)
    trainer = SimpleNamespace(args=args, epochs=100, epoch=0, data={"nc": 7})
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    trainer.optimizer = BaseTrainer.build_optimizer(trainer, model, name="AdamW", lr=.01, momentum=.937, decay=.0005)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Schedule audit must never call an optimizer update")
    trainer.optimizer.register_step_pre_hook(forbidden)
    BaseTrainer._setup_scheduler(trainer)
    require(BaseTrainer._get_warmup_iterations(trainer, 21) == 63, "Native warmup iterations differ")
    trainer.scheduler.last_epoch = -1  # exact _setup_train assignment before matched_batches
    rows = []
    for epoch in range(100):
        trainer.epoch = epoch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer.scheduler.step()  # matched_batches: scheduler before new epoch loader
        for batch in range(21):
            iteration = epoch * 21 + batch
            warmup(trainer, iteration, 63)
            groups = [{"param_group": g["param_group"], "lr": g["lr"], "initial_lr": g["initial_lr"],
                       "weight_decay": g["weight_decay"], "betas": list(g["betas"]),
                       "momentum_key_present": "momentum" in g} for g in trainer.optimizer.param_groups]
            rows.append({"groups": groups, "scheduler": trainer.scheduler.state_dict()})
    return rows, {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                  "native_trainer_file_sha256": _sha(Path(inspect.getfile(BaseTrainer))),
                  "native_scheduler_method": "BaseTrainer._setup_scheduler",
                  "native_optimizer_method": "BaseTrainer.build_optimizer",
                  "native_warmup_method": "BoundedSmokeTrainer.warmup",
                  "optimizer_steps": 0, "model_forwards": 0, "CUDA_calls": 0}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(repo, output):
    """Generate all 2100 immutable rows after code/math/history consistency checks."""
    repo, output = Path(repo), Path(output)
    canonical = (repo / "results/exp15_jb/formal_protocol.json").read_bytes().replace(b"\r\n", b"\n")
    require(hashlib.sha256(canonical).hexdigest() == FROZEN_J1_PROTOCOL_SHA256, "J1 canonical protocol SHA differs")
    frozen_protocol = json.loads(canonical)
    profile = ScheduleProfile.from_protocol(frozen_protocol)
    inherited_sources = {}
    for name, expected_sha in frozen_protocol["identity"]["files"].items():
        path = repo / "tools/exp15" / name
        content = path.read_bytes()
        canonical_sha = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
        require(canonical_sha == expected_sha, "Inherited J1 source differs from frozen protocol")
        inherited_sources[f"tools/exp15/{name}"] = {"canonical_LF_sha256": canonical_sha,
                                                    "working_bytes_sha256": _sha(path)}
    for name in ("a11_smoke_train.py", "a11_full_train.py", "a11_full_protocol.py"):
        path = repo / "tools/exp14" / name
        inherited_sources[f"tools/exp14/{name}"] = {"working_bytes_sha256": _sha(path),
            "canonical_LF_sha256": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    ledger = repo / "results/exp15_jc_formal_seed42/matched_batch_ledger.csv"
    with ledger.open(encoding="utf-8", newline="") as f:
        historical = list(csv.DictReader(f))
    require(len(historical) == 2100, "Frozen actual J1 ledger is incomplete")
    native, source = native_code_trace(repo)
    math_error, ledger_error = 0., 0.
    all_rows = []
    for update in range(1, 2101):
        point, math_point = profile.point(update), profile.mathematical_point(update)
        ref = native[update - 1]
        require(point["groups"] == ref["groups"] and point["scheduler"] == ref["scheduler"],
                f"J2 candidate/native J1 code mismatch at update {update}")
        old = historical[update - 1]
        require(int(old["step"]) == update and int(old["epoch"]) == point["epoch"]
                and int(old["images"]) == point["images"], "Actual J1 batch/epoch/tail differs")
        group_lrs = [g["lr"] for g in point["groups"]]
        old_lrs = [float(old[k]) for k in ("lr_no_decay", "lr_decay", "lr_bias")]
        error = max(abs(a-b) for a,b in zip(group_lrs, old_lrs));ledger_error = max(ledger_error, error)
        require(group_lrs == old_lrs, f"Actual J1 ledger LR differs at update {update}")
        m_error = max(abs(g-math_point[k]) for g,k in zip(group_lrs, ("lr_weight", "lr_bn", "lr_bias")))
        math_error = max(math_error, m_error)
        require(m_error < 2e-16, "Independent decimal mathematics differs")
        row = {"update": update, "epoch": point["epoch"], "batch_index_zero_based": point["batch_index"],
               "images": point["images"], "warmup": point["warmup"],
               "warmup_iteration_zero_based": point["warmup_iteration_zero_based"],
               "lr_weight": group_lrs[0], "lr_bn": group_lrs[1], "lr_bias": group_lrs[2],
               "initial_lr": .01, "beta1_weight": .937, "beta1_bn": .937, "beta1_bias": .937,
               "beta2": .999, "SGD_momentum_key_present": False,
               "configured_warmup_momentum": .8, "weight_decay_weight": .0005,
               "weight_decay_bn": 0., "weight_decay_bias": 0., "accumulate": 1,
               "scheduler_last_epoch": ref["scheduler"]["last_epoch"],
               "scheduler_step_count": ref["scheduler"]["_step_count"],
               "scheduler_last_lr": ref["scheduler"]["_last_lr"][0],
               "scheduler_state_sha256": hashlib.sha256(json.dumps(ref["scheduler"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "native_code_exact": True, "historical_J1_exact": True, "independent_math_abs_error": m_error}
        all_rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "lr_schedule_2100.csv"
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    summary = {"schema": "exp17g_f1_schedule_audit_v1", "status": "PASS_EXACT_NATIVE_J1_2100_UPDATE_SCHEDULE",
               "profile": asdict(profile), "updates": 2100, "groups": list(GROUP_ORDER),
               "native_J1_vs_J2_candidate_bit_exact_rows": 2100, "historical_J1_bit_exact_rows": 2100,
               "historical_J1_max_lr_absolute_error": ledger_error,
               "independent_decimal_math_max_absolute_error": math_error,
               "independent_decimal_math_tolerance": 2e-16,
               "independent_math_note": "Decimal derivation verifies real arithmetic; native C/Python binary operation order may differ by <=1 ULP. Code-vs-code and actual historical LR comparisons are bit exact.",
               "warmup": {"iterations": 63, "condition": "iteration < 63", "zero_based_positions": [0, 62],
                          "one_based_updates": [1, 63], "positions_51_through_63_checked": 13,
                          "first_non_warmup_update": 64, "last_warmup_bias_lr": all_rows[62]["lr_bias"],
                          "first_non_warmup_lr": all_rows[63]["lr_weight"],
                          "no_GPU_extension_of_exhausted_smoke": True},
               "lr_endpoints": {"first_update": [all_rows[0][k] for k in ("lr_weight", "lr_bn", "lr_bias")],
                                "final_epoch100_update": [all_rows[-1][k] for k in ("lr_weight", "lr_bn", "lr_bias")],
                                "theoretical_lambda_at_epoch_index100_unused": .0001},
               "AdamW": {"betas": [.937, .999], "weight_decay_by_group": [.0005, 0., 0.],
                         "weight_decay_scaling": ".0005 * batch32 * accumulate1 / nbs32 = .0005",
                         "warmup_momentum_is_inactive": True,
                         "reason": "Native warmup updates only a literal momentum key. AdamW groups contain betas, no momentum key; beta1 stays .937, never .8.",
                         "epsilon": 1e-8, "amsgrad": False},
               "batch_tail": {"full_batches_per_epoch": 20, "tail_images": 28, "drop_last": False,
                              "epochs_verified": 100, "total_image_exposures": sum(r["images"] for r in all_rows)},
               "call_order": ["construction: LambdaLR initial step", "_setup_train sets scheduler.last_epoch=-1",
                              "epoch start: model.train then scheduler.step", "new_epoch_loader",
                              "before each update: warmup(global_zero_based_iteration,63)",
                              "native AdamW step/GradScaler/EMA once; no scheduler.step per batch"],
               "historical_column_naming_note": "Historical columns lr_no_decay/lr_decay have equal LR throughout. Authoritative native optimizer order is weight(decay), bn(no_decay), bias(no_decay); historical LR column names do not prove weight-decay assignment.",
               "code_reference": source, "source_sha256": _sha(Path(__file__)),
               "inherited_source_identity": inherited_sources,
               "source_locations": {"warmup": "tools/exp14/a11_smoke_train.py:193",
                                    "matched_batches": "tools/exp15/jb.py:176",
                                    "epoch_seeded_loader": "tools/exp14/a11_full_train.py:163",
                                    "scheduler": "ultralytics/engine/trainer.py:245",
                                    "optimizer": "ultralytics/engine/trainer.py:1093"},
               "J1_protocol_canonical_sha256": FROZEN_J1_PROTOCOL_SHA256,
               "historical_J1_ledger_raw_sha256": _sha(ledger), "trajectory_csv_sha256": _sha(path),
               "metadata_only": True, "formal_training_authorized": False,
               "real_data_reads": 0, "model_forwards": 0, "optimizer_updates": 0, "GPU_calls": 0}
    with (output / "lr_schedule_audit.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False);f.write("\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.repository, args.output)
    print(json.dumps({k:report[k] for k in ("status", "updates", "native_J1_vs_J2_candidate_bit_exact_rows", "independent_decimal_math_max_absolute_error")}))


if __name__ == "__main__":
    main()
