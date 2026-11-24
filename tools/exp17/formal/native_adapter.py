"""Future authorized J2 run: unchanged native J1 A stream plus frozen Pair4.

This module is an adapter, never an authorization bypass or a standalone entry.
The common executor must validate the independent Gate and permanent run claim
BEFORE constructing NativeAdapter. G1 only imports/inspects this code; it never
constructs a native trainer, reads a VAL image or performs a real update.
"""
from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path

import torch
import ultralytics
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a11_full_numerics import validate_scaler_trajectory
from tools.exp14.a11_smoke_numerics import NumericalAudit, finite_tree, optimizer_group_names
from tools.exp14.formal_e0_e1 import hash_state
from tools.exp15.ja import JATrainer, checked_train
from tools.exp17.engineering.augmentation import IndexIdentityResolver, install_native_replay
from tools.exp17.engineering.bn import bn_modules, buffer_snapshot, buffers_equal, rng_equal, rng_snapshot
from tools.exp17.engineering.coverage import compare_reference, native_evidence, read_reference
from tools.exp17.engineering.data import TrustedPairIndex, load, require, sha
from tools.exp17.engineering.model import export_native_model, forward_joint
from tools.exp17.engineering.runtime import (
    INITIAL_NATIVE, INITIAL_PROJECTION, DecodedAssetCache, prepare_batch, runtime_asset_map,
)
from tools.exp17.formal.diagnostics import fixed_gradient_diagnostic
from tools.exp17.formal.sampler import SealedEpochCache
from tools.exp17.formal.schedule import ScheduleProfile, verify_optimizer_point
from tools.exp17.formal.validation import FrozenValAdapter, due, run_isolated


# Immutable original Engineering fixed CPU batches, never result-dependent.
FIXED_BATCH_SHA256 = {
    1: "802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929",
    7: "07cff77e68290e0281cc6f8ac49a54dde0dbb25600c5c4932a0d5ba7b712ff76",
    11: "5395b0c3d0d7e0d22d3eb546f78e02fab29c9298aa886977328b140af6ff3f08",
    21: "c93789fb512af377a6d0793f3199126cd23e89ab1844fd08ecf30654e85aa90a",
}


def _parameter_identity(model, projection=False):
    return hash_state({n: p for n, p in model.named_parameters() if p.requires_grad and
        (n.startswith("region_projection.") if projection else
         n.startswith("model.") and int(n.split(".")[1]) <= 10)})


class _AuditState:
    """Expose non-model training observers to the callback isolation guard."""
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        trainer = self.trainer
        value = {"counters": copy.deepcopy(dict(trainer.counters)), "epoch": trainer.epoch,
                 "audit": {k: getattr(trainer.audit, k) for k in
                           ("optimizer_attempts", "native_steps", "effective_updates", "amp_skips")}}
        foreground = getattr(trainer, "foreground", None)
        value["foreground_counts"] = None if foreground is None or foreground.counts is None else foreground.counts.detach().clone()
        return value

    def load_state_dict(self, value):
        trainer = self.trainer
        trainer.counters.clear(); trainer.counters.update(value["counters"])
        trainer.epoch = value["epoch"]
        for key, item in value["audit"].items(): setattr(trainer.audit, key, item)
        if getattr(trainer, "foreground", None) is not None:
            trainer.foreground.counts = value["foreground_counts"]


class NativeJ2Trainer(JATrainer):
    """Independent 2100-budget wrapper; historical engineering cap is untouched."""
    def train(self, *_args, **_kwargs):
        raise RuntimeError("Only the separately authorized common J2 executor may drive training")

    def run_formal(self, *_args, **_kwargs):
        raise RuntimeError("Use the separate protocol- and journal-bound J2 executor")


class NativeAdapter:
    bn_mode = "replay_a"
    validator_class = FrozenValAdapter

    def __init__(self, config, protocol, output, bindings):
        self.config, self.protocol, self.bindings = config, protocol, copy.deepcopy(bindings)
        self.output = Path(output)
        self.trainer, self.sampler, self.validator = None, None, None
        self.profile = ScheduleProfile.from_protocol(protocol)
        framework_root = Path(ultralytics.__file__).parent
        require(all(sha(framework_root / name) == expected for name, expected in
                    protocol["framework_sources_sha256"].items()), "Installed frozen framework source changed")
        expected_method = {"lambda_supcon": 8, "temperature": .1, "lambda_pair": 4, "margin": .2,
                           "bn_mode": self.bn_mode, "checkpoint_prefix": False, "N_prefix": [0, 16]}
        require(protocol["method"] == expected_method, "Frozen J2 method changed")
        require(bindings["seed"] == 42 and bindings["epochs"] == 100 and bindings["effective_updates"] == 2100,
                "Native adapter budget/seed differs from the authorized bindings")
        supplied = config["val_authorization"]
        val_authorization = load(supplied) if isinstance(supplied, (str, Path)) else supplied
        # Metadata/path/scope validation only. Discover an invalid future VAL
        # scope before constructing a model or consuming a training update.
        self.validator = self.validator_class(authorization=val_authorization,
            protocol_sha256=bindings["protocol_sha256"], val_copy=config["val_copy"],
            val_root=config["val_root"], output=config["val_output"], ledger_root=config["val_ledger_root"])
        require(sha(config["manifest"]) == bindings["manifest_sha256"] == config["manifest_sha256"]
                and sha(config["native_identity"]) == bindings["native_identity_sha256"]
                and sha(config["weight"]) == bindings["coco_sha256"], "Frozen native data/model identity mismatch")
        release = Path(config["release"]).resolve(strict=True)
        self.index = TrustedPairIndex.from_files(config["manifest"], config["native_identity"],
            expected_manifest_sha256=bindings["manifest_sha256"], runtime_assets=runtime_asset_map(config),
            native_dataset_root=release / "dataset")
        self.groups = checked_train(release)
        require(sha(config["baseline_ledger"]) == protocol["identity"]["native_batch_ledger_sha256"],
                "Historical 2100-batch reference ledger identity changed")
        self.references = read_reference(config["baseline_ledger"])
        self.fixed_batches = {int(r["batch_position"]): r for r in config["fixed_batches"]}
        require({n: r["sha256"] for n, r in self.fixed_batches.items()} == FIXED_BATCH_SHA256,
                "Frozen diagnostic batch identity or position changed")
        for spec in self.fixed_batches.values():
            require(sha(spec["path"]) == spec["sha256"], "Frozen diagnostic CPU batch changed")
        # A recovery reconstructs the architecture/optimizer before strictly
        # restoring its full state. Preserve every prior setup cache/log, and
        # never mistake a new initialization folder for a new experiment claim.
        initializations = self.output / "native_initializations"
        require(not initializations.is_symlink(), "Native initialization root is a symlink")
        initializations.mkdir(parents=True, exist_ok=True)
        number = 0
        while True:
            native_output = initializations / f"init_{number:04d}"
            if native_output.exists():
                require(not native_output.is_symlink() and native_output.is_dir(),
                        "Existing native initialization path changed type")
                number += 1
                continue
            try:
                native_output.mkdir(exist_ok=False)
                break
            except FileExistsError:
                # The common executor owns a process lease, but retain an
                # exclusive directory claim instead of relying on that alone.
                number += 1
        try:
            trainer = NativeJ2Trainer(release, Path(config["weight"]).resolve(strict=True), native_output,
                                      "j1", self.groups, {"stage": "EXP17_G1_FUTURE_FORMAL_NATIVE_ADAPTER", **bindings})
            self.trainer = trainer
            trainer.setup()
            require(trainer.runtime == protocol["environment"], "Formal environment differs from the frozen protocol")
            audited = protocol["training"]
            names = ("seed","epochs","batch","nbs","imgsz","optimizer","lr0","lrf","cos_lr",
                     "warmup_epochs","warmup_bias_lr","warmup_momentum","momentum","weight_decay",
                     "amp","workers","deterministic","compile","patience")
            require(all(getattr(trainer.args, key) == audited[key] for key in names)
                    and all(getattr(trainer.args, key) == value for key,value in audited["augmentation"].items()),
                    "Actual native training arguments differ from the audited candidate protocol")
            trainer.audit.close()
            trainer.audit = NumericalAudit(trainer, update_limit=2100)
            trainer.epoch = 0
            projection = _parameter_identity(trainer.model, projection=True)
            require(trainer.initial_native == INITIAL_NATIVE == protocol["identity"]["initial_native_sha256"]
                    and projection == INITIAL_PROJECTION == protocol["identity"]["initial_projection_sha256"],
                    "Fresh COCO seed42 native/projection initialization differs")
            require(trainer.batch_size == 32 and trainer.args.imgsz == 640 and trainer.args.seed == 42
                    and len(trainer.train_loader) == 21 and trainer._get_warmup_iterations(21) == 63
                    and trainer.accumulate == 1 and trainer.amp and not trainer.optimizer.state,
                    "Native J1 initialization, schedule or numerical path changed")
            self.optimizer_names = copy.deepcopy(optimizer_group_names(trainer.model, trainer.optimizer))
            require(self.optimizer_names == trainer.optimizer_names, "Native initial AdamW grouping differs")
            install_native_replay(trainer.train_loader.dataset, IndexIdentityResolver(self.index))
            self.asset_cache = DecodedAssetCache()
            self.audit_state = _AuditState(trainer)
        except BaseException:
            self.close()
            raise

    def start_epoch(self, epoch):
        trainer = self.trainer
        require(type(epoch) is int and 0 <= epoch < 100
                and trainer.audit.effective_updates == epoch * 21,
                "Native epoch start is not the exact completed-update boundary")
        require(trainer.scheduler.last_epoch == epoch - 1, "Epoch scheduler would be repeated or skipped")
        trainer.epoch = epoch
        trainer._model_train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer.scheduler.step()

    def build_cache(self, epoch, directory):
        """Drain the exact original two-worker stream, with no warmup/optimizer use."""
        trainer = self.trainer
        require(trainer.epoch == epoch and trainer.scheduler.last_epoch == epoch,
                "Cache must follow exactly one native epoch-start scheduling call")
        require(trainer.audit.effective_updates == epoch * 21, "Epoch cache built after an unexpected update")
        def fingerprint(batch_index, batch):
            step = epoch * 21 + batch_index + 1
            actual = native_evidence(trainer, batch, step)
            # LR is intentionally not advanced during prefetch. All six native
            # tensors, membership/order and instance exposure must already match.
            fields = ("step", "epoch", "images", "raw_targets", "augmented_targets",
                      "ordered_sources_sha256", "augmented_tensors_sha256")
            require(all(actual[k] == self.references[step - 1][k] for k in fields),
                    "Sealed prefetch changed the original J1 batch trajectory")
            return {key: actual[key] for key in fields}
        return SealedEpochCache.build(directory,
            trainer.new_epoch_loader(trainer.train_loader.dataset, epoch), epoch=epoch,
            bindings=self.bindings, batch_fingerprint=fingerprint, expected_batches=21)

    def set_sampler(self, cursor):
        """Set a new or restored sealed cursor without rerunning a scheduler call."""
        state = cursor.state_dict()
        require(state["epoch"] == self.trainer.scheduler.last_epoch,
                "Sealed cursor epoch and recovered scheduler disagree")
        self.trainer.epoch = state["epoch"]
        self.sampler = cursor

    def train_batch(self, batch, step):
        trainer = self.trainer
        require(type(step) is int and 1 <= step <= 2100
                and trainer.audit.effective_updates == step - 1
                and trainer.epoch == (step - 1) // 21, "Native update order or effective count differs")
        trainer.warmup(step - 1, 63)
        schedule = verify_optimizer_point(trainer.optimizer, trainer.scheduler, step, self.profile)
        original = native_evidence(trainer, batch, step)
        compare_reference(original, self.references[step - 1], trainer.formal)
        trainer.audit_batch(batch)  # count logical consumption only, never prefetch
        require(trainer.counters["batches_read"] == step, "Native batch consumption counter differs")
        replay_rng = rng_snapshot()
        prepared = prepare_batch(batch, self.index, self.groups, decode_N=True,
                                 device=trainer.device, asset_cache=self.asset_cache)
        require(rng_equal(replay_rng), "Auxiliary replay consumed a main-task RNG stream")
        self.check_prepared(prepared, original, step)
        trainer.optimizer.zero_grad(set_to_none=True)
        before_bn = {name: int(module.num_batches_tracked) for name, module in bn_modules(trainer.model).items()}
        before_backbone = _parameter_identity(trainer.model)
        before_projection = _parameter_identity(trainer.model, projection=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.counters["forwards"] += 1
        with autocast(trainer.amp, device="cuda"):
            require(torch.is_autocast_enabled("cuda"), "Native CUDA AMP is not enabled")
            result = forward_joint(trainer.model, prepared.batch, prepared.a_regions, prepared.normal_images,
                prepared.normal_masks, prepared.normal_regions, prepared.bindings,
                bn_mode=self.bn_mode, lambda_pair=4., checkpoint_prefix=False)
        require(finite_tree((result.total, result.yolo, result.supcon, result.pair, result.parts)),
                "Nonfinite native/SupCon/Pair/total loss")
        trainer.loss = result.total
        trainer.audit.check_forward(result.total, {"yolo": result.yolo, "supcon": result.supcon,
                                                   "pair": result.pair, "total": result.total})
        trainer.counters["finite_forwards"] += 1
        after_forward_bn = buffer_snapshot(trainer.model)
        if result.n_p3 is not None: result.n_p3.retain_grad()
        trainer.counters["backwards"] += 1
        trainer.scaler.scale(result.total).backward()
        trainer.counters["completed_backwards"] += 1
        require(buffers_equal(trainer.model, after_forward_bn), "Auxiliary backward changed native BN buffers")
        n_norm = 0.
        if result.n_p3 is not None:
            require(result.n_p3.grad is not None and bool(torch.isfinite(result.n_p3.grad).all()),
                    "Sparse N lost its finite shared-Backbone gradient path")
            n_norm = float(result.n_p3.grad.float().norm()) / trainer.scaler.get_scale()
            require(float(result.pair.detach()) == 0 or n_norm > 0, "Active Pair hinge has no N gradient")
        projection = [p for n, p in trainer.model.named_parameters() if n.startswith("region_projection.")]
        projection_nonzero = sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in projection)
        require(len(projection) == projection_nonzero == 4, "Shared projection gradient missing or zero")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in trainer.model.parameters() if p.requires_grad),
                "A trainable master gradient is missing or nonfinite")
        numerical = trainer.audit.optimizer_step()  # native clip10/scaler/AdamW/zero_grad/EMA, once
        validate_scaler_trajectory(numerical["scaler_state"], step)
        after_bn = {name: int(module.num_batches_tracked) for name, module in bn_modules(trainer.model).items()}
        require(all(after_bn[name] - before_bn[name] == 1 for name in result.bn_statistics),
                "A BN must advance once; N must add zero running-stat updates")
        require(_parameter_identity(trainer.model) != before_backbone
                and _parameter_identity(trainer.model, projection=True) != before_projection,
                "Backbone or projection parameters did not really update")
        require(trainer.audit.optimizer_attempts == trainer.audit.native_steps == trainer.audit.effective_updates
                == trainer.ema.updates == step and trainer.audit.amp_skips == 0, "Native optimizer/EMA budget mismatch or AMP skip")
        require(optimizer_group_names(trainer.model, trainer.optimizer) == self.optimizer_names,
                "Native AdamW parameter names or groups changed")
        if step % 21 == 0:
            trainer.completed_epochs = trainer.epoch + 1
            trainer.counters["complete_epochs"] = trainer.completed_epochs
        by_class = {str(c): 0 for c in range(7)}
        a_groups = {str(c): set() for c in range(7)}
        for binding in prepared.bindings:
            by_class[str(binding.class_id)] += 1
            a_groups[str(binding.class_id)].add(binding.group)
        per_class_loss = {str(c): float(value.detach()) for c, value in result.per_class_pair_tensors.items()}
        torch.cuda.synchronize()
        norm = numerical["gradient_norm"]
        record = {"step": step, "epoch": trainer.epoch + 1, "batch_index": (step - 1) % 21,
            "images": original["images"], "raw_targets": original["raw_targets"],
            "augmented_targets": original["augmented_targets"],
            "native_batch_fingerprint": original, "yolo_loss": float(result.yolo.detach()),
            "supcon_loss": float(result.supcon.detach()), "pair_loss": float(result.pair.detach()),
            "total_loss": float(result.total.detach()), "lambda_pair": 4., "gradient_norm": norm,
            "clip_coefficient": min(1., 10. / (norm + 1e-6)), "amp_scale": numerical["scale_after"],
            "optimizer_attempts": trainer.audit.optimizer_attempts, "effective_updates": step,
            "amp_skipped": numerical["amp_skipped"], "lr": [g["lr"] for g in trainer.optimizer.param_groups],
            "scheduler": schedule["scheduler"], "BN_advances_per_A": 1, "N_extra_BN_updates": 0,
            "bn_state_sha256": hash_state({n: v for n, v in trainer.model.named_buffers()
                if n.endswith(("running_mean", "running_var", "num_batches_tracked"))}),
            "N_P3_unscaled_gradient_norm": n_norm, "native_RNG_unchanged_by_N": True,
            "projection_nonzero_gradients": projection_nonzero,
            "parameter_state_sha256": hash_state(dict(trainer.model.named_parameters())),
            "effective_A": prepared.coverage["admitted_A"],
            "effective_B": prepared.coverage["distinct_B_region_rows"],
            "effective_N": prepared.coverage["N_image_slots"],
            "effective_ABN": prepared.coverage["actual_valid_ABN"],
            "pair_by_class": by_class, "A_groups_by_class": {c: sorted(v) for c, v in a_groups.items()},
            "pair_loss_by_class": per_class_loss, "pair_skips": prepared.coverage["skip"],
            "coverage": prepared.coverage, "pair": result.report, "numeric_audit": numerical,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "seconds": time.perf_counter() - started,
            "explicit_augmentation_traces": [t.compact_record() for t in prepared.trace]}
        require(finite_tree(record), "Nonfinite per-update monitoring record")
        # Native loss tensor is not a recovery state; release the completed graph.
        trainer.loss = result.total.detach()
        del result, prepared
        return record

    def _protected(self):
        trainer = self.trainer
        return {"model": trainer.model, "ema": trainer.ema, "optimizer": trainer.optimizer,
                "scheduler": trainer.scheduler, "scaler": trainer.scaler, "sampler": self.sampler,
                "extras": {"training_observers": self.audit_state}}

    def check_prepared(self, prepared, original, step):
        """Versioned experiments can enforce additional frozen input evidence."""

    def gradient_diagnostics(self):
        trainer = self.trainer
        require(trainer.audit.effective_updates % 105 == 0 and 0 < trainer.audit.effective_updates <= 2100,
                "Gradient diagnostics are fixed to complete five-epoch boundaries")
        reports = []
        for position, spec in sorted(self.fixed_batches.items()):
            require(sha(spec["path"]) == FIXED_BATCH_SHA256[position], "Frozen diagnostic batch changed")
            def diagnostic(clone):
                # deepcopy retains Python closures in the original foreground
                # observer. Rebind the same native criterion on the clone before
                # any forward, rather than touching that live observer/criterion.
                clone.criterion = clone.init_criterion()
                def batch_factory(_clone):
                    packet = torch.load(spec["path"], map_location="cpu", weights_only=False)
                    require(packet["schema"] == "EXP17_J2_NATIVE_CPU_BATCH_CACHE_V1"
                            and packet["epoch"] == 0 and packet["batch_position"] == position,
                            "Frozen diagnostic batch metadata changed")
                    return prepare_batch(packet["batch"], self.index, self.groups, decode_N=True,
                        device=trainer.device, asset_cache=DecodedAssetCache(max_bytes=0))
                return fixed_gradient_diagnostic(clone, batch_factory, amp=trainer.amp,
                                                 gradient_scale=trainer.scaler.get_scale(), bn_mode=self.bn_mode)
            report = run_isolated(diagnostic, trainer.model, allow_grad=True, **self._protected())
            reports.append({"batch_position": position, "batch_sha256": spec["sha256"], **report})
        return reports

    def evaluate(self, epoch, role):
        trainer = self.trainer
        require(role in ("master", "ema") and due(epoch, trainer.audit.effective_updates),
                "VAL role/cadence does not match completed training")
        if self.validator is None:
            supplied = self.config["val_authorization"]
            authorization = load(supplied) if isinstance(supplied, (str, Path)) else supplied
            self.validator = self.validator_class(authorization=authorization,
                protocol_sha256=self.bindings["protocol_sha256"], val_copy=self.config["val_copy"],
                val_root=self.config["val_root"], output=self.config["val_output"],
                ledger_root=self.config["val_ledger_root"])
        source = trainer.model if role == "master" else trainer.ema.ema
        def callback(clone):
            native = export_native_model(clone)
            native.names = copy.deepcopy(clone.names)
            native.to(trainer.device).float().eval()
            return self.validator.evaluate(native, epoch=epoch, role=role)
        return run_isolated(callback, source, **self._protected())

    def close(self):
        if self.trainer is not None:
            self.trainer.close()
