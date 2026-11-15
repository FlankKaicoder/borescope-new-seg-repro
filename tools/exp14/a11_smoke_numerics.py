"""Audit a bounded AdamW/AMP segmentation update and strictly reload its checkpoint."""

import json
import math
from collections import Counter, defaultdict

import torch

from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.diagnose_amp_first_batch import parameters_sha
from tools.exp14.formal_e0_e1 import cpu_state, hash_state, layer_gradient_statistics
from tools.exp14.label_plan import CLASSES
from tools.exp14.revised_amp_smoke import NATIVE_DEFAULTS, clip_native_gradients, parameter_vector

CHECKPOINT_SCHEMA = "exp14_a11_1_bounded_smoke_v1"
ZERO_DETAIL_AT = 0.01
MAX_UPDATES = 50


class NumericalFailure(ValueError):
    """Stop the arm and preserve the measured failure in its event log."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def gradient_report(model):
    parameters = list(model.parameters())
    values = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    report = {"tensors": len(values), "missing_trainable_tensors": sum(p.requires_grad and p.grad is None for p in parameters),
              "elements": 0, "nonfinite": 0, "zero": 0, "zero_fraction": None, "master_dtype_fp32": False}
    if not values:
        return report
    flat = torch.cat(values)
    report.update(elements=flat.numel(), nonfinite=int((~torch.isfinite(flat)).sum()),
                  zero=flat.numel() - int(flat.count_nonzero()), master_dtype_fp32=all(g.dtype == torch.float32 for g in values))
    report["zero_fraction"] = report["zero"] / report["elements"]
    return report


def finite_tree(value):
    """Check recursively stored tensors, gathering checks per device and dtype."""
    grouped = defaultdict(list)

    def walk(item):
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                x = item.detach().reshape(-1)
                grouped[x.device, x.dtype].append(x)
        elif isinstance(item, dict):
            return all(walk(x) for x in item.values())
        elif isinstance(item, (list, tuple)):
            return all(walk(x) for x in item)
        elif isinstance(item, float):
            return math.isfinite(item)
        return True

    return walk(value) and all(bool(torch.isfinite(torch.cat(v)).all()) for v in grouped.values())


def model_fp32_finite(model):
    return all(p.dtype == torch.float32 for p in model.parameters()) and finite_tree(model.state_dict())


def finite_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


class NumericalAudit:
    def __init__(self, trainer, update_limit=MAX_UPDATES):
        require(type(update_limit) is int and update_limit > 0, "Invalid audited update limit")
        self.trainer = trainer
        self.model, self.optimizer, self.scaler = trainer.model, trainer.optimizer, trainer.scaler
        self.update_limit = update_limit
        state = self.scaler.state_dict()
        require(self.scaler.is_enabled() and state == {"scale": 1.0, "_growth_tracker": 0, **NATIVE_DEFAULTS},
                "Smoke requires enabled native GradScaler with initial scale 1 and standard growth/backoff")
        require(type(self.optimizer) is torch.optim.AdamW
                and not getattr(self.optimizer, "_step_supports_amp_scaling", False), "Expected ordinary native AdamW")
        require(model_fp32_finite(self.model), "Initial FP32 model state is non-finite or has a different dtype")
        self.initial_state_sha256 = hash_state(self.model.state_dict())
        self.initial_parameters_sha256 = parameters_sha(self.model)
        self.optimizer_attempts = 0
        self.native_steps = 0
        self.amp_skips = 0
        self.effective_updates = 0
        self.step_hook = self.optimizer.register_step_pre_hook(self.observe_step)

    def close(self):
        self.step_hook.remove()

    def observe_step(self, optimizer, args, kwargs):
        self.native_steps += 1

    def check_forward(self, loss, parts):
        public_parts = ({k: finite_number(v.detach()) for k, v in parts.items()} if isinstance(parts, dict)
                        else [finite_number(x) for x in parts.detach().reshape(-1)])
        values = public_parts.values() if isinstance(public_parts, dict) else public_parts
        report = {"loss": finite_number(loss.detach()), "loss_items": public_parts}
        if report["loss"] is None or not values or None in values:
            raise NumericalFailure("Non-finite loss or loss component", report)
        foreground = getattr(self.trainer, "foreground", None)
        if foreground is not None:
            report["foreground"] = foreground.read()
        return report

    def inspect_backward(self):
        report = {"scale_before": self.scaler.get_scale(), "optimizer_attempted": False,
                  "native_step_called": False, "parameter_changed": False, "amp_skipped": False,
                  "scaled_gradients": gradient_report(self.model)}
        if not report["scaled_gradients"]["master_dtype_fp32"]:
            raise NumericalFailure("Scaled gradients missing or master gradient dtype is not FP32", report)
        self.scaler.unscale_(self.optimizer)
        checks = self.scaler._per_optimizer_states[id(self.optimizer)]["found_inf_per_device"]
        if not checks:
            raise NumericalFailure("GradScaler did not record native overflow detection", report)
        report["native_found_inf"] = {str(device): float(value) for device, value in checks.items()}
        report["overflow"] = any(report["native_found_inf"].values())
        stats = report["unscaled_gradients"] = gradient_report(self.model)
        if stats["zero_fraction"] >= ZERO_DETAIL_AT or report["overflow"]:
            report["layer_gradients"] = layer_gradient_statistics(self.model, by_scale=True)
            for layer in report["layer_gradients"].values():
                for metric in ("finite_norm", "finite_max_abs"):
                    if layer[metric] is not None:
                        layer[metric] = finite_number(layer[metric])
        if not report["overflow"] and (report["scaled_gradients"]["nonfinite"] or stats["nonfinite"]
                                      or stats["zero"] == stats["elements"] or not stats["master_dtype_fp32"]):
            raise NumericalFailure("Backward yielded no finite, nonzero FP32 gradient signal", report)
        return report

    def diagnostic_backward(self):
        """Inspect and clip one diagnostic gradient without advancing the scaler or optimizer."""
        report = self.inspect_backward()
        if report["overflow"]:
            raise NumericalFailure("GradScaler detected overflow during diagnostic backward", report)
        self.clip_gradients(report)
        return report

    def clip_gradients(self, report):
        try:
            norm = clip_native_gradients(self.model.parameters())
        except RuntimeError as error:
            raise NumericalFailure("Native gradient clipping failed", report) from error
        report["gradient_norm"] = finite_number(norm)
        if report["gradient_norm"] is None or norm <= 0:
            raise NumericalFailure("Global gradient norm is non-finite or zero", report)

    def optimizer_step(self):
        require(self.optimizer_attempts < self.update_limit and self.effective_updates < self.update_limit,
                "Audited optimizer update budget exhausted")
        report = self.inspect_backward()
        snapshots = []
        if not report["overflow"]:
            self.clip_gradients(report)
            for idx, group in enumerate(self.optimizer.param_groups):
                active = [p for p in group["params"] if p.requires_grad]
                if active and group["lr"] > 0:
                    snapshots.append((group.get("param_group", str(idx)), active, parameter_vector(active)))
            if not snapshots:
                raise NumericalFailure("AdamW has no trainable parameter group at positive LR", report)

        native_before = self.native_steps
        self.optimizer_attempts += 1
        report["optimizer_attempted"] = True
        self.scaler.step(self.optimizer)
        self.scaler.update()
        report["scale_after"] = self.scaler.get_scale()
        observed_native_steps = self.native_steps - native_before
        report["native_step_called"] = observed_native_steps > 0
        if report["overflow"] or observed_native_steps != 1:
            if not observed_native_steps:
                self.amp_skips += 1
                report["amp_skipped"] = True
            expected_backoff = report["scale_before"] * NATIVE_DEFAULTS["backoff_factor"]
            report["native_overflow_handling_as_expected"] = (report["overflow"] and not observed_native_steps
                                                              and report["scale_after"] == expected_backoff)
            raise NumericalFailure("Native GradScaler overflow or unexplained AdamW skip", report)

        changes = []
        for name, params, before in snapshots:
            after = parameter_vector(params)
            changes.append({"group": name, "changed": int((after != before).sum()),
                            "max_abs_delta": finite_number((after - before).abs().max())})
        report.update(parameter_changes=changes, parameter_changed=any(g["changed"] for g in changes),
                      model_state_fp32_finite=model_fp32_finite(self.model),
                      optimizer_state_finite=finite_tree(self.optimizer.state),
                      scaler_state=self.scaler.state_dict())
        if not report["model_state_fp32_finite"] or not report["optimizer_state_finite"] or not finite_tree(report["scaler_state"]):
            raise NumericalFailure("Optimizer produced non-finite model, AdamW or scaler state", report)
        if not report["parameter_changed"]:
            raise NumericalFailure("Native AdamW step did not change any active master parameter", report)
        self.effective_updates += 1
        report["effective_update"] = self.effective_updates
        self.optimizer.zero_grad(set_to_none=True)
        if getattr(self.trainer, "ema", None) is not None:
            self.trainer.ema.update(self.model)
        return report


def optimizer_group_names(model, optimizer):
    by_id = {id(p): name for name, p in model.named_parameters()}
    return [[by_id[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def exact_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    return left == right


def validate_checkpoint(path, arm):
    from ultralytics.nn.tasks import SegmentationModel

    record = torch.load(path, map_location="cpu", weights_only=True)
    require(record["schema"] == CHECKPOINT_SCHEMA and record["engineering_only"] and record["arm"] == arm
            and record["optimizer_attempts"] == record["native_steps"] == record["effective_updates"] == MAX_UPDATES
            and record["amp_skips"] == 0 and record["initial_parameters_sha256"] != record["parameters_sha256"],
            "Smoke checkpoint lineage or measured update budget differs")
    require(record["state_sha256"] == hash_state(record["model_state"])
            and finite_tree([record["model_state"], record["optimizer"], record["scaler"]]),
            "Smoke checkpoint contains invalid or mismatched state")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(record["model_config"], ch=3, nc=7, verbose=False)
        model.load_state_dict(record["model_state"], strict=True)
    require(record["class_names"] == dict(enumerate(CLASSES)), "Smoke checkpoint class mapping differs")
    model.names = record["class_names"]
    require(model.model[-1].nc == record["nc"] == len(model.names) == 7 and model_fp32_finite(model)
            and parameters_sha(model) == record["parameters_sha256"]
            and hash_state(model.state_dict()) == record["state_sha256"], "Smoke checkpoint strict model reload disagrees")

    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [named[n] for n in names]} for names in record["optimizer_group_names"]])
    optimizer.load_state_dict(record["optimizer"])
    require(exact_tree(optimizer.state_dict(), record["optimizer"]), "Strict CPU AdamW reload changed state")
    steps = Counter(int(v["step"].item()) for v in optimizer.state.values())
    require(steps and min(steps) > 0 and max(steps) == MAX_UPDATES, "AdamW state corroborates no bounded step history")
    scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
    scaler.load_state_dict(record["scaler"])
    require(scaler.state_dict() == record["scaler"] and record["scaler"]["scale"] == 1.0
            and record["scaler"]["_growth_tracker"] == MAX_UPDATES, "Strict CPU GradScaler reload disagrees")
    return steps


def save_checkpoint(trainer, path, audit=None):
    audit = audit if audit is not None else trainer.audit
    model = trainer.model
    state = cpu_state(model.state_dict())
    record = {"schema": CHECKPOINT_SCHEMA, "engineering_only": True, "arm": trainer.arm,
              "model_config": json.loads(json.dumps(model.yaml)), "nc": model.model[-1].nc,
              "class_names": dict(model.names),
              "model_state": state, "state_sha256": hash_state(state), "parameters_sha256": parameters_sha(model),
              "initial_state_sha256": audit.initial_state_sha256, "initial_parameters_sha256": audit.initial_parameters_sha256,
              "optimizer": cpu_state(trainer.optimizer.state_dict()),
              "optimizer_group_names": optimizer_group_names(model, trainer.optimizer),
              "scaler": trainer.scaler.state_dict(), "optimizer_attempts": audit.optimizer_attempts, "native_steps": audit.native_steps,
              "amp_skips": audit.amp_skips, "effective_updates": audit.effective_updates,
              "identity": json.loads(json.dumps(getattr(trainer, "checkpoint_identity", {})))}
    with path.open("xb") as f:
        torch.save(record, f)
    step_counts = validate_checkpoint(path, trainer.arm)
    return {"path": str(path), "sha256": sha256_file(path), "model_state_sha256": record["state_sha256"],
            "parameters_sha256": record["parameters_sha256"], "initial_parameters_sha256": audit.initial_parameters_sha256,
            "initial_state_sha256": audit.initial_state_sha256, "strict_model_reload": True, "strict_optimizer_reload": True,
            "strict_scaler_reload": True, "adam_step_counts": dict(step_counts), "engineering_only": True}
