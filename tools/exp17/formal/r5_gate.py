"""Predeclared R5 early gate, observed within existing isolated TRAIN probes."""
import math
from contextlib import contextmanager

import torch

FOCUS = ("model.0.bn", "model.4.cv1.bn", "model.16.cv1.bn", "model.16.cv2.bn")
RULE = {"layers": list(FOCUS), "channel_reduction": "median_abs_normalized_mean_gap",
        "threshold": 3.0, "minimum_batches": 3, "positions": [1, 7, 11, 21],
        "required_master_mask_AP50_95": "strictly_positive",
        "threshold_status": "PREDECLARED_ENGINEERING_BOUND_NOT_HEALTHY_BASELINE_CALIBRATED"}


@contextmanager
def observe_mean_gap(model):
    modules = dict(model.named_modules())
    observations, handles = {}, []
    try:
        for name in FOCUS:
            module = modules[name]
            saved_mean = module.running_mean.detach().clone()
            saved_var = module.running_var.detach().clone()
            if not bool(torch.isfinite(saved_var).all() and (saved_var >= 0).all()):
                raise ValueError("Invalid saved R5 BN variance")
            def observe(_module, inputs, key=name, mean=saved_mean, var=saved_var):
                if key in observations:
                    return  # Only A, which precedes N within this same probe.
                x = inputs[0].detach().float()
                with torch.autocast(x.device.type, enabled=False):
                    batch_mean = x.mean((0,) + tuple(range(2, x.ndim)))
                    values = (batch_mean - mean.float()).abs() / (var.float() + _module.eps).sqrt()
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("Nonfinite R5 normalized A mean gap")
                observations[key] = {"D": float(values.median()), "epsilon": _module.eps,
                                     "elements_per_channel": x.numel() // x.shape[1],
                                     "path": "TRAIN_A_INPUT_VS_PRE_PROBE_MASTER_BUFFERS"}
            handles.append(module.register_forward_pre_hook(observe))
        yield observations
    finally:
        for handle in handles:
            handle.remove()


def decide(reports, master_result):
    if [r["batch_position"] for r in reports] != RULE["positions"]:
        raise ValueError("Missing or reordered R5 frozen probes")
    ap = float(master_result["standard"]["overall"]["metrics/mAP50-95(M)"])
    if not math.isfinite(ap):
        raise ValueError("Nonfinite R5 early VAL")
    bad = []
    for name in FOCUS:
        values = [float(r["bn_mean_gap"][name]["D"]) for r in reports]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid R5 mean gap")
        if sum(v >= RULE["threshold"] for v in values) >= RULE["minimum_batches"]:
            bad.append(name)
    return {"status": "PASS" if ap > 0 and not bad else "HOLD",
            "master_mask_AP50_95": ap, "mismatched_layers": bad, "rule": RULE}
