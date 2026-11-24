"""Within-step BN diagnostics for a shared, sparse counterfactual prefix.

Mode B detaches the A statistics: they are a fixed within-step normalizer, not
another gradient route from N to A. Gradients still flow through N inputs,
shared weights and BN affine parameters; A retains its own native loss graph.
No second persistent buffers, in-place buffer restoration or RNG restoration.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


@dataclass(frozen=True)
class BatchStatistics:
    mean: torch.Tensor
    biased_variance: torch.Tensor
    elements_per_channel: int


def bn_modules(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)}


def buffer_snapshot(model):
    return {name: {key: None if getattr(m, key) is None else getattr(m, key).detach().clone()
                   for key in ("running_mean", "running_var", "num_batches_tracked")}
            for name, m in bn_modules(model).items()}


def buffers_equal(model, snapshot):
    modules = bn_modules(model)
    return set(modules) == set(snapshot) and all(
        (getattr(modules[name], key) is None and value is None) or
        (isinstance(getattr(modules[name], key), torch.Tensor) and value is not None
         and torch.equal(getattr(modules[name], key), value))
        for name, values in snapshot.items() for key, value in values.items())


def rng_snapshot():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def rng_equal(state):
    other = rng_snapshot()
    return (state["python"] == other["python"] and state["numpy"][0] == other["numpy"][0]
            and np.array_equal(state["numpy"][1], other["numpy"][1])
            and state["numpy"][2:] == other["numpy"][2:]
            and torch.equal(state["torch_cpu"], other["torch_cpu"])
            and len(state["torch_cuda"]) == len(other["torch_cuda"])
            and all(torch.equal(a, b) for a, b in zip(state["torch_cuda"], other["torch_cuda"], strict=True)))


@contextmanager
def capture_native_bn(model, *, connected_prefix=False):
    """Observe actual BN inputs without replacing A normalization or buffer writes."""
    modules = bn_modules(model)
    require(modules and all(m.training and m.track_running_stats for m in modules.values()),
            "A statistics capture requires native TRAIN BatchNorm with tracked buffers")
    captured, handles = {}, []
    def hook(name):
        def capture(_module, inputs):
            prefix = name.startswith("model.") and int(name.split(".")[1]) <= 16
            x = inputs[0].float() if connected_prefix and prefix else inputs[0].detach().float()
            dimensions = (0,) + tuple(range(2, x.ndim))
            variance, mean = torch.var_mean(x, dim=dimensions, unbiased=False)
            require(bool(torch.isfinite(mean).all()) and bool(torch.isfinite(variance).all()), "Nonfinite A BN statistics")
            captured.setdefault(name, []).append(BatchStatistics(mean, variance, x.numel() // x.shape[1]))
        return capture
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(hook(name)))
        yield captured
    finally:
        for handle in handles: handle.remove()


class StatisticsGradient(torch.autograd.Function):
    """Native forward and base VJP, plus only the two A-statistic partials.

    This first-order bridge is reentrant for retained component-gradient graphs.
    Detached N/affine inputs cannot receive a second direct derivative.
    """
    @staticmethod
    def forward(ctx, base, mean, variance, normal, weight, epsilon):
        ctx.save_for_backward(mean.detach(), variance.detach(), normal.detach(), weight.detach())
        ctx.epsilon = epsilon
        return base.clone()

    @staticmethod
    def backward(ctx, upstream):
        mean, variance, normal, weight = ctx.saved_tensors
        dtype = torch.float64 if mean.dtype == torch.float64 else torch.float32
        with torch.autocast(upstream.device.type, enabled=False):
            h, n, gamma = upstream.to(dtype), normal.to(dtype), weight.to(dtype)
            dimensions = (0,) + tuple(range(2, n.ndim))
            shape = (1, -1) + (1,) * (n.ndim - 2)
            invstd = (variance.to(dtype) + ctx.epsilon).rsqrt()
            gm = -gamma * invstd * h.sum(dimensions)
            gv = -.5 * gamma * invstd.pow(3) * (h * (n - mean.to(dtype).view(shape))).sum(dimensions)
        return upstream, gm.to(mean.dtype), gv.to(variance.dtype), None, None, None


@contextmanager
def auxiliary_bn(model, mode, captured=None):
    """Mode independent uses N batch statistics; replay_a uses detached A statistics.

    BN flags/buffers never change. The temporary forward override calls the same
    PyTorch BN primitive and is removed, including on exceptions. Every RNG stream
    is checked, never restored to conceal auxiliary stochastic operations.
    """
    require(mode in ("independent", "replay_a", "replay_a_connected"), "Unknown auxiliary BN mode")
    modules = bn_modules(model)
    require(modules and all(m.training and m.track_running_stats for m in modules.values()),
            "N route requires native TRAIN BatchNorm")
    require(mode == "independent" or isinstance(captured, dict), "A statistics required for replay mode")
    before, rng = buffer_snapshot(model), rng_snapshot()
    flags = {name: (m.training, m.track_running_stats) for name, m in modules.items()}
    overrides = {name: ("forward" in m.__dict__, m.__dict__.get("forward")) for name, m in modules.items()}
    counts = {}
    def replacement(name):
        def normalize(module, x):
            if mode == "independent":
                return F.batch_norm(x, None, None, module.weight, module.bias, True, 0.0, module.eps)
            count = counts.get(name, 0)
            require(name in captured and count < len(captured[name]), "N called a BN without matching A statistics: " + name)
            item = captured[name][count]; counts[name] = count + 1
            require(item.mean.shape == item.biased_variance.shape == (x.shape[1],)
                    and item.mean.device == x.device, "Invalid replay statistics")
            if mode == "replay_a":
                require(not item.mean.requires_grad and not item.biased_variance.requires_grad
                        and item.mean.grad_fn is None and item.biased_variance.grad_fn is None,
                        "Invalid or graph-connected replay statistics")
            else:
                require(item.mean.requires_grad and item.biased_variance.requires_grad,
                        "Connected replay lost its A statistics graph")
            base = F.batch_norm(x, item.mean.detach(), item.biased_variance.detach(),
                                module.weight, module.bias, False, 0.0, module.eps)
            if mode == "replay_a_connected":
                return StatisticsGradient.apply(base, item.mean, item.biased_variance,
                                                x.detach(), module.weight.detach(), module.eps)
            return base
        return normalize
    try:
        for name, module in modules.items():
            module.forward = MethodType(replacement(name), module)
        yield {"mode": mode, "A_statistics_detached": mode != "replay_a_connected", "replayed_calls": counts}
    finally:
        for name, module in modules.items():
            existed, old = overrides[name]
            if existed: module.forward = old
            else: del module.forward
        require(all((m.training, m.track_running_stats) == flags[name] for name, m in modules.items()), "N changed BN training flags")
        require(buffers_equal(model, before), "N changed native BN running buffers")
        require(rng_equal(rng), "N consumed an RNG stream; state was not silently restored")
