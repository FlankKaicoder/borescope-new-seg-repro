"""Connected A-minibatch BN statistics without CUDA discovery or initialization.

The statistics cover batch AND spatial axes, independently for each channel.
The auxiliary N traversal never writes running statistics. This is the R5
first-order statistics bridge. CPU-only runs never query CUDA generators;
an already initialized native CUDA runtime also has its generators observed.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def require(value, message):
    if not value:
        raise ValueError(message)


def cpu_rng_state():
    return {"torch": torch.get_rng_state().clone(), "python": random.getstate(),
            "numpy": np.random.get_state()}


def restore_cpu_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def rng_equal(left, right):
    return (torch.equal(left["torch"], right["torch"]) and left["python"] == right["python"]
            and left["numpy"][0] == right["numpy"][0]
            and np.array_equal(left["numpy"][1], right["numpy"][1])
            and left["numpy"][2:] == right["numpy"][2:])


def model_rng_state():
    """Observe CUDA generators only when CUDA is already initialized.

    CPU-only construction never asks whether a GPU is available and never
    invokes a CUDA RNG getter. This observer does not restore or conceal draws.
    """
    initialized = torch.cuda.is_initialized()
    return {"cpu": cpu_rng_state(), "cuda_initialized": initialized,
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()] if initialized else None}


def rng_model_equal(left, right):
    if not rng_equal(left["cpu"], right["cpu"]) or left["cuda_initialized"] != right["cuda_initialized"]:
        return False
    if not left["cuda_initialized"]:
        return left["cuda"] is None and right["cuda"] is None
    return len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"], strict=True))


def bn_modules(model):
    return {name: value for name, value in model.named_modules()
            if isinstance(value, nn.modules.batchnorm._BatchNorm)}


def buffer_snapshot(model):
    return {name: {key: None if getattr(module, key) is None else getattr(module, key).detach().clone()
                   for key in ("running_mean", "running_var", "num_batches_tracked")}
            for name, module in bn_modules(model).items()}


def buffers_equal(model, snapshot):
    current = buffer_snapshot(model)
    return current.keys() == snapshot.keys() and all(
        (current[name][key] is None and value is None)
        or (current[name][key] is not None and value is not None and torch.equal(current[name][key], value))
        for name, values in snapshot.items() for key, value in values.items())


@dataclass
class BatchStatistics:
    mean: torch.Tensor
    biased_variance: torch.Tensor
    elements_per_channel: int


def native_prefix(name):
    """Native 0..16, or all BN modules in an explicitly synthetic graph."""
    return not name.startswith("model.") or int(name.split(".")[1]) <= 16


@contextmanager
def capture_native_bn(model, *, connected_prefix=False, prefix_predicate=native_prefix):
    modules = bn_modules(model)
    require(modules and all(m.training and m.track_running_stats for m in modules.values()),
            "A requires native TRAIN BN with tracked running buffers")
    captured, handles = {}, []

    def hook(name):
        def capture(_module, inputs):
            original = inputs[0]
            dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
            x = original.to(dtype) if connected_prefix and prefix_predicate(name) else original.detach().to(dtype)
            variance, mean = torch.var_mean(x, dim=(0,) + tuple(range(2, x.ndim)), unbiased=False)
            require(bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()), "Nonfinite A BN statistics")
            captured.setdefault(name, []).append(BatchStatistics(mean, variance, x.numel() // x.shape[1]))
        return capture

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(hook(name)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class StatisticsGradient(torch.autograd.Function):
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
            dims = (0,) + tuple(range(2, n.ndim))
            shape = (1, -1) + (1,) * (n.ndim - 2)
            invstd = (variance.to(dtype) + ctx.epsilon).rsqrt()
            gm = -gamma * invstd * h.sum(dims)
            gv = -.5 * gamma * invstd.pow(3) * (h * (n - mean.to(dtype).view(shape))).sum(dims)
        return upstream, gm.to(mean.dtype), gv.to(variance.dtype), None, None, None


@contextmanager
def auxiliary_bn(model, mode="replay_a_connected", captured=None):
    require(mode in ("replay_a_connected", "replay_a"), "Only preregistered A-statistic replay is supported")
    modules = bn_modules(model)
    require(modules and all(m.training and m.track_running_stats for m in modules.values()), "N requires TRAIN BN")
    require(isinstance(captured, dict), "A-minibatch statistics required")
    before, rng = buffer_snapshot(model), model_rng_state()
    flags = {name: (m.training, m.track_running_stats) for name, m in modules.items()}
    overrides = {name: ("forward" in m.__dict__, m.__dict__.get("forward")) for name, m in modules.items()}
    calls = {}

    def replacement(name):
        def normalize(module, x):
            count = calls.get(name, 0)
            require(name in captured and count < len(captured[name]), "N BN has no matching A call: " + name)
            item = captured[name][count]
            calls[name] = count + 1
            require(item.mean.shape == item.biased_variance.shape == (x.shape[1],), "BN channel mismatch")
            require(item.mean.device == x.device, "BN statistic device mismatch")
            connected = mode == "replay_a_connected"
            require(item.mean.requires_grad == connected and item.biased_variance.requires_grad == connected,
                    "A-statistics graph does not match replay mode")
            base = F.batch_norm(x, item.mean.detach(), item.biased_variance.detach(),
                                module.weight, module.bias, False, 0., module.eps)
            if connected:
                return StatisticsGradient.apply(base, item.mean, item.biased_variance,
                                                x.detach(), module.weight.detach(), module.eps)
            return base
        return normalize

    try:
        for name, module in modules.items():
            module.forward = MethodType(replacement(name), module)
        yield {"calls": calls, "statistics_scope": "WHOLE_A_MINIBATCH_AND_SPATIAL_AXES_PER_CHANNEL"}
    finally:
        for name, module in modules.items():
            existed, old = overrides[name]
            if existed:
                module.forward = old
            else:
                module.__dict__.pop("forward", None)
        require(all((m.training, m.track_running_stats) == flags[name] for name, m in modules.items()), "N changed BN flags")
        require(buffers_equal(model, before), "N changed running BN buffers")
        require(rng_model_equal(rng, model_rng_state()), "N consumed a model RNG stream (CPU or initialized CUDA)")
