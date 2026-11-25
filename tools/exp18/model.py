"""EXP18 mathematical and injected-forward machinery; no native model imports.

Pool every N ROI in its own image before indexing the frozen permutation.
J1 keeps its original projection; state supervision uses normalized raw P3.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .bn import capture_native_bn, auxiliary_bn

ARMS = ("j1", "a1_true", "a1_shuffled")
TRAINING_ONLY_PREFIXES = ("region_projection.", "state_head.")


def require(value, message):
    if not value:
        raise ValueError(message)


def add_state_head(model, *, seed):
    require(not hasattr(model, "state_head"), "State head already exists")
    require(type(seed) is int and seed >= 0, "Explicit nonnegative head seed required")
    # Generator and construction are CPU-only. Native CPU RNG is exactly restored.
    before = torch.get_rng_state()
    try:
        head = nn.Linear(64, 1, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            head.weight.uniform_(-1 / 8, 1 / 8, generator=generator)
    finally:
        torch.set_rng_state(before)
    reference = next(model.parameters(), None)
    if reference is not None:
        head.to(device=reference.device, dtype=reference.dtype)
    model.add_module("state_head", head)
    return head


def native_state_dict(model):
    return {k: value.detach().clone() for k, value in model.state_dict().items()
            if not k.startswith(TRAINING_ONLY_PREFIXES)}


def pool_raw_regions(feature, masks, regions):
    """Same polygon coverage reduction as EXP15; cells>0 compact row order."""
    require(feature.ndim == 4 and feature.shape[1] == 64 and masks.ndim == 3,
            "Expected P3 [B,64,H,W] and native overlap masks [B,2H,2W]")
    require(feature.shape[0] == masks.shape[0] and tuple(feature.shape[-2:]) == tuple(v // 2 for v in masks.shape[-2:]),
            "Native mask/P3 dimensions differ")
    dtype = torch.float64 if feature.dtype == torch.float64 else torch.float32
    with torch.autocast(feature.device.type, enabled=False):
        ids = torch.arange(1, int(regions.max_instances) + 1, device=feature.device)
        binary = (masks[:, None] == ids[None, :, None, None]).to(dtype)
        weights = F.avg_pool2d(binary, 2).flatten(2)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(.25)
        pooled = weights.bmm(feature.to(dtype).flatten(2).transpose(1, 2))
        cells = torch.as_tensor(regions.cells, device=feature.device)
        keep = cells > 0
        slots = torch.as_tensor(regions.slots, device=feature.device)[keep].long()
        ordinals = torch.as_tensor(regions.ordinals, device=feature.device)[keep].long()
        return F.normalize(pooled[slots, ordinals], dim=1)


def _weights(weights, count, reference):
    result = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
    require(result.shape == (count,) and bool(torch.isfinite(result).all()) and bool((result > 0).all()),
            "State reduction weights must be finite and positive")
    require(count > 0 and abs(float(result.sum().detach()) - 1.) < 1e-6, "State weights must sum to one")
    return result


def a1_loss(head, u_a, u_n, weights):
    require(u_a.shape == u_n.shape and u_a.ndim == 2 and u_a.shape[1] == 64, "Raw state features must be paired 64-D rows")
    dtype = torch.float64 if u_a.dtype == torch.float64 else torch.float32
    with torch.autocast(u_a.device.type, enabled=False):
        gap = (head(u_a.to(dtype)) - head(u_n.to(dtype))).squeeze(-1)
        return (_weights(weights, len(gap), gap) * F.softplus(-gap)).sum()


def a0_loss(head, u_a, u_n, weights):
    """Mathematical marginal BCE reference, never a fourth formal arm."""
    require(u_a.shape == u_n.shape and u_a.ndim == 2 and u_a.shape[1] == 64, "Raw state features must be paired 64-D rows")
    dtype = torch.float64 if u_a.dtype == torch.float64 else torch.float32
    with torch.autocast(u_a.device.type, enabled=False):
        score_a, score_n = head(u_a.to(dtype)).squeeze(-1), head(u_n.to(dtype)).squeeze(-1)
        return (_weights(weights, len(u_a), score_a) * .5 * (F.softplus(-score_a) + F.softplus(score_n))).sum()


def _get(plan, key):
    return plan[key] if isinstance(plan, Mapping) else getattr(plan, key)


@dataclass
class AForward:
    yolo: torch.Tensor
    supcon: torch.Tensor
    p3: torch.Tensor
    parts: object = None


@dataclass
class JointForward:
    yolo: torch.Tensor
    supcon: torch.Tensor
    state: torch.Tensor
    total: torch.Tensor
    head_active: bool
    a_raw: object
    n_raw: object
    a_p3: torch.Tensor
    n_p3: object
    matching_sha256: str
    bn_statistics: dict
    parts: object = None


def forward_joint(model, batch, a_regions, normal_images, normal_masks, normal_regions, matching,
                  *, arm, lambda_state, a_forward, n_forward, pool_regions=pool_raw_regions):
    """Callbacks perform native/full-A and shared/prefix-N traversals after gates.

    a_forward(model,batch,a_regions) -> AForward, preserving original J1 losses.
    n_forward(model,normal_images) -> raw fused P3. This function controls BN.
    Only injected synthetic graphs are exercised by the current CPU stage.
    """
    require(arm in ARMS, "Unknown EXP18 arm")
    require(model.training and torch.is_grad_enabled(), "Joint forward requires TRAIN autograd")
    require(lambda_state > 0, "Candidate state coefficient must be positive")
    a_rows = tuple(_get(matching, "a_rows")) if matching is not None else ()
    active = arm != "j1" and bool(a_rows)
    if arm != "j1":
        require(isinstance(getattr(model, "state_head", None), nn.Linear)
                and model.state_head.in_features == 64 and model.state_head.out_features == 1
                and model.state_head.bias is None, "Expected no-bias raw64 state head")
    require(active or normal_images is None or len(normal_images) == 0, "Empty/J1 batch must not expose N images")
    with capture_native_bn(model, connected_prefix=active) as statistics:
        a = a_forward(model, batch, a_regions)
    require(isinstance(a, AForward), "A callback must return AForward")
    if active:
        n_rows = _get(matching, "n_rows")
        require(n_rows is not None and len(n_rows) == len(a_rows), "Matching N own-ROI indices are unbound")
        permutation = tuple(_get(matching, "permutation"))
        require(sorted(permutation) == list(range(len(a_rows))) and all(i != j for i, j in enumerate(permutation)),
                "Matching permutation must be a complete derangement")
        with auxiliary_bn(model, "replay_a_connected", statistics):
            n_p3 = n_forward(model, normal_images)
        a_raw = pool_regions(a.p3, batch["masks"], a_regions)
        n_raw = pool_regions(n_p3, normal_masks, normal_regions)
        a_index = torch.tensor(a_rows, dtype=torch.long, device=a_raw.device)
        n_index = torch.tensor(tuple(n_rows), dtype=torch.long, device=n_raw.device)
        require(bool((a_index >= 0).all() and (a_index < len(a_raw)).all()
                     and (n_index >= 0).all() and (n_index < len(n_raw)).all()), "ROI compact index out of bounds")
        selected_n = n_raw[n_index]
        if arm == "a1_shuffled":
            selected_n = selected_n[torch.tensor(permutation, dtype=torch.long, device=n_raw.device)]
        state = a1_loss(model.state_head, a_raw[a_index], selected_n, _get(matching, "weights"))
    else:
        a_raw, n_raw, n_p3 = None, None, None
        # No reference to head parameters: empty head grad MUST remain None.
        state = a.yolo.new_zeros(())
    total = a.yolo + 8. * a.supcon + lambda_state * state
    require(bool(torch.isfinite(total)), "Nonfinite joint loss")
    sha = _get(matching, "sha256") if matching is not None else ""
    return JointForward(a.yolo, a.supcon, state, total, active, a_raw, n_raw, a.p3,
                        n_p3, sha, statistics, a.parts)
