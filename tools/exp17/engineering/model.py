"""Real shared-weight A/full + sparse-N/fused-P3 prefix computation.

No optimizer, data sampling, augmentation draws, checkpoint selection or training
authorization lives here. N stops after original layer16, including its true neck
dependencies; original J1 pooling, projection and SupCon are imported unchanged.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tools.exp15 import regions as j1
from tools.exp17.pair_loss import tensor_identity
from tools.exp17.engineering.bn import auxiliary_bn, capture_native_bn
from tools.exp17.engineering.losses import joint_loss, per_class_pair_losses, sparse_source_ranking


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def check_projection(model):
    head = getattr(model, "region_projection", None)
    require(isinstance(head, nn.Sequential) and len(head) == 3
            and isinstance(head[0], nn.Linear) and isinstance(head[1], nn.ReLU) and isinstance(head[2], nn.Linear)
            and (head[0].in_features, head[0].out_features, head[2].in_features, head[2].out_features) == (64, 128, 128, 128),
            "Expected the unchanged J1 shared 64-128-128 projection")
    return head


def ensure_projection(model):
    if not hasattr(model, "region_projection"):
        j1.add_projection(model)
        model.region_projection.to(device=next(model.parameters()).device)
    return check_projection(model)


def prefix_p3(model, images):
    """Exactly the native dependency routing, truncated AFTER fused P3 layer16."""
    require(torch.is_grad_enabled(), "N prefix must retain the shared-weight autograd graph")
    require(images.ndim == 4 and images.shape[1] == 3 and len(images) > 0
            and images.shape[-2] % 32 == images.shape[-1] % 32 == 0, "Invalid sparse N image shape")
    require(hasattr(model, "model") and len(model.model) > j1.P3_LAYER, "Missing native YOLO module graph")
    previous, saved = images, []
    for index, module in enumerate(model.model):
        if index > j1.P3_LAYER: break
        require(module.i == index, "Native module index identity changed")
        if module.f != -1:
            previous = saved[module.f] if isinstance(module.f, int) else [previous if f == -1 else saved[f] for f in module.f]
            require(previous is not None and (not isinstance(previous, list) or all(v is not None for v in previous)), "Prefix dependency was not retained")
        previous = module(previous)
        saved.append(previous if module.i in model.save else None)
    require(isinstance(previous, torch.Tensor) and previous.shape == (len(images), j1.P3_CHANNELS,
            images.shape[-2] // j1.P3_STRIDE, images.shape[-1] // j1.P3_STRIDE)
            and previous.grad_fn is not None, "Missing gradient-enabled fused P3 prefix")
    return previous


@dataclass
class AForward:
    yolo: torch.Tensor
    parts: torch.Tensor
    p3: torch.Tensor
    embedding: torch.Tensor
    supcon: torch.Tensor
    bn_statistics: dict


@dataclass
class JointForward:
    yolo: torch.Tensor
    parts: torch.Tensor
    supcon: torch.Tensor
    pair: torch.Tensor
    total: torch.Tensor | None
    report: dict
    a_p3: torch.Tensor
    n_p3: torch.Tensor | None
    a_embedding: torch.Tensor
    n_embedding: torch.Tensor | None
    bn_statistics: dict
    per_class_pair_tensors: dict


def forward_a(model, batch, regions, *, connected_prefix=False):
    check_projection(model)
    require(model.training and torch.is_grad_enabled(), "A needs the original native TRAIN graph")
    features = []
    handle = model.model[j1.P3_LAYER].register_forward_hook(lambda _m, _i, value: features.append(value))
    try:
        with capture_native_bn(model, connected_prefix=connected_prefix) as statistics:
            native, parts = model(batch)
    finally:
        handle.remove()
    require(len(features) == 1 and features[0].grad_fn is not None, "Native A did not expose exactly one differentiable fused P3")
    embedding = j1.project_regions(features[0], batch["masks"], regions, model.region_projection)
    supcon = j1.supervised_contrast(embedding, regions.classes, regions)
    return AForward(native.sum(), parts, features[0], embedding, supcon, statistics)


def forward_n(model, images, *, bn_mode, a_statistics=None, checkpoint_prefix=False):
    def compute(value):
        # The context must also surround non-reentrant backward recomputation.
        with auxiliary_bn(model, bn_mode, a_statistics):
            return prefix_p3(model, value)
    require(type(checkpoint_prefix) is bool, "checkpoint_prefix must be explicit bool")
    if checkpoint_prefix:
        return checkpoint(compute, images, use_reentrant=False, preserve_rng_state=False)
    return compute(images)


def validate_tensor_bindings(batch, a_regions, normal_images, n_regions, bindings):
    """Verify actual augmented A/N tensor bytes and one N image per active A slot."""
    require(normal_images.ndim == 4 and normal_images.shape[1:] == batch["img"].shape[1:]
            and normal_images.device == batch["img"].device
            and normal_images.dtype == batch["img"].dtype == torch.float32
            and bool(torch.isfinite(normal_images).all()) and bool(torch.isfinite(batch["img"]).all())
            and float(normal_images.min()) >= 0 and float(normal_images.max()) <= 1, "Invalid augmented sparse N image tensor")
    a_keep, n_keep = a_regions.cells > 0, n_regions.cells > 0
    a_slots, n_slots = a_regions.slots[a_keep], n_regions.slots[n_keep]
    classes, groups = a_regions.classes[a_keep], a_regions.groups[a_keep]
    n_classes, n_groups = n_regions.classes[n_keep], n_regions.groups[n_keep]
    seen_a_rows, seen_n_rows = set(), set()
    by_a, by_n, actual_a_hashes, actual_n_hashes = {}, {}, {}, {}
    for item in bindings:
        require(type(item.a_row) is type(item.n_row) is type(item.class_id) is int, "Binding row/class type differs")
        require(0 <= item.a_row < len(a_slots) and 0 <= item.n_row < len(n_slots), "Bound A/N region row is out of bounds")
        require(item.a_row not in seen_a_rows and item.n_row not in seen_n_rows, "Duplicate A/N bound Polygon row")
        require(int(classes[item.a_row]) == int(n_classes[item.n_row]) == item.class_id
                and str(groups[item.a_row]) == str(n_groups[item.n_row]) == item.group, "Bound A/N class or group differs")
        require(getattr(item, "support_complete", True) is True, "Unsupported Pair binding")
        seen_a_rows.add(item.a_row); seen_n_rows.add(item.n_row)
        a_slot, n_slot = int(a_slots[item.a_row]), int(n_slots[item.n_row])
        require(0 <= a_slot < len(batch["img"]) and 0 <= n_slot < len(normal_images), "Bound A/N image slot is out of bounds")
        require(getattr(item, "a_slot", a_slot) == a_slot and getattr(item, "n_slot", n_slot) == n_slot, "Bound image slot metadata differs")
        require(bool(np.any((classes == item.class_id) & (groups != item.group))), "Pair planner exposed N despite no cross-group same-class B")
        require(by_a.get(a_slot, n_slot) == n_slot and by_n.get(n_slot, a_slot) == a_slot, "A source mixed N slots or an N slot was shared across A images")
        by_a[a_slot], by_n[n_slot] = n_slot, a_slot
        if a_slot not in actual_a_hashes: actual_a_hashes[a_slot] = tensor_identity(batch["img"][a_slot])
        if n_slot not in actual_n_hashes: actual_n_hashes[n_slot] = tensor_identity(normal_images[n_slot])
        require(actual_a_hashes[a_slot] == item.a_tensor_sha256 and actual_n_hashes[n_slot] == item.n_tensor_sha256,
                "Actual augmented A/N tensor bytes differ from replay binding")
    require(set(by_n) == set(range(len(normal_images))), "Unpaired N image exposure is forbidden")


def forward_joint(model, batch, a_regions, normal_images, normal_masks, normal_regions, bindings,
                  *, bn_mode, lambda_pair=None, checkpoint_prefix=False):
    """One joint graph. lambda_pair=None exposes components for read-only calibration."""
    if bindings:
        require(normal_images is not None and normal_regions is not None and normal_masks is not None, "Missing sparse N replay")
        validate_tensor_bindings(batch, a_regions, normal_images, normal_regions, bindings)
    else:
        require(normal_images is None or len(normal_images) == 0, "No-Pair batch must not expose extra N images")
    require(bn_mode in ("independent", "replay_a", "replay_a_connected"), "Unknown J2 BN mode")
    require(bn_mode != "replay_a_connected" or not checkpoint_prefix, "Connected replay requires the original single traversal")
    a = forward_a(model, batch, a_regions,
                  connected_prefix=bn_mode == "replay_a_connected" and bool(bindings))
    if bindings:
        n_feature = forward_n(model, normal_images, bn_mode=bn_mode, a_statistics=a.bn_statistics,
                              checkpoint_prefix=checkpoint_prefix)
        n_embedding = j1.project_regions(n_feature, normal_masks, normal_regions, model.region_projection)
        pair, report = sparse_source_ranking(a.embedding, n_embedding, a_regions, normal_regions, bindings)
        per_class = per_class_pair_losses(a.embedding, n_embedding, a_regions, normal_regions, bindings)
    else:
        n_feature, n_embedding = None, None
        pair = a.embedding.sum() * 0
        per_class = {}
        report = {"paired_anchors": 0, "independent_A_groups": 0, "classes": [], "per_anchor": []}
    report = {**report, "bn_mode": bn_mode, "A_statistics_detached_on_N": bn_mode != "replay_a_connected",
              "N_executed_layers": list(range(j1.P3_LAYER + 1)) if bindings else [],
              "N_prefix_activation_checkpoint": checkpoint_prefix,
              "N_images": len(normal_images) if bindings else 0, "N_has_native_detection_or_segmentation_head": False}
    total = None if lambda_pair is None else joint_loss(a.yolo, a.supcon, pair, lambda_pair)
    return JointForward(a.yolo, a.parts, a.supcon, pair, total, report, a.p3, n_feature,
                        a.embedding, n_embedding, a.bn_statistics, per_class)


def native_state_dict(model):
    check_projection(model)
    state = model.state_dict()
    require({k for k in state if k.startswith("region_projection.")} == {
        "region_projection.0.weight", "region_projection.0.bias", "region_projection.2.weight", "region_projection.2.bias"},
        "Unexpected training-only projection state")
    require(all(not value.is_floating_point() or value.dtype == torch.float32 for value in state.values()),
            "Native checkpoint extraction requires FP32 master parameters and buffers")
    return {k: v.detach().clone() for k, v in state.items() if not k.startswith("region_projection.")}


@contextmanager
def _native_activation_scope():
    """Give a newly constructed native graph its own default activation.

    Pinned Ultralytics Conv instances otherwise reuse a class-level SiLU,
    including the activation owned by the live training graph. Constructor
    mode changes and later native eval must only reach the new graph. Restore
    the exact class attribute on success, exception and nested construction.
    """
    from ultralytics.nn.modules.conv import Conv
    original = Conv.default_act
    Conv.default_act = copy.deepcopy(original)
    try:
        yield
    finally:
        Conv.default_act = original


def export_native_model(model):
    """Strictly reload a native seven-class skeleton, without the auxiliary module."""
    from ultralytics.nn.tasks import SegmentationModel
    require(model.model[-1].nc == 7, "Native class mapping changed")
    state = {k: v.cpu() for k, v in native_state_dict(model).items()}
    with torch.random.fork_rng(devices=[]), _native_activation_scope():
        native = SegmentationModel(copy.deepcopy(model.yaml), ch=3, nc=7, verbose=False)
    native.load_state_dict(state, strict=True)
    require(not hasattr(native, "region_projection") and native.model[-1].nc == 7, "Projection stripping/native class identity failed")
    return native
