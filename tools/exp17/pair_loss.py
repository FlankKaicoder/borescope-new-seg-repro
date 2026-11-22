"""Small source-aware J2 components. Real Pair training remains disabled in Exp17."""

import hashlib
import json
import random
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from tools.exp14.a11_full_numerics import capture_rng_state, numpy_rng_state, restore_rng_state
from tools.exp14.audit_admission import require
from tools.exp15.regions import P3_LAYER, project_regions, supervised_contrast

MARGIN = 0.20  # Engineering candidate, not a calibrated formal research coefficient.
SUPCON_WEIGHT = 8.0


def restore_auxiliary_rng(state):
    # The historical formal restorer deliberately requires one CUDA device.
    # CPU unit tests have zero devices but must preserve the same three CPU streams.
    if state["torch_cuda"]:
        restore_rng_state(state)
    else:
        require(torch.cuda.device_count() == 0, "CPU-only captured RNG no longer matches the visible topology")
        random.setstate(state["python"])
        np.random.set_state(numpy_rng_state(state["numpy"]))
        torch.set_rng_state(state["torch_cpu"])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_identity(value):
    tensor = value.detach().cpu().contiguous()
    header = json.dumps({"shape":list(tensor.shape),"dtype":str(tensor.dtype)},sort_keys=True).encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class PairBinding:
    """A/N rows refer to the same valid pooled Polygon in full matched batch slots."""

    a_row: int
    n_row: int
    group: str
    polygon_id: str
    class_id: int
    repair_version: str
    source_sha256: str
    repair_sha256: str
    matrix_sha256: str
    valid_mask_sha256: str
    augmentation_sha256: str
    a_tensor_sha256: str = ""
    n_tensor_sha256: str = ""
    synthetic: bool = False
    support_complete: bool = False
    contamination_clear: bool = False
    unambiguous_region: bool = False

    @property
    def identity(self):
        return digest(asdict(self))


def real_pair_training_gate(*_args, **_kwargs):
    raise ValueError("Exp17 cannot authorize real Pair training; user G1 review and an independent Data/Formal Gate are required")


def select_synthetic_bindings(bindings, regions, *, seed=17):
    """One version per source/Polygon using a local RNG; never change the native sampler."""
    classes = regions.classes[regions.cells > 0]
    groups = regions.groups[regions.cells > 0]
    identities, variants, skipped = set(), {}, Counter()
    for item in bindings:
        require(isinstance(item, PairBinding) and item.synthetic,
                "This prototype only permits explicitly synthetic Pair bindings")
        require(type(item.a_row) is type(item.n_row) is type(item.class_id) is int
                and isinstance(item.group,str) and item.group,
                "Pair row/class/group identity types differ")
        require(item.identity not in identities, "Duplicate Pair identity")
        identities.add(item.identity)
        require(0 <= item.a_row < len(classes) and item.n_row == item.a_row
                and item.class_id == int(classes[item.a_row]) and item.group == str(groups[item.a_row]),
                "A/N slot, Polygon class or source group is misbound")
        require(item.polygon_id and item.repair_version and all(
            len(getattr(item, name)) == 64 and all(c in "0123456789abcdef" for c in getattr(item, name))
            for name in ("source_sha256", "repair_sha256", "matrix_sha256", "valid_mask_sha256", "augmentation_sha256",
                         "a_tensor_sha256", "n_tensor_sha256")),
            "Incomplete synthetic provenance/trace identity")
        reason = ("incomplete_support" if not item.support_complete else "contamination_or_unknown"
                  if not item.contamination_clear else "ambiguous_region" if not item.unambiguous_region else None)
        if reason:
            skipped[reason] += 1
            continue
        key = item.group, item.polygon_id
        variants.setdefault(key, []).append(item)
    rng = random.Random(seed)
    chosen = [rng.choice(sorted(options, key=lambda b: b.identity)) for _, options in sorted(variants.items())]
    require(len({b.a_row for b in chosen}) == len(chosen), "More than one Polygon binding points to the same A row")
    slots = regions.slots[regions.cells > 0]
    by_slot = {}
    for item in chosen:
        by_slot.setdefault(int(slots[item.a_row]),[]).append(item)
    conflicting = {slot for slot,items in by_slot.items() if len({
        (b.repair_version,b.repair_sha256,b.matrix_sha256,b.valid_mask_sha256,b.augmentation_sha256,b.n_tensor_sha256)
        for b in items}) != 1}
    skipped["counterfactual_slot_version_conflict"] += sum(len(by_slot[s]) for s in conflicting)
    chosen = [b for b in chosen if int(slots[b.a_row]) not in conflicting]
    return chosen, dict(skipped)


def source_ranking(embedding, normal_embedding, regions, bindings, margin=MARGIN):
    """d(A,B) is averaged first within each other-source group, then across groups.

    Anchor losses are averaged per Polygon, then per A group, then per represented class.
    No hard mining, unlabelled-normal assumption, extra image sampling or class-specific margin.
    """
    require(0 < margin < 2 and embedding.ndim == 2 and embedding.shape == normal_embedding.shape,
            "Unexpected A/N embeddings or margin")
    classes, groups = regions.classes[regions.cells > 0], regions.groups[regions.cells > 0]
    require(len(classes) == len(embedding), "Region-to-embedding row count differs")
    grouped, details, skipped = {}, [], Counter()
    with torch.autocast(embedding.device.type, enabled=False):
        a, n = embedding.float(), normal_embedding.float()
        for item in bindings:
            indexes = np.flatnonzero((classes == item.class_id) & (groups != item.group))
            if not len(indexes):
                skipped["no_cross_source_same_class_B"] += 1
                continue
            by_b_group = []
            for group in sorted(set(groups[indexes])):
                positions = torch.as_tensor(indexes[groups[indexes] == group], device=a.device)
                by_b_group.append((1 - (a[item.a_row] * a[positions]).sum(1)).mean())
            d_ab = torch.stack(by_b_group).mean()
            d_an = 1 - (a[item.a_row] * n[item.n_row]).sum()
            loss = torch.relu(d_ab - d_an + margin)
            grouped.setdefault(item.class_id, {}).setdefault(item.group, []).append(loss)
            details.append({"binding_sha256": item.identity, "class_id": item.class_id,
                            "b_groups": len(by_b_group), "d_ab": float(d_ab.detach()),
                            "d_an": float(d_an.detach()), "hinge": float(loss.detach())})
        class_losses = [torch.stack([torch.stack(polygons).mean() for polygons in per_group.values()]).mean()
                        for per_group in grouped.values()]
        loss = torch.stack(class_losses).mean() if class_losses else (a.sum() + n.sum()) * 0
    return loss, {"paired_anchors": len(details), "independent_A_groups": len({b.group for b in bindings
                    if any(d["binding_sha256"] == b.identity for d in details)}),
                  "classes": sorted(grouped), "skipped": dict(skipped), "per_anchor": details,
                  "reduction": "mean_class(mean_A_group(mean_polygon(hinge(mean_B_group_mean_distance-d_AN+margin))))"}


@contextmanager
def auxiliary_batch_statistics(model):
    """Same TRAIN batch-stat semantics as A, without a second running-stat update.

    Buffers are never restored in-place: that would invalidate saved autograd versions.
    All RNG streams and original flags are restored even if the auxiliary forward fails.
    """
    modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    require(modules and all(m.training and m.track_running_stats for m in modules),
            "Auxiliary route requires native TRAIN-mode BatchNorm with tracked statistics")
    before = [{name: None if getattr(m, name) is None else getattr(m, name).detach().clone()
               for name in ("running_mean", "running_var", "num_batches_tracked")} for m in modules]
    flags = [(m.training, m.track_running_stats) for m in modules]
    rng = capture_rng_state()
    try:
        for m in modules:
            m.track_running_stats = False
        yield
    finally:
        for m, (training, tracked) in zip(modules, flags, strict=True):
            m.training, m.track_running_stats = training, tracked
        restore_auxiliary_rng(rng)
        require(all((getattr(m, key) is None and value is None) or
                    (isinstance(getattr(m, key), torch.Tensor) and value is not None
                     and torch.equal(getattr(m, key), value))
                    for m, state in zip(modules, before, strict=True) for key, value in state.items()),
                "Auxiliary route mutated native BatchNorm buffers")


def forward_joint(model, batch, regions, normal_images, bindings, *, sampler_seed=17):
    """Synthetic prototype of the native main route plus shared-head A/B/N loss.

    normal_images is a FULL same-slot counterfactual batch; unpaired slots retain A.
    Real paired augmentations are blocked pending G1 admission and a native replay Gate.
    """
    selected, skipped = select_synthetic_bindings(bindings, regions, seed=sampler_seed)
    require(normal_images.shape == batch["img"].shape and normal_images.device == batch["img"].device
            and normal_images.dtype == batch["img"].dtype == torch.float32
            and bool(torch.isfinite(normal_images).all()) and bool(torch.isfinite(batch["img"]).all())
            and float(normal_images.min())>=0 and float(normal_images.max())<=1,
            "N must retain every native batch slot and spatial geometry")
    slots = regions.slots[regions.cells > 0]
    paired_slots = {int(slots[b.a_row]) for b in selected}
    for slot in range(len(normal_images)):
        if slot not in paired_slots:
            require(torch.equal(normal_images[slot],batch["img"][slot]),
                    "An unpaired/unadmitted N slot changed the counterfactual batch")
    for item in selected:
        slot = int(slots[item.a_row])
        require(tensor_identity(batch["img"][slot]) == item.a_tensor_sha256
                and tensor_identity(normal_images[slot]) == item.n_tensor_sha256,
                "A/N augmented tensor bytes differ from the bound source/Polygon/version")
    features = []
    hook = model.model[P3_LAYER].register_forward_hook(lambda _m, _i, value: features.append(value))
    try:
        native, parts = model(batch)
    finally:
        hook.remove()
    require(len(features) == 1 and features[0].grad_fn is not None, "Missing shared native P3 graph")
    a_feature = features[0]
    a = project_regions(a_feature, batch["masks"], regions, model.region_projection)
    supcon = supervised_contrast(a, regions.classes, regions)  # unchanged Exp15 objective/temperature
    if selected:
        normal_features = []
        hook = model.model[P3_LAYER].register_forward_hook(lambda _m, _i, value: normal_features.append(value))
        try:
            with auxiliary_batch_statistics(model):
                model(normal_images)
        finally:
            hook.remove()
        require(len(normal_features) == 1 and normal_features[0].grad_fn is not None,
                "N lost the shared Backbone gradient path")
        n_feature = normal_features[0]
        n = project_regions(n_feature, batch["masks"], regions, model.region_projection)
        pair, report = source_ranking(a, n, regions, selected)
    else:
        n_feature, n = None, None
        pair = a.sum() * 0
        report = {"paired_anchors": 0, "independent_A_groups": 0, "classes": [], "per_anchor": []}
    return native.sum(), parts, supcon, pair, {**report, "binding_skipped": skipped}, (a_feature, n_feature, a, n)


@dataclass(frozen=True)
class ReplayTrace:
    """Explicit synthetic augmentation contract, not proof of native YOLO replay."""

    source_to_augmented: tuple
    output_hw: tuple
    rgb_gain: tuple = (1.0, 1.0, 1.0)
    rgb_bias: tuple = (0.0, 0.0, 0.0)
    kind: str = "SYNTHETIC_MATRIX_AND_RGB_AFFINE_CONTRACT"

    @property
    def identity(self):
        return digest(asdict(self))


def replay_aligned_pair(a, normal, target_mask, valid_mask, trace):
    """Apply ONE frozen geometry/photo trace to already source-aligned A/N bytes.

    No registration is estimated or historical H changed. A future real loader must
    prove its trace matches the audited native augmentation, including pixel conventions.
    """
    import cv2
    require(a.shape == normal.shape and a.ndim == 3 and a.shape[2] == 3
            and target_mask.shape == valid_mask.shape == a.shape[:2], "A/N/mask source coordinates differ")
    matrix = np.asarray(trace.source_to_augmented, dtype=np.float64).reshape(3, 3)
    require(np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-10
            and len(trace.output_hw) == 2 and min(trace.output_hw) > 0, "Invalid frozen geometric trace")
    height, width = trace.output_hw
    def image(x):
        value = cv2.warpPerspective(x, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))
        return np.clip(value.astype(np.float32)*np.asarray(trace.rgb_gain)+np.asarray(trace.rgb_bias),0,255).astype(np.uint8)
    def mask(x):
        return cv2.warpPerspective(x.astype(np.uint8), matrix, (width,height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    target, valid = mask(target_mask), mask(valid_mask)
    return image(a), image(normal), target, valid, {"trace_sha256": trace.identity,
        "target_pixels": int(target.sum()), "valid_target_pixels": int((target & valid).sum()),
        "support_complete": bool(target.any() and np.all(valid[target])),
        "scope": "SYNTHETIC_CONTRACT_ONLY_NATIVE_AUGMENTATION_REPLAY_REQUIRES_FUTURE_GATE"}
