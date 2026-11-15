"""Group-aware supervised contrastive learning on augmented TRAIN polygons."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tools.exp14.audit_admission import require

P3_LAYER = 16
P3_CHANNELS = 64
P3_STRIDE = 8
MASK_RATIO = 4
TEMPERATURE = 0.10


@dataclass
class Regions:
    slots: np.ndarray
    ordinals: np.ndarray
    classes: np.ndarray
    groups: np.ndarray
    cells: np.ndarray
    foreground_fraction: np.ndarray
    max_instances: int

    def pairs(self):
        keep = self.cells > 0
        return pair_masks(self.classes[keep], self.groups[keep])

    def report(self):
        keep = self.cells > 0
        positive, negative, active = self.pairs()
        selected = self.classes[keep]
        return {
            "targets": len(keep), "missing_native_raster": int((~keep).sum()),
            "p3_cells_1": int((self.cells == 1).sum()),
            "p3_cells_2_to_4": int(((self.cells >= 2) & (self.cells <= 4)).sum()),
            "p3_cells_over_4": int((self.cells > 4).sum()),
            "positives": int(positive.sum()) // 2, "negatives": int(negative.sum()) // 2,
            "eligible_anchors": int(active.sum()),
            "box_fraction_q10_q50_q90": np.quantile(self.foreground_fraction[keep], [0.1, 0.5, 0.9]).tolist() if keep.any() else [],
            "targets_by_class": np.bincount(self.classes, minlength=7).tolist(),
            "raster_missing_by_class": np.bincount(self.classes[~keep], minlength=7).tolist(),
            "eligible_anchors_by_class": np.bincount(selected[active], minlength=7).tolist(),
            "eligible_groups_by_class": [len(set(self.groups[keep][active & (selected == c)])) for c in range(7)],
        }


def pair_masks(classes, groups):
    different_group = groups[:, None] != groups[None, :]
    same_class = classes[:, None] == classes[None, :]
    positive = different_group & same_class
    negative = different_group & ~same_class
    return positive, negative, positive.any(axis=1) & negative.any(axis=1)


def inspect_regions(batch, source_groups):
    """Bind overlap-mask ID to the matching, area-ordered native target in each slot."""
    masks = batch["masks"].numpy()
    boxes = batch["bboxes"].numpy()
    cls = batch["cls"].view(-1).numpy()
    slots = batch["batch_idx"].numpy()
    images = batch["im_file"]
    height, width = batch["img"].shape[-2:]
    require(masks.shape == (len(images), height // MASK_RATIO, width // MASK_RATIO)
            and height % P3_STRIDE == width % P3_STRIDE == 0, "Unexpected native TRAIN segmentation mask shape")
    require(slots.shape == cls.shape and boxes.shape == (len(cls), 4)
            and np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            and np.isin(slots, np.arange(len(images))).all() and np.isin(cls, np.arange(7)).all(),
            "Invalid collated ROI coordinates, classes or source slots")
    slots = slots.astype(np.int64)
    classes = cls.astype(np.int64)
    groups = np.asarray([source_groups[str(Path(name).resolve(strict=True))] for name in images])[slots]
    ordinals = np.empty(len(cls), dtype=np.int64)
    cells = np.zeros(len(cls), dtype=np.int64)
    fractions = np.zeros(len(cls), dtype=np.float64)
    max_instances = 0
    mask_h, mask_w = masks.shape[-2:]
    for slot, mask in enumerate(masks):
        indexes = np.flatnonzero(slots == slot)
        ordinals[indexes] = np.arange(len(indexes))
        max_instances = max(max_instances, len(indexes))
        require(np.isin(mask, np.arange(len(indexes) + 1)).all(), "Native overlap mask has an unbound instance ID")
        for local, index in enumerate(indexes, 1):
            y, x = np.nonzero(mask == local)
            if not len(x):  # Thin/occluded polygons can disappear at the native raster resolution.
                continue
            cx, cy, w, h = boxes[index] * [mask_w, mask_h, mask_w, mask_h]
            require(x.min() >= cx - w / 2 - 2 and x.max() <= cx + w / 2 + 2
                    and y.min() >= cy - h / 2 - 2 and y.max() <= cy + h / 2 + 2,
                    "Augmented raster pixels escape their matching native target box")
            cells[index] = len(np.unique((y // 2) * (mask_w // 2) + x // 2))
            fractions[index] = len(x) / (max(1, np.ceil(w)) * max(1, np.ceil(h)))
    return Regions(slots, ordinals, classes, groups, cells, fractions, max_instances)


def project_regions(feature, masks, regions, head):
    """Weight each P3 cell by the fraction of its four native mask pixels in the polygon."""
    shape = (masks.shape[0], P3_CHANNELS, masks.shape[-2] // 2, masks.shape[-1] // 2)
    require(feature.shape == shape, f"Unexpected fused P3 dimensions: {tuple(feature.shape)} vs {shape}")
    device = feature.device
    ids = torch.arange(1, regions.max_instances + 1, device=device)
    with torch.autocast(device.type, enabled=False):
        binary = (masks[:, None] == ids[None, :, None, None]).float()
        weights = F.avg_pool2d(binary, 2).flatten(2)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(0.25)
        pooled = weights.bmm(feature.float().flatten(2).transpose(1, 2))
        keep = regions.cells > 0
        rows = torch.as_tensor(regions.slots[keep], device=device)
        cols = torch.as_tensor(regions.ordinals[keep], device=device)
        return F.normalize(head(pooled[rows, cols]), dim=1)


def supervised_contrast(embedding, classes, regions):
    positive, negative, active = regions.pairs()
    require(active.any(), "TRAIN batch has no cross-group positive anchor with a negative")
    dev = embedding.device
    with torch.autocast(dev.type, enabled=False):
        logits = embedding.float() @ embedding.float().T / TEMPERATURE
        p = torch.as_tensor(positive[active], device=dev)
        candidates = torch.as_tensor((positive | negative)[active], device=dev)
        logits = logits[torch.as_tensor(active, device=dev)]
        per_anchor = logits.masked_fill(~candidates, -torch.inf).logsumexp(dim=1)
        per_anchor = per_anchor - logits.masked_fill(~p, 0).sum(dim=1) / p.sum(dim=1)
        active_classes = classes[regions.cells > 0][active]
        inverse_count = 1 / np.bincount(active_classes, minlength=7)[active_classes]
        weights = torch.as_tensor(inverse_count / inverse_count.sum(), dtype=torch.float32, device=dev)
        return (weights * per_anchor).sum()


def add_projection(model):
    # Initialization of J1-only parameters must leave the native RNG untouched.
    with torch.random.fork_rng(devices=[]):
        model.add_module("region_projection", nn.Sequential(nn.Linear(P3_CHANNELS, 128), nn.ReLU(), nn.Linear(128, 128)))


def forward(model, batch, regions=None, read_bn=None):
    if regions is None:
        native, parts = model(batch)
        return native.sum(), parts, None
    features = []
    handle = model.model[P3_LAYER].register_forward_hook(lambda _m, _i, out: features.append(out))
    try:
        native, parts = model(batch)
    finally:
        handle.remove()
    require(len(features) == 1 and features[0].grad_fn is not None, "Missing gradient-enabled fused P3 feature")
    bn_native = read_bn() if read_bn else None
    embedding = project_regions(features[0], batch["masks"], regions, model.region_projection)
    aux = supervised_contrast(embedding, regions.classes, regions)
    if read_bn:
        require(read_bn() == bn_native, "Auxiliary computation changed native BatchNorm state")
    return native.sum(), parts, aux

