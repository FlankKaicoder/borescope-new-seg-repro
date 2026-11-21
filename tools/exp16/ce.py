"""Ordinary class-mean CE on all valid frozen J1 pooled Polygon regions."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tools.exp14.audit_admission import require
from tools.exp15.regions import P3_CHANNELS, P3_LAYER, project_regions

HEAD_PREFIX = "region_projection."
HEAD_TENSORS = 6


def add_head(model):
    # The first three modules match J1's projection. The classifier consumes the
    # same normalized 128-D representation. No BN, dropout or new RNG stream.
    with torch.random.fork_rng(devices=[]):
        model.add_module("region_projection", nn.Sequential(
            nn.Linear(P3_CHANNELS, 128), nn.ReLU(), nn.Linear(128, 128), nn.Linear(128, 7)))


def region_cross_entropy(embedding, regions, classifier):
    """Average classes equally over all valid regions, without pair eligibility."""
    labels = regions.classes[regions.cells > 0]
    require(len(labels) > 0 and embedding.shape == (len(labels), 128),
            "No valid CE regions or unexpected projected feature shape")
    weights = 1 / np.bincount(labels, minlength=7)[labels]
    with torch.autocast(embedding.device.type, enabled=False):
        logits = classifier(embedding.float())
        loss = F.cross_entropy(logits, torch.as_tensor(labels, device=embedding.device), reduction="none")
        weight = torch.as_tensor(weights / weights.sum(), dtype=torch.float32, device=embedding.device)
        return (loss * weight).sum()


def coverage(regions):
    keep = regions.cells > 0
    classes, groups = regions.classes[keep], regions.groups[keep]
    return {"valid_regions": int(keep.sum()),
            "ce_regions_by_class": np.bincount(classes, minlength=7).tolist(),
            "ce_groups_by_class": [len(set(groups[classes == c])) for c in range(7)]}


def summarize_coverage(rows):
    return {"batches": len(rows), "zero_valid_batches": sum(r["valid_regions"] == 0 for r in rows),
            "valid_regions": sum(r["valid_regions"] for r in rows),
            "ce_regions_by_class": np.sum([r["ce_regions_by_class"] for r in rows], axis=0).tolist(),
            "ce_group_exposures_by_class": np.sum([r["ce_groups_by_class"] for r in rows], axis=0).tolist(),
            "batches_with_regions_by_class": np.count_nonzero([r["ce_regions_by_class"] for r in rows], axis=0).tolist()}


def forward(model, batch, regions, read_bn=None):
    features = []
    handle = model.model[P3_LAYER].register_forward_hook(lambda _m, _i, value: features.append(value))
    try:
        native, parts = model(batch)
    finally:
        handle.remove()
    require(len(features) == 1 and features[0].grad_fn is not None,
            "Missing gradient-enabled fused P3 feature")
    bn_native = read_bn() if read_bn else None
    embedding = project_regions(features[0], batch["masks"], regions, model.region_projection[:3])
    auxiliary = region_cross_entropy(embedding, regions, model.region_projection[3])
    if read_bn:
        require(read_bn() == bn_native, "CE region computation changed native BatchNorm state")
    return native.sum(), parts, auxiliary
