"""Ordinary region CE on exactly the frozen J1 polygon/anchor exposure."""

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
    """Average classes equally, then average the same eligible anchors as J1."""
    _, _, active = regions.pairs()
    require(active.any() and embedding.shape == (int((regions.cells > 0).sum()), 128),
            "No eligible CE regions or unexpected projected feature shape")
    labels = regions.classes[regions.cells > 0][active]
    weights = 1 / np.bincount(labels, minlength=7)[labels]
    with torch.autocast(embedding.device.type, enabled=False):
        logits = classifier(embedding.float()[torch.as_tensor(active, device=embedding.device)])
        loss = F.cross_entropy(logits, torch.as_tensor(labels, device=embedding.device), reduction="none")
        weight = torch.as_tensor(weights / weights.sum(), dtype=torch.float32, device=embedding.device)
        return (loss * weight).sum()


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
