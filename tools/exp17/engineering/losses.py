"""Thin sparse-N adapter around the frozen J1 and source-ranking definitions."""
from dataclasses import dataclass
import math

import numpy as np
import torch

from tools.exp15 import regions as j1
from tools.exp17.pair_loss import MARGIN, SUPCON_WEIGHT, source_ranking


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


@dataclass(frozen=True)
class RankingRow:
    a_row: int
    n_row: int
    group: str
    class_id: int
    identity: str


def sparse_source_ranking(a_embedding, n_embedding, a_regions, n_regions, bindings):
    """Scatter only bound N rows to A's row domain; reduction remains frozen.

    Zero placeholders are never ranked, never treated as normal samples and never
    forwarded as images. The index_copy remains differentiable through sparse N.
    """
    require(a_embedding.ndim == n_embedding.ndim == 2 and a_embedding.shape[1] == n_embedding.shape[1], "A/N embedding width differs")
    a_keep, n_keep = a_regions.cells > 0, n_regions.cells > 0
    require(len(a_embedding) == int(a_keep.sum()) and len(n_embedding) == int(n_keep.sum()), "Embedding/native region row counts differ")
    require(a_embedding.device == n_embedding.device, "A/N embedding device differs")
    a_rows, n_rows, ranking, seen_a, seen_n = [], [], [], set(), set()
    ac, ag = a_regions.classes[a_keep], a_regions.groups[a_keep]
    nc, ng = n_regions.classes[n_keep], n_regions.groups[n_keep]
    for item in bindings:
        require(type(item.a_row) is type(item.n_row) is type(item.class_id) is int, "Pair row or class identity has wrong type")
        require(0 <= item.a_row < len(a_embedding) and 0 <= item.n_row < len(n_embedding), "Sparse binding row is out of bounds")
        require(item.a_row not in seen_a and item.n_row not in seen_n, "Duplicate A or N Polygon binding")
        require(int(ac[item.a_row]) == int(nc[item.n_row]) == item.class_id
                and str(ag[item.a_row]) == str(ng[item.n_row]) == item.group, "Sparse A/N group or class is misbound")
        seen_a.add(item.a_row); seen_n.add(item.n_row)
        a_rows.append(item.a_row); n_rows.append(item.n_row)
        ranking.append(RankingRow(item.a_row, item.a_row, item.group, item.class_id, item.identity))
    aligned = a_embedding.new_zeros(a_embedding.shape)
    if ranking:
        a_index = torch.as_tensor(a_rows, device=a_embedding.device)
        n_index = torch.as_tensor(n_rows, device=a_embedding.device)
        aligned = aligned.index_copy(0, a_index, n_embedding.index_select(0, n_index))
    else:
        aligned = aligned + n_embedding.sum() * 0
    loss, report = source_ranking(a_embedding, aligned, a_regions, ranking, margin=MARGIN)
    return loss, {**report, "normal_row_domain": "differentiable_sparse_N_to_A_index_copy",
                  "sparse_normal_regions": len(n_embedding), "bound_normal_regions": len(ranking),
                  "margin": MARGIN, "supcon_weight": SUPCON_WEIGHT, "supcon_temperature": j1.TEMPERATURE}


def joint_loss(yolo, supcon, pair, lambda_pair):
    require(type(lambda_pair) in (int, float) and math.isfinite(lambda_pair) and lambda_pair >= 0,
            "A fixed, explicit finite nonnegative lambda_pair is required")
    require(all(value.ndim == 0 and bool(torch.isfinite(value.detach())) for value in (yolo, supcon, pair)), "Nonfinite or nonscalar joint loss component")
    return yolo + SUPCON_WEIGHT * supcon + lambda_pair * pair


def per_class_pair_losses(a_embedding, n_embedding, a_regions, n_regions, bindings):
    """Represented-class tensors using exactly the same frozen reduction.

    Filtering only the A bindings does not filter the native batch's same-class B
    regions. Their arithmetic mean reproduces the global Pair loss; no new forward.
    """
    values = {}
    for class_id in dict.fromkeys(item.class_id for item in bindings):
        selected = [item for item in bindings if item.class_id == class_id]
        value, report = sparse_source_ranking(a_embedding, n_embedding, a_regions, n_regions, selected)
        if report["classes"]: values[class_id] = value
    return values
