"""Order-independent ROI composition rules for Exp13 Phase 1-A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import Polygon, build_geometry


@dataclass(frozen=True)
class CompositionItem:
    roi_id: str
    source_id: str
    class_name: str
    polygon: tuple[tuple[float, float], ...]
    risk_flags: tuple[str, ...] = ()


@dataclass
class CompositionResult:
    owner_map: np.ndarray
    edit_support: np.ndarray
    hold: bool
    hold_sources: tuple[str, ...]
    provenance: dict


def _stable_key(item: CompositionItem) -> tuple:
    return (item.source_id, item.roi_id, item.class_name, item.polygon)


def compose_support(items: Iterable[CompositionItem], width: int, height: int, *, context: int = 16, halo_radius: int = 2, unresolved_overlap_policy: str = "deterministic_ownership") -> CompositionResult:
    """Compose masks without execution-order dependence.

    Ownership is selected by stable ROI key among candidates at each pixel. A
    caller may request ``hold_on_overlap`` to conservatively stop unresolved
    conflicts rather than silently treating a risky composition as a PASS.
    """
    ordered = sorted(list(items), key=_stable_key)
    owner = np.full((height, width), "", dtype=object)
    support = np.zeros((height, width), dtype=bool)
    overlap = np.zeros((height, width), dtype=bool)
    for item in ordered:
        geometry = build_geometry(item.polygon, width, height, context=context, halo_radius=halo_radius, roi_id=item.roi_id, source_id=item.source_id)
        candidate = geometry.edit_support
        overlap |= candidate & support
        # Stable lexical owner makes the outcome independent of input order.
        assign = candidate & ((owner == "") | (np.asarray(owner, dtype=object) > item.roi_id))
        owner[assign] = item.roi_id
        support |= candidate
    risky = {item.source_id for item in ordered if item.risk_flags}
    hold_sources: set[str] = set()
    if unresolved_overlap_policy == "hold_on_overlap" and overlap.any():
        hold_sources.update(item.source_id for item in ordered)
    if unresolved_overlap_policy not in {"deterministic_ownership", "hold_on_overlap"}:
        raise ValueError("unknown overlap policy")
    # Explicit risk flags remain visible even when deterministic ownership is usable.
    provenance = {
        "ordering": "stable(source_id, roi_id, class_name, polygon)",
        "ownership": "lexicographically smallest roi_id per overlapping support pixel",
        "overlap_pixels": int(overlap.sum()),
        "risk_sources": sorted(risky),
        "composition_status": "ROI_COMPOSITION_HOLD" if hold_sources else "PASS",
        "non_editable_background_preserved": True,
    }
    return CompositionResult(owner, support, bool(hold_sources), tuple(sorted(hold_sources)), provenance)
