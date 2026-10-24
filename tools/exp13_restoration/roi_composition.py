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


@dataclass(frozen=True)
class RGBROIEdit:
    """One deterministic native-ROI RGB edit returned by a model or mock."""
    roi_id: str
    source_id: str
    polygon: tuple[tuple[float, float], ...]
    mock_rgb: np.ndarray
    roi_support_mask: np.ndarray | None = None
    risk_flags: tuple[str, ...] = ()


@dataclass
class FullImageCompositionResult:
    image: np.ndarray
    owner_map: np.ndarray
    edit_support: np.ndarray
    overlap_pixels: int
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


def compose_full_image(
    native_rgb: np.ndarray,
    source_id: str,
    edits: Iterable[RGBROIEdit],
    *,
    context: int = 16,
    halo_radius: int = 2,
    unresolved_overlap_policy: str = "deterministic_ownership",
) -> FullImageCompositionResult:
    """Paste every native ROI back into one RGB image with immutable background."""
    image = np.asarray(native_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("native_rgb must be HxWx3 uint8")
    h, w, _ = image.shape
    ordered = sorted(list(edits), key=lambda e: (e.source_id, e.roi_id, e.polygon))
    if any(edit.source_id != source_id for edit in ordered):
        raise ValueError("cross-source ROI composition is forbidden")
    owner = np.full((h, w), "", dtype=object)
    support = np.zeros((h, w), dtype=bool)
    overlap = np.zeros((h, w), dtype=bool)
    geometries = []
    for edit in ordered:
        geom = build_geometry(edit.polygon, w, h, context=context, halo_radius=halo_radius, roi_id=edit.roi_id, source_id=edit.source_id)
        mask = geom.edit_support if edit.roi_support_mask is None else np.asarray(edit.roi_support_mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"ROI {edit.roi_id} support mask shape mismatch")
        box = geom.roi_box
        mock = np.asarray(edit.mock_rgb)
        if mock.shape != (box.height, box.width, 3) or mock.dtype != np.uint8:
            raise ValueError(f"ROI {edit.roi_id} mock RGB must match native crop and be uint8")
        if not np.all(mask[geom.polygon_mask]):
            raise ValueError(f"ROI {edit.roi_id} support does not cover target polygon")
        overlap |= mask & support
        assign = mask & ((owner == "") | (np.asarray(owner, dtype=object) > edit.roi_id))
        owner[assign] = edit.roi_id
        support |= mask
        geometries.append((edit, geom, mask, mock))
    hold_sources = set()
    if unresolved_overlap_policy == "hold_on_overlap" and overlap.any():
        hold_sources.add(source_id)
    if unresolved_overlap_policy not in {"deterministic_ownership", "hold_on_overlap"}:
        raise ValueError("unknown overlap policy")
    out = image.copy()
    if not hold_sources:
        for edit, geom, mask, mock in geometries:
            box = geom.roi_box
            local_owner = owner[box.y0:box.y1, box.x0:box.x1] == edit.roi_id
            local_mask = mask[box.y0:box.y1, box.x0:box.x1] & local_owner
            crop = out[box.y0:box.y1, box.x0:box.x1]
            crop[local_mask] = mock[local_mask]
    immutable = ~support
    if not np.array_equal(out[immutable], image[immutable]):
        raise AssertionError("pixels outside edit support changed")
    provenance = {
        "source_id": source_id,
        "pipeline": "Native RGB -> ROI Crop -> Mock Canvas -> Mock ROI RGB -> Native ROI -> Full Image",
        "roi_count": len(ordered),
        "overlap_pixels": int(overlap.sum()),
        "owner_map_values": sorted(str(x) for x in np.unique(owner) if x),
        "composition_status": "ROI_COMPOSITION_HOLD" if hold_sources else "PASS",
        "background_immutable": bool(np.array_equal(out[immutable], image[immutable])),
    }
    return FullImageCompositionResult(out, owner, support, int(overlap.sum()), bool(hold_sources), tuple(sorted(hold_sources)), provenance)
