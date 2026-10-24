"""Order-independent ROI composition rules for Exp13 Phase 1-A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from .geometry import Polygon, GeometryBundle, build_geometry, inverse_canvas_mask, map_mask_to_canvas


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


@dataclass(frozen=True)
class CanvasMockResult:
    """Deterministic non-identity canvas edit and its auditable masks."""
    native_roi_rgb: np.ndarray
    canvas_rgb: np.ndarray
    mock_canvas_rgb: np.ndarray
    canvas_polygon_mask: np.ndarray
    canvas_padding_mask: np.ndarray
    native_changed_mask: np.ndarray
    native_polygon_mask: np.ndarray
    provenance: dict


def deterministic_canvas_mock(
    native_roi_rgb: np.ndarray,
    geometry: GeometryBundle,
    *,
    padding_rgb: tuple[int, int, int] = (0, 0, 0),
    xor_value: int = 0x5A,
) -> CanvasMockResult:
    """Run ROI crop -> 1024 letterbox -> non-identity mock -> inverse map."""
    crop = np.asarray(native_roi_rgb)
    transform = geometry.transform
    if crop.shape != (transform.native_height, transform.native_width, 3) or crop.dtype != np.uint8:
        raise ValueError("native ROI must match geometry dimensions and be uint8 RGB")
    resized = np.asarray(Image.fromarray(crop, mode="RGB").resize((transform.resized_width, transform.resized_height), Image.Resampling.BILINEAR), dtype=np.uint8)
    canvas = np.empty((transform.canvas_size, transform.canvas_size, 3), dtype=np.uint8)
    canvas[...] = np.asarray(padding_rgb, dtype=np.uint8)
    y0, y1 = transform.pad_top, transform.pad_top + transform.resized_height
    x0, x1 = transform.pad_left, transform.pad_left + transform.resized_width
    canvas[y0:y1, x0:x1] = resized
    crop_polygon = geometry.polygon_mask[geometry.roi_box.y0:geometry.roi_box.y1, geometry.roi_box.x0:geometry.roi_box.x1]
    canvas_polygon = map_mask_to_canvas(crop_polygon, transform)
    content = np.zeros((transform.canvas_size, transform.canvas_size), dtype=bool)
    content[y0:y1, x0:x1] = True
    padding = ~content
    mock_canvas = canvas.copy()
    mock_canvas[canvas_polygon] ^= np.uint8(xor_value)
    yy = np.minimum(transform.canvas_size - 1, transform.pad_top + ((np.arange(transform.native_height) + 0.5) * transform.scale - 0.5)).astype(int)
    xx = np.minimum(transform.canvas_size - 1, transform.pad_left + ((np.arange(transform.native_width) + 0.5) * transform.scale - 0.5)).astype(int)
    yy = np.maximum(yy, 0); xx = np.maximum(xx, 0)
    inverse = mock_canvas[np.ix_(yy, xx)]
    inverse_mask = inverse_canvas_mask(canvas_polygon, transform)
    native_out = crop.copy()
    native_out[inverse_mask] = inverse[inverse_mask]
    changed = np.any(native_out != crop, axis=2)
    return CanvasMockResult(
        native_roi_rgb=native_out,
        canvas_rgb=canvas,
        mock_canvas_rgb=mock_canvas,
        canvas_polygon_mask=canvas_polygon,
        canvas_padding_mask=padding,
        native_changed_mask=changed,
        native_polygon_mask=crop_polygon,
        provenance={
            "pipeline": "Native RGB -> ROI Crop -> Aspect-preserving Resize -> 1024 Canvas -> Non-identity Mock Canvas -> Inverse Mapping -> Native ROI RGB -> Full Image",
            "padding_rgb": list(padding_rgb),
            "padding_edited_pixels": int(np.count_nonzero(np.any(mock_canvas[padding] != canvas[padding], axis=1))),
            "canvas_polygon_pixels": int(canvas_polygon.sum()),
            "native_polygon_pixels": int(crop_polygon.sum()),
            "native_changed_pixels": int(np.count_nonzero(changed & crop_polygon)),
            "mask_rgb_space_synchronized": bool(np.all(~changed | inverse_mask)),
        },
    )


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
