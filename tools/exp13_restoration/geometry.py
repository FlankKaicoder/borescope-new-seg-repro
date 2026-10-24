"""Deterministic native-ROI geometry for Exp13 Phase 1-A.

This module contains only geometry and mask bookkeeping.  It deliberately has
no model or image-generation dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Polygon:
    roi_id: str
    points: tuple[tuple[float, float], ...]
    class_name: str = "unknown"
    source_id: str = "unknown"


@dataclass(frozen=True)
class ROIBox:
    x0: int
    y0: int
    x1: int
    y1: int
    image_width: int
    image_height: int
    context: int = 0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class CanvasTransform:
    native_width: int
    native_height: int
    canvas_size: int = 1024
    scale: float = 1.0
    resized_width: int = 1
    resized_height: int = 1
    pad_left: int = 0
    pad_top: int = 0
    resize_interpolation: str = "lanczos_image_nearest_mask"
    mask_interpolation: str = "nearest"
    padding_mode: str = "reflect_or_constant_declared_by_runner"

    @property
    def pad_right(self) -> int:
        return self.canvas_size - self.pad_left - self.resized_width

    @property
    def pad_bottom(self) -> int:
        return self.canvas_size - self.pad_top - self.resized_height


@dataclass
class GeometryBundle:
    polygon_mask: np.ndarray
    edit_core: np.ndarray
    transition_halo: np.ndarray
    edit_support: np.ndarray
    non_editable_background: np.ndarray
    fov_uncertain: np.ndarray
    roi_box: ROIBox
    transform: CanvasTransform
    provenance: dict = field(default_factory=dict)


def _as_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("polygon must contain at least three (x, y) points")
    if not np.isfinite(arr).all():
        raise ValueError("polygon points must be finite")
    return arr


def polygon_mask(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """Rasterize a polygon using pixel-center ray casting, deterministically."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    p = _as_points(points)
    # Pixel centers keep thin, non-integer polygons deterministic and avoid
    # dependence on a graphics backend.
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(float) + 0.5
    y = yy.astype(float) + 0.5
    inside = np.zeros((height, width), dtype=bool)
    x0, y0 = p[-1]
    for x1, y1 in p:
        cross = ((y1 > y) != (y0 > y))
        denom = y0 - y1
        safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        intersects = x < (x0 - x1) * (y - y1) / safe + x1
        inside ^= cross & intersects
        x0, y0 = x1, y1
    return inside


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Small dependency-free square dilation used for transition halos."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        ys = max(0, dy)
        ye = min(h, h + dy)
        ysrc = slice(max(0, -dy), min(h, h - dy))
        for dx in range(-radius, radius + 1):
            xs = max(0, dx)
            xe = min(w, w + dx)
            xsrc = slice(max(0, -dx), min(w, w - dx))
            out[ys:ye, xs:xe] |= mask[ysrc, xsrc]
    return out


def bbox(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    p = _as_points(points)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def compute_roi_box(points: Sequence[Sequence[float]], image_width: int, image_height: int, context: int = 16) -> ROIBox:
    """Return a context box with explicit image-boundary clamping."""
    x0, y0, x1, y1 = bbox(points)
    left = max(0, int(np.floor(x0)) - int(context))
    top = max(0, int(np.floor(y0)) - int(context))
    right = min(image_width, int(np.ceil(x1)) + 1 + int(context))
    bottom = min(image_height, int(np.ceil(y1)) + 1 + int(context))
    if right <= left or bottom <= top:
        raise ValueError("polygon has no positive-area clamped ROI")
    return ROIBox(left, top, right, bottom, image_width, image_height, int(context))


def canvas_transform(box: ROIBox, canvas_size: int = 1024) -> CanvasTransform:
    if canvas_size <= 0:
        raise ValueError("canvas size must be positive")
    scale = min(canvas_size / box.width, canvas_size / box.height)
    rw = max(1, int(round(box.width * scale)))
    rh = max(1, int(round(box.height * scale)))
    # Correct rounding at the edge while preserving the declared aspect policy.
    rw, rh = min(rw, canvas_size), min(rh, canvas_size)
    return CanvasTransform(
        native_width=box.width,
        native_height=box.height,
        canvas_size=canvas_size,
        scale=scale,
        resized_width=rw,
        resized_height=rh,
        pad_left=(canvas_size - rw) // 2,
        pad_top=(canvas_size - rh) // 2,
    )


def map_mask_to_canvas(mask: np.ndarray, transform: CanvasTransform) -> np.ndarray:
    """Nearest-neighbor map, implemented without an image/model dependency."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (transform.native_height, transform.native_width):
        raise ValueError("mask shape does not match transform native dimensions")
    y = np.minimum(transform.native_height - 1, (np.arange(transform.resized_height) + 0.5) / transform.scale - 0.5).astype(int)
    x = np.minimum(transform.native_width - 1, (np.arange(transform.resized_width) + 0.5) / transform.scale - 0.5).astype(int)
    y = np.maximum(y, 0)
    x = np.maximum(x, 0)
    out = np.zeros((transform.canvas_size, transform.canvas_size), dtype=bool)
    out[transform.pad_top:transform.pad_top + transform.resized_height, transform.pad_left:transform.pad_left + transform.resized_width] = mask[np.ix_(y, x)]
    return out


def inverse_canvas_mask(mask: np.ndarray, transform: CanvasTransform) -> np.ndarray:
    """Map a canvas mask back to native crop coordinates with nearest sampling."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (transform.canvas_size, transform.canvas_size):
        raise ValueError("canvas mask has unexpected shape")
    y = np.minimum(transform.canvas_size - 1, transform.pad_top + ((np.arange(transform.native_height) + 0.5) * transform.scale - 0.5)).astype(int)
    x = np.minimum(transform.canvas_size - 1, transform.pad_left + ((np.arange(transform.native_width) + 0.5) * transform.scale - 0.5)).astype(int)
    y = np.maximum(y, 0)
    x = np.maximum(x, 0)
    return mask[np.ix_(y, x)]


def build_geometry(points: Sequence[Sequence[float]], image_width: int, image_height: int, *, context: int = 16, halo_radius: int = 2, canvas_size: int = 1024, gray_border_width: int = 0, roi_id: str = "unknown", source_id: str = "unknown") -> GeometryBundle:
    """Build explicit edit masks; gray borders are flagged, never auto-edited."""
    full = polygon_mask(points, image_width, image_height)
    box = compute_roi_box(points, image_width, image_height, context=context)
    halo = dilate(full, halo_radius) & ~full
    support = full | halo
    fov = np.zeros_like(full)
    if gray_border_width > 0:
        # This is an uncertainty marker only.  It does not change edit support.
        fov[:gray_border_width, :] = True
        fov[-gray_border_width:, :] = True
        fov[:, :gray_border_width] = True
        fov[:, -gray_border_width:] = True
    # A clamped ROI is geometric evidence of a border risk, not padding proof.
    clamped = box.x0 == 0 or box.y0 == 0 or box.x1 == image_width or box.y1 == image_height
    transform = canvas_transform(box, canvas_size)
    return GeometryBundle(
        polygon_mask=full,
        edit_core=full,
        transition_halo=halo,
        edit_support=support,
        non_editable_background=~support,
        fov_uncertain=fov,
        roi_box=box,
        transform=transform,
        provenance={
            "roi_id": roi_id,
            "source_id": source_id,
            "polygon_coverage": 1.0,
            "halo_policy": "dilate_outside_polygon_only",
            "gray_padding_status": "UNRESOLVED_HEURISTIC",
            "fov_status": "FOV_UNCERTAIN" if clamped or gray_border_width else "UNASSESSED",
            "roi_clamped": clamped,
            "resize_interpolation": transform.resize_interpolation,
            "mask_interpolation": transform.mask_interpolation,
        },
    )


def _bbox_intersects(a: Polygon, b: Polygon, gap: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = bbox(a.points)
    bx0, by0, bx1, by1 = bbox(b.points)
    return not (ax1 + gap < bx0 or bx1 + gap < ax0 or ay1 + gap < by0 or by1 + gap < ay0)


def group_polygons(polygons: Iterable[Polygon], gap: float = 0.0) -> list[list[Polygon]]:
    """Connected components over bbox proximity, independent of input order."""
    items = sorted(list(polygons), key=lambda p: (p.source_id, p.roi_id, p.class_name, p.points))
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_intersects(items[i], items[j], gap=gap):
                union(i, j)
    groups: dict[int, list[Polygon]] = {}
    for i, poly in enumerate(items):
        groups.setdefault(find(i), []).append(poly)
    return [sorted(v, key=lambda p: (p.source_id, p.roi_id)) for _, v in sorted(groups.items())]
