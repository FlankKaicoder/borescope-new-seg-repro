"""CPU geometry for the unchanged native stride-4 mask / stride-8 P3 ROI.

No image, feature, checkpoint or model is loaded here.  A small polygon is
never resized: its native overlap mask supplies fractional P3 coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu_array(value):
    if hasattr(value, "device") and hasattr(value, "detach"):
        _require(value.device.type == "cpu", "Geometry requires an existing CPU tensor")
        value = value.detach().numpy()
    return np.asarray(value)


def array_sha256(value):
    """The same dtype/shape/contiguous-byte convention as EXP17 replay."""
    array = np.ascontiguousarray(_cpu_array(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def native_polygon_area_fraction(normalized_native_coordinates):
    """Area of the frozen YOLO polygon, using native float32 coordinates.

    The return value is a frame fraction, so differently sized source images
    are comparable.  This is not an augmented mask area or a P3 cell count.
    Invalid geometry is rejected rather than clipped, reordered or repaired.
    """
    original = _cpu_array(normalized_native_coordinates)
    _require(original.dtype.kind in "fiu", "Native polygon coordinates must be numeric")
    _require(original.ndim in (1, 2), "Native coordinates must be flat pairs or Nx2")
    if original.ndim == 1:
        _require(original.size >= 6 and original.size % 2 == 0, "Invalid flat polygon coordinates")
        original = original.reshape(-1, 2)
    _require(original.shape[1:] == (2,) and len(original) >= 3, "Polygon requires at least three Nx2 points")
    with np.errstate(over="ignore", invalid="ignore"):
        points = original.astype(np.float32).astype(np.float64)
    _require(np.isfinite(points).all() and ((points >= 0) & (points <= 1)).all(),
             "Frozen native normalized coordinates are outside the source frame")
    x, y = points.T
    area = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
    _require(math.isfinite(area) and 0 < area <= 1, "Native polygon has zero or invalid area")
    return area


def _binary(value, name):
    array = _cpu_array(value)
    _require(array.ndim == 2 and array.size > 0, f"{name} must be a nonempty 2-D mask")
    _require(array.dtype.kind in "biuf" and np.isin(array, [0, 1, 255]).all(),
             f"{name} must be binary, without unknown values")
    return array > 0


def _overlap_array(value, dimensions):
    """Native Format can represent integer instance IDs in float32 masks.

    Validate their values without converting the original array or changing
    its bytes; fractional/unknown values never become repaired instance IDs.
    """
    array = _cpu_array(value)
    _require(array.ndim in dimensions and array.dtype.kind in "iuf",
             "Native overlap masks require numeric integer-valued IDs")
    _require(np.isfinite(array).all() and (array >= 0).all(),
             "Overlap masks contain negative or nonfinite instance IDs")
    if array.dtype.kind == "f":
        _require((array == np.floor(array)).all(), "Overlap masks contain fractional instance IDs")
    return array


def coverage_weights(binary_native_mask):
    """Return raw and normalized 2x2 coverage, never an interpolated ROI.

    For the fixed 640 image this maps native 160x160 masks to 80x80 P3.
    Smaller even synthetic masks are accepted for CPU verification.
    """
    mask = _binary(binary_native_mask, "Native instance mask")
    height, width = mask.shape
    _require(height % 2 == width % 2 == 0, "Native mask dimensions must be divisible by two")
    raw = mask.astype(np.float32).reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
    # Quarter-integer sums are exact at these native sizes, as in the FP32
    # avg_pool2d(binary, 2) / sum(...).clamp_min(.25) implementation in J1.
    mass = raw.sum(dtype=np.float32)
    normalized = raw / np.maximum(mass, np.float32(.25))
    return np.ascontiguousarray(raw), np.ascontiguousarray(normalized)


def stride4_valid_mask(valid_full):
    """All 16 source-grid pixels must be valid in each native mask cell."""
    valid = _binary(valid_full, "Full-resolution Valid")
    height, width = valid.shape
    _require(height % 4 == width % 4 == 0, "Full Valid dimensions must be divisible by four")
    return np.ascontiguousarray(valid.reshape(height // 4, 4, width // 4, 4).all(axis=(1, 3)))


def is_complete_valid_support(binary_native_mask, valid_full):
    mask = _binary(binary_native_mask, "Native instance mask")
    valid_native = stride4_valid_mask(valid_full)
    _require(mask.shape == valid_native.shape, "Native instance / replayed Valid shapes differ")
    return bool(mask.any() and valid_native[mask].all())


@dataclass(frozen=True)
class RegionGeometry:
    cells: int
    native_mask_pixels: int
    coverage_mass: float
    effective_cells: float
    coverage_weights: np.ndarray
    normalized_weights: np.ndarray
    support_complete: bool | None
    mask_sha256: str

    @property
    def p3_cells(self):
        return self.cells

    def to_dict(self):
        """A bounded record; do not repeat 80x80 arrays in every ledger row."""
        return {
            "p3_cells": self.cells,
            "native_mask_pixels": self.native_mask_pixels,
            "coverage_mass": self.coverage_mass,
            "effective_cells": self.effective_cells,
            "support_complete": self.support_complete,
            "mask_sha256": self.mask_sha256,
            "coverage_weights_sha256": array_sha256(self.coverage_weights),
            "normalized_weights_sha256": array_sha256(self.normalized_weights),
            "p3_shape": list(self.coverage_weights.shape),
            "pooling": "NATIVE_OVERLAP_STRIDE4_2X2_COVERAGE_TO_STRIDE8_NO_ROI_RESIZE",
            "valid_contract": "ALL_STRIDE4_BLOCKS_UNDER_ENTIRE_FOREGROUND; NOT_FULL_STRIDE8_OR_RECEPTIVE_FIELD",
        }


def region_geometry(overlap_mask, mask_id, valid_full=None):
    overlap = _overlap_array(overlap_mask, (2,))
    _require(type(mask_id) is int and mask_id > 0, "A positive native mask ID is required")
    binary = overlap == mask_id
    raw, normalized = coverage_weights(binary)
    pixels = int(binary.sum())
    cells = int(np.count_nonzero(raw))
    mass = float(raw.sum(dtype=np.float64))
    squares = float(np.square(normalized.astype(np.float64)).sum())
    effective = 1 / squares if squares else 0.
    complete = None if valid_full is None else is_complete_valid_support(binary, valid_full)
    raw.setflags(write=False)
    normalized.setflags(write=False)
    return RegionGeometry(cells, pixels, mass, effective, raw, normalized, complete,
                          array_sha256(binary))


def native_polygon_geometry(masks, instances_per_image=None, valid_full=None):
    """Map (image slot, native mask ID) to geometry, including missing rasters.

    Pass actual target counts to include polygons whose native raster vanished.
    With counts omitted only mask IDs actually present are described.
    """
    overlap = _overlap_array(masks, (2, 3))
    if overlap.ndim == 2:
        overlap = overlap[None]
    if instances_per_image is None:
        ids = [sorted(int(x) for x in np.unique(mask) if x > 0) for mask in overlap]
    else:
        counts = list(instances_per_image)
        _require(len(counts) == len(overlap) and all(type(n) is int and n >= 0 for n in counts),
                 "Target counts must have one nonnegative integer per image")
        _require(all(not mask.size or int(mask.max()) <= n for mask, n in zip(overlap, counts)),
                 "Native overlap mask contains an unbound target ID")
        ids = [range(1, n + 1) for n in counts]
    validity = None
    if valid_full is not None:
        validity = _cpu_array(valid_full)
        if validity.ndim == 2:
            validity = validity[None]
        _require(validity.ndim == 3 and len(validity) == len(overlap), "Valid batch size differs")
    return {(slot, mask_id): region_geometry(mask, mask_id, None if validity is None else validity[slot])
            for slot, (mask, image_ids) in enumerate(zip(overlap, ids)) for mask_id in image_ids}
