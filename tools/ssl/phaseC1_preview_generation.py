#!/usr/bin/env python3
"""Phase C-1-B morphology-aware Local Change preview generation.

This tool is preview-only. It never trains, never reads labels, and never
touches VAL or TEST. Its purpose is to inspect the redesigned synthetic change
distribution before any proxy smoke run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
FAMILIES = ["A_small_low_contrast", "B_diffuse_texture", "C_elongated_ribbon", "D_irregular_boundary"]
FAMILY_PROBABILITY = {
    "A_small_low_contrast": 0.30,
    "B_diffuse_texture": 0.30,
    "C_elongated_ribbon": 0.25,
    "D_irregular_boundary": 0.15,
}
FAMILY_RANGES = {
    "A_small_low_contrast": {"area": (0.0005, 0.0030), "aspect": (1.0, 2.0), "vertices": (10, 16)},
    "B_diffuse_texture": {"area": (0.0020, 0.1700), "aspect": (1.0, 4.0), "vertices": (16, 32)},
    "C_elongated_ribbon": {"area": (0.0005, 0.0180), "aspect": (2.0, 8.0), "vertices": (40, 80)},
    "D_irregular_boundary": {"area": (0.0030, 0.1000), "aspect": (1.0, 4.0), "vertices": (24, 48)},
}
MAX_ATTEMPTS = 5


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def relative_to_root(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def grayscale_array(image_array: np.ndarray) -> np.ndarray:
    return np.dot(image_array[..., :3].astype(np.float32), [0.299, 0.587, 0.114])


def count_changed_pixels(reference: np.ndarray, current: np.ndarray, threshold: float = 2.0) -> int:
    difference = np.abs(grayscale_array(reference) - grayscale_array(current))
    return int(np.count_nonzero(difference >= threshold))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_train_stems(path: Path) -> set[str]:
    stems = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = (row.get("split") or "").strip().lower()
            stem = (row.get("stem") or "").strip()
            if split == "train":
                if not stem:
                    raise RuntimeError("MANIFEST_EMPTY_TRAIN_STEM")
                if stem in stems:
                    raise RuntimeError(f"MANIFEST_DUPLICATE_TRAIN_STEM: {stem}")
                stems.add(stem)
    return stems


def collect_train_images(
    train_dir: Path,
    expected_stems: set[str],
    approved_source_root: Path,
    approved_physical_source_root: Path,
) -> list[Path]:
    resolved_dir = train_dir.resolve()
    if resolved_dir.name != "train" or resolved_dir.parent.name != "images":
        raise RuntimeError(f"TRAIN_DIR_GATE: {resolved_dir}")
    if not approved_source_root.is_dir():
        raise RuntimeError(f"APPROVED_SOURCE_ROOT_NOT_FOUND: {approved_source_root}")
    if not approved_physical_source_root.is_dir():
        raise RuntimeError(f"APPROVED_PHYSICAL_SOURCE_ROOT_NOT_FOUND: {approved_physical_source_root}")
    if not path_is_within(resolved_dir, approved_source_root):
        raise RuntimeError(f"TRAIN_DIR_OUTSIDE_APPROVED_SOURCE_ROOT: {resolved_dir}")

    paths = []
    seen = set()
    for path in resolved_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path_is_within(path.parent, resolved_dir):
            raise RuntimeError(f"SOURCE_NOT_IN_TRAIN_DIRECTORY: {path}")
        resolved = path.resolve()
        lowered_parts = [part.lower() for part in resolved.parts]
        if "val" in lowered_parts or "test" in lowered_parts:
            raise RuntimeError(f"FORBIDDEN_SPLIT_PATH: {resolved}")
        if not (
            path_is_within(resolved, approved_source_root)
            or path_is_within(resolved, approved_physical_source_root)
        ):
            raise RuntimeError(f"SOURCE_OUTSIDE_APPROVED_ROOTS: {resolved}")
        stem = path.stem
        if stem in seen:
            raise RuntimeError(f"DUPLICATE_STEM: {stem}")
        seen.add(stem)
        paths.append(resolved)

    if len(paths) != 668:
        raise RuntimeError(f"TRAIN_IMAGE_COUNT_GATE: {len(paths)} != 668")
    if seen != expected_stems:
        missing = sorted(expected_stems - seen)
        extra = sorted(seen - expected_stems)
        raise RuntimeError(f"TRAIN_STEM_SET_GATE: missing={missing[:10]} extra={extra[:10]}")
    return sorted(paths, key=lambda path: path.stem)


def letterbox(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def shared_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image
    if rng.random() < 0.50:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.15:
        result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if rng.random() < 0.50:
        array = np.asarray(result, dtype=np.float32)
        gain = rng.uniform(0.90, 1.10)
        bias = rng.uniform(-8.0, 8.0)
        array = np.clip(array * gain + bias, 0, 255).astype(np.uint8)
        result = Image.fromarray(array)
    return result


def scale_points(points: list[tuple[float, float]], target_area: float) -> list[tuple[float, float]]:
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area2 += x1 * y2 - x2 * y1
    area = abs(area2) / 2.0
    if area <= 0:
        return points
    scale = math.sqrt(target_area / area)
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in points]


def clamp_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    return [
        (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0))
        for x, y in points
    ]


def blob_points(
    rng: random.Random,
    width: int,
    height: int,
    target_area: float,
    aspect: float,
    vertex_count: int,
    jitter_low: float,
    jitter_high: float,
) -> list[tuple[float, float]]:
    base_radius = math.sqrt(target_area / math.pi)
    rx = base_radius * math.sqrt(aspect)
    ry = base_radius / math.sqrt(aspect)
    rotation = rng.uniform(0.0, 2.0 * math.pi)
    cx = rng.uniform(max(rx, 4.0), max(width - rx - 1.0, 5.0))
    cy = rng.uniform(max(ry, 4.0), max(height - ry - 1.0, 5.0))
    harmonic_phase = rng.uniform(0.0, 2.0 * math.pi)
    harmonic_amplitude = rng.uniform(0.08, 0.28)
    points = []
    for index in range(vertex_count):
        angle = rotation + 2.0 * math.pi * index / vertex_count
        jitter = rng.uniform(jitter_low, jitter_high)
        harmonic = 1.0 + harmonic_amplitude * math.sin(3.0 * angle + harmonic_phase)
        radius_factor = jitter * harmonic
        x = cx + rx * radius_factor * math.cos(angle)
        y = cy + ry * radius_factor * math.sin(angle)
        points.append((x, y))
    points = scale_points(points, target_area)
    return clamp_points(points, width, height)


def ribbon_points(
    rng: random.Random,
    width: int,
    height: int,
    target_area: float,
    aspect: float,
) -> list[tuple[float, float]]:
    length = math.sqrt(max(target_area * aspect, 1.0))
    thickness = max(2.0, target_area / max(length, 1.0))
    angle = rng.uniform(0.0, math.pi)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    margin = max(length / 2.0, thickness) + 4.0
    cx = rng.uniform(margin, max(width - margin, margin + 1.0))
    cy = rng.uniform(margin, max(height - margin, margin + 1.0))
    sample_count = rng.randint(40, 80)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    amplitude = min(length * 0.18, 35.0)
    centerline = []
    for index in range(sample_count):
        t = (index / (sample_count - 1) - 0.5) * length
        lateral = amplitude * math.sin(2.0 * math.pi * (index / sample_count) + phase)
        point = np.array([cx, cy]) + direction * t + normal * lateral
        centerline.append(point)
    left = []
    right = []
    half_thickness = thickness / 2.0
    for index, point in enumerate(centerline):
        if index == 0:
            tangent = centerline[1] - centerline[0]
        else:
            tangent = centerline[index] - centerline[index - 1]
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            local_normal = normal
        else:
            local_normal = np.array([-tangent[1], tangent[0]]) / norm
        left.append(tuple(point + local_normal * half_thickness))
        right.append(tuple(point - local_normal * half_thickness))
    points = left + right[::-1]
    points = scale_points([(float(x), float(y)) for x, y in points], target_area)
    return clamp_points(points, width, height)


def make_mask(
    rng: random.Random,
    size: int,
    family: str,
    attempt: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ranges = FAMILY_RANGES[family]
    area_low, area_high = ranges["area"]
    aspect_low, aspect_high = ranges["aspect"]
    # Retries become slightly larger to avoid invisible edits.
    area_ratio = rng.uniform(area_low, min(area_high, area_high * (1.0 + 0.08 * attempt)))
    area_ratio = min(area_ratio, 0.18)
    aspect = rng.uniform(aspect_low, aspect_high)
    target_area = area_ratio * size * size
    if family == "C_elongated_ribbon":
        points = ribbon_points(rng, size, size, target_area, aspect)
        blur_radius = rng.uniform(0.8, 1.8)
    else:
        vertex_low, vertex_high = ranges["vertices"]
        vertex_count = rng.randint(vertex_low, vertex_high)
        if family == "A_small_low_contrast":
            points = blob_points(rng, size, size, target_area, aspect, vertex_count, 0.72, 1.20)
            blur_radius = rng.uniform(1.0, 2.0)
        elif family == "B_diffuse_texture":
            points = blob_points(rng, size, size, target_area, aspect, vertex_count, 0.68, 1.28)
            blur_radius = rng.uniform(1.2, 2.5)
        else:
            points = blob_points(rng, size, size, target_area, aspect, vertex_count, 0.60, 1.38)
            blur_radius = rng.uniform(0.8, 1.8)

    mask_image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask_image).polygon(points, fill=255)
    alpha_image = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    intended = alpha >= 0.20
    ys, xs = np.where(intended)
    if len(xs) == 0:
        raise RuntimeError("EMPTY_INTENDED_MASK")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    metadata = {
        "family": family,
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_aspect_ratio": max(x1 - x0 + 1, y1 - y0 + 1) / max(min(x1 - x0 + 1, y1 - y0 + 1), 1),
        "attempt": attempt + 1,
    }
    return alpha, intended, metadata


def connected_component_count(mask: np.ndarray) -> int:
    parent = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    previous_runs: list[tuple[int, int, int]] = []
    for y, row in enumerate(mask):
        diffs = np.diff(np.concatenate(([0], row.astype(np.uint8), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        current_runs = []
        for start, end in zip(starts, ends):
            label = len(parent)
            parent.append(label)
            current_runs.append((int(start), int(end) - 1, label))
        for current_start, current_end, current_label in current_runs:
            for previous_start, previous_end, previous_label in previous_runs:
                if current_start <= previous_end + 1 and previous_start <= current_end + 1:
                    union(current_label, previous_label)
        previous_runs = current_runs
    roots = {find(index) for index in range(len(parent))}
    return len(roots)


def mask_geometry(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("EMPTY_REFINED_MASK")
    area = int(mask.sum())
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    padded = np.pad(mask.astype(np.uint8), 1)
    neighbor_sum = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    boundary = int(np.sum((mask.astype(np.uint8) == 1) & (neighbor_sum < 4)))
    perimeter = float(boundary)
    compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    return {
        "mask_area_ratio": area / mask.size,
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_aspect_ratio": max(bw, bh) / max(min(bw, bh), 1),
        "relative_bbox_width": bw / mask.shape[1],
        "relative_bbox_height": bh / mask.shape[0],
        "perimeter": perimeter,
        "compactness": compactness,
        "connected_components": connected_component_count(mask),
    }


def apply_shading(original: np.ndarray, alpha: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    height, width = alpha.shape
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return original
    cy = float(ys.mean())
    cx = float(xs.mean())
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = max(float(radius[alpha > 0].max()), 1.0)
    profile = np.clip(1.0 - radius / max_radius, 0.0, 1.0)
    sign = -1.0 if rng.random() < 0.65 else 1.0
    delta = sign * strength * profile * (0.8 + 0.4 * rng.random())
    changed = original.astype(np.float32) + delta[..., None] * alpha[..., None]
    return np.clip(changed, 0, 255).astype(np.uint8)


def apply_texture_transplant(
    original: np.ndarray,
    alpha: np.ndarray,
    intended: np.ndarray,
    rng: random.Random,
    strength: float,
) -> np.ndarray:
    height, width, _ = original.shape
    ys, xs = np.where(intended)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    patch_height = y1 - y0 + 1
    patch_width = x1 - x0 + 1
    dest_cx = (x0 + x1) / 2.0
    dest_cy = (y0 + y1) / 2.0
    best = None
    best_distance = -1.0
    for _ in range(30):
        sy = rng.randint(0, max(height - patch_height, 0))
        sx = rng.randint(0, max(width - patch_width, 0))
        source_cx = sx + patch_width / 2.0
        source_cy = sy + patch_height / 2.0
        distance = math.hypot(source_cx - dest_cx, source_cy - dest_cy)
        if distance > best_distance:
            best_distance = distance
            best = (sy, sx)
        if distance > 2.0 * max(patch_width, patch_height):
            break
    sy, sx = best
    source = original[sy:sy + patch_height, sx:sx + patch_width].astype(np.float32)
    source = np.clip(source * rng.uniform(0.88, 1.12) + rng.uniform(-8.0, 8.0), 0, 255)
    changed = original.astype(np.float32).copy()
    target_alpha = alpha[y0:y1 + 1, x0:x1 + 1, None] * strength
    changed[y0:y1 + 1, x0:x1 + 1] = (
        changed[y0:y1 + 1, x0:x1 + 1] * (1.0 - target_alpha) + source * target_alpha
    )
    return np.clip(changed, 0, 255).astype(np.uint8)


def apply_local_blur(original: Image.Image, alpha: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    radius = 1.0 + strength * rng.uniform(0.8, 1.4)
    blurred = np.asarray(original.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    changed = np.asarray(original, dtype=np.float32).copy()
    changed = changed * (1.0 - alpha[..., None]) + blurred * alpha[..., None]
    return np.clip(changed, 0, 255).astype(np.uint8)


def apply_local_displacement(
    original: np.ndarray,
    alpha: np.ndarray,
    rng: random.Random,
    strength: float,
) -> np.ndarray:
    dx = int(round(rng.uniform(1.5, 4.0) * strength))
    dy = int(round(rng.uniform(1.0, 3.0) * strength))
    shifted = np.roll(original, shift=(dy, dx), axis=(0, 1)).astype(np.float32)
    changed = original.astype(np.float32).copy()
    changed = changed * (1.0 - alpha[..., None]) + shifted * alpha[..., None]
    return np.clip(changed, 0, 255).astype(np.uint8)


def apply_edit(
    original_image: Image.Image,
    original_array: np.ndarray,
    alpha: np.ndarray,
    intended: np.ndarray,
    family: str,
    rng: random.Random,
    attempt: int,
) -> np.ndarray:
    strength = 1.0 + 0.20 * attempt
    if family == "A_small_low_contrast":
        return apply_shading(original_array, alpha, rng, 14.0 * strength)
    if family == "B_diffuse_texture":
        if rng.random() < 0.60:
            return apply_texture_transplant(original_array, alpha, intended, rng, min(strength, 1.25))
        return apply_shading(original_array, alpha, rng, 18.0 * strength)
    if family == "C_elongated_ribbon":
        if rng.random() < 0.65:
            return apply_local_blur(original_image, alpha, rng, strength)
        return apply_shading(original_array, alpha, rng, 16.0 * strength)
    return apply_local_displacement(original_array, alpha, rng, strength)


def difference_stats(original: np.ndarray, changed: np.ndarray, mask: np.ndarray) -> dict:
    difference = np.abs(original.astype(np.int16) - changed.astype(np.int16))
    gray_original = grayscale_array(original)
    gray_changed = grayscale_array(changed)
    gray_difference = np.abs(gray_original - gray_changed)
    selected = gray_difference[mask]
    rgb_selected = difference[mask]
    if selected.size == 0:
        raise RuntimeError("EMPTY_DIFFERENCE_MASK")
    return {
        "gray_difference_mean": float(selected.mean()),
        "gray_difference_median": float(np.median(selected)),
        "gray_difference_max": float(selected.max()),
        "rgb_difference_mean": float(np.abs(rgb_selected).mean()),
        "pct_diff_ge_2": float(np.mean(selected >= 2) * 100.0),
        "pct_diff_ge_5": float(np.mean(selected >= 5) * 100.0),
        "pct_diff_ge_10": float(np.mean(selected >= 10) * 100.0),
        "single_edit_total_changed_pixels": int(np.count_nonzero(gray_difference >= 2.0)),
    }


def generate_pair(
    image_path: Path,
    seed: int,
    image_size: int,
) -> tuple[Image.Image, list[dict], list[dict], list[dict], str]:
    rng = random.Random(f"phaseC1A:{seed}:{image_path.stem}")
    with Image.open(image_path) as source:
        base_image = letterbox(source, image_size)
    base_image = shared_augmentation(base_image, rng)
    original_array = np.asarray(base_image, dtype=np.uint8)
    changed_array = original_array.copy()
    edit_rows = []
    failures = []
    image_status = "PASS"
    edit_count = rng.randint(1, 3)
    for edit_index in range(edit_count):
        family = rng.choices(list(FAMILY_PROBABILITY), weights=list(FAMILY_PROBABILITY.values()), k=1)[0]
        before_edit_array = changed_array.copy()
        before_change_pixels = count_changed_pixels(original_array, before_edit_array)
        success = False
        for attempt in range(MAX_ATTEMPTS):
            try:
                alpha, intended, intended_meta = make_mask(rng, image_size, family, attempt)
                current_image = Image.fromarray(before_edit_array)
                candidate = apply_edit(
                    current_image,
                    before_edit_array,
                    alpha,
                    intended,
                    family,
                    rng,
                    attempt,
                )
                gray_before = grayscale_array(before_edit_array)
                gray_candidate = grayscale_array(candidate)
                visible = (np.abs(gray_candidate - gray_before) >= 2.0) & intended
                if int(visible.sum()) < 12:
                    raise RuntimeError("VISIBLE_CHANGE_TOO_SMALL")
                refined = visible
                geometry = mask_geometry(refined)
                after_change_pixels = count_changed_pixels(original_array, candidate)
                row = {
                    "image_stem": image_path.stem,
                    "seed": seed,
                    "edit_index": edit_index,
                    "visible_change_ratio": int(refined.sum()) / max(int(intended.sum()), 1),
                    "before_change_pixels": before_change_pixels,
                    "after_change_pixels": after_change_pixels,
                    "delta_pixels": after_change_pixels - before_change_pixels,
                    **intended_meta,
                    **geometry,
                    **difference_stats(before_edit_array, candidate, refined),
                }
                edit_rows.append(row)
                changed_array = candidate
                success = True
            except Exception as error:
                failures.append({
                    "stem": image_path.stem,
                    "seed": seed,
                    "edit_index": edit_index,
                    "family": family,
                    "attempt": attempt + 1,
                    "reason": type(error).__name__,
                    "message": str(error),
                })
                continue
            break
        if not success:
            image_status = "FAILED"
            break
    if not edit_rows:
        raise RuntimeError(f"PAIR_HAS_NO_VALID_EDIT: seed={seed} stem={image_path.stem}")
    original_image = Image.fromarray(original_array)
    changed_image = Image.fromarray(changed_array)
    pair_image = Image.new("RGB", (image_size * 2 + 8, image_size), (255, 255, 255))
    pair_image.paste(original_image, (0, 0))
    pair_image.paste(changed_image, (image_size + 8, 0))
    image_rows = [{
        "image_stem": image_path.stem,
        "seed": seed,
        "edit_count": len(edit_rows),
        "families": ",".join(row["family"] for row in edit_rows),
        "total_refined_area_ratio": sum(row["mask_area_ratio"] for row in edit_rows),
        "min_visible_change_ratio": min(row["visible_change_ratio"] for row in edit_rows),
        "mean_gray_difference": float(np.mean([row["gray_difference_mean"] for row in edit_rows])),
        "status": image_status,
    }]
    return pair_image, edit_rows, image_rows, failures, image_status


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    families = Counter(row["family"] for row in rows)
    visible = [float(row["visible_change_ratio"]) for row in rows]
    return {
        "edit_count": len(rows),
        "family_counts": dict(families),
        "family_fractions": {family: count / len(rows) for family, count in families.items()},
        "mask_area_ratio": quantiles([float(row["mask_area_ratio"]) for row in rows]),
        "bbox_aspect_ratio": quantiles([float(row["bbox_aspect_ratio"]) for row in rows]),
        "perimeter": quantiles([float(row["perimeter"]) for row in rows]),
        "compactness": quantiles([float(row["compactness"]) for row in rows]),
        "connected_components": dict(Counter(int(row["connected_components"]) for row in rows)),
        "visible_change_ratio": quantiles(visible),
        "visible_change_ratio_below_0_5": float(np.mean(np.asarray(visible) < 0.5)),
        "gray_difference_mean": quantiles([float(row["gray_difference_mean"]) for row in rows]),
        "gray_difference_max": quantiles([float(row["gray_difference_max"]) for row in rows]),
        "pct_diff_ge_2": quantiles([float(row["pct_diff_ge_2"]) for row in rows]),
    }


def select_contact_examples(rows: list[dict]) -> list[dict]:
    selected = []
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        family_rows = sorted(family_rows, key=lambda row: float(row["mask_area_ratio"]))
        candidates = [
            family_rows[0],
            family_rows[len(family_rows) // 4],
            family_rows[len(family_rows) // 2],
            family_rows[(3 * len(family_rows)) // 4],
            family_rows[-1],
            max(family_rows, key=lambda row: float(row["bbox_aspect_ratio"])),
        ]
        unique = []
        seen = set()
        for row in candidates:
            key = (row["seed"], row["image_stem"], row["edit_index"])
            if key not in seen:
                unique.append(row)
                seen.add(key)
        selected.extend(unique[:4])
    return selected


def make_contact_sheet(
    rows: list[dict],
    image_dir: Path,
    output_path: Path,
    cell_width: int = 320,
    cell_height: int = 160,
) -> None:
    selected = select_contact_examples(rows)
    if not selected:
        raise RuntimeError("NO_EDITS_AVAILABLE_FOR_CONTACT_SHEET")
    columns = 4
    rows_per_sheet = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows_per_sheet * cell_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        col = index % columns
        row_index = index // columns
        path = image_dir / f"seed{row['seed']:02d}_{row['image_stem']}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"MISSING_PREVIEW_FOR_CONTACT_SHEET: {path}")
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width, cell_height - 18), Image.Resampling.BILINEAR)
        x = col * cell_width + (cell_width - image.width) // 2
        y = row_index * cell_height + 18
        sheet.paste(image, (x, y))
        label = f"{row['family']} a={row['mask_area_ratio']:.4f} ar={row['bbox_aspect_ratio']:.2f}"
        draw.text((col * cell_width + 4, row_index * cell_height + 3), label, fill=(0, 0, 0))
    sheet.save(output_path, quality=90)


def render_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    lines = []
    lines.append("# Phase C-1-B Morphology-aware Local Change Preview Report")
    lines.append("")
    lines.append("## Input Audit")
    lines.append("")
    lines.append(f"- TRAIN-only verification: **{payload['input_audit']['status']}**")
    lines.append(f"- TRAIN image count: **{payload['input_audit']['image_count']}**")
    lines.append(f"- Manifest stem set match: **{payload['input_audit']['manifest_stem_set_match']}**")
    lines.append(f"- Forbidden split paths found: **{payload['input_audit']['forbidden_split_path_count']}**")
    lines.append(f"- VAL images seen: **0**")
    lines.append(f"- TEST images seen: **0**")
    lines.append(f"- `test_accessed`: **false**")
    lines.append("")
    lines.append("## Generation Summary")
    lines.append("")
    lines.append(f"- Seeds: `{','.join(str(seed) for seed in payload['run_config']['seeds'])}`")
    lines.append(f"- Synthetic pairs: **{payload['generation_summary']['pair_count']}**")
    lines.append(f"- Valid edits: **{payload['generation_summary']['edit_count']}**")
    lines.append(f"- Failed images: **{payload['generation_summary']['failed_image_count']}**")
    lines.append(f"- Failed edit attempts: **{payload['generation_summary']['failed_attempt_count']}**")
    lines.append("")
    lines.append("## Family Distribution")
    lines.append("")
    lines.append("| Family | Edits | Fraction |")
    lines.append("|---|---:|---:|")
    for family in FAMILIES:
        count = summary["family_counts"].get(family, 0)
        fraction = summary["family_fractions"].get(family, 0.0)
        lines.append(f"| {family} | {count} | {fraction:.4f} |")
    lines.append("")
    lines.append("## Mask Geometry")
    lines.append("")
    lines.append("| Metric | q05 | q25 | q50 | q75 | q90 | q95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric in ["mask_area_ratio", "bbox_aspect_ratio", "perimeter", "compactness", "visible_change_ratio"]:
        values = summary[metric]
        lines.append(f"| {metric} | {values['q05']:.4f} | {values['q25']:.4f} | {values['q50']:.4f} | {values['q75']:.4f} | {values['q90']:.4f} | {values['q95']:.4f} |")
    lines.append("")
    lines.append("## Pixel Difference")
    lines.append("")
    lines.append("| Metric | q05 | q25 | q50 | q75 | q90 | q95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric in ["gray_difference_mean", "gray_difference_max", "pct_diff_ge_2"]:
        values = summary[metric]
        lines.append(f"| {metric} | {values['q05']:.4f} | {values['q25']:.4f} | {values['q50']:.4f} | {values['q75']:.4f} | {values['q90']:.4f} | {values['q95']:.4f} |")
    lines.append("")
    lines.append("## Gate Notes")
    lines.append("")
    lines.append("- Gate B numeric audit: **PASS** if the input audit is PASS and every pair has at least one valid edit.")
    lines.append("- Gate C numeric audit: **PASS** if all four families are present and visible-change statistics are nondegenerate.")
    lines.append("- Qualitative shortcut review still requires human inspection of the contact sheet.")
    lines.append("- This report makes no downstream performance claim.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_write_json(path: Path, payload: dict, failures: list[dict]) -> bool:
    try:
        write_json(path, payload)
        return True
    except Exception as error:
        failures.append({
            "artifact": str(path),
            "reason": type(error).__name__,
            "message": str(error),
        })
        return False


def try_write_csv(path: Path, rows: list[dict], failures: list[dict]) -> bool:
    try:
        write_csv(path, rows)
        return True
    except Exception as error:
        failures.append({
            "artifact": str(path),
            "reason": type(error).__name__,
            "message": str(error),
        })
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-source-root", type=Path, required=True)
    parser.add_argument(
        "--approved-physical-source-root",
        type=Path,
        default=None,
        help="Read-only root for resolved symlink targets; defaults to --approved-source-root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 0])
    parser.add_argument("--mode", choices=["debug", "formal"], default="debug")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="Debug-only image limit; 0 means all 668 images.")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()
    if args.image_size != 640:
        raise RuntimeError("PREVIEW_IMAGE_SIZE_MUST_BE_640")
    if args.limit < 0 or args.limit > 668:
        raise RuntimeError(f"LIMIT_OUT_OF_RANGE: {args.limit}")
    if args.mode == "formal" and args.limit != 0:
        raise RuntimeError("FORMAL_MODE_FORBIDS_LIMIT")
    if not args.seeds:
        raise RuntimeError("SEED_LIST_MUST_NOT_BE_EMPTY")
    if any(seed < 0 for seed in args.seeds):
        raise RuntimeError("SEEDS_MUST_BE_NONNEGATIVE")
    if len(set(args.seeds)) != len(args.seeds):
        raise RuntimeError(f"DUPLICATE_SEEDS: {args.seeds}")
    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    train_dir = args.train_dir.resolve()
    manifest = args.manifest.resolve()
    approved_source_root = args.approved_source_root.resolve()
    approved_physical_source_root = (
        args.approved_physical_source_root.resolve()
        if args.approved_physical_source_root is not None
        else approved_source_root
    )
    output_root = args.output_root.resolve()
    image_size = args.image_size
    results_root = (repo_root / "results").resolve()
    if not repo_root.is_dir():
        raise RuntimeError(f"REPO_ROOT_NOT_FOUND: {repo_root}")
    if not train_dir.is_dir():
        raise RuntimeError(f"TRAIN_DIR_NOT_FOUND: {train_dir}")
    if not manifest.is_file():
        raise RuntimeError(f"MANIFEST_NOT_FOUND: {manifest}")
    if not approved_source_root.is_dir():
        raise RuntimeError(f"APPROVED_SOURCE_ROOT_NOT_FOUND: {approved_source_root}")
    if not approved_physical_source_root.is_dir():
        raise RuntimeError(f"APPROVED_PHYSICAL_SOURCE_ROOT_NOT_FOUND: {approved_physical_source_root}")
    if not path_is_within(output_root, results_root):
        raise RuntimeError(f"OUTPUT_ROOT_OUTSIDE_REPO_RESULTS: {output_root}")
    if path_is_within(output_root, approved_source_root) or path_is_within(approved_source_root, output_root):
        raise RuntimeError(f"OUTPUT_AND_SOURCE_OVERLAP: {output_root}")
    if path_is_within(output_root, approved_physical_source_root) or path_is_within(approved_physical_source_root, output_root):
        raise RuntimeError(f"OUTPUT_AND_PHYSICAL_SOURCE_OVERLAP: {output_root}")

    expected_stems = load_manifest_train_stems(manifest)
    if len(expected_stems) != 668:
        raise RuntimeError(f"MANIFEST_TRAIN_COUNT_GATE: {len(expected_stems)} != 668")
    image_paths = collect_train_images(
        train_dir,
        expected_stems,
        approved_source_root,
        approved_physical_source_root,
    )
    selected_image_paths = image_paths[:args.limit] if args.limit else image_paths
    image_count = len(selected_image_paths)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"preview_{timestamp}_seed{'_'.join(str(seed) for seed in args.seeds)}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise RuntimeError(f"INVALID_RUN_NAME: {run_name}")
    output_dir = output_root / run_name
    if output_dir.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_OUTPUT: {output_dir}")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)

    metadata_save_failures: list[dict] = []

    def save_json(label: str, path: Path, payload: dict) -> bool:
        return try_write_json(path, payload, metadata_save_failures)

    def save_csv(label: str, path: Path, rows: list[dict]) -> bool:
        return try_write_csv(path, rows, metadata_save_failures)

    all_edit_rows = []
    all_image_rows = []
    all_failures = []
    per_image_records = []
    pair_count = 0
    saved_preview_count = 0
    failed_image_count = 0
    for seed in args.seeds:
        for image_path in selected_image_paths:
            generated_output = None
            edit_rows = []
            image_rows = []
            failures = []
            image_status = "FAILED"
            try:
                pair_image, edit_rows, image_rows, failures, image_status = generate_pair(
                    image_path,
                    seed,
                    image_size,
                )
            except Exception as error:
                failures = [{
                    "stem": image_path.stem,
                    "seed": seed,
                    "edit_index": None,
                    "reason": type(error).__name__,
                    "message": str(error),
                }]
                edit_rows = []
                image_rows = [{
                    "image_stem": image_path.stem,
                    "seed": seed,
                    "edit_count": 0,
                    "families": "",
                    "total_refined_area_ratio": 0.0,
                    "min_visible_change_ratio": 0.0,
                    "mean_gray_difference": 0.0,
                    "status": "FAILED",
                }]
                pair_image = None

            record_status = "PASS" if image_status == "PASS" and pair_image is not None else "FAILED"
            if pair_image is not None:
                output_path = image_dir / f"seed{seed:02d}_{image_path.stem}.jpg"
                try:
                    pair_image.save(output_path, quality=88)
                    generated_output = output_path.relative_to(output_dir).as_posix()
                    saved_preview_count += 1
                except Exception as error:
                    failures.append({
                        "stem": image_path.stem,
                        "seed": seed,
                        "edit_index": None,
                        "reason": type(error).__name__,
                        "message": f"IMAGE_SAVE_FAILURE: {error}",
                    })
                    record_status = "FAILED"

            if image_rows:
                for row in image_rows:
                    row["status"] = record_status
            else:
                image_rows = [{
                    "image_stem": image_path.stem,
                    "seed": seed,
                    "edit_count": len(edit_rows),
                    "families": ",".join(row["family"] for row in edit_rows),
                    "total_refined_area_ratio": sum(row["mask_area_ratio"] for row in edit_rows),
                    "min_visible_change_ratio": min(row["visible_change_ratio"] for row in edit_rows),
                    "mean_gray_difference": float(np.mean([row["gray_difference_mean"] for row in edit_rows])) if edit_rows else 0.0,
                    "status": record_status,
                }]

            if record_status == "PASS":
                pair_count += 1
            else:
                failed_image_count += 1
            all_edit_rows.extend(edit_rows)
            all_image_rows.extend(image_rows)
            all_failures.extend(failures)
            edit_types = [row["family"] for row in edit_rows]
            per_image_records.append({
                "seed": seed,
                "source_image": str(image_path),
                "generated_output": generated_output,
                "edit_type": ",".join(edit_types),
                "edit_types": edit_types,
                "edit_count": len(edit_rows),
                "status": record_status,
                "failures": failures,
            })

    input_audit = {
        "status": "PASS",
        "train_dir": str(train_dir),
        "approved_source_root": str(approved_source_root),
        "approved_physical_source_root": str(approved_physical_source_root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_train_stem_count": len(expected_stems),
        "frozen_train_image_count": len(image_paths),
        "image_count": image_count,
        "input_image_limit": args.limit,
        "manifest_stem_set_match": True,
        "forbidden_split_path_count": 0,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    generation_summary = {
        "pair_count": pair_count,
        "saved_preview_count": saved_preview_count,
        "edit_count": len(all_edit_rows),
        "failed_image_count": failed_image_count,
        "failed_attempt_count": len(all_failures),
    }
    summary = summarize(all_edit_rows)
    run_config = {
        "scope": "PHASE_C1_B_MORPHOLOGY_AWARE_LOCAL_CHANGE_PREVIEW",
        "mode": args.mode,
        "gate_mode": "FORMAL_GATE" if args.mode == "formal" else "DEBUG_ONLY",
        "seeds": args.seeds,
        "primary_seed": args.seeds[0],
        "image_size": image_size,
        "families": FAMILIES,
        "family_probability": FAMILY_PROBABILITY,
        "family_ranges": FAMILY_RANGES,
        "max_attempts": MAX_ATTEMPTS,
        "input_image_limit": args.limit,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    payload = {
        "run_config": run_config,
        "input_audit": input_audit,
        "generation_summary": generation_summary,
        "summary": summary,
    }
    save_json("run_config", output_dir / "run_config.json", run_config)
    save_json("input_audit", output_dir / "input_audit.json", input_audit)
    save_json("generation_summary", output_dir / "generation_summary.json", generation_summary)
    save_json("summary", output_dir / "summary.json", summary)
    save_csv("edit_metrics", output_dir / "edit_metrics.csv", all_edit_rows)
    save_csv("image_metrics", output_dir / "image_metrics.csv", all_image_rows)
    save_csv("failed_attempts", output_dir / "failed_attempts.csv", all_failures)
    artifacts_not_generated = []
    if all_edit_rows:
        try:
            make_contact_sheet(all_edit_rows, image_dir, output_dir / "contact_sheet_families.jpg")
        except Exception as error:
            metadata_save_failures.append({
                "artifact": "contact_sheet_families.jpg",
                "reason": type(error).__name__,
                "message": str(error),
            })
        try:
            render_report(output_dir / "preview_report.md", payload)
        except Exception as error:
            metadata_save_failures.append({
                "artifact": "preview_report.md",
                "reason": type(error).__name__,
                "message": str(error),
            })
    else:
        artifacts_not_generated = ["contact_sheet_families.jpg", "preview_report.md"]

    successful_edit_keys = {
        (row["seed"], row["image_stem"], row["edit_index"])
        for row in all_edit_rows
    }
    failed_edit_keys = {
        (row["seed"], row["stem"], row["edit_index"])
        for row in all_failures
        if row.get("edit_index") is not None
    } - successful_edit_keys
    failed_source_keys = {
        (row["seed"], row["stem"])
        for row in all_failures
        if row.get("edit_index") is None
    }
    hard_gate_reasons = []
    if args.mode == "formal" and image_count != 668:
        hard_gate_reasons.append(f"FORMAL_IMAGE_COUNT_GATE:{image_count}_!=_668")
    if failed_image_count:
        hard_gate_reasons.append(f"FAILED_IMAGE_COUNT:{failed_image_count}")
    if failed_edit_keys:
        hard_gate_reasons.append(f"FAILED_EDIT_COUNT:{len(failed_edit_keys)}")
    if failed_source_keys:
        hard_gate_reasons.append(f"FAILED_SOURCE_READ_COUNT:{len(failed_source_keys)}")
    if metadata_save_failures:
        hard_gate_reasons.append(f"METADATA_SAVE_FAILURE_COUNT:{len(metadata_save_failures)}")
    if artifacts_not_generated:
        hard_gate_reasons.append("AUDIT_ARTIFACTS_NOT_GENERATED")
    final_status_value = "FAIL" if hard_gate_reasons else "PASS"
    formal_gate = args.mode == "formal" and final_status_value == "PASS"
    generation_manifest = {
        "schema_version": 1,
        "run_id": run_name,
        "timestamp": timestamp,
        "mode": args.mode,
        "formal_gate": formal_gate,
        "seed_policy": "DETERMINISTIC_PER_IMAGE_WITH_SEED_PREFIX",
        "seed": args.seeds[0],
        "seeds": args.seeds,
        "source_root": str(approved_source_root),
        "physical_source_root": str(approved_physical_source_root),
        "train_dir": str(train_dir),
        "manifest": str(manifest),
        "manifest_sha256": input_audit["manifest_sha256"],
        "image_count": image_count,
        "edit_count": len(all_edit_rows),
        "failed_edit_count": len(failed_edit_keys),
        "failed_source_image_count": len(failed_source_keys),
        "per_image": per_image_records,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    save_json("generation_manifest", output_dir / "generation_manifest.json", generation_manifest)
    final_status = {
        "schema_version": 1,
        "status": final_status_value,
        "mode": args.mode,
        "formal_gate": formal_gate,
        "run_id": run_name,
        "image_count": image_count,
        "seed": args.seeds[0],
        "seeds": args.seeds,
        "failed_edits": len(failed_edit_keys),
        "failed_images": failed_image_count,
        "failed_source_reads": len(failed_source_keys),
        "failed_attempts": len(all_failures),
        "metadata_save_failures": len(metadata_save_failures),
        "metadata_save_failure_details": metadata_save_failures,
        "artifacts_not_generated": artifacts_not_generated,
        "hard_gate_reasons": hard_gate_reasons,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    try:
        write_json(output_dir / "final_status.json", final_status)
    except Exception as error:
        print(json.dumps({
            "status": "FAIL",
            "reason": "FINAL_STATUS_WRITE_FAILURE",
            "message": str(error),
        }, ensure_ascii=False))
        return 4
    print(json.dumps({
        "status": final_status_value,
        "formal_gate": formal_gate,
        "output_dir": str(output_dir),
        **generation_summary,
    }, ensure_ascii=False))
    return 0 if final_status_value == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
