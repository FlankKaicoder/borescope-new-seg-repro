#!/usr/bin/env python3
"""Phase C1-R1 Local Change V2 preview generation.

This is an isolated, TRAIN-only generator experiment. A/B reuse the frozen V1
implementation. C and D use V2 morphology plus acceptance checks before a
candidate can enter a final preview pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FAMILIES = [
    "A_small_low_contrast",
    "B_diffuse_texture",
    "C_crack_like_structural_evolution",
    "D_boundary_evolution",
]
FAMILY_PROBABILITY = {
    "A_small_low_contrast": 0.30,
    "B_diffuse_texture": 0.30,
    "C_crack_like_structural_evolution": 0.25,
    "D_boundary_evolution": 0.15,
}
MAX_ATTEMPTS = 10
V2_VERSION = "phaseC1-r1-local-change-v2"

C_ACCEPTANCE = {
    "component_count": 1,
    "minimum_skeleton_length": 24,
    "minimum_elongation_score": 2.0,
}
D_ACCEPTANCE = {
    "component_count": 1,
    "minimum_largest_component_ratio": 0.95,
    "minimum_boundary_score": 0.18,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
V1_A_B_RANGES = {
    "A_small_low_contrast": {"area": (0.0005, 0.0030), "aspect": (1.0, 2.0), "vertices": (10, 16)},
    "B_diffuse_texture": {"area": (0.0020, 0.1700), "aspect": (1.0, 4.0), "vertices": (16, 32)},
}


# Frozen V1 support is embedded so this independently versioned repair can run
# against the project's older runtime checkout without modifying V1.
def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def grayscale_array(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3].astype(np.float32), [0.299, 0.587, 0.114])


def count_changed_pixels(reference: np.ndarray, current: np.ndarray, threshold: float = 2.0) -> int:
    return int(np.count_nonzero(np.abs(grayscale_array(reference) - grayscale_array(current)) >= threshold))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_train_stems(path: Path) -> set[str]:
    stems = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("split") or "").strip().lower() != "train":
                continue
            stem = (row.get("stem") or "").strip()
            if not stem:
                raise RuntimeError("MANIFEST_EMPTY_TRAIN_STEM")
            if stem in stems:
                raise RuntimeError(f"MANIFEST_DUPLICATE_TRAIN_STEM:{stem}")
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
        raise RuntimeError(f"TRAIN_DIR_GATE:{resolved_dir}")
    if not path_is_within(resolved_dir, approved_source_root):
        raise RuntimeError(f"TRAIN_DIR_OUTSIDE_APPROVED_SOURCE_ROOT:{resolved_dir}")
    paths: list[Path] = []
    seen = set()
    for path in resolved_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        lowered_parts = {part.lower() for part in resolved.parts}
        if "val" in lowered_parts or "test" in lowered_parts:
            raise RuntimeError(f"FORBIDDEN_SPLIT_PATH:{resolved}")
        if not (path_is_within(resolved, approved_source_root) or path_is_within(resolved, approved_physical_source_root)):
            raise RuntimeError(f"SOURCE_OUTSIDE_APPROVED_ROOTS:{resolved}")
        if path.stem in seen:
            raise RuntimeError(f"DUPLICATE_STEM:{path.stem}")
        seen.add(path.stem)
        paths.append(resolved)
    if len(paths) != 668:
        raise RuntimeError(f"TRAIN_IMAGE_COUNT_GATE:{len(paths)}!=668")
    if seen != expected_stems:
        raise RuntimeError("TRAIN_STEM_SET_GATE")
    return sorted(paths, key=lambda item: item.stem)


def letterbox(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def shared_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image
    if rng.random() < 0.50:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.15:
        result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if rng.random() < 0.50:
        array = np.asarray(result, dtype=np.float32)
        array = np.clip(array * rng.uniform(0.90, 1.10) + rng.uniform(-8.0, 8.0), 0, 255).astype(np.uint8)
        result = Image.fromarray(array)
    return result


def scale_points(points: list[tuple[float, float]], target_area: float) -> list[tuple[float, float]]:
    area = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2.0
    if area <= 0:
        return points
    factor = math.sqrt(target_area / area)
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points]


def v1_blob_points(rng: random.Random, size: int, target_area: float, aspect: float, vertex_count: int, jitter_low: float, jitter_high: float) -> list[tuple[float, float]]:
    base_radius = math.sqrt(target_area / math.pi)
    rx, ry = base_radius * math.sqrt(aspect), base_radius / math.sqrt(aspect)
    rotation, phase, amplitude = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(0.0, 2.0 * math.pi), rng.uniform(0.08, 0.28)
    cx = rng.uniform(max(rx, 4.0), max(size - rx - 1.0, 5.0))
    cy = rng.uniform(max(ry, 4.0), max(size - ry - 1.0, 5.0))
    points = []
    for index in range(vertex_count):
        angle = rotation + 2.0 * math.pi * index / vertex_count
        radius = rng.uniform(jitter_low, jitter_high) * (1.0 + amplitude * math.sin(3.0 * angle + phase))
        points.append((cx + rx * radius * math.cos(angle), cy + ry * radius * math.sin(angle)))
    return [(min(max(x, 0.0), size - 1.0), min(max(y, 0.0), size - 1.0)) for x, y in scale_points(points, target_area)]


def make_v1_mask(rng: random.Random, size: int, family: str, attempt: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if family not in V1_A_B_RANGES:
        raise RuntimeError(f"V1_MASK_UNSUPPORTED_FAMILY:{family}")
    ranges = V1_A_B_RANGES[family]
    area_ratio = min(rng.uniform(ranges["area"][0], ranges["area"][1] * (1.0 + 0.08 * attempt)), 0.18)
    aspect = rng.uniform(*ranges["aspect"])
    vertices = rng.randint(*ranges["vertices"])
    jitter = (0.72, 1.20) if family == "A_small_low_contrast" else (0.68, 1.28)
    points = v1_blob_points(rng, size, area_ratio * size * size, aspect, vertices, *jitter)
    mask_image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask_image).polygon(points, fill=255)
    blur = rng.uniform(1.0, 2.0) if family == "A_small_low_contrast" else rng.uniform(1.2, 2.5)
    alpha = np.asarray(mask_image.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
    intended = alpha >= 0.20
    ys, xs = np.where(intended)
    if not len(xs):
        raise RuntimeError("EMPTY_INTENDED_MASK")
    return alpha, intended, {"family": family, "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "bbox_aspect_ratio": max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / max(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), 1), "attempt": attempt + 1}


def apply_v1_shading(original: np.ndarray, alpha: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    ys, xs = np.where(alpha > 0)
    if not len(xs):
        return original
    yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]].astype(np.float32)
    radius = np.sqrt((yy - float(ys.mean())) ** 2 + (xx - float(xs.mean())) ** 2)
    profile = np.clip(1.0 - radius / max(float(radius[alpha > 0].max()), 1.0), 0.0, 1.0)
    delta = (-1.0 if rng.random() < 0.65 else 1.0) * strength * profile * (0.8 + 0.4 * rng.random())
    return np.clip(original.astype(np.float32) + delta[..., None] * alpha[..., None], 0, 255).astype(np.uint8)


def apply_v1_texture(original: np.ndarray, alpha: np.ndarray, intended: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    ys, xs = np.where(intended)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    height, width = y1 - y0 + 1, x1 - x0 + 1
    best, distance = (0, 0), -1.0
    for _ in range(30):
        sy, sx = rng.randint(0, max(original.shape[0] - height, 0)), rng.randint(0, max(original.shape[1] - width, 0))
        candidate_distance = math.hypot(sx + width / 2.0 - (x0 + x1) / 2.0, sy + height / 2.0 - (y0 + y1) / 2.0)
        if candidate_distance > distance:
            best, distance = (sy, sx), candidate_distance
        if distance > 2.0 * max(width, height):
            break
    sy, sx = best
    source = np.clip(original[sy:sy + height, sx:sx + width].astype(np.float32) * rng.uniform(0.88, 1.12) + rng.uniform(-8.0, 8.0), 0, 255)
    changed = original.astype(np.float32).copy()
    target_alpha = alpha[y0:y1 + 1, x0:x1 + 1, None] * strength
    changed[y0:y1 + 1, x0:x1 + 1] = changed[y0:y1 + 1, x0:x1 + 1] * (1.0 - target_alpha) + source * target_alpha
    return np.clip(changed, 0, 255).astype(np.uint8)


def apply_edit(original_image: Image.Image, original: np.ndarray, alpha: np.ndarray, intended: np.ndarray, family: str, rng: random.Random, attempt: int) -> np.ndarray:
    strength = 1.0 + 0.20 * attempt
    if family == "A_small_low_contrast":
        return apply_v1_shading(original, alpha, rng, 14.0 * strength)
    if family == "B_diffuse_texture":
        return apply_v1_texture(original, alpha, intended, rng, min(strength, 1.25)) if rng.random() < 0.60 else apply_v1_shading(original, alpha, rng, 18.0 * strength)
    raise RuntimeError(f"V1_EDIT_UNSUPPORTED_FAMILY:{family}")


def difference_stats(original: np.ndarray, changed: np.ndarray, mask: np.ndarray) -> dict:
    difference = np.abs(original.astype(np.int16) - changed.astype(np.int16))
    gray_difference = np.abs(grayscale_array(original) - grayscale_array(changed))
    selected, rgb_selected = gray_difference[mask], difference[mask]
    if not selected.size:
        raise RuntimeError("EMPTY_DIFFERENCE_MASK")
    return {"gray_difference_mean": float(selected.mean()), "gray_difference_median": float(np.median(selected)), "gray_difference_max": float(selected.max()), "rgb_difference_mean": float(np.abs(rgb_selected).mean()), "pct_diff_ge_2": float(np.mean(selected >= 2) * 100.0), "pct_diff_ge_5": float(np.mean(selected >= 5) * 100.0), "pct_diff_ge_10": float(np.mean(selected >= 10) * 100.0), "single_edit_total_changed_pixels": int(np.count_nonzero(gray_difference >= 2.0))}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grayscale(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3].astype(np.float32), [0.299, 0.587, 0.114])


def component_metrics(mask: np.ndarray) -> tuple[int, float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return count - 1, float(areas.max() / max(int(mask.sum()), 1))


def skeleton_length(mask: np.ndarray) -> int:
    image = (mask.astype(np.uint8) * 255).copy()
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return int(cv2.countNonZero(skeleton))


def elongation_score(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0
    points = np.column_stack((xs, ys)).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
    if eigenvalues[0] <= 1e-9:
        return float("inf")
    return float(math.sqrt(eigenvalues[1] / eigenvalues[0]))


def boundary_score(mask: np.ndarray) -> float:
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
    boundary = binary & (1 - eroded)
    return float(boundary.sum() / max(int(binary.sum()), 1))


def geometry(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("EMPTY_REFINED_MASK")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    area = int(mask.sum())
    perimeter = int((mask.astype(np.uint8) & (1 - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)))).sum())
    components, largest_ratio = component_metrics(mask)
    return {
        "mask_area_ratio": area / mask.size,
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_aspect_ratio": max(width, height) / max(min(width, height), 1),
        "relative_bbox_width": width / mask.shape[1],
        "relative_bbox_height": height / mask.shape[0],
        "perimeter": float(perimeter),
        "compactness": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0,
        "connected_components": components,
        "largest_component_ratio": largest_ratio,
        "largest_component_ratio_lower_bound": largest_ratio,
        "skeleton_length": skeleton_length(mask),
        "elongation_score": elongation_score(mask),
        "boundary_score": boundary_score(mask),
    }


def visible_mask(before: np.ndarray, candidate: np.ndarray, intended: np.ndarray) -> np.ndarray:
    return (np.abs(grayscale(candidate) - grayscale(before)) >= 2.0) & intended


def curve_centerline(rng: random.Random, size: int) -> tuple[list[tuple[float, float]], list[float], float]:
    length = rng.uniform(90.0, 270.0)
    base_width = rng.uniform(3.5, 8.5)
    margin = length / 2.0 + 2.5 * base_width + 18.0
    angle = rng.uniform(0.0, math.pi)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    center = np.array([
        rng.uniform(margin, size - margin),
        rng.uniform(margin, size - margin),
    ])
    amplitude = rng.uniform(0.04, 0.13) * length
    phase = rng.uniform(0.0, 2.0 * math.pi)
    points: list[tuple[float, float]] = []
    widths: list[float] = []
    sample_count = 64
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        longitudinal = (fraction - 0.5) * length
        lateral = amplitude * math.sin(2.0 * math.pi * fraction + phase)
        point = center + direction * longitudinal + normal * lateral
        width = base_width * (0.72 + 0.45 * math.sin(2.0 * math.pi * fraction + phase / 2.0))
        points.append((float(point[0]), float(point[1])))
        widths.append(max(2.5, width))
    centerline_length = float(sum(
        math.dist(left, right) for left, right in zip(points, points[1:])
    ))
    return points, widths, centerline_length


def make_c_mask(rng: random.Random, size: int, attempt: int) -> tuple[np.ndarray, np.ndarray, dict]:
    points, widths, centerline_length = curve_centerline(rng, size)
    mask_image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_image)
    for left, right, width_left, width_right in zip(points, points[1:], widths, widths[1:]):
        width = max(3, int(round((width_left + width_right) / 2.0 + attempt * 0.25)))
        draw.line([left, right], fill=255, width=width)
    for point, width in zip(points[::4], widths[::4]):
        radius = max(1, int(round(width / 2.0)))
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=255)
    alpha = np.asarray(mask_image.filter(ImageFilter.GaussianBlur(0.65)), dtype=np.float32) / 255.0
    intended = alpha >= 0.20
    raw = geometry(intended)
    raw.update({
        "family": "C_crack_like_structural_evolution",
        "morphology_variant": "CENTERLINE_VARIABLE_WIDTH_CRACK",
        "centerline_length": centerline_length,
        "attempt": attempt + 1,
    })
    return alpha, intended, raw


def make_irregular_seed(rng: random.Random, size: int) -> np.ndarray:
    target_area = rng.uniform(0.009, 0.055) * size * size
    aspect = rng.uniform(1.0, 2.8)
    radius = math.sqrt(target_area / math.pi)
    rx = radius * math.sqrt(aspect)
    ry = radius / math.sqrt(aspect)
    margin_x = rx + 20.0
    margin_y = ry + 20.0
    cx = rng.uniform(margin_x, size - margin_x)
    cy = rng.uniform(margin_y, size - margin_y)
    rotation = rng.uniform(0.0, math.pi)
    points = []
    vertex_count = rng.randint(36, 60)
    for index in range(vertex_count):
        angle = rotation + 2.0 * math.pi * index / vertex_count
        harmonic = 1.0 + 0.16 * math.sin(3.0 * angle + rng.uniform(-0.25, 0.25))
        jitter = rng.uniform(0.84, 1.18) * harmonic
        points.append((cx + rx * jitter * math.cos(angle), cy + ry * jitter * math.sin(angle)))
    image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(image).polygon(points, fill=255)
    seed = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)
    return cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))


def smooth_contour_deformation(seed: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = seed.shape
    coarse_x = np.asarray([[rng.uniform(-1.0, 1.0) for _ in range(18)] for _ in range(18)], dtype=np.float32)
    coarse_y = np.asarray([[rng.uniform(-1.0, 1.0) for _ in range(18)] for _ in range(18)], dtype=np.float32)
    dx = cv2.resize(coarse_x, (width, height), interpolation=cv2.INTER_CUBIC) * rng.uniform(1.5, 4.0)
    dy = cv2.resize(coarse_y, (width, height), interpolation=cv2.INTER_CUBIC) * rng.uniform(1.5, 4.0)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped = cv2.remap(seed * 255, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return (warped >= 128).astype(np.uint8)


def make_d_mask(rng: random.Random, size: int, attempt: int) -> tuple[np.ndarray, np.ndarray, dict]:
    seed = make_irregular_seed(rng, size)
    deformed = smooth_contour_deformation(seed, rng)
    radius = rng.randint(2, min(7 + attempt, 11))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    operation = "dilation" if rng.random() < 0.5 else "erosion"
    evolved = cv2.dilate(deformed, kernel) if operation == "dilation" else cv2.erode(deformed, kernel)
    intended = np.logical_xor(seed.astype(bool), evolved.astype(bool))
    alpha = np.asarray(Image.fromarray(intended.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32) / 255.0
    intended = alpha >= 0.20
    seed_components, _ = component_metrics(seed.astype(bool))
    raw = geometry(intended)
    raw.update({
        "family": "D_boundary_evolution",
        "morphology_variant": "SYNTHETIC_SEED_CONTOUR_DEFORMATION_" + operation.upper(),
        "base_component_count": seed_components,
        "boundary_operation": operation,
        "attempt": attempt + 1,
    })
    return alpha, intended, raw


def apply_structural_contrast(before: np.ndarray, alpha: np.ndarray, rng: random.Random, attempt: int, family: str) -> np.ndarray:
    magnitude = rng.uniform(28.0, 48.0) * (1.0 + 0.08 * attempt)
    if family == "D_boundary_evolution" and rng.random() < 0.35:
        magnitude *= -1.0
    else:
        magnitude *= -1.0
    shifted = np.roll(before, shift=(rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])), axis=(0, 1)).astype(np.float32)
    structured = np.clip(0.75 * before.astype(np.float32) + 0.25 * shifted + magnitude, 0, 255)
    changed = before.astype(np.float32) * (1.0 - alpha[..., None]) + structured * alpha[..., None]
    return np.clip(changed, 0, 255).astype(np.uint8)


def accept_candidate(family: str, refined: np.ndarray, quality: dict) -> list[str]:
    reasons = []
    if int(refined.sum()) < 24:
        reasons.append("VISIBLE_CHANGE_TOO_SMALL")
    if family == "C_crack_like_structural_evolution":
        if quality["connected_components"] != C_ACCEPTANCE["component_count"]:
            reasons.append("C_COMPONENT_COUNT")
        if quality["skeleton_length"] < C_ACCEPTANCE["minimum_skeleton_length"]:
            reasons.append("C_SKELETON_LENGTH")
        if quality["elongation_score"] < C_ACCEPTANCE["minimum_elongation_score"]:
            reasons.append("C_ELONGATION_SCORE")
    elif family == "D_boundary_evolution":
        if quality["connected_components"] != D_ACCEPTANCE["component_count"]:
            reasons.append("D_COMPONENT_COUNT")
        if quality["largest_component_ratio"] < D_ACCEPTANCE["minimum_largest_component_ratio"]:
            reasons.append("D_LARGEST_COMPONENT_RATIO")
        if quality["boundary_score"] < D_ACCEPTANCE["minimum_boundary_score"]:
            reasons.append("D_BOUNDARY_SCORE")
    return reasons


def candidate_for_family(
    family: str,
    before: np.ndarray,
    rng: random.Random,
    image_size: int,
    attempt: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if family == "C_crack_like_structural_evolution":
        alpha, intended, metadata = make_c_mask(rng, image_size, attempt)
        return apply_structural_contrast(before, alpha, rng, attempt, family), intended, metadata
    if family == "D_boundary_evolution":
        alpha, intended, metadata = make_d_mask(rng, image_size, attempt)
        return apply_structural_contrast(before, alpha, rng, attempt, family), intended, metadata
    alpha, intended, metadata = make_v1_mask(rng, image_size, family, attempt)
    candidate = apply_edit(Image.fromarray(before), before, alpha, intended, family, rng, attempt)
    metadata["morphology_variant"] = "V1_UNCHANGED"
    return candidate, intended, metadata


def generate_pair(image_path: Path, seed: int, image_size: int) -> tuple[Image.Image | None, list[dict], dict, list[dict], list[dict], str]:
    rng = random.Random(f"phaseC1R1:{seed}:{image_path.stem}")
    with Image.open(image_path) as source:
        base = shared_augmentation(letterbox(source, image_size), rng)
    original = np.asarray(base, dtype=np.uint8)
    changed = original.copy()
    rows: list[dict] = []
    rejected: list[dict] = []
    failures: list[dict] = []
    requested_edits = rng.randint(1, 3)
    for edit_index in range(requested_edits):
        family = rng.choices(list(FAMILY_PROBABILITY), weights=list(FAMILY_PROBABILITY.values()), k=1)[0]
        before = changed.copy()
        before_change_pixels = count_changed_pixels(original, before)
        accepted = False
        for attempt in range(MAX_ATTEMPTS):
            try:
                candidate, intended, metadata = candidate_for_family(family, before, rng, image_size, attempt)
                refined = visible_mask(before, candidate, intended)
                quality = geometry(refined)
                reasons = accept_candidate(family, refined, quality)
                if reasons:
                    rejected.append({
                        "image_stem": image_path.stem,
                        "seed": seed,
                        "edit_index": edit_index,
                        "family": family,
                        "attempt": attempt + 1,
                        "reasons": "|".join(reasons),
                        **quality,
                    })
                    continue
                after_change_pixels = count_changed_pixels(original, candidate)
                row = {
                    "image_stem": image_path.stem,
                    "seed": seed,
                    "edit_index": edit_index,
                    "family": family,
                    "visible_change_ratio": int(refined.sum()) / max(int(intended.sum()), 1),
                    "before_change_pixels": before_change_pixels,
                    "after_change_pixels": after_change_pixels,
                    "delta_pixels": after_change_pixels - before_change_pixels,
                    "quality_filter_attempt": attempt + 1,
                    "quality_filter_rejections_before_acceptance": sum(
                        item["image_stem"] == image_path.stem and item["seed"] == seed and item["edit_index"] == edit_index
                        for item in rejected
                    ),
                    "acceptance_status": "ACCEPTED",
                    **metadata,
                    **quality,
                    **difference_stats(before, candidate, refined),
                }
                rows.append(row)
                changed = candidate
                accepted = True
                break
            except Exception as error:
                rejected.append({
                    "image_stem": image_path.stem,
                    "seed": seed,
                    "edit_index": edit_index,
                    "family": family,
                    "attempt": attempt + 1,
                    "reasons": f"EXCEPTION:{type(error).__name__}:{error}",
                    "mask_area_ratio": "",
                    "connected_components": "",
                    "largest_component_ratio": "",
                    "skeleton_length": "",
                    "elongation_score": "",
                    "boundary_score": "",
                })
        if not accepted:
            failures.append({
                "image_stem": image_path.stem,
                "seed": seed,
                "edit_index": edit_index,
                "family": family,
                "reason": "QUALITY_FILTER_EXHAUSTED",
                "attempt_count": MAX_ATTEMPTS,
            })
            return None, rows, {"edit_count": len(rows), "status": "FAILED"}, rejected, failures, "FAILED"
    pair = Image.new("RGB", (image_size * 2 + 8, image_size), (255, 255, 255))
    pair.paste(Image.fromarray(original), (0, 0))
    pair.paste(Image.fromarray(changed), (image_size + 8, 0))
    image_row = {
        "image_stem": image_path.stem,
        "seed": seed,
        "edit_count": len(rows),
        "families": ",".join(row["family"] for row in rows),
        "total_refined_area_ratio": sum(row["mask_area_ratio"] for row in rows),
        "min_visible_change_ratio": min(row["visible_change_ratio"] for row in rows),
        "mean_gray_difference": float(np.mean([row["gray_difference_mean"] for row in rows])),
        "status": "PASS",
    }
    return pair, rows, image_row, rejected, failures, "PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-source-root", type=Path, required=True)
    parser.add_argument("--approved-physical-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 0])
    parser.add_argument("--mode", choices=["debug", "formal"], default="debug")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.image_size != 640:
        raise RuntimeError("PREVIEW_IMAGE_SIZE_MUST_BE_640")
    if args.limit < 0 or args.limit > 668:
        raise RuntimeError("LIMIT_OUT_OF_RANGE")
    if args.mode == "formal" and args.limit:
        raise RuntimeError("FORMAL_MODE_FORBIDS_LIMIT")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise RuntimeError("INVALID_SEED_LIST")
    if Path(args.run_name).name != args.run_name:
        raise RuntimeError("INVALID_RUN_NAME")
    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    train_dir = args.train_dir.resolve()
    source_root = args.approved_source_root.resolve()
    physical_root = args.approved_physical_source_root.resolve()
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    results_root = (repo_root / "results").resolve()
    if not path_is_within(output_root, results_root):
        raise RuntimeError("OUTPUT_ROOT_OUTSIDE_REPO_RESULTS")
    if path_is_within(output_root, source_root) or path_is_within(output_root, physical_root):
        raise RuntimeError("OUTPUT_ROOT_OVERLAPS_SOURCE")
    expected_stems = load_manifest_train_stems(manifest_path)
    if len(expected_stems) != 668:
        raise RuntimeError("MANIFEST_TRAIN_COUNT_GATE")
    images = collect_train_images(train_dir, expected_stems, source_root, physical_root)
    selected = images[:args.limit] if args.limit else images
    output_dir = output_root / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_OUTPUT:{output_dir}")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)

    all_edits: list[dict] = []
    all_images: list[dict] = []
    all_rejected: list[dict] = []
    all_failures: list[dict] = []
    per_image = []
    for seed in args.seeds:
        for image_path in selected:
            pair, rows, image_row, rejected, failures, status = generate_pair(image_path, seed, args.image_size)
            all_edits.extend(rows)
            all_rejected.extend(rejected)
            all_failures.extend(failures)
            if pair is not None and status == "PASS":
                output_path = image_dir / f"seed{seed:02d}_{image_path.stem}.jpg"
                pair.save(output_path, quality=88)
                generated_output = output_path.relative_to(output_dir).as_posix()
            else:
                generated_output = None
            all_images.append(image_row)
            per_image.append({
                "seed": seed,
                "source_image": str(image_path),
                "generated_output": generated_output,
                "edit_type": image_row.get("families", ""),
                "edit_types": image_row.get("families", "").split(",") if image_row.get("families") else [],
                "edit_count": image_row["edit_count"],
                "status": status,
                "failures": failures,
            })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failed_images = sum(record["status"] != "PASS" for record in per_image)
    hard_reasons = []
    if args.mode == "formal" and len(selected) != 668:
        hard_reasons.append("FORMAL_IMAGE_COUNT_GATE")
    if failed_images:
        hard_reasons.append(f"FAILED_IMAGE_COUNT:{failed_images}")
    if all_failures:
        hard_reasons.append(f"FAILED_EDIT_COUNT:{len(all_failures)}")
    status = "PASS" if not hard_reasons else "FAIL"
    formal_gate = args.mode == "formal" and status == "PASS"
    input_audit = {
        "status": "PASS",
        "train_dir": str(train_dir),
        "approved_source_root": str(source_root),
        "approved_physical_source_root": str(physical_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_train_stem_count": len(expected_stems),
        "frozen_train_image_count": len(images),
        "image_count": len(selected),
        "input_image_limit": args.limit,
        "manifest_stem_set_match": True,
        "forbidden_split_path_count": 0,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    run_config = {
        "scope": "PHASE_C1_R1_LOCAL_CHANGE_V2_REPAIR_PREVIEW",
        "generator_version": V2_VERSION,
        "mode": args.mode,
        "seeds": args.seeds,
        "image_size": args.image_size,
        "families": FAMILIES,
        "family_probability": FAMILY_PROBABILITY,
        "max_attempts": MAX_ATTEMPTS,
        "C_acceptance": C_ACCEPTANCE,
        "D_acceptance": D_ACCEPTANCE,
        "A_B_status": "V1_UNCHANGED",
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    generation_manifest = {
        "schema_version": 2,
        "run_id": args.run_name,
        "timestamp": timestamp,
        "mode": args.mode,
        "formal_gate": formal_gate,
        "generator_version": V2_VERSION,
        "seed_policy": "DETERMINISTIC_PER_IMAGE_WITH_SEED_PREFIX",
        "seed": args.seeds[0],
        "seeds": args.seeds,
        "source_root": str(source_root),
        "physical_source_root": str(physical_root),
        "train_dir": str(train_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": input_audit["manifest_sha256"],
        "image_count": len(selected),
        "edit_count": len(all_edits),
        "failed_edit_count": len(all_failures),
        "failed_source_image_count": 0,
        "rejected_candidate_count": len(all_rejected),
        "per_image": per_image,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    final_status = {
        "schema_version": 2,
        "status": status,
        "mode": args.mode,
        "formal_gate": formal_gate,
        "run_id": args.run_name,
        "generator_version": V2_VERSION,
        "image_count": len(selected),
        "seed": args.seeds[0],
        "seeds": args.seeds,
        "failed_edits": len(all_failures),
        "failed_images": failed_images,
        "rejected_candidate_count": len(all_rejected),
        "hard_gate_reasons": hard_reasons,
        "val_images_seen": 0,
        "test_images_seen": 0,
        "test_accessed": False,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "input_audit.json", input_audit)
    write_json(output_dir / "generation_manifest.json", generation_manifest)
    write_json(output_dir / "final_status.json", final_status)
    write_csv(output_dir / "edit_metrics.csv", all_edits)
    write_csv(output_dir / "image_metrics.csv", all_images)
    write_csv(output_dir / "rejected_candidates.csv", all_rejected)
    write_csv(output_dir / "failed_attempts.csv", all_failures)
    print(json.dumps({
        "status": status,
        "formal_gate": formal_gate,
        "output_dir": str(output_dir),
        "pair_count": sum(record["status"] == "PASS" for record in per_image),
        "edit_count": len(all_edits),
        "rejected_candidate_count": len(all_rejected),
        "failed_edit_count": len(all_failures),
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
