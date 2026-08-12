#!/usr/bin/env python3
"""Read-only Exp00.1--00.3 audit for polygon annotation datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ExifTags


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PHASH_DISTANCE = 6
DHASH_DISTANCE = 8
CORRELATION_MIN = 0.98
TINY_RELATIVE_AREA = 0.001


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a

    def groups(self, minimum: int = 2) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            grouped[self.find(item)].append(item)
        return sorted(
            (sorted(group) for group in grouped.values() if len(group) >= minimum),
            key=lambda group: (-len(group), group[0]),
        )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos((math.pi / size) * (x + 0.5) * k)
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


DCT32 = dct_matrix(32)


def image_fingerprints(image: Image.Image) -> tuple[int, int, np.ndarray]:
    gray32 = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    dct = DCT32 @ gray32 @ DCT32.T
    low = dct[:8, :8]
    threshold = np.median(low.ravel()[1:])
    phash = 0
    for bit in (low >= threshold).ravel():
        phash = (phash << 1) | int(bit)

    gray9 = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = 0
    for bit in (gray9[:, 1:] >= gray9[:, :-1]).ravel():
        dhash = (dhash << 1) | int(bit)

    feature = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    feature -= feature.mean()
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return phash, dhash, feature


def clean_points(raw: Any) -> tuple[list[tuple[float, float]], bool, int]:
    points: list[tuple[float, float]] = []
    nonfinite = False
    invalid = 0
    if not isinstance(raw, list):
        return points, False, 1
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            invalid += 1
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            nonfinite = True
            continue
        current = (x, y)
        if not points or current != points[-1]:
            points.append(current)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points, nonfinite, invalid


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2.0


def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(p1: tuple[float, float], q1: tuple[float, float], p2: tuple[float, float], q2: tuple[float, float]) -> bool:
    o1, o2 = orientation(p1, q1, p2), orientation(p1, q1, q2)
    o3, o4 = orientation(p2, q2, p1), orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(p1, p2, q1))
        or (o2 == 0 and on_segment(p1, q2, q1))
        or (o3 == 0 and on_segment(p2, p1, q2))
        or (o4 == 0 and on_segment(p2, q1, q2))
    )


def self_intersects(points: list[tuple[float, float]]) -> bool:
    size = len(points)
    if size < 4:
        return False
    for i in range(size):
        a1, a2 = points[i], points[(i + 1) % size]
        for j in range(i + 1, size):
            if j == i or (j + 1) % size == i or (i + 1) % size == j:
                continue
            b1, b2 = points[j], points[(j + 1) % size]
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False


def canonical_polygon(label: str, points: list[tuple[float, float]]) -> str:
    rounded = [(round(x, 6), round(y, 6)) for x, y in points]
    if not rounded:
        return f"{label}|"
    rotations = [rounded[i:] + rounded[:i] for i in range(len(rounded))]
    reverse = list(reversed(rounded))
    rotations.extend(reverse[i:] + reverse[:i] for i in range(len(reverse)))
    return f"{label}|{min(rotations)!r}"


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p90": None, "p95": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def numeric_stem_runs(images: list[Path], minimum: int = 3) -> list[dict[str, Any]]:
    values: list[tuple[int, str]] = []
    for path in images:
        if re.fullmatch(r"\d+", path.stem):
            values.append((int(path.stem), path.name))
    values.sort()
    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for item in values:
        if not current or item[0] == current[-1][0] + 1:
            current.append(item)
        else:
            if len(current) >= minimum:
                runs.append(current)
            current = [item]
    if len(current) >= minimum:
        runs.append(current)
    return [{"start": run[0][0], "end": run[-1][0], "count": len(run), "first_file": run[0][1], "last_file": run[-1][1]} for run in runs]


def safe_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: safe_relative(p, root).casefold())
    images = [path for path in all_files if path.suffix.lower() in IMAGE_SUFFIXES]
    json_files = [path for path in all_files if path.suffix.lower() == ".json"]
    other_files = [path for path in all_files if path not in images and path not in json_files]

    stem_images: dict[str, list[Path]] = defaultdict(list)
    stem_json: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        stem_images[path.stem.casefold()].append(path)
    for path in json_files:
        stem_json[path.stem.casefold()].append(path)

    image_without_json = [path for stem, paths in stem_images.items() if stem not in stem_json for path in paths]
    json_without_image = [path for stem, paths in stem_json.items() if stem not in stem_images for path in paths]
    ambiguous_image_stems = {stem: paths for stem, paths in stem_images.items() if len(paths) > 1}
    ambiguous_json_stems = {stem: paths for stem, paths in stem_json.items() if len(paths) > 1}

    pairs: list[tuple[Path, Path]] = []
    for stem in sorted(set(stem_images) & set(stem_json)):
        image_candidates = stem_images[stem]
        json_candidates = stem_json[stem]
        if len(image_candidates) == len(json_candidates) == 1:
            pairs.append((image_candidates[0], json_candidates[0]))
            continue
        unused = set(image_candidates)
        for json_path in json_candidates:
            same_parent = [path for path in unused if path.parent == json_path.parent]
            if len(same_parent) == 1:
                pairs.append((same_parent[0], json_path))
                unused.remove(same_parent[0])

    image_meta: dict[Path, dict[str, Any]] = {}
    decode_failed: list[dict[str, str]] = []
    fingerprint_paths: list[Path] = []
    features: list[np.ndarray] = []
    phashes: list[int] = []
    dhashes: list[int] = []
    sha_by_path: dict[Path, str] = {}
    exif_datetime: Counter[str] = Counter()
    extension_counts = Counter(path.suffix for path in images)
    dimension_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    for index, path in enumerate(images, 1):
        relative = safe_relative(path, root)
        try:
            digest = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                phash, dhash, feature = image_fingerprints(image)
                exif = image.getexif()
                for key, value in exif.items():
                    if ExifTags.TAGS.get(key) in {"DateTime", "DateTimeOriginal", "DateTimeDigitized"}:
                        exif_datetime[str(value)] += 1
            sha_by_path[path] = digest
            image_meta[path] = {"width": width, "height": height, "mode": mode, "sha256": digest}
            dimension_counts[f"{width}x{height}"] += 1
            mode_counts[mode] += 1
            fingerprint_paths.append(path)
            phashes.append(phash)
            dhashes.append(dhash)
            features.append(feature)
        except Exception as exc:
            decode_failed.append({"image_path": relative, "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0:
            print(f"images_scanned={index}/{len(images)}", flush=True)

    json_data: dict[Path, Any] = {}
    json_parse_failed: list[dict[str, str]] = []
    top_field_counts: Counter[str] = Counter()
    version_counts: Counter[str] = Counter()
    shape_type_counts: Counter[str] = Counter()
    shape_field_counts: Counter[str] = Counter()
    group_id_presence = Counter()
    image_data_presence = Counter()
    image_path_values: list[str] = []
    schema_sample_rows: list[dict[str, Any]] = []

    sample_indices = set(np.linspace(0, max(0, len(json_files) - 1), min(20, len(json_files)), dtype=int).tolist())
    for index, path in enumerate(json_files):
        relative = safe_relative(path, root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            json_data[path] = payload
            if not isinstance(payload, dict):
                raise ValueError(f"top level is {type(payload).__name__}, expected object")
            top_field_counts.update(payload.keys())
            version_counts[str(payload.get("version", "<missing>"))] += 1
            image_data_presence["present_non_null" if payload.get("imageData") is not None else "missing_or_null"] += 1
            if "imagePath" in payload:
                image_path_values.append(str(payload.get("imagePath")))
            shapes = payload.get("shapes", [])
            if isinstance(shapes, list):
                for shape in shapes:
                    if isinstance(shape, dict):
                        shape_field_counts.update(shape.keys())
                        shape_type_counts[str(shape.get("shape_type", "<missing>"))] += 1
                        group_id_presence["non_null" if shape.get("group_id") is not None else "null_or_missing"] += 1
                    else:
                        shape_type_counts["<non_object>"] += 1
            if index in sample_indices:
                schema_sample_rows.append({
                    "json_path": relative,
                    "top_level_type": type(payload).__name__,
                    "top_level_fields": "|".join(sorted(payload.keys())),
                    "version": payload.get("version"),
                    "imagePath": payload.get("imagePath"),
                    "imageWidth": payload.get("imageWidth"),
                    "imageHeight": payload.get("imageHeight"),
                    "shapes_count": len(shapes) if isinstance(shapes, list) else None,
                    "imageData_non_null": payload.get("imageData") is not None,
                })
        except Exception as exc:
            json_parse_failed.append({"json_path": relative, "error": f"{type(exc).__name__}: {exc}"})

    class_instances: Counter[str] = Counter()
    class_images: Counter[str] = Counter()
    image_labels: dict[str, set[str]] = {}
    per_image_instances: list[int] = []
    empty_json: list[str] = []
    instance_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    class_vertex_values: dict[str, list[float]] = defaultdict(list)
    class_polygon_area_values: dict[str, list[float]] = defaultdict(list)
    class_bbox_area_values: dict[str, list[float]] = defaultdict(list)
    class_relative_area_values: dict[str, list[float]] = defaultdict(list)
    class_aspect_values: dict[str, list[float]] = defaultdict(list)
    image_mask_coverage: dict[str, float] = {}
    matched_json_set = {json_path for _, json_path in pairs}
    image_for_json = {json_path: image_path for image_path, json_path in pairs}

    for json_path, payload in json_data.items():
        relative_json = safe_relative(json_path, root)
        shapes = payload.get("shapes", []) if isinstance(payload, dict) else []
        if not isinstance(shapes, list):
            issue_rows.append({"json_path": relative_json, "shape_index": "", "label": "", "issue_type": "shapes_not_list", "detail": type(shapes).__name__})
            shapes = []
        if not shapes:
            empty_json.append(relative_json)
        per_image_instances.append(len(shapes))
        labels_here: set[str] = set()
        image_path = image_for_json.get(json_path)
        meta = image_meta.get(image_path) if image_path else None
        true_width = meta["width"] if meta else None
        true_height = meta["height"] if meta else None
        declared_width = payload.get("imageWidth") if isinstance(payload, dict) else None
        declared_height = payload.get("imageHeight") if isinstance(payload, dict) else None
        if meta and (declared_width != true_width or declared_height != true_height):
            issue_rows.append({"json_path": relative_json, "shape_index": "", "label": "", "issue_type": "image_dimension_mismatch", "detail": f"declared={declared_width}x{declared_height}; actual={true_width}x{true_height}"})
        seen_polygons: dict[str, int] = {}
        summed_area = 0.0
        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                issue_rows.append({"json_path": relative_json, "shape_index": shape_index, "label": "", "issue_type": "shape_not_object", "detail": type(shape).__name__})
                continue
            label = str(shape.get("label", ""))
            labels_here.add(label)
            class_instances[label] += 1
            points, nonfinite, invalid_points = clean_points(shape.get("points"))
            shape_type = str(shape.get("shape_type", "<missing>"))
            issue_types: list[str] = []
            if shape_type != "polygon":
                issue_types.append("non_polygon_shape")
            if not label:
                issue_types.append("empty_label")
            if nonfinite:
                issue_types.append("nan_or_inf_coordinate")
            if invalid_points:
                issue_types.append("invalid_point_structure")
            if len(set(points)) < 3:
                issue_types.append("fewer_than_3_unique_points")
            area = polygon_area(points)
            if area <= 0:
                issue_types.append("zero_area_polygon")
            intersects = self_intersects(points)
            if intersects:
                issue_types.append("self_intersection")
            out_of_bounds = False
            if true_width is not None and true_height is not None:
                out_of_bounds = any(x < 0 or y < 0 or x > true_width or y > true_height for x, y in points)
                if out_of_bounds:
                    issue_types.append("polygon_out_of_bounds")
            polygon_key = canonical_polygon(label, points)
            if polygon_key in seen_polygons:
                issue_types.append("duplicate_polygon")
            else:
                seen_polygons[polygon_key] = shape_index
            for issue_type in issue_types:
                issue_rows.append({"json_path": relative_json, "shape_index": shape_index, "label": label, "issue_type": issue_type, "detail": ""})

            if points:
                xs, ys = zip(*points)
                bbox_width, bbox_height = max(xs) - min(xs), max(ys) - min(ys)
                bbox_area = max(0.0, bbox_width) * max(0.0, bbox_height)
                aspect = bbox_width / bbox_height if bbox_height > 0 else None
            else:
                bbox_width = bbox_height = bbox_area = 0.0
                aspect = None
            image_area = float(true_width * true_height) if true_width and true_height else None
            relative_area = area / image_area if image_area else None
            relative_bbox_area = bbox_area / image_area if image_area else None
            tiny = relative_area is not None and relative_area < TINY_RELATIVE_AREA
            summed_area += area
            instance_rows.append({
                "json_path": relative_json,
                "image_path": safe_relative(image_path, root) if image_path else "",
                "shape_index": shape_index,
                "label": label,
                "shape_type": shape_type,
                "vertex_count": len(points),
                "polygon_area_px": area,
                "bbox_area_px": bbox_area,
                "relative_polygon_area": relative_area,
                "relative_bbox_area": relative_bbox_area,
                "bbox_aspect_ratio": aspect,
                "is_tiny_lt_0_001": tiny,
                "has_issue": bool(issue_types),
                "issue_types": "|".join(issue_types),
            })
            class_vertex_values[label].append(float(len(points)))
            class_polygon_area_values[label].append(area)
            class_bbox_area_values[label].append(bbox_area)
            if relative_area is not None:
                class_relative_area_values[label].append(relative_area)
            if aspect is not None:
                class_aspect_values[label].append(aspect)
        for label in labels_here:
            class_images[label] += 1
        image_labels[relative_json] = labels_here
        if true_width and true_height:
            image_mask_coverage[relative_json] = summed_area / (true_width * true_height)

    class_names = sorted(class_instances)
    class_id = {label: index for index, label in enumerate(class_names)}
    total_instances = sum(class_instances.values())
    annotated_image_count = len([labels for labels in image_labels.values() if labels])
    class_rows: list[dict[str, Any]] = []
    for label in class_names:
        class_rows.append({
            "class_id": class_id[label],
            "class_name": label,
            "image_count": class_images[label],
            "instance_count": class_instances[label],
            "percentage_images": 100 * class_images[label] / annotated_image_count if annotated_image_count else 0,
            "percentage_instances": 100 * class_instances[label] / total_instances if total_instances else 0,
            "tiny_instance_count": sum(value < TINY_RELATIVE_AREA for value in class_relative_area_values[label]),
            "tiny_instance_percentage": 100 * sum(value < TINY_RELATIVE_AREA for value in class_relative_area_values[label]) / len(class_relative_area_values[label]) if class_relative_area_values[label] else 0,
            "vertices_median": describe(class_vertex_values[label])["median"],
            "polygon_area_median_px": describe(class_polygon_area_values[label])["median"],
            "relative_area_median": describe(class_relative_area_values[label])["median"],
            "bbox_aspect_median": describe(class_aspect_values[label])["median"],
        })

    cooccurrence_rows: list[dict[str, Any]] = []
    for left in class_names:
        row: dict[str, Any] = {"class_name": left}
        for right in class_names:
            row[right] = sum(left in labels and right in labels for labels in image_labels.values())
        cooccurrence_rows.append(row)

    sha_groups: dict[str, list[Path]] = defaultdict(list)
    for path, digest in sha_by_path.items():
        sha_groups[digest].append(path)
    exact_groups = sorted((paths for paths in sha_groups.values() if len(paths) >= 2), key=lambda group: (-len(group), safe_relative(group[0], root)))
    exact_rows: list[dict[str, Any]] = []
    exact_uf = UnionFind([safe_relative(path, root) for path in fingerprint_paths])
    for group_index, group in enumerate(exact_groups, 1):
        rels = [safe_relative(path, root) for path in group]
        for rel in rels[1:]:
            exact_uf.union(rels[0], rel)
        for path in group:
            exact_rows.append({"group_id": f"exact_{group_index:04d}", "group_size": len(group), "sha256": sha_by_path[path], "image_path": safe_relative(path, root), "stem": path.stem})

    near_uf = UnionFind([safe_relative(path, root) for path in fingerprint_paths])
    near_pair_rows: list[dict[str, Any]] = []
    feature_matrix = np.stack(features) if features else np.empty((0, 1024), dtype=np.float32)
    for i in range(len(fingerprint_paths)):
        if i % 100 == 0:
            print(f"near_duplicate_scan={i}/{len(fingerprint_paths)}", flush=True)
        correlations = feature_matrix[i + 1:] @ feature_matrix[i] if len(feature_matrix) > i + 1 else np.array([])
        for offset, correlation in enumerate(correlations, start=i + 1):
            phash_distance = hamming(phashes[i], phashes[offset])
            if phash_distance > PHASH_DISTANCE:
                continue
            dhash_distance = hamming(dhashes[i], dhashes[offset])
            correlation_value = float(correlation)
            if correlation_value < CORRELATION_MIN or dhash_distance > DHASH_DISTANCE:
                continue
            left = safe_relative(fingerprint_paths[i], root)
            right = safe_relative(fingerprint_paths[offset], root)
            if sha_by_path[fingerprint_paths[i]] == sha_by_path[fingerprint_paths[offset]]:
                continue
            near_uf.union(left, right)
            near_pair_rows.append({
                "image_a": left,
                "image_b": right,
                "phash_distance": phash_distance,
                "dhash_distance": dhash_distance,
                "gray_correlation": correlation_value,
            })
    near_groups = near_uf.groups()
    near_group_for_path: dict[str, str] = {}
    for group_index, group in enumerate(near_groups, 1):
        for path in group:
            near_group_for_path[path] = f"near_{group_index:04d}"
    near_pair_output_rows: list[dict[str, Any]] = []
    for row in near_pair_rows:
        near_pair_output_rows.append({
            "group_id": near_group_for_path[row["image_a"]],
            **row,
        })
    fingerprint_index = {safe_relative(path, root): index for index, path in enumerate(fingerprint_paths)}
    near_member_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(near_groups, 1):
        group_id = f"near_{group_index:04d}"
        for relative in group:
            index = fingerprint_index[relative]
            path = fingerprint_paths[index]
            near_member_rows.append({
                "group_id": group_id,
                "group_size": len(group),
                "image_path": relative,
                "stem": path.stem,
                "sha256": sha_by_path[path],
                "phash_hex": f"{phashes[index]:016x}",
                "dhash_hex": f"{dhashes[index]:016x}",
            })

    numeric_runs = numeric_stem_runs(images)
    numeric_stems = sum(bool(re.fullmatch(r"\d+", path.stem)) for path in images)
    prefix_counts: Counter[str] = Counter()
    for path in images:
        match = re.match(r"^(.*?)(?:[_-]?\d+)$", path.stem)
        if match and match.group(1):
            prefix_counts[match.group(1)] += 1

    valid_relative_areas = [row["relative_polygon_area"] for row in instance_rows if row["relative_polygon_area"] is not None]
    size_quantiles = describe(valid_relative_areas)
    size_thresholds = {"tiny_max_q25": size_quantiles["p25"], "small_max_q50": size_quantiles["median"], "medium_max_q75": size_quantiles["p75"], "large_above_q75": size_quantiles["p75"]}
    issue_counts = Counter(row["issue_type"] for row in issue_rows)
    shape_count_per_image = Counter(per_image_instances)
    single_class_images = sum(len(labels) == 1 for labels in image_labels.values())
    multi_class_images = sum(len(labels) > 1 for labels in image_labels.values())
    nonempty_instances = list(class_instances.values())
    imbalance_ratio = max(nonempty_instances) / min(nonempty_instances) if nonempty_instances and min(nonempty_instances) else None

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "audit_parameters": {"image_suffixes_case_insensitive": sorted(IMAGE_SUFFIXES), "phash_distance_max": PHASH_DISTANCE, "dhash_distance_max": DHASH_DISTANCE, "gray_correlation_min": CORRELATION_MIN, "tiny_relative_polygon_area_lt": TINY_RELATIVE_AREA},
        "files": {
            "total_files": len(all_files), "image_count": len(images), "json_count": len(json_files), "other_file_count": len(other_files),
            "matched_pair_count": len(pairs), "image_without_json_count": len(image_without_json), "json_without_image_count": len(json_without_image),
            "decode_failed_images_count": len(decode_failed), "json_parse_failed_count": len(json_parse_failed),
            "ambiguous_image_stem_count": len(ambiguous_image_stems), "ambiguous_json_stem_count": len(ambiguous_json_stems),
            "extension_counts": dict(extension_counts), "dimension_counts": dict(dimension_counts), "mode_counts": dict(mode_counts),
        },
        "schema": {
            "sampled_json_count": len(schema_sample_rows), "top_level_field_counts": dict(top_field_counts), "version_counts": dict(version_counts),
            "shape_field_counts": dict(shape_field_counts), "shape_type_counts": dict(shape_type_counts), "group_id_presence": dict(group_id_presence),
            "image_data_presence": dict(image_data_presence), "imagePath_absolute_count": sum(os.path.isabs(value) for value in image_path_values),
        },
        "annotations": {
            "actual_class_count": len(class_names), "class_names": class_names, "total_instances": total_instances,
            "annotated_image_count": annotated_image_count, "empty_json_count": len(empty_json), "single_class_image_count": single_class_images,
            "multi_class_image_count": multi_class_images, "class_imbalance_max_to_min_ratio": imbalance_ratio,
            "instances_per_image": describe([float(value) for value in per_image_instances]), "instances_per_image_histogram": dict(sorted(shape_count_per_image.items())),
            "relative_polygon_area": describe(valid_relative_areas), "mask_coverage_sum_per_image": describe(list(image_mask_coverage.values())),
            "size_quantile_thresholds": size_thresholds, "polygon_issue_counts": dict(issue_counts), "polygon_issue_row_count": len(issue_rows),
        },
        "duplicates": {
            "exact_group_count": len(exact_groups), "exact_image_count": sum(len(group) for group in exact_groups),
            "near_group_count": len(near_groups), "near_image_count": sum(len(group) for group in near_groups), "near_pair_count": len(near_pair_rows),
        },
        "source_clues": {
            "numeric_stem_image_count": numeric_stems, "numeric_stem_percentage": 100 * numeric_stems / len(images) if images else 0,
            "numeric_consecutive_runs": numeric_runs, "filename_prefix_counts": dict(prefix_counts.most_common(20)),
            "exif_datetime_value_counts": dict(exif_datetime.most_common(20)),
        },
        "lists": {
            "other_files": [safe_relative(path, root) for path in other_files],
            "image_without_json": [safe_relative(path, root) for path in image_without_json],
            "json_without_image": [safe_relative(path, root) for path in json_without_image],
            "empty_json": empty_json,
        },
    }

    write_csv(output / "class_stats.csv", list(class_rows[0].keys()) if class_rows else ["class_id", "class_name"], class_rows)
    write_csv(output / "instance_stats.csv", list(instance_rows[0].keys()) if instance_rows else ["json_path"], instance_rows)
    write_csv(output / "cooccurrence.csv", ["class_name", *class_names], cooccurrence_rows)
    write_csv(output / "polygon_issues.csv", ["json_path", "shape_index", "label", "issue_type", "detail"], issue_rows)
    write_csv(output / "schema_samples.csv", list(schema_sample_rows[0].keys()) if schema_sample_rows else ["json_path"], schema_sample_rows)
    write_csv(output / "decode_failed_images.csv", ["image_path", "error"], decode_failed)
    write_csv(output / "json_parse_failed.csv", ["json_path", "error"], json_parse_failed)
    write_csv(output / "missing_pairs.csv", ["type", "path"], ([{"type": "image_without_json", "path": safe_relative(path, root)} for path in image_without_json] + [{"type": "json_without_image", "path": safe_relative(path, root)} for path in json_without_image]))
    write_csv(output / "duplicate_groups.csv", ["group_id", "group_size", "sha256", "image_path", "stem"], exact_rows)
    write_csv(output / "near_duplicate_groups.csv", ["group_id", "group_size", "image_path", "stem", "sha256", "phash_hex", "dhash_hex"], near_member_rows)
    write_csv(output / "near_duplicate_pairs.csv", ["group_id", "image_a", "image_b", "phash_distance", "dhash_distance", "gray_correlation"], near_pair_output_rows)
    write_csv(output / "image_hashes.csv", ["image_path", "sha256", "phash_hex", "dhash_hex"], ({"image_path": safe_relative(path, root), "sha256": sha_by_path[path], "phash_hex": f"{phashes[index]:016x}", "dhash_hex": f"{dhashes[index]:016x}"} for index, path in enumerate(fingerprint_paths)))
    json_dump(output / "dataset_summary.json", summary)

    def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
        header = "| " + " | ".join(fields) + " |"
        separator = "|" + "|".join("---" for _ in fields) + "|"
        body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
        return "\n".join([header, separator, *body])

    required_before_exp01: list[str] = []
    if image_without_json or json_without_image:
        required_before_exp01.append("确认缺失配对样本的处置（排除、补标或找回文件），禁止静默纳入训练。")
    if decode_failed or json_parse_failed:
        required_before_exp01.append("处理无法解码的图片或无法解析的 JSON。")
    if issue_counts:
        required_before_exp01.append("逐项复核 polygon 异常；退化、非有限坐标和无法可靠转换的标注必须排除并留痕。")
    if exact_groups or near_groups:
        required_before_exp01.append("冻结 duplicate/near-duplicate group，同组样本必须整体进入同一 split。")
    if not required_before_exp01:
        required_before_exp01.append("未发现阻塞性数据问题；Exp01 仍需进行至少 50 张 overlay 反向验证。")

    report = f"""# Exp00 数据集完整审计

生成时间（UTC）：`{summary['generated_at_utc']}`  
原始数据：`{root}`（全程只读）  
审计脚本：`tools/dataset/audit_dataset.py`

## 结论摘要

- 图片：**{len(images)}**；JSON：**{len(json_files)}**；按 stem 成功唯一配对：**{len(pairs)}**。
- 实际类别：**{len(class_names)} 类**：{', '.join(f'`{label}`' for label in class_names)}。
- 实例总数：**{total_instances}**；最多类/最少类实例比：**{imbalance_ratio:.2f}:1**。
- 图片缺 JSON：**{len(image_without_json)}**；JSON 缺图片：**{len(json_without_image)}**；图片解码失败：**{len(decode_failed)}**；JSON 解析失败：**{len(json_parse_failed)}**。
- polygon 问题记录：**{len(issue_rows)}** 条；空标注 JSON：**{len(empty_json)}**。
- exact duplicate：**{len(exact_groups)} 组 / {sum(len(group) for group in exact_groups)} 张**；near duplicate：**{len(near_groups)} 组 / {sum(len(group) for group in near_groups)} 张 / {len(near_pair_rows)} 对**。

## 类别与实例

{md_table(class_rows, ['class_id', 'class_name', 'image_count', 'instance_count', 'percentage_instances', 'tiny_instance_percentage', 'relative_area_median'])}

类别 ID 只是本次审计中的稳定字典序展示，尚未执行 Exp01，未冻结 `class_mapping.yaml`。

## 文件与 JSON schema

- 图片后缀（保留原始大小写）：`{dict(extension_counts)}`。
- 图片尺寸：`{dict(dimension_counts)}`。
- JSON 抽样：按排序等间距抽取 **{len(schema_sample_rows)}** 份，详见 `schema_samples.csv`。
- 顶层字段出现次数：`{dict(top_field_counts)}`。
- version：`{dict(version_counts)}`。
- shape_type：`{dict(shape_type_counts)}`。
- imageData：`{dict(image_data_presence)}`；group_id：`{dict(group_id_presence)}`。
- `imagePath` 为绝对路径：**{sum(os.path.isabs(value) for value in image_path_values)}** 份。配对实际按 stem 完成，不依赖该字段。

## Polygon、尺度与标注异常

- 每图实例数：`{summary['annotations']['instances_per_image']}`。
- 单类图：**{single_class_images}**；多类图：**{multi_class_images}**。
- 相对 polygon 面积分布：`{summary['annotations']['relative_polygon_area']}`。
- 极小实例定义：polygon 面积 / 图像面积 `< {TINY_RELATIVE_AREA}`；总计 **{sum(row['is_tiny_lt_0_001'] for row in instance_rows)} / {len(valid_relative_areas)}**。
- 后续尺度分箱建议采用本数据分位数：tiny ≤ q25 `{size_thresholds['tiny_max_q25']}`，small ≤ q50 `{size_thresholds['small_max_q50']}`，medium ≤ q75 `{size_thresholds['medium_max_q75']}`，large > q75。
- 异常类型计数：`{dict(issue_counts)}`。
- `mask_coverage_sum_per_image` 是 polygon 面积求和/图像面积；重叠 polygon 会重复计入，仅用于审计，不等价于 union mask 覆盖率。

完整逐实例数据见 `instance_stats.csv`，异常定位见 `polygon_issues.csv`，类别共现见 `cooccurrence.csv`。

## 重复、近重复和泄漏风险

- exact duplicate 使用文件 SHA256。
- near duplicate 聚类阈值固定为：pHash Hamming ≤ {PHASH_DISTANCE}、dHash Hamming ≤ {DHASH_DISTANCE}，且 32×32 标准化灰度相关性 ≥ {CORRELATION_MIN}；exact pair 不重复计入 near pair。
- 数字 stem 图片：**{numeric_stems}/{len(images)}（{100 * numeric_stems / len(images) if images else 0:.2f}%）**；连续编号段：`{numeric_runs}`。
- 文件名前缀线索：`{dict(prefix_counts.most_common(20))}`；EXIF 时间线索：`{dict(exif_datetime.most_common(20))}`。

大量数字连续命名只能证明存在序列化导出/采集线索，不能单独证明连续帧；视觉近重复组则必须作为 split 的硬分组边界。

## 推荐 split 方案

Exp01 使用固定 `seed=42` 的 **multilabel-stratified + group-aware 70/15/15 split**：

1. exact SHA256、near-duplicate 连通组整体绑定到同一 `group_id`；
2. 若原始来源能补充视频/发动机/工件/采集批次 ID，应优先用真实来源组覆盖视觉推断组；
3. 以每图多标签向量做分层，使所有类别，尤其稀有类，尽量覆盖 val/test；
4. test 一次冻结，后续不参与阈值、hard mining、SSL 或模型选择；
5. 生成并哈希唯一 `split_manifest.csv`。

## 进入 Exp01 前必须处理

{chr(10).join(f'{index}. {item}' for index, item in enumerate(required_before_exp01, 1))}

此外，建议向数据提供方索取真实采集来源字段（视频/发动机/部位/时间段）。仅凭数字文件名与视觉哈希无法完全排除跨序列泄漏。

## 下一步（尚未执行）

Exp01 将冻结类别映射，转换合法 polygon，排除项逐条留痕，创建 group-aware split，并对至少 50 张样本反向 rasterize overlay 验证。当前报告完成后必须先停止并由用户确认。
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "image_count": len(images), "json_count": len(json_files), "class_count": len(class_names), "instance_count": total_instances, "issue_count": len(issue_rows), "exact_group_count": len(exact_groups), "near_group_count": len(near_groups)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
