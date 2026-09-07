#!/usr/bin/env python3
"""Read-only Gate C audit for an existing Phase C-1 preview run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


FAMILIES = [
    "A_small_low_contrast",
    "B_diffuse_texture",
    "C_elongated_ribbon",
    "D_irregular_boundary",
]
CLASS_NAMES = [
    "Burn",
    "Crack",
    "Dent",
    "Material missing",
    "Tears",
    "Tip curl",
    "corrosion",
]


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
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, probability))


def statistics(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p1": quantile(values, 0.01),
        "p5": quantile(values, 0.05),
        "p95": quantile(values, 0.95),
    }


def load_edit_metrics(path: Path, image_size: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = [
        "visible_change_ratio",
        "before_change_pixels",
        "after_change_pixels",
        "delta_pixels",
        "bbox_aspect_ratio",
        "attempt",
        "mask_area_ratio",
        "relative_bbox_width",
        "relative_bbox_height",
        "perimeter",
        "compactness",
        "connected_components",
        "gray_difference_mean",
        "gray_difference_median",
        "gray_difference_max",
        "rgb_difference_mean",
        "pct_diff_ge_2",
        "pct_diff_ge_5",
        "pct_diff_ge_10",
        "single_edit_total_changed_pixels",
    ]
    for row in rows:
        row["edit_index"] = int(row["edit_index"])
        row["seed"] = int(row["seed"])
        for field in numeric_fields:
            row[field] = float(row[field])
        row["connected_components"] = int(row["connected_components"])
        row["before_change_pixels"] = int(row["before_change_pixels"])
        row["after_change_pixels"] = int(row["after_change_pixels"])
        row["delta_pixels"] = int(row["delta_pixels"])
        row["attempt"] = int(row["attempt"])
        row["single_edit_total_changed_pixels"] = int(row["single_edit_total_changed_pixels"])
        row["changed_pixel_ratio"] = row["single_edit_total_changed_pixels"] / (
            image_size * image_size
        )
        row["largest_component_ratio_lower_bound"] = row["mask_area_ratio"] / max(
            row["connected_components"], 1
        )
    return rows


def load_split_classes(manifest_path: Path, expected_count: int) -> tuple[dict[str, set[str]], list[str]]:
    stem_classes: dict[str, set[str]] = {}
    integrity_errors: list[str] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() != "train":
                continue
            stem = row["stem"].strip()
            if stem in stem_classes:
                integrity_errors.append(f"DUPLICATE_TRAIN_STEM:{stem}")
                continue
            labels = {part.strip() for part in row["labels_present"].split("|") if part.strip()}
            invalid = sorted(labels - set(CLASS_NAMES))
            if invalid:
                integrity_errors.append(f"INVALID_TRAIN_LABELS:{stem}:{','.join(invalid)}")
            if not labels:
                integrity_errors.append(f"EMPTY_TRAIN_LABELS:{stem}")
            stem_classes[stem] = labels
    if len(stem_classes) != expected_count:
        integrity_errors.append(f"TRAIN_STEM_COUNT:{len(stem_classes)}!={expected_count}")
    return stem_classes, integrity_errors


def load_image_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["edit_count"] = int(row["edit_count"])
        row["total_refined_area_ratio"] = float(row["total_refined_area_ratio"])
        row["min_visible_change_ratio"] = float(row["min_visible_change_ratio"])
        row["mean_gray_difference"] = float(row["mean_gray_difference"])
        row["families"] = row["families"].split(",") if row["families"] else []
    return rows


def analyze_changed_regions(
    manifest: dict,
    preview_dir: Path,
    image_size: int,
) -> tuple[dict[tuple[int, str], dict], list[str]]:
    total_pixels = image_size * image_size
    half_width = image_size
    right_offset = image_size + 8
    results: dict[tuple[int, str], dict] = {}
    errors: list[str] = []
    for record in manifest["per_image"]:
        stem = Path(record["source_image"]).stem
        output_path = preview_dir / record["generated_output"]
        try:
            pair = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            if pair is None:
                raise RuntimeError("OUTPUT_IMAGE_NOT_READABLE")
            if pair.shape[:2] != (image_size, image_size * 2 + 8):
                raise RuntimeError(f"UNEXPECTED_PREVIEW_SHAPE:{pair.shape}")
            original = pair[:, :half_width]
            changed = pair[:, right_offset:right_offset + half_width]
            gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_changed = cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)
            changed_mask = (np.abs(
                gray_changed.astype(np.int16) - gray_original.astype(np.int16)
            ) >= 2).astype(np.uint8)
            component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
                changed_mask,
                connectivity=8,
            )
            actual_components = max(component_count - 1, 0)
            if actual_components:
                largest_area = int(component_stats[1:, cv2.CC_STAT_AREA].max())
                largest_ratio = largest_area / total_pixels
            else:
                largest_area = 0
                largest_ratio = 0.0
            results[(record["seed"], stem)] = {
                "seed": record["seed"],
                "image_stem": stem,
                "changed_pixel_count": int(np.count_nonzero(changed_mask)),
                "changed_pixel_ratio": int(np.count_nonzero(changed_mask)) / total_pixels,
                "connected_components": actual_components,
                "largest_component_area": largest_area,
                "largest_component_ratio": largest_ratio,
            }
        except Exception as error:
            errors.append(
                f"CHANGED_REGION_ANALYSIS_FAILURE:seed={record['seed']}:stem={stem}:"
                f"{type(error).__name__}:{error}"
            )
    return results, errors


def family_distribution(edit_rows: list[dict]) -> list[dict]:
    counts = Counter(row["family"] for row in edit_rows)
    return [
        {
            "family": family,
            "count": counts.get(family, 0),
            "percentage": counts.get(family, 0) / len(edit_rows) if edit_rows else 0.0,
        }
        for family in FAMILIES
    ]


def visible_change_rows(edit_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in edit_rows:
        groups[row["family"]].append(row)
    groups["ALL"] = edit_rows
    output = []
    for family in [*FAMILIES, "ALL"]:
        rows = groups[family]
        changed_ratios = [row["changed_pixel_ratio"] for row in rows]
        delta_pixels = [row["delta_pixels"] for row in rows]
        changed_pixels = [row["single_edit_total_changed_pixels"] for row in rows]
        mean_differences = [row["gray_difference_mean"] for row in rows]
        visible_ratios = [row["visible_change_ratio"] for row in rows]
        components = [row["connected_components"] for row in rows]
        largest_lower_bounds = [row["largest_component_ratio_lower_bound"] for row in rows]
        row_output = {
            "family": family,
            "edit_count": len(rows),
            "changed_pixel_ratio_min": min(changed_ratios),
            "changed_pixel_ratio_max": max(changed_ratios),
            "changed_pixel_ratio_mean": float(np.mean(changed_ratios)),
            "changed_pixel_ratio_median": float(np.median(changed_ratios)),
            "changed_pixel_ratio_p1": quantile(changed_ratios, 0.01),
            "changed_pixel_ratio_p5": quantile(changed_ratios, 0.05),
            "changed_pixel_ratio_p95": quantile(changed_ratios, 0.95),
            "changed_pixel_count_min": min(changed_pixels),
            "changed_pixel_count_max": max(changed_pixels),
            "changed_pixel_count_mean": float(np.mean(changed_pixels)),
            "delta_pixels_min": min(delta_pixels),
            "delta_pixels_max": max(delta_pixels),
            "delta_pixels_mean": float(np.mean(delta_pixels)),
            "mean_absolute_difference_mean": float(np.mean(mean_differences)),
            "visible_change_ratio_below_0_5_fraction": float(np.mean(
                np.asarray(visible_ratios) < 0.5
            )),
            "connected_components_min": min(components),
            "connected_components_max": max(components),
            "connected_components_mean": float(np.mean(components)),
            "largest_component_ratio_lower_bound_min": min(largest_lower_bounds),
            "largest_component_ratio_lower_bound_max": max(largest_lower_bounds),
            "largest_component_ratio_lower_bound_mean": float(np.mean(largest_lower_bounds)),
        }
        output.append(row_output)
    return output


def class_distribution(
    manifest: dict,
    edit_rows: list[dict],
    stem_classes: dict[str, set[str]],
) -> list[dict]:
    edits_by_preview: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in edit_rows:
        edits_by_preview[(row["seed"], row["image_stem"])].append(row)
    source_stems: set[str] = set()
    for record in manifest["per_image"]:
        source_stems.add(Path(record["source_image"]).stem)
    output = []
    for class_name in CLASS_NAMES:
        class_stems = {stem for stem, labels in stem_classes.items() if class_name in labels}
        class_previews = [
            record
            for record in manifest["per_image"]
            if Path(record["source_image"]).stem in class_stems
        ]
        class_edits = []
        for record in class_previews:
            key = (record["seed"], Path(record["source_image"]).stem)
            class_edits.extend(edits_by_preview[key])
        family_counts = Counter(row["family"] for row in class_edits)
        for family in FAMILIES:
            output.append({
                "class_name": class_name,
                "family": family,
                "source_image_count": len(class_stems),
                "preview_count": len(class_previews),
                "edit_count": len(class_edits),
                "family_count": family_counts.get(family, 0),
                "family_percentage": family_counts.get(family, 0) / len(class_edits) if class_edits else 0.0,
                "morphology_family_coverage_count": sum(family_counts.get(family, 0) > 0 for family in FAMILIES),
            })
    return output


def check_manifest_integrity(
    manifest: dict,
    final_status: dict,
    preview_dir: Path,
    stem_classes: dict[str, set[str]],
    expected_train_count: int,
) -> dict:
    per_image = manifest["per_image"]
    source_paths = [Path(record["source_image"]) for record in per_image]
    source_stems = [path.stem for path in source_paths]
    generated_outputs = [Path(record["generated_output"]) for record in per_image]
    preview_pairs = [(record["seed"], Path(record["source_image"]).stem) for record in per_image]
    source_paths_by_seed: dict[int, list[Path]] = defaultdict(list)
    for record in per_image:
        source_paths_by_seed[record["seed"]].append(Path(record["source_image"]))
    expected_preview_count = manifest["image_count"] * len(manifest["seeds"])
    missing_sources = sorted({str(path) for path in source_paths if not path.is_file()})
    missing_outputs = sorted({
        str(preview_dir / output)
        for output in generated_outputs
        if not (preview_dir / output).is_file()
    })
    unexpected_class_stems = sorted(set(source_stems) - set(stem_classes))
    missing_class_stems = sorted(set(stem_classes) - set(source_stems))
    failed_records = [
        {"seed": record["seed"], "stem": Path(record["source_image"]).stem, "status": record["status"]}
        for record in per_image
        if record["status"] != "PASS"
    ]
    seed_mismatches = sorted({
        record["seed"] for record in per_image if record["seed"] not in manifest["seeds"]
    })
    return {
        "run_id": manifest["run_id"],
        "final_status": final_status["status"],
        "formal_gate": final_status["formal_gate"],
        "manifest_sha256": sha256_file(preview_dir / "generation_manifest.json"),
        "final_status_sha256": sha256_file(preview_dir / "final_status.json"),
        "expected_train_image_count": expected_train_count,
        "manifest_image_count": manifest["image_count"],
        "expected_preview_count": expected_preview_count,
        "manifest_per_image_count": len(per_image),
        "filesystem_preview_count": len([
            path for path in (preview_dir / "images").iterdir() if path.is_file()
        ]),
        "unique_source_image_count": len(set(source_paths)),
        "unique_source_stem_count": len(set(source_stems)),
        "unique_generated_output_count": len(set(generated_outputs)),
        "source_path_repeat_count": len(source_paths) - len(set(source_paths)),
        "duplicate_source_path_within_seed_count": sum(
            len(paths) - len(set(paths))
            for paths in source_paths_by_seed.values()
        ),
        "duplicate_generated_output_count": len(generated_outputs) - len(set(generated_outputs)),
        "duplicate_preview_key_count": len(preview_pairs) - len(set(preview_pairs)),
        "missing_source_image_count": len(missing_sources),
        "missing_source_images": missing_sources,
        "missing_generated_output_count": len(missing_outputs),
        "missing_generated_outputs": missing_outputs,
        "failed_per_image_record_count": len(failed_records),
        "failed_per_image_records": failed_records,
        "seed_mismatches": seed_mismatches,
        "manifest_seed_list": manifest["seeds"],
        "final_status_seed_list": final_status["seeds"],
        "unexpected_class_stems": unexpected_class_stems,
        "missing_class_stems": missing_class_stems,
        "val_images_seen": manifest["val_images_seen"],
        "test_images_seen": manifest["test_images_seen"],
        "test_accessed": manifest["test_accessed"],
    }


def outlier_rows(
    edit_rows: list[dict],
    changed_region_results: dict[tuple[int, str], dict],
    weak_threshold: float,
    strong_threshold: float,
    noise_component_threshold: int,
) -> list[dict]:
    rows = []
    for row in edit_rows:
        reasons = []
        if row["changed_pixel_ratio"] <= weak_threshold:
            reasons.append(("WEAK_CHANGE_CANDIDATE", "changed_pixel_ratio_at_or_below_weak_threshold"))
        if row["changed_pixel_ratio"] >= strong_threshold or row["mask_area_ratio"] > 0.20:
            reasons.append(("OVER_STRONG_CHANGE_CANDIDATE", "changed_pixel_ratio_or_mask_area_over_threshold"))
        region = changed_region_results.get((row["seed"], row["image_stem"]), {})
        if (
            row["connected_components"] >= noise_component_threshold
            and row["largest_component_ratio_lower_bound"] <= 0.20
        ):
            reasons.append(("NOISE_PATTERN_CANDIDATE", "fragmented_refined_mask"))
        for outlier_type, reason in reasons:
            rows.append({
                "image_id": row["image_stem"],
                "edit_id": f"seed{row['seed']:02d}_edit{row['edit_index']:02d}",
                "seed": row["seed"],
                "family": row["family"],
                "outlier_type": outlier_type,
                "reason": reason,
                "changed_pixel_ratio": row["changed_pixel_ratio"],
                "mask_area_ratio": row["mask_area_ratio"],
                "visible_change_ratio": row["visible_change_ratio"],
                "connected_components": row["connected_components"],
                "largest_component_ratio_lower_bound": row["largest_component_ratio_lower_bound"],
                "image_level_changed_pixel_ratio": region.get("changed_pixel_ratio", float("nan")),
                "image_level_largest_component_ratio": region.get("largest_component_ratio", float("nan")),
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--expected-train-image-count", type=int, default=668)
    parser.add_argument("--weak-change-threshold", type=float, default=0.0005)
    parser.add_argument("--strong-change-threshold", type=float, default=0.20)
    parser.add_argument("--noise-component-threshold", type=int, default=20)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    if args.image_size <= 0:
        raise RuntimeError("IMAGE_SIZE_MUST_BE_POSITIVE")
    return args


def main() -> int:
    args = parse_args()
    preview_dir = args.preview_run_dir.resolve()
    split_manifest = args.split_manifest.resolve()
    output_root = args.output_root.resolve()
    if not preview_dir.is_dir():
        raise RuntimeError(f"PREVIEW_RUN_DIR_NOT_FOUND:{preview_dir}")
    if not split_manifest.is_file():
        raise RuntimeError(f"SPLIT_MANIFEST_NOT_FOUND:{split_manifest}")
    generation_manifest_path = preview_dir / "generation_manifest.json"
    final_status_path = preview_dir / "final_status.json"
    edit_metrics_path = preview_dir / "edit_metrics.csv"
    image_metrics_path = preview_dir / "image_metrics.csv"
    required_inputs = [
        generation_manifest_path,
        final_status_path,
        edit_metrics_path,
        image_metrics_path,
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise RuntimeError(f"MISSING_AUDIT_INPUTS:{missing_inputs}")

    manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    final_status = json.loads(final_status_path.read_text(encoding="utf-8"))
    edit_rows = load_edit_metrics(edit_metrics_path, args.image_size)
    image_rows = load_image_metrics(image_metrics_path)
    stem_classes, class_mapping_errors = load_split_classes(
        split_manifest,
        args.expected_train_image_count,
    )
    integrity = check_manifest_integrity(
        manifest,
        final_status,
        preview_dir,
        stem_classes,
        args.expected_train_image_count,
    )
    changed_region_results, changed_region_errors = analyze_changed_regions(
        manifest,
        preview_dir,
        args.image_size,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"phaseC1_distribution_audit_{timestamp}"
    output_dir = output_root / run_name
    if output_dir.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_AUDIT_OUTPUT:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    family_rows = family_distribution(edit_rows)
    visible_rows = visible_change_rows(edit_rows)
    class_rows = class_distribution(manifest, edit_rows, stem_classes)
    outlier_candidates = outlier_rows(
        edit_rows,
        changed_region_results,
        args.weak_change_threshold,
        args.strong_change_threshold,
        args.noise_component_threshold,
    )
    write_csv(output_dir / "morphology_family_distribution.csv", family_rows)
    write_csv(output_dir / "visible_change_statistics.csv", visible_rows)
    write_csv(output_dir / "class_conditioned_preview_distribution.csv", class_rows)
    write_csv(output_dir / "outlier_candidates.csv", outlier_candidates)
    changed_region_rows = [
        {"seed": seed, "image_stem": stem, **stats}
        for (seed, stem), stats in sorted(changed_region_results.items())
    ]
    write_csv(output_dir / "changed_region_component_stats.csv", changed_region_rows)

    family_counts = {row["family"]: row["count"] for row in family_rows}
    class_edit_counts = {
        class_name: next(
            row["edit_count"]
            for row in class_rows
            if row["class_name"] == class_name
        )
        for class_name in CLASS_NAMES
    }
    class_family_coverage = {
        class_name: next(
            row["morphology_family_coverage_count"]
            for row in class_rows
            if row["class_name"] == class_name
        )
        for class_name in CLASS_NAMES
    }
    image_level_changed_ratios = [row["changed_pixel_ratio"] for row in changed_region_rows]
    image_level_largest_ratios = [row["largest_component_ratio"] for row in changed_region_rows]
    integrity_failures = []
    if final_status["status"] != "PASS" or not final_status["formal_gate"]:
        integrity_failures.append("FINAL_STATUS_NOT_FORMAL_PASS")
    if integrity["manifest_image_count"] != args.expected_train_image_count:
        integrity_failures.append("TRAIN_IMAGE_COUNT_MISMATCH")
    if integrity["manifest_per_image_count"] != integrity["expected_preview_count"]:
        integrity_failures.append("PER_IMAGE_COUNT_MISMATCH")
    if integrity["filesystem_preview_count"] != integrity["manifest_per_image_count"]:
        integrity_failures.append("FILESYSTEM_PREVIEW_COUNT_MISMATCH")
    if integrity["missing_source_image_count"]:
        integrity_failures.append("MISSING_SOURCE_IMAGES")
    if integrity["missing_generated_output_count"]:
        integrity_failures.append("MISSING_GENERATED_OUTPUTS")
    if integrity["unique_source_image_count"] != args.expected_train_image_count:
        integrity_failures.append("UNIQUE_SOURCE_IMAGE_COUNT_MISMATCH")
    if integrity["duplicate_source_path_within_seed_count"]:
        integrity_failures.append("DUPLICATE_SOURCE_PATHS_WITHIN_SEED")
    if integrity["duplicate_generated_output_count"]:
        integrity_failures.append("DUPLICATE_GENERATED_OUTPUTS")
    if integrity["duplicate_preview_key_count"]:
        integrity_failures.append("DUPLICATE_PREVIEW_KEYS")
    if integrity["failed_per_image_record_count"]:
        integrity_failures.append("FAILED_PER_IMAGE_RECORDS")
    if integrity["seed_mismatches"]:
        integrity_failures.append("SEED_MISMATCHES")
    if manifest["seeds"] != final_status["seeds"]:
        integrity_failures.append("SEED_LIST_MISMATCH")
    if unexpected_class_stems := integrity["unexpected_class_stems"]:
        integrity_failures.append(f"UNEXPECTED_CLASS_STEMS:{unexpected_class_stems[:10]}")
    if missing_class_stems := integrity["missing_class_stems"]:
        integrity_failures.append(f"MISSING_CLASS_STEMS:{missing_class_stems[:10]}")
    if class_mapping_errors:
        integrity_failures.extend(class_mapping_errors)
    if changed_region_errors:
        integrity_failures.extend(changed_region_errors)

    hard_failures = list(integrity_failures)
    if any(count == 0 for count in family_counts.values()):
        hard_failures.append("MISSING_MORPHOLOGY_FAMILY")
    for class_name in CLASS_NAMES:
        if class_edit_counts[class_name] == 0:
            hard_failures.append(f"NO_EDITS_FOR_CLASS:{class_name}")

    conditional_reasons = []
    for class_name in CLASS_NAMES:
        if class_family_coverage[class_name] < 2:
            conditional_reasons.append(f"CLASS_HAS_FEWER_THAN_TWO_FAMILIES:{class_name}")
    for row in visible_rows:
        if row["family"] != "ALL" and row["visible_change_ratio_below_0_5_fraction"] > 0.05:
            conditional_reasons.append(
                f"VISIBLE_RATIO_BELOW_0_5_FRACTION_OVER_5_PERCENT:{row['family']}:"
                f"{row['visible_change_ratio_below_0_5_fraction']:.6f}"
            )
    if outlier_candidates:
        counts = Counter(row["outlier_type"] for row in outlier_candidates)
        conditional_reasons.append(
            "OUTLIER_CANDIDATES_PRESENT:" + ",".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
    if image_level_changed_ratios and max(image_level_changed_ratios) > 0.30:
        conditional_reasons.append(f"IMAGE_LEVEL_CHANGED_RATIO_OVER_30_PERCENT:{max(image_level_changed_ratios):.6f}")

    if hard_failures:
        gate_decision = "FAIL"
    elif conditional_reasons:
        gate_decision = "CONDITIONAL_PASS"
    else:
        gate_decision = "PASS"

    report = {
        "schema_version": 1,
        "audit_run_id": run_name,
        "timestamp": timestamp,
        "preview_run_dir": str(preview_dir),
        "preview_run_id": manifest["run_id"],
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "generation_manifest_sha256": integrity["manifest_sha256"],
        "final_status_sha256": integrity["final_status_sha256"],
        "mode": "READ_ONLY_DISTRIBUTION_AUDIT",
        "thresholds": {
            "weak_change": args.weak_change_threshold,
            "strong_change": args.strong_change_threshold,
            "noise_components": args.noise_component_threshold,
            "visible_change_low": 0.50,
            "image_level_changed_ratio_strong": 0.30,
        },
        "summary": {
            "train_image_count": integrity["manifest_image_count"],
            "preview_pair_count": integrity["manifest_per_image_count"],
            "edit_count": len(edit_rows),
            "family_counts": family_counts,
            "class_edit_counts": class_edit_counts,
            "class_family_coverage": class_family_coverage,
            "changed_pixel_ratio": next(
                {
                    key: value
                    for key, value in row.items()
                    if key != "family"
                }
                for row in visible_rows
                if row["family"] == "ALL"
            ),
            "image_level_changed_pixel_ratio": statistics(image_level_changed_ratios),
            "image_level_largest_component_ratio": statistics(image_level_largest_ratios),
            "outlier_counts": dict(Counter(row["outlier_type"] for row in outlier_candidates)),
        },
        "manifest_integrity": integrity,
        "integrity_failures": integrity_failures,
        "hard_failures": hard_failures,
        "conditional_reasons": conditional_reasons,
        "gate_decision": gate_decision,
    }
    integrity_report = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "integrity": integrity,
        "failures": integrity_failures,
    }
    write_json(output_dir / "phaseC1_distribution_audit_report.json", report)
    write_json(output_dir / "manifest_integrity_report.json", integrity_report)

    lines = [
        "# Phase C-1 Distribution Audit Summary",
        "",
        f"- Audit run: `{run_name}`",
        f"- Preview run: `{manifest['run_id']}`",
        f"- Split manifest SHA256: `{report['split_manifest_sha256']}`",
        f"- Preview pairs: **{integrity['manifest_per_image_count']}**",
        f"- Edits: **{len(edit_rows)}**",
        f"- Family counts: `{family_counts}`",
        f"- Class edit counts: `{class_edit_counts}`",
        f"- Class family coverage: `{class_family_coverage}`",
        f"- Outlier candidate counts: `{report['summary']['outlier_counts']}`",
        "",
        "## Gate Decision",
        "",
        f"**{gate_decision}**",
        "",
    ]
    if hard_failures:
        lines.append("### Hard failures")
        lines.extend(f"- {failure}" for failure in hard_failures)
        lines.append("")
    if conditional_reasons:
        lines.append("### Conditional reasons")
        lines.extend(f"- {reason}" for reason in conditional_reasons)
        lines.append("")
    lines.extend([
        "This audit is read-only. It does not authorize proxy training when the decision is not PASS.",
    ])
    (output_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": gate_decision,
        "audit_output_dir": str(output_dir),
        "edit_count": len(edit_rows),
        "outlier_counts": report["summary"]["outlier_counts"],
        "hard_failures": hard_failures,
        "conditional_reasons": conditional_reasons,
    }, ensure_ascii=False))
    return 0 if gate_decision != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
