#!/usr/bin/env python3
"""Prepare read-only, deterministic Phase C1 manual-review artifacts.

This tool reads the completed quality-review CSV and existing preview pairs. It
does not generate previews, modify their inputs, or infer human labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


WEAK = "WEAK_CHANGE_CANDIDATE"
NOISE = "NOISE_PATTERN_CANDIDATE"
C_FAMILY = "C_elongated_ribbon"
D_FAMILY = "D_irregular_boundary"
SAMPLE_FIELDS = [
    "review_group",
    "selection_order",
    "edit_id",
    "source_id",
    "class",
    "family",
    "changed_ratio",
    "component_count",
    "candidate_type",
    "human_label",
    "comment",
]
ANNOTATION_FIELDS = [
    "edit_id",
    "source_id",
    "class",
    "family",
    "changed_ratio",
    "component_count",
    "candidate_type",
    "human_label",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-review-dir", type=Path, required=True)
    parser.add_argument("--preview-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--noise-sample-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--panel-size", type=int, default=192)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    if args.noise_sample_count <= 0 or args.image_size <= 0:
        raise RuntimeError("COUNTS_AND_IMAGE_SIZE_MUST_BE_POSITIVE")
    if args.panel_size < 96 or args.columns <= 0:
        raise RuntimeError("CONTACT_SHEET_LAYOUT_INVALID")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_edit_key(row: dict[str, str]) -> tuple[int, str, int]:
    match = re.fullmatch(r"seed(\d+)_edit(\d+)", row["edit_id"])
    if not match:
        raise RuntimeError(f"INVALID_EDIT_ID:{row['edit_id']}")
    return int(match.group(1)), row["source_id"], int(match.group(2))


def sort_key(row: dict[str, str]) -> tuple[int, int, str, int]:
    seed, source_id, edit_index = parse_edit_key(row)
    source_sort = int(source_id) if source_id.isdigit() else 10**12
    return seed, source_sort, source_id, edit_index


def deterministic_sample(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    if len(rows) < count:
        raise RuntimeError(f"INSUFFICIENT_SAMPLE_POOL:{len(rows)}<{count}")
    return random.Random(seed).sample(sorted(rows, key=sort_key), count)


def prepare_rows(rows: list[dict[str, str]], group: str) -> list[dict[str, object]]:
    prepared = []
    for index, row in enumerate(rows, start=1):
        prepared.append({
            "review_group": group,
            "selection_order": index,
            "edit_id": row["edit_id"],
            "source_id": row["source_id"],
            "class": row["class"],
            "family": row["family"],
            "changed_ratio": row["changed_ratio"],
            "component_count": row["component_count"],
            "candidate_type": row["candidate_type"],
            "human_label": "",
            "comment": "",
        })
    return prepared


def annotation_template(groups: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    deduplicated: dict[tuple[str, str], dict[str, object]] = {}
    for rows in groups:
        for row in rows:
            key = (str(row["source_id"]), str(row["edit_id"]))
            existing = deduplicated.get(key)
            if existing is None:
                deduplicated[key] = {
                    field: row[field] for field in ANNOTATION_FIELDS
                }
                continue
            existing_types = set(str(existing["candidate_type"]).split("|"))
            existing_types.add(str(row["candidate_type"]))
            existing["candidate_type"] = "|".join(sorted(existing_types))
    return sorted(
        deduplicated.values(),
        key=lambda row: sort_key({
            "edit_id": str(row["edit_id"]),
            "source_id": str(row["source_id"]),
        }),
    )


def load_edit_metrics(path: Path) -> dict[tuple[int, str, int], dict[str, str]]:
    metrics = {}
    for row in load_csv(path):
        key = (int(row["seed"]), row["image_stem"], int(row["edit_index"]))
        if key in metrics:
            raise RuntimeError(f"DUPLICATE_EDIT_METRIC:{key}")
        metrics[key] = row
    return metrics


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def crop_bounds(box: list[int], image_size: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width = max(x1 - x0 + 1, 1)
    height = max(y1 - y0 + 1, 1)
    side = min(image_size, max(192, int(max(width, height) * 2.0)))
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = max(0, min(image_size - side, center_x - side // 2))
    top = max(0, min(image_size - side, center_y - side // 2))
    return left, top, left + side, top + side


def mark_candidate(image: Image.Image, box: list[int], crop: tuple[int, int, int, int]) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    left, top, _, _ = crop
    x0, y0, x1, y1 = box
    draw.rectangle((x0 - left, y0 - top, x1 - left, y1 - top), outline=(220, 35, 35), width=3)
    return marked


def render_tile(
    row: dict[str, object],
    metric: dict[str, str],
    preview_dir: Path,
    image_size: int,
    panel_size: int,
    metadata_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    seed, source_id, _ = parse_edit_key({
        "edit_id": str(row["edit_id"]),
        "source_id": str(row["source_id"]),
    })
    preview_path = preview_dir / "images" / f"seed{seed:02d}_{source_id}.jpg"
    if not preview_path.is_file():
        raise RuntimeError(f"PREVIEW_NOT_READABLE:{preview_path}")
    with Image.open(preview_path) as preview:
        pair = preview.convert("RGB")
    expected_width = image_size * 2 + 8
    if pair.size != (expected_width, image_size):
        raise RuntimeError(f"UNEXPECTED_PREVIEW_SHAPE:{preview_path}:{pair.size}")
    source = pair.crop((0, 0, image_size, image_size))
    edited = pair.crop((image_size + 8, 0, expected_width, image_size))
    source_gray = np.asarray(ImageOps.grayscale(source), dtype=np.int16)
    edited_gray = np.asarray(ImageOps.grayscale(edited), dtype=np.int16)
    mask = (np.abs(edited_gray.astype(np.int16) - source_gray.astype(np.int16)) >= 2).astype(np.uint8) * 255
    box = json.loads(metric["bbox_xyxy"])
    if len(box) != 4:
        raise RuntimeError(f"INVALID_BBOX:{row['edit_id']}")
    crop = crop_bounds([int(value) for value in box], image_size)
    source_image = mark_candidate(source.crop(crop), box, crop)
    edited_image = mark_candidate(edited.crop(crop), box, crop)
    mask_image = mark_candidate(Image.fromarray(mask).convert("RGB").crop(crop), box, crop)
    panels = [source_image, edited_image, mask_image]
    labels = ["source context", "edited context", "combined change mask"]
    header_height = 92
    label_height = 20
    tile = Image.new("RGB", (panel_size * 3, header_height + panel_size + label_height), (246, 246, 246))
    draw = ImageDraw.Draw(tile)
    lines = [
        f"edit_id: {row['edit_id']}    source_id: {row['source_id']}",
        f"class: {row['class']}",
        f"family: {row['family']}",
        f"changed ratio: {float(str(row['changed_ratio'])):.6f}    component count: {row['component_count']}",
    ]
    for line_index, line in enumerate(lines):
        draw.text((6, 4 + line_index * 21), line, fill=(20, 20, 20), font=metadata_font)
    for panel_index, (panel, label) in enumerate(zip(panels, labels)):
        rendered = ImageOps.pad(
            panel,
            (panel_size, panel_size),
            method=Image.Resampling.LANCZOS,
            color=(232, 232, 232),
        )
        x = panel_index * panel_size
        tile.paste(rendered, (x, header_height))
        draw.text((x + 4, header_height + panel_size + 2), label, fill=(50, 50, 50), font=label_font)
    return tile


def make_contact_sheet(
    rows: list[dict[str, object]],
    metrics: dict[tuple[int, str, int], dict[str, str]],
    preview_dir: Path,
    output_path: Path,
    image_size: int,
    panel_size: int,
    columns: int,
) -> dict[str, object]:
    metadata_font = load_font(15)
    label_font = load_font(12)
    tile_height = 92 + panel_size + 20
    rows_per_sheet = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_size * 3, rows_per_sheet * tile_height), (246, 246, 246))
    for index, row in enumerate(rows):
        key = parse_edit_key({
            "edit_id": str(row["edit_id"]),
            "source_id": str(row["source_id"]),
        })
        metric = metrics.get(key)
        if metric is None:
            raise RuntimeError(f"MISSING_EDIT_METRIC:{key}")
        tile = render_tile(row, metric, preview_dir, image_size, panel_size, metadata_font, label_font)
        x = (index % columns) * tile.width
        y = (index // columns) * tile.height
        sheet.paste(tile, (x, y))
    sheet.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path),
        "sample_count": len(rows),
        "columns": columns,
        "rows": rows_per_sheet,
        "sha256": sha256_file(output_path),
    }


def main() -> int:
    args = parse_args()
    quality_dir = args.quality_review_dir.resolve()
    preview_dir = args.preview_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    manual_csv = quality_dir / "outlier_manual_review.csv"
    metrics_csv = preview_dir / "edit_metrics.csv"
    manifest_json = preview_dir / "generation_manifest.json"
    required = [manual_csv, metrics_csv, manifest_json, preview_dir / "images"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"MISSING_MANUAL_REVIEW_INPUTS:{missing}")
    if output_dir.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_MANUAL_REVIEW_OUTPUT:{output_dir}")

    input_rows = load_csv(manual_csv)
    required_columns = {
        "edit_id", "source_id", "family", "class", "changed_ratio", "component_count", "candidate_type", "review_label"
    }
    if not input_rows or set(input_rows[0]) != required_columns:
        raise RuntimeError("UNEXPECTED_QUALITY_REVIEW_SCHEMA")
    if any(row["review_label"].strip() for row in input_rows):
        raise RuntimeError("QUALITY_REVIEW_LABELS_MUST_REMAIN_BLANK")

    weak_rows = [row for row in input_rows if row["candidate_type"] == WEAK]
    c_noise_pool = [row for row in input_rows if row["candidate_type"] == NOISE and row["family"] == C_FAMILY]
    d_noise_pool = [row for row in input_rows if row["candidate_type"] == NOISE and row["family"] == D_FAMILY]
    if len(weak_rows) != 96:
        raise RuntimeError(f"WEAK_CHANGE_COUNT:{len(weak_rows)}!=96")
    c_noise_rows = deterministic_sample(c_noise_pool, args.noise_sample_count, args.selection_seed)
    d_noise_rows = deterministic_sample(d_noise_pool, args.noise_sample_count, args.selection_seed)

    weak_output = prepare_rows(sorted(weak_rows, key=sort_key), "weak_change")
    c_output = prepare_rows(c_noise_rows, "C_noise")
    d_output = prepare_rows(d_noise_rows, "D_noise")
    annotations = annotation_template([weak_output, c_output, d_output])
    if any(row["human_label"] or row["comment"] for row in [*weak_output, *c_output, *d_output, *annotations]):
        raise RuntimeError("AUTOMATIC_HUMAN_LABEL_WRITE_DETECTED")

    metrics = load_edit_metrics(metrics_csv)
    for row in [*weak_output, *c_output, *d_output]:
        key = parse_edit_key({
            "edit_id": str(row["edit_id"]),
            "source_id": str(row["source_id"]),
        })
        if key not in metrics:
            raise RuntimeError(f"MISSING_SELECTED_EDIT_METRIC:{key}")

    output_dir.mkdir(parents=True, exist_ok=False)
    contact_sheet_dir = output_dir / "contact_sheet"
    contact_sheet_dir.mkdir()
    write_csv(output_dir / "weak_change_manual_review.csv", weak_output, SAMPLE_FIELDS)
    write_csv(output_dir / "C_noise_manual_review.csv", c_output, SAMPLE_FIELDS)
    write_csv(output_dir / "D_noise_manual_review.csv", d_output, SAMPLE_FIELDS)
    write_csv(output_dir / "manual_annotation_template.csv", annotations, ANNOTATION_FIELDS)
    sheets = {
        "weak_change": make_contact_sheet(weak_output, metrics, preview_dir, contact_sheet_dir / "weak_change_review.png", args.image_size, args.panel_size, args.columns),
        "C_noise": make_contact_sheet(c_output, metrics, preview_dir, contact_sheet_dir / "C_noise_review.png", args.image_size, args.panel_size, args.columns),
        "D_noise": make_contact_sheet(d_output, metrics, preview_dir, contact_sheet_dir / "D_noise_review.png", args.image_size, args.panel_size, args.columns),
    }
    manifest = {
        "schema_version": 1,
        "mode": "READ_ONLY_MANUAL_REVIEW_PREPARATION",
        "selection_seed": args.selection_seed,
        "quality_review_dir": str(quality_dir),
        "preview_run_dir": str(preview_dir),
        "input_sha256": {
            "outlier_manual_review.csv": sha256_file(manual_csv),
            "edit_metrics.csv": sha256_file(metrics_csv),
            "generation_manifest.json": sha256_file(manifest_json),
        },
        "selection": {
            "weak_change": {"selection_mode": "ALL", "count": len(weak_output)},
            "C_noise": {"selection_mode": "RANDOM_SAMPLE_FIXED_SEED", "pool_count": len(c_noise_pool), "count": len(c_output)},
            "D_noise": {"selection_mode": "RANDOM_SAMPLE_FIXED_SEED", "pool_count": len(d_noise_pool), "count": len(d_output)},
        },
        "annotation_template_unique_count": len(annotations),
        "human_labels_populated": 0,
        "contact_sheets": sheets,
        "sheet_mask_scope": "COMBINED_FINAL_PREVIEW_DIFFERENCE_WITH_CANDIDATE_BBOX_CONTEXT",
        "selected_candidate_type_counts": dict(Counter(str(row["candidate_type"]) for row in [*weak_output, *c_output, *d_output])),
    }
    (output_dir / "manual_review_preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPLETE",
        "output_dir": str(output_dir),
        "weak_change_count": len(weak_output),
        "C_noise_count": len(c_output),
        "D_noise_count": len(d_output),
        "annotation_template_unique_count": len(annotations),
        "human_labels_populated": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
