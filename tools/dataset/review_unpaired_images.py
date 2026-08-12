#!/usr/bin/env python3
"""Build Exp00.4 evidence for images without JSON; never assigns final semantics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_dataset import IMAGE_SUFFIXES, hamming, image_fingerprints, sha256_file


def labels_for_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return sorted({str(shape.get("label", "")) for shape in payload.get("shapes", []) if isinstance(shape, dict)})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["filename"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_near_groups(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["image_path"]] = row["group_id"]
    return result


def program_hint(phash_distance: int, dhash_distance: int, correlation: float) -> tuple[str, str]:
    if phash_distance <= 6 and dhash_distance <= 8 and correlation >= 0.98:
        return "possible_missing_json", "Meets the frozen high-confidence near-duplicate thresholds against a labeled image."
    if phash_distance <= 12 and dhash_distance <= 14 and correlation >= 0.95:
        return "possible_missing_annotation", "Moderately similar to a labeled image but does not meet the high-confidence group threshold."
    return "unknown", "No sufficiently close labeled neighbor; background versus missing annotation cannot be inferred without human review."


def multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font: ImageFont.ImageFont) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill="black", font=font)
        y += 15


def render_summary(data_root: Path, rows: list[dict[str, Any]], output: Path) -> None:
    columns, cell_w, cell_h, image_h, margin = 4, 280, 245, 170, 12
    page_rows = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (margin * 2 + columns * cell_w, 42 + page_rows * cell_h + margin), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((margin, margin), f"Exp00.4 images without JSON | {len(rows)} | hints are NOT final labels", fill="black", font=font)
    for index, row in enumerate(rows):
        col, grid_row = index % columns, index // columns
        x, y = margin + col * cell_w, 42 + grid_row * cell_h
        with Image.open(data_root / row["path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_w - 8, image_h - 8), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (cell_w - image.width) // 2, y + (image_h - image.height) // 2))
        draw.rectangle((x, y, x + cell_w - 1, y + image_h - 1), outline="#777777")
        multiline(draw, (x + 3, y + image_h + 3), [
            f"#{index + 1} {row['filename']} | {row['width']}x{row['height']}",
            f"near_group: {row['near_duplicate_group']}",
            f"nearest: {row['nearest_labeled_image']}",
            f"labels: {row['nearest_image_labels']}",
            f"pH/dH/corr: {row['nearest_phash_distance']}/{row['nearest_dhash_distance']}/{float(row['nearest_correlation']):.4f}",
            f"hint: {row['program_hint']}",
        ], font)
    canvas.save(output, quality=92)


def render_pairs(data_root: Path, rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row["program_hint"] != "unknown"]
    font, cell_w, image_h, text_h, margin = ImageFont.load_default(), 360, 250, 72, 12
    if not selected:
        canvas = Image.new("RGB", (900, 120), "white")
        ImageDraw.Draw(canvas).text((margin, margin), "No unpaired image met the programmatic moderate/high similarity thresholds.", fill="black", font=font)
        canvas.save(output, quality=92)
        return
    canvas = Image.new("RGB", (margin * 2 + 2 * cell_w, margin + len(selected) * (image_h + text_h + margin)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(selected):
        y = margin + row_index * (image_h + text_h + margin)
        for col, (relative, title) in enumerate(((row["path"], "UNPAIRED"), (row["nearest_labeled_image"], "NEAREST LABELED"))):
            x = margin + col * cell_w
            with Image.open(data_root / relative) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w - 8, image_h - 8), Image.Resampling.LANCZOS)
                canvas.paste(image, (x + (cell_w - image.width) // 2, y + (image_h - image.height) // 2))
            draw.rectangle((x, y, x + cell_w - 1, y + image_h - 1), outline="#777777")
            lines = [f"{title}: {relative}"]
            if col == 1:
                lines.append(f"labels: {row['nearest_image_labels']}")
            lines.extend([f"pH/dH/corr: {row['nearest_phash_distance']}/{row['nearest_dhash_distance']}/{float(row['nearest_correlation']):.4f}", f"hint: {row['program_hint']}"])
            multiline(draw, (x + 3, y + image_h + 3), lines, font)
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--near-groups", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    near_groups = load_near_groups(args.near_groups)
    images = sorted(path for path in args.data_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    unpaired = [path for path in images if not path.with_suffix(".json").exists()]
    labeled = [path for path in images if path.with_suffix(".json").exists()]
    labeled_data = []
    for path in labeled:
        with Image.open(path) as image:
            phash, dhash, feature = image_fingerprints(image)
        labeled_data.append((path, phash, dhash, feature, labels_for_json(path.with_suffix(".json"))))
    feature_matrix = np.stack([item[3] for item in labeled_data])
    rows: list[dict[str, Any]] = []
    for path in unpaired:
        with Image.open(path) as image:
            width, height = image.size
            phash, dhash, feature = image_fingerprints(image)
        correlations = feature_matrix @ feature
        best_index = int(np.argmax(correlations))
        nearest, near_phash, near_dhash, _, labels = labeled_data[best_index]
        phash_distance = hamming(phash, near_phash)
        dhash_distance = hamming(dhash, near_dhash)
        correlation = float(correlations[best_index])
        hint, reason = program_hint(phash_distance, dhash_distance, correlation)
        relative = path.relative_to(args.data_root).as_posix()
        nearest_relative = nearest.relative_to(args.data_root).as_posix()
        rows.append({
            "filename": path.name,
            "path": relative,
            "sha256": sha256_file(path),
            "width": width,
            "height": height,
            "near_duplicate_group": near_groups.get(relative, "NONE"),
            "nearest_labeled_image": nearest_relative,
            "nearest_distance": f"phash={phash_distance};dhash={dhash_distance};gray_correlation={correlation:.8f}",
            "nearest_phash_distance": phash_distance,
            "nearest_dhash_distance": dhash_distance,
            "nearest_correlation": f"{correlation:.8f}",
            "nearest_image_labels": "|".join(labels),
            "program_hint": hint,
            "program_hint_reason": reason,
            "human_decision": "PENDING_HUMAN_REVIEW",
            "human_notes": "",
        })
    write_csv(args.output_dir / "image_without_json.csv", rows)
    render_summary(args.data_root, rows, args.output_dir / "image_without_json_contact_sheet.jpg")
    render_pairs(args.data_root, rows, args.output_dir / "nearest_labeled_pair_comparisons.jpg")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["program_hint"]] = counts.get(row["program_hint"], 0) + 1
    summary = {"image_count": len(rows), "program_hint_counts": counts, "final_semantic_decisions_made": 0, "warning": "Program hints are evidence triage only and must not become labels without human review."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

