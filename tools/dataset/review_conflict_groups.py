#!/usr/bin/env python3
"""Build Exp00.5 visual/structured review packages; never edits annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = ["#ff3030", "#00cc66", "#2080ff", "#ffb000", "#cc33ff", "#00b8c8", "#ff66aa"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def annotation(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    shapes = [shape for shape in payload.get("shapes", []) if isinstance(shape, dict)]
    return payload, shapes, Counter(str(shape.get("label", "")) for shape in shapes)


def draw_overlay(image: Image.Image, shapes: list[dict[str, Any]], color_by_label: dict[str, str]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result, "RGBA")
    font = ImageFont.load_default()
    for shape in shapes:
        label = str(shape.get("label", ""))
        points = [(float(point[0]), float(point[1])) for point in shape.get("points", []) if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) < 3:
            continue
        color = color_by_label[label]
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        draw.polygon(points, fill=(*rgb, 65), outline=(*rgb, 255), width=3)
        draw.text(points[0], label, fill=(*rgb, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 220))
    return result


def normalized_centroids(shapes: list[dict[str, Any]], width: int, height: int) -> list[tuple[float, float]]:
    values = []
    for shape in shapes:
        points = [(float(p[0]), float(p[1])) for p in shape.get("points", []) if isinstance(p, (list, tuple)) and len(p) >= 2]
        if points:
            values.append((sum(p[0] for p in points) / len(points) / width, sum(p[1] for p in points) / len(points) / height))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--pairs-csv", required=True, type=Path)
    parser.add_argument("--consistency-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sheet_dir = args.output_dir / "contact_sheets"
    sheet_dir.mkdir()
    consistency = read_csv(args.consistency_csv)
    conflict_ids = sorted({row["group_id"] for row in consistency if row["group_has_label_set_mismatch"].lower() == "true"})
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.groups_csv):
        if row["group_id"] in conflict_ids:
            groups[row["group_id"]].append(row)
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.pairs_csv):
        if row["group_id"] in conflict_ids:
            pairs[row["group_id"]].append(row)
    member_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    all_labels: set[str] = set()
    cached: dict[str, tuple[dict[str, Any], list[dict[str, Any]], Counter[str], tuple[int, int], list[tuple[float, float]]]] = {}
    for group_id in conflict_ids:
        for member in groups[group_id]:
            relative = member["image_path"]
            payload, shapes, counts = annotation((args.data_root / relative).with_suffix(".json"))
            with Image.open(args.data_root / relative) as image:
                dimensions = image.size
            cached[relative] = (payload, shapes, counts, dimensions, normalized_centroids(shapes, *dimensions))
            all_labels.update(counts)
    color_by_label = {label: COLORS[index % len(COLORS)] for index, label in enumerate(sorted(all_labels))}
    for group_id in conflict_ids:
        members = sorted(groups[group_id], key=lambda row: row["image_path"])
        edge_rows = pairs[group_id]
        phash_values = [int(row["phash_distance"]) for row in edge_rows]
        dhash_values = [int(row["dhash_distance"]) for row in edge_rows]
        correlations = [float(row["gray_correlation"]) for row in edge_rows]
        signatures = []
        class_sets = []
        count_values = []
        centroid_means = []
        for member in members:
            relative = member["image_path"]
            payload, shapes, counts, (width, height), centroids = cached[relative]
            signature = "|".join(f"{key}:{counts[key]}" for key in sorted(counts))
            class_set = "|".join(sorted(counts))
            signatures.append(signature)
            class_sets.append(class_set)
            count_values.append(len(shapes))
            if centroids:
                centroid_means.append((float(np.mean([x for x, _ in centroids])), float(np.mean([y for _, y in centroids]))))
            member_rows.append({
                "group_id": group_id,
                "group_size": len(members),
                "filename": Path(relative).name,
                "path": relative,
                "labels": class_set,
                "annotation_signature": signature,
                "polygon_count": len(shapes),
                "image_width": width,
                "image_height": height,
                "json_version": payload.get("version"),
                "group_min_phash_distance": min(phash_values) if phash_values else "",
                "group_max_phash_distance": max(phash_values) if phash_values else "",
                "group_min_dhash_distance": min(dhash_values) if dhash_values else "",
                "group_max_dhash_distance": max(dhash_values) if dhash_values else "",
                "group_min_gray_correlation": min(correlations) if correlations else "",
                "group_max_gray_correlation": max(correlations) if correlations else "",
                "human_decision": "PENDING_HUMAN_REVIEW",
                "human_notes": "",
            })
        centroid_spread = 0.0
        for i in range(len(centroid_means)):
            for j in range(i + 1, len(centroid_means)):
                centroid_spread = max(centroid_spread, math.dist(centroid_means[i], centroid_means[j]))
        subset_relation = any(set(left.split("|")) < set(right.split("|")) for left in class_sets for right in class_sets if left and right)
        review_rows.append({
            "group_id": group_id,
            "member_count": len(members),
            "members": "|".join(row["image_path"] for row in members),
            "class_sets": ";".join(sorted(set(class_sets))),
            "annotation_signatures": ";".join(sorted(set(signatures))),
            "min_phash_distance": min(phash_values) if phash_values else "",
            "max_phash_distance": max(phash_values) if phash_values else "",
            "min_dhash_distance": min(dhash_values) if dhash_values else "",
            "max_dhash_distance": max(dhash_values) if dhash_values else "",
            "min_gray_correlation": min(correlations) if correlations else "",
            "max_gray_correlation": max(correlations) if correlations else "",
            "same_or_near_same_visual_content_program_hint": True,
            "same_defect_location_program_hint": "LIKELY" if centroid_means and centroid_spread <= 0.12 else "UNCERTAIN",
            "normalized_mean_centroid_spread": f"{centroid_spread:.6f}",
            "annotation_classes_inconsistent": len(set(class_sets)) > 1,
            "one_image_seems_incompletely_annotated_program_hint": "POSSIBLE" if subset_relation else "NOT_INFERRED",
            "different_true_defect_despite_visual_similarity": "PENDING_HUMAN_REVIEW",
            "uncertain": True,
            "human_group_decision": "PENDING_HUMAN_REVIEW",
            "human_notes": "",
        })

        columns, tile_w, image_h, text_h, margin = min(4, len(members)), 330, 220, 80, 12
        rows_count = math.ceil(len(members) / columns)
        canvas = Image.new("RGB", (margin * 2 + columns * tile_w, 40 + rows_count * (image_h * 2 + text_h + margin)), "white")
        draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
        draw.text((margin, margin), f"{group_id} | {len(members)} members | GT labels conflict | HUMAN REVIEW REQUIRED", fill="black", font=font)
        for index, member in enumerate(members):
            relative = member["image_path"]
            payload, shapes, counts, _, _ = cached[relative]
            with Image.open(args.data_root / relative) as image:
                original = image.convert("RGB")
                overlay = draw_overlay(original, shapes, color_by_label)
            col, row = index % columns, index // columns
            x, y = margin + col * tile_w, 40 + row * (image_h * 2 + text_h + margin)
            for layer_index, layer in enumerate((original, overlay)):
                layer.thumbnail((tile_w - 8, image_h - 8), Image.Resampling.LANCZOS)
                layer_y = y + layer_index * image_h
                canvas.paste(layer, (x + (tile_w - layer.width) // 2, layer_y + (image_h - layer.height) // 2))
                draw.rectangle((x, layer_y, x + tile_w - 1, layer_y + image_h - 1), outline="#777777")
            text_y = y + image_h * 2 + 3
            draw.text((x + 3, text_y), f"{relative} | version={payload.get('version')}", fill="black", font=font)
            draw.text((x + 3, text_y + 15), f"labels/counts: {dict(counts)}", fill="black", font=font)
            draw.text((x + 3, text_y + 30), f"polygons={len(shapes)}", fill="black", font=font)
        canvas.save(sheet_dir / f"{group_id}.jpg", quality=92)

    def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    save_csv(args.output_dir / "conflict_group_members.csv", member_rows)
    save_csv(args.output_dir / "conflict_group_review.csv", review_rows)
    save_csv(args.output_dir / "conflict_group_similarity_edges.csv", [row for group in conflict_ids for row in pairs[group]])
    summary = {"conflict_group_count": len(conflict_ids), "member_row_count": len(member_rows), "contact_sheet_count": len(conflict_ids), "final_label_changes": 0, "human_decisions_pending": len(conflict_ids)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

