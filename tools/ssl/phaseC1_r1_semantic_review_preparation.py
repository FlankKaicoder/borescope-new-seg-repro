#!/usr/bin/env python3
"""Prepare read-only semantic-review artifacts for Phase C1-R1 V2 previews.

The R1 formal run stores only the source/final preview pair, not per-edit
intermediate images or masks. This tool never regenerates an edit. It uses the
saved source/final panels and marks the selected edit bbox; the displayed mask
is explicitly a combined final-preview difference and is not edit-isolated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FAMILY_SPECS = {
    "A_small_low_contrast": {"output": "A", "count": 50},
    "B_diffuse_texture": {"output": "B", "count": 50},
    "C_crack_like_structural_evolution": {"output": "C", "count": 100},
    "D_boundary_evolution": {"output": "D", "count": 100},
}
SAMPLE_FIELDS = [
    "review_group",
    "selection_order",
    "edit_id",
    "source_id",
    "class",
    "class_provenance",
    "family",
    "changed_ratio",
    "component_count",
    "seed",
    "edit_index",
    "bbox_xyxy",
    "preview_path",
    "human_label",
    "comment",
]
ANNOTATION_FIELDS = [
    "edit_id",
    "class",
    "family",
    "changed_ratio",
    "component_count",
    "human_label",
    "comment",
]
CLASS_UNAVAILABLE = "CLASS_PROVENANCE_UNAVAILABLE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def canonical_edit_id(row: dict[str, str]) -> str:
    return f"seed{int(row['seed']):02d}_{row['image_stem']}_edit{int(row['edit_index']):02d}"


def parse_bbox(row: dict[str, object]) -> list[int]:
    value = json.loads(str(row["bbox_xyxy"]))
    if len(value) != 4:
        raise RuntimeError(f"INVALID_BBOX:{row['edit_id']}")
    return [int(item) for item in value]


def crop_bounds(box: list[int], image_size: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width, height = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
    side = min(image_size, max(192, int(max(width, height) * 2.0)))
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, min(image_size - side, center_x - side // 2))
    top = max(0, min(image_size - side, center_y - side // 2))
    return left, top, left + side, top + side


def mark_bbox(image: Image.Image, box: list[int], crop: tuple[int, int, int, int]) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    left, top, _, _ = crop
    x0, y0, x1, y1 = box
    draw.rectangle((x0 - left, y0 - top, x1 - left, y1 - top), outline=(225, 38, 38), width=3)
    return marked


def source_context(image: Image.Image, box: list[int]) -> Image.Image:
    marked = image.copy()
    ImageDraw.Draw(marked).rectangle(tuple(box), outline=(225, 38, 38), width=4)
    return marked


def render_tile(
    row: dict[str, object],
    preview_dir: Path,
    image_size: int,
    panel_size: int,
    metadata_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    preview_path = preview_dir / str(row["preview_path"])
    with Image.open(preview_path) as opened:
        pair = opened.convert("RGB")
    expected_shape = (image_size * 2 + 8, image_size)
    if pair.size != expected_shape:
        raise RuntimeError(f"UNEXPECTED_PREVIEW_SHAPE:{preview_path}:{pair.size}")
    source = pair.crop((0, 0, image_size, image_size))
    edited = pair.crop((image_size + 8, 0, image_size * 2 + 8, image_size))
    difference = np.abs(
        np.asarray(ImageOps.grayscale(edited), dtype=np.int16)
        - np.asarray(ImageOps.grayscale(source), dtype=np.int16)
    ) >= 2
    mask = Image.fromarray((difference.astype(np.uint8) * 255), mode="L").convert("RGB")
    box = parse_bbox(row)
    crop = crop_bounds(box, image_size)
    panels = [
        source_context(source, box),
        mark_bbox(source.crop(crop), box, crop),
        mark_bbox(edited.crop(crop), box, crop),
        mark_bbox(mask.crop(crop), box, crop),
    ]
    labels = ["source image", "original preview", "edited preview", "change mask"]
    header_height, label_height = 66, 16
    tile = Image.new("RGB", (panel_size * 4, header_height + panel_size + label_height), (246, 246, 246))
    draw = ImageDraw.Draw(tile)
    lines = [
        f"{row['edit_id']}  source:{row['source_id']}",
        f"class: {row['class']}",
        f"family: {row['family']}",
        f"changed ratio: {float(str(row['changed_ratio'])):.6f}  components: {row['component_count']}",
    ]
    for line_number, line in enumerate(lines):
        draw.text((4, 2 + line_number * 16), line, fill=(20, 20, 20), font=metadata_font)
    for panel_index, (panel, label) in enumerate(zip(panels, labels)):
        rendered = ImageOps.pad(panel, (panel_size, panel_size), method=Image.Resampling.LANCZOS, color=(232, 232, 232))
        x = panel_index * panel_size
        tile.paste(rendered, (x, header_height))
        draw.text((x + 3, header_height + panel_size + 1), label, fill=(50, 50, 50), font=label_font)
    return tile


def make_contact_sheet(
    rows: list[dict[str, object]],
    preview_dir: Path,
    output_path: Path,
    image_size: int,
    panel_size: int,
    columns: int,
) -> dict[str, object]:
    metadata_font, label_font = load_font(10), load_font(9)
    tile_height = 66 + panel_size + 16
    row_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_size * 4, row_count * tile_height), (246, 246, 246))
    for index, row in enumerate(rows):
        tile = render_tile(row, preview_dir, image_size, panel_size, metadata_font, label_font)
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path),
        "sample_count": len(rows),
        "columns": columns,
        "rows": row_count,
        "sha256": sha256_file(output_path),
    }


def load_accepted_seed_rows(metrics_path: Path, seed: int) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if int(row["seed"]) == seed and row["acceptance_status"] == "ACCEPTED"]
    if not accepted:
        raise RuntimeError("NO_ACCEPTED_SEED_ROWS")
    return accepted


def source_unique_sample(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row["image_stem"], []).append(row)
    if len(by_source) < count:
        raise RuntimeError(f"INSUFFICIENT_UNIQUE_SOURCES:{len(by_source)}<{count}")
    rng = random.Random(seed)
    sources = sorted(by_source)
    selected_sources = rng.sample(sources, count)
    selected = []
    for source in selected_sources:
        candidates = sorted(by_source[source], key=lambda item: int(item["edit_index"]))
        selected.append(candidates[rng.randrange(len(candidates))])
    return selected


def prepare_rows(rows: list[dict[str, str]], group: str) -> list[dict[str, object]]:
    prepared = []
    for order, row in enumerate(rows, start=1):
        prepared.append({
            "review_group": group,
            "selection_order": order,
            "edit_id": canonical_edit_id(row),
            "source_id": row["image_stem"],
            "class": CLASS_UNAVAILABLE,
            "class_provenance": "NOT_STORED_IN_R1_FORMAL_ARTIFACTS",
            "family": row["family"],
            "changed_ratio": float(row["single_edit_total_changed_pixels"]) / (640 * 640),
            "component_count": int(row["connected_components"]),
            "seed": int(row["seed"]),
            "edit_index": int(row["edit_index"]),
            "bbox_xyxy": row["bbox_xyxy"],
            "preview_path": f"images/seed{int(row['seed']):02d}_{row['image_stem']}.jpg",
            "human_label": "",
            "comment": "",
        })
    return prepared


def main() -> int:
    args = parse_args()
    run_dir, output_dir = args.repair_run_dir.resolve(), args.output_dir.resolve()
    manifest_path = run_dir / "generation_manifest.json"
    status_path = run_dir / "final_status.json"
    metrics_path = run_dir / "edit_metrics.csv"
    required = [manifest_path, status_path, metrics_path, run_dir / "images"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"MISSING_R1_FORMAL_INPUTS:{missing}")
    if output_dir.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_OUTPUT:{output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "formal" or not manifest.get("formal_gate"):
        raise RuntimeError("FORMAL_MANIFEST_GATE_FAILED")
    if status.get("status") != "PASS" or not status.get("formal_gate"):
        raise RuntimeError("FORMAL_STATUS_GATE_FAILED")
    if manifest.get("generator_version") != "phaseC1-r1-local-change-v2":
        raise RuntimeError("UNEXPECTED_GENERATOR_VERSION")
    if int(manifest.get("seed", -1)) != args.selection_seed or args.selection_seed not in manifest.get("seeds", []):
        raise RuntimeError("SELECTION_SEED_NOT_IN_FORMAL_RUN")
    seed_rows = load_accepted_seed_rows(metrics_path, args.selection_seed)
    selections: dict[str, list[dict[str, object]]] = {}
    for family, spec in FAMILY_SPECS.items():
        pool = [row for row in seed_rows if row["family"] == family]
        sampled = source_unique_sample(pool, spec["count"], args.selection_seed)
        selections[spec["output"]] = prepare_rows(sampled, f"{spec['output']}_semantic")
    if any(row["human_label"] or row["comment"] for rows in selections.values() for row in rows):
        raise RuntimeError("AUTOMATIC_HUMAN_LABEL_WRITE_DETECTED")
    output_dir.mkdir(parents=True, exist_ok=False)
    contact_dir = output_dir / "contact_sheet"
    contact_dir.mkdir()
    sheets = {}
    for output_key, rows in selections.items():
        write_csv(output_dir / f"{output_key}_semantic_review.csv", rows, SAMPLE_FIELDS)
        sheets[output_key] = make_contact_sheet(
            rows, run_dir, contact_dir / f"{output_key}_semantic_review.png",
            args.image_size, args.panel_size, args.columns,
        )
    annotation_rows = []
    for key in ("C", "D", "A", "B"):
        annotation_rows.extend({field: row[field] for field in ANNOTATION_FIELDS} for row in selections[key])
    write_csv(output_dir / "semantic_annotation_template.csv", annotation_rows, ANNOTATION_FIELDS)
    manifest_output = {
        "schema_version": 1,
        "status": "PREPARED_AWAITING_HUMAN_ANNOTATION",
        "mode": "READ_ONLY_R1_SEMANTIC_REVIEW_PREPARATION",
        "repair_run_dir": str(run_dir),
        "repair_run_id": manifest["run_id"],
        "selection_seed": args.selection_seed,
        "input_sha256": {
            "generation_manifest.json": sha256_file(manifest_path),
            "final_status.json": sha256_file(status_path),
            "edit_metrics.csv": sha256_file(metrics_path),
        },
        "selection": {
            key: {
                "family": next(name for name, spec in FAMILY_SPECS.items() if spec["output"] == key),
                "count": len(rows),
                "unique_source_count": len({row["source_id"] for row in rows}),
                "selection_mode": "FIXED_SEED_42_SOURCE_UNIQUE_SAMPLE",
            }
            for key, rows in selections.items()
        },
        "class_coverage": {
            "status": "UNAVAILABLE_FROM_R1_FORMAL_ARTIFACTS",
            "reason": "R1 formal manifest stores source IDs and R1 audit stores only aggregate class statistics; no source_id-to-class mapping is present in the allowed inputs.",
            "requested_seven_class_coverage_verified": False,
        },
        "annotation_template_count": len(annotation_rows),
        "human_labels_populated": 0,
        "contact_sheets": sheets,
        "preview_evidence_scope": "SOURCE_AND_FINAL_PREVIEW_PANELS_ONLY; CHANGE_MASK_IS_COMBINED_FINAL_PREVIEW_DIFFERENCE_WITH_SELECTED_BBOX_CONTEXT; PER_EDIT_INTERMEDIATE_PREVIEW_AND_MASK_NOT_STORED",
        "automatic_semantic_decision": "NOT_MADE",
    }
    (output_dir / "semantic_review_preparation_manifest.json").write_text(
        json.dumps(manifest_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest_output["status"],
        "output_dir": str(output_dir),
        "counts": {key: len(rows) for key, rows in selections.items()},
        "class_coverage": manifest_output["class_coverage"]["status"],
        "human_labels_populated": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
