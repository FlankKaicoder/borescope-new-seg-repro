#!/usr/bin/env python3
"""Read-only quality review for Phase C-1 preview outliers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


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
SHORT_FAMILY = {
    "A_small_low_contrast": "A",
    "B_diffuse_texture": "B",
    "C_elongated_ribbon": "C",
    "D_irregular_boundary": "D",
}


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_split_classes(manifest_path: Path, expected_count: int) -> dict[str, set[str]]:
    stem_classes: dict[str, set[str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() != "train":
                continue
            stem = row["stem"].strip()
            if stem in stem_classes:
                raise RuntimeError(f"DUPLICATE_TRAIN_STEM:{stem}")
            labels = {part.strip() for part in row["labels_present"].split("|") if part.strip()}
            invalid = sorted(labels - set(CLASS_NAMES))
            if invalid:
                raise RuntimeError(f"INVALID_TRAIN_LABELS:{stem}:{invalid}")
            stem_classes[stem] = labels
    if len(stem_classes) != expected_count:
        raise RuntimeError(f"TRAIN_STEM_COUNT:{len(stem_classes)}!={expected_count}")
    return stem_classes


def load_edit_metrics(path: Path, image_size: int) -> dict[tuple[int, str, int], dict]:
    rows: dict[tuple[int, str, int], dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["image_stem"], int(row["edit_index"]))
            if key in rows:
                raise RuntimeError(f"DUPLICATE_EDIT_METRIC_KEY:{key}")
            row["seed"] = key[0]
            row["edit_index"] = key[2]
            row["mask_area_ratio"] = float(row["mask_area_ratio"])
            row["visible_change_ratio"] = float(row["visible_change_ratio"])
            row["connected_components"] = int(row["connected_components"])
            row["single_edit_total_changed_pixels"] = int(row["single_edit_total_changed_pixels"])
            row["changed_pixel_ratio"] = row["single_edit_total_changed_pixels"] / (
                image_size * image_size
            )
            rows[key] = row
    return rows


def load_outliers(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match = re.fullmatch(r"seed(\d+)_edit(\d+)", row["edit_id"])
            if not match:
                raise RuntimeError(f"INVALID_EDIT_ID:{row['edit_id']}")
            row["seed"] = int(match.group(1))
            row["edit_index"] = int(match.group(2))
            row["changed_pixel_ratio"] = float(row["changed_pixel_ratio"])
            row["connected_components"] = int(row["connected_components"])
            rows.append(row)
    return rows


def stratified_selection(
    rows: list[dict],
    target_count: int,
    review_seed: int,
    required_families: list[str] | None = None,
) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        stem = row["image_id"]
        for class_name in row["classes"]:
            groups[(row["family"], class_name)].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["seed"], row["image_id"], row["edit_index"]))

    selected: list[dict] = []
    selected_keys: set[tuple[int, str, int]] = set()
    required = required_families or []
    for family in required:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        rng = random.Random(f"{review_seed}:{family}:required")
        for row in rng.sample(family_rows, min(1, len(family_rows))):
            key = (row["seed"], row["image_id"], row["edit_index"])
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)

    remaining = target_count - len(selected)
    if remaining <= 0:
        return selected[:target_count]

    keys = sorted(groups)
    rng = random.Random(f"{review_seed}:stratified")
    # Deterministic stratified round-robin preserves family/class coverage.
    while remaining and any(groups[key] for key in keys):
        for key in keys:
            if not remaining:
                break
            group = groups[key]
            if not group:
                continue
            row = group.pop(0)
            row_key = (row["seed"], row["image_id"], row["edit_index"])
            if row_key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(row_key)
            remaining -= 1
    if remaining:
        pool = [row for row in rows if (row["seed"], row["image_id"], row["edit_index"]) not in selected_keys]
        rng.shuffle(pool)
        selected.extend(pool[:remaining])
    return selected[:target_count]


def make_contact_sheet(
    rows: list[dict],
    manifest_by_key: dict[tuple[int, str], dict],
    preview_dir: Path,
    output_path: Path,
    image_size: int,
    panel_size: int = 128,
    columns: int = 8,
) -> dict:
    label_height = 36
    tile_width = panel_size * 4
    tile_height = panel_size + label_height
    rows_per_sheet = max(1, (len(rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_width, rows_per_sheet * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    labels = ["source", "original", "edited", "change mask"]
    for index, row in enumerate(rows):
        seed = row["seed"]
        stem = row["image_id"]
        record = manifest_by_key[(seed, stem)]
        source_path = Path(record["source_image"])
        output_path_image = preview_dir / record["generated_output"]
        source = Image.open(source_path).convert("RGB")
        source.thumbnail((panel_size, panel_size), Image.Resampling.BILINEAR)
        pair = cv2.imread(str(output_path_image), cv2.IMREAD_COLOR)
        if pair is None:
            raise RuntimeError(f"PREVIEW_NOT_READABLE:{output_path_image}")
        original = cv2.cvtColor(pair[:, :image_size], cv2.COLOR_BGR2RGB)
        changed = cv2.cvtColor(pair[:, image_size + 8:], cv2.COLOR_BGR2RGB)
        gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        gray_changed = cv2.cvtColor(changed, cv2.COLOR_RGB2GRAY)
        mask_array = (np.abs(
            gray_changed.astype(np.int16) - gray_original.astype(np.int16)
        ) >= 2).astype(np.uint8) * 255
        mask = Image.fromarray(mask_array, mode="L").convert("RGB")
        original_image = Image.fromarray(original)
        changed_image = Image.fromarray(changed)
        panels = [source, original_image, changed_image, mask]
        x0 = (index % columns) * tile_width
        y0 = (index // columns) * tile_height
        for panel_index, panel in enumerate(panels):
            panel = panel.copy()
            panel.thumbnail((panel_size, panel_size), Image.Resampling.BILINEAR)
            x = x0 + panel_index * panel_size + (panel_size - panel.width) // 2
            y = y0 + label_height + (panel_size - panel.height) // 2
            sheet.paste(panel, (x, y))
        for panel_index, label in enumerate(labels):
            draw.text(
                (x0 + panel_index * panel_size + 3, y0 + 22),
                label,
                fill=(60, 60, 60),
            )
        class_text = "|".join(sorted(row["classes"]))
        line1 = f"{stem} {row['edit_id']}"
        line2 = f"{SHORT_FAMILY[row['family']]} {class_text} cr={row['changed_pixel_ratio']:.4f}"
        draw.text((x0 + 3, y0 + 2), line1, fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 12), line2, fill=(0, 0, 0))
    sheet.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path),
        "sample_count": len(rows),
        "columns": columns,
        "rows": rows_per_sheet,
        "sha256": sha256_file(output_path),
    }


def component_analysis(edit_rows: list[dict]) -> list[dict]:
    output = []
    for family in FAMILIES:
        rows = [row for row in edit_rows if row["family"] == family]
        components = [row["connected_components"] for row in rows]
        largest = [
            row["mask_area_ratio"] / max(row["connected_components"], 1)
            for row in rows
        ]
        mean_components = float(np.mean(components))
        output.append({
            "family": family,
            "edit_count": len(rows),
            "mean_component_count": mean_components,
            "median_component_count": float(np.median(components)),
            "p95_component_count": float(np.quantile(components, 0.95)),
            "mean_largest_component_ratio": float(np.mean(largest)),
            "fragmentation_score": mean_components - 1.0,
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-run-dir", type=Path, required=True)
    parser.add_argument("--audit-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--expected-train-image-count", type=int, default=668)
    parser.add_argument("--review-seed", type=int, default=42)
    parser.add_argument("--noise-sample-count", type=int, default=100)
    parser.add_argument("--family-sample-count", type=int, default=50)
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    if args.image_size <= 0 or args.panel_size <= 0:
        raise RuntimeError("SIZES_MUST_BE_POSITIVE")
    return args


def main() -> int:
    args = parse_args()
    preview_dir = args.preview_run_dir.resolve()
    audit_dir = args.audit_run_dir.resolve()
    split_manifest = args.split_manifest.resolve()
    output_root = args.output_root.resolve()
    required = [
        preview_dir / "generation_manifest.json",
        preview_dir / "final_status.json",
        preview_dir / "edit_metrics.csv",
        audit_dir / "outlier_candidates.csv",
        audit_dir / "phaseC1_distribution_audit_report.json",
        split_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"MISSING_REVIEW_INPUTS:{missing}")

    manifest = json.loads((preview_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    final_status = json.loads((preview_dir / "final_status.json").read_text(encoding="utf-8"))
    audit_report = json.loads(
        (audit_dir / "phaseC1_distribution_audit_report.json").read_text(encoding="utf-8")
    )
    if audit_report.get("preview_run_id") != manifest.get("run_id"):
        raise RuntimeError("PREVIEW_RUN_ID_MISMATCH")
    if final_status.get("status") != "PASS" or not final_status.get("formal_gate"):
        raise RuntimeError("PREVIEW_RUN_NOT_FORMAL_PASS")
    stem_classes = load_split_classes(split_manifest, args.expected_train_image_count)
    edit_rows_by_key = load_edit_metrics(
        preview_dir / "edit_metrics.csv",
        args.image_size,
    )
    manifest_by_key: dict[tuple[int, str], dict] = {}
    for record in manifest["per_image"]:
        key = (record["seed"], Path(record["source_image"]).stem)
        if key in manifest_by_key:
            raise RuntimeError(f"DUPLICATE_PREVIEW_KEY:{key}")
        manifest_by_key[key] = record

    outlier_rows = load_outliers(audit_dir / "outlier_candidates.csv")
    for row in outlier_rows:
        classes = stem_classes.get(row["image_id"])
        if classes is None:
            raise RuntimeError(f"UNKNOWN_SOURCE_STEM:{row['image_id']}")
        row["classes"] = classes
        key = (row["seed"], row["image_id"], row["edit_index"])
        if key not in edit_rows_by_key:
            raise RuntimeError(f"MISSING_EDIT_METRIC:{key}")
        row["edit_metric"] = edit_rows_by_key[key]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"phaseC1_quality_review_{timestamp}"
    output_dir = output_root / run_name
    if output_dir.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_REVIEW_OUTPUT:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    weak_rows = [row for row in outlier_rows if row["outlier_type"] == "WEAK_CHANGE_CANDIDATE"]
    noise_rows = [row for row in outlier_rows if row["outlier_type"] == "NOISE_PATTERN_CANDIDATE"]
    family_c_rows = [row for row in outlier_rows if row["family"] == "C_elongated_ribbon"]
    family_d_rows = [row for row in outlier_rows if row["family"] == "D_irregular_boundary"]
    weak_selection = weak_rows
    noise_selection = stratified_selection(
        noise_rows,
        args.noise_sample_count,
        args.review_seed,
        required_families=FAMILIES,
    )
    family_c_selection = stratified_selection(
        family_c_rows,
        args.family_sample_count,
        args.review_seed,
    )
    family_d_selection = stratified_selection(
        family_d_rows,
        args.family_sample_count,
        args.review_seed,
    )

    sheet_summaries = {
        "weak_change": make_contact_sheet(
            weak_selection,
            manifest_by_key,
            preview_dir,
            output_dir / "weak_change_contact_sheet.png",
            args.image_size,
            args.panel_size,
        ),
        "noise_pattern": make_contact_sheet(
            noise_selection,
            manifest_by_key,
            preview_dir,
            output_dir / "noise_pattern_contact_sheet.png",
            args.image_size,
            args.panel_size,
        ),
        "family_C": make_contact_sheet(
            family_c_selection,
            manifest_by_key,
            preview_dir,
            output_dir / "family_C_contact_sheet.png",
            args.image_size,
            args.panel_size,
        ),
        "family_D": make_contact_sheet(
            family_d_selection,
            manifest_by_key,
            preview_dir,
            output_dir / "family_D_contact_sheet.png",
            args.image_size,
            args.panel_size,
        ),
    }

    selection_summary = {
        "review_seed": args.review_seed,
        "weak_change": {
            "selection_mode": "ALL",
            "sample_count": len(weak_selection),
            "family_counts": dict(Counter(row["family"] for row in weak_selection)),
            "class_counts": {
                class_name: sum(class_name in row["classes"] for row in weak_selection)
                for class_name in CLASS_NAMES
            },
        },
        "noise_pattern": {
            "selection_mode": "DETERMINISTIC_STRATIFIED",
            "sample_count": len(noise_selection),
            "family_counts": dict(Counter(row["family"] for row in noise_selection)),
            "class_counts": {
                class_name: sum(class_name in row["classes"] for row in noise_selection)
                for class_name in CLASS_NAMES
            },
        },
        "family_C": {
            "selection_mode": "DETERMINISTIC_STRATIFIED",
            "sample_count": len(family_c_selection),
            "class_counts": {
                class_name: sum(class_name in row["classes"] for row in family_c_selection)
                for class_name in CLASS_NAMES
            },
        },
        "family_D": {
            "selection_mode": "DETERMINISTIC_STRATIFIED",
            "sample_count": len(family_d_selection),
            "class_counts": {
                class_name: sum(class_name in row["classes"] for row in family_d_selection)
                for class_name in CLASS_NAMES
            },
        },
    }
    write_json(output_dir / "contact_sheet_selection.json", selection_summary)

    component_rows = component_analysis(list(edit_rows_by_key.values()))
    write_csv(output_dir / "family_component_analysis.csv", component_rows)
    manual_review_rows = []
    for row in outlier_rows:
        manual_review_rows.append({
            "edit_id": row["edit_id"],
            "source_id": row["image_id"],
            "family": row["family"],
            "class": "|".join(sorted(row["classes"])),
            "changed_ratio": row["changed_pixel_ratio"],
            "component_count": row["connected_components"],
            "candidate_type": row["outlier_type"],
            "review_label": "",
        })
    write_csv(output_dir / "outlier_manual_review.csv", manual_review_rows)

    review_report = {
        "schema_version": 1,
        "review_run_id": run_name,
        "timestamp": timestamp,
        "preview_run_dir": str(preview_dir),
        "preview_run_id": manifest["run_id"],
        "audit_run_dir": str(audit_dir),
        "audit_run_id": audit_report.get("audit_run_id"),
        "mode": "READ_ONLY_QUALITY_REVIEW",
        "review_seed": args.review_seed,
        "generation_manifest_sha256": sha256_file(preview_dir / "generation_manifest.json"),
        "final_status_sha256": sha256_file(preview_dir / "final_status.json"),
        "outlier_input_sha256": sha256_file(audit_dir / "outlier_candidates.csv"),
        "gate_c_decision": audit_report.get("gate_decision"),
        "outlier_input_counts": dict(Counter(row["outlier_type"] for row in outlier_rows)),
        "contact_sheets": sheet_summaries,
        "selection": selection_summary,
        "component_analysis": component_rows,
        "manual_review_row_count": len(manual_review_rows),
    }
    write_json(output_dir / "quality_review_report.json", review_report)
    print(json.dumps({
        "status": "COMPLETE",
        "output_dir": str(output_dir),
        "contact_sheets": {key: value["sample_count"] for key, value in sheet_summaries.items()},
        "manual_review_rows": len(manual_review_rows),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
