#!/usr/bin/env python3
"""Build and verify the immutable Exp01 YOLO segmentation dataset v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
COLORS = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#00a7e1", "#5856d6", "#ff2d55"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_points(raw: Any) -> tuple[list[tuple[float, float]], int]:
    points: list[tuple[float, float]] = []
    removed = 0
    if not isinstance(raw, list):
        return [], 0
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return [], removed
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return [], removed
        if not math.isfinite(x) or not math.isfinite(y):
            return [], removed
        current = (x, y)
        if points and current == points[-1]:
            removed += 1
            continue
        points.append(current)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
        removed += 1
    return points, removed


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2.0 if len(points) >= 3 else 0.0


def load_frozen_inputs(data_root: Path, raw_manifest: Path, near_groups: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = read_csv(raw_manifest)
    image_rows = {row["relative_path"]: row for row in manifest_rows if row["file_type"] == "image"}
    json_rows = {row["relative_path"]: row for row in manifest_rows if row["file_type"] == "json"}
    if len(image_rows) != 993 or len(json_rows) != 969:
        raise RuntimeError(f"Frozen manifest cardinality mismatch: images={len(image_rows)}, json={len(json_rows)}")

    near_by_image = {row["image_path"]: row["group_id"] for row in read_csv(near_groups)}
    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for image_rel, image_meta in sorted(image_rows.items(), key=lambda item: int(Path(item[0]).stem)):
        image_path = data_root / image_rel
        if not image_path.is_file() or sha256_file(image_path) != image_meta["sha256"]:
            raise RuntimeError(f"Frozen source image changed or missing: {image_rel}")
        json_rel = str(Path(image_rel).with_suffix(".json")).replace("\\", "/")
        if json_rel not in json_rows:
            exclusions.append({
                "image_path": image_rel, "json_path": "", "stem": Path(image_rel).stem,
                "split": "excluded", "source_status": "excluded_unpaired",
                "exclusion_reason": "No paired authoritative JSON; excluded from supervised dataset.",
                "sha256": image_meta["sha256"],
            })
            continue
        json_path = data_root / json_rel
        if not json_path.is_file() or sha256_file(json_path) != json_rows[json_rel]["sha256"]:
            raise RuntimeError(f"Frozen source JSON changed or missing: {json_rel}")
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        with Image.open(image_path) as image:
            width, height = image.size
        if payload.get("imageWidth") != width or payload.get("imageHeight") != height:
            raise RuntimeError(f"Image/JSON dimensions mismatch: {image_rel}")
        labels = [str(shape.get("label")) for shape in payload.get("shapes", [])]
        unknown = sorted(set(labels) - set(CLASSES))
        if unknown:
            raise RuntimeError(f"Unexpected classes in {json_rel}: {unknown}")
        samples.append({
            "image_path": image_rel, "json_path": json_rel, "stem": Path(image_rel).stem,
            "sha256": image_meta["sha256"], "json_sha256": json_rows[json_rel]["sha256"],
            "labels_present": "|".join(sorted(set(labels), key=CLASSES.index)),
            "instance_count": len(labels), "annotation_version": str(payload.get("version", "")),
            "image_suffix": Path(image_rel).suffix, "image_width": width, "image_height": height,
            "near_duplicate_group_id": near_by_image.get(image_rel, ""),
            "source_status": "matched_authoritative", "payload": payload,
        })
    if len(samples) != 969 or len(exclusions) != 24:
        raise RuntimeError(f"Policy pool mismatch: supervised={len(samples)}, excluded={len(exclusions)}")
    return samples, exclusions


def convert(args: argparse.Namespace) -> None:
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    labels_dir = output / "labels_all"
    labels_dir.mkdir(parents=True)
    samples, exclusions = load_frozen_inputs(args.data_root, args.raw_manifest, args.near_groups)
    issues: list[dict[str, Any]] = []
    converted_instances = 0
    removed_consecutive_points = 0
    conversion_rows: list[dict[str, Any]] = []
    for sample in samples:
        lines: list[str] = []
        for index, shape in enumerate(sample["payload"].get("shapes", [])):
            label = str(shape.get("label"))
            points, removed = clean_points(shape.get("points"))
            removed_consecutive_points += removed
            reason = ""
            if shape.get("shape_type") != "polygon": reason = "non_polygon"
            elif len(points) < 3: reason = "fewer_than_3_valid_points"
            elif polygon_area(points) <= 0: reason = "degenerate_polygon"
            elif any(x < 0 or x > sample["image_width"] or y < 0 or y > sample["image_height"] for x, y in points): reason = "coordinate_out_of_bounds"
            if reason:
                issues.append({"json_path": sample["json_path"], "shape_index": index, "label": label, "reason": reason})
                continue
            normalized = [(x / sample["image_width"], y / sample["image_height"]) for x, y in points]
            if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in normalized):
                issues.append({"json_path": sample["json_path"], "shape_index": index, "label": label, "reason": "normalized_out_of_range"})
                continue
            coords = " ".join(f"{value:.10f}" for point in normalized for value in point)
            lines.append(f"{CLASSES.index(label)} {coords}")
            converted_instances += 1
        label_path = labels_dir / f"{sample['stem']}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        row = {key: value for key, value in sample.items() if key != "payload"}
        row["label_path"] = f"labels_all/{sample['stem']}.txt"
        row["label_sha256"] = sha256_file(label_path)
        conversion_rows.append(row)
    write_csv(output / "conversion_manifest.csv", conversion_rows)
    write_csv(output / "conversion_issues.csv", issues, ["json_path", "shape_index", "label", "reason"])
    write_csv(output / "exclusion_manifest.csv", exclusions)
    mapping = "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    (output / "class_mapping.yaml").write_text(mapping, encoding="utf-8")
    mapping_hash = sha256_file(output / "class_mapping.yaml")
    (output / "class_mapping_sha256.txt").write_text(mapping_hash + "  class_mapping.yaml\n", encoding="utf-8")
    summary = {
        "status": "PASS" if not issues and converted_instances == 1847 else "FAIL",
        "supervised_samples": len(samples), "excluded_unpaired": len(exclusions),
        "source_instances": sum(row["instance_count"] for row in samples),
        "converted_instances": converted_instances, "conversion_errors": len(issues),
        "removed_consecutive_or_closing_duplicate_points": removed_consecutive_points,
        "class_mapping_sha256": mapping_hash,
    }
    dump_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS": raise SystemExit(2)


def build_groups(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row["near_duplicate_group_id"] or f"singleton_{row['stem']}"
        grouped[group_id].append(row)
    groups = []
    for group_id, members in grouped.items():
        label_images = Counter()
        label_instances = Counter()
        for member in members:
            for label in member["labels_present"].split("|"):
                if label: label_images[label] += 1
            payload = json.loads(member["instance_labels_json"])
            label_instances.update(payload)
        groups.append({"group_id": group_id, "members": members, "size": len(members), "label_images": label_images, "label_instances": label_instances})
    return groups


def score_assignment(assignments: dict[str, str], groups: list[dict[str, Any]], totals: dict[str, Any], ratio_weight: float) -> float:
    counts = {split: 0 for split in SPLITS}
    image_cls = {split: Counter() for split in SPLITS}
    inst_cls = {split: Counter() for split in SPLITS}
    for group in groups:
        split = assignments[group["group_id"]]
        counts[split] += group["size"]
        image_cls[split].update(group["label_images"])
        inst_cls[split].update(group["label_instances"])
    score = 0.0
    total_images = totals["images"]
    for split in SPLITS:
        score += ratio_weight * ((counts[split] - TARGET[split] * total_images) / total_images) ** 2
        for label in CLASSES:
            expected_images = TARGET[split] * totals["image_cls"][label]
            expected_instances = TARGET[split] * totals["inst_cls"][label]
            score += ((image_cls[split][label] - expected_images) / max(3.0, expected_images)) ** 2
            score += 0.35 * ((inst_cls[split][label] - expected_instances) / max(3.0, expected_instances)) ** 2
            if image_cls[split][label] == 0: score += 1000.0
    return score


def split(args: argparse.Namespace) -> None:
    output = args.output
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = read_csv(args.conversion / "conversion_manifest.csv")
    for row in rows:
        payload = json.loads((args.data_root / row["json_path"]).read_text(encoding="utf-8-sig"))
        row["instance_labels_json"] = json.dumps(Counter(str(s["label"]) for s in payload["shapes"]), sort_keys=True)
    groups = build_groups(rows)
    totals = {
        "images": len(rows),
        "image_cls": Counter(label for row in rows for label in row["labels_present"].split("|") if label),
        "inst_cls": Counter(),
    }
    for row in rows: totals["inst_cls"].update(json.loads(row["instance_labels_json"]))
    rng = random.Random(args.seed)
    best: tuple[float, dict[str, str]] | None = None
    ordered_base = sorted(groups, key=lambda g: (-g["size"], -sum(1 / totals["image_cls"][c] for c in g["label_images"]), g["group_id"]))
    for restart in range(args.restarts):
        ordered = ordered_base[:]
        if restart:
            rng.shuffle(ordered)
            ordered.sort(key=lambda g: -g["size"])
        assignment: dict[str, str] = {}
        for group in ordered:
            candidates = []
            for candidate in SPLITS:
                trial = dict(assignment); trial[group["group_id"]] = candidate
                remaining = [g for g in groups if g["group_id"] not in trial]
                for rest in remaining: trial[rest["group_id"]] = "train"
                candidates.append((score_assignment(trial, groups, totals, args.ratio_weight), rng.random(), candidate))
            assignment[group["group_id"]] = min(candidates)[2]
        current = score_assignment(assignment, groups, totals, args.ratio_weight)
        for _ in range(args.iterations):
            group = rng.choice(groups); old = assignment[group["group_id"]]; new = rng.choice([s for s in SPLITS if s != old])
            assignment[group["group_id"]] = new
            candidate = score_assignment(assignment, groups, totals, args.ratio_weight)
            if candidate <= current or rng.random() < math.exp(min(0.0, (current - candidate) / 0.02)):
                current = candidate
            else: assignment[group["group_id"]] = old
        if best is None or current < best[0]: best = (current, dict(assignment))
    assert best is not None
    assignment = best[1]
    final_rows = []
    for row in rows:
        group_id = row["near_duplicate_group_id"] or f"singleton_{row['stem']}"
        final = {key: value for key, value in row.items() if key != "instance_labels_json"}
        final["split"] = assignment[group_id]
        final["group_id"] = group_id
        final_rows.append(final)
    fields = ["image_path", "json_path", "stem", "split", "group_id", "sha256", "labels_present", "instance_count", "near_duplicate_group_id", "source_status", "annotation_version", "image_suffix", "image_width", "image_height", "json_sha256", "label_path", "label_sha256"]
    final_rows.sort(key=lambda row: (SPLITS.index(row["split"]), int(row["stem"])))
    write_csv(output / "split_manifest.csv", final_rows, fields)
    digest = sha256_file(output / "split_manifest.csv")
    (output / "split_manifest_sha256.txt").write_text(digest + "  split_manifest.csv\n", encoding="utf-8")
    summary: dict[str, Any] = {"status": "PASS", "seed": args.seed, "objective_score": best[0], "split_manifest_sha256": digest, "splits": {}}
    for split_name in SPLITS:
        selected = [row for row in final_rows if row["split"] == split_name]
        class_images = Counter(label for row in selected for label in row["labels_present"].split("|") if label)
        class_instances = Counter()
        for row in selected:
            payload = json.loads((args.data_root / row["json_path"]).read_text(encoding="utf-8-sig"))
            class_instances.update(str(shape["label"]) for shape in payload["shapes"])
        summary["splits"][split_name] = {"images": len(selected), "instances": sum(class_instances.values()), "class_images": dict(class_images), "class_instances": dict(class_instances)}
    near_split = defaultdict(set)
    for row in final_rows:
        if row["near_duplicate_group_id"]: near_split[row["near_duplicate_group_id"]].add(row["split"])
    summary["near_duplicate_group_count"] = len(near_split)
    summary["cross_split_near_duplicate_leakage"] = sum(len(value) > 1 for value in near_split.values())
    if summary["cross_split_near_duplicate_leakage"] or any(summary["splits"][s]["class_images"].get(c, 0) == 0 for s in SPLITS for c in CLASSES):
        summary["status"] = "FAIL"
    dump_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS": raise SystemExit(2)


def freeze(args: argparse.Namespace) -> None:
    if args.dataset_root.exists(): raise FileExistsError(f"Refusing to overwrite {args.dataset_root}")
    rows = read_csv(args.split_dir / "split_manifest.csv")
    for split_name in SPLITS:
        (args.dataset_root / "images" / split_name).mkdir(parents=True)
        (args.dataset_root / "labels" / split_name).mkdir(parents=True)
    for row in rows:
        image_link = args.dataset_root / "images" / row["split"] / Path(row["image_path"]).name
        os.symlink(args.data_root / row["image_path"], image_link)
        shutil.copy2(args.conversion / row["label_path"], args.dataset_root / "labels" / row["split"] / f"{row['stem']}.txt")
    shutil.copy2(args.conversion / "class_mapping.yaml", args.dataset_root / "class_mapping.yaml")
    shutil.copy2(args.conversion / "class_mapping_sha256.txt", args.dataset_root / "class_mapping_sha256.txt")
    shutil.copy2(args.split_dir / "split_manifest.csv", args.dataset_root / "split_manifest.csv")
    shutil.copy2(args.split_dir / "split_manifest_sha256.txt", args.dataset_root / "split_manifest_sha256.txt")
    data_yaml = f"path: {args.dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    (args.dataset_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    raw_hash = args.raw_manifest_hash.read_text(encoding="utf-8").strip().split()[0]
    provenance = {
        "raw_dataset_path": str(args.data_root), "raw_manifest_sha256": raw_hash,
        "raw_image_count": 993, "raw_json_count": 969, "matched_annotated_count": 969,
        "excluded_unpaired_count": 24, "excluded_unpaired_policy": "excluded_unpaired; not background; no empty JSON; no split membership",
        "actual_classes": CLASSES, "class_mapping_sha256": sha256_file(args.dataset_root / "class_mapping.yaml"),
        "split_manifest_sha256": sha256_file(args.dataset_root / "split_manifest.csv"),
        "conversion_git_commit": args.git_commit, "split_seed": 42,
        "grouping_strategy": "Exp00 pHash/dHash/correlation connected-component near-duplicate group; singleton otherwise",
        "near_duplicate_group_count": 88, "real_acquisition_ids_available": False,
        "leakage_control_limitation": "Real video/engine/workpiece/acquisition-batch IDs unavailable; visual near-duplicate grouping is the leakage-control proxy.",
        "cross_label_near_duplicate_groups": "22 groups; original professional labels retained unchanged",
        "image_materialization": "absolute symbolic links to immutable raw images",
        "dataset_freeze_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(args.dataset_root / "provenance.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


def parse_label(path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) < 7 or len(values) % 2 == 0: raise ValueError(f"Invalid label shape: {path}:{line_number}")
        class_id = int(values[0]); coords = [float(value) for value in values[1:]]
        if class_id < 0 or class_id >= len(CLASSES): raise ValueError(f"Invalid class id: {path}:{line_number}")
        if any(value < 0 or value > 1 for value in coords): raise ValueError(f"Coordinate outside [0,1]: {path}:{line_number}")
        points = list(zip(coords[0::2], coords[1::2]))
        if len(points) < 3 or polygon_area(points) <= 0: raise ValueError(f"Degenerate YOLO polygon: {path}:{line_number}")
        result.append((class_id, points))
    if not result: raise ValueError(f"Empty label: {path}")
    return result


def choose_overlay_rows(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed); chosen: list[dict[str, str]] = []; selected_stems: set[str] = set()
    per_split = {"train": 20, "val": 15, "test": 15}
    split_counts = Counter()

    def priority(row: dict[str, str]) -> float:
        labels = row["labels_present"].split("|")
        return 4 * ("Tip curl" in labels) + 3 * ("Tears" in labels) + 2 * (len(labels) > 1) + int(row["instance_count"]) + 2 * bool(row["near_duplicate_group_id"]) + rng.random()

    # The verification set must cover every frozen class at least once.
    for label in CLASSES:
        pool = [row for row in rows if label in row["labels_present"].split("|") and row["stem"] not in selected_stems and split_counts[row["split"]] < per_split[row["split"]]]
        if not pool: raise RuntimeError(f"Cannot cover class in overlay verification: {label}")
        row = max(pool, key=priority)
        chosen.append(row); selected_stems.add(row["stem"]); split_counts[row["split"]] += 1

    for split_name, n in per_split.items():
        pool = [row for row in rows if row["split"] == split_name and row["stem"] not in selected_stems]
        ranked = []
        for row in pool:
            ranked.append((priority(row), row))
        needed = n - split_counts[split_name]
        added = [row for _, row in sorted(ranked, key=lambda item: item[0], reverse=True)[:needed]]
        chosen.extend(added); selected_stems.update(row["stem"] for row in added)
    return chosen[:count]


def verify(args: argparse.Namespace) -> None:
    rows = read_csv(args.dataset_root / "split_manifest.csv")
    issues: list[dict[str, str]] = []
    seen_images: dict[str, str] = {}; group_splits: dict[str, set[str]] = defaultdict(set)
    split_stats: dict[str, Any] = {}
    for split_name in SPLITS:
        image_dir = args.dataset_root / "images" / split_name; label_dir = args.dataset_root / "labels" / split_name
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        labels = sorted(label_dir.glob("*.txt"))
        if len(images) != len(labels): issues.append({"type": "pair_count", "detail": f"{split_name}: images={len(images)}, labels={len(labels)}"})
        instances = 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists(): issues.append({"type": "missing_label", "detail": str(image)}); continue
            try: instances += len(parse_label(label))
            except Exception as exc: issues.append({"type": "invalid_label", "detail": str(exc)})
        split_stats[split_name] = {"images": len(images), "labels": len(labels), "instances": instances}
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        previous = seen_images.setdefault(row["sha256"], row["split"])
        if previous != row["split"]: issues.append({"type": "sha256_leakage", "detail": row["sha256"]})
    leakage_groups = [group for group, values in group_splits.items() if len(values) > 1]
    for group in leakage_groups: issues.append({"type": "group_leakage", "detail": group})

    overlay_dir = args.output / "gt_overlays"; overlay_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_overlay_rows(rows, 50, 42)
    overlay_manifest = []
    font = ImageFont.load_default()
    for index, row in enumerate(selected, 1):
        image_path = args.dataset_root / "images" / row["split"] / Path(row["image_path"]).name
        label_path = args.dataset_root / "labels" / row["split"] / f"{row['stem']}.txt"
        with Image.open(image_path) as source:
            image = source.convert("RGB"); draw = ImageDraw.Draw(image)
            for class_id, normalized in parse_label(label_path):
                points = [(x * image.width, y * image.height) for x, y in normalized]
                draw.polygon(points, outline=COLORS[class_id], width=max(2, image.width // 400))
                draw.text(points[0], CLASSES[class_id], fill=COLORS[class_id], font=font)
            image.thumbnail((960, 720), Image.Resampling.LANCZOS)
            out_name = f"{index:03d}_{row['split']}_{row['stem']}.jpg"; image.save(overlay_dir / out_name, quality=92)
        overlay_manifest.append({"overlay": out_name, "split": row["split"], "stem": row["stem"], "labels_present": row["labels_present"], "instance_count": row["instance_count"], "near_duplicate_group_id": row["near_duplicate_group_id"]})
    write_csv(args.output / "overlay_manifest.csv", overlay_manifest)

    smoke = {"status": "NOT_RUN", "detail": ""}
    try:
        from ultralytics.cfg import DEFAULT_CFG
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset
        loaded = check_det_dataset(str(args.dataset_root / "data.yaml"), autodownload=False)
        loaded_splits = {}
        cache_paths = [args.dataset_root / "labels" / f"{split_name}.cache" for split_name in SPLITS]
        cache_existed = {path: path.exists() for path in cache_paths}
        for split_name in SPLITS:
            dataset = YOLODataset(
                img_path=loaded[split_name], data=loaded, task="segment", imgsz=640,
                augment=False, rect=False, cache=False, batch_size=1, hyp=DEFAULT_CFG,
            )
            sample = dataset[0]
            loaded_splits[split_name] = {
                "dataset_length": len(dataset),
                "sample_image_shape": list(sample["img"].shape),
                "sample_instance_count": int(sample["cls"].shape[0]),
                "sample_has_masks": "masks" in sample,
            }
        for path in cache_paths:
            if not cache_existed[path] and path.exists():
                path.unlink()
        smoke = {"status": "PASS", "detail": {
            "nc": loaded["nc"], "names": loaded["names"], "loaded_splits": loaded_splits,
            "transient_label_caches_removed": True,
        }}
    except Exception as exc:
        smoke = {"status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}
        issues.append({"type": "ultralytics_dataset_load", "detail": smoke["detail"]})
    summary = {
        "status": "PASS" if not issues and len(rows) == 969 and len(selected) >= 50 else "FAIL",
        "manifest_rows": len(rows), "split_stats": split_stats, "invalid_issue_count": len(issues),
        "cross_split_group_leakage": len(leakage_groups), "cross_split_sha256_leakage": sum(i["type"] == "sha256_leakage" for i in issues),
        "overlay_count": len(selected), "overlay_pipeline": "YOLO txt read -> denormalize -> polygon overlay",
        "ultralytics_dataset_load_smoke": smoke,
    }
    selected_classes = sorted({label for row in selected for label in row["labels_present"].split("|")}, key=CLASSES.index)
    summary["overlay_classes_covered"] = selected_classes
    if selected_classes != CLASSES:
        summary["status"] = "FAIL"
        issues.append({"type": "overlay_class_coverage", "detail": "|".join(selected_classes)})
    write_csv(args.output / "verification_issues.csv", issues, ["type", "detail"])
    dump_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS": raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, required=True)
    p = sub.add_parser("convert", parents=[common]); p.add_argument("--raw-manifest", type=Path, required=True); p.add_argument("--near-groups", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=convert)
    p = sub.add_parser("split", parents=[common]); p.add_argument("--conversion", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=42); p.add_argument("--restarts", type=int, default=6); p.add_argument("--iterations", type=int, default=300); p.add_argument("--ratio-weight", type=float, default=800.0); p.set_defaults(func=split)
    p = sub.add_parser("freeze", parents=[common]); p.add_argument("--conversion", type=Path, required=True); p.add_argument("--split-dir", type=Path, required=True); p.add_argument("--dataset-root", type=Path, required=True); p.add_argument("--raw-manifest-hash", type=Path, required=True); p.add_argument("--git-commit", required=True); p.set_defaults(func=freeze)
    p = sub.add_parser("verify"); p.add_argument("--dataset-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=verify)
    return root


if __name__ == "__main__":
    args = parser().parse_args(); args.func(args)
