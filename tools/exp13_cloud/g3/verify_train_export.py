"""Export and verify the frozen 668-image TRAIN set without model access.

The exporter reads the authoritative split manifest and immutable source root,
then copies only TRAIN image/JSON pairs to a new directory.  The verifier can
run on Windows against that directory and never treats a missing JSON as
background.  VAL/TEST rows are read from the manifest for leakage checks only;
their image files are never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_MANIFEST_SHA256 = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
EXPECTED_TRAIN_IMAGES = 668
CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_source(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def load_class_mapping(path: Path | None) -> dict[int, str]:
    if path is None:
        return dict(enumerate(CLASSES))
    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
        names = payload.get("names", payload) if isinstance(payload, dict) else payload
        if isinstance(names, list):
            return dict(enumerate(str(item) for item in names))
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
    except json.JSONDecodeError:
        pass
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue
        if in_names and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip().isdigit():
                names[int(key.strip())] = value.strip().strip("'\"")
    if not names:
        raise ValueError(f"cannot parse class mapping: {path}")
    return names


def validate_annotation(path: Path, width: int, height: int, classes: set[str]) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    issues: list[str] = []
    labels: list[str] = []
    if payload.get("imageWidth") not in (None, width):
        issues.append(f"annotation imageWidth={payload.get('imageWidth')} != {width}")
    if payload.get("imageHeight") not in (None, height):
        issues.append(f"annotation imageHeight={payload.get('imageHeight')} != {height}")
    shapes = payload.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        return labels, issues + ["missing or empty shapes; image is not background"]
    for index, shape in enumerate(shapes):
        label = str(shape.get("label", ""))
        labels.append(label)
        if label not in classes:
            issues.append(f"shape {index}: unknown class {label!r}")
        if shape.get("shape_type") != "polygon":
            issues.append(f"shape {index}: shape_type is not polygon")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 3:
            issues.append(f"shape {index}: fewer than three polygon points")
            continue
        for point_index, point in enumerate(points):
            if not isinstance(point, list) or len(point) < 2:
                issues.append(f"shape {index} point {point_index}: malformed point")
                continue
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                issues.append(f"shape {index} point {point_index}: nonnumeric point")
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                issues.append(f"shape {index} point {point_index}: nonfinite point")
            elif not (0 <= x <= width and 0 <= y <= height):
                issues.append(f"shape {index} point {point_index}: out of bounds")
    return labels, issues


def row_paths(row: dict[str, str], source_root: Path, image_root: Path | None, json_root: Path | None) -> tuple[Path, Path]:
    image_rel = row.get("export_image_path") or row.get("image_path", "")
    json_rel = row.get("export_json_path") or row.get("json_path", "")
    image_path = resolve_source(image_root or source_root, image_rel)
    json_path = resolve_source(json_root or source_root, json_rel)
    return image_path, json_path


def validate_manifest(rows: list[dict[str, str]], expected_train: int) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != expected_train:
        issues.append({"type": "train_count", "detail": f"expected {expected_train}, got {len(train)}"})
    seen_stems: set[str] = set()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        stem = row.get("stem", "")
        group = row.get("group_id", "")
        if row.get("split") == "train":
            if not stem or stem in seen_stems:
                issues.append({"type": "duplicate_or_missing_stem", "detail": stem or "<empty>"})
            seen_stems.add(stem)
            if not group:
                issues.append({"type": "missing_group_id", "detail": stem})
        if group:
            groups[group].add(row.get("split", ""))
    for group, splits in groups.items():
        if len(splits) > 1:
            issues.append({"type": "group_split_leakage", "detail": f"{group}: {sorted(splits)}"})
    return issues


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_hash = sha256_file(args.manifest)
    rows = read_rows(args.manifest)
    issues = validate_manifest(rows, args.expected_count)
    class_mapping = load_class_mapping(args.class_mapping)
    classes = set(class_mapping.values())
    split_stats: Counter[str] = Counter()
    label_stats: Counter[str] = Counter()
    checked_rows = 0
    for row in rows:
        if row.get("split") != "train":
            continue
        checked_rows += 1
        image_path, json_path = row_paths(row, args.source_root, args.image_root, args.json_root)
        if not image_path.is_file():
            issues.append({"type": "missing_image", "detail": str(image_path)})
            continue
        if not json_path.is_file():
            issues.append({"type": "missing_professional_json", "detail": str(json_path)})
            continue
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            labels, annotation_issues = validate_annotation(json_path, width, height, classes)
        except Exception as exc:  # decode/JSON errors are recorded, never guessed through
            issues.append({"type": "decode_or_annotation_error", "detail": f"{image_path}: {exc}"})
            continue
        for issue in annotation_issues:
            issues.append({"type": "annotation", "detail": f"{json_path}: {issue}"})
        split_stats[row.get("split", "")] += 1
        label_stats.update(labels)
        expected_image_sha = row.get("sha256", "")
        expected_json_sha = row.get("json_sha256", "")
        if expected_image_sha and sha256_file(image_path) != expected_image_sha:
            issues.append({"type": "image_hash_mismatch", "detail": str(image_path)})
        if expected_json_sha and sha256_file(json_path) != expected_json_sha:
            issues.append({"type": "json_hash_mismatch", "detail": str(json_path)})
        for key, actual in (("image_width", width), ("image_height", height)):
            if row.get(key) and int(row[key]) != actual:
                issues.append({"type": "dimension_mismatch", "detail": f"{row.get('stem')}: {key}"})
        if row.get("instance_count") and int(row["instance_count"]) != len(labels):
            issues.append({"type": "instance_count_mismatch", "detail": row.get("stem", "")})
    report = {
        "status": "PASS" if manifest_hash == args.expected_manifest_sha256 and not issues and checked_rows == args.expected_count else "FAIL",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "expected_train_images": args.expected_count,
        "checked_train_images": checked_rows,
        "class_mapping": class_mapping,
        "split_stats": dict(split_stats),
        "label_instance_counts": dict(sorted(label_stats.items())),
        "issues": issues,
        "test_image_access": 0,
        "raw_write": 0,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    return report


def export_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing export directory: {args.output}")
    rows = read_rows(args.manifest)
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError(f"manifest SHA256 mismatch: {manifest_hash}")
    manifest_issues = validate_manifest(rows, args.expected_count)
    if manifest_issues:
        raise RuntimeError("manifest validation failed: " + json.dumps(manifest_issues, ensure_ascii=False))
    class_mapping = load_class_mapping(args.class_mapping)
    args.output.mkdir(parents=True)
    images_dir = args.output / "images" / "train"
    json_dir = args.output / "annotations" / "train"
    images_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    exported: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") != "train":
            continue
        image_src, json_src = row_paths(row, args.source_root, None, None)
        if not image_src.is_file() or not json_src.is_file():
            raise FileNotFoundError(f"missing TRAIN pair: {image_src} / {json_src}")
        image_dst = images_dir / Path(row["image_path"]).name
        json_dst = json_dir / (Path(row["json_path"]).stem + ".json")
        if image_dst.exists() or json_dst.exists():
            raise FileExistsError(f"export collision: {image_dst} / {json_dst}")
        shutil.copy2(image_src, image_dst)
        shutil.copy2(json_src, json_dst)
        exported_row = dict(row)
        exported_row["export_image_path"] = str(image_dst.relative_to(args.output)).replace("\\", "/")
        exported_row["export_json_path"] = str(json_dst.relative_to(args.output)).replace("\\", "/")
        exported.append(exported_row)
    with (args.output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(exported[0]) if exported else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)
    (args.output / "class_mapping.json").write_text(json.dumps({"names": class_mapping}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    export_record = {
        "status": "EXPORTED_TRAIN_ONLY",
        "source_root": str(args.source_root),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": manifest_hash,
        "train_images": len(exported),
        "class_mapping": class_mapping,
        "val_test_image_access": 0,
        "raw_write": 0,
        "next_step": "Run verify_train_export.py --mode verify on Windows export directory.",
    }
    write_json(args.output / "export_record.json", export_record)
    return export_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("verify", "export"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--json-root", type=Path)
    parser.add_argument("--class-mapping", type=Path)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TRAIN_IMAGES)
    args = parser.parse_args()
    result = export_train(args) if args.mode == "export" else verify(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
