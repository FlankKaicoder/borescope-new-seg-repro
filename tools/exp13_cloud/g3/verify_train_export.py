"""Immutable frozen-manifest TRAIN export, archive, and Windows verification."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_MANIFEST_SHA256 = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
EXPECTED_TRAIN_IMAGES = 668
EXPECTED_TRAIN_INSTANCES = 1266
CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
FROZEN_COLUMNS = {"image_path", "json_path", "stem", "split", "group_id", "sha256", "json_sha256", "labels_present", "instance_count", "image_width", "image_height"}
EXPORT_COLUMNS = ["source_id", "source_image_path", "source_json_path", "source_image_sha256", "source_json_sha256", "export_image_path", "export_json_path", "export_image_sha256", "export_json_sha256", "split", "group_id", "labels_present", "instance_count", "image_width", "image_height", "annotation_version"]


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
    path = Path(relative)
    return path if path.is_absolute() else root / path


def load_class_mapping(path: Path | None) -> dict[int, str]:
    if path is None:
        return dict(enumerate(CLASSES))
    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
        names = payload.get("names", payload) if isinstance(payload, dict) else payload
        if isinstance(names, list):
            return dict(enumerate(map(str, names)))
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
    except json.JSONDecodeError:
        pass
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        value = line.strip()
        if value == "names:":
            in_names = True
        elif in_names and ":" in value:
            key, label = value.split(":", 1)
            if key.strip().isdigit():
                names[int(key.strip())] = label.strip().strip("'\"")
    if not names:
        raise ValueError(f"cannot parse class mapping: {path}")
    return names


def validate_annotation(path: Path, source_image_path: str, width: int, height: int, classes: set[str]) -> tuple[list[str], list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    issues: list[str] = []
    labels: list[str] = []
    image_path = payload.get("imagePath")
    if not isinstance(image_path, str) or not image_path:
        issues.append("imagePath missing or invalid")
    elif Path(image_path.replace("\\", "/")).name != Path(source_image_path.replace("\\", "/")).name:
        issues.append(f"imagePath does not identify source image {Path(source_image_path).name}")
    if payload.get("imageWidth") != width:
        issues.append(f"imageWidth={payload.get('imageWidth')} != {width}")
    if payload.get("imageHeight") != height:
        issues.append(f"imageHeight={payload.get('imageHeight')} != {height}")
    shapes = payload.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        return labels, issues + ["missing or empty shapes; image is not background"], payload
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            issues.append(f"shape {index}: not an object")
            continue
        label = str(shape.get("label", ""))
        labels.append(label)
        if label not in classes:
            issues.append(f"shape {index}: unknown class {label!r}")
        if shape.get("shape_type") != "polygon":
            issues.append(f"shape {index}: shape_type is not polygon")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 3:
            issues.append(f"shape {index}: fewer than three points")
            continue
        for point in points:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                issues.append(f"shape {index}: malformed point")
                continue
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
                issues.append(f"shape {index}: nonfinite or out-of-bounds point")
    return labels, issues, payload


def validate_frozen_rows(rows: list[dict[str, str]], expected_count: int) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not rows:
        return [{"type": "empty_manifest", "detail": "no rows"}]
    missing = sorted(FROZEN_COLUMNS - set(rows[0]))
    if missing:
        issues.append({"type": "frozen_schema", "detail": f"missing columns {missing}"})
    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != expected_count:
        issues.append({"type": "train_count", "detail": f"expected {expected_count}, got {len(train)}"})
    stems: set[str] = set()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split, stem, group = row.get("split", ""), row.get("stem", ""), row.get("group_id", "")
        if split not in {"train", "val", "test"}:
            issues.append({"type": "split_value", "detail": f"{stem}: {split!r}"})
        if split == "train":
            if not stem or stem in stems:
                issues.append({"type": "duplicate_or_missing_stem", "detail": stem})
            stems.add(stem)
            if not group:
                issues.append({"type": "missing_group_id", "detail": stem})
        if group:
            groups[group].add(split)
    for group, splits in groups.items():
        if len(splits) > 1:
            issues.append({"type": "group_split_leakage", "detail": f"{group}: {sorted(splits)}"})
    return issues


def _safe_relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _files(root: Path, directory: str, suffix: str | None = None) -> set[str]:
    base = root / directory
    if not base.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in base.rglob("*") if path.is_file() and (suffix is None or path.suffix.lower() == suffix)}


def validate_export_data(frozen_manifest: Path, export_manifest: Path, data_root: Path, class_mapping: Path | None, expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256, expected_count: int = EXPECTED_TRAIN_IMAGES, expected_instances: int = EXPECTED_TRAIN_INSTANCES) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    frozen_hash = sha256_file(frozen_manifest) if frozen_manifest.is_file() else "MISSING"
    if frozen_hash != expected_manifest_sha256:
        issues.append({"type": "frozen_manifest_sha256", "detail": f"expected {expected_manifest_sha256}, got {frozen_hash}"})
    frozen_rows = read_rows(frozen_manifest) if frozen_manifest.is_file() else []
    issues.extend(validate_frozen_rows(frozen_rows, expected_count))
    train_rows = [row for row in frozen_rows if row.get("split") == "train"]
    export_rows = read_rows(export_manifest) if export_manifest.is_file() else []
    if not export_rows:
        issues.append({"type": "export_manifest", "detail": "missing or empty export_manifest.csv"})
    elif set(EXPORT_COLUMNS) - set(export_rows[0]):
        issues.append({"type": "export_schema", "detail": f"missing columns {sorted(set(EXPORT_COLUMNS) - set(export_rows[0]))}"})
    if len(export_rows) != expected_count:
        issues.append({"type": "export_count", "detail": f"expected {expected_count}, got {len(export_rows)}"})
    frozen_by_stem = {row.get("stem", ""): row for row in train_rows}
    export_by_stem: dict[str, dict[str, str]] = {}
    class_counts: Counter[str] = Counter()
    checked_images = checked_instances = 0
    for row in export_rows:
        source_id = row.get("source_id", "")
        if source_id in export_by_stem:
            issues.append({"type": "duplicate_export_source", "detail": source_id})
        export_by_stem[source_id] = row
        frozen = frozen_by_stem.get(source_id)
        if frozen is None:
            issues.append({"type": "source_set", "detail": f"not in frozen TRAIN: {source_id}"})
            continue
        identity_pairs = (("source_image_path", "image_path"), ("source_json_path", "json_path"), ("source_image_sha256", "sha256"), ("source_json_sha256", "json_sha256"), ("split", "split"), ("group_id", "group_id"), ("labels_present", "labels_present"), ("instance_count", "instance_count"), ("image_width", "image_width"), ("image_height", "image_height"))
        for export_key, frozen_key in identity_pairs:
            if row.get(export_key, "") != frozen.get(frozen_key, ""):
                issues.append({"type": "source_identity", "detail": f"{source_id}: {export_key}"})
        image_rel, json_rel = row.get("export_image_path", ""), row.get("export_json_path", "")
        if not _safe_relative(image_rel) or not _safe_relative(json_rel):
            issues.append({"type": "unsafe_export_path", "detail": source_id})
            continue
        image_path, json_path = data_root / image_rel, data_root / json_rel
        if not image_path.is_file():
            issues.append({"type": "missing_export_image", "detail": source_id})
            continue
        if not json_path.is_file():
            issues.append({"type": "missing_export_json", "detail": source_id})
            continue
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            labels, annotation_issues, annotation = validate_annotation(json_path, row.get("source_image_path", ""), width, height, set(load_class_mapping(class_mapping).values()))
        except Exception as exc:
            issues.append({"type": "decode_or_annotation_error", "detail": f"{source_id}: {exc}"})
            continue
        for detail in annotation_issues:
            issues.append({"type": "annotation", "detail": f"{source_id}: {detail}"})
        checked_images += 1; checked_instances += len(labels); class_counts.update(labels)
        if row.get("annotation_version") and annotation.get("version") != row["annotation_version"]:
            issues.append({"type": "annotation_version", "detail": source_id})
        for key, actual in (("image_width", width), ("image_height", height)):
            if row.get(key) and int(row[key]) != actual:
                issues.append({"type": "dimensions", "detail": f"{source_id}: {key}"})
        if int(row.get("instance_count") or -1) != len(labels):
            issues.append({"type": "instance_count", "detail": source_id})
        if set(filter(None, row.get("labels_present", "").split("|"))) != set(labels):
            issues.append({"type": "labels_present", "detail": source_id})
        for hash_key, path in (("export_image_sha256", image_path), ("export_json_sha256", json_path)):
            if sha256_file(path) != row.get(hash_key):
                issues.append({"type": "export_file_hash", "detail": f"{source_id}: {hash_key}"})
        if row.get("export_image_sha256") != row.get("source_image_sha256") or row.get("export_json_sha256") != row.get("source_json_sha256"):
            issues.append({"type": "source_export_hash", "detail": source_id})
    missing_stems = sorted(set(frozen_by_stem) - set(export_by_stem))
    if missing_stems:
        issues.append({"type": "exact_set", "detail": f"{len(missing_stems)} frozen TRAIN source_ids missing"})
    expected_images = {row.get("export_image_path", "") for row in export_rows}
    expected_jsons = {row.get("export_json_path", "") for row in export_rows}
    if _files(data_root, "images/train") != expected_images:
        issues.append({"type": "image_exact_set", "detail": "actual TRAIN image set differs from export manifest"})
    if _files(data_root, "annotations/train", ".json") != expected_jsons:
        issues.append({"type": "json_exact_set", "detail": "actual TRAIN JSON set differs from export manifest"})
    if checked_instances != expected_instances:
        issues.append({"type": "train_instance_count", "detail": f"expected {expected_instances}, got {checked_instances}"})
    result = {
        "status": "PASS" if not issues and frozen_hash == expected_manifest_sha256 and checked_images == expected_count else "FAIL",
        "frozen_manifest_sha256": frozen_hash, "expected_frozen_manifest_sha256": expected_manifest_sha256,
        "export_manifest_sha256": sha256_file(export_manifest) if export_manifest.is_file() else "MISSING",
        "expected_train_images": expected_count, "checked_train_images": checked_images,
        "expected_train_instances": expected_instances, "checked_train_instances": checked_instances,
        "class_instance_counts": dict(sorted(class_counts.items())), "source_count": len(export_by_stem),
        "frozen_train_exact_set": not missing_stems and len(export_rows) == expected_count,
        "export_file_hashes_match": not any(item["type"] in {"export_file_hash", "source_export_hash"} for item in issues),
        "issues": issues, "val_test_image_access": 0, "val_test_json_access": 0, "raw_write": 0,
    }
    return result


def create_archive(root: Path, archive: Path) -> int:
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != archive:
                handle.write(path, path.relative_to(root).as_posix())
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii")
    return len(zipfile.ZipFile(archive).namelist())


def verify_archive(archive: Path, report: dict[str, Any], data_root: Path) -> list[dict[str, str]]:
    if not archive.is_file():
        return [{"type": "archive", "detail": "archive missing"}]
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    if not checksum_path.is_file():
        return [{"type": "archive_checksum", "detail": "archive SHA256 sidecar missing"}]
    expected_archive_hash = checksum_path.read_text(encoding="ascii").split()[0]
    if sha256_file(archive) != expected_archive_hash:
        return [{"type": "archive_checksum", "detail": "archive SHA256 mismatch"}]
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        names = {info.filename for info in infos}
        if len(names) != len(infos):
            return [{"type": "archive_members", "detail": "duplicate ZIP member names"}]
    required = {"frozen_split_manifest.csv", "export_manifest.csv", "export_verification.json", "export_record.json", "class_mapping.json"}
    export_rows = read_rows(Path(report["export_manifest"]))
    required.update(row.get("export_image_path", "") for row in export_rows)
    required.update(row.get("export_json_path", "") for row in export_rows)
    if names != required:
        return [{"type": "archive_members", "detail": f"ZIP member set differs (missing={len(required - names)}, extra={len(names - required)})"}]
    mismatches = []
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(required):
            expected_path = data_root / member
            if not expected_path.is_file() or hashlib.sha256(handle.read(member)).hexdigest() != sha256_file(expected_path):
                mismatches.append(member)
    return [{"type": "archive_content", "detail": f"{len(mismatches)} ZIP members differ from verified extracted files"}] if mismatches else []


def verify(args: argparse.Namespace) -> dict[str, Any]:
    report = validate_export_data(args.manifest, args.export_manifest, args.source_root, args.class_mapping, args.expected_manifest_sha256, args.expected_count, args.expected_instances)
    report.update({"frozen_manifest": str(args.manifest), "export_manifest": str(args.export_manifest)})
    if args.archive:
        report["archive_issues"] = verify_archive(args.archive, report, args.source_root)
        report["issues"].extend(report["archive_issues"])
        if report["archive_issues"]:
            report["status"] = "FAIL"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True); write_json(args.output, report)
    return report


def export_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source_root, output = args.source_root.resolve(), args.output.resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("export output must be outside the immutable source root")
    frozen_hash = sha256_file(args.manifest)
    if frozen_hash != args.expected_manifest_sha256:
        raise RuntimeError(f"frozen manifest SHA256 mismatch: {frozen_hash}")
    frozen_bytes = args.manifest.read_bytes()
    frozen_rows = read_rows(args.manifest)
    issues = validate_frozen_rows(frozen_rows, args.expected_count)
    if issues:
        raise RuntimeError("frozen manifest validation failed: " + json.dumps(issues, ensure_ascii=False))
    output.mkdir(parents=True)
    images_dir, json_dir = output / "images" / "train", output / "annotations" / "train"
    images_dir.mkdir(parents=True); json_dir.mkdir(parents=True)
    mapping = load_class_mapping(args.class_mapping)
    export_rows: list[dict[str, str]] = []
    for row in frozen_rows:
        if row.get("split") != "train":
            continue
        image_src, json_src = resolve_source(source_root, row["image_path"]), resolve_source(source_root, row["json_path"])
        if not image_src.is_file() or not json_src.is_file():
            raise FileNotFoundError(f"missing source pair: {row.get('stem')}")
        if sha256_file(image_src) != row.get("sha256") or sha256_file(json_src) != row.get("json_sha256"):
            raise RuntimeError(f"frozen source hash mismatch: {row.get('stem')}")
        image_dst, json_dst = images_dir / image_src.name, json_dir / f"{Path(row['json_path']).stem}.json"
        if image_dst.exists() or json_dst.exists():
            raise RuntimeError(f"export basename collision: {row.get('stem')}")
        shutil.copy2(image_src, image_dst); shutil.copy2(json_src, json_dst)
        export_rows.append({
            "source_id": row["stem"], "source_image_path": row["image_path"], "source_json_path": row["json_path"],
            "source_image_sha256": row["sha256"], "source_json_sha256": row["json_sha256"],
            "export_image_path": image_dst.relative_to(output).as_posix(), "export_json_path": json_dst.relative_to(output).as_posix(),
            "export_image_sha256": sha256_file(image_dst), "export_json_sha256": sha256_file(json_dst),
            "split": row["split"], "group_id": row["group_id"], "labels_present": row["labels_present"],
            "instance_count": row["instance_count"], "image_width": row["image_width"], "image_height": row["image_height"],
            "annotation_version": json.loads(json_dst.read_text(encoding="utf-8-sig")).get("version", ""),
        })
    frozen_copy = output / "frozen_split_manifest.csv"
    frozen_copy.write_bytes(frozen_bytes)
    if frozen_copy.read_bytes() != frozen_bytes:
        raise RuntimeError("frozen manifest copy is not byte-identical")
    export_manifest = output / "export_manifest.csv"
    with export_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS); writer.writeheader(); writer.writerows(export_rows)
    mapping_path = output / "class_mapping.json"
    write_json(mapping_path, {"names": mapping})
    validation = validate_export_data(frozen_copy, export_manifest, output, mapping_path, args.expected_manifest_sha256, args.expected_count, args.expected_instances)
    if validation["status"] != "PASS":
        raise RuntimeError("export self-verification failed: " + json.dumps(validation["issues"], ensure_ascii=False))
    validation["frozen_manifest"] = "frozen_split_manifest.csv"
    validation["export_manifest"] = "export_manifest.csv"
    write_json(output / "export_verification.json", validation)
    record = {"status": "EXPORTED_TRAIN_ONLY_AND_VERIFIED", "frozen_manifest_sha256": frozen_hash, "export_manifest_sha256": sha256_file(export_manifest), "train_images": validation["checked_train_images"], "train_instances": validation["checked_train_instances"], "class_instance_counts": validation["class_instance_counts"], "source_count": validation["source_count"], "archive_name": "train_export.zip", "val_test_image_access": 0, "val_test_json_access": 0, "raw_write": 0}
    write_json(output / "export_record.json", record)
    archive = output / "train_export.zip"
    members = create_archive(output, archive)
    record["archive_members"] = members
    record["archive_sha256"] = sha256_file(archive)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("export", "verify"), required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="original frozen split_manifest.csv")
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--class-mapping", type=Path)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TRAIN_IMAGES)
    parser.add_argument("--expected-instances", type=int, default=EXPECTED_TRAIN_INSTANCES)
    args = parser.parse_args()
    if args.mode == "export":
        result = export_train(args)
    else:
        args.export_manifest = args.export_manifest or args.source_root / "export_manifest.csv"
        args.class_mapping = args.class_mapping or args.source_root / "class_mapping.json"
        result = verify(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
