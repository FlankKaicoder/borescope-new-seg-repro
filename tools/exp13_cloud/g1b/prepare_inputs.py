#!/usr/bin/env python3
"""Prepare the fixed corrosion23 image and transparent edit mask from the handoff ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

from PIL import Image, ImageDraw


EXPECTED_ZIP_SHA256 = "d9f049bda223f6485644da71839aed670a78b169b7411f722593c32fabe04f18"
IMAGE_MEMBER = "exp13_cloud_pilot4/sources/corrosion23/original_image.png"
JSON_MEMBER = "exp13_cloud_pilot4/sources/corrosion23/original_annotation.json"
H1_PREFIX = "exp13_cloud_pilot4/historical_reference/H1_corrosion23/"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    if sha256_file(args.zip) != EXPECTED_ZIP_SHA256:
        raise ValueError("Pilot-4 ZIP SHA256 mismatch")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: args.run_dir / name for name in [
        "original.png", "original_annotation.json", "polygon_mask.png",
        "api_mask.png", "overlay_check.png", "prepare_report.json",
    ]}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    with zipfile.ZipFile(args.zip) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        names = archive.namelist()
        for name in [IMAGE_MEMBER, JSON_MEMBER]:
            if name not in names:
                raise FileNotFoundError(f"Missing ZIP member: {name}")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe ZIP member: {name}")
        image_bytes = archive.read(IMAGE_MEMBER)
        json_bytes = archive.read(JSON_MEMBER)
        h1_references = sorted(name for name in names if name.startswith(H1_PREFIX))

    outputs["original.png"].write_bytes(image_bytes)
    outputs["original_annotation.json"].write_bytes(json_bytes)
    annotation = json.loads(json_bytes.decode("utf-8-sig"))

    with Image.open(outputs["original.png"]) as source:
        source.load()
        image = source.convert("RGB")
        width, height = image.size
        image_format = source.format
    if [annotation.get("imageWidth"), annotation.get("imageHeight")] != [width, height]:
        raise ValueError("Original image size does not match JSON")

    polygons = []
    for index, shape in enumerate(annotation.get("shapes", [])):
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"Shape {index} is not a polygon")
        points = shape.get("points", [])
        if len(points) < 3:
            raise ValueError(f"Shape {index} has fewer than three points")
        normalized = [[float(x), float(y)] for x, y in points]
        in_bounds = all(0.0 <= x <= width and 0.0 <= y <= height for x, y in normalized)
        if not in_bounds:
            raise ValueError(f"Shape {index} is out of image bounds")
        if shape.get("label") != "corrosion":
            raise ValueError(f"Unexpected label in corrosion23: {shape.get('label')}")
        polygons.append({"index": index, "label": shape["label"], "points": normalized})
    if not polygons:
        raise ValueError("No corrosion polygons found")

    binary = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(binary)
    for polygon in polygons:
        vertices = [(round(x), round(y)) for x, y in polygon["points"]]
        draw.polygon(vertices, fill=255)
    histogram = binary.histogram()
    target_pixels = int(histogram[255])
    total_pixels = width * height
    if target_pixels <= 0 or target_pixels >= total_pixels:
        raise ValueError("Polygon mask must be nonempty and smaller than the full image")
    binary.save(outputs["polygon_mask.png"], format="PNG")

    # The fixed prompt defines transparent pixels as the repair region.
    alpha = binary.point(lambda value: 0 if value > 0 else 255, mode="L")
    api_mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    api_mask.putalpha(alpha)
    api_mask.save(outputs["api_mask.png"], format="PNG")
    alpha_histogram = alpha.histogram()
    if alpha_histogram[0] != target_pixels or alpha_histogram[255] != total_pixels - target_pixels:
        raise ValueError("API alpha mask polarity/count mismatch")

    overlay = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(binary.point(lambda value: 112 if value > 0 else 0, mode="L"))
    overlay = Image.alpha_composite(overlay, red).convert("RGB")
    outline = ImageDraw.Draw(overlay)
    for polygon in polygons:
        vertices = [(round(x), round(y)) for x, y in polygon["points"]]
        outline.line(vertices + [vertices[0]], fill=(255, 255, 0), width=2)
    overlay.save(outputs["overlay_check.png"], format="PNG")

    report = {
        "status": "PASS",
        "source": "corrosion23",
        "source_zip": str(args.zip.resolve()),
        "source_zip_sha256": EXPECTED_ZIP_SHA256,
        "zip_crc_status": "PASS",
        "zip_members": {"image": IMAGE_MEMBER, "annotation": JSON_MEMBER},
        "original": {
            "image_format": image_format,
            "size_wh": [width, height],
            "image_sha256": sha256_bytes(image_bytes),
            "annotation_sha256": sha256_bytes(json_bytes),
            "json_size_match": True,
        },
        "annotation": {
            "polygon_count": len(polygons),
            "labels": sorted({item["label"] for item in polygons}),
            "all_points_in_bounds": True,
            "polygons": polygons,
        },
        "polygon_mask": {
            "path": str(outputs["polygon_mask.png"].resolve()),
            "sha256": sha256_file(outputs["polygon_mask.png"]),
            "mode": "L",
            "target_value": 255,
            "target_pixels": target_pixels,
            "total_pixels": total_pixels,
            "nonempty": True,
            "not_full_image": True,
        },
        "api_mask": {
            "path": str(outputs["api_mask.png"].resolve()),
            "sha256": sha256_file(outputs["api_mask.png"]),
            "mode": "RGBA",
            "format": "PNG",
            "size_matches_image": True,
            "transparent_repair_pixels": alpha_histogram[0],
            "opaque_preserve_pixels": alpha_histogram[255],
            "polarity": "alpha=0 edit polygon; alpha=255 preserve outside",
            "official_requirement_basis": "OpenAI image generation guide: same size/format and alpha channel",
        },
        "overlay": {
            "path": str(outputs["overlay_check.png"].resolve()),
            "sha256": sha256_file(outputs["overlay_check.png"]),
            "render_status": "PASS",
            "meaning": "red fill/yellow outline show the exact source-coordinate target polygons",
        },
        "historical_h1_reference_members": h1_references,
        "historical_h1_used_for_generation": False,
    }
    write_json(outputs["prepare_report.json"], report)
    print(json.dumps({"status": "PASS", "size_wh": [width, height], "polygon_count": len(polygons), "target_pixels": target_pixels}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
