#!/usr/bin/env python3
"""CPU-only technical and preliminary visual-risk audit for one saved Gate B output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


STATUS = "FIRST_CALL_RAW / PENDING_HUMAN_REVIEW"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_nonzero(mask: Image.Image) -> int:
    histogram = mask.convert("L").histogram()
    return sum(histogram[1:])


def logical_and(first: Image.Image, second: Image.Image) -> Image.Image:
    return ImageChops.multiply(first.convert("L"), second.convert("L"))


def labeled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", (size[0], size[1] + 34), "white")
    fitted = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    panel.paste(fitted, (0, 34))
    ImageDraw.Draw(panel).text((10, 9), label, fill="black", font=ImageFont.load_default())
    return panel


def audit_run(run_dir: Path) -> dict:
    original_path = run_dir / "original.png"
    mask_path = run_dir / "api_mask.png"
    polygon_path = run_dir / "polygon_mask.png"
    overlay_path = run_dir / "overlay_check.png"
    output_path = run_dir / "raw_output.png"
    response_path = run_dir / "response_manifest.json"
    request_path = run_dir / "request_manifest.json"
    for path in [original_path, mask_path, polygon_path, overlay_path, output_path, response_path, request_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    audit_path = run_dir / "audit_report.json"
    sheet_path = run_dir / "review_sheet.png"
    summary_path = run_dir / "first_call_summary.md"
    for path in [audit_path, sheet_path, summary_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    with Image.open(original_path) as handle:
        handle.load()
        original = handle.convert("RGB")
        original_format = handle.format
    with Image.open(output_path) as handle:
        handle.load()
        output = handle.convert("RGB")
        output_format = handle.format
        output_mode = handle.mode
    with Image.open(mask_path) as handle:
        handle.load()
        api_mask = handle.convert("RGBA")
    with Image.open(polygon_path) as handle:
        handle.load()
        polygon = handle.convert("L").point(lambda value: 255 if value > 0 else 0)
    with Image.open(overlay_path) as handle:
        handle.load()
        overlay = handle.convert("RGB")

    same_size = output.size == original.size == api_mask.size == polygon.size
    pixel_audit: dict = {"status": "PASS" if same_size else "UNAVAILABLE_SIZE_MISMATCH"}
    if same_size:
        alpha = api_mask.getchannel("A")
        transparent_edit = alpha.point(lambda value: 255 if value == 0 else 0)
        if ImageChops.difference(transparent_edit, polygon).getbbox() is not None:
            raise ValueError("API mask transparent region does not match polygon mask")
        preserve = ImageChops.invert(transparent_edit)
        difference = ImageChops.difference(original, output)
        channels = difference.split()
        max_difference = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        changed_gt5 = max_difference.point(lambda value: 255 if value > 5 else 0)
        changed_gt10 = max_difference.point(lambda value: 255 if value > 10 else 0)
        edit_pixels = count_nonzero(transparent_edit)
        preserve_pixels = count_nonzero(preserve)
        inside_gt5 = count_nonzero(logical_and(changed_gt5, transparent_edit))
        outside_gt5 = count_nonzero(logical_and(changed_gt5, preserve))
        inside_gt10 = count_nonzero(logical_and(changed_gt10, transparent_edit))
        outside_gt10 = count_nonzero(logical_and(changed_gt10, preserve))
        inside_mean = sum(ImageStat.Stat(difference, mask=transparent_edit).mean) / 3.0
        outside_mean = sum(ImageStat.Stat(difference, mask=preserve).mean) / 3.0
        outside_fraction = outside_gt5 / preserve_pixels if preserve_pixels else None
        pixel_audit.update({
            "edit_pixels": edit_pixels,
            "preserve_pixels": preserve_pixels,
            "inside_mask_changed_pixels_max_channel_gt5": inside_gt5,
            "outside_mask_changed_pixels_max_channel_gt5": outside_gt5,
            "inside_mask_changed_pixels_max_channel_gt10": inside_gt10,
            "outside_mask_changed_pixels_max_channel_gt10": outside_gt10,
            "inside_mask_mean_absolute_channel_difference": inside_mean,
            "outside_mask_mean_absolute_channel_difference": outside_mean,
            "outside_mask_changed_fraction_gt5": outside_fraction,
        })
        amplified = difference.point(lambda value: min(255, value * 4))
        panels = [
            labeled_panel(original, "Original", original.size),
            labeled_panel(overlay, "Polygon overlay", original.size),
            labeled_panel(output, "Raw API output", original.size),
            labeled_panel(amplified, "Absolute difference x4", original.size),
        ]
        sheet = Image.new("RGB", (original.width * 2, (original.height + 34) * 2), (230, 230, 230))
        sheet.paste(panels[0], (0, 0))
        sheet.paste(panels[1], (original.width, 0))
        sheet.paste(panels[2], (0, original.height + 34))
        sheet.paste(panels[3], (original.width, original.height + 34))
        sheet.save(sheet_path, format="PNG")
        scene_risk = (
            "HIGH_RISK_BY_OUTSIDE_MASK_PIXEL_CHANGE_HEURISTIC"
            if outside_fraction is not None and outside_fraction > 0.15
            else "NOT_RESOLVED_REQUIRES_HUMAN_REVIEW"
        )
    else:
        scene_risk = "HIGH_RISK_OUTPUT_SIZE_MISMATCH"
        sheet = Image.new("RGB", (1280, 720), "white")
        ImageDraw.Draw(sheet).text((20, 20), "Output size mismatch; pixel comparison unavailable", fill="red")
        sheet.save(sheet_path, format="PNG")

    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    audit = {
        "status": STATUS,
        "source": "corrosion23",
        "model": request["model"],
        "post_request_count": request["post_request_count"],
        "image_result_received": True,
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "format": output_format,
            "mode": output_mode,
            "size_wh": list(output.size),
            "original_format": original_format,
        },
        "latency_seconds": response.get("latency_seconds"),
        "usage": response.get("usage"),
        "pixel_audit": pixel_audit,
        "preliminary_semantic_audit": {
            "obvious_overall_scene_reconstruction": scene_risk,
            "new_mechanical_structure_risk": "PENDING_HUMAN_REVIEW",
            "target_corrosion_removed": "PENDING_HUMAN_REVIEW",
            "geometry_preserved": "PENDING_HUMAN_REVIEW",
            "outside_region_preserved": "PENDING_HUMAN_REVIEW",
            "automatic_visual_pass": False,
        },
        "training_pair_eligibility": "NOT_EVALUATED_NOT_FROZEN",
        "review_sheet": {"path": str(sheet_path.resolve()), "sha256": sha256(sheet_path)},
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        "# Exp13-G1B corrosion23 first live call\n\n"
        f"Status: `{STATUS}`\n\n"
        f"- Model: `{request['model']}`\n"
        f"- POST requests: `{request['post_request_count']}`\n"
        f"- Output SHA256: `{audit['output']['sha256']}`\n"
        f"- Output: `{output_format}`, `{output_mode}`, `{output.width}x{output.height}`\n"
        f"- Latency: `{response.get('latency_seconds')}` seconds\n"
        f"- Usage: `{json.dumps(response.get('usage'), ensure_ascii=False)}`\n"
        f"- Outside-mask changed fraction (>5): `{pixel_audit.get('outside_mask_changed_fraction_gt5')}`\n"
        f"- Scene reconstruction risk: `{scene_risk}`\n"
        "- Mechanical structure risk: `PENDING_HUMAN_REVIEW`\n"
        "- Training-pair eligibility: `NOT_EVALUATED_NOT_FROZEN`\n\n"
        "No second request, retry, fallback, or automatic pair admission is authorized.\n",
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    audit = audit_run(args.run_dir)
    print(json.dumps({"status": audit["status"], "output_sha256": audit["output"]["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
