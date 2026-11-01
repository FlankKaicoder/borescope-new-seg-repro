#!/usr/bin/env python3
"""Offline audit for the two completed G1E mask-control conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


LEFT = (100, 160, 220, 320)
RIGHT = (420, 160, 540, 320)
CANVAS_SIZE = (640, 480)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, mode: str = "RGB") -> Image.Image:
    with Image.open(path) as handle:
        handle.load()
        return handle.convert(mode)


def save_new(image: Image.Image, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    image.save(path, format="PNG")


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def observation_view(raw: Image.Image) -> tuple[Image.Image, dict]:
    if raw.size == CANVAS_SIZE:
        return raw.copy(), {"method": "IDENTITY", "raw_size_wh": list(raw.size), "observation_size_wh": list(CANVAS_SIZE)}
    exact_aspect = raw.width * CANVAS_SIZE[1] == CANVAS_SIZE[0] * raw.height
    if exact_aspect:
        result = raw.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        return result, {"method": "LANCZOS_EXACT_ASPECT_RESIZE_FOR_OBSERVATION_ONLY", "raw_size_wh": list(raw.size), "observation_size_wh": list(CANVAS_SIZE)}
    contained = ImageOps.contain(raw, CANVAS_SIZE, Image.Resampling.LANCZOS)
    result = Image.new("RGB", CANVAS_SIZE, (211, 211, 211))
    offset = ((CANVAS_SIZE[0] - contained.width) // 2, (CANVAS_SIZE[1] - contained.height) // 2)
    result.paste(contained, offset)
    return result, {"method": "LANCZOS_ASPECT_PRESERVING_CONTAIN_WITH_GRAY_LETTERBOX_FOR_OBSERVATION_ONLY", "raw_size_wh": list(raw.size), "observation_size_wh": list(CANVAS_SIZE), "content_size_wh": list(contained.size), "offset_xy": list(offset)}


def region_color_stats(image: Image.Image, box: tuple[int, int, int, int]) -> dict:
    pixels = list(image.crop(box).convert("RGB").getdata())
    total = len(pixels)
    red = sum(1 for r, g, b in pixels if r >= 180 and g <= 110 and b <= 110)
    blue = sum(1 for r, g, b in pixels if b >= 160 and r <= 110 and g <= 130)
    mean = [sum(pixel[index] for pixel in pixels) / total for index in range(3)]
    return {"pixels": total, "red_fraction": red / total, "blue_fraction": blue / total, "mean_rgb": mean}


def background_mask() -> Image.Image:
    mask = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(mask)
    for box in [LEFT, RIGHT]:
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=0)
    return mask


def changed_fraction(first: Image.Image, second: Image.Image, mask: Image.Image, threshold: int = 10) -> float:
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    red, green, blue = difference.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    changed = maximum.point(lambda value: 255 if value > threshold else 0)
    selected = ImageChops.multiply(changed, mask)
    mask_pixels = sum(mask.histogram()[1:])
    changed_pixels = sum(selected.histogram()[1:])
    return changed_pixels / mask_pixels if mask_pixels else 0.0


def panel(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (640, 520), "white")
    canvas.paste(image.resize(CANVAS_SIZE, Image.Resampling.LANCZOS), (0, 40))
    ImageDraw.Draw(canvas).text((10, 12), title, fill="black", font=ImageFont.load_default())
    return canvas


def audit(root: Path) -> dict:
    inputs = root / "inputs"
    reports = root / "reports"
    run_a = root / "runs" / "condition_A"
    run_b = root / "runs" / "condition_B"
    original = load(inputs / "synthetic_original.png")
    overlay_a = load(inputs / "overlay_left.png")
    overlay_b = load(inputs / "overlay_right.png")
    raw_a = load(run_a / "raw_output.png")
    raw_b = load(run_b / "raw_output.png")
    view_a, transform_a = observation_view(raw_a)
    view_b, transform_b = observation_view(raw_b)
    save_new(view_a, run_a / "observation_640x480.png")
    save_new(view_b, run_b / "observation_640x480.png")

    bg = background_mask()
    stats = {
        "A": {"left": region_color_stats(view_a, LEFT), "right": region_color_stats(view_a, RIGHT), "background_changed_fraction_gt10": changed_fraction(original, view_a, bg), "observation_transform": transform_a},
        "B": {"left": region_color_stats(view_b, LEFT), "right": region_color_stats(view_b, RIGHT), "background_changed_fraction_gt10": changed_fraction(original, view_b, bg), "observation_transform": transform_b},
    }
    a_localized = stats["A"]["left"]["red_fraction"] >= 0.50 and stats["A"]["right"]["red_fraction"] <= 0.15
    b_localized = stats["B"]["right"]["red_fraction"] >= 0.50 and stats["B"]["left"]["red_fraction"] <= 0.15
    spatial_switch = (
        stats["A"]["left"]["red_fraction"] - stats["A"]["right"]["red_fraction"] >= 0.35
        and stats["B"]["right"]["red_fraction"] - stats["B"]["left"]["red_fraction"] >= 0.35
    )
    both_red_a = stats["A"]["left"]["red_fraction"] >= 0.40 and stats["A"]["right"]["red_fraction"] >= 0.40
    both_red_b = stats["B"]["left"]["red_fraction"] >= 0.40 and stats["B"]["right"]["red_fraction"] >= 0.40
    if a_localized and b_localized and spatial_switch:
        conclusion = "MASK_CONDITIONED_SPATIAL_RESPONSE_OBSERVED"
    elif both_red_a and both_red_b:
        conclusion = "MASK_SPATIAL_CONTROL_NOT_DEMONSTRATED"
    else:
        conclusion = "MASK_SPATIAL_CONTROL_INCONCLUSIVE"

    panels = [
        panel(original, "Synthetic original"),
        panel(overlay_a, "Condition A mask overlay (review only)"),
        panel(view_a, f"Condition A observation | L red={stats['A']['left']['red_fraction']:.3f} R red={stats['A']['right']['red_fraction']:.3f}"),
        panel(overlay_b, "Condition B mask overlay (review only)"),
        panel(view_b, f"Condition B observation | L red={stats['B']['left']['red_fraction']:.3f} R red={stats['B']['right']['red_fraction']:.3f}"),
        panel(Image.new("RGB", CANVAS_SIZE, "white"), f"Approximate result: {conclusion}"),
    ]
    sheet = Image.new("RGB", (1920, 1040), (225, 225, 225))
    for index, item in enumerate(panels):
        sheet.paste(item, ((index % 3) * 640, (index // 3) * 520))
    save_new(sheet, reports / "g1e_review_sheet.png")

    result = {
        "status": conclusion,
        "scientific_boundary": {
            "strict_mask_compliance_proven": False,
            "upstream_mask_forwarding_directly_proven": False,
            "statistical_significance_claimed": False,
            "geometry_warning": "Fixed rectangle statistics are approximate if generated output drifts geometrically.",
        },
        "thresholds": {"red": "R>=180,G<=110,B<=110", "blue": "B>=160,R<=110,G<=130", "target_red_min": 0.50, "non_target_red_max": 0.15, "spatial_difference_min": 0.35},
        "statistics": stats,
        "raw_outputs": {
            "A": {"path": str(run_a / "raw_output.png"), "sha256": sha256(run_a / "raw_output.png"), "size_wh": list(raw_a.size)},
            "B": {"path": str(run_b / "raw_output.png"), "sha256": sha256(run_b / "raw_output.png"), "size_wh": list(raw_b.size)},
        },
        "review_sheet": {"path": str(reports / "g1e_review_sheet.png"), "sha256": sha256(reports / "g1e_review_sheet.png")},
        "human_review": {"background_redraw": "PENDING", "rectangle_geometry_preserved": "PENDING", "unrelated_shapes_added": "PENDING"},
    }
    write_json_new(reports / "g1e_pair_audit.json", result)
    final_path = reports / "G1E_FINAL_REPORT.md"
    if final_path.exists():
        raise FileExistsError(final_path)
    final_path.write_text(
        "# Exp13-G1E mask spatial-control result\n\n"
        f"Status: `{conclusion}`\n\n"
        "This is a two-condition exploratory control, not a statistically significant experiment. Fixed-coordinate color metrics are approximate when generated geometry drifts.\n\n"
        f"- A left/right red fractions: `{stats['A']['left']['red_fraction']:.6f}` / `{stats['A']['right']['red_fraction']:.6f}`\n"
        f"- B left/right red fractions: `{stats['B']['left']['red_fraction']:.6f}` / `{stats['B']['right']['red_fraction']:.6f}`\n"
        f"- A/B background changed fractions (>10): `{stats['A']['background_changed_fraction_gt10']:.6f}` / `{stats['B']['background_changed_fraction_gt10']:.6f}`\n\n"
        "`STRICT_MASK_COMPLIANCE_PROVEN` is not claimed. `UPSTREAM_MASK_FORWARDING_DIRECTLY_PROVEN` is not claimed. Human review remains required.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps({"status": result["status"], "review_sheet": result["review_sheet"]["path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
