#!/usr/bin/env python3
"""CPU-only G1C resize, geometry audit, local composites, and review assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat


EXPECTED = {
    "original.png": "c57384471d743fc179add88589fa0d7c8d59e59d14c6947ab9f6d4fd692dbddb",
    "polygon_mask.png": "69a8275afe293cb9086a5730cf5c18aa4db54bd3b109cafd7c55cb3bb3df5021",
    "api_mask.png": "cdd794f0432f09d3fbb5167112a4b03e9692970357871abb133728d92991bc02",
    "raw_output.png": "492ad7779bd026d767e93ff86e059b02f09fbbb5e3be7880bde828acb54903f6",
}
STATUS = "G1C_POSTPROCESS_COMPLETE / PENDING_HUMAN_REVIEW"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_new(image: Image.Image, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    image.save(path, format="PNG")


def load_image(path: Path, mode: str | None = None) -> Image.Image:
    with Image.open(path) as handle:
        handle.load()
        image = handle.copy()
    return image.convert(mode) if mode else image


def count_nonzero(mask: Image.Image) -> int:
    histogram = mask.convert("L").histogram()
    return sum(histogram[1:])


def difference_metrics(first: Image.Image, second: Image.Image, region: Image.Image) -> dict:
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    bands = difference.split()
    maximum = ImageChops.lighter(ImageChops.lighter(bands[0], bands[1]), bands[2])
    region_l = region.convert("L")
    pixels = count_nonzero(region_l)
    changed_any = count_nonzero(ImageChops.multiply(maximum.point(lambda v: 255 if v > 0 else 0), region_l))
    changed_gt5 = count_nonzero(ImageChops.multiply(maximum.point(lambda v: 255 if v > 5 else 0), region_l))
    changed_gt10 = count_nonzero(ImageChops.multiply(maximum.point(lambda v: 255 if v > 10 else 0), region_l))
    mean = sum(ImageStat.Stat(difference, mask=region_l).mean) / 3.0 if pixels else None
    return {
        "region_pixels": pixels,
        "changed_any_count": changed_any,
        "changed_max_channel_gt5_count": changed_gt5,
        "changed_max_channel_gt10_count": changed_gt10,
        "changed_gt5_fraction": changed_gt5 / pixels if pixels else None,
        "mean_absolute_channel_difference": mean,
        "max_channel_difference": maximum.getextrema()[1],
    }


def edge_map(image: Image.Image) -> Image.Image:
    edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
    draw = ImageDraw.Draw(edge)
    draw.rectangle((0, 0, edge.width - 1, edge.height - 1), outline=0, width=2)
    return edge


def binary_edges(image: Image.Image, threshold: int = 30) -> Image.Image:
    return edge_map(image).point(lambda value: 255 if value >= threshold else 0)


def translate_mask(image: Image.Image, dx: int, dy: int) -> Image.Image:
    shifted = Image.new("L", image.size, 0)
    shifted.paste(image, (dx, dy))
    return shifted


def edge_translation_search(original: Image.Image, generated: Image.Image, valid: Image.Image, radius: int = 12) -> dict:
    first = ImageChops.multiply(binary_edges(original), valid)
    second = ImageChops.multiply(binary_edges(generated), valid)
    results = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = ImageChops.multiply(translate_mask(first, dx, dy), valid)
            intersection = count_nonzero(ImageChops.multiply(shifted, second))
            first_count = count_nonzero(shifted)
            second_count = count_nonzero(second)
            dice = 2.0 * intersection / (first_count + second_count) if first_count + second_count else 0.0
            results.append((dice, dx, dy, intersection, first_count, second_count))
    results.sort(reverse=True)
    best = results[0]
    zero = next(item for item in results if item[1] == 0 and item[2] == 0)
    return {
        "method": "binary FIND_EDGES threshold=30, Dice search on non-mask region excluded by 10px",
        "search_radius_pixels": radius,
        "best_translation_dx_dy": [best[1], best[2]],
        "best_dice": best[0],
        "best_edge_intersection_count": best[3],
        "original_edge_count_at_best": best[4],
        "generated_edge_count": best[5],
        "zero_translation_dice": zero[0],
        "dice_improvement_over_zero": best[0] - zero[0],
        "interpretation_limit": "Global edge overlap is not feature correspondence and cannot prove pixel registration.",
    }


def patch_values(image: Image.Image, cx: int, cy: int, radius: int) -> list[float]:
    return [float(value) for value in image.crop((cx - radius, cy - radius, cx + radius + 1, cy + radius + 1)).getdata()]


def ncc(first: list[float], second: list[float]) -> float:
    mean_a = sum(first) / len(first)
    mean_b = sum(second) / len(second)
    centered_a = [value - mean_a for value in first]
    centered_b = [value - mean_b for value in second]
    denom = math.sqrt(sum(value * value for value in centered_a) * sum(value * value for value in centered_b))
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denom if denom > 1e-9 else -1.0


def local_patch_matches(original: Image.Image, generated: Image.Image, valid: Image.Image) -> dict:
    first = edge_map(original)
    second = edge_map(generated)
    radius = 7
    search = 12
    candidates = []
    valid_px = valid.load()
    for cy in range(24, original.height - 24, 32):
        for cx in range(24, original.width - 24, 32):
            if any(valid_px[x, y] == 0 for y in range(cy - radius - search, cy + radius + search + 1, 4) for x in range(cx - radius - search, cx + radius + search + 1, 4)):
                continue
            values = patch_values(first, cx, cy, radius)
            variance = statistics.pvariance(values)
            if variance >= 180.0:
                candidates.append((variance, cx, cy, values))
    candidates.sort(reverse=True)
    matches = []
    for variance, cx, cy, reference in candidates[:30]:
        scores = []
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                score = ncc(reference, patch_values(second, cx + dx, cy + dy, radius))
                scores.append((score, dx, dy))
        scores.sort(reverse=True)
        best = scores[0]
        separated = next((item for item in scores[1:] if abs(item[1] - best[1]) + abs(item[2] - best[2]) > 3), scores[-1])
        margin = best[0] - separated[0]
        if best[0] >= 0.55 and margin >= 0.03:
            matches.append({"center_xy": [cx, cy], "dx_dy": [best[1], best[2]], "ncc": best[0], "uniqueness_margin": margin, "edge_variance": variance})
    if matches:
        dxs = [item["dx_dy"][0] for item in matches]
        dys = [item["dx_dy"][1] for item in matches]
        median_dx = statistics.median(dxs)
        median_dy = statistics.median(dys)
        distances = [math.hypot(dx - median_dx, dy - median_dy) for dx, dy in zip(dxs, dys)]
        inliers = [distance <= 2.5 for distance in distances]
        inlier_count = sum(inliers)
        inlier_ratio = inlier_count / len(matches)
    else:
        median_dx = median_dy = None
        inlier_count = 0
        inlier_ratio = 0.0
    reliable = len(matches) >= 8 and inlier_ratio >= 0.65 and math.hypot(median_dx or 0, median_dy or 0) <= 2.0
    return {
        "method": "Pillow FIND_EDGES 15x15 normalized-correlation patches; fixed +/-12px search",
        "candidate_high_texture_patches": min(30, len(candidates)),
        "accepted_unique_matches": len(matches),
        "acceptance_thresholds": {"ncc_min": 0.55, "uniqueness_margin_min": 0.03},
        "median_displacement_dx_dy": [median_dx, median_dy],
        "inlier_radius_pixels": 2.5,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reliable_for_bounded_translation_check": reliable,
        "matches": matches,
        "limitation": "This is a lightweight photometric edge-patch check, not ORB/SIFT or dense registration.",
    }


def erode(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.MinFilter(radius * 2 + 1))


def dilate(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))


def internal_ring(mask: Image.Image, width: int) -> Image.Image:
    return ImageChops.subtract(mask, erode(mask, width))


def boundary_band(mask: Image.Image, width: int) -> Image.Image:
    return ImageChops.subtract(dilate(mask, width), erode(mask, width))


def changed_counts(candidate: Image.Image, original: Image.Image, mask: Image.Image) -> dict:
    outside = ImageChops.invert(mask)
    return {
        "inside": difference_metrics(candidate, original, mask),
        "outside": difference_metrics(candidate, original, outside),
    }


def boundary_metrics(original: Image.Image, aligned: Image.Image, hard: Image.Image, feather: Image.Image, mask: Image.Image) -> dict:
    results = {}
    for width in [1, 3, 5, 10]:
        inside = internal_ring(mask, width)
        band = boundary_band(mask, width)
        row = {"inside_ring_pixels": count_nonzero(inside), "two_sided_band_pixels": count_nonzero(band)}
        for name, candidate in [("aligned_raw", aligned), ("hard", hard), ("feather_inward_3px", feather)]:
            row[name] = {
                "inside_ring_difference_from_original": difference_metrics(candidate, original, inside),
                "two_sided_boundary_edge_mean": ImageStat.Stat(edge_map(candidate), mask=band).mean[0],
            }
        row["original_two_sided_boundary_edge_mean"] = ImageStat.Stat(edge_map(original), mask=band).mean[0]
        results[f"width_{width}px"] = row
    return results


def add_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image.convert("RGB"), (0, 34))
    ImageDraw.Draw(canvas).text((10, 10), title, fill="black", font=ImageFont.load_default())
    return canvas


def fit_crop(image: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    crop = image.crop(box)
    scale = min(size[0] / crop.width, size[1] / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (32, 32, 32))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def make_review_sheet(original: Image.Image, overlay: Image.Image, raw: Image.Image, aligned: Image.Image, hard: Image.Image, feather: Image.Image) -> Image.Image:
    panels = [
        add_title(original, "Original 640x480"),
        add_title(overlay, "Original polygon overlay"),
        add_title(raw.resize(original.size, Image.Resampling.LANCZOS), "FIRST_CALL_RAW preview (1448x1086 -> panel)"),
        add_title(aligned, "RESIZED_FOR_ALIGNMENT_ONLY (LANCZOS)"),
        add_title(hard, "POSTPROCESS_ONLY hard composite"),
        add_title(feather, "POSTPROCESS_ONLY inward feather 3px"),
    ]
    sheet = Image.new("RGB", (original.width * 3, (original.height + 34) * 2), (220, 220, 220))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * original.width, (index // 3) * (original.height + 34)))
    return sheet


def make_local_review(original: Image.Image, overlay: Image.Image, aligned: Image.Image, hard: Image.Image, feather: Image.Image, polygons: list[dict]) -> Image.Image:
    variants = [("Original", original), ("Overlay", overlay), ("Aligned raw", aligned), ("Hard", hard), ("Feather 3px", feather)]
    panel_size = (360, 240)
    rows = []
    for index, polygon in enumerate(polygons):
        xs = [point[0] for point in polygon["points"]]
        ys = [point[1] for point in polygon["points"]]
        box = (max(0, int(min(xs)) - 24), max(0, int(min(ys)) - 24), min(original.width, int(max(xs)) + 25), min(original.height, int(max(ys)) + 25))
        panels = [add_title(fit_crop(image, box, panel_size), f"Target {index + 1}: {name}") for name, image in variants]
        row = Image.new("RGB", (panel_size[0] * len(panels), panel_size[1] + 34), (220, 220, 220))
        for column, panel in enumerate(panels):
            row.paste(panel, (column * panel_size[0], 0))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), (220, 220, 220))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1b-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    run = args.g1b_run.resolve()
    root = args.output_root.resolve()
    inputs_dir, outputs_dir, reports_dir = root / "inputs", root / "outputs", root / "reports"
    for directory in [inputs_dir, outputs_dir, reports_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    expected_outputs = [
        inputs_dir / "input_manifest.json",
        outputs_dir / "aligned_output_640x480.png",
        outputs_dir / "composite_hard.png",
        outputs_dir / "composite_feather_inward_3px.png",
        outputs_dir / "review_sheet_g1c.png",
        outputs_dir / "local_boundary_review.png",
        reports_dir / "alignment_metadata.json",
        reports_dir / "geometry_alignment_report.json",
        reports_dir / "mask_composite_report.json",
        reports_dir / "boundary_audit_report.json",
        reports_dir / "G1C_FINAL_REPORT.md",
    ]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("One or more G1C outputs already exist; refusing overwrite")

    required = ["original.png", "original_annotation.json", "polygon_mask.png", "api_mask.png", "overlay_check.png", "raw_output.png", "prepare_report.json", "audit_report.json", "response_manifest.json"]
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    for name, expected in EXPECTED.items():
        actual = sha256(run / name)
        if actual != expected:
            raise ValueError(f"Frozen input hash mismatch for {name}: {actual}")

    prepare = json.loads((run / "prepare_report.json").read_text(encoding="utf-8"))
    annotation = json.loads((run / "original_annotation.json").read_text(encoding="utf-8-sig"))
    response = json.loads((run / "response_manifest.json").read_text(encoding="utf-8"))
    original = load_image(run / "original.png", "RGB")
    raw = load_image(run / "raw_output.png", "RGB")
    mask = load_image(run / "polygon_mask.png", "L").point(lambda value: 255 if value > 0 else 0)
    api_mask = load_image(run / "api_mask.png", "RGBA")
    overlay = load_image(run / "overlay_check.png", "RGB")
    if original.size != (640, 480) or mask.size != original.size or api_mask.size != original.size:
        raise ValueError("G1B original/mask dimensions are not the frozen 640x480")
    if [annotation["imageWidth"], annotation["imageHeight"]] != list(original.size):
        raise ValueError("Original JSON dimensions do not match original image")
    polygons = []
    for index, shape in enumerate(annotation.get("shapes", [])):
        points = [[float(x), float(y)] for x, y in shape["points"]]
        if shape.get("label") != "corrosion" or shape.get("shape_type") != "polygon" or not all(0 <= x <= original.width and 0 <= y <= original.height for x, y in points):
            raise ValueError(f"Invalid frozen polygon {index}")
        polygons.append({"index": index, "label": shape["label"], "points": points})
    if len(polygons) != 2 or count_nonzero(mask) != prepare["polygon_mask"]["target_pixels"]:
        raise ValueError("Mask does not reproduce both frozen corrosion polygons")
    transparent = api_mask.getchannel("A").point(lambda value: 255 if value == 0 else 0)
    if ImageChops.difference(transparent, mask).getbbox() is not None:
        raise ValueError("API mask transparent region differs from polygon mask")

    aspect_cross_left = original.width * raw.height
    aspect_cross_right = raw.width * original.height
    same_aspect = aspect_cross_left == aspect_cross_right
    if not same_aspect:
        raise ValueError("Raw API output and original do not have the same exact aspect ratio")
    aligned = raw.resize(original.size, Image.Resampling.LANCZOS)
    aligned_path = outputs_dir / "aligned_output_640x480.png"
    save_new(aligned, aligned_path)
    alignment_metadata = {
        "status": "RESIZED_FOR_ALIGNMENT_ONLY",
        "original_input_size_wh": list(original.size),
        "api_raw_output_size_wh": list(raw.size),
        "exact_same_aspect_ratio": same_aspect,
        "aspect_ratio": original.width / original.height,
        "scale_x": original.width / raw.width,
        "scale_y": original.height / raw.height,
        "interpolation": "Pillow Image.Resampling.LANCZOS",
        "aligned_output_path": str(aligned_path),
        "aligned_output_sha256": sha256(aligned_path),
        "warning": "Size equality after resizing does not prove geometric or pixel alignment.",
    }
    write_json_new(reports_dir / "alignment_metadata.json", alignment_metadata)

    preserve = ImageChops.invert(mask)
    geometry_valid = ImageChops.invert(dilate(mask, 10))
    raw_change = {
        "inside_mask": difference_metrics(original, aligned, mask),
        "outside_mask": difference_metrics(original, aligned, preserve),
    }
    edge_search = edge_translation_search(original, aligned, geometry_valid)
    patch_matches = local_patch_matches(original, aligned, geometry_valid)
    partially_supported = (
        patch_matches["reliable_for_bounded_translation_check"]
        and edge_search["zero_translation_dice"] >= 0.30
        and math.hypot(*(patch_matches["median_displacement_dx_dy"] or [99, 99])) <= 2.0
    )
    geometry_status = "GEOMETRIC_ALIGNMENT_PARTIALLY_SUPPORTED_NOT_PIXEL_EXACT" if partially_supported else "GEOMETRIC_ALIGNMENT_UNVERIFIED"
    geometry_report = {
        "status": geometry_status,
        "input_role": "FIRST_CALL_RAW",
        "comparison_role": "RESIZED_FOR_ALIGNMENT_ONLY",
        "no_geometric_warp_applied": True,
        "dependencies": {"Pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown", "NumPy": "NOT_INSTALLED", "OpenCV": "NOT_INSTALLED"},
        "raw_global_change": raw_change,
        "edge_translation_search": edge_search,
        "local_patch_matching": patch_matches,
        "decision_basis": "Bounded CPU evidence only; no non-rigid transform, homography, or corrective warp was applied.",
    }
    write_json_new(reports_dir / "geometry_alignment_report.json", geometry_report)

    hard = Image.composite(aligned, original, mask)
    hard_path = outputs_dir / "composite_hard.png"
    save_new(hard, hard_path)
    e1 = erode(mask, 1)
    e2 = erode(mask, 2)
    feather_alpha = Image.new("L", mask.size, 0)
    feather_alpha.paste(85, mask=mask)
    feather_alpha.paste(170, mask=e1)
    feather_alpha.paste(255, mask=e2)
    feather = Image.composite(aligned, original, feather_alpha)
    feather_path = outputs_dir / "composite_feather_inward_3px.png"
    save_new(feather, feather_path)
    hard_changes = changed_counts(hard, original, mask)
    feather_changes = changed_counts(feather, original, mask)
    if hard_changes["outside"]["changed_any_count"] != 0 or feather_changes["outside"]["changed_any_count"] != 0:
        raise RuntimeError("Outside-mask pixels changed; composite implementation invalid")
    feather_changed = ImageChops.difference(feather, original).convert("RGB")
    feather_changed_max = ImageChops.lighter(ImageChops.lighter(*feather_changed.split()[:2]), feather_changed.split()[2])
    changed_domain = feather_changed_max.point(lambda value: 255 if value > 0 else 0)
    if ImageChops.multiply(changed_domain, preserve).getbbox() is not None:
        raise RuntimeError("Feather changed domain extends outside polygon mask")
    weights = feather_alpha.histogram()
    composite_report = {
        "status": "POSTPROCESS_ONLY",
        "hard_composite": {"path": str(hard_path), "sha256": sha256(hard_path), "size_wh": list(hard.size), "changes": hard_changes, "outside_mask_changed_pixel_count": hard_changes["outside"]["changed_any_count"]},
        "inward_feather_composite": {
            "path": str(feather_path), "sha256": sha256(feather_path), "size_wh": list(feather.size),
            "feather_width_pixels": 3,
            "method": "Fixed morphological inward distance layers: boundary weight=85/255, second layer=170/255, depth>=3px=255/255; outside=0",
            "weight_pixel_counts": {str(index): count for index, count in enumerate(weights) if count},
            "changes": feather_changes,
            "outside_mask_changed_pixel_count": feather_changes["outside"]["changed_any_count"],
            "changed_domain_strictly_within_mask": True,
        },
        "scientific_limit": "Zero outside-mask changes are a postprocess compositing guarantee, not evidence that the API model preserved the background.",
    }
    write_json_new(reports_dir / "mask_composite_report.json", composite_report)

    bands = boundary_metrics(original, aligned, hard, feather, mask)
    boundary_report = {
        "status": "PENDING_HUMAN_REVIEW",
        "ring_definition": "Internal ring = mask minus erosion(width); two-sided band = dilation(width) minus erosion(width).",
        "ring_widths_pixels": [1, 3, 5, 10],
        "metrics": bands,
        "preliminary_visual_fields": {
            "obvious_seam": "PENDING_DIRECT_REVIEW",
            "texture_discontinuity": "PENDING_DIRECT_REVIEW",
            "blade_edge_misalignment": "PENDING_DIRECT_REVIEW",
            "color_match": "PENDING_DIRECT_REVIEW",
            "new_mechanical_structure_risk": "PENDING_DIRECT_REVIEW",
            "abnormal_smoothing": "PENDING_DIRECT_REVIEW",
            "residual_corrosion": "PENDING_DIRECT_REVIEW",
        },
        "metric_limit": "No single ring metric is a semantic or quality PASS criterion.",
    }
    write_json_new(reports_dir / "boundary_audit_report.json", boundary_report)

    save_new(make_review_sheet(original, overlay, raw, aligned, hard, feather), outputs_dir / "review_sheet_g1c.png")
    save_new(make_local_review(original, overlay, aligned, hard, feather, polygons), outputs_dir / "local_boundary_review.png")
    input_manifest = {
        "status": "PASS_FROZEN_INPUTS",
        "g1b_run": str(run),
        "source": "corrosion23",
        "original_json_relationship": {"imagePath": annotation.get("imagePath"), "imageWidth": annotation.get("imageWidth"), "imageHeight": annotation.get("imageHeight"), "polygon_count": len(polygons), "labels": sorted({item["label"] for item in polygons})},
        "files": {name: {"path": str(run / name), "size_bytes": (run / name).stat().st_size, "sha256": sha256(run / name)} for name in required},
        "raw_output_frozen_sha256_before": sha256(run / "raw_output.png"),
        "g1b_post_request_count": 1,
        "g1c_api_post_request_count": 0,
        "historical_inputs_modified": False,
    }
    write_json_new(inputs_dir / "input_manifest.json", input_manifest)
    final_report = reports_dir / "G1C_FINAL_REPORT.md"
    if final_report.exists():
        raise FileExistsError(final_report)
    final_report.write_text(
        "# Exp13-G1C corrosion23 local composite and geometry audit\n\n"
        f"Status: `{STATUS}`\n\n"
        "## Frozen inputs\n\n"
        f"- Original: 640x480, `{EXPECTED['original.png']}`.\n"
        f"- FIRST_CALL_RAW: 1448x1086, `{EXPECTED['raw_output.png']}`.\n"
        "- Both frozen corrosion polygons are represented by the 36,018-pixel mask.\n"
        "- G1C API POST requests: `0`.\n\n"
        "## Alignment\n\n"
        "The exact aspect ratio is 4:3 for both inputs. The API output was resized once to 640x480 with Pillow LANCZOS and labelled `RESIZED_FOR_ALIGNMENT_ONLY`. No geometric warp was applied.\n\n"
        f"Geometry status: `{geometry_status}`. Lightweight edge/patch evidence is recorded in `geometry_alignment_report.json`; resizing is not treated as proof of registration.\n\n"
        "## Composites\n\n"
        f"- Hard composite: `{sha256(hard_path)}`; outside-mask changed pixels: `{hard_changes['outside']['changed_any_count']}`.\n"
        f"- Fixed 3px inward feather: `{sha256(feather_path)}`; outside-mask changed pixels: `{feather_changes['outside']['changed_any_count']}`.\n"
        "- Both are `POSTPROCESS_ONLY`, not direct model outputs and not trusted training pairs.\n\n"
        "## Human review\n\n"
        "Boundary seam, geometry, texture continuity, residual corrosion, smoothing, and new-mechanical-structure risks remain `PENDING_DIRECT_REVIEW` until the generated review sheets are inspected.\n\n"
        "`TRAIN_PAIR_ELIGIBLE` is not granted. `RESTORATION_METHOD_PASS` is not granted.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": STATUS, "geometry_status": geometry_status, "hard_outside_changes": hard_changes["outside"]["changed_any_count"], "feather_outside_changes": feather_changes["outside"]["changed_any_count"], "raw_sha256_after": sha256(run / "raw_output.png")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
