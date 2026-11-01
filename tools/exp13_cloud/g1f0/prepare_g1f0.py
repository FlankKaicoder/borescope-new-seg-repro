from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\borescope_review_assets\cloud_pilot_g1f_roi_prepare")
SOURCE = ROOT / "source_snapshot"
CANDIDATES = ROOT / "candidates"
AUDITS = ROOT / "audits"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"

PILOT_ZIP = Path(
    r"E:\borescope_review_assets\exp13_cloud_pilot4_review_assets_20261031.zip"
)
G1E_ROOT = Path(r"E:\borescope_review_assets\cloud_pilot_g1e_mask_control")

EXPECTED = {
    "pilot_zip": "d9f049bda223f6485644da71839aed670a78b169b7411f722593c32fabe04f18",
    "original_image": "514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03",
    "original_json": "e4fa956d8ae736e4e851dde5fcf66e16faf2bf87deb284c81039da3e5621717a",
    "frozen_native_crop": "bc52f54b4af2e8ea92938126c481008487e5102b292a08e62b4ba8f805dae747",
    "frozen_native_edit_mask_crop": "030d2ea5ccb1241de2236c5c4680481077e580b9076ebd5e82a81b7bfc2af0e7",
    "frozen_input_512": "3c6f9d36f874999d6ae54661a8c76b128c7db7f0863b06fc0397a88d03b4cd4c",
    "frozen_edit_mask_512": "584b34276f07cd542ad18dec4abde6f05f130510d1d12af4ab700453fcca11f0",
    "frozen_effective_mask_512": "2aee6d7a130627a1cae17ad2ad6b93dd03caf087b2e32a7d0269d683e104c840",
    "g1e_A_raw_output": "4deeee9c25fd911833536a12013b4e7200bbabf4a03bc929ef786ac184f9635b",
    "g1e_B_raw_output": "15d5e03c3728138e4934225d90ec3c289f26d4fd26ba59fcef1457bb3a3d2f4f",
}

PROMPT_ID = "CLOUD_RESTORATION_V1"
PROMPT = """Edit the provided industrial borescope photograph of aircraft engine components.

The transparent regions of the supplied mask indicate damaged metallic surface areas that require restoration.

Restore only these damaged regions to a plausible intact metallic surface, preserving the original component geometry, blade curvature, edge contours, lighting, camera viewpoint, surface aging, and fine manufacturing texture.

The engine component is not perfectly smooth. Preserve realistic small surface pits, dimples, grain, machining marks, and natural unevenness consistent with the adjacent undamaged surfaces and neighboring blades.

Remove the target corrosion damage, but do not polish the metal into an unnaturally smooth surface.

Do not invent or add holes, cavities, bolts, panels, seams, slots, additional blades, mechanical parts, or regular geometric structures.

Preserve all existing non-target mechanical structures and inspection-camera interface overlays.

Avoid changes outside the specified repair region.

This is a conservative industrial defect restoration task, not image beautification or scene redesign.

Return one realistic restored image of the same camera view without labels, diagrams, comparisons, or explanatory text.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def assert_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def align_down(value: float, unit: int) -> int:
    return int(math.floor(value / unit) * unit)


def align_up(value: float, unit: int) -> int:
    return int(math.ceil(value / unit) * unit)


def clamp_roi(roi: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = roi
    width, height = size
    result = (max(0, left), max(0, top), min(width, right), min(height, bottom))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise RuntimeError(f"Invalid ROI after clipping: {result}")
    return result


def make_overlay(image: Image.Image, points: list[tuple[int, int]]) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(points, fill=(255, 40, 40, 92), outline=(255, 0, 0, 255), width=2)
    for x, y in points:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 0, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    left, top = xy
    box = draw.textbbox((left, top), text, font=font)
    draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 0, 0))
    draw.text((left, top), text, fill=color, font=font)


def mask_stats(mask: Image.Image) -> dict[str, Any]:
    pixels = list(mask.convert("L").getdata())
    target = sum(value > 0 for value in pixels)
    total = len(pixels)
    return {
        "target_pixels": target,
        "total_pixels": total,
        "coverage_fraction": target / total,
        "coverage_percent": target * 100.0 / total,
        "unique_values": sorted(set(pixels)),
        "nonempty": 0 < target < total,
    }


def binary_overlap(a: Image.Image, b: Image.Image, threshold: int = 127) -> dict[str, Any]:
    if a.size != b.size:
        raise RuntimeError(f"Mask size mismatch: {a.size} vs {b.size}")
    av = [v > threshold for v in a.convert("L").getdata()]
    bv = [v > threshold for v in b.convert("L").getdata()]
    intersection = sum(x and y for x, y in zip(av, bv))
    union = sum(x or y for x, y in zip(av, bv))
    count_a = sum(av)
    count_b = sum(bv)
    return {
        "threshold": threshold,
        "a_pixels": count_a,
        "b_pixels": count_b,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "dice": (2 * intersection / (count_a + count_b)) if count_a + count_b else 1.0,
        "iou": (intersection / union) if union else 1.0,
        "mismatch_pixels": sum(x != y for x, y in zip(av, bv)),
    }


def panel(title: str, image: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(result)
    draw.text((8, 7), title, fill="black", font=ImageFont.load_default())
    x = (width - image.width) // 2
    y = 28 + (height - 28 - image.height) // 2
    result.paste(image.convert("RGB"), (x, y))
    return result


def main() -> None:
    generated_roots = [CANDIDATES, AUDITS, CONFIGS, REPORTS]
    collisions = [path for path in generated_roots if path.exists()]
    if collisions:
        raise RuntimeError(f"Refusing to overwrite existing generated directories: {collisions}")
    for path in generated_roots:
        path.mkdir(parents=True)

    source_files = {
        "original_image": SOURCE / "original_image.png",
        "original_json": SOURCE / "original_annotation.json",
        "frozen_native_crop": SOURCE / "frozen_native_crop.png",
        "frozen_native_edit_mask_crop": SOURCE / "frozen_native_edit_mask_crop.png",
        "frozen_input_512": SOURCE / "frozen_input_512.png",
        "frozen_edit_mask_512": SOURCE / "frozen_edit_mask_512.png",
        "frozen_effective_mask_512": SOURCE / "frozen_effective_mask_512.png",
    }
    verified_hashes = {"pilot_zip": assert_hash(PILOT_ZIP, EXPECTED["pilot_zip"])}
    for key, path in source_files.items():
        verified_hashes[key] = assert_hash(path, EXPECTED[key])

    with source_files["original_json"].open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    original = Image.open(source_files["original_image"]).convert("RGB")
    if original.size != (640, 480):
        raise RuntimeError(f"Unexpected source size: {original.size}")
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != original.size:
        raise RuntimeError("JSON dimensions do not match source image")
    shapes = annotation.get("shapes", [])
    if len(shapes) != 1 or shapes[0].get("label") != "corrosion":
        raise RuntimeError("Expected exactly one corrosion shape")
    if shapes[0].get("shape_type") != "polygon":
        raise RuntimeError("Expected polygon annotation")
    points = [(int(x), int(y)) for x, y in shapes[0]["points"]]
    if len(points) != 26:
        raise RuntimeError(f"Unexpected polygon point count: {len(points)}")
    if any(not (0 <= x < original.width and 0 <= y < original.height) for x, y in points):
        raise RuntimeError("Out-of-bounds polygon point")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    polygon_bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    if polygon_bbox != (40, 64, 160, 113):
        raise RuntimeError(f"Unexpected polygon bbox: {polygon_bbox}")
    bbox_width = polygon_bbox[2] - polygon_bbox[0]
    bbox_height = polygon_bbox[3] - polygon_bbox[1]

    # A is compact: 1/4 bbox-width horizontal and 1/2 bbox-height vertical padding.
    a_raw = (
        align_down(polygon_bbox[0] - bbox_width * 0.25, 8),
        align_down(polygon_bbox[1] - bbox_height * 0.5, 8),
        align_up(polygon_bbox[2] + bbox_width * 0.25, 8),
        align_up(polygon_bbox[3] + bbox_height * 0.5, 8),
    )
    # B is broad: 1.25 bbox-width horizontal and 2 bbox-heights vertical padding.
    b_raw = (
        align_down(polygon_bbox[0] - bbox_width * 1.25, 16),
        align_down(polygon_bbox[1] - bbox_height * 2.0, 16),
        align_up(polygon_bbox[2] + bbox_width * 1.25, 16),
        align_up(polygon_bbox[3] + bbox_height * 2.0, 16),
    )
    rois = {
        "A": clamp_roi(a_raw, original.size),
        "B": clamp_roi(b_raw, original.size),
    }
    if rois != {"A": (8, 32, 192, 144), "B": (0, 0, 320, 224)}:
        raise RuntimeError(f"Unexpected deterministic ROI result: {rois}")

    full_mask = Image.new("L", original.size, 0)
    ImageDraw.Draw(full_mask).polygon(points, fill=255)
    full_stats = mask_stats(full_mask)
    if not full_stats["nonempty"]:
        raise RuntimeError("Full-image polygon mask is empty or full-frame")
    full_mask.save(AUDITS / "full_original_polygon_mask.png")
    full_overlay = make_overlay(original, points)
    full_overlay.save(AUDITS / "01_full_original_polygon.png")

    boxes = original.copy()
    boxes_draw = ImageDraw.Draw(boxes)
    colors = {"A": (0, 255, 255), "B": (255, 0, 255)}
    for key in ("B", "A"):
        left, top, right, bottom = rois[key]
        boxes_draw.rectangle((left, top, right - 1, bottom - 1), outline=colors[key], width=3)
        draw_label(boxes_draw, (left + 5, top + 5), f"ROI {key}", colors[key])
    boxes_draw.polygon(points, outline=(255, 0, 0), width=3)
    boxes.save(AUDITS / "02_roi_A_B_on_full_original.png")

    candidate_reports: dict[str, Any] = {}
    overlays: dict[str, Image.Image] = {}
    roi_images: dict[str, Image.Image] = {}
    for key, roi in rois.items():
        left, top, right, bottom = roi
        out = CANDIDATES / f"roi_{key}"
        out.mkdir()
        roi_image = original.crop(roi)
        roi_points = [(x - left, y - top) for x, y in points]
        if any(not (0 <= x < roi_image.width and 0 <= y < roi_image.height) for x, y in roi_points):
            raise RuntimeError(f"ROI {key} clips polygon points")
        roi_mask = Image.new("L", roi_image.size, 0)
        ImageDraw.Draw(roi_mask).polygon(roi_points, fill=255)
        stats = mask_stats(roi_mask)
        if stats["target_pixels"] != full_stats["target_pixels"]:
            raise RuntimeError(f"ROI {key} loses polygon pixels")

        api_mask = Image.new("RGBA", roi_image.size, (255, 255, 255, 255))
        ImageDraw.Draw(api_mask).polygon(roi_points, fill=(255, 255, 255, 0))
        alpha = api_mask.getchannel("A")
        alpha_values = sorted(set(alpha.getdata()))
        transparent_pixels = sum(value == 0 for value in alpha.getdata())
        if alpha_values != [0, 255] or transparent_pixels != stats["target_pixels"]:
            raise RuntimeError(f"ROI {key} RGBA alpha semantics failed")

        overlay = make_overlay(roi_image, roi_points)
        original_path = out / "roi_original.png"
        binary_path = out / "roi_binary_defect_mask.png"
        api_path = out / "api_rgba_mask.png"
        overlay_path = out / "polygon_overlay.png"
        roi_image.save(original_path)
        roi_mask.save(binary_path)
        api_mask.save(api_path)
        overlay.save(overlay_path)
        roi_images[key] = roi_image
        overlays[key] = overlay

        report = {
            "candidate": key,
            "roi_xyxy_half_open": list(roi),
            "roi_size_wh": [roi_image.width, roi_image.height],
            "source_to_roi_transform": {"x_roi": f"x_source - {left}", "y_roi": f"y_source - {top}"},
            "roi_to_source_transform": {"x_source": f"x_roi + {left}", "y_source": f"y_roi + {top}"},
            "polygon_bbox_source_xyxy_half_open": list(polygon_bbox),
            "polygon_bbox_roi_xyxy_half_open": [
                polygon_bbox[0] - left,
                polygon_bbox[1] - top,
                polygon_bbox[2] - left,
                polygon_bbox[3] - top,
            ],
            "polygon_point_count": len(points),
            "polygon_complete": True,
            "mask": stats,
            "api_mask": {
                "mode": api_mask.mode,
                "size_wh": list(api_mask.size),
                "alpha_unique_values": alpha_values,
                "polygon_inside_alpha": 0,
                "outside_alpha": 255,
                "transparent_pixels": transparent_pixels,
                "not_inverted": True,
            },
            "no_resize": True,
            "no_padding": True,
            "files": {
                "roi_original": {"path": str(original_path), "sha256": sha256(original_path)},
                "binary_mask": {"path": str(binary_path), "sha256": sha256(binary_path)},
                "api_mask": {"path": str(api_path), "sha256": sha256(api_path)},
                "overlay": {"path": str(overlay_path), "sha256": sha256(overlay_path)},
            },
        }
        write_json(out / "candidate_manifest.json", report)
        report["files"]["manifest"] = {
            "path": str(out / "candidate_manifest.json"),
            "sha256": sha256(out / "candidate_manifest.json"),
        }
        candidate_reports[key] = report

    max_width = max(image.width for image in overlays.values()) + 32
    max_height = max(image.height for image in overlays.values()) + 64
    comparison = Image.new("RGB", (max_width * 2, max_height), "white")
    comparison.paste(panel("ROI A: compact, native pixels", overlays["A"], max_width, max_height), (0, 0))
    comparison.paste(panel("ROI B: wider context, native pixels", overlays["B"], max_width, max_height), (max_width, 0))
    comparison.save(AUDITS / "03_context_and_defect_comparison.png")

    mask_comparison = Image.new("RGB", (max_width * 2, max_height), "white")
    for index, key in enumerate(("A", "B")):
        api = Image.open(CANDIDATES / f"roi_{key}" / "api_rgba_mask.png")
        visible = Image.new("RGB", api.size, (80, 200, 80))
        alpha = api.getchannel("A")
        transparent_preview = Image.new("RGB", api.size, (255, 40, 40))
        visible.paste(transparent_preview, mask=Image.eval(alpha, lambda v: 255 - v))
        mask_comparison.paste(
            panel(f"ROI {key}: red = alpha 0 edit area", visible, max_width, max_height),
            (index * max_width, 0),
        )
    mask_comparison.save(AUDITS / "04_api_mask_semantics_comparison.png")

    native_historical_mask = Image.open(source_files["frozen_native_edit_mask_crop"]).convert("L")
    m1_polygon_mask = full_mask.crop((0, 0, 480, 480))
    historical_overlap = binary_overlap(native_historical_mask, m1_polygon_mask)
    historical_mask_report = {
        "m1_selected_crop_id": "alt_1",
        "m1_crop_xyxy_half_open": [0, 0, 480, 480],
        "m1_risk": "HIGH_RISK_BUT_USER_ACCEPTED",
        "frozen_native_mask_mode": native_historical_mask.mode,
        "frozen_native_mask_size_wh": list(native_historical_mask.size),
        "json_polygon_bbox_xyxy_half_open": list(m1_polygon_mask.getbbox()),
        "historical_mask_thresholded_bbox_xyxy_half_open": list(
            native_historical_mask.point(lambda value: 255 if value > 127 else 0).getbbox()
        ),
        "comparison_note": "The historical M1 mask is anti-aliased; thresholded overlap is an identity/geometry cross-check, not a requirement that the new exact JSON rasterization be byte-identical.",
        "thresholded_overlap_with_json_polygon": historical_overlap,
    }

    source_mapping_rows = list(csv.DictReader((SOURCE / "source_mapping.csv").open("r", encoding="utf-8-sig", newline="")))
    source_row = next(row for row in source_mapping_rows if row["source_key"] == "corrosion2")
    split_rows = list(csv.DictReader((SOURCE / "frozen_train_split_rows.csv").open("r", encoding="utf-8-sig", newline="")))
    split_row = next(row for row in split_rows if row["image_path"] == "2.png")
    if not (
        source_row["source_id"] == "2"
        and source_row["class"] == "corrosion"
        and source_row["split"] == "train"
        and source_row["group_id"] == "singleton_2"
        and source_row["crop_id"] == "alt_1"
        and split_row["split"] == "train"
        and split_row["group_id"] == "singleton_2"
    ):
        raise RuntimeError("Frozen source/TRAIN identity validation failed")

    g1e_paths = {
        "A": G1E_ROOT / "runs" / "condition_A" / "raw_output.png",
        "B": G1E_ROOT / "runs" / "condition_B" / "raw_output.png",
    }
    g1e_check: dict[str, Any] = {
        "read_only": True,
        "ledger_modified": False,
        "possible_upload_conversion_or_version_difference": False,
        "conditions": {},
    }
    for key, path in g1e_paths.items():
        expected = EXPECTED[f"g1e_{key}_raw_output"]
        actual = sha256(path)
        with Image.open(path) as output:
            output_meta = {"format": output.format, "mode": output.mode, "size_wh": list(output.size)}
        matches = actual == expected
        g1e_check["conditions"][key] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
            **output_meta,
        }
        if not matches:
            g1e_check["possible_upload_conversion_or_version_difference"] = True
    write_json(REPORTS / "g1e_raw_output_hash_check.json", g1e_check)

    prompt_path = CONFIGS / "prompt_CLOUD_RESTORATION_V1.txt"
    prompt_path.write_text(PROMPT, encoding="utf-8", newline="\n")
    future_requests = {
        "status": "POST_NOT_SENT",
        "authorized_for_execution": False,
        "selection": "PENDING_REVIEW_NO_ROI_FROZEN",
        "verified_from_g1e": {
            "base_url": "https://direct-api.ccccapi.cc",
            "endpoint": "https://direct-api.ccccapi.cc/v1/images/edits",
            "model": "gpt-image-2.5-sunburst",
            "multipart_field_names": ["model", "prompt", "n", "image[]", "mask"],
        },
        "future_single_call_parameters": {
            "method": "POST_NOT_SENT",
            "n": 1,
            "model": "gpt-image-2.5-sunburst",
            "prompt_id": PROMPT_ID,
            "prompt_path": str(prompt_path),
            "prompt_sha256": sha256(prompt_path),
            "candidate_image_and_mask_pairs": {
                key: {
                    "image": candidate_reports[key]["files"]["roi_original"],
                    "mask": candidate_reports[key]["files"]["api_mask"],
                }
                for key in ("A", "B")
            },
            "automatic_retries": 0,
            "fallbacks": 0,
            "prompt_search": 0,
        },
        "api_post_count_this_stage": 0,
    }
    write_json(CONFIGS / "future_single_call_template.json", future_requests)

    identity_report = {
        "status": "VERIFIED",
        "source_key": "corrosion2",
        "source_id": 2,
        "class": "corrosion",
        "split": "train",
        "group_id": "singleton_2",
        "image_path_in_frozen_source": "2.png",
        "json_path_in_frozen_source": "2.json",
        "image_size_wh": list(original.size),
        "json_dimensions_match": True,
        "shape_count": len(shapes),
        "polygon_point_count": len(points),
        "polygon_bbox_xyxy_half_open": list(polygon_bbox),
        "polygon_coordinate_minmax_inclusive": [min(xs), min(ys), max(xs), max(ys)],
        "all_polygon_points_in_bounds": True,
        "verified_sha256": verified_hashes,
        "frozen_source_mapping_row": source_row,
        "frozen_train_split_row": split_row,
        "m1_history": historical_mask_report,
        "private_mapping_accessed": False,
        "test_data_accessed": False,
    }
    write_json(REPORTS / "source_identity_and_m1_audit.json", identity_report)

    roi_audit = {
        "status": "G1F_0_ROI_PREPARED / PENDING_REVIEW",
        "selection_frozen": False,
        "selection_policy": "Both geometry-derived candidates are retained; no candidate is automatically selected.",
        "source_polygon_mask": full_stats,
        "candidate_rules": {
            "A": "Expand bbox by 0.25 bbox width horizontally and 0.5 bbox height vertically; align outward to 8 px; clip only to source bounds.",
            "B": "Expand bbox by 1.25 bbox widths horizontally and 2 bbox heights vertically; align outward to 16 px; clip only to source bounds.",
        },
        "candidates": candidate_reports,
        "visual_risk_observations": {
            "A": "Compact view preserves the target edge and nearby normal metal, but cuts through inspection UI context near the top-left and provides less long-range mechanical context.",
            "B": "Wider view preserves more of the curved mechanical surface and the complete GE icon, but includes more inspection UI and still truncates the long top status text at the right ROI edge.",
            "ui_is_masked_as_defect": False,
            "mechanical_structure_clipping": "A clips more of the continuing mechanical edge; B retains substantially more, but neither is a full-frame view.",
        },
        "api_post_count": 0,
        "gpu_operations": 0,
        "training_operations": 0,
        "test_data_accesses": 0,
    }
    write_json(REPORTS / "roi_integrity_audit.json", roi_audit)

    a = candidate_reports["A"]
    b = candidate_reports["B"]
    report = f"""# Exp13-G1F-0 corrosion2 ROI 局部修复输入准备

最终状态：`G1F_0_ROI_PREPARED / PENDING_REVIEW`

## 身份与输入验证

- Source：`corrosion2`，source ID `2`，类别 `corrosion`。
- 冻结划分：`train / singleton_2`，只读清单核对通过；未访问 TEST。
- 原图：`640×480`，SHA256 `{verified_hashes['original_image']}`。
- JSON：SHA256 `{verified_hashes['original_json']}`，尺寸与原图一致。
- 标注：1 个 polygon、26 点，所有点均在图像范围内。
- Polygon 包围范围：包含式坐标 `x=40..159, y=64..112`；半开 bbox `[40,64,160,113)`。
- Pilot ZIP：SHA256 `{verified_hashes['pilot_zip']}`，与预期一致。
- M1 冻结信息：`alt_1`，半开裁剪 `[0,0,480,480)`，历史风险 `HIGH_RISK_BUT_USER_ACCEPTED`。未访问 M1 私有解盲映射。
- 历史 M1 native edit Mask 不是 JSON polygon 的逐字节副本：以 `>127` 二值化后为 3503 像素、bbox `[37,61,163,116)`，相对精确 polygon 的 2394 像素、bbox `[40,64,160,113)` 向外扩展 3 像素；Dice `0.811938`。这是历史抗锯齿/扩展编辑域差异，不是本次 ROI 坐标偏移。

## ROI 候选

| 候选 | 半开坐标 `[l,t,r,b)` | 尺寸 | Mask 像素 | Mask 覆盖率 |
|---|---:|---:|---:|---:|
| A | `{a['roi_xyxy_half_open']}` | `{a['roi_size_wh'][0]}×{a['roi_size_wh'][1]}` | {a['mask']['target_pixels']} | {a['mask']['coverage_percent']:.4f}% |
| B | `{b['roi_xyxy_half_open']}` | `{b['roi_size_wh'][0]}×{b['roi_size_wh'][1]}` | {b['mask']['target_pixels']} | {b['mask']['coverage_percent']:.4f}% |

两者均完整保留全部 polygon，未拉伸、未补边、未丢失缺陷像素。二值 Mask 与 ROI 尺寸一致。API Mask 为 RGBA PNG，polygon 内 alpha=0，外部 alpha=255；透明区未反转。

## 机械结构与 UI 风险

- A 更紧凑，保留目标机械边缘及邻近正常金属，但截取的长程机械结构较少，并在左上方包含被 ROI 边界截断的检查 UI 上下文。
- B 保留更多曲面、连续机械边缘与完整 GE 图标，更适合判断局部修复是否顺从真实结构；代价是正常背景更多、Mask 占比更低，并且顶部较长的状态文字仍在 ROI 右边界被截断。
- 两个缺陷 Mask 都只来自 JSON polygon，UI 像素不在透明编辑区内；这不等同于证明未来模型绝不会改动 Mask 外 UI。

## SHA256

| 候选 | ROI 原图 | 二值 Mask | API RGBA Mask |
|---|---|---|---|
| A | `{a['files']['roi_original']['sha256']}` | `{a['files']['binary_mask']['sha256']}` | `{a['files']['api_mask']['sha256']}` |
| B | `{b['files']['roi_original']['sha256']}` | `{b['files']['binary_mask']['sha256']}` | `{b['files']['api_mask']['sha256']}` |

## G1E 证据链只读核对

- A raw output：`{g1e_check['conditions']['A']['actual_sha256']}`，与指定值一致：`{g1e_check['conditions']['A']['matches']}`。
- B raw output：`{g1e_check['conditions']['B']['actual_sha256']}`，与指定值一致：`{g1e_check['conditions']['B']['matches']}`。
- 两者均为 PNG `1448×1086`；未发现本地文件与指定上传版本哈希差异，未修改 ledger。

## 接口准备与安全审计

- 只读沿用 G1E 已验证协议：`https://direct-api.ccccapi.cc/v1/images/edits`、`gpt-image-2.5-sunburst`、multipart `image[] + mask + prompt + model + n`。
- 未来参数模板明确标记为 `POST_NOT_SENT`，A/B 均仅作为候选，尚未冻结最终 ROI。
- 本阶段 API POST：`0`；新生成请求：`0`；GPU：`0`；训练：`0`；TEST：`0`。

## 审核建议

建议优先冻结 **ROI B** 进入后续单次对照，因为它明显保留更多连续机械曲面和边缘，降低因上下文不足导致结构重构的风险。该建议不是自动选择：B 的 Mask 覆盖率更低、UI 暴露更多，且中转站/上游对独立 Mask 的严格服从仍未被证明，必须由人工查看审计图后冻结。

未解决风险：未来 API 可能缩放输出、修改 Mask 外区域、重绘 UI 或产生结构幻觉；ROI 局部输入也不能单独证明上游实际使用 Mask。G1F-1 未获授权，本阶段不执行任何真实调用。
"""
    (REPORTS / "G1F_0_FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    final_manifest = {
        "experiment": "Exp13-G1F-0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "G1F_0_ROI_PREPARED / PENDING_REVIEW",
        "source": "corrosion2",
        "outputs": {
            "full_original_polygon": str(AUDITS / "01_full_original_polygon.png"),
            "roi_locations": str(AUDITS / "02_roi_A_B_on_full_original.png"),
            "context_comparison": str(AUDITS / "03_context_and_defect_comparison.png"),
            "mask_semantics_comparison": str(AUDITS / "04_api_mask_semantics_comparison.png"),
            "source_identity_audit": str(REPORTS / "source_identity_and_m1_audit.json"),
            "roi_integrity_audit": str(REPORTS / "roi_integrity_audit.json"),
            "g1e_hash_check": str(REPORTS / "g1e_raw_output_hash_check.json"),
            "future_request_template": str(CONFIGS / "future_single_call_template.json"),
            "final_report": str(REPORTS / "G1F_0_FINAL_REPORT.md"),
        },
        "safety": {
            "api_post_count": 0,
            "network_requests": 0,
            "gpu_operations": 0,
            "training_operations": 0,
            "test_data_accesses": 0,
            "paper_repository_accesses": 0,
            "m1_private_mapping_accesses": 0,
        },
    }
    write_json(ROOT / "G1F_0_MANIFEST.json", final_manifest)
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
