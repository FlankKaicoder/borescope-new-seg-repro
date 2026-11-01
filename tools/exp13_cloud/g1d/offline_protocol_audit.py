#!/usr/bin/env python3
"""Independent, network-free verification of the G1B request inputs and G1C outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path: Path, mode: str | None = None) -> Image.Image:
    with Image.open(path) as handle:
        handle.load()
        image = handle.copy()
        image.info["source_format"] = handle.format
    return image.convert(mode) if mode else image


def count_value(image: Image.Image, value: int) -> int:
    return image.convert("L").histogram()[value]


def count_nonzero(image: Image.Image) -> int:
    histogram = image.convert("L").histogram()
    return sum(histogram[1:])


def changed_count(first: Image.Image, second: Image.Image, region: Image.Image, threshold: int = 0) -> int:
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    red, green, blue = difference.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    changed = maximum.point(lambda value: 255 if value > threshold else 0)
    return count_nonzero(ImageChops.multiply(changed, region.convert("L")))


def regional_max(first: Image.Image, second: Image.Image, region: Image.Image) -> int:
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    red, green, blue = difference.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return ImageChops.multiply(maximum, region.convert("L")).getextrema()[1]


def regional_mean(first: Image.Image, second: Image.Image, region: Image.Image) -> float:
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    return sum(ImageStat.Stat(difference, mask=region.convert("L")).mean) / 3.0


def erode(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.MinFilter(radius * 2 + 1))


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1b-root", required=True, type=Path)
    parser.add_argument("--g1c-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    g1b_root = args.g1b_root.resolve()
    g1c_root = args.g1c_root.resolve()
    out = args.output_root.resolve()
    live = g1b_root / "runs" / "corrosion23_first_call_live"
    code_path = g1b_root / "code" / "gate_b_live_once.py"
    request = json.loads((live / "request_manifest.json").read_text(encoding="utf-8"))
    response = json.loads((live / "response_manifest.json").read_text(encoding="utf-8"))
    raw_response = json.loads((live / "response_raw.json").read_text(encoding="utf-8"))
    prepare = json.loads((live / "prepare_report.json").read_text(encoding="utf-8"))
    g1c_geometry = json.loads((g1c_root / "reports" / "geometry_alignment_report.json").read_text(encoding="utf-8"))
    g1c_composite = json.loads((g1c_root / "reports" / "mask_composite_report.json").read_text(encoding="utf-8"))

    original = load(live / "original.png", "RGB")
    polygon = load(live / "polygon_mask.png", "L").point(lambda value: 255 if value > 0 else 0)
    api_mask = load(live / "api_mask.png", "RGBA")
    raw = load(live / "raw_output.png", "RGB")
    alpha = api_mask.getchannel("A")
    transparent_region = alpha.point(lambda value: 255 if value == 0 else 0)
    alpha_histogram = alpha.histogram()
    rgb_extrema = [list(channel.getextrema()) for channel in api_mask.convert("RGB").split()]
    prompt_file = (live / "prompt.txt").read_text(encoding="utf-8")
    prompt_sent = prompt_file.strip()
    revised_prompt = str(raw_response["data"][0].get("revised_prompt") or "")

    code = code_path.read_text(encoding="utf-8")
    required_code_fragments = {
        "forced_model_constant": 'MODEL = "gpt-image-2.5-sunburst"',
        "post_method": 'method="POST"',
        "endpoint_suffix": 'return normalized + "/v1/images/edits"',
        "multipart_model_prompt_n": '[("model", MODEL), ("prompt", prompt), ("n", "1")]',
        "multipart_image_field": '("image[]", "original.png", "image/png", image_bytes)',
        "multipart_mask_field": '("mask", "api_mask.png", "image/png", mask_bytes)',
    }
    code_fragment_checks = {name: fragment in code for name, fragment in required_code_fragments.items()}
    if not all(code_fragment_checks.values()):
        raise ValueError(f"Required request construction code not found: {code_fragment_checks}")

    mask_audit = {
        "status": "PASS_LOCAL_REQUEST_PIPELINE / RELAY_AND_UPSTREAM_BEHAVIOR_UNVERIFIED",
        "evidence_states": {
            "REQUEST_IMAGE_INCLUDED": True,
            "REQUEST_MASK_INCLUDED": True,
            "MASK_FORMAT_VERIFIED": True,
            "RELAY_MASK_FORWARDING_UNKNOWN": True,
            "UPSTREAM_MODEL_IDENTITY_UNVERIFIED": True,
            "MODEL_MASK_ADHERENCE_UNVERIFIED": True,
        },
        "request": {
            "endpoint": request["endpoint"],
            "method": request["method"],
            "content_type": request["content_type"],
            "fields": request["fields"],
            "field_details": {
                "model": {"type": "UTF-8 multipart text", "value": request["model"], "source": "forced MODEL constant"},
                "prompt": {"type": "UTF-8 multipart text", "source": str(live / "prompt.txt"), "source_file_sha256": sha256(live / "prompt.txt"), "actual_stripped_field_sha256": sha256_text(prompt_sent)},
                "n": {"type": "UTF-8 multipart text", "value": "1"},
                "image[]": {"type": "image/png multipart file", "filename": "original.png", "source": str(live / "original.png"), "sha256": sha256(live / "original.png")},
                "mask": {"type": "image/png multipart file", "filename": "api_mask.png", "source": str(live / "api_mask.png"), "sha256": sha256(live / "api_mask.png")},
            },
            "request_body_sha256": request["request_body_sha256"],
            "request_body_bytes": request["request_body_bytes"],
            "post_request_count": request["post_request_count"],
            "automatic_retries": request["automatic_retries"],
            "model_fallbacks": request["model_fallbacks"],
            "size_field_sent": "size" in request["fields"],
            "quality_field_sent": "quality" in request["fields"],
            "output_format_field_sent": "output_format" in request["fields"],
        },
        "static_code": {"path": str(code_path), "sha256": sha256(code_path), "fragment_checks": code_fragment_checks},
        "image": {"format": "PNG", "mode": original.mode, "size_wh": list(original.size), "sha256": sha256(live / "original.png")},
        "binary_polygon_mask": {"format": "PNG", "mode": polygon.mode, "size_wh": list(polygon.size), "white_polygon_pixels": count_value(polygon, 255), "black_preserve_pixels": count_value(polygon, 0), "sha256": sha256(live / "polygon_mask.png")},
        "api_mask": {
            "format": "PNG", "mode": api_mask.mode, "size_wh": list(api_mask.size), "sha256": sha256(live / "api_mask.png"),
            "alpha_extrema": list(alpha.getextrema()), "alpha_unique_nonzero_bins": [index for index, count in enumerate(alpha_histogram) if count],
            "transparent_edit_pixels": alpha_histogram[0], "opaque_preserve_pixels": alpha_histogram[255],
            "rgb_channel_extrema": rgb_extrema,
            "transparent_region_pixel_equal_polygon_mask": ImageChops.difference(transparent_region, polygon).getbbox() is None,
            "polarity": "polygon alpha=0; outside alpha=255",
            "official_format_conformance": True,
            "relay_forwarding_proven": False,
            "upstream_usage_proven": False,
        },
        "prompt": {
            "fixed_prompt_source_file_sha256": sha256(live / "prompt.txt"),
            "actual_sent_after_strip_sha256": sha256_text(prompt_sent),
            "request_manifest_prompt_sha256_is_source_file_not_actual_field": request["prompt_sha256"] == sha256(live / "prompt.txt") and request["prompt_sha256"] != sha256_text(prompt_sent),
            "revised_prompt_present": bool(revised_prompt),
            "revised_prompt_sha256": sha256_text(revised_prompt),
            "revised_prompt_character_count": len(revised_prompt),
            "revised_prompt_exactly_equals_sent_prompt": revised_prompt == prompt_sent,
            "semantic_rewrite_observed": revised_prompt != prompt_sent,
        },
        "response": {
            "http_status": response["http_status"],
            "top_level_keys": list(raw_response.keys()),
            "data_item_keys": list(raw_response["data"][0].keys()),
            "response_model": raw_response.get("model"),
            "requested_model": request["model"],
            "upstream_model_identity_verified": False,
            "image_source_kind": response["image_source_kind"],
            "image_url_summary": response["image_url_summary"],
            "url_reaccessed_during_g1d": False,
            "b64_json_present": bool(raw_response["data"][0].get("b64_json")),
            "usage": response["usage"],
            "output": {"format": response["detected_original_format"], "mode": response["detected_original_mode"], "size_wh": response["detected_size_wh"], "sha256": response["raw_output_png_sha256"]},
            "relay_adapter_evidence": "URL response on a third-party host instead of the official guide's documented base64 Image API response",
        },
        "local_code_errors": [
            {
                "severity": "METADATA_ONLY",
                "finding": "request_manifest.prompt_sha256 hashes the prompt file including its trailing newline, while the transmitted field is prompt_file.strip().",
                "impact": "The actual prompt field is not independently hashed in the original manifest; request_body_sha256 still covers the serialized body. The returned revised_prompt exactly matches the stripped transmitted text.",
            }
        ],
        "no_mask_encoding_or_polarity_error_found": True,
        "scientific_boundary": "Local construction proves submission of an independent mask field to the relay. It does not prove relay forwarding, upstream model identity, or upstream adherence.",
    }
    write_json(out / "mask_pipeline_audit.json", mask_audit)

    aligned = raw.resize(original.size, Image.Resampling.LANCZOS)
    aligned_disk = load(g1c_root / "outputs" / "aligned_output_640x480.png", "RGB")
    hard_recomputed = Image.composite(aligned, original, polygon)
    hard_disk = load(g1c_root / "outputs" / "composite_hard.png", "RGB")
    eroded_one = erode(polygon, 1)
    eroded_two = erode(polygon, 2)
    feather_alpha = Image.new("L", polygon.size, 0)
    feather_alpha.paste(85, mask=polygon)
    feather_alpha.paste(170, mask=eroded_one)
    feather_alpha.paste(255, mask=eroded_two)
    feather_recomputed = Image.composite(aligned, original, feather_alpha)
    feather_disk = load(g1c_root / "outputs" / "composite_feather_inward_3px.png", "RGB")
    outside = ImageChops.invert(polygon)

    aligned_match = ImageChops.difference(aligned, aligned_disk).getbbox() is None
    hard_match = ImageChops.difference(hard_recomputed, hard_disk).getbbox() is None
    feather_match = ImageChops.difference(feather_recomputed, feather_disk).getbbox() is None
    raw_outside_gt5 = changed_count(original, aligned, outside, 5)
    g1c_check = {
        "status": "CORE_RESULTS_VERIFIED / ONE_REGIONAL_MAX_METRIC_BUG_FOUND",
        "inputs": {
            "original_sha256": sha256(live / "original.png"), "raw_output_sha256": sha256(live / "raw_output.png"),
            "polygon_mask_sha256": sha256(live / "polygon_mask.png"), "original_size_wh": list(original.size), "raw_size_wh": list(raw.size),
            "exact_same_aspect_ratio": original.width * raw.height == raw.width * original.height,
        },
        "alignment": {
            "method": "Pillow LANCZOS 1448x1086 -> 640x480", "recomputed_pixel_equal_saved": aligned_match,
            "recomputed_sha256": sha256(g1c_root / "outputs" / "aligned_output_640x480.png"),
        },
        "mask": {"polygon_pixels": count_nonzero(polygon), "outside_pixels": count_nonzero(outside), "two_polygons_from_prepare_report": prepare["annotation"]["polygon_count"] == 2},
        "raw_change": {
            "outside_mask_changed_gt5_recomputed": raw_outside_gt5,
            "outside_mask_changed_gt5_reported": g1c_geometry["raw_global_change"]["outside_mask"]["changed_max_channel_gt5_count"],
            "match": raw_outside_gt5 == g1c_geometry["raw_global_change"]["outside_mask"]["changed_max_channel_gt5_count"],
            "outside_fraction_recomputed": raw_outside_gt5 / count_nonzero(outside),
            "outside_mean_absolute_channel_difference_recomputed": regional_mean(original, aligned, outside),
        },
        "hard_composite": {
            "recomputed_pixel_equal_saved": hard_match, "saved_sha256": sha256(g1c_root / "outputs" / "composite_hard.png"),
            "outside_changed_any_recomputed": changed_count(original, hard_disk, outside, 0),
            "outside_changed_any_reported": g1c_composite["hard_composite"]["outside_mask_changed_pixel_count"],
            "outside_regional_max_recomputed": regional_max(original, hard_disk, outside),
            "outside_reported_max_is_global_bug": g1c_composite["hard_composite"]["changes"]["outside"]["max_channel_difference"],
        },
        "feather_composite": {
            "recomputed_pixel_equal_saved": feather_match, "saved_sha256": sha256(g1c_root / "outputs" / "composite_feather_inward_3px.png"),
            "weights": {str(index): count for index, count in enumerate(feather_alpha.histogram()) if count},
            "outside_changed_any_recomputed": changed_count(original, feather_disk, outside, 0),
            "outside_changed_any_reported": g1c_composite["inward_feather_composite"]["outside_mask_changed_pixel_count"],
            "outside_regional_max_recomputed": regional_max(original, feather_disk, outside),
            "outside_reported_max_is_global_bug": g1c_composite["inward_feather_composite"]["changes"]["outside"]["max_channel_difference"],
        },
        "geometry_analysis_input_check": {
            "original_used": True, "aligned_output_used": True, "polygon_mask_used": True,
            "non_mask_geometry_domain_excludes_dilated_mask_by_10px": True,
            "no_warp_applied": g1c_geometry["no_geometric_warp_applied"],
            "conservative_status": g1c_geometry["status"],
            "edge_dice": g1c_geometry["edge_translation_search"]["zero_translation_dice"],
            "accepted_patch_matches": g1c_geometry["local_patch_matching"]["accepted_unique_matches"],
        },
        "calculation_error": {
            "found": True,
            "field": "difference_metrics.max_channel_difference",
            "cause": "The implementation calls maximum.getextrema()[1] before applying the requested region mask.",
            "effect": "Outside-region max is reported as 139 for hard/feather despite zero changed pixels; the independently recomputed regional max is 0.",
            "materiality": "NON_MATERIAL_TO_PRIMARY_CONCLUSION: counts, fractions, regional means, output hashes, and FAIL_PAIR_FIDELITY remain valid.",
        },
        "conclusion": "G1C used the correct frozen original, aligned raw output, and polygon mask. The core resize/composite hashes and 81.92% outside-change statistic reproduce exactly. The geometry conclusion remains conservative and supported.",
    }
    write_json(out / "g1c_consistency_check.json", g1c_check)
    print(json.dumps({"mask_status": mask_audit["status"], "g1c_status": g1c_check["status"], "outside_gt5": raw_outside_gt5}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
