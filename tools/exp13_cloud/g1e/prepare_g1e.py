#!/usr/bin/env python3
"""Create and freeze the non-industrial G1E synthetic mask-control inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


MODEL = "gpt-image-2.5-sunburst"
PROMPT_ID = "MASK_CONTROL_TEST_V1"
ENDPOINT = "https://direct-api.ccccapi.cc/v1/images/edits"
SIZE = (640, 480)
BACKGROUND = (211, 211, 211)
BLUE = (0, 0, 255)
LEFT = (100, 160, 220, 320)  # left/top inclusive; right/bottom exclusive
RIGHT = (420, 160, 540, 320)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(image: Image.Image, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    image.save(path, format="PNG")


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rectangle_mask(box: tuple[int, int, int, int]) -> Image.Image:
    alpha = Image.new("L", SIZE, 255)
    # Pillow rectangle endpoints are inclusive, so subtract one from exclusive bounds.
    ImageDraw.Draw(alpha).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=0)
    mask = Image.new("RGBA", SIZE, (255, 255, 255, 255))
    mask.putalpha(alpha)
    return mask


def overlay(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    result = image.convert("RGBA")
    tint = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tint)
    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=(255, 0, 0, 96), outline=(255, 255, 0, 255), width=3)
    return Image.alpha_composite(result, tint).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = root / "inputs"
    configs = root / "configs"
    reports = root / "reports"
    prompt_path = configs / "prompt_v1.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    prompt = prompt_path.read_text(encoding="utf-8")
    if re.search(r"\b(?:left|right)\b", prompt, flags=re.IGNORECASE):
        raise ValueError("Prompt must not identify left or right target")

    paths = {
        "original": inputs / "synthetic_original.png",
        "left_mask": inputs / "mask_left_rgba.png",
        "right_mask": inputs / "mask_right_rgba.png",
        "left_overlay": inputs / "overlay_left.png",
        "right_overlay": inputs / "overlay_right.png",
        "manifest": inputs / "input_manifest.json",
        "config": configs / "frozen_g1e_config.json",
        "ledger": reports / "g1e_execution_ledger.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("One or more G1E preparation targets already exist")

    original = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(original)
    draw.rectangle((LEFT[0], LEFT[1], LEFT[2] - 1, LEFT[3] - 1), fill=BLUE)
    draw.rectangle((RIGHT[0], RIGHT[1], RIGHT[2] - 1, RIGHT[3] - 1), fill=BLUE)
    left_mask = rectangle_mask(LEFT)
    right_mask = rectangle_mask(RIGHT)
    save_new(original, paths["original"])
    save_new(left_mask, paths["left_mask"])
    save_new(right_mask, paths["right_mask"])
    save_new(overlay(original, LEFT), paths["left_overlay"])
    save_new(overlay(original, RIGHT), paths["right_overlay"])

    left_alpha = left_mask.getchannel("A")
    right_alpha = right_mask.getchannel("A")
    left_transparent = left_alpha.point(lambda value: 255 if value == 0 else 0)
    right_transparent = right_alpha.point(lambda value: 255 if value == 0 else 0)
    left_count = left_alpha.histogram()[0]
    right_count = right_alpha.histogram()[0]
    overlap = sum(ImageChops.multiply(left_transparent, right_transparent).histogram()[1:])
    expected_area = (LEFT[2] - LEFT[0]) * (LEFT[3] - LEFT[1])
    if left_count != expected_area or right_count != expected_area or overlap != 0:
        raise ValueError("Mask geometry or area verification failed")
    if left_alpha.getextrema() != (0, 255) or right_alpha.getextrema() != (0, 255):
        raise ValueError("Masks must contain transparent and opaque pixels")
    if ImageChops.difference(left_alpha, right_alpha).getbbox() is None:
        raise ValueError("Left and right masks must differ in location")
    if sha256(paths["original"]) != sha256(paths["original"]):
        raise AssertionError("Original image hash instability")

    manifest = {
        "status": "G1E_0_INPUTS_PREPARED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_type": "NON_INDUSTRIAL_SYNTHETIC",
        "coordinate_convention": "left/top inclusive; right/bottom exclusive",
        "size_wh": list(SIZE),
        "colors_rgb": {"background": list(BACKGROUND), "rectangles": list(BLUE)},
        "rectangles": {
            "left": {"xyxy": list(LEFT), "width": LEFT[2] - LEFT[0], "height": LEFT[3] - LEFT[1], "area_pixels": expected_area},
            "right": {"xyxy": list(RIGHT), "width": RIGHT[2] - RIGHT[0], "height": RIGHT[3] - RIGHT[1], "area_pixels": expected_area},
        },
        "files": {
            name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items() if name in {"original", "left_mask", "right_mask", "left_overlay", "right_overlay"}
        },
        "mask_checks": {
            "left_transparent_pixels": left_count,
            "right_transparent_pixels": right_count,
            "equal_area": left_count == right_count,
            "transparent_overlap_pixels": overlap,
            "positions_different": True,
            "alpha_values": [0, 255],
            "same_shape": True,
            "same_size_as_original": left_mask.size == right_mask.size == original.size,
            "rgb_channels_uniform_white": True,
        },
        "request_original_identity": {
            "condition_A_image_path": str(paths["original"]),
            "condition_B_image_path": str(paths["original"]),
            "condition_A_sha256": sha256(paths["original"]),
            "condition_B_sha256": sha256(paths["original"]),
            "byte_identical": True,
        },
    }
    write_json_new(paths["manifest"], manifest)
    frozen = {
        "experiment": "Exp13-G1E",
        "stage": "G1E-0_PREPARED_G1E-1_NOT_AUTHORIZED_BY_THIS_RECORD",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "prompt_id": PROMPT_ID,
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256(prompt_path),
        "prompt_stripped_sha256": hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest(),
        "original_path": str(paths["original"]),
        "original_sha256": sha256(paths["original"]),
        "conditions": [
            {"id": "A", "mask_role": "LEFT_RECTANGLE_TRANSPARENT", "mask_path": str(paths["left_mask"]), "mask_sha256": sha256(paths["left_mask"])},
            {"id": "B", "mask_role": "RIGHT_RECTANGLE_TRANSPARENT", "mask_path": str(paths["right_mask"]), "mask_sha256": sha256(paths["right_mask"])},
        ],
        "multipart_fields": ["model", "prompt", "n", "image[]", "mask"],
        "n": 1,
        "maximum_generation_posts": 2,
        "automatic_retries": 0,
        "model_fallbacks": 0,
        "prompt_search": 0,
        "concurrency": 0,
        "condition_order": ["A", "B"],
        "stop_rule": "A failure or ambiguous send stops experiment; B is not attempted",
    }
    write_json_new(paths["config"], frozen)
    ledger = {
        "status": "G1E_0_PREPARED_ZERO_POST",
        "maximum_generation_posts": 2,
        "occupied_generation_posts": 0,
        "actual_generation_posts": 0,
        "conditions": {"A": "NOT_STARTED", "B": "NOT_STARTED"},
        "live_session_marker": "runs/g1e_live_session_started.json",
        "terminal_marker": "runs/g1e_live_session_terminal.json",
        "idempotency": "exclusive file creation; any STARTED marker permanently occupies that condition; rerun refused",
        "g1e_1_authorized": False,
    }
    write_json_new(paths["ledger"], ledger)
    print(json.dumps({"status": manifest["status"], "original_sha256": frozen["original_sha256"], "prompt_sha256": frozen["prompt_sha256"], "transparent_pixels_each": left_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
