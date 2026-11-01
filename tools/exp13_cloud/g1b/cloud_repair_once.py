#!/usr/bin/env python3
"""Create a single-image edit request manifest. Gate A implements dry-run only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


PROMPT_ID = "CLOUD_RESTORATION_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "live":
        raise RuntimeError("LIVE_NOT_IMPLEMENTED_IN_GATE_A; no network request was sent")
    if args.manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {args.manifest}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_url = os.getenv("OPENAI_BASE_URL") or config["base_url"]
    model = os.getenv("OPENAI_IMAGE_MODEL") or config["image_model"]
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    if urlparse(base_url).scheme != "https":
        raise ValueError("Base URL must use HTTPS")
    endpoint = base_url.rstrip("/") + "/v1/images/edits"
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt is empty")
    with Image.open(args.image) as image:
        image.load()
        image_info = {"format": image.format, "mode": image.mode, "size_wh": list(image.size)}
    with Image.open(args.mask) as mask:
        mask.load()
        mask_info = {"format": mask.format, "mode": mask.mode, "size_wh": list(mask.size), "alpha_extrema": list(mask.getchannel("A").getextrema())}
    if image_info["size_wh"] != mask_info["size_wh"] or mask_info["mode"] != "RGBA":
        raise ValueError("Image/mask size mismatch or mask lacks RGBA alpha")
    if mask_info["alpha_extrema"] != [0, 255]:
        raise ValueError("Mask must contain both transparent edit and opaque preserve pixels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [str(path) for path in args.output_dir.iterdir() if path.is_file()]
    request_structure = {
        "method": "POST",
        "endpoint": endpoint,
        "content_type": "multipart/form-data",
        "headers": {"Authorization": "Bearer <REDACTED_NOT_LOGGED>"},
        "fields": {
            "model": model,
            "prompt": f"<{PROMPT_ID}: {len(prompt)} characters; see prompt file>",
            "image": {"filename": args.image.name, "mime_type": "image/png", "sha256": sha256(args.image)},
            "mask": {"filename": args.mask.name, "mime_type": "image/png", "sha256": sha256(args.mask)},
            "n": 1,
        },
    }
    manifest = {
        "status": "DRY_RUN_PASS",
        "mode": "dry-run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "corrosion23",
        "prompt_id": PROMPT_ID,
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": sha256(args.prompt),
        "prompt_modified": False,
        "environment": {
            "openai_api_key_present": api_key_present,
            "openai_base_url": base_url,
            "openai_image_model": model,
            "api_key_value_logged": False,
            "environment_visibility_note": None if api_key_present else "OPENAI_API_KEY was not inherited by this Codex process; not required for dry-run",
        },
        "input": {"path": str(args.image.resolve()), "sha256": sha256(args.image), **image_info},
        "mask": {"path": str(args.mask.resolve()), "sha256": sha256(args.mask), **mask_info, "transparent_pixels_mean_edit": True},
        "request_structure_not_sent": request_structure,
        "output_directory": {"path": str(args.output_dir.resolve()), "existing_files_before_request": existing_outputs, "writable": True},
        "network": {"request_sent": False, "real_image_generation_requests": 0, "automatic_retries": 0},
        "gate_a_enforcement": "Live mode deliberately raises LIVE_NOT_IMPLEMENTED_IN_GATE_A",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DRY_RUN_PASS", "endpoint": endpoint, "model": model, "request_sent": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
