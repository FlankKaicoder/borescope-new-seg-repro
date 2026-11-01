#!/usr/bin/env python3
"""Write the future single-output audit schema without requiring an output image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    template = {
        "status": "TEMPLATE_ONLY_NO_GENERATED_IMAGE",
        "source": "corrosion23",
        "expected_generation_count": 1,
        "generated_image": {"path": None, "sha256": None, "decode_status": "PENDING", "size_match": "PENDING"},
        "request_provenance": {"endpoint": None, "model": None, "request_id": None, "prompt_id": "CLOUD_RESTORATION_V1", "input_sha256": None, "mask_sha256": None},
        "pixel_audit": {"inside_mask_change": "PENDING", "outside_mask_change": "PENDING", "outside_mask_change_fraction": None},
        "human_review": {
            "target_corrosion_removed": "PENDING",
            "geometry_preserved": "PENDING",
            "texture_plausible": "PENDING",
            "hallucinated_structures_absent": "PENDING",
            "outside_region_preserved": "PENDING",
            "final_quality": "PENDING",
            "comment": "",
        },
        "training_pair_eligibility": "NOT_EVALUATED_NOT_FROZEN",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": template["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
