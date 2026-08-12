#!/usr/bin/env python3
"""Inspect optional provenance-like JSON fields without modifying source data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    types: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    values: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    image_paths: Counter[str] = Counter()
    for path in sorted(args.data_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        value = payload.get("img_name_list", "<missing>")
        types[type(value).__name__] += 1
        if isinstance(value, (list, dict, str)):
            lengths[len(value)] += 1
        if value != "<missing>" and len(samples) < 20:
            samples.append({"json_path": path.relative_to(args.data_root).as_posix(), "value": value})
        if isinstance(value, list):
            values.update(str(item) for item in value)
        elif value != "<missing>":
            values[str(value)] += 1
        image_paths[str(payload.get("imagePath"))] += 1
    result = {
        "img_name_list_type_counts": dict(types),
        "img_name_list_length_counts": dict(lengths),
        "img_name_list_unique_values": len(values),
        "img_name_list_top_values": values.most_common(50),
        "samples": samples,
        "imagePath_unique_values": len(image_paths),
        "imagePath_top_values": image_paths.most_common(20),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
