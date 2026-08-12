#!/usr/bin/env python3
"""Render a CSV image list into contact sheets for audit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Image audit")
    args = parser.parse_args()
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        paths = [row[args.path_column] for row in csv.DictReader(handle)]
    columns, thumb_w, thumb_h, label_h, margin = 5, 200, 150, 24, 12
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new("RGB", (2 * margin + columns * thumb_w, 48 + rows * (thumb_h + label_h) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), f"{args.title} | {len(paths)} images", fill="black", font=font)
    for index, relative in enumerate(paths):
        col, row = index % columns, index // columns
        x, y = margin + col * thumb_w, 48 + row * (thumb_h + label_h)
        with Image.open(args.data_root / relative) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w - 6, thumb_h - 6), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (thumb_w - image.width) // 2, y + (thumb_h - image.height) // 2))
        draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline="#777777")
        draw.text((x + 3, y + thumb_h + 3), relative, fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(f"images={len(paths)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
