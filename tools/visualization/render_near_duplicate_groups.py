#!/usr/bin/env python3
"""Render near-duplicate group contact sheets for manual review."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--groups-per-page", type=int, default=12)
    args = parser.parse_args()
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.groups_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["group_id"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    font = ImageFont.load_default()
    thumb_w, thumb_h = 180, 135
    header_h, label_h, margin = 28, 24, 12
    rows_per_group = []
    for group_id, members in ordered:
        rows_per_group.append((group_id, members, math.ceil(len(members) / 5)))
    pages: list[list[tuple[str, list[dict[str, str]], int]]] = []
    page: list[tuple[str, list[dict[str, str]], int]] = []
    page_rows = 0
    for item in rows_per_group:
        needed = item[2]
        if page and (len(page) >= args.groups_per_page or page_rows + needed > 10):
            pages.append(page)
            page, page_rows = [], 0
        page.append(item)
        page_rows += needed
    if page:
        pages.append(page)
    for page_index, page_groups in enumerate(pages, 1):
        height = margin + sum(header_h + row_count * (thumb_h + label_h) + margin for _, _, row_count in page_groups)
        canvas = Image.new("RGB", (margin * 2 + 5 * thumb_w, height), "white")
        draw = ImageDraw.Draw(canvas)
        y = margin
        for group_id, members, row_count in page_groups:
            draw.text((margin, y), f"{group_id} | {len(members)} images", fill="black", font=font)
            y += header_h
            for member_index, member in enumerate(members):
                col, row = member_index % 5, member_index // 5
                x = margin + col * thumb_w
                item_y = y + row * (thumb_h + label_h)
                image_path = args.data_root / member["image_path"]
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((thumb_w - 6, thumb_h - 6), Image.Resampling.LANCZOS)
                    offset_x = x + (thumb_w - image.width) // 2
                    offset_y = item_y + (thumb_h - image.height) // 2
                    canvas.paste(image, (offset_x, offset_y))
                draw.rectangle((x, item_y, x + thumb_w - 1, item_y + thumb_h - 1), outline="#777777")
                draw.text((x + 3, item_y + thumb_h + 3), member["image_path"], fill="black", font=font)
            y += row_count * (thumb_h + label_h) + margin
        canvas.save(args.output_dir / f"near_duplicate_contact_sheet_{page_index:03d}.jpg", quality=90)
    print(f"groups={len(ordered)} pages={len(pages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
