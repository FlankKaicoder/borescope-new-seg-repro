"""Use the existing Exp14 offline reviewer for the third repair round."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.exp14 import targeted_v2_review as review
from tools.exp14.large_scale_review import validate_review
from tools.exp14.targeted_v3 import A7, frozen_plan
from tools.exp14.targeted_v3_alignment import historical_record


def history_for_page(scope):
    notes = defaultdict(list)
    for route in scope["routes"]:
        for entry in route["human_evidence"]:
            row = entry["row"]
            notes[route["source_id"]].append({
                **{key: row[key] for key in ("variant_id", "decision", "reason", "notes")},
                "round": entry["round"], "targets": entry["targets"],
            })
    return notes


def context(root, scope, route, record, notes, *, versions=("V1", "V2"), alignment_suffix="_A7ALIGN"):
    def previous(variant):
        folder, item = historical_record(root, scope, variant)
        return review.prior_context(root, folder.parents[2], item, notes)

    parents = route["prior_variants"]
    if record.get("alignment_parent_review_id"):
        variant = next(job["parent_variant"] for job in route["alignments"]
                       if job["parent_variant"] + alignment_suffix == record["variant_id"])
        prior = previous(variant)
        comparison = [prior]
    else:
        comparison = []
        for version in versions:
            if parents[version]:
                comparison.append(previous(parents[version][0]))
            else:
                reason = "上一轮请求未返回可用图像" if route["delivery"] else "此前无该版本"
                comparison.append({"missing": version + " · " + reason})
        prior = next(item for item in reversed(comparison) if "assets" in item)
    request = next((row for row in scope["requests"] if row["variant_id"] == record["variant_id"].split("_OUT", 1)[0]), None)
    return {"prior": prior, "comparison": comparison, "input_mode": request["input_mode"] if request else ""}


def render(root, item):
    folder = root / item["asset_directory"]
    prior = item["prior"]
    aligned = bool(item["identity"]["aligned_sha256"])
    bounded = "bounded.png" in item["file_sha256"]
    rescue = item["rescue"]
    final = len(item["comparison"]) == 3
    step, top, crop_height = (218 if final else 264), 176, 105
    canvas = Image.new("RGB", (review.WIDTH, 62 + top + len(item["polygons"]) * crop_height), "#18232a")
    draw = ImageDraw.Draw(canvas)
    heading, small = ImageFont.load_default(size=20), ImageFont.load_default(size=13)
    draw.text((10, 5), item["variant_id"] + "  |  " + " / ".join(item["class_names"]), fill="white", font=heading)
    draw.text((780, 8), item["input_mode"], fill="#f0d5a0", font=small)
    names = [(folder / "polygon_overlay.png", "Original + Polygon")]
    if rescue:
        previous = root / prior["assets"]
        names.extend([(previous / prior["raw"], prior["variant_id"] + " Raw"),
                      (previous / prior["image"] if prior["aligned"] else folder / item["original"],
                       "Old Aligned" if prior["aligned"] else "Original")])
    else:
        names.extend((root / old["assets"] / old["image"], old["variant_id"]) if "assets" in old else
                     (None, old["missing"][:2] + " unavailable") for old in item["comparison"])
    names.append((folder / ("aligned.png" if aligned else item["raw"]), "New Aligned" if aligned else "New Raw"))
    names.append((folder / "bounded.png", "Bounded") if bounded else
                 (folder / "valid.png", "Valid Mask") if aligned and rescue else (folder / item["original"], "Original"))
    for index, (path, title) in enumerate(names):
        x = 7 + index * step
        draw.text((x, 34), title, fill="white", font=small)
        if path:
            review.add_image(canvas, path, x, 52, step - 8, top)
    for i, p in enumerate(item["polygons"]):
        pid = p["polygon_id"]
        y = 58 + top + i * crop_height
        action = item["targets"].get(pid, "context")
        earlier = item["comparison"] if not rescue else [{}, prior]
        paths = [folder / p["before"]]
        for old in earlier:
            crop = old.get("crops", {}).get(pid)
            paths.append(root / old["assets"] / crop if crop else None)
        paths.append(folder / p["after"] if aligned else None)
        for col, path in enumerate(paths):
            if path:
                labels = (f"{pid} {p['class']} {action}", "V1 crop", "V2 crop", "V3 crop", "New crop") if final else (
                    f"{pid} {p['class']} {action}", "V1 crop", "Previous crop", "New crop")
                label = labels[col]
                draw.text((7 + col * step, y), label, fill="#f0d5a0", font=small)
                review.add_image(canvas, path, 7 + col * step, y + 17, step - 8, crop_height - 19)
        if bounded:
            bounded_col = len(paths)
            draw.text((7 + bounded_col * step, y), "Bounded crop", fill="#f0d5a0", font=small)
            with Image.open(folder / "bounded.png") as image:
                crop = ImageOps.contain(image.crop(p["box"]).convert("RGB"), (step - 8, crop_height - 19))
            canvas.paste(crop, (7 + bounded_col * step + (step - 8 - crop.width) // 2, y + 17))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "verify-export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        review.prepare(root, args.limit, plan_loader=frozen_plan, build_context=context, render=render)
    elif args.action == "publish":
        scope, _, _ = frozen_plan(root)
        review.publish(root, phase=A7, name="review_v3_ai_assisted", title="Exp14-A7 第三轮修复复核",
                       history=history_for_page(scope))
    else:
        if not args.export:
            parser.error("verify-export requires --export")
        rows = validate_review(root, args.export, phase=A7, records=review.reviewable(root).values())
        print("Verified third-round export", len(rows))


if __name__ == "__main__":
    main()
