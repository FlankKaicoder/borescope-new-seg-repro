"""Build a private, offline seven-class TRAIN candidate review atlas."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tools.exp13_cloud.g3.verify_train_export import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_TRAIN_INSTANCES,
    validate_export_data,
)

CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
# These are public-record call histories, not quality decisions.
HISTORY = {
    "2": "G1F-1",
    "23": "G1B,G1C,G1D,G2",
    "260": "G2",
    "470": "G2",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2.0)


def make_overlay(image: Any, points: list[tuple[float, float]]) -> Any:
    from PIL import Image, ImageDraw

    overlay = image.convert("RGBA")
    layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(points, fill=(255, 40, 40, 80), outline=(255, 230, 0, 255), width=max(2, round(min(image.size) / 400)))
    return Image.alpha_composite(overlay, layer).convert("RGB")


def candidate_record(row: dict[str, str], shape_index: int, shape: dict[str, Any], width: int, height: int) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    points = [(float(point[0]), float(point[1])) for point in shape.get("points", [])]
    if len(points) < 3:
        raise ValueError(f"{row.get('stem')} shape {shape_index}: invalid polygon")
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    area = polygon_area(points)
    padding = max(8, round(0.10 * max(x2 - x1, y2 - y1)))
    crop_box = (max(0, math.floor(x1 - padding)), max(0, math.floor(y1 - padding)), min(width, math.ceil(x2 + padding)), min(height, math.ceil(y2 + padding)))
    record = {
        "candidate_id": f"{row.get('stem')}_{shape_index:03d}",
        "source_id": row.get("stem", ""),
        "class": str(shape.get("label", "")),
        "split": row.get("split", ""),
        "group_id": row.get("group_id", ""),
        "near_duplicate_group_id": row.get("near_duplicate_group_id") or row.get("group_id") or "UNKNOWN",
        "polygon_index": shape_index,
        "polygon_point_count": len(points),
        "area_polygon_px": round(area, 3),
        "area_fraction_polygon": round(area / max(1, width * height), 8),
        "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "centroid_xy": [round(sum(xs) / len(xs), 3), round(sum(ys) / len(ys), 3)],
        "padding_px": padding,
        "padding_crop_xyxy": list(crop_box),
        "padding_clipped": crop_box[0] == 0 or crop_box[1] == 0 or crop_box[2] == width or crop_box[3] == height,
        "padding_risk": "CLIPPED" if (crop_box[0] == 0 or crop_box[1] == 0 or crop_box[2] == width or crop_box[3] == height) else "LOW",
        "ui_risk": "UNKNOWN (no source-specific UI map in frozen manifest)",
        "historical_g1_g2_calls": HISTORY.get(row.get("stem", ""), "NO_RECORDED_G1_G2_CALL"),
        "review_status": "UNREVIEWED",
        "review_notes": "",
        "image_width": width,
        "image_height": height,
    }
    return record, points


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing atlas directory: {args.output}")
    if not args.verification_report.is_file():
        raise RuntimeError("formal candidate atlas requires a completed TRAIN integrity report")
    gate = json.loads(args.verification_report.read_text(encoding="utf-8-sig"))
    if gate.get("status") != "PASS":
        raise RuntimeError("TRAIN integrity report did not PASS")
    if gate.get("frozen_manifest_sha256") != args.expected_manifest_sha256:
        raise RuntimeError("verification report frozen Manifest SHA256 mismatch")
    if gate.get("export_manifest_sha256") != __import__("hashlib").sha256(args.export_manifest.read_bytes()).hexdigest():
        raise RuntimeError("verification report export Manifest SHA256 mismatch")
    if gate.get("checked_train_images") != args.expected_count or gate.get("checked_train_instances") != args.expected_instances:
        raise RuntimeError("verification report TRAIN image/instance counts mismatch")
    recomputed = validate_export_data(args.manifest, args.export_manifest, args.data_root, args.class_mapping,
                                      args.expected_manifest_sha256, args.expected_count, args.expected_instances)
    if recomputed.get("status") != "PASS":
        raise RuntimeError("revalidated TRAIN export failed: " + json.dumps(recomputed.get("issues", []), ensure_ascii=False))
    if recomputed.get("export_manifest_sha256") != gate.get("export_manifest_sha256"):
        raise RuntimeError("verification report does not match current export Manifest")
    rows = read_rows(args.export_manifest)
    from PIL import Image

    args.output.mkdir(parents=True)
    image_dir, crop_dir, preview_dir = args.output / "images", args.output / "crops", args.output / "previews"
    image_dir.mkdir()
    crop_dir.mkdir()
    preview_dir.mkdir()
    records: list[dict[str, Any]] = []
    for row in rows:
        image_value = row.get("export_image_path", "")
        json_value = row.get("export_json_path", "")
        image_path, json_path = resolve(args.data_root, image_value), resolve(args.data_root, json_value)
        if not image_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"missing TRAIN image/JSON pair: {image_path} / {json_path}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        candidate_row = {
            **row,
            "stem": row["source_id"],
            "image_path": row["source_image_path"],
            "json_path": row["source_json_path"],
        }
        for shape_index, shape in enumerate(payload.get("shapes", [])):
            label = str(shape.get("label", ""))
            if label not in CLASSES:
                raise ValueError(f"unknown class {label!r} in {json_path}")
            if shape.get("shape_type") != "polygon":
                raise ValueError(f"non-polygon annotation in {json_path}: shape {shape_index}")
            record, points = candidate_record(candidate_row, shape_index, shape, image.width, image.height)
            stem = record["candidate_id"]
            overlay_path = image_dir / f"{stem}_overlay.png"
            crop_path = crop_dir / f"{stem}_defect.png"
            make_overlay(image, points).save(overlay_path, format="PNG")
            box = tuple(record["padding_crop_xyxy"])
            make_overlay(image.crop(box), [(x - box[0], y - box[1]) for x, y in points]).save(crop_path, format="PNG")
            record["overlay_path"] = str(overlay_path.relative_to(args.output)).replace("\\", "/")
            record["crop_path"] = str(crop_path.relative_to(args.output)).replace("\\", "/")
            for key, source_path in (("overlay_preview_path", overlay_path), ("crop_preview_path", crop_path)):
                with Image.open(source_path) as preview:
                    preview.thumbnail((640, 480))
                    preview_path = preview_dir / source_path.name
                    preview.save(preview_path, format="JPEG", quality=82, optimize=True)
                record[key] = str(preview_path.relative_to(args.output)).replace("\\", "/")
            records.append(record)
    fields = [
        "candidate_id", "source_id", "class", "split", "group_id", "near_duplicate_group_id", "polygon_index",
        "polygon_point_count", "area_polygon_px", "area_fraction_polygon", "bbox_xyxy", "centroid_xy", "padding_px",
        "padding_crop_xyxy", "padding_clipped", "padding_risk", "ui_risk", "historical_g1_g2_calls", "overlay_path",
        "crop_path", "overlay_preview_path", "crop_preview_path", "image_width", "image_height", "review_status", "review_notes",
    ]
    with (args.output / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ("bbox_xyxy", "centroid_xy", "padding_crop_xyxy"):
                row[key] = json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
            writer.writerow({key: row.get(key, "") for key in fields})
    cards = []
    for record in records:
        card = (
            '<article class="candidate" data-class="{cls}" data-source="{source}" data-status="UNREVIEWED">'
            '<h2>{cid} <span>{cls}</span></h2><p>source_id={source} | group_id={group} | near_duplicate={near} | '
            'history={history}</p><p>area={area} px ({fraction:.4%}) | bbox={bbox} | padding={padding} ({risk}) | UI={ui}</p>'
            '<div class="images"><figure><img loading="lazy" src="{overlay}"><figcaption>Original + polygon overlay</figcaption></figure>'
            '<figure><img loading="lazy" src="{crop}"><figcaption>Defect crop</figcaption></figure></div></article>'
        ).format(
            cls=html.escape(record["class"]), source=html.escape(record["source_id"]), cid=html.escape(record["candidate_id"]),
            group=html.escape(record["group_id"]), near=html.escape(record["near_duplicate_group_id"]), history=html.escape(record["historical_g1_g2_calls"]),
            area=record["area_polygon_px"], fraction=record["area_fraction_polygon"], bbox=html.escape(str(record["bbox_xyxy"])),
            padding=record["padding_px"], risk=html.escape(record["padding_risk"]), ui=html.escape(record["ui_risk"]),
            overlay=html.escape(record["overlay_preview_path"]), crop=html.escape(record["crop_preview_path"]),
        )
        cards.append(card)
    options = "".join(f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in CLASSES)
    index = """<!doctype html>
<meta charset="utf-8"><title>Exp13 G3-A TRAIN candidate review</title>
<style>body{font:14px Arial;margin:20px;background:#f5f5f5}header{position:sticky;top:0;background:white;padding:12px;z-index:2}.candidate{background:white;margin:16px 0;padding:12px;border:1px solid #bbb}.candidate h2{margin:0 0 6px}.candidate h2 span{font-size:14px;color:#555}.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}img{max-width:520px;max-height:420px;border:1px solid #888}figcaption{font-size:12px}</style>
<header><b>Exp13-G3-A TRAIN candidates</b> <label>Class <select id="class"><option value="">All</option>__OPTIONS__</select></label> <label>Source ID <input id="source" placeholder="search"></label> <span id="count"></span><p>Review decisions are stored in review.csv: SELECT / MAYBE / REJECT. All candidates start UNREVIEWED.</p></header>
<main>__CARDS__</main><script>
const cards=[...document.querySelectorAll('.candidate')], cls=document.querySelector('#class'), source=document.querySelector('#source'), count=document.querySelector('#count');
function filter(){const c=cls.value.toLowerCase(),s=source.value.toLowerCase();let n=0;for(const card of cards){const show=(!c||card.dataset.class.toLowerCase()===c)&&(!s||card.dataset.source.toLowerCase().includes(s));card.hidden=!show;if(show)n++}count.textContent=`${n} shown / ${cards.length} total`;}cls.onchange=source.oninput=filter;filter();
</script>""".replace("__OPTIONS__", options).replace("__CARDS__", "\n".join(cards))
    (args.output / "index.html").write_text(index, encoding="utf-8")
    summary = {
        "status": "READY_FOR_HUMAN_SELECTION",
        "train_images": len(rows),
        "train_instances": sum(int(row["instance_count"]) for row in rows),
        "candidate_polygons": len(records),
        "independent_source_count": len({row["source_id"] for row in rows}),
        "class_source_counts": {label: len({row["source_id"] for row in rows if label in row["labels_present"].split("|")}) for label in CLASSES},
        "class_candidate_counts": dict(Counter(record["class"] for record in records)),
        "review_csv": str(args.output / "review.csv"),
        "index_html": str(args.output / "index.html"),
        "all_default_status": "UNREVIEWED",
        "api_posts": 0,
        "gpu_training": 0,
        "val_test_image_access": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TRAIN_IMAGES)
    parser.add_argument("--expected-instances", type=int, default=EXPECTED_TRAIN_INSTANCES)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
