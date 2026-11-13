"""Prepare, record, and publish Exp14 A6 old/new visual evidence."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import _csv_bytes
from tools.exp13_cloud.g3.verify_train_export import CLASSES
from tools.exp14.ai_visual_review import (
    MANIFEST, RESULT_FIELDS, WIDTH, add_image, load_observations, now, save_sheet,
)
from tools.exp14.large_scale_review import FIELDS, IDENTITY, REASONS, STATES, initial_row, validate_review
from tools.exp14.postprocess_large_scale_repair import finished_candidates
from tools.exp14.prepare_large_scale_repair import REPO, hash_file, load_json
from tools.exp14.targeted_v2 import frozen_plan

PHASE = "exp14_a6_targeted_v2"


def reviewable(root):
    records = {}
    for receipt in sorted((root / "postprocess_receipts").glob("*.json")):
        result = load_json(receipt)
        record = result["record"]
        if record and (not record.get("alignment_parent_review_id") or record["aligned_sha256"]):
            records[record["variant_id"]] = record
    return records


def waves(root):
    prepared, observations = {}, {}
    for path in sorted((root / "visual_waves").glob(f"*/{MANIFEST}")):
        manifest = load_json(path)
        for item in manifest["records"]:
            key = item["variant_id"]
            g3.require(key not in prepared, "Duplicate A6 visual evidence identity")
            prepared[key] = item
        for key, result in load_observations(path.parent, manifest).items():
            evidence = result["visual_evidence_paths"]
            sheet = path.parent / evidence["sheet"]
            hash_file(sheet, evidence["sheet_sha256"])
            evidence["sheet"] = sheet.relative_to(root).as_posix()
            evidence["prior_assets"] = prepared[key]["prior"]
            if prepared[key].get("comparison"):
                evidence["comparison_assets"] = prepared[key]["comparison"]
            observations[key] = result
    return prepared, observations


def verified_files(folder, record):
    for name, sha in record["file_sha256"].items():
        hash_file(g3.safe_file(folder, name), sha)
    return record["file_sha256"]


def prior_context(root, parent_root, parent, note_rows):
    folder = parent_root / parent["asset_directory"]
    image = parent["raw"] if not parent["aligned_sha256"] else "aligned.png"
    crops = {p["polygon_id"]: p["after"] for p in parent["polygons"] if p["after"]}
    assets = {name: parent["file_sha256"][name] for name in {image, parent["raw"], *crops.values()}}
    for name, sha in assets.items():
        hash_file(g3.safe_file(folder, name), sha)
    return {"variant_id": parent["variant_id"], "review_id": parent["review_id"],
            "assets": Path(os.path.relpath(folder, root)).as_posix(), "image": image,
            "raw": parent["raw"], "aligned": bool(parent["aligned_sha256"]),
            "crops": crops, "sha256": assets, "notes": note_rows}


def render_case(root, item):
    folder = root / item["asset_directory"]
    prior = item["prior"]
    prior_folder = root / prior["assets"]
    aligned = bool(item["identity"]["aligned_sha256"])
    bounded = "bounded.png" in item["file_sha256"]
    rescue = item["rescue"]
    columns = 4
    step, top, crop_height = 330, 220, 118
    canvas = Image.new("RGB", (WIDTH, 62 + top + len(item["polygons"]) * crop_height), "#18232a")
    draw = ImageDraw.Draw(canvas)
    heading, small = ImageFont.load_default(size=21), ImageFont.load_default(size=15)
    draw.text((10, 5), item["variant_id"] + "  |  " + " / ".join(item["class_names"]), fill="white", font=heading)
    previous = prior["raw"] if rescue else prior["image"]
    old_title = prior["variant_id"] + (" Raw" if rescue or not prior["aligned"] else " Aligned")
    names = [(folder / "polygon_overlay.png", "Original + Polygon"), (prior_folder / previous, old_title),
             (folder / ("aligned.png" if aligned else item["raw"]), "New Aligned" if aligned else "New Raw")]
    names.append((folder / "bounded.png", "New bounded edit") if bounded else
                 (folder / item["original"], "Original without overlay"))
    for i, (path, label) in enumerate(names):
        x = 8 + i * step
        draw.text((x, 34), label, fill="white", font=small)
        add_image(canvas, path, x, 54, step - 10, top)
    for index, polygon in enumerate(item["polygons"]):
        pid = polygon["polygon_id"]
        y = 57 + top + index * crop_height
        action = item["targets"].get(pid, "context")
        old_crop = prior["crops"].get(pid)
        images = [folder / polygon["before"], prior_folder / old_crop if old_crop else None,
                  folder / polygon["after"] if polygon["after"] else None]
        for col in range(columns):
            label = (f"{pid} {polygon['class']} {action}", "Old crop" if old_crop else "",
                     "New aligned crop" if aligned else "", "Bounded crop" if bounded else "")[col]
            if label:
                draw.text((8 + col * step, y), label, fill="#f0d5a0", font=small)
        for col, path in enumerate(images):
            if path:
                add_image(canvas, path, 8 + col * step, y + 20, step - 10, crop_height - 24)
        if bounded:
            with Image.open(folder / "bounded.png") as image:
                crop = ImageOps.contain(image.crop(polygon["box"]).convert("RGB"), (step - 10, crop_height - 24))
            canvas.paste(crop, (8 + 3 * step + (step - 10 - crop.width) // 2, y + 20))
    return canvas


def prepare(root, limit, *, plan_loader=frozen_plan, build_context=None, render=render_case):
    scope, _, _ = plan_loader(root)
    if build_context is None:
        parents = finished_candidates(Path(scope["parent_root"]))
    routes = {item["source_id"]: item for item in scope["routes"]}
    human = defaultdict(list)
    for route in scope["routes"]:
        for evidence in route["human_evidence"]:
            row = evidence["row"]
            human[row["source_id"]].append({
                **{key: row[key] for key in ("variant_id", "decision", "reason", "notes")},
                "targets": evidence["targets"],
            })
    prepared, _ = waves(root)
    records = [r for key, r in reviewable(root).items() if key not in prepared]
    if not records:
        print("NO_NEW_EVIDENCE")
        return
    records.sort(key=lambda r: (not bool(r.get("alignment_parent_review_id")), int(r["source_id"]), r["variant_id"]))
    root_waves = root / "visual_waves"
    output = root_waves / f"wave_{len(list(root_waves.glob('*/' + MANIFEST))) + 1:03}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    sheets, items, tiles = [], [], []
    for record in records[:limit]:
        sid = record["source_id"]
        route = routes[sid]
        files = verified_files(root / record["asset_directory"], record)
        context = (build_context(root, scope, route, record, human[sid]) if build_context else
                   {"prior": prior_context(root, Path(scope["parent_root"]), parents[route["parent_variant"]], human[sid])})
        targets = {pid: label for label, ids in (("keep", route["preserve_targets"]), ("align", route["alignment_targets"]),
                                               ("repair", route["repair_targets"])) for pid in ids}
        item = {"variant_id": record["variant_id"], "identity": {key: record[key] or "" for key in IDENTITY},
                "class_names": record["classes"], "geometry_status": record["geometry_status"],
                "polygons": record["polygons"], "asset_directory": record["asset_directory"],
                "raw": record["raw"], "original": record["original"], "file_sha256": files,
                **context, "rescue": bool(record.get("alignment_parent_review_id")), "targets": targets}
        tile = render(root, item)
        if tiles and (len(tiles) == 3 or sum(im.height + 6 for _, im in tiles) + tile.height > 1770):
            sheets.append(save_sheet(output, len(sheets) + 1, tiles))
            tiles = []
        tiles.append((item, tile))
        items.append(item)
    if tiles:
        sheets.append(save_sheet(output, len(sheets) + 1, tiles))
    g3.write_json_exclusive(output / MANIFEST, {"repair_root": str(root), "started_utc": now(),
                                               "records": items, "sheets": sheets})
    print(output, len(items), len(sheets))


def publish(root, *, phase=PHASE, name="review_v2_ai_assisted", title="Exp14-A6 定向修复复核", history=None):
    prepared, observed = waves(root)
    records = reviewable(root)
    g3.require(set(prepared) == set(observed) == set(records) and records,
               "Finish viewing every reviewable result before publishing")
    destination = root / name
    destination.mkdir()
    (destination / "exports").mkdir()
    data = {"phase": phase, "title": title, "class_names": CLASSES, "identity": IDENTITY,
            "fields": FIELDS, "states": sorted(STATES), "reasons": sorted(REASONS), "records": [],
            "ai_review": {"completed": len(observed)}}
    results = []
    for variant, record in sorted(records.items(), key=lambda row: (int(row[1]["source_id"]), row[0])):
        item = prepared[variant]
        g3.require(all(item["identity"][key] == (record[key] or "") for key in IDENTITY),
                   "Evidence identity mismatch")
        verified_files(root / record["asset_directory"], record)
        prior = {**item["prior"], "assets": "../" + item["prior"]["assets"] + "/"}
        if history is not None:
            prior["notes"] = history[record["source_id"]]
        for name, sha in prior["sha256"].items():
            hash_file(g3.safe_file(root / item["prior"]["assets"], name), sha)
        comparison = []
        for earlier in item.get("comparison", []):
            if earlier.get("assets"):
                for name, sha in earlier["sha256"].items():
                    hash_file(g3.safe_file(root / earlier["assets"], name), sha)
                earlier = {**earlier, "assets": "../" + earlier["assets"] + "/"}
            comparison.append(earlier)
        ai = observed[variant]
        results.append(ai)
        row = initial_row(record)
        row.update(decision="UNREVIEWED", whole_image_normal="", b_region_review="")
        data["records"].append({**record, "assets": "../" + record["asset_directory"] + "/",
                                "aligned": bool(record["aligned_sha256"]), "initial": row, "prior": prior,
                                "comparison": comparison, "input_mode": item.get("input_mode", ""),
                                "rescue": item["rescue"], "bounded": "bounded.png" in record["file_sha256"],
                                "ai": {**ai, "sheet_url": "../" + ai["visual_evidence_paths"]["sheet"]}})
    script_root = REPO / "tools" / "exp14"
    template = (script_root / "large_scale_review.html").read_text(encoding="utf-8")
    script = (script_root / "large_scale_review.js").read_text(encoding="utf-8")
    page = template.replace("__BUNDLE_JSON__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    g3.write_exclusive(destination / "index.html", page.replace("__REVIEW_SCRIPT__", script).encode("utf-8"))
    initial = [row["initial"] for row in data["records"]]
    g3.write_exclusive(destination / "review_template.csv", _csv_bytes(initial, FIELDS))
    g3.write_json_exclusive(destination / "review_template.json", {"phase": phase, "records": initial})
    instructions = ("双击上一级 index.html。可以采纳 AI 建议，也可以载入后修改并确认人工决定。\n"
                    "保存请导出 CSV，并备份 JSON。重新打开页面后可以导入恢复。\n"
                    "浏览器通常将文件下载到 Downloads，请复制到本 exports 目录。\n")
    g3.write_exclusive(destination / "exports" / "保存与恢复.txt", instructions.encode("utf-8"))
    g3.write_json_exclusive(root / "ai_review_results.json", {"phase": phase, "records": results})
    csv_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                 for key, value in row.items()} for row in results]
    g3.write_exclusive(root / "ai_review_results.csv", _csv_bytes(csv_rows, RESULT_FIELDS))
    print(destination, len(records), dict(Counter(row["ai_decision"] for row in results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "verify-export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, args.limit)
    elif args.action == "publish":
        frozen_plan(root)
        publish(root)
    else:
        if args.export is None:
            parser.error("verify-export requires --export")
        rows = validate_review(root, args.export, phase=PHASE, records=reviewable(root).values())
        print(f"VERIFIED_A6_REVIEW_EXPORT {len(rows)} identities")


if __name__ == "__main__":
    main()
