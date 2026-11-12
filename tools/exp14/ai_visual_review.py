"""Read-only evidence preparation and durable observation ledger for the Exp14 AI review."""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.exp13_cloud.g3.g3_batch_executor import digest, safe_file, write_exclusive, write_json_exclusive
from tools.exp14.export_ai_review_pilot import frozen_bundle
from tools.exp14.large_scale_review import IDENTITY, REASONS
from tools.exp14.prepare_large_scale_repair import canonical_digest, hash_file, load_json, unique_index

MANIFEST = "ai_review_manifest.json"
STATES = {"P": "PASS", "R": "RETRY_REPAIR", "A": "RETRY_ALIGNMENT", "H": "HOLD"}
CONFIDENCE = {"H": "HIGH", "M": "MEDIUM", "L": "LOW"}
CANDIDATE = {"Y": "YES", "N": "NO", "U": "UNSURE"}
RESULT_FIELDS = (*IDENTITY, "class_names", "polygon_ids", "geometry_status", "ai_decision", "ai_reason",
                 "ai_notes", "ai_confidence", "ai_a_candidate", "ai_b_candidate", "ai_polygon_decisions",
                 "visual_evidence_paths", "visual_review_completed", "needs_human_attention")
READY = "EXP14_A5_AI_REVIEW_COMPLETE_PENDING_HUMAN_CONFIRMATION"
PARTIAL = "EXP14_A5_AI_REVIEW_PARTIAL_PENDING_HUMAN_CONFIRMATION"
WIDTH = 1332


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def evidence_record(root, page, ledger):
    folder = (root / ledger["asset_directory"]).resolve()
    require(root / "derived" in folder.parents, "Unsafe derived path")
    record = load_json(safe_file(folder, "record.json"))
    require(page["assets"] == f"../{record['asset_directory']}/" and
            record["asset_directory"] == ledger["asset_directory"] and
            all(page["initial"][key] == (record[key] or "") for key in IDENTITY) and
            page["geometry_status"] == record["geometry_status"] == ledger["geometry_status"] and
            page["polygons"] == record["polygons"] and page["classes"] == record["classes"], "Frozen record mismatch")
    require(record["review_id"] == canonical_digest({k: record[k] for k in IDENTITY
            if k not in {"review_id", "source_id", "version"}}), "Frozen review ID mismatch")
    aligned = bool(record["aligned_sha256"])
    require(page["aligned"] == aligned and
            page["geometry_status"] == ("GEOMETRY_CANDIDATE" if aligned else "ALIGNMENT_UNVERIFIED"),
            "Frozen aligned state mismatch")
    files = {record["original"]: record["source_sha256"],
             record["raw"]: record["raw_sha256"],
             "polygon_overlay.png": record["file_sha256"]["polygon_overlay.png"]}
    if aligned:
        files["aligned.png"] = record["aligned_sha256"]
    for polygon in record["polygons"]:
        require(bool(polygon["after"]) == aligned, "Polygon crop alignment mismatch")
        for key in ("before", "after"):
            if polygon[key]:
                files[polygon[key]] = record["file_sha256"][polygon[key]]
    for name, sha in files.items():
        require(sha == record["file_sha256"][name], f"Frozen file SHA mismatch for {name}")
        hash_file(safe_file(folder, name), sha)
    return {"variant_id": record["variant_id"], "identity": {k: page["initial"][k] for k in IDENTITY},
            "class_names": record["classes"], "geometry_status": record["geometry_status"],
            "polygons": record["polygons"], "asset_directory": record["asset_directory"],
            "original": record["original"], "raw": record["raw"], "file_sha256": files}


def add_image(canvas, path, x, y, width, height):
    with Image.open(path) as image:
        fitted = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (x + (width - fitted.width) // 2, y + (height - fitted.height) // 2))


def render_case(root, item):
    folder = root / item["asset_directory"]
    aligned = bool(item["identity"]["aligned_sha256"])
    polygons = item["polygons"]
    cells = 3 if aligned else 6
    top_height, crop_height = 249, 150
    rows = (len(polygons) + cells - 1) // cells
    canvas = Image.new("RGB", (WIDTH, 60 + top_height + rows * crop_height + 10), "#1a2429")
    draw = ImageDraw.Draw(canvas)
    heading, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    draw.text((12, 5), item["variant_id"] + "  |  " + " / ".join(item["class_names"]) + "  |  " +
              ("Aligned" if aligned else "Raw / ALIGNMENT_UNVERIFIED"), fill="white", font=heading)
    for i, (name, title) in enumerate(((item["original"], "Original"), ("polygon_overlay.png", "Selected targets in red"),
            ("aligned.png" if aligned else item["raw"], "Aligned" if aligned else "Raw (no coordinate match)"))):
        x = 8 + i * 440
        draw.text((x, 34), title, fill="white", font=small)
        add_image(canvas, folder / name, x, 57, 430, top_height)
    for i, polygon in enumerate(polygons):
        step = 440 if aligned else 220
        x, y = 8 + (i % cells) * step, 61 + top_height + (i // cells) * crop_height
        draw.text((x, y), polygon["polygon_id"] + "  " + polygon["class"] + ("  before | after" if aligned else "  original"),
                  fill="#e4cc9f", font=small)
        add_image(canvas, folder / polygon["before"], x, y + 22, 207, crop_height - 24)
        if aligned:
            add_image(canvas, folder / polygon["after"], x + 216, y + 22, 207, crop_height - 24)
    return canvas


def save_sheet(output, number, tiles):
    path = Path("ai_review_evidence") / f"sheet_{number:03}.jpg"
    sheet = Image.new("RGB", (WIDTH, sum(image.height + 6 for _, image in tiles)), "#597480")
    top = 0
    for item, image in tiles:
        sheet.paste(image, (0, top))
        item["visual_sheet"] = path.as_posix()
        item["sheet_y_range"] = [top, top + image.height]
        top += image.height + 6
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=95, subsampling=0)
    blob = buffer.getvalue()
    write_exclusive(output / path, blob)
    return {"sheet": path.as_posix(), "sha256": digest(blob), "variants": [item["variant_id"] for item, _ in tiles]}


def prepare(root, output, controls):
    require(root.parent == output.parent and not output.exists(), "Use a new sibling private directory")
    pages, ledgers, frozen_hashes = frozen_bundle(root)
    reference = [r["variant_id"] for r in load_json(controls)["records"]
                 if r["reference_consistency_eligible"] or r["previously_discussed"]]
    require(len(set(reference)) == len(reference) == 18 and set(reference) <= pages.keys(),
            "Historical reference set differs")
    order = sorted(reference, key=lambda v: digest(v.encode()))
    order += [v for v in pages if v not in reference]
    output.mkdir()
    (output / "logs").mkdir()
    items, sheets, tiles = [], [], []
    for index, variant in enumerate(order):
        item = evidence_record(root, pages[variant], ledgers[variant])
        tile = render_case(root, item)
        if tiles and (index == len(reference) or len(tiles) == 4 or sum(t.height + 6 for _, t in tiles) + tile.height > 1970):
            sheets.append(save_sheet(output, len(sheets) + 1, tiles))
            tiles = []
        tiles.append((item, tile))
        items.append(item)
    if tiles:
        sheets.append(save_sheet(output, len(sheets) + 1, tiles))
    write_json_exclusive(output / MANIFEST, {"repair_root": str(root), "frozen_hashes": frozen_hashes,
        "started_utc": now(), "reference_count": len(reference), "records": items, "sheets": sheets})
    print(f"VERIFIED {len(items)} identities, {len(sheets)} evidence sheets")


def expanded_result(item, row, sheet):
    variant, decision, reason, notes, confidence, a, b, polygons, priority = row[:9]
    supplementary = row[9] if len(row) == 10 else []
    require(item["variant_id"] == variant and reason in REASONS and isinstance(notes, str) and notes.strip()
            and type(priority) is bool, "Invalid observation")
    ids = [p["polygon_id"] for p in item["polygons"]]
    require(len(polygons) == len(ids), "A decision is required for each viewed Polygon")
    states = {key: STATES[value] for key, value in zip(ids, polygons, strict=True)}
    require(decision != "P" or all(value == "PASS" for value in states.values()), "PASS requires all targets to pass")
    require(b != "Y" or (item["identity"]["aligned_sha256"] and "PASS" in states.values()), "B needs an aligned passed target")
    require(a != "Y" or decision == "P", "Whole-image A candidate needs a visual PASS")
    require(set(supplementary) <= {item["raw"], "valid.png"}, "Unknown supplementary image")
    return {**item["identity"], "class_names": item["class_names"], "polygon_ids": ids,
            "geometry_status": item["geometry_status"], "ai_decision": STATES[decision], "ai_reason": reason,
            "ai_notes": notes, "ai_confidence": CONFIDENCE[confidence], "ai_a_candidate": CANDIDATE[a],
            "ai_b_candidate": CANDIDATE[b], "ai_polygon_decisions": states,
            "visual_evidence_paths": {"sheet": sheet["sheet"], "sheet_sha256": sheet["sha256"],
               "sheet_y_range": item["sheet_y_range"], "asset_directory": item["asset_directory"],
               "source_assets": list(item["file_sha256"]), "supplementary_viewed": supplementary},
            "visual_review_completed": True, "needs_human_attention": priority}


def load_observations(output, manifest):
    items = unique_index(manifest["records"], "variant_id")
    sheets = {item["sheet"]: item for item in manifest["sheets"]}
    observed = {}
    for path in sorted((output / "logs").glob("observations_*.json")):
        batch = load_json(path)
        for row in batch["observations"]:
            variant = row[0]
            require(variant not in observed, "Duplicate observation")
            item = items[variant]
            observed[variant] = expanded_result(item, row, sheets[item["visual_sheet"]])
    return observed


def save_observations(output, rows):
    manifest = load_json(output / MANIFEST)
    items = unique_index(manifest["records"], "variant_id")
    sheets = {item["sheet"]: item for item in manifest["sheets"]}
    observed = load_observations(output, manifest)
    require(len({row[0] for row in rows}) == len(rows) and rows, "Duplicate or empty observation batch")
    results = []
    for row in rows:
        variant = row[0]
        require(variant not in observed, "Review identity was already recorded")
        item = items[variant]
        sheet = sheets[item["visual_sheet"]]
        hash_file(output / sheet["sheet"], sheet["sha256"])
        result = expanded_result(item, row, sheet)
        folder = Path(manifest["repair_root"]) / item["asset_directory"]
        for name in result["visual_evidence_paths"]["supplementary_viewed"]:
            hash_file(safe_file(folder, name), load_json(folder / "record.json")["file_sha256"][name])
        results.append(result)
    label = f"{len(observed) + 1:03}-{len(observed) + len(rows):03}"
    write_json_exclusive(output / "logs" / f"observations_{label}.json",
                         {"saved_utc": now(), "observations": rows, "records": results})
    progress = {"completed": len(observed) + len(rows), "total": len(items), "last_batch": label, "updated_utc": now()}
    temp = output / "logs" / f"progress_{label}.json"
    write_json_exclusive(temp, progress)
    # The immutable batch is authoritative. The root progress file is only a convenience snapshot.
    (output / "ai_review_progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    print(f"{progress['completed']}/{progress['total']} saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "save"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path)
    parser.add_argument("--controls", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.repair_root is None or args.controls is None:
            parser.error("prepare requires --repair-root and --controls")
        prepare(args.repair_root.resolve(), args.output.resolve(), args.controls)
    else:
        save_observations(args.output.resolve(), json.load(sys.stdin))


if __name__ == "__main__":
    main()
