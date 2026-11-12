"""Export blinded visual-review evidence from an immutable Exp14 repair bundle."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from tools.exp13_cloud.g3.g3_batch_executor import digest, safe_file, write_exclusive, write_json_exclusive
from tools.exp13_cloud.g3.prepare_g3b2a_batch import _csv_bytes, mask_for_selection
from tools.exp13_cloud.g3.verify_train_export import CLASSES
from tools.exp14.large_scale_review import IDENTITY
from tools.exp14.prepare_large_scale_repair import canonical_digest, hash_file, load_json, unique_index

STATUS = "EXP14_A5_PILOT40_VISUAL_EVIDENCE_READY"
BATCH = "EXP14_A5_PILOT40"
FIELDS = (
    "case_id", "review_id", "variant_id", "source_id", "ai_decision", "ai_reason",
    "ai_notes", "ai_confidence", "ai_a_candidate", "ai_b_candidate",
    "ai_polygon_decisions", "needs_human_attention",
)
IDENTIFIERS = FIELDS[:4]
GUIDE = """# Exp14-A5 视觉证据包

逐条打开 overview.png 查看原图、原图叠加 Polygon 和修复结果。红线圈出选中目标，蓝线圈出其他目标。整图与局部证据均可通过 manifest 中的相对路径单独打开。AI 建议应基于实际查看的图像。

ALIGNMENT_UNVERIFIED 案例提供未配准 Raw 整图和原图局部。原图的目标坐标只适用于原图。Aligned 案例的局部前后图使用同一原图坐标框。

在 ai_review_template.csv 或 ai_review_template.json 的对应行填入独立建议，保留 case_id、review_id、variant_id、source_id。ai_decision 使用 PASS、RETRY_REPAIR、RETRY_ALIGNMENT 或 HOLD。ai_notes 使用中文并指出观察到的位置。ai_confidence 使用 HIGH、MEDIUM 或 LOW，表示主观把握程度。ai_a_candidate 和 ai_b_candidate 可用 YES、NO 或 UNSURE。needs_human_attention 可用 YES 或 NO。

有逐实例建议时，ai_polygon_decisions 使用以原始 Polygon ID 为键、以建议状态为值的 JSON 对象。AI 建议留待人工复核。
"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_bundle(root):
    page = root / "review/index.html"
    match = re.search(r'<script id="bundle" type="application/json">(.*?)</script>',
                      page.read_text(encoding="utf-8"), re.S)
    require(match is not None, "Frozen review bundle is missing")
    bundle = json.loads(match[1])
    template = root / "review/review_template.json"
    ledger = root / "variant_status.csv"
    snapshot = load_json(template)
    by_variant = unique_index(bundle["records"], "variant_id")
    initial = unique_index(snapshot["records"], "variant_id")
    csv_initial = unique_index(read_csv(root / "review/review_template.csv"), "variant_id")
    variants = unique_index(read_csv(ledger), "variant_id")
    require(bundle["phase"] == snapshot["phase"] and
            set(by_variant) == set(initial) == set(csv_initial) == set(variants) and len(variants) == 530,
            "Frozen review page, templates and ledger disagree")
    require(all(row["initial"] == initial[key] == csv_initial[key] for key, row in by_variant.items()),
            "Frozen template values differ from the published page")
    require(len({row["review_id"] for row in initial.values()}) == len(initial),
            "Frozen review IDs are not unique")
    return by_variant, variants, {path.name: hash_file(path) for path in (page, template, ledger)}


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def asset(folder, record, filename, expected=None):
    blob = safe_file(folder, filename).read_bytes()
    sha = digest(blob)
    require(sha == record["file_sha256"][filename] and (expected is None or sha == expected),
            f"Image hash mismatch for {filename}")
    return blob


def verify_case(root, page, ledger, source, selection):
    variant = selection["variant_id"]
    path = safe_file(root / "derived", str(Path(ledger["asset_directory"]).relative_to("derived") / "record.json"))
    folder = path.parent
    record = load_json(path)
    require(ledger["asset_directory"] == record["asset_directory"] == folder.relative_to(root).as_posix() and
            page["assets"] == f"../{record['asset_directory']}/", "Derived directory identity mismatch")
    require(all(page["initial"][key] == (record[key] or "") for key in IDENTITY) and
            all(ledger[key] == (record[key] or "") for key in
                ("variant_id", "source_id", "version", "geometry_status", "raw_sha256",
                 "aligned_sha256", "request_sha256", "mask_sha256")),
            "Frozen review identity differs from the derived record or ledger")
    require(record["review_id"] == canonical_digest({key: record[key] for key in IDENTITY
            if key not in {"review_id", "source_id", "version"}}), "Review ID digest mismatch")
    require(page["initial"]["decision"] == selection["expected_decision"], "Frozen decision differs from the cohort definition")
    require(page["version"] == record["version"] and variant == f"{record['source_id']}_{record['version']}" and
            page["polygons"] == record["polygons"] and page["classes"] == record["classes"] == source["classes"] and
            source["split"] == "train" and source["decision"] == "SELECT" and
            source["source_id"] == record["source_id"] and source["image_sha256"] == record["source_sha256"] and
            source["annotation_sha256"] == record["annotation_sha256"],
            "Source, version or Polygon metadata changed")
    require(source["selected_polygon_ids"] == [p["polygon_id"] for p in record["polygons"]],
            "Selected Polygon exact-set mismatch")
    annotation_path = Path(source["annotation"]).resolve()
    require(root.parent in annotation_path.parents, "Annotation is outside the private asset root")
    hash_file(annotation_path, record["annotation_sha256"])
    annotation = load_json(annotation_path)
    original_name = record["original"]
    require(page["original"] == original_name and page["raw"] == record["raw"], "Display filename differs")
    original = rgb(asset(folder, record, original_name, record["source_sha256"]))
    overlay = rgb(asset(folder, record, "polygon_overlay.png"))
    aligned = bool(record["aligned_sha256"])
    geometry_status = "GEOMETRY_CANDIDATE" if aligned else "ALIGNMENT_UNVERIFIED"
    require(page["aligned"] == aligned and page["geometry_status"] == record["geometry_status"] == geometry_status,
            "Geometry status and file evidence disagree")
    raw = rgb(asset(folder, record, record["raw"], record["raw_sha256"]))
    repaired = rgb(asset(folder, record, "aligned.png", record["aligned_sha256"])) if aligned else raw
    require(list(original.size) == source["size"] and overlay.size == original.size and
            (not aligned or repaired.size == original.size), "Source-grid dimensions disagree")
    width, height = original.size
    require([annotation["imageWidth"], annotation["imageHeight"]] == [width, height],
            "Annotation and original dimensions disagree")
    pad = max(8, round(min(original.size) * 0.04))
    for polygon, index in zip(record["polygons"], source["indexes"], strict=True):
        label = annotation["shapes"][index]["label"]
        require(label == polygon["class"] == source["polygon_labels"][polygon["polygon_id"]] and
                polygon["polygon_id"] == f"{source['source_id']}_{index:03d}", "Polygon label or ID changed")
        raster, _ = mask_for_selection(original.size, annotation["shapes"], {index})
        left, top, right, bottom = raster.getbbox()
        box = (max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad))
        require(tuple(polygon["box"]) == box, "Frozen crop does not match the selected Polygon raster")
        before = rgb(asset(folder, record, polygon["before"]))
        require(before.size == original.crop(box).size and not ImageChops.difference(before, original.crop(box)).getbbox(),
                "Original crop pixels differ from the frozen coordinate box")
        require(bool(polygon["after"]) == aligned, "After crop is inconsistent with alignment")
        if aligned:
            after = rgb(asset(folder, record, polygon["after"]))
            expected = repaired.crop(box)
            require(after.size == expected.size and not ImageChops.difference(after, expected).getbbox(),
                    "Aligned crop pixels differ from the frozen coordinate box")
    return record, folder


def overview(case, paths, output):
    original, overlay, repaired = [rgb(path.read_bytes()) for path in paths]
    panels = [ImageOps.contain(image, (860, 640), Image.Resampling.LANCZOS)
              for image in (original, overlay, repaired)]
    gap, panel_width, start_y = 16, 860, 105
    canvas = Image.new("RGB", (3 * panel_width + 4 * gap, start_y + max(i.height for i in panels) + 42), "#172329")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.load_default(size=29)
    label = ImageFont.load_default(size=23)
    note = ImageFont.load_default(size=20)
    draw.text((gap, 10), f"{case['case_id']}  |  Source {case['source_id']}  |  {case['variant_id']}  |  "
              + " / ".join(case["class_names"]), font=heading, fill="#f8fbfc")
    draw.text((gap, 49), case["geometry_status"], font=note, fill="#f1c983")
    labels = ("Original", "Original + Polygon", "Aligned" if case["aligned_or_raw"] == "ALIGNED" else "Raw (unaligned)")
    for i, (panel, title) in enumerate(zip(panels, labels, strict=True)):
        x = gap + i * (panel_width + gap)
        draw.text((x, 76), title, font=label, fill="#f8fbfc")
        canvas.paste(panel, (x + (panel_width - panel.width) // 2, start_y))
    foot = canvas.height - 31
    draw.text((gap, foot), "Red outline: selected target", font=note, fill="#ff503c")
    draw.text((gap + 340, foot), "Blue outline: other annotated target", font=note, fill="#5ac8ff")
    with output.open("xb") as handle:
        canvas.save(handle, format="PNG")


def export_case(output, case_id, record, folder):
    prefix = f"items/{case_id}"
    aligned = bool(record["aligned_sha256"])
    case = {
        "case_id": case_id, "variant_id": record["variant_id"], "review_id": record["review_id"],
        "source_id": record["source_id"], "version": record["version"], "class_names": record["classes"],
        "polygon_ids": [p["polygon_id"] for p in record["polygons"]], "geometry_status": record["geometry_status"],
        "aligned_or_raw": "ALIGNED" if aligned else "RAW", "image_sha256": {},
    }

    def copy(filename, destination):
        path = f"{prefix}/{destination}"
        blob = asset(folder, record, filename)
        write_exclusive(output / path, blob)
        case["image_sha256"][path] = digest(blob)
        return path

    chosen = "aligned.png" if aligned else record["raw"]
    case["original_image"] = copy(record["original"], "original" + Path(record["original"]).suffix.lower())
    case["polygon_overlay_image"] = copy("polygon_overlay.png", "polygon_overlay.png")
    case["aligned_or_raw_image"] = copy(chosen, "aligned_or_raw" + Path(chosen).suffix.lower())
    case["polygons"] = []
    for polygon in record["polygons"]:
        pid = polygon["polygon_id"]
        target = {"polygon_id": pid, "class": polygon["class"], "original_box_xyxy": polygon["box"],
                  "before_image": copy(polygon["before"], f"crops/{pid}_before.png")}
        if aligned:
            target["after_image"] = copy(polygon["after"], f"crops/{pid}_after.png")
        case["polygons"].append(target)
    case["overview_image"] = f"{prefix}/overview.png"
    overview(case, [output / case[key] for key in ("original_image", "polygon_overlay_image", "aligned_or_raw_image")],
             output / case["overview_image"])
    case["image_sha256"][case["overview_image"]] = hash_file(output / case["overview_image"])
    return case


def upload_documents(cases, part):
    rows = [{key: case[key] if key in IDENTIFIERS else "" for key in FIELDS} for case in cases]
    manifest = {"batch_id": BATCH, "part": part, "record_count": len(cases), "records": cases}
    template = {"batch_id": BATCH, "records": rows}
    return {
        "pilot_manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        "ai_review_template.csv": _csv_bytes(rows, FIELDS),
        "ai_review_template.json": (json.dumps(template, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        "README.md": GUIDE.encode("utf-8"),
    }


def write_archive(output, part, cases):
    archive = output / f"pilot40_part{part:02}.zip"
    docs = upload_documents(cases, part)
    images = {name: sha for case in cases for name, sha in case["image_sha256"].items()}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
        for name, blob in docs.items():
            zipped.writestr(name, blob)
        for name in images:
            zipped.write(output / name, arcname=name)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        require(len(names) == len(set(names)) and set(names) == set(docs) | set(images) and
                all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts and
                    "\\" not in name and ":" not in name for name in names),
                "Archive contains unexpected or unsafe entries")
        require(zipped.testzip() is None, "Archive CRC check failed")
        require(all(zipped.read(name) == blob for name, blob in docs.items()), "Archived manifest or template differs")
        for name, sha in images.items():
            blob = zipped.read(name)
            require(digest(blob) == sha, f"Archived image hash mismatch for {name}")
            with Image.open(io.BytesIO(blob)) as image:
                image.verify()
            rgb(blob)
    return {"file": archive.name, "records": len(cases), "images": len(images),
            "bytes": archive.stat().st_size, "sha256": hash_file(archive)}


def export(root, output, selection, c0_path, a1_path):
    require(root.parent == output.parent and not output.exists(), "Choose a new directory in the private asset root")
    by_selection = unique_index(selection, "variant_id")
    require(len(by_selection) == 40, "The Pilot must contain exactly 40 different variants")
    output.mkdir()
    write_json_exclusive(output / "private_control/selection.json", selection)
    report = {"batch_id": BATCH, "api_post": 0, "gpu_training": 0, "new_alignment": 0, "review_changes": 0}
    active = "frozen bundle"
    try:
        pages, ledger, frozen_hashes = frozen_bundle(root)
        scope_path = root / "scope.json"
        scope = load_json(scope_path)
        c0 = unique_index(read_csv(c0_path), "source_id")
        a1 = unique_index(read_csv(a1_path), "variant_id")
        # Stable ordering independent of the private reference group and decision.
        ordered = sorted(by_selection, key=lambda variant: digest(f"{BATCH}:{variant}".encode()))
        verified = []
        for active in ordered:
            page = pages[active]
            source = scope["sources"][page["source_id"]]
            record, folder = verify_case(root, page, ledger[active], source, by_selection[active])
            verified.append((record, folder, page))
        require(sum(bool(record["aligned_sha256"]) for record, _, _ in verified) == 31,
                "Expected 31 Aligned and 9 Alignment Unverified cases")
        active = "export"
        cases = []
        control = []
        for index, (record, folder, page) in enumerate(verified, start=1):
            active = record["variant_id"]
            case = export_case(output, f"case_{index:03}", record, folder)
            cases.append(case)
            historical = {
                **{key: case[key] for key in IDENTIFIERS}, **by_selection[active],
                "frozen_review": page["initial"], "historical_visual": record["historical_visual"],
                "a2_status": record["a2_status"], "historical_support": record["historical_support"],
                "source_hold": record["source_hold"],
            }
            if record["origin"] == "G3_V1":
                require(scope["historical_variants"][active]["raw_sha256"] == record["raw_sha256"] and
                        c0[record["source_id"]]["visual_review_status"] == record["historical_visual"],
                        "Historical visual notes do not bind the same repair")
                historical["c0_review"] = c0[record["source_id"]]
            if active in a1:
                entry = a1[active]
                require(entry["source_id"] == record["source_id"] and
                        (entry["image_verified"] != "True" or entry["final_image_sha256"] == record["aligned_sha256"]),
                        "A1 review notes do not bind the same repair")
                historical["a1_review"] = entry
            control.append(historical)
        write_json_exclusive(output / "private_control/historical_decisions.json", {
            "batch_id": BATCH, "records": control,
            "reference_note": "Previously discussed examples are excluded from independent agreement. HOLD can reflect geometric support.",
            "frozen_bundle_sha256": frozen_hashes,
            "historical_source_sha256": {"c0_review": hash_file(c0_path), "a1_review": hash_file(a1_path)},
        })
        active = "archive validation"
        for name, blob in upload_documents(cases, None).items():
            write_exclusive(output / name, blob)
        archives = [write_archive(output, part, cases[start:start + 20]) for part, start in enumerate((0, 20), start=1)]
        require(all(hash_file(root / "review" / name) == sha for name, sha in frozen_hashes.items()
                    if name != "variant_status.csv") and hash_file(root / "variant_status.csv") == frozen_hashes["variant_status.csv"],
                "Frozen review inputs changed during export")
        report.update({
            "status": STATUS, "records": len(cases), "sources": len({case["source_id"] for case in cases}),
            "aligned": 31, "alignment_unverified": 9,
            "polygon_count": sum(len(case["polygons"]) for case in cases),
            "multi_polygon_records": sum(len(case["polygons"]) > 1 for case in cases),
            "multi_class_records": sum(len(case["class_names"]) > 1 for case in cases),
            "class_records": {label: sum(label in c["class_names"] for c in cases) for label in CLASSES},
            "class_aligned_records": {label: sum(label in c["class_names"] and c["aligned_or_raw"] == "ALIGNED" for c in cases)
                                      for label in CLASSES},
            "frozen_bundle_decisions": dict(Counter(row["initial"]["decision"] for row in pages.values())),
            "frozen_bundle_geometry": dict(Counter(row["geometry_status"] for row in pages.values())),
            "pilot_frozen_decisions": dict(Counter(row["frozen_review"]["decision"] for row in control)),
            "identity_checks": 40, "crop_coordinate_and_pixel_checks": sum(
                len(case["polygons"]) * (2 if case["aligned_or_raw"] == "ALIGNED" else 1) for case in cases),
            "archives": archives, "archive_image_decode_and_sha256": sum(item["images"] for item in archives),
            "private_controls_excluded": True, "ai_suggestions_written": 0,
        })
    except (OSError, ValueError, KeyError) as error:
        report.update(status="CONFLICT", conflict_context=active, detail=str(error))
        write_json_exclusive(output / "validation_report.json", report)
        raise
    write_json_exclusive(output / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", default="-", help="Private cohort JSON path or - for stdin")
    parser.add_argument("--c0-review", type=Path, required=True)
    parser.add_argument("--a1-review", type=Path, required=True)
    args = parser.parse_args()
    selection = json.load(sys.stdin) if args.selection == "-" else load_json(Path(args.selection))
    export(args.repair_root.resolve(), args.output.resolve(), selection, args.c0_review, args.a1_review)


if __name__ == "__main__":
    main()
