"""Read-only Exp14-A0 audit of frozen G3 assets. Writes review plans only."""

import argparse
import csv
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from tools.exp13_cloud.g3.closeout_g3c1_review import EXPECTED_REVIEW_SHA256, analyze
from tools.exp13_cloud.g3.verify_train_export import (
    EXPECTED_MANIFEST_SHA256, load_class_mapping, read_rows, sha256_file, validate_export_data,
)
from tools.exp14.label_plan import CLASSES, HOLD, PENDING, Review, approved_label_plan, polygon_rows


TRAIN_DIR = "g3a_train_export_unpacked_20261107_retry1"
C1_DIR = "g3c1_visual_closeout_20261108_verified"
D2_DIR = "g3d2_closeout_20261109_partial"
EXPORT_SHA = "594899f749934ac31d819c2eb804aaf3e2c72f45d73f3decd87ce92a8e31e98c"
BATCHES = {
    "V1": ("g3b2b_batch_review_20261108", "g3b2_batch_generation_20261108",
           "g3c0_alignment_20261108_release", "index.html",
           "2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa"),
    "V2": ("g3c2_v2_live_20261109", "g3c2_v2_execution_20261109",
           "g3c2_v2_alignment_20261109_retry2", "v1_v2_comparison.html",
           "1d6ee5f11c347ce1957f78667f7e66c7ee267ea372627811141b8667d0f4064f"),
}
DERIVED = {"aligned_output.png": "RGB", "hard_composite.png": "RGB",
           "feather_composite.png": "RGB", "valid_mask.png": "L", "valid_target_mask.png": "L"}
GATE = "TRAINING_DATA_NOT_YET_APPROVED"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def under(root, value):
    relative = Path(value)
    path = (root / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root),
            "Asset path escapes its expected directory")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def index(rows):
    by_id = {r["source_id"]: r for r in rows}
    require(len(by_id) == len(rows) and all(re.fullmatch(r"[A-Za-z0-9_-]+", k) for k in by_id),
            "Duplicate or invalid source ID")
    return by_id


def verify_train(root):
    train = under(root, TRAIN_DIR)
    frozen = under(train, "frozen_split_manifest.csv")
    exported = under(train, "export_manifest.csv")
    mapping = under(train, "class_mapping.json")
    require(sha256_file(frozen) == EXPECTED_MANIFEST_SHA256 and sha256_file(exported) == EXPORT_SHA,
            "Frozen TRAIN inputs changed")
    require(load_class_mapping(mapping) == dict(enumerate(CLASSES)), "Frozen class IDs changed")
    exported_rows = index(read_rows(exported))
    for row in exported_rows.values():
        require(row["split"] == "train", "Non-TRAIN export row")
        for key, prefix in (("export_image_path", "images/train"), ("export_json_path", "annotations/train")):
            require(under(train, row[key]).is_relative_to(train / prefix), "Export path outside TRAIN")
    report = validate_export_data(frozen, exported, train, mapping)
    require(report["status"] == "PASS", f"TRAIN export validation failed ({len(report['issues'])} issues)")
    return train, exported_rows, report


def load_image(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def identical(left, right):
    return left.size == right.size and ImageChops.difference(left, right).getbbox() is None


def validate_source(row, exported):
    pairs = (("image_path", "export_image_path"), ("json_path", "export_json_path"),
             ("source_image_sha256", "export_image_sha256"), ("source_json_sha256", "export_json_sha256"),
             ("group_id", "group_id"), ("image_width", "image_width"), ("image_height", "image_height"))
    require(row["split"] == exported["split"] == "train"
            and all(row[key] == exported[other] for key, other in pairs), "Frozen TRAIN source binding mismatch")


def verify_variant(train, exported, batch, generation, gallery, version, row, diag):
    sid = row["source_id"]
    variant = f"{sid}_{version}"
    require(sid in exported, f"{variant}: absent from frozen TRAIN")
    validate_source(row, exported[sid])
    source = under(train, row["image_path"])
    annotation = under(train, row["json_path"])
    shapes = read_json(annotation)["shapes"]
    size = int(row["image_width"]), int(row["image_height"])
    labels = polygon_rows(sid, shapes, size)
    all_ids = row["all_polygon_ids"].split("|")
    selected = row["selected_polygon_ids"].split("|")
    require(len(all_ids) == len(set(all_ids)) == len(labels) == int(row["all_polygon_count"])
            and set(all_ids) == labels.keys(), f"{variant}: full Polygon IDs differ")
    require(len(selected) == len(set(selected)) == int(row["selected_polygon_count"])
            and set(selected) <= labels.keys()
            and all(CLASSES[labels[pid][0]] == row["class"] for pid in selected),
            f"{variant}: selected Polygon IDs or repair class differ")
    mask = under(batch, row["mask_path"])
    require(sha256_file(mask) == row["mask_sha256"], f"{variant}: edit mask hash mismatch")
    with Image.open(mask) as rgba:
        require(rgba.mode == "RGBA" and rgba.size == size, f"{variant}: invalid RGBA edit mask")
        target = ImageChops.invert(rgba.getchannel("A"))
    expected = Image.new("L", size)
    draw = ImageDraw.Draw(expected)
    for pid in selected:
        points = shapes[int(pid.rsplit("_", 1)[1])]["points"]
        draw.polygon([tuple(map(float, point)) for point in points], fill=255)
    require(identical(expected, target) and target.histogram()[255] == int(row["mask_target_pixels"]),
            f"{variant}: repair mask differs from selected original Polygons")

    generated = under(generation, f"live/{sid}")
    raw = under(generated, "raw_output.png")
    raw_sha = sha256_file(raw)
    receipt = read_json(under(generated, "response_manifest.json"))
    raw_image = load_image(raw, "RGB")
    require(receipt["http_status"] == 200 and receipt["raw_output_sha256"] == raw_sha
            and receipt["output_size_wh"] == list(raw_image.size), f"{variant}: raw output identity differs")
    folder = under(gallery, sid)
    saved_diag = read_json(under(folder, "geometry_diagnostics.json"))
    expected_diag = dict(diag)
    if version == "V2":
        require(expected_diag.pop("v2_geometry_status") == diag["geometry_status"],
                f"{variant}: V2 comparison alias differs")
    require(expected_diag == saved_diag, f"{variant}: geometry summary and per-source diagnostic differ")
    require(all(diag[key] == row[key] for key in ("class", "source_image_sha256", "mask_sha256"))
            and diag["polygon_ids"] == row["selected_polygon_ids"]
            and diag["raw_output_sha256"] == raw_sha and diag["source_size"] == list(size)
            and diag["output_size"] == list(raw_image.size)
            and diag["trusted_training_pair_status"] == "NOT_GRANTED", f"{variant}: geometry identity differs")
    state = diag["geometry_status"]
    require(state in {"GEOMETRY_CANDIDATE", "ALIGNMENT_UNVERIFIED"}, f"{variant}: unknown geometry state")
    original = load_image(source, "RGB")
    atlas_original = under(folder, "original.png")
    atlas_raw = under(folder, "raw_output.png")
    overlay = under(folder, "polygon_overlay.png")
    require(identical(original, load_image(atlas_original, "RGB"))
            and identical(raw_image, load_image(atlas_raw, "RGB"))
            and load_image(overlay, "RGB").size == size, f"{variant}: atlas image identity differs")
    hashes = diag.get("derived_sha256", {})
    has_composite = state == "GEOMETRY_CANDIDATE"
    require(set(hashes) == (set(DERIVED) if has_composite else set()), f"{variant}: derived hash set differs")
    files = {}
    images = {}
    for name, mode in DERIVED.items():
        path = under(folder, name)
        require(path.is_file() == has_composite, f"{variant}: unexpected presence or absence of {name}")
        if has_composite:
            require(sha256_file(path) == hashes[name], f"{variant}: {name} hash mismatch")
            images[name] = load_image(path, mode)
            require(images[name].size == size, f"{variant}: {name} dimensions differ")
            files[f"{path.stem}_path"] = str(path)
            files[f"{path.stem}_sha256"] = hashes[name]
        else:
            files[f"{path.stem}_path"] = ""
            files[f"{path.stem}_sha256"] = ""
    omitted = ""
    if has_composite:
        valid_target = images["valid_target_mask.png"]
        require(identical(ImageChops.multiply(target, images["valid_mask.png"]), valid_target),
                f"{variant}: invalid target/support intersection")
        omitted = target.histogram()[255] - valid_target.histogram()[255]
        hard_expected = Image.composite(images["aligned_output.png"], original, valid_target)
        require(identical(hard_expected, images["hard_composite.png"]), f"{variant}: Hard pixels differ")
        unchanged = Image.composite(original, images["feather_composite.png"], valid_target)
        require(identical(unchanged, original), f"{variant}: Feather changes pixels outside target support")
    record = {
        "variant_id": variant, "source_id": sid, "group_id": row["group_id"], "version": version,
        "split": "train", "repair_class": row["class"], "class_id": CLASSES.index(row["class"]),
        "source_path": str(source), "source_sha256": row["source_image_sha256"],
        "annotation_path": str(annotation), "annotation_sha256": row["source_json_sha256"],
        "source_size_wh": "x".join(map(str, size)), "raw_size_wh": "x".join(map(str, raw_image.size)),
        "all_polygon_ids": row["all_polygon_ids"], "selected_polygon_ids": row["selected_polygon_ids"],
        "edit_mask_path": str(mask), "edit_mask_sha256": row["mask_sha256"],
        "raw_output_path": str(raw), "raw_output_sha256": raw_sha,
        "atlas_original_path": str(atlas_original), "atlas_original_sha256": sha256_file(atlas_original),
        "atlas_raw_path": str(atlas_raw), "atlas_raw_sha256": sha256_file(atlas_raw),
        "polygon_overlay_path": str(overlay), "polygon_overlay_sha256": sha256_file(overlay),
        "geometry_diagnostic_path": str(folder / "geometry_diagnostics.json"),
        "geometry_diagnostic_sha256": sha256_file(folder / "geometry_diagnostics.json"),
        "geometry_status": state, "full_target_pixels": target.histogram()[255],
        "omitted_target_pixels": omitted, **files,
    }
    return record, labels, set(selected)


def make_review(record, visual_status, historical_note, reasons):
    composite = bool(record["hard_composite_path"])
    return {
        "variant_id": record["variant_id"], "source_id": record["source_id"],
        "historical_geometry_status": record["geometry_status"],
        "historical_visual_status": visual_status, "historical_visual_note": historical_note,
        "priority_flags": "|".join(reasons),
        "a_technical_status": "SOURCE_GRID_COMPOSITE_AVAILABLE" if composite else "RAW_ONLY",
        "b_technical_status": "REGION_EVIDENCE_AVAILABLE" if composite else "REGION_ALIGNMENT_UNVERIFIED",
        "a_status": PENDING, "b_status": PENDING, "final_image_sha256": "",
        "final_image_choice": "", "a_whole_image_verified": "", "a_source_coordinates_verified": "",
        "a_empty_target_confirmed": "", "b_region_correspondence_verified": "",
        "b_artifact_shortcuts_reviewed": "", "a_reviewer": "", "b_reviewer": "",
        "a_notes": "", "b_notes": "",
    }


def class_counts(records):
    return {name: {"sources": len({r["source_id"] for r in records if r["repair_class"] == name}),
                   "v1": sum(r["repair_class"] == name and r["version"] == "V1" for r in records),
                   "v2": sum(r["repair_class"] == name and r["version"] == "V2" for r in records)}
            for name in CLASSES}


def audit(root):
    train, exported, train_report = verify_train(root)
    c1 = under(root, C1_DIR)
    frozen_review = under(c1, "frozen_review.csv")
    require(sha256_file(frozen_review) == EXPECTED_REVIEW_SHA256, "Frozen C1 review hash mismatch")
    historical = index(read_rows(frozen_review))
    d2 = read_json(under(root, f"{D2_DIR}/sanitized_closeout_summary.json"))
    require(d2["status"] == "G3D2_CLOSED_BOUNDED_PARTIAL_HUMAN_EVIDENCE"
            and d2["formal_tre_gate"] == "NOT_ESTABLISHED" and d2["trusted_pair_pass"] == 0,
            "Historical D2 closeout status differs")
    records, reviews, plans = [], [], []
    atlas_paths = {}
    v1_by_id = {}
    blocked_rows = Counter()
    for version, (batch_name, generation_name, gallery_name, atlas_name, manifest_sha) in BATCHES.items():
        batch, generation, gallery = [under(root, name) for name in (batch_name, generation_name, gallery_name)]
        manifest = under(batch, "expanded_batch_manifest.csv")
        require(sha256_file(manifest) == manifest_sha, f"{version}: historical batch hash mismatch")
        rows = index(read_rows(manifest))
        geom_summary = read_json(under(gallery, "geometry_summary.json"))
        geometry = index(geom_summary["records"])
        atlas = under(gallery, atlas_name)
        atlas_paths[version] = str(atlas)
        expected_count = 36 if version == "V1" else 3
        generated_ids = {p.parent.name for p in under(generation, "live").glob("*/raw_output.png")}
        require(len(rows) == expected_count and rows.keys() == geometry.keys() == generated_ids
                and atlas.is_file() and geom_summary["trusted_training_pair_pass"] == 0,
                f"{version}: candidate asset set differs")
        if version == "V1":
            require(geom_summary["manifest_sha256"] == manifest_sha, "C0 batch binding differs")
            c1_analysis = analyze(list(historical.values()), list(geometry.values()))
            shortlist = set(c1_analysis["visual_pass_pair_review_shortlist"])
            conflicts = {r["source_id"] for r in c1_analysis["decision_note_discrepancies"]}
            ui_overlap = set(c1_analysis["mask_ui_overlap_hold_sources"])
            retries = {r["source_id"]: r["failure_category"] for r in c1_analysis["needs_retry_sources"]}
            require(len(shortlist) == 23 and len(conflicts) == len(ui_overlap) == 1,
                    "Historical C1 review priorities differ")
            v1_ledger = index(read_rows(under(root, "g3b2_batch_review_atlas_20261108_final/review.csv")))
            require(v1_ledger.keys() == rows.keys(), "V1 generation ledger source set differs")
        else:
            v2_visual = index(read_rows(under(gallery, "review.csv")))
            require(v2_visual.keys() == rows.keys() <= v1_by_id.keys(), "V2 source/review set differs")
        for sid, row in rows.items():
            diag = geometry[sid]
            record, labels, selected = verify_variant(train, exported, batch, generation, gallery, version, row, diag)
            reasons = []
            if version == "V1":
                v1_by_id[sid] = record
                visual = historical[sid]
                ledger = v1_ledger[sid]
                require(ledger["raw_output_sha256"] == record["raw_output_sha256"]
                        and ledger["source_image_sha256"] == record["source_sha256"]
                        and ledger["polygon_ids"] == record["selected_polygon_ids"]
                        and ledger["generation_status"] == "SUCCEEDED", "V1 generation ledger differs")
                if sid in shortlist:
                    reasons.append("C1_VISUAL_GEOMETRY_SHORTLIST")
                if sid in conflicts:
                    reasons.append("C1_STATUS_NOTE_CONFLICT")
                if sid in ui_overlap:
                    reasons.append("POLYGON_UI_OVERLAP")
                if sid in retries:
                    reasons.append("C1_NEEDS_RETRY")
                    if row["class"] == "Material missing":
                        reasons.append("MATERIAL_RECONSTRUCTION_INCOMPLETE")
            else:
                old = v1_by_id[sid]
                visual = v2_visual[sid]
                require(row["v1_raw_output_sha256"] == old["raw_output_sha256"] == diag["v1_raw_output_sha256"]
                        and row["v1_mask_sha256"] == old["edit_mask_sha256"] == record["edit_mask_sha256"]
                        and row["selected_polygon_ids"] == old["selected_polygon_ids"]
                        and row["class"] == old["repair_class"] and row["attempt_number"] == "2"
                        and visual["geometry_status"] == diag["geometry_status"]
                        and visual["visual_review_status"] == diag["visual_review_status"] == "UNREVIEWED",
                        "V2 lineage or historical review state differs")
                reasons.append("V2_UNREVIEWED")
                if record["hard_composite_path"]:
                    reasons.append("V2_BOUNDARY_AND_SEAM_REVIEW")
            if record["omitted_target_pixels"]:
                reasons.append("FULL_TARGET_COVERAGE_INCOMPLETE")
            if not record["hard_composite_path"]:
                reasons.append("ALIGNMENT_UNVERIFIED")
            if not reasons:
                reasons.append("WHOLE_IMAGE_AND_INSTANCE_REVIEW")
            record["historical_visual_status"] = visual["visual_review_status"]
            record["atlas_path"] = str(atlas)
            records.append(record)
            reviews.append(make_review(record, visual["visual_review_status"], visual["review_notes"], reasons))
            decisions = dict.fromkeys(labels, HOLD)
            for pid, (class_id, _) in labels.items():
                plans.append({
                    "variant_id": record["variant_id"], "source_id": sid, "group_id": row["group_id"],
                    "polygon_id": pid, "class_id": class_id, "class": CLASSES[class_id],
                    "repair_target": "YES" if pid in selected else "NO",
                    "proposed_review_action": "CHECK_FOR_REMOVAL" if pid in selected else "CHECK_FOR_RETENTION",
                    "human_decision": PENDING, "current_label_action": HOLD,
                    "confirmed_image_sha256": "", "notes": "",
                })
            for route in ("A", "B"):
                try:
                    approved_label_plan(labels, decisions, Review(), final_sha256=record["raw_output_sha256"],
                                        source_size=None, final_size=None, route=route)
                except ValueError:
                    blocked_rows[route] += 1
    require(len(records) == 39 and blocked_rows == {"A": 39, "B": 39}, "Pending-review gate failed")
    groups = {r["group_id"] for r in records}
    require(len(groups) == len(v1_by_id) == 36, "Candidate independence differs")
    counts = class_counts(records)
    require(sum(item["sources"] for item in counts.values()) == 36
            and all(counts[name]["v1"] == (0 if name == "Tip curl" else 6) for name in CLASSES),
            "Expected V1 repair class balance differs")
    geometry_counts = dict(Counter(r["geometry_status"] for r in records))
    require(geometry_counts == {"GEOMETRY_CANDIDATE": 29, "ALIGNMENT_UNVERIFIED": 10},
            "Historical geometry candidate counts differ")
    summary = {
        "experiment": "Exp14-A0", "data_gate": GATE, "automatic_file_integrity": "PASS",
        "train_export_images_verified": train_report["checked_train_images"],
        "train_export_instances_verified": train_report["checked_train_instances"],
        "frozen_split_sha256": EXPECTED_MANIFEST_SHA256,
        "sources": len(v1_by_id), "groups": len(groups), "variants": len(records),
        "v1": 36, "v2": 3, "class_counts": counts, "geometry_counts": geometry_counts,
        "historical_visual_counts": dict(Counter(r["historical_visual_status"] for r in records)),
        "priority_flag_counts": dict(Counter(flag for r in reviews for flag in r["priority_flags"].split("|"))),
        "a_technical_counts": dict(Counter(r["a_technical_status"] for r in reviews)),
        "b_technical_counts": dict(Counter(r["b_technical_status"] for r in reviews)),
        "a_human_pending": len(reviews), "b_human_pending": len(reviews),
        "a_eligible_sources": 0, "b_eligible_sources": 0, "a_eligible_variants": 0, "b_eligible_variants": 0,
        "pending_samples_blocked_from_training": dict(blocked_rows),
        "label_plan_rows_on_hold": len(plans),
        "multi_target_variants": sum("|" in r["selected_polygon_ids"] for r in records),
        "partial_target_variants": len({r["variant_id"] for r in plans if r["repair_target"] == "NO"}),
        "approved_empty_labels": 0, "historical_formal_tre_gate": d2["formal_tre_gate"],
        "historical_trusted_pair_pass": d2["trusted_pair_pass"],
    }
    return records, reviews, plans, summary, atlas_paths


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def write_review(root, records, reviews, plans, summary, atlases):
    output = root / "exp14_a0_data_admission"
    try:
        output.mkdir()
    except FileExistsError:
        output = Path(tempfile.mkdtemp(prefix="exp14_a0_data_admission_", dir=root))
    for name, rows in (("candidate_manifest", records), ("candidate_review", reviews), ("label_change_plan", plans)):
        write_csv(output / f"{name}.csv", rows)
    write_text(output / "audit_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(output / "review_guide.md", f"""# Exp14-A0 人工审核

当前状态：{GATE}。文件核验完成，A/B 均等待人工审核。

- V1 图册：<{Path(atlases['V1']).as_uri()}>
- V2 图册：<{Path(atlases['V2']).as_uri()}>

在 candidate_review.csv 查找 variant_id。优先查看 C1 视觉及几何 shortlist、V2
有合成变体、覆盖不足、C1 备注冲突、UI 重叠和材料缺失案例。图册用于浏览历史证据。
请在本阶段 CSV 记录新判断，保留历史图册和审核记录原样。

1. 对照 candidate_manifest.csv 中的原图、Polygon 叠图、Raw、Hard 和 Feather，
   审核同一个最终变体并记录其文件哈希。无配准的 Raw 仍可接受整图复核，
   其中原坐标标签需要独立验证，尺寸变化时需重新标注或暂停。
2. 对每个 Polygon 在 label_change_plan.csv 记录是否确认消失、确认仍在原位置，
   或范围改变/不确定。确认消失才可计划删除。确认仍在原位置可保留。
   范围改变或不确定时保留 HOLD，交由人工处理。复核机械结构、残留、新异常、
   UI、接缝、目标覆盖和最终标签。只在整图确认无目标后记录显式空标签。
3. A 审核绑定最终图及全部实例判断。B 在同一最终图上另外复核原 Polygon 与
   伪正常区域的机械对应关系、目标状态和伪影捷径。两列独立填写。
4. 同一来源的 V1/V2 使用相同来源组。后续训练冻结变体时按来源计数。
   私有 CSV、坐标、图片和审核备注留在此资产根目录。
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve(strict=True)
    results = audit(root)
    write_review(root, *results)
    print(json.dumps(results[3], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
