"""Reconcile exact historical pixel evidence and package the Exp14 A recovery review."""

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.admission_diagnosis import verify_frozen_a9
from tools.exp14.admission_diagnosis_a import bounded_metrics, mask_rasters
from tools.exp14.prepare_large_scale_repair import hash_file, load_json

REUSABLE = "HUMAN_REVIEWED_ASSET_REUSABLE"
PROPOSED = "RECOVERY_PROPOSED_PENDING_HUMAN"
DECISION = "NEEDS_USER_DECISION"
EXCLUDED = "NOT_RECOVERABLE_AS_WHOLE_NORMAL_PROPOSED"
COMPLETE = "EXP14_A9D1_FULL392_RECOVERY_COMPLETE_PENDING_HUMAN_CONFIRMATION"
PARTIAL = "EXP14_A9D1_FULL392_RECOVERY_PARTIAL_PENDING_HUMAN_CONFIRMATION"
KINDS = {"RAW": None, "ALIGNED": "aligned.png", "BOUNDED": "bounded.png"}


def csv_rows(path):
    return g3.read_csv(path)[1]


def write_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    g3.write_exclusive(path, buffer.getvalue().encode("utf-8-sig"))
    _, restored = g3.read_csv(path)
    g3.require(restored == [{k: str(row.get(k, "")) for k in fields} for row in rows], "CSV round-trip differs")


def files_for(item):
    record, audit = item["record"], item["audit"]
    files = audit["assets"]
    return {kind: files[name or record["raw"]] for kind, name in KINDS.items() if name or record.get("raw")
            if (name or record["raw"]) in files}


def ordered_source_ids(ids):
    return sorted(ids, key=int)


def source_digest(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def baseline(a9, d0):
    verify_frozen_a9(a9)
    receipt = load_json(d0 / "delivery_receipt.json")
    for name in ("a_192_root_cause_summary.json", "a_recoverable_candidate_queue.csv",
                 "evidence/a_agent_notes/visual_observations.json"):
        hash_file(d0 / name, receipt["private_sha256"][name])
    evidence = load_json(a9 / "audits/verified_evidence_reconciled.json")
    grouped = defaultdict(list)
    for item in evidence["reviews"]:
        grouped[item["record"]["source_id"]].append(item)
    whole_rows = {sid: [x for x in rows if x["human"]["decision"] == "PASS" and x["human"]["whole_image_normal"] == "YES"]
                  for sid, rows in grouped.items()}
    whole_rows = {sid: rows for sid, rows in whole_rows.items() if rows}
    a_items = {x["source_id"]: json.loads(x["recommended"]) for x in csv_rows(a9 / "registry/a_candidate_manifest.csv")}
    holds = {x["source_id"]: x for x in load_json(d0 / "a_192_root_cause_summary.json")["source_details"]}
    shortlist = {x["source_id"] for x in csv_rows(d0 / "a_recoverable_candidate_queue.csv") if x["screened_potential"] == "true"}
    g3.require(len(whole_rows) == 392 and sum(map(len, whole_rows.values())) == 393
               and len(a_items) == 200 and len(holds) == 192 and len(shortlist) == 119
               and (a_items.keys() | holds.keys()) == whole_rows.keys() and not (a_items.keys() & holds.keys())
               and shortlist <= holds.keys(), "Exact A whole-review cohort changed")
    d0_notes = {x["review_id"]: x for x in load_json(d0 / "evidence/a_agent_notes/visual_observations.json")}
    reusable = [sid for sid, item in a_items.items() if item["pixel_asset_state"] == "ALIGNED_REVIEW_ID_BOUND"]
    pending = a_items.keys() - set(reusable)
    g3.require(len(reusable) == 138 and len(pending) == 62, "A9 asset-binding count changed")
    review_order = ordered_source_ids(shortlist) + ordered_source_ids(holds.keys() - shortlist) + ordered_source_ids(pending)
    return evidence, grouped, whole_rows, a_items, holds, shortlist, reusable, review_order, d0_notes


def open_verified(asset):
    data = Path(asset["path"]).read_bytes()
    g3.require(g3.digest(data) == asset["sha256"], "Opened asset differs from verified identity")
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def save_image(path, image, file_format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=file_format, **({"quality": 93} if file_format == "JPEG" else {}))
    data = buffer.getvalue()
    if path.exists():
        hash_file(path, g3.digest(data))
    else:
        g3.write_exclusive(path, data)
    return {"path": str(path), "sha256": g3.digest(data)}


def all_polygon_overlay(original, source, shapes):
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    for pid, shape in zip(source["all_polygon_ids"], shapes, strict=True):
        points = [tuple(x) for x in shape["points"]]
        closed = points + [points[0]]
        draw.line(closed, fill="#000000", width=4)
        draw.line(closed, fill="#4ce8f8", width=2)
        x, y = min(p[0] for p in points), min(p[1] for p in points)
        draw.text((x, max(0, y - 13)), pid.rsplit("_", 1)[-1], fill="#ffffff", stroke_width=2, stroke_fill="#000000")
    return overlay


def crop_boxes(source, shapes):
    width, height = source["size"]
    boxes = {}
    for pid, shape in zip(source["all_polygon_ids"], shapes, strict=True):
        points = shape["points"]
        x0, y0 = min(p[0] for p in points), min(p[1] for p in points)
        x1, y1 = max(p[0] for p in points), max(p[1] for p in points)
        pad = max(18, int(max(x1 - x0, y1 - y0) * 0.22))
        boxes[pid] = (max(0, int(x0) - pad), max(0, int(y0) - pad),
                      min(width, int(x1 + 1) + pad), min(height, int(y1 + 1) + pad))
    return boxes


def shape_labels(source, shapes):
    return {pid: shape["label"] for pid, shape in zip(source["all_polygon_ids"], shapes, strict=True)}


def fit(canvas, im, x, y, width, height):
    thumb = ImageOps.contain(im, (width, height))
    canvas.paste(thumb, (x + (width - thumb.width) // 2, y))


def evidence_window(row, images, evidence_dir):
    boxes = row["boxes"]
    images = {key: im for key, im in images.items() if key != "RAW"}
    if not boxes or not images:
        return
    keys = ["Original", "ALIGNED", "BOUNDED"]
    for start in range(0, len(boxes), 5):
        chunk = list(boxes.items())[start:start + 5]
        canvas = Image.new("RGB", (1140, 30 + 239 * len(chunk)), "#20262d")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f'{row["source_id"]} {row["primary"]["variant_id"]} Original-grid Polygon crops', fill="white")
        for rank, (pid, box) in enumerate(chunk):
            y = 30 + rank * 239
            for col, label in enumerate(keys):
                if label in images:
                    draw.text((380 * col + 7, y), f"{pid}  {label}", fill="white")
                    fit(canvas, images[label].crop(box), 380 * col + 4, y + 18, 372, 215)
        path = evidence_dir / f'{row["source_id"]}_targets_{start//5:02}.jpg'
        row["target_sheets"].append(save_image(path, canvas, "JPEG"))


def render_full_sheet(sheet, path):
    canvas = Image.new("RGB", (1520, 325 * len(sheet)), "#20262d")
    draw = ImageDraw.Draw(canvas)
    for index, (row, images) in enumerate(sheet):
        draw.text((8, index * 325 + 5), f'{row["ordinal"]:03}    {row["source_id"]}    {row["primary"]["variant_id"]}    {row["group"]}', fill="#ffffff")
        for col, key in enumerate(("Polygon", "RAW", "ALIGNED", "BOUNDED")):
            x, y = 380 * col, index * 325 + 26
            draw.text((x + 7, y), key, fill="#ffffff")
            if key in images:
                fit(canvas, images[key], x + 4, y + 19, 372, 276)
    return save_image(path, canvas, "JPEG")


def prepare(a9, d0, output):
    evidence, grouped, whole_rows, a_items, holds, shortlist, reusable, review_order, d0_notes = baseline(a9, d0)
    pinned = load_json(a9 / "registry/candidate_asset_sha256.json")["files"]
    ordinal_by_id = {sid: index for index, sid in enumerate(review_order)}
    visual_dir = output / "visual_inputs"
    rows, sheets = [], defaultdict(list)
    for sid in review_order + ordered_source_ids(reusable):
        source = evidence["sources"][sid]
        g3.require(source["split"] == "train" and source["source_id"] == sid, "A recovery source outside frozen TRAIN")
        original = {"path": source["image"], "sha256": source["image_sha256"]}
        annotation = {"path": source["annotation"], "sha256": source["annotation_sha256"]}
        hash_file(Path(annotation["path"]), annotation["sha256"])
        im = open_verified(original)
        shapes = evidence["source_info"][sid]["shapes"]
        g3.require(im.size == tuple(source["size"]), "Original dimensions changed")
        overlay = all_polygon_overlay(im, source, shapes)
        overlay_ref = save_image(output / "review_full392" / "assets" / f"{sid}_all_polygons.png", overlay)
        historical_review = a_items[sid]["review_id"] if sid in a_items else holds[sid]["version_reason_sets"][0]["review_id"]
        if sid in holds:
            historic = holds[sid]
            historical_review = next(x["review_id"] for x in historic["version_reason_sets"]
                                     if x["variant_id"] == historic["diagnostic_variant"])
        variants = []
        for item in sorted(grouped[sid], key=lambda x: (x["record"]["review_id"] != historical_review,
                                                       x["human"]["decision"] != "PASS", -int(x["record"]["version"][1]))):
            record, audit, human = item["record"], item["audit"], item["human"]
            assets = files_for(item)
            for asset in assets.values():
                g3.require(pinned.get(asset["path"]) == asset["sha256"], "Unpinned asset in version audit")
            valid = audit["assets"].get("valid.png")
            metadata = {key: record[key] for key in ("review_id", "variant_id", "version", "geometry_status")}
            variants.append({**metadata, "assets": assets, "valid": valid, "human": human, "ai": item["ai"],
                             "support": audit["support"], "invalid_pixels": audit["invalid_pixels"],
                             "visible_loss": audit["visible_loss"], "d0_visual": d0_notes.get(record["review_id"])})
        primary = variants[0]
        g3.require(primary["review_id"] == historical_review, "Historical review missing")
        group = ("A9_BOUND" if sid in reusable else "A9_PENDING" if sid in a_items
                 else "HOLD_SHORTLIST" if sid in shortlist else "HOLD_OTHER")
        row = {"source_id": sid, "group_id": source["group_id"], "group": group, "ordinal": ordinal_by_id.get(sid),
               "original": original, "annotation": annotation, "size": source["size"], "polygon": overlay_ref,
               "labels": shape_labels(source, shapes), "boxes": crop_boxes(source, shapes), "target_sheets": [],
               "primary": primary, "variants": variants, "historical_whole_review_ids": [x["record"]["review_id"] for x in whole_rows[sid]],
               "original_reasons": holds.get(sid, {}).get("all_reasons", []),
               "b_candidates": 0, "historical_binding_basis": ""}
        if sid in a_items:
            a_candidate = a_items[sid]
            g3.require(a_candidate["asset"] == primary["assets"]["ALIGNED"]
                       and primary["human"]["aligned_sha256"] == a_candidate["asset"]["sha256"],
                       "Historical approval asset mismatch")
            hash_file(Path(a_candidate["asset"]["path"]), a_candidate["asset"]["sha256"])
            if group == "A9_BOUND":
                g3.require("BOUNDED" not in primary["assets"], "An additional historical primary asset was shown")
                row["historical_binding_basis"] = "EXACT_HUMAN_REVIEW_ID_ALIGNED_SHA256_ONLY_REPAIR_MAIN_VIEW"
            else:
                row["historical_binding_basis"] = "ALIGNED_AND_BOUNDED_SHOWN_NO_ASSET_CHOICE"
        if row["ordinal"] is not None:
            images = {k: open_verified(v) for k, v in primary["assets"].items()}
            images.update({"Original": im, "Polygon": overlay})
            if group == "A9_PENDING":
                row["pending_image_choices_differ"] = not np.array_equal(np.asarray(images["ALIGNED"]), np.asarray(images["BOUNDED"]))
                row["pending_notes_match_ai"] = primary["human"]["notes"].strip() == primary["ai"].get("ai_notes", "").strip()
            evidence_window(row, images, visual_dir)
            shard = row["ordinal"] % 3
            index = row["ordinal"] // 3 // 3
            sheets[shard].append((row, images))
            row["visual_sheet"] = str(visual_dir / f"s{shard}_{index:02}.jpg")
            if len(sheets[shard]) == 3:
                render_full_sheet(sheets.pop(shard), Path(row["visual_sheet"]))
        rows.append(row)
    for tail in sheets.values():
        render_full_sheet(tail, Path(tail[0][0]["visual_sheet"]))
    b_by_source = Counter(x["source_id"] for x in csv_rows(a9 / "registry/b_pair_candidate_manifest.csv"))
    for row in rows:
        row["b_candidates"] = b_by_source[row["source_id"]]
    g3.write_json_exclusive(visual_dir / "cases.json", {"source_set_sha256": source_digest(whole_rows), "rows": rows})
    print("Prepared visual evidence and identity records", len(rows), len(review_order))


def suggested_crop(output, row, variant, observation):
    edge = (variant.get("d0_geometry") or {}).get("edge") or {}
    box = edge.get("proposed_dark_margin_crop_xyxy")
    if not (observation.get("edge_only_confirmed") is True and edge.get("a1_dark_margin_crop_screen_pass") and box):
        return None
    parent = variant["assets"]["ALIGNED"]
    valid = np.asarray(open_verified(variant["valid"]))[:, :, 0] != 0
    x0, y0, x1, y1 = box
    wx, wy = row["size"]
    g3.require(0 <= x0 < x1 <= wx and 0 <= y0 < y1 <= wy and np.all(valid[y0:y1, x0:x1])
               and all(x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1 for bx0, by0, bx1, by1 in row["boxes"].values()),
               "Crop fails valid support or would remove original target context")
    image = open_verified(parent).crop(tuple(box))
    path = output / "review_full392" / "assets" / f'{row["source_id"]}_{parent["sha256"][:16]}_crop.png'
    return {**save_image(path, image), "parent_sha256": parent["sha256"], "crop_box_xyxy": box,
            "parent_to_crop_xy": [-x0, -y0], "size": list(image.size)}


def select_disposition(row, observation, output):
    if row["group"] == "A9_BOUND":
        return REUSABLE, row["primary"], "ALIGNED", None
    if observation is None:
        return "", None, None, None
    sid = row["source_id"]
    versions = {v["review_id"]: v for v in row["variants"]}
    rid, status = observation["review_id"], observation["disposition"]
    kind = observation.get("proposal_asset_type")
    reviewed = set(observation["observed_review_ids"])
    g3.require(observation["source_id"] == sid and rid in reviewed and reviewed <= versions.keys()
               and status in {PROPOSED, DECISION, EXCLUDED} and kind in {*KINDS, "CROP", None}
               and isinstance(observation.get("note"), str) and observation["note"].strip()
               and isinstance(observation.get("asset_observations"), dict), "Invalid visual observation for " + sid)
    variant = versions[rid]
    crop = None
    if kind == "CROP":
        crop = suggested_crop(output, row, variant, observation)
        if crop is None:
            kind, status = "ALIGNED", DECISION
    if status == EXCLUDED:
        kind = None
        available = {v["review_id"] for v in row["variants"] if v["assets"]}
        g3.require(available <= reviewed, "Exclusion fails to check available historical alternatives for " + sid)
    g3.require((status != PROPOSED or kind is not None)
               and (kind is None or kind == "CROP" or kind in variant["assets"]),
               "Visual recommendation has no exact asset for " + sid)
    if status == PROPOSED:
        g3.require(bool(observation["asset_observations"].get("ALIGNED" if kind == "CROP" else kind)),
                   "Recommended asset lacks a specific visual observation for " + sid)
    return status, variant, kind, crop


def verify_bounded_recommendation(a9, row, variant):
    cached = (variant["d0_geometry"] or {}).get("bounded_metrics")
    current = variant["assets"]["BOUNDED"]
    if cached and cached["bounded"] == current:
        hash_file(Path(cached["base"]["path"]), cached["base"]["sha256"])
        return cached
    source_data = a9["sources"][row["source_id"]]
    primary = next(x for x in a9["reviews"] if x["record"]["review_id"] == variant["review_id"])
    rasters = mask_rasters(source_data, a9["source_info"][row["source_id"]]["shapes"])
    im = np.asarray(open_verified(row["original"]))
    aligned = np.asarray(open_verified(variant["assets"]["ALIGNED"]))
    bounded = np.asarray(open_verified(current))
    valid = np.asarray(open_verified(variant["valid"]))[:, :, 0] > 0
    return bounded_metrics(primary["record"], primary["audit"], im, aligned, valid, rasters, bounded)


def web_asset(asset, page, verified):
    path = Path(asset["path"])
    if path not in verified:
        verified[path] = hash_file(path, asset["sha256"])
    g3.require(verified[path] == asset["sha256"], "One image path has conflicting hashes")
    try:
        url = Path(os.path.relpath(path, page)).as_posix()
    except ValueError:
        url = path.resolve().as_uri()
    return {"url": url, "sha256": asset["sha256"]}


def asset_id(review_id, kind, sha):
    return f"{review_id}:{kind}:{sha}"


def display_assets(row, chosen, crop):
    assets = []
    for variant in row["variants"]:
        for kind, location in variant["assets"].items():
            assets.append({"variant_id": variant["variant_id"], "review_id": variant["review_id"],
                           "asset_type": kind, **location, "asset_id": asset_id(variant["review_id"], kind, location["sha256"])})
    if crop:
        assets.append({"variant_id": chosen["variant_id"], "review_id": chosen["review_id"], "asset_type": "CROP",
                       **crop, "asset_id": asset_id(chosen["review_id"], "CROP", crop["sha256"])})
    return assets


def make_review_row(row, main, assets, status, note, page, verified):
    primary = row["primary"]
    historical_type = "ALIGNED" if "ALIGNED" in primary["assets"] else "RAW"
    historical = primary["assets"].get(historical_type)
    historical_id = asset_id(primary["review_id"], historical_type, historical["sha256"]) if historical else None
    decisions = {v["review_id"]: v["human"]["decision"] for v in row["variants"]}
    review_assets = [{**{k: x[k] for k in ("asset_id", "asset_type", "variant_id", "review_id")},
                      **web_asset(x, page, verified),
                      "label": f'{x["asset_type"]} · 旧版本记录 {decisions[x["review_id"]]}'}
                     for x in assets]
    diagnostics = []
    for variant in row["variants"]:
        if variant["valid"]:
            diagnostics.append({**web_asset(variant["valid"], page, verified), "label": f'{variant["variant_id"]} · Valid Mask'})
    crops = []
    if row["boxes"]:
        original_image = open_verified(row["original"])
        selected = [x for x in assets if x["asset_type"] in {"ALIGNED", "BOUNDED", "CROP"}
                    and x["asset_id"] in {historical_id, main.get("asset_id")}]
        selected_images = {x["asset_id"]: open_verified(x) for x in selected}
        for pid, box in row["boxes"].items():
            before = save_image(page / "assets" / f"{pid}_before.png", original_image.crop(box))
            local = {"label": f'{pid} · {row["labels"][pid]}', "before": web_asset(before, page, verified), "assets": {}}
            for x in selected:
                key = x["asset_id"]
                image = selected_images[key]
                target = box
                if x["asset_type"] == "CROP":
                    dx, dy = x["parent_to_crop_xy"]
                    target = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
                if target[2] > image.width or target[3] > image.height:
                    continue
                after = save_image(page / "assets" / f'{pid}_{x["sha256"][:16]}.png', image.crop(target))
                local["assets"][key] = web_asset(after, page, verified)
            crops.append(local)
    reasons = row["original_reasons"] or ([row["historical_binding_basis"]] if row["group"] == "A9_PENDING" else [])
    return {"source_id": row["source_id"], "group_id": row["group_id"], "disposition": status or "VISUAL_REVIEW_INCOMPLETE",
            "original": web_asset(row["original"], page, verified), "polygon": web_asset(row["polygon"], page, verified),
            "historical_asset_id": historical_id, "historical_asset_ambiguous": row["group"] == "A9_PENDING" or (
                "BOUNDED" in primary["assets"] or historical_type == "RAW"),
            "recommended_asset_id": main.get("asset_id"), "assets": review_assets, "crops": crops,
            "original_reasons": reasons, "summary": note, "diagnostics": diagnostics}


def finish(a9, d0, output, allow_partial):
    verify_frozen_a9(a9)
    d0_receipt = load_json(d0 / "delivery_receipt.json")
    d0_geometry_path = d0 / "a_hold_version_asset_geometry.json"
    hash_file(d0_geometry_path, d0_receipt["private_sha256"][d0_geometry_path.name])
    d0_geometry = {x["review_id"]: x for x in load_json(d0_geometry_path)["records"]}
    source_file = load_json(output / "visual_inputs" / "cases.json")
    rows = source_file["rows"]
    exact_sha = source_file["source_set_sha256"]
    g3.require(len(rows) == len({r["source_id"] for r in rows}) == 392
               and source_digest(r["source_id"] for r in rows) == exact_sha, "Recovery source set diverged")
    evidence = load_json(a9 / "audits/verified_evidence_reconciled.json")
    observations = {p.stem: load_json(p) for p in (output / "visual_notes").glob("*.json")}
    active = {r["source_id"] for r in rows if r["group"] != "A9_BOUND"}
    g3.require(observations.keys() <= active and (allow_partial or observations.keys() == active), "Visual review identity coverage is incomplete")
    registry = output / "registry"
    page = output / "review_full392"
    verified, dispositions, recommended, alternates, unresolved, web_rows, coordinates = {}, [], [], [], [], [], []
    by_origin, types, issues = defaultdict(Counter), defaultdict(Counter), Counter()
    for row in rows:
        sid = row["source_id"]
        source = evidence["sources"][sid]
        row["labels"] = shape_labels(source, evidence["source_info"][sid]["shapes"])
        for variant in row["variants"]:
            variant["d0_geometry"] = d0_geometry.get(variant["review_id"])
        row["primary"]["d0_geometry"] = d0_geometry.get(row["primary"]["review_id"])
        observation = observations.get(sid)
        status, variant, kind, crop = select_disposition(row, observation, output)
        assets = display_assets(row, variant, crop)
        chosen = next((x for x in assets if kind and x["review_id"] == variant["review_id"] and x["asset_type"] == kind), None)
        explanation = observation["note"] if observation else (
            "冻结记录的整图人工通过与当时唯一主要修复视图的 Aligned 哈希一致，可沿用该资产。" if status == REUSABLE
            else "图像证据已整理，目视处置尚未完成。")
        bounded = None
        if chosen and chosen["asset_type"] == "BOUNDED":
            bounded = verify_bounded_recommendation(evidence, row, variant)
            if status == PROPOSED and not bounded["all_gt_accounted"]:
                status = DECISION
                explanation += " Bounded 的合成记录不能解释全部原标注，请复核未完全覆盖的目标。"
        if observation and observation.get("proposal_asset_type") == "CROP" and crop is None:
            explanation += " 外围裁剪证据不足，保留原 Aligned 等待选择。"
        if row["original_reasons"]:
            issues.update(row["original_reasons"])
        if row["group"] == "A9_PENDING":
            g3.require(row["pending_image_choices_differ"] and row["pending_notes_match_ai"],
                       "Historical pending pixel semantics changed")
        by_origin[row["group"]][status or "VISUAL_REVIEW_INCOMPLETE"] += 1
        main = {k: chosen[k] for k in ("asset_id", "variant_id", "review_id", "asset_type", "path", "sha256")} if chosen else {}
        by_source = {"source_id": sid, "group_id": row["group_id"]}
        if main:
            recommended.append({**by_source, "disposition": status, **main, "basis": status,
                                "parent_sha256": crop["parent_sha256"] if crop else "", "formal_admission": "false"})
            types[status][chosen["asset_type"]] += 1
        disposition = {**by_source, "starting_group": row["group"], "disposition": status or "VISUAL_REVIEW_INCOMPLETE",
                       "original_sha256": row["original"]["sha256"], "annotation_sha256": row["annotation"]["sha256"],
                       "historical_review_id": row["primary"]["review_id"],
                       "historical_whole_review_ids": json.dumps(row["historical_whole_review_ids"]),
                       "recommended_variant_id": main.get("variant_id", ""), "asset_type": main.get("asset_type", ""),
                       "asset_sha256": main.get("sha256", ""), "original_reasons": json.dumps(row["original_reasons"]),
                       "historical_binding_basis": row["historical_binding_basis"], "visual_note": explanation,
                       "observed_review_ids": json.dumps(observation["observed_review_ids"] if observation else []),
                       "asset_observations": json.dumps(observation["asset_observations"] if observation else {}, ensure_ascii=False),
                       "alternate_observations": json.dumps(observation.get("alternate_observations", {}) if observation else {}, ensure_ascii=False),
                       "bounded_provenance": json.dumps(bounded or {}, ensure_ascii=False),
                       "existing_b_candidates": row["b_candidates"],
                       "local_only_proposed": "true" if observation and observation.get("local_only") else "false",
                       "new_human_confirmation": "false", "formal_admission": "false"}
        dispositions.append(disposition)
        if status not in {REUSABLE, PROPOSED}:
            unresolved.append({k: disposition[k] for k in ("source_id", "group_id", "starting_group", "disposition", "historical_review_id",
                                                           "original_reasons", "visual_note", "existing_b_candidates", "local_only_proposed")})
        for asset in assets:
            if asset["asset_id"] != main.get("asset_id"):
                v = next(v for v in row["variants"] if v["review_id"] == asset["review_id"])
                alternates.append({**by_source, **{k: asset[k] for k in ("asset_id", "variant_id", "review_id", "asset_type", "path", "sha256")},
                                   "historical_variant_decision": v["human"]["decision"], "pixel_specific_human_pass": "false",
                                   "visual_observation": (observation["asset_observations"].get(asset["asset_type"], "") if observation
                                                          and asset["review_id"] == observation["review_id"] else "")})
        if crop:
            coordinates.append({**by_source, **crop, "variant_id": variant["variant_id"], "review_id": variant["review_id"]})
        web_rows.append(make_review_row(row, main, assets, status, explanation, page, verified))
    counts = Counter(x["disposition"] for x in dispositions)
    final_status = COMPLETE if observations.keys() == active else PARTIAL
    g3.require(sum(counts.values()) == 392 and len({r["source_id"] for r in recommended}) == len(recommended)
               and len({r["sha256"] for r in recommended}) == len(recommended)
               and len(verified) > 392 and counts[REUSABLE] == 138, "Recovery output identity does not reconcile")
    batch_id = f"exp14-a9d1-full392-{exact_sha[:12]}"
    g3.write_json_exclusive(registry / "exact_source_set.json", {"source_ids": sorted(r["source_id"] for r in rows),
                             "sha256": exact_sha, "canonical_form": "lexicographically sorted UTF-8 source IDs, LF with final newline"})
    write_csv(registry / "source_392_disposition.csv", dispositions, list(dispositions[0]))
    write_csv(registry / "recommended_asset_manifest.csv", recommended, list(recommended[0]))
    write_csv(registry / "alternate_assets.csv", alternates, list(alternates[0]))
    write_csv(registry / "unresolved_or_excluded.csv", unresolved, list(unresolved[0]) if unresolved else [
              "source_id", "group_id", "starting_group", "disposition", "historical_review_id", "original_reasons",
              "visual_note", "existing_b_candidates", "local_only_proposed"])
    g3.write_json_exclusive(registry / "derived_crop_provenance.json", coordinates)
    summary = {"status": final_status, "batch_id": batch_id, "source_set_sha256": exact_sha, "source_count": 392,
               "starting_groups_to_disposition": by_origin, "dispositions": counts, "recommended_asset_type_counts": types,
               "distinct_frozen_source_groups": len({r["group_id"] for r in rows}),
               "observed_distinct_review_ids": len({rid for n in observations.values() for rid in n["observed_review_ids"]}),
               "historical_human_asset_reused": counts[REUSABLE], "new_ai_asset_recommendations": counts[PROPOSED],
               "outstanding_visual_reviews": len(active) - len(observations),
               "image_review_notes_completed": len(observations), "image_review_required": len(active),
               "new_human_confirmation": 0, "formal_admissions": 0, "derived_candidate_crop_count": len(coordinates),
               "historical_whole_pass_review_rows": 393, "original_a9_hold_reason_counts": issues,
               "new_api_calls": 0, "new_registration": 0, "new_api_generated_images": 0, "new_training": 0, "new_b_pilots": 0,
               "verified_view_asset_file_count": len(verified)}
    g3.write_json_exclusive(output / "audits" / "recovery_summary.json", summary)
    template = Path(__file__).with_name("full392_recovery.html").read_text(encoding="utf-8")
    bundle = json.dumps({"batch_id": batch_id, "exact_set_sha256": exact_sha, "rows": web_rows}, ensure_ascii=False)
    g3.write_exclusive(page / "index.html", template.replace("__BUNDLE__", bundle.replace("<", "\\u003c")).encode("utf-8"))
    verify_frozen_a9(a9)
    for name, sha in d0_receipt["private_sha256"].items():
        hash_file(d0 / name, sha)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "finish"))
    parser.add_argument("--a9", required=True, type=Path)
    parser.add_argument("--d0", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    paths = tuple(path.resolve() for path in (args.a9, args.d0, args.output))
    if args.action == "prepare":
        prepare(*paths)
    else:
        finish(*paths, args.allow_partial)


if __name__ == "__main__":
    main()
