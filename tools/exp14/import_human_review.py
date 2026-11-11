"""Import a private Exp14 review export into candidate plans without training data."""

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from tools.exp14.audit_admission import (
    TRAIN_DIR, read_json, read_rows, require, sha256_file, under, verify_train, write_text,
)
from tools.exp14.label_plan import (
    A_ELIGIBLE, B_ELIGIBLE, CLASSES, HOLD, PENDING, REMOVED, RETAINED,
    Review, approved_label_plan, polygon_rows,
)


SCHEMA = "exp14_a0_priority_human_review"
A0_DIR = "exp14_a0_data_admission"
INPUTS = {"candidate_manifest.csv", "candidate_review.csv", "label_change_plan.csv"}
IMAGES = ("aligned_output", "hard_composite", "feather_composite", "raw_output")
DECISIONS = {"PASS", HOLD, "REJECT", PENDING}
LABELS = {"DELETE": REMOVED, "KEEP": RETAINED, HOLD: HOLD, PENDING: HOLD}
FLAGS = ("a_whole_image_reviewed", "a_source_coordinates_confirmed", "a_empty_target_confirmed",
         "b_correspondence_confirmed", "b_artifact_shortcuts_reviewed")
RECORD_FIELDS = {
    "variant_id", "source_id", "group_id", "source_image_sha256", "annotation_sha256",
    "final_image_choice", "final_image_sha256", "a_decision", "b_decision", "a_note",
    "b_note", "polygons", "unresolved_checks", *FLAGS,
}
GATE = "CANDIDATES_VERIFIED_PENDING_CHATGPT_DATA_GATE"


def unique_pairs(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate JSON object key")
    return result


def unique_index(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows) and all(result), f"Duplicate or empty {key}")
    return result


def validate_schema(raw):
    export = json.loads(raw, object_pairs_hook=unique_pairs)
    require(isinstance(export, dict) and set(export) == {
        "schema", "bundle_id", "review_state", "reviewer", "exported_at", "input_sha256", "records",
    }, "Invalid review export object")
    require(export["schema"] == SCHEMA and export["review_state"] == "PRIVATE_REVIEW_RECORD",
            "Unsupported review schema or state")
    require(isinstance(export["reviewer"], str) and isinstance(export["exported_at"], str)
            and datetime.fromisoformat(export["exported_at"]).tzinfo is not None,
            "Invalid review metadata")
    require(isinstance(export["records"], list), "Review records must be a list")
    for record in export["records"]:
        require(isinstance(record, dict) and set(record) == RECORD_FIELDS, "Invalid review record fields")
        require(all(type(record[k]) is bool for k in FLAGS), "Invalid review flag type")
        require(all(isinstance(record[k], str) for k in RECORD_FIELDS - {*FLAGS, "polygons", "unresolved_checks"}),
                "Invalid review field type")
        require({record["a_decision"], record["b_decision"]} <= DECISIONS, "Unknown A/B decision")
        require(isinstance(record["unresolved_checks"], list)
                and all(isinstance(x, str) for x in record["unresolved_checks"]), "Invalid exported check list")
        require(isinstance(record["polygons"], list), "Invalid polygon decisions")
        for polygon in record["polygons"]:
            require(isinstance(polygon, dict) and set(polygon) == {"polygon_id", "decision", "note"}
                    and all(isinstance(v, str) for v in polygon.values())
                    and polygon["decision"] in LABELS, "Invalid polygon decision")
        unique_index(record["polygons"], "polygon_id")
    unique_index(export["records"], "variant_id")
    return export


def read_page_bundle(page):
    match = re.search(r'<script type="application/json" id="bundle">(.*?)</script>',
                      page.read_text(encoding="utf-8"), re.S)
    require(match is not None, "Missing HTML review bundle")
    return json.loads(match[1], object_pairs_hook=unique_pairs)


def verify_bundle(export, page, input_hashes, expected_ids):
    require(set(input_hashes) == INPUTS and page["input_sha256"] == export["input_sha256"] == input_hashes,
            "Review inputs do not match frozen A0 files")
    entries = unique_index(page["variants"], "variant_id")
    order = list(entries)
    require(set(order) == expected_ids and [r["variant_id"] for r in export["records"]] == order,
            "Review variant set or ordering differs from the priority bundle")
    identity = hashlib.sha256(json.dumps({"inputs": input_hashes, "order": order}, sort_keys=True).encode()).hexdigest()
    require(export["bundle_id"] == page["bundle"] == identity, "Bundle ID does not match frozen inputs")
    return entries


def expected_priority(manifest, reviews):
    require(manifest.keys() == reviews.keys(), "A0 review and manifest sets differ")
    shortlist = {key for key, row in reviews.items()
                 if "C1_VISUAL_GEOMETRY_SHORTLIST" in row["priority_flags"].split("|")}
    special = {key for key, row in reviews.items()
               if {"V2_BOUNDARY_AND_SEAM_REVIEW", "FULL_TARGET_COVERAGE_INCOMPLETE"}
               & set(row["priority_flags"].split("|"))}
    require(len(manifest) == 39 and len(shortlist) == 23 and len(shortlist | special) == 24,
            "A0 priority counts differ")
    return shortlist | special


def size_wh(value):
    width, height = map(int, value.split("x"))
    require(min(width, height) > 0, "Invalid image dimensions")
    return width, height


def trusted_path(root, name):
    path = Path(name).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Asset outside private root")
    return path


def verified_file(root, name, digest):
    path = trusted_path(root, name)
    require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)) and sha256_file(path) == digest,
            "Actual file SHA256 mismatch")
    return path


@dataclass(frozen=True)
class FinalImage:
    choice: str
    path: Path
    sha256: str
    size: tuple[int, int]


def final_image(root, row, review, entry):
    choice = review["final_image_choice"]
    require(choice in IMAGES, "FINAL_IMAGE_CHOICE_MISSING_OR_UNSUPPORTED")
    image = entry["images"].get(choice)
    require(isinstance(image, dict) and set(image) == {"sha256", "size"}, "FINAL_IMAGE_MISSING_IN_HTML_BUNDLE")
    size_text = row["raw_size_wh"] if choice == "raw_output" else row["source_size_wh"]
    digest = row.get(f"{choice}_sha256")
    require(bool(digest) and image["sha256"] == digest == review["final_image_sha256"]
            and image["size"] == size_text, "FINAL_IMAGE_MANIFEST_BUNDLE_REVIEW_MISMATCH")
    require(choice == "raw_output" or row["geometry_status"] == "GEOMETRY_CANDIDATE",
            "SOURCE_GRID_IMAGE_WITHOUT_REGISTERED_GEOMETRY")
    path = verified_file(root, row[f"{choice}_path"], digest)
    with Image.open(path) as decoded:
        require(decoded.size == size_wh(size_text), "FINAL_IMAGE_DIMENSIONS_MISMATCH")
        decoded.verify()
    return FinalImage(choice, path, digest, size_wh(size_text))


def source_annotations(root, row, train):
    require(row["split"] == train["split"] == "train" and row["group_id"] == train["group_id"]
            and row["source_size_wh"] == f"{train['image_width']}x{train['image_height']}"
            and row["source_sha256"] == train["export_image_sha256"]
            and row["annotation_sha256"] == train["export_json_sha256"], "FROZEN_TRAIN_IDENTITY_MISMATCH")
    train_root = under(root, TRAIN_DIR)
    source = trusted_path(root, row["source_path"])
    annotation = trusted_path(root, row["annotation_path"])
    require(source == under(train_root, train["export_image_path"])
            and annotation == under(train_root, train["export_json_path"]), "FROZEN_TRAIN_PATH_MISMATCH")
    size = size_wh(row["source_size_wh"])
    doc = read_json(annotation)
    require((doc["imageWidth"], doc["imageHeight"]) == size, "ANNOTATION_DIMENSIONS_MISMATCH")
    shapes = doc["shapes"]
    labels = polygon_rows(row["source_id"], shapes, size)
    require(set(row["all_polygon_ids"].split("|")) == labels.keys(), "A0_ORIGINAL_POLYGON_SET_MISMATCH")
    return labels, dict(zip(labels, shapes, strict=True)), size


def verify_record_identity(row, review, entry, labels, label_plan):
    require(all(row[k] == review[k] == entry[k] for k in ("variant_id", "source_id", "group_id"))
            and row["source_sha256"] == review["source_image_sha256"] == entry["source_image_sha256"]
            and row["annotation_sha256"] == review["annotation_sha256"] == entry["annotation_sha256"]
            and row["source_size_wh"] == entry["source_size"], "REVIEW_SOURCE_IDENTITY_MISMATCH")
    decisions = unique_index(review["polygons"], "polygon_id")
    require(decisions.keys() == labels.keys() == label_plan.keys() == set(entry["polygon_ids"])
            and len(entry["polygon_ids"]) == len(labels), "REVIEW_POLYGON_SET_MISMATCH")
    require(all(p["source_id"] == row["source_id"] and p["group_id"] == row["group_id"]
                and p["class_id"] == str(labels[pid][0]) and p["class"] == CLASSES[labels[pid][0]]
                for pid, p in label_plan.items()), "A0_LABEL_CLASS_OR_GROUP_MISMATCH")
    return {pid: p["decision"] for pid, p in decisions.items()}


def omitted_polygon_pixels(support, shape):
    polygon = Image.new("L", support.size)
    ImageDraw.Draw(polygon).polygon([tuple(point) for point in shape["points"]], fill=255)
    missing = ImageChops.multiply(polygon, ImageChops.invert(support))
    return sum(missing.histogram()[1:])


def geometry_support(root, row, size, shapes, ids):
    path = verified_file(root, row["valid_mask_path"], row["valid_mask_sha256"])
    with Image.open(path) as image:
        support = image.convert("L")
    require(support.size == size and not any(support.histogram()[1:255]), "INVALID_GEOMETRY_SUPPORT_MASK")
    return {pid: omitted_polygon_pixels(support, shapes[pid]) for pid in ids}


def evaluate_routes(review, labels, decisions, image, source_size, omitted, selected_ids, support_error=""):
    """Recompute approval using the actual reviewed image and the A0 label-plan gate."""
    a_reasons, b_reasons = [], []
    removed = [pid for pid, decision in decisions.items() if decision == "DELETE"]
    kept = [pid for pid, decision in decisions.items() if decision == "KEEP"]
    confirmed = {pid: LABELS[decision] for pid, decision in decisions.items()}
    evidence = Review(
        image_sha256=review["final_image_sha256"],
        a_status=A_ELIGIBLE if review["a_decision"] == "PASS" else review["a_decision"],
        b_status=B_ELIGIBLE if review["b_decision"] == "PASS" else review["b_decision"],
        whole_image_verified=review["a_whole_image_reviewed"],
        source_coordinates_verified=review["a_source_coordinates_confirmed"],
        empty_confirmed=review["a_empty_target_confirmed"],
        region_correspondence_verified=review["b_correspondence_confirmed"],
        artifact_shortcuts_reviewed=review["b_artifact_shortcuts_reviewed"],
    )
    if review["a_decision"] == "PASS":
        if image is None:
            a_reasons.append("FINAL_IMAGE_NOT_VERIFIED")
        if not evidence.whole_image_verified:
            a_reasons.append("WHOLE_IMAGE_CHECK_MISSING")
        if set(confirmed.values()) - {REMOVED, RETAINED}:
            a_reasons.append("POLYGON_DECISION_UNRESOLVED")
        if not removed:
            a_reasons.append("NO_CONFIRMED_REMOVAL")
        if kept and (image is None or image.size != source_size or not evidence.source_coordinates_verified):
            a_reasons.append("RETAINED_POLYGON_COORDINATES_UNVERIFIED")
        if kept and image and image.choice == "raw_output":
            a_reasons.append("RAW_RETAINED_LABELS_NEED_INDEPENDENT_COORDINATE_BINDING")
        if kept and image and image.choice == "aligned_output" and (support_error or any(omitted.get(p, 0) for p in kept)):
            a_reasons.append("ALIGNED_RETAINED_POLYGON_SUPPORT_INCOMPLETE")
        if evidence.empty_confirmed != all(d == "DELETE" for d in decisions.values()):
            a_reasons.append("EMPTY_TARGET_CONFIRMATION_INCONSISTENT")
        if not a_reasons:
            approved_label_plan(labels, confirmed, evidence, final_sha256=image.sha256,
                                source_size=source_size, final_size=image.size, route="A")
    a_pass = review["a_decision"] == "PASS" and not a_reasons
    if review["b_decision"] == "PASS":
        if not a_pass:
            b_reasons.append("A_NOT_VERIFIED")
        if not evidence.region_correspondence_verified:
            b_reasons.append("REGION_CORRESPONDENCE_OR_STATE_UNCONFIRMED")
        if not evidence.artifact_shortcuts_reviewed:
            b_reasons.append("ARTIFACT_SHORTCUTS_UNREVIEWED")
        if image and image.choice == "raw_output":
            b_reasons.append("RAW_REGION_MAPPING_UNVERIFIED")
        if removed and not set(removed) <= selected_ids:
            b_reasons.append("REMOVED_POLYGON_OUTSIDE_REVIEWED_EDIT_TARGET")
        if support_error and image and image.choice != "raw_output":
            b_reasons.append("VALID_REGION_SUPPORT_UNVERIFIED")
        elif any(omitted.get(pid, 0) for pid in removed):
            b_reasons.append("REMOVED_POLYGON_VALID_SUPPORT_INCOMPLETE")
        if not b_reasons:
            approved_label_plan(labels, confirmed, evidence, final_sha256=image.sha256,
                                source_size=source_size, final_size=image.size, route="B")
    return a_reasons, b_reasons


def disposition(human, reasons):
    if human == "PASS":
        return HOLD if reasons else "VERIFIED_CANDIDATE"
    return human


def block_duplicate_sources(details):
    counts = Counter(d["source_id"] for d in details if d["a_status"] == "VERIFIED_CANDIDATE")
    for d in details:
        if d["a_status"] == "VERIFIED_CANDIDATE" and counts[d["source_id"]] > 1:
            d["a_reasons"].append("MULTIPLE_VERIFIED_VARIANTS_NEED_SOURCE_SELECTION")
            d["a_status"] = HOLD
            if d["b_human_decision"] == "PASS":
                d["b_reasons"].append("A_NOT_VERIFIED")
                d["b_status"] = HOLD


def candidate_row(row, detail, image, removed, kept):
    return {
        "variant_id": row["variant_id"], "source_id": row["source_id"], "group_id": row["group_id"],
        "version": row["version"], "split": row["split"], "repair_class": row["repair_class"],
        "final_image_choice": image.choice, "final_image_path": str(image.path),
        "final_image_sha256": image.sha256, "final_image_size_wh": "x".join(map(str, image.size)),
        "source_image_path": row["source_path"], "source_image_sha256": row["source_sha256"],
        "annotation_path": row["annotation_path"], "annotation_sha256": row["annotation_sha256"],
        "delete_polygon_ids": removed, "keep_polygon_ids": kept,
        "label_plan": "KEEP_CONFIRMED_INSTANCES" if kept else "EMPTY_TARGET_CONFIRMED",
        "review_sha256": detail["review_sha256"], "bundle_id": detail["bundle_id"],
        "release_status": "PENDING_CHATGPT_DATA_GATE",
    }


def audit(root, a0, raw, page):
    export = validate_schema(raw)
    review_sha = hashlib.sha256(raw).hexdigest()
    inputs = {name: sha256_file(under(a0, name)) for name in INPUTS}
    manifest = unique_index(read_rows(under(a0, "candidate_manifest.csv")), "variant_id")
    a0_reviews = unique_index(read_rows(under(a0, "candidate_review.csv")), "variant_id")
    entries = verify_bundle(export, page, inputs, expected_priority(manifest, a0_reviews))
    a0_plan = read_rows(under(a0, "label_change_plan.csv"))
    require(sum(len(r["polygons"]) for r in export["records"]) == 27
            and len(a0_plan) == 44, "A0 Polygon count differs")
    _, train, _ = verify_train(root)
    details, decisions_out = [], []
    verified = {}
    by_variant = unique_index(export["records"], "variant_id")
    for key, row in manifest.items():
        identity = {k: row[k] for k in ("variant_id", "source_id", "group_id", "repair_class")}
        detail = {**identity, "review_sha256": review_sha, "bundle_id": export["bundle_id"],
                  "review_scope": "PRIORITY" if key in by_variant else "OUTSIDE_PRIORITY",
                  "a_human_decision": PENDING, "b_human_decision": PENDING, "a_status": PENDING,
                  "b_status": PENDING, "a_reasons": [], "b_reasons": [], "conflicts": [],
                  "final_image_choice": "", "final_image_sha256": "", "image_verified": False,
                  "html_unresolved_checks": [], "label_decisions": {}, "a_note": "", "b_note": "",
                  "unsupported_pixels_by_polygon": {}, "human_flags": {}}
        details.append(detail)
        if key not in by_variant:
            continue
        user = by_variant[key]
        for key_name in ("a_note", "b_note", "final_image_choice", "final_image_sha256"):
            detail[key_name] = user[key_name]
        detail.update(a_human_decision=user["a_decision"], b_human_decision=user["b_decision"],
                      html_unresolved_checks=user["unresolved_checks"], human_flags={k: user[k] for k in FLAGS},
                      label_decisions={p["polygon_id"]: p["decision"] for p in user["polygons"]})
        try:
            labels, shapes, size = source_annotations(root, row, train[row["source_id"]])
            plan = unique_index([p for p in a0_plan if p["variant_id"] == key], "polygon_id")
            decisions = verify_record_identity(row, user, entries[key], labels, plan)
        except (ValueError, KeyError, OSError) as exc:
            detail["conflicts"].append(f"REVIEW_IDENTITY: {exc}")
            for route in ("a", "b"):
                detail[f"{route}_reasons"].append("REVIEW_IDENTITY_MISMATCH")
                detail[f"{route}_status"] = disposition(user[f"{route}_decision"], detail[f"{route}_reasons"])
            continue
        image = None
        try:
            image = final_image(root, row, user, entries[key])
        except (ValueError, KeyError, OSError) as exc:
            detail["conflicts"].append(f"FINAL_IMAGE: {exc}")
        detail["image_verified"] = image is not None
        omitted, support_error = {}, ""
        if image and image.choice != "raw_output":
            try:
                omitted = geometry_support(root, row, size, shapes, labels)
            except (ValueError, KeyError, OSError) as exc:
                support_error = str(exc)
                detail["conflicts"].append(f"GEOMETRY_SUPPORT: {exc}")
        detail["unsupported_pixels_by_polygon"] = {pid: n for pid, n in omitted.items() if n}
        a_reasons, b_reasons = evaluate_routes(user, labels, decisions, image, size, omitted,
                                              set(row["selected_polygon_ids"].split("|")), support_error)
        for route, reasons in (("a", a_reasons), ("b", b_reasons)):
            detail[f"{route}_reasons"] = reasons
            detail[f"{route}_status"] = disposition(user[f"{route}_decision"], reasons)
        for polygon in user["polygons"]:
            pid = polygon["polygon_id"]
            decisions_out.append({**identity, **polygon, "class_id": labels[pid][0],
                                  "class": CLASSES[labels[pid][0]], "final_image_sha256": user["final_image_sha256"]})
        if detail["a_status"] == "VERIFIED_CANDIDATE":
            removed = [pid for pid, action in decisions.items() if action == "DELETE"]
            kept = [pid for pid, action in decisions.items() if action == "KEEP"]
            verified[key] = candidate_row(row, detail, image, removed, kept)
    block_duplicate_sources(details)
    a_rows = [verified[d["variant_id"]] for d in details if d["a_status"] == "VERIFIED_CANDIDATE"]
    b_rows = [{**verified[d["variant_id"]], "region_polygon_ids": verified[d["variant_id"]]["delete_polygon_ids"],
               "region_coordinate_grid": "ORIGINAL_SOURCE_PIXELS",
               "valid_support_mask_path": manifest[d["variant_id"]]["valid_mask_path"],
               "valid_support_mask_sha256": manifest[d["variant_id"]]["valid_mask_sha256"]}
              for d in details if d["b_status"] == "VERIFIED_CANDIDATE"]
    return details, decisions_out, a_rows, b_rows, export, review_sha, inputs


def summarize(details, labels, a_rows, b_rows):
    def route_summary(route, candidates):
        reviewed = [d for d in details if d["review_scope"] == "PRIORITY"]
        status = ("PASS", HOLD, "REJECT", PENDING)
        return {
            "human": {s: sum(d[f"{route}_human_decision"] == s for d in reviewed) for s in status},
            "verified": {s: sum(d[f"{route}_status"] == s for d in reviewed)
                         for s in ("VERIFIED_CANDIDATE", HOLD, "REJECT", PENDING)},
            "pending_including_outside_priority": sum(d[f"{route}_status"] == PENDING for d in details),
            "image_use": {name: sum(c["final_image_choice"] == name for c in candidates) for name in IMAGES},
            "independent_sources": len({c["source_id"] for c in candidates}),
            "source_groups": len({c["group_id"] for c in candidates}),
            "by_class": {name: {"sources": len({c["source_id"] for c in candidates if c["repair_class"] == name}),
                               "variants": sum(c["repair_class"] == name for c in candidates)}
                         for name in CLASSES},
            "reason_counts": dict(Counter(reason for d in reviewed for reason in d[f"{route}_reasons"])),
        }
    submitted_labels = Counter(value for d in details for value in d["label_decisions"].values())
    return {
        "experiment": "Exp14-A1", "gate": GATE, "training_data_released": False,
        "historical_trusted_pair_pass": 0,
        "coverage": {"a0_variants": len(details),
                     "priority_variants": sum(d["review_scope"] == "PRIORITY" for d in details),
                     "priority_sources": len({d["source_id"] for d in details if d["review_scope"] == "PRIORITY"}),
                     "outside_priority_pending": sum(d["review_scope"] == "OUTSIDE_PRIORITY" for d in details),
                     "polygon_decisions_verified": len(labels),
                     "final_images_verified": sum(d["image_verified"] for d in details)},
        "A": route_summary("a", a_rows),
        "B": route_summary("b", b_rows),
        "submitted_image_choices": {name: sum(d["final_image_choice"] == name for d in details) for name in IMAGES},
        "submitted_polygon_decisions": {s: submitted_labels[s] for s in LABELS},
        "a_candidate_deleted_polygons": sum(len(c["delete_polygon_ids"]) for c in a_rows),
        "a_candidate_kept_polygons": sum(len(c["keep_polygon_ids"]) for c in a_rows),
        "a_empty_targets": sum(c["label_plan"] == "EMPTY_TARGET_CONFIRMED" for c in a_rows),
        "a_partial_repairs": sum(c["label_plan"] == "KEEP_CONFIRMED_INSTANCES" for c in a_rows),
        "b_pair_candidates": len(b_rows),
        "b_removed_region_candidates": sum(len(c["region_polygon_ids"]) for c in b_rows),
        "a_verified_b_not_verified": len(a_rows) - len(b_rows),
        "identity_conflict_variants": sum(bool(d["conflicts"]) for d in details),
        "priority_variants_needing_followup": sum(d["review_scope"] == "PRIORITY"
            and (d["a_status"] in {HOLD, PENDING} or d["b_status"] in {HOLD, PENDING}) for d in details),
    }


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def write_outputs(root, review_path, page_path, result):
    details, labels, a_rows, b_rows, export, review_sha, inputs = result
    summary = summarize(details, labels, a_rows, b_rows)
    review_date = datetime.fromisoformat(export["exported_at"]).strftime("%Y%m%d")
    folder = Path(tempfile.mkdtemp(prefix=f"exp14_a1_review_import_{review_date}_", dir=root))
    pending = [d for d in details if d["b_status"] != "VERIFIED_CANDIDATE"]
    a_fields = ("variant_id", "source_id", "group_id", "version", "split", "repair_class",
                "final_image_choice", "final_image_path", "final_image_sha256", "final_image_size_wh",
                "source_image_path", "source_image_sha256", "annotation_path", "annotation_sha256",
                "delete_polygon_ids", "keep_polygon_ids", "label_plan", "review_sha256", "bundle_id", "release_status")
    write_csv(folder / "review_import_details.csv", details, list(details[0]))
    write_csv(folder / "label_decision_summary.csv", labels,
              ["variant_id", "source_id", "group_id", "repair_class", "polygon_id", "decision",
               "note", "class_id", "class", "final_image_sha256"])
    write_csv(folder / "train_a_candidate_manifest.csv", a_rows, a_fields)
    write_csv(folder / "train_b_pair_candidate_manifest.csv", b_rows,
              (*a_fields, "region_polygon_ids", "region_coordinate_grid", "valid_support_mask_path",
               "valid_support_mask_sha256"))
    write_csv(folder / "pending_review_manifest.csv", pending, list(details[0]))
    audit_record = {
        "review_file": str(review_path), "review_sha256": review_sha, "html_bundle_file": str(page_path),
        "html_sha256": sha256_file(page_path), "bundle_id": export["bundle_id"],
        "exported_at": export["exported_at"], "schema": export["schema"], "input_sha256": inputs,
        "audit_summary": summary,
    }
    write_text(folder / "review_import_audit.json", json.dumps(audit_record, ensure_ascii=False, indent=2) + "\n")
    write_text(folder / "exp14_a1_gate_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return folder, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve(strict=True)
    a0 = under(root, A0_DIR)
    review_path, page_path = [trusted_path(a0, p) for p in (args.review, args.html)]
    raw = review_path.read_bytes()
    page_sha = sha256_file(page_path)
    result = audit(root, a0, raw, read_page_bundle(page_path))
    require(sha256_file(review_path) == result[5] and sha256_file(page_path) == page_sha,
            "Review inputs changed during import")
    for name, digest in result[6].items():
        require(sha256_file(under(a0, name)) == digest, "A0 input changed during import")
    folder, summary = write_outputs(root, review_path, page_path, result)
    print(folder)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
