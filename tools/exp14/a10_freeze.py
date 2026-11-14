"""Freeze the asset-specific A10 human selection and publish read-only data entry checks."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tools.exp13_cloud.g3.g3_batch_executor import require, write_exclusive, write_json_exclusive
from tools.exp14.a10_dataset_entries import EMPTY_SHA256, build_dataset_entries
from tools.exp14.a10_selection import FINAL, import_selections
from tools.exp14.admission_diagnosis_a import mask_rasters, bounded_metrics
from tools.exp14.freeze_core import copy_verified
from tools.exp14.full392_recovery import crop_boxes, csv_rows, write_csv
from tools.exp14.prepare_large_scale_repair import hash_file, load_json

READY = "EXP14_A10_A_DATASET_FROZEN_READY_FOR_EXPERIMENT_GATE"
EXCEPTIONS = "EXP14_A10_A_DATASET_FROZEN_WITH_EXCEPTIONS"
LABEL_POLICY = "EMPTY_SEVEN_CLASS_INSTANCES_FROM_PIXEL_SPECIFIC_WHOLE_IMAGE_HUMAN_DECISION"
EXCEPTION_CODES = {
    "RESIDUAL_ANNOTATED_DEFECT_EXACT_ASSET", "UNEDITED_ANNOTATED_DEFECT_EXACT_ASSET",
    "VISIBLE_DEFECT_LIKE_STRUCTURE_AT_ANNOTATED_SITE",
}


def load_pixels(path, expected_sha):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha, "Image bytes differ from the selected hash")
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        require(img.format in {"PNG", "JPEG"}, "Unexpected training image format")
        return img.convert("RGB"), img.format


def crop_proof(row, image, a9):
    record = row["crop"]
    aligned = row["variant"]["assets"]["ALIGNED"]
    mask = row["variant"]["valid"]
    parent, _ = load_pixels(aligned["path"], aligned["sha256"])
    valid, _ = load_pixels(mask["path"], mask["sha256"])
    x0, y0, x1, y1 = record["crop_box_xyxy"]
    require(record["parent_sha256"] == aligned["sha256"] and record["parent_to_crop_xy"] == [-x0, -y0]
            and record["sha256"] == row["sha256"] and 0 <= x0 < x1 <= parent.width and 0 <= y0 < y1 <= parent.height
            and image.size == (x1 - x0, y1 - y0) == tuple(record["size"])
            and image.tobytes() == parent.crop((x0, y0, x1, y1)).tobytes(), "D1 crop bytes or coordinate derivation differ")
    valid = np.asarray(valid)[:, :, 0] > 0
    src = a9["sources"][row["source_id"]]
    boxes = crop_boxes(src, a9["source_info"][row["source_id"]]["shapes"])
    require(valid.shape == (parent.height, parent.width) and np.all(valid[y0:y1, x0:x1])
            and all(x0 <= ax0 < ax1 <= x1 and y0 <= ay0 < ay1 <= y1 for ax0, ay0, ax1, ay1 in boxes.values()),
            "D1 crop omits original target context or includes invalid pixels")
    return {"parent": aligned, "valid_mask": mask, "crop_box_xyxy": record["crop_box_xyxy"],
            "parent_to_crop_xy": record["parent_to_crop_xy"], "full_target_contexts_retained": len(boxes)}


def bounded_proof(row, image, a9, a9_reviews):
    src = a9["sources"][row["source_id"]]
    item = a9_reviews[row["review_id"]]
    original, _ = load_pixels(row["original"]["path"], row["original"]["sha256"])
    aligned = row["variant"]["assets"]["ALIGNED"]
    mask = row["variant"]["valid"]
    aligned_image, _ = load_pixels(aligned["path"], aligned["sha256"])
    valid, _ = load_pixels(mask["path"], mask["sha256"])
    result = bounded_metrics(item["record"], item["audit"], np.asarray(original), np.asarray(aligned_image),
                             np.asarray(valid)[:, :, 0] > 0,
                             mask_rasters(src, a9["source_info"][row["source_id"]]["shapes"]), np.asarray(image))
    require(result["bounded"]["sha256"] == row["sha256"] and result["all_gt_accounted"],
            "Bounded provenance does not cover all immutable targets")
    return {**result, "edit_mask": {"path": item["audit"]["origin_paths"]["mask"], "sha256": item["record"]["mask_sha256"]},
            "valid_mask": mask, "request_sha256": item["record"]["request_sha256"]}


def bind_exceptions(doc, receipt, selected, a9, d1, review_root):
    require(doc["selected_input_sha256"] == receipt["input_csv_sha256"], "Exception review concerns another human input")
    by_id = {r["source_id"]: r for r in selected}
    result = {}
    for exception in doc["entries"]:
        sid = exception["source_id"]
        row = by_id[sid]
        identities = [exception["asset_sha256"] == row["sha256"],
                      exception["original_sha256"] == row["original"]["sha256"],
                      exception["annotation_sha256"] == row["annotation"]["sha256"],
                      Path(exception["selected_asset_path"]).resolve() == Path(row["path"]).resolve()]
        require(sid not in result and all(identities) and row["decision_origin"] == FINAL
                and all(exception[k] == row[k] for k in ("review_id", "variant_id", "asset_id", "asset_type"))
                and exception["reason_code"] in EXCEPTION_CODES and exception["reason"].strip()
                and exception["polygon_ids"] and set(exception["polygon_ids"]) <= set(a9["sources"][sid]["all_polygon_ids"]),
                "Exception is not bound to exact human-selected pixels and original professional annotation")
        roots = (review_root, d1)
        require(len(exception["evidence"]) >= 2, "Exception lacks independent evidence references")
        for item in exception["evidence"]:
            evidence = Path(item["path"]).resolve()
            require(any(evidence.is_relative_to(root) for root in roots), "Exception evidence escapes its reviewed roots")
            hash_file(evidence, item["sha256"])
        result[sid] = exception
    return result


def source_entry(row, output, width, height, suffix, specialty):
    name = f'exp14_a10_{row["source_id"]}_{row["sha256"][:16]}'
    image = output / "dataset" / "images" / "a10" / (name + suffix)
    label = output / "dataset" / "labels" / "a10" / (name + ".txt")
    image.parent.mkdir(parents=True, exist_ok=True)
    copy_verified(Path(row["path"]), image, row["sha256"])
    write_exclusive(label, b"")
    require(hash_file(image) == row["sha256"] and hash_file(label) == EMPTY_SHA256, "Published bytes differ")
    keep = ("source_id", "group_id", "split", "asset_id", "review_id", "variant_id", "asset_type",
            "decision_origin", "recorded_utc", "action", "human_input_sha256", "historical_input_name",
            "historical_review_id", "human_reason", "human_notes", "advisory_state", "selected_note", "old_visual_note",
            "existing_b_candidates", "original_reasons", "original_classes")
    return {**{k: row[k] for k in keep}, "source_image_path": row["original"]["path"],
            "source_image_sha256": row["original"]["sha256"], "annotation_path": row["annotation"]["path"],
            "annotation_sha256": row["annotation"]["sha256"], "selected_asset_id": row["asset_id"],
            "selected_asset_path": row["path"], "selected_asset_sha256": row["sha256"],
            "published_path": image.relative_to(output).as_posix(), "published_sha256": row["sha256"],
            "width": width, "height": height, "label_path": label.relative_to(output).as_posix(),
            "label_sha256": EMPTY_SHA256, "label_policy": LABEL_POLICY, "specialty_provenance": specialty,
            "admission_state": "FROZEN_A_TRAIN_EMPTY_LABEL"}


def flatten(rows, fields):
    return [{k: (json.dumps(row[k], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                 if isinstance(row.get(k), (list, dict)) else row.get(k, "")) for k in fields} for row in rows]


def csv_table(path, rows, fields):
    write_csv(path, flatten(rows, fields), fields)


def b_references(a9, frozen, selected):
    b_rows = csv_rows(a9 / "registry/b_pair_candidate_manifest.csv")
    by_source = {row["source_id"]: row for row in selected}
    frozen_ids = {r["source_id"] for r in frozen}
    result = []
    for b in b_rows:
        recommended = json.loads(b["recommended"])
        a = by_source.get(b["source_id"])
        result.append({"source_id": b["source_id"], "polygon_id": b["polygon_id"],
                       "b_review_id": recommended["review_id"], "b_variant_id": recommended["variant_id"],
                       "b_aligned_sha256": recommended["aligned"]["sha256"], "b_geometry_status": recommended["geometry_status"],
                       "a_asset_id": a["asset_id"] if a else "", "a_frozen": bool(a and b["source_id"] in frozen_ids)})
    return result


def content_receipt(output):
    """Hash released pixel and label bytes by portable relative path."""
    inventory = load_json(output / "configs" / "train_entry_files.json")
    files = {}
    for row in inventory["baseline"] + inventory["derived"]:
        for path_key, sha_key in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
            name = (Path("dataset") / row[path_key]).as_posix()
            require(name not in files and (output / name).resolve().is_relative_to(output / "dataset"),
                    "Released file is duplicate or outside the dataset")
            files[name] = hash_file(output / name, row[sha_key])
    observed = {x.relative_to(output).as_posix() for x in (output / "dataset").rglob("*") if x.is_file()}
    require(files.keys() == observed, "Released dataset includes a missing or unexpected file")
    hash_map = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"canonical_encoding": "UTF-8 canonical JSON object, lexicographic relative paths, no extra whitespace",
            "sha256": hashlib.sha256(hash_map).hexdigest(), "file_count": len(files), "files": files}


def stratification(rows):
    by_asset = {}
    for kind in sorted({r["asset_type"] for r in rows}):
        group = [r for r in rows if r["asset_type"] == kind]
        by_asset[kind] = {"images": len(group), "source_groups": len({r["group_id"] for r in group}),
                          "original_source_classes": dict(Counter(c for r in group for c in set(r["original_classes"]))),
                          "historical_risks_nonexclusive": dict(Counter(k for r in group for k in r["original_reasons"]))}
    group_counts = Counter(r["group_id"] for r in rows)
    return {"by_asset_type": by_asset, "frozen_groups_by_image_count": dict(Counter(group_counts.values()))}


def freeze(a9, d1, csv_path, asset_root, output, review):
    output.mkdir(parents=True, exist_ok=True)
    require(set(p.name for p in output.iterdir()) <= {"review_work"}, "Use a fresh A10 output directory")
    imported = import_selections(d1, csv_path, a9)
    summary = imported["audit"]
    selected = imported["selected"]
    evidence = load_json(a9 / "audits/verified_evidence_reconciled.json")
    a9_reviews = {r["record"]["review_id"]: r for r in evidence["reviews"]}
    review_doc = load_json(review)
    conflicts = bind_exceptions(review_doc, summary, selected, evidence, d1, review.parent)
    exceptions = [dict(row) for row in imported["exceptions"]]
    frozen, integrity = [], []
    used_hashes = set()
    for row in selected:
        sid, sha = row["source_id"], row["sha256"]
        try:
            image, image_format = load_pixels(row["path"], sha)
            hash_file(Path(row["original"]["path"]), row["original"]["sha256"])
            hash_file(Path(row["annotation"]["path"]), row["annotation"]["sha256"])
            width, height = image.size
            specialty = {}
            if row["asset_type"] == "CROP":
                specialty = crop_proof(row, image, evidence)
            elif row["asset_type"] == "BOUNDED":
                specialty = bounded_proof(row, image, evidence, a9_reviews)
            require(sha not in used_hashes, "A10 chosen bytes already occur in this source cohort")
        except (OSError, ValueError, KeyError) as error:
            exceptions.append({"source_id": sid, "group_id": row["group_id"], "asset_id": row["asset_id"],
                               "asset_sha256": sha, "reason_code": "SELECTED_FILE_INTEGRITY", "reason": str(error)})
            continue
        state = "EMPTY_LABEL_CONFLICT" if sid in conflicts else "VERIFIED_FOR_FREEZE"
        integrity.append({"source_id": sid, "asset_id": row["asset_id"], "asset_sha256": sha,
                          "asset_type": row["asset_type"], "path": row["path"], "width": width, "height": height,
                          "decoded_format": image_format, "sha256_match": True, "specialty_provenance": specialty, "status": state})
        used_hashes.add(sha)
        if sid in conflicts:
            exceptions.append({"group_id": row["group_id"], **conflicts[sid]})
            continue
        suffix = ".png" if image_format == "PNG" else ".jpg"
        frozen.append(source_entry(row, output, width, height, suffix, specialty))
    require(len(frozen) + len(exceptions) == summary["source_count"]
            and len({r["source_id"] for r in frozen + exceptions}) == summary["source_count"], "Freeze dispositions do not reconcile")
    dataset_report = build_dataset_entries(asset_root, output, frozen)
    inputs = output / "inputs"
    original_csv = inputs / "final_human_selection.csv"
    write_exclusive(original_csv, imported["input_bytes"])
    hash_file(original_csv, summary["input_csv_sha256"])
    original_csv.chmod(0o444)
    evidence_copy = inputs / "exact_pixel_label_exceptions.json"
    copy_verified(review, evidence_copy, hash_file(review))
    write_json_exclusive(inputs / "input_hashes.json", {
        "final_human_selection.csv": summary["input_csv_sha256"], "exact_pixel_label_exceptions.json": hash_file(evidence_copy),
        "d1_delivery_receipt_sha256": hash_file(d1 / "audits" / "delivery_receipt.json"),
        "a9_delivery_receipt_sha256": hash_file(a9 / "handoff" / "delivery_receipt.json"),
        "frozen_split_manifest_sha256": dataset_report["frozen_split_sha256"]})
    audits, registry = output / "audits", output / "registry"
    write_json_exclusive(audits / "selection_import_audit.json", summary)
    write_json_exclusive(audits / "historical_138_reuse_receipt.json", imported["history_receipt"])
    write_json_exclusive(audits / "new_254_selection_receipt.json", imported["selection_receipt"])
    write_json_exclusive(audits / "asset_integrity_audit.json", integrity)
    exception_by_id = {r["source_id"]: r for r in exceptions}
    for row in imported["override_rows"]:
        ex = exception_by_id.get(row["source_id"], {})
        row["freeze_resolution"] = "ISOLATED" if ex else "FROZEN"
        row["hard_exception_code"] = ex.get("reason_code", ex.get("reason", ""))
    overrides = imported["override_rows"]
    csv_table(audits / "human_override_trace.csv", overrides, list(overrides[0]))
    write_json_exclusive(registry / "a_frozen_manifest.json", frozen)
    csv_table(registry / "a_frozen_manifest.csv", frozen, list(frozen[0]))
    exception_columns = ["source_id", "group_id", "asset_id", "review_id", "variant_id", "asset_type", "asset_sha256",
                         "reason_code", "reason", "message", "polygon_ids", "evidence", "selected_asset_path", "selection"]
    csv_table(registry / "a_freeze_exceptions.csv", exceptions, exception_columns)
    b_rows = b_references(a9, frozen, selected)
    csv_table(registry / "a_to_existing_b_references.csv", b_rows, list(b_rows[0]))
    labels = {"label_policy": LABEL_POLICY, "explicit_empty_labels": len(frozen), "empty_label_sha256": EMPTY_SHA256,
              "isolated_no_empty_label": len(exceptions), "original_train_labels_preserved": dataset_report["arms"]["e0"]["train_labels"],
              "polygons_copied_to_repairs": 0, "class_names": dataset_report["class_names"], "existing_b_candidates": len(b_rows),
              "new_b_trusted_pairs": 0}
    write_json_exclusive(audits / "label_policy_audit.json", labels)
    write_json_exclusive(audits / "frozen_stratification.json", stratification(frozen))
    results = {"stage": "EXP14_A10_HUMAN_SELECTION_AND_A_DATASET_FREEZE",
               "status": EXCEPTIONS if exceptions else READY, "identity": summary, "n_frozen": len(frozen),
               "frozen_groups": len({r["group_id"] for r in frozen}),
               "frozen_asset_types": dict(Counter(r["asset_type"] for r in frozen)),
               "frozen_decision_origins": dict(Counter(r["decision_origin"] for r in frozen)),
               "original_source_class_coverage": dict(Counter(k for r in frozen for k in set(r["original_classes"]))),
               "historical_risk_counts_nonexclusive": dict(Counter(k for r in frozen for k in r["original_reasons"])),
               "label_exception_counts": dict(Counter(r.get("reason_code") or r.get("reason", "") for r in exceptions)),
               "exception_count": len(exceptions), "labels": labels, "data_entries": dataset_report,
               "repair_api_calls": 0, "new_registrations": 0, "new_generated_images": 0, "new_crop_assets": 0,
               "optimizer_updates": 0}
    write_json_exclusive(registry / "freeze_summary.json", results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("a9", "d1", "csv", "asset-root", "output", "exception-review"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    results = freeze(*(p.resolve() for p in (a.a9, a.d1, a.csv, a.asset_root, a.output, a.exception_review)))
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
