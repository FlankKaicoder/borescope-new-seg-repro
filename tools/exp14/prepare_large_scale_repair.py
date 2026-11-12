"""Freeze Exp14 TRAIN inputs and reconcile permanent historical request evidence."""
from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.prepare_g3b2a_batch import (
    _csv_bytes, _load_prompts, _manifest_hash_check, mask_for_selection,
)
from tools.exp13_cloud.g3.prepare_g3c2_v2 import (
    BASE_MANIFEST_SHA256, FROZEN_REVIEW_SHA256,
)
from tools.exp13_cloud.g3.verify_train_export import (
    CLASSES, EXPECTED_MANIFEST_SHA256, validate_annotation,
)

PHASE = "exp14_large_scale_repair_20261112"
PRIVATE_ROOT = Path(r"E:\borescope_review_assets")
REPO = Path(__file__).resolve().parents[2]
NATIVE = "FULL_NATIVE_FRAME_NO_CROP_RESIZE_PADDING_OR_COORDINATE_TRANSFORM"
FINAL_SHA = "9f249b8833b3c397e3d568cb20b338af0edad15f2fcdcfa08a41af72003b2dae"
V2_PREPARATION_SHA = "2689a94ead5f8e395c20a0c38d807acec0af958d65d9001b2598f2e3cc5c0f52"
MULTI = Path("configs/exp14/repair/MULTICLASS_RESTORATION_V1.txt")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    return g3.read_csv(path)[1]


def hash_file(path, expected=None):
    actual = g3.sha256_file(path)
    g3.require(expected is None or expected == actual, f"File identity mismatch: {path.name}")
    return actual


def unique_index(rows, key):
    index = {row[key]: row for row in rows}
    g3.require(len(index) == len(rows), f"Duplicate {key}")
    return index


def canonical_digest(record):
    return g3.digest(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def target_mask(source, shapes):
    return mask_for_selection(tuple(source["size"]), shapes, set(source["indexes"]))[0]


def verify_mask(path, expected, mask_sha):
    hash_file(path, mask_sha)
    with Image.open(path) as image:
        g3.require(image.format == "PNG" and image.mode == "RGBA" and image.size == expected.size,
                   "Invalid native RGBA mask")
        g3.require(ImageChops.difference(ImageOps.invert(image.getchannel("A")), expected).getbbox() is None,
                   "Mask alpha differs from the selected Polygon raster")


def freeze_sources(private):
    final = private / "g3b1_final_selection_20261107"
    csv_path = final / "final_selection_review.csv"
    audit = _manifest_hash_check(csv_path, final / "final_selection_audit.json")
    hash_file(csv_path, FINAL_SHA)
    backups = private / "g3b1_review_backups"
    backup = backups / "g3b1_review_emergency_backup3.json"
    hash_file(backup, audit["backup_sha256"])
    decisions = unique_index(load_json(backup), "source_id")
    earlier = unique_index(load_json(backups / "g3b1_review_emergency_backup.json"), "source_id")
    train = private / "g3a_train_export_unpacked_20261107_retry1"
    export_csv = train / "export_manifest.csv"
    split_csv = train / "frozen_split_manifest.csv"
    hash_file(export_csv, audit["export_manifest_sha256"])
    hash_file(split_csv, EXPECTED_MANIFEST_SHA256)
    exports = unique_index(read_rows(export_csv), "source_id")
    frozen = unique_index([row for row in read_rows(split_csv) if row["split"] == "train"], "stem")
    reviews = unique_index(read_rows(csv_path), "source_id")
    g3.require(set(reviews) == set(exports) == set(frozen) == set(decisions) == set(earlier),
               "Frozen TRAIN and G3-B1 source sets differ")
    states = Counter(row["user_decision"] for row in reviews.values())
    g3.require(dict(states) == audit["decision_counts"], "Frozen review counts differ")
    sources = {}
    for sid, row in reviews.items():
        backup_row = decisions[sid]
        selected = row["selected_polygon_ids"].split("|") if row["selected_polygon_ids"] else []
        for field in ("user_decision", "repair_priority", "user_notes"):
            g3.require(row[field] == backup_row[field], f"Backup mismatch in {field}")
        g3.require(selected == backup_row["selected_polygon_ids"], "Backup target set differs")
        if row["user_decision"] != "SELECT":
            continue
        g3.require(re.fullmatch(r"[0-9]+", sid) is not None, "Unsafe source ID")
        ex, fr = exports[sid], frozen[sid]
        g3.require(ex["split"] == "train" and row["group_id"] == fr["group_id"] == ex["group_id"],
                   "TRAIN source metadata differs")
        image = g3.safe_file(train / "images/train", Path(ex["export_image_path"]).name)
        annotation = g3.safe_file(train / "annotations/train", Path(ex["export_json_path"]).name)
        image_hash, json_hash = hash_file(image, fr["sha256"]), hash_file(annotation, fr["json_sha256"])
        g3.require(image_hash == row["source_image_sha256"] == ex["export_image_sha256"] and
                   json_hash == ex["export_json_sha256"] and
                   Path(ex["export_image_path"]).as_posix() == f"images/train/{image.name}" and
                   Path(ex["export_json_path"]).as_posix() == f"annotations/train/{annotation.name}",
                   "G3-B1 source and TRAIN export identity differ")
        size = tuple(int(ex[key]) for key in ("image_width", "image_height"))
        labels, issues, ann = validate_annotation(annotation, image.name, *size, set(CLASSES))
        g3.require(not issues and size == (int(row["image_width"]), int(row["image_height"])),
                   f"Invalid frozen annotation or dimensions: {sid}")
        all_ids = [f"{sid}_{i:03d}" for i in range(len(labels))]
        g3.require(row["polygon_ids"].split("|") == all_ids and selected and
                   len(set(selected)) == len(selected) and set(selected) <= set(all_ids),
                   f"Invalid SELECT Polygon identity: {sid}")
        indexes = [all_ids.index(pid) for pid in selected]
        polygon_labels = {pid: labels[i] for pid, i in zip(selected, indexes)}
        sources[sid] = {
            "source_id": sid, "group_id": row["group_id"], "split": "train", "decision": "SELECT",
            "priority": row["repair_priority"], "source_note": row["user_notes"],
            "earlier_backup_decision": earlier[sid]["user_decision"],
            "image": str(image), "image_sha256": image_hash,
            "annotation": str(annotation), "annotation_sha256": json_hash,
            "selected_polygon_ids": selected, "all_polygon_ids": all_ids,
            "polygon_labels": polygon_labels, "indexes": indexes, "size": list(size),
            "classes": [label for label in CLASSES if label in polygon_labels.values()],
        }
        target_mask(sources[sid], ann["shapes"])
        if len(selected) != len(all_ids):
            other, _ = mask_for_selection(size, ann["shapes"], set(range(len(labels))) - set(indexes))
            if ImageChops.multiply(target_mask(sources[sid], ann["shapes"]), other).getbbox():
                sources[sid]["hold"] = "SELECTED_MASK_INTERSECTS_UNSELECTED_POLYGON"
    return sources, {"final_csv_sha256": FINAL_SHA, "backup_sha256": audit["backup_sha256"],
                     "earlier_backup_sha256": hash_file(backups / "g3b1_review_emergency_backup.json"),
                     "split_sha256": EXPECTED_MANIFEST_SHA256, "decisions": dict(states)}


def verify_source_row(row, source):
    g3.require(row["split"] == "train" and row["decision"] == "SELECT" and
               row["source_image_sha256"] == source["image_sha256"] and
               row["source_json_sha256"] == source["annotation_sha256"] and
               row["selected_polygon_ids"].split("|") == source["selected_polygon_ids"],
               "Historical row does not bind the frozen source and targets")


def catalogue_history(private, sources):
    batch_v1 = private / "g3b2b_batch_review_20261108"
    batch_v2 = private / "g3c2_v2_live_20261109"
    a0 = unique_index(read_rows(private / "exp14_a0_data_admission/candidate_manifest.csv"), "variant_id")
    a1 = unique_index(read_rows(private / "exp14_a1_review_import_20261111_fprl0fi_/review_import_details.csv"),
                      "variant_id")
    a2 = unique_index(read_rows(private / "exp14_a2_core_freeze_20261111_xpeawdm_/core_freeze_manifest.csv"),
                      "variant_id")
    hash_file(batch_v1 / "expanded_batch_manifest.csv", BASE_MANIFEST_SHA256)
    hash_file(batch_v2 / "expanded_batch_manifest.csv",
              "1d6ee5f11c347ce1957f78667f7e66c7ee267ea372627811141b8667d0f4064f")
    variants = {}
    holds = {}
    for version, batch, dirname in (
        ("V1", batch_v1, "g3b2_batch_generation_20261108"),
        ("V2", batch_v2, "g3c2_v2_execution_20261109"),
    ):
        execution = private / dirname
        rows = unique_index(read_rows(batch / "expanded_batch_manifest.csv"), "source_id")
        lock_ids = {lock.stem for lock in (execution / "locks").glob("*.json")}
        g3.require(lock_ids <= set(rows), "Unmapped historical request lock")
        for sid, row in rows.items():
            source = sources[sid]
            verify_source_row(row, source)
            variant = f"{sid}_{version}"
            evidence = a0[variant]
            mask = g3.safe_file(batch, row["mask_path"])
            verify_mask(mask, target_mask(source, load_json(Path(source["annotation"]))["shapes"]), row["mask_sha256"])
            folder = execution / "live" / sid
            lock_path = execution / "locks" / f"{sid}.json"
            if sid not in lock_ids or not (folder / "TERMINAL.json").is_file():
                holds[sid] = "HISTORICAL_REQUEST_TERMINAL_UNVERIFIED"
                continue
            request, response = (load_json(folder / f"{name}_manifest.json") for name in ("request", "response"))
            lock = load_json(lock_path)
            g3.require(lock["request_sha256"] == g3.digest(json.dumps(request, sort_keys=True).encode()) and
                       request["input_sha256"] == source["image_sha256"] and
                       request["mask_sha256"] == row["mask_sha256"] and
                       request["prompt_wire_sha256"] == row["prompt_wire_sha256"] and
                       load_json(folder / "TERMINAL.json")["status"] == "SUCCEEDED",
                       "Historical request/lock/terminal mismatch")
            raw = g3.safe_file(folder, "raw_output.png")
            raw_sha = hash_file(raw, response["raw_output_sha256"])
            g3.require(raw_sha == evidence["raw_output_sha256"] and
                       evidence["edit_mask_sha256"] == row["mask_sha256"] and
                       evidence["source_sha256"] == source["image_sha256"] and
                       evidence["annotation_sha256"] == source["annotation_sha256"] and
                       evidence["selected_polygon_ids"].split("|") == source["selected_polygon_ids"],
                       "Exp14 A0 evidence differs from executed inputs")
            derived = {}
            for key in ("aligned_output", "valid_mask", "hard_composite", "feather_composite",
                        "geometry_diagnostic"):
                if evidence[f"{key}_path"]:
                    path = Path(evidence[f"{key}_path"]).resolve()
                    g3.require(private in path.parents, "Historical derived file outside private root")
                    derived[key] = {"path": str(path), "sha256": hash_file(path, evidence[f"{key}_sha256"])}
            approval = a1[variant]
            if approval["image_verified"] == "True":
                g3.require(approval["final_image_sha256"] == derived["aligned_output"]["sha256"],
                           "A1 decision refers to a different image")
            core = a2.get(variant, {})
            if core:
                g3.require(core["final_image_sha256"] == derived["aligned_output"]["sha256"] and
                           core["source_image_sha256"] == source["image_sha256"],
                           "A2 decision refers to a different image")
            variants[variant] = {
                "source_id": sid, "version": version, "origin": f"G3_{version}",
                "request_path": str(folder / "request_manifest.json"), "lock_path": str(lock_path),
                "request_sha256": hash_file(folder / "request_manifest.json"),
                "prompt_id": row["prompt_id"], "prompt_wire_sha256": request["prompt_wire_sha256"],
                "mask": str(mask), "mask_sha256": row["mask_sha256"], "raw": str(raw), "raw_sha256": raw_sha,
                "historical_visual": evidence["historical_visual_status"], "derived": derived,
                "a1_status": [approval["a_status"], approval["b_status"]],
                "a2_status": core.get("training_status", ""), "historical_support": core.get("finding", ""),
                "historical_ui": core.get("ui_flag", ""),
            }

    # G2 used the same native frame and target alpha for these earlier requests.
    pilot_inputs = private / "cloud_pilot_g2_three_source_20261101_retry1/sources"
    pilot_exec = private / "cloud_pilot_g2_execution_20261102"
    for lock in sorted((pilot_exec / "locks").glob("*/STARTED.json")):
        sid = lock.parent.name
        if sid not in sources:
            continue
        source = sources[sid]
        folder = pilot_exec / "live" / sid
        if not (folder / "TERMINAL.json").is_file():
            holds[sid] = "HISTORICAL_REQUEST_TERMINAL_UNVERIFIED"
            continue
        request_path = folder / "request_manifest.json"
        req = load_json(request_path)
        request, hashes = req["request"], req["input_hashes"]
        g3.require(hashes["original_image"] == source["image_sha256"] and
                   hashes["original_professional_json"] == source["annotation_sha256"] and
                   request["multipart_fields"]["image[]"]["sha256"] == source["image_sha256"] and
                   load_json(folder / "TERMINAL.json")["status"] == "FIRST_CALL_RAW / PENDING_HUMAN_REVIEW",
                   "G2 native frame/terminal evidence mismatch")
        mask = pilot_inputs / sid / "api_rgba_mask.png"
        verify_mask(mask, target_mask(source, load_json(Path(source["annotation"]))["shapes"]), hashes["api_rgba_mask"])
        raw = folder / "raw_output.png"
        sha = hash_file(raw, load_json(folder / "response_manifest.json")["raw_output_sha256"])
        prompt = request["multipart_fields"]["prompt"]
        variants[f"{sid}_G2_V1"] = {
            "source_id": sid, "version": "V1", "origin": "G2_NATIVE", "derived": {},
            "request_path": str(request_path), "request_sha256": hash_file(request_path),
            "lock_path": str(lock), "prompt_id": prompt["prompt_id"],
            "prompt_wire_sha256": prompt["transmitted_string_sha256"],
            "mask": str(mask), "mask_sha256": hashes["api_rgba_mask"], "raw": str(raw), "raw_sha256": sha,
            "historical_visual": "G2_VISUAL_ACCEPTED",
        }

    # The earlier corrosion ROI has a different uploaded frame and is preserved as exposure evidence.
    pilot_roi = private / "cloud_pilot_g1f_1a_preflight"
    roi_terminal = load_json(pilot_roi / "live_result/TERMINAL.json")
    roi_request = load_json(pilot_roi / "live_result/request_manifest.json")
    roi_response = load_json(pilot_roi / "live_result/response_manifest.json")
    g3.require((pilot_roi / "STARTED.json").is_file() and
               roi_terminal["status"] == "FIRST_CALL_RAW / PENDING_HUMAN_REVIEW" and
               roi_request["fields"]["image[]"]["sha256"] != sources["2"]["image_sha256"],
               "Earlier ROI exposure could not be distinguished from the native request")
    sources["2"]["prior_roi"] = {
        "lock": str(pilot_roi / "STARTED.json"), "request": str(pilot_roi / "live_result/request_manifest.json"),
        "image_sha256": roi_request["fields"]["image[]"]["sha256"],
        "mask_sha256": roi_request["fields"]["mask"]["sha256"],
        "raw_sha256": hash_file(pilot_roi / "live_result/raw_output.png", roi_response["raw_output_sha256"]),
    }
    return variants, holds


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def request_identity(row):
    # Filenames and PNG encoder metadata are outside the semantic request identity.
    return canonical_digest({key: row[key] for key in (
        "source_image_sha256", "target_alpha_sha256", "prompt_wire_sha256", "endpoint", "model", "n",
    )})


def input_row(root, source, version, prompt, target, mask_bytes, protocol):
    sid = source["source_id"]
    variant = f"{sid}_{version}"
    image = Path(source["image"])
    image_path = root / "inputs" / image.name
    mask_path = root / "masks" / f"{variant}.png"
    if not image_path.exists():
        g3.write_exclusive(image_path, image.read_bytes())
    g3.write_exclusive(mask_path, mask_bytes)
    row = {
        "source_id": sid, "version": version, "variant_id": variant, "split": "train", "decision": "SELECT",
        "source_image_sha256": source["image_sha256"], "source_json_sha256": source["annotation_sha256"],
        "image_width": source["size"][0], "image_height": source["size"][1],
        "selected_polygon_ids": "|".join(source["selected_polygon_ids"]),
        "batch_image_path": image_path.relative_to(root).as_posix(),
        "mask_path": mask_path.relative_to(root).as_posix(),
        "mask_sha256": g3.digest(mask_bytes), "target_alpha_sha256": g3.digest(target.tobytes()),
        "prompt_id": prompt["prompt_id"], "prompt_file": prompt["path"],
        "prompt_file_sha256": prompt["file_sha256"], "prompt_wire_sha256": prompt["wire_sha256"],
        "status": "PROPOSED", "manually_approved": "NO", "authorized_for_paid_post": "NO",
        "request_count_proposed": "1", "roi_policy": NATIVE, "n": "1",
        "endpoint": protocol["base_url"] + protocol["endpoint"], "model": protocol["model"],
    }
    row["request_id"] = request_identity(row)
    return row


def prepare(private, root, authorization_task):
    protocol, protocol_sha = g3.load_protocol(REPO)
    sources, provenance = freeze_sources(private)
    variants, holds = catalogue_history(private, sources)
    eligible = private / "g3c2_v2_preparation_20261109_retry4"
    v2_path = eligible / "expanded_batch_manifest.csv"
    hash_file(v2_path, V2_PREPARATION_SHA)
    historical_failures = private / "g3c0_alignment_20261108_release/g3c0_review_recovered.txt"
    hash_file(historical_failures, FROZEN_REVIEW_SHA256)
    failure_rows = unique_index(read_rows(historical_failures), "source_id")
    v2_prepared = unique_index(read_rows(v2_path), "source_id")
    for held in load_json(eligible / "held_sources.json"):
        holds[held["source_id"]] = held["reason_code"]

    prompts, _ = _load_prompts(REPO)
    wrapper = (REPO / MULTI).read_bytes()
    historical_sources = {row["source_id"] for row in variants.values()}
    # A fixed run path and exclusive creation preserve identity across resumes.
    root.mkdir(exist_ok=False)
    rows = []
    for sid, source in sources.items():
        if sid in holds:
            source["hold"] = holds[sid]
        if source.get("hold"):
            continue
        if sid in historical_sources and (sid not in v2_prepared or f"{sid}_V2" in variants):
            continue
        shapes = load_json(Path(source["annotation"]))["shapes"]
        target = target_mask(source, shapes)
        if sid in v2_prepared:
            version = "V2"
            old = v2_prepared[sid]
            verify_source_row(old, source)
            g3.require(failure_rows[sid]["visual_review_status"] == "NEEDS_RETRY" and
                       old["v1_raw_output_sha256"] == variants[f"{sid}_V1"]["raw_sha256"],
                       "Prepared V2 lacks verified human repair-failure evidence")
            mask_path = g3.safe_file(eligible, old["mask_path"])
            verify_mask(mask_path, target, old["mask_sha256"])
            prompt = {"prompt_id": old["prompt_id"], "path": old["prompt_file"],
                      "file_sha256": old["prompt_file_sha256"], "wire_sha256": old["prompt_wire_sha256"]}
            mask_bytes = mask_path.read_bytes()
        else:
            version = "V1"
            labels = source["classes"]
            prompt = prompts[labels[0]]
            rgba = Image.new("RGBA", tuple(source["size"]), (255, 255, 255, 255))
            rgba.putalpha(ImageOps.invert(target))
            mask_bytes = png_bytes(rgba)
            if len(labels) > 1:
                # Use exact historical class text with one versioned multi-target instruction.
                sections = [wrapper.rstrip(b"\n")]
                for label in labels:
                    sections.append(label.encode("utf-8") + b"\n" + (REPO / prompts[label]["path"]).read_bytes().rstrip(b"\n"))
                payload = b"\n\n".join(sections) + b"\n"
                relative = Path("prompts") / ("_".join(str(CLASSES.index(label)) for label in labels) + "_V1.txt")
                path = root / relative
                if not path.exists():
                    g3.write_exclusive(path, payload)
                prompt = {"prompt_id": "MULTICLASS_RESTORATION_V1",
                          "path": relative.as_posix(), "file_sha256": g3.digest(payload),
                          "wire_sha256": g3.digest(payload[:-1]), "private_prompt": True}
        row = input_row(root, source, version, prompt, target, mask_bytes, protocol["request"])
        row["prompt_root"] = "run" if prompt.get("private_prompt") else "repo"
        g3.validate_input(row, root, root if row["prompt_root"] == "run" else REPO)
        rows.append(row)

    # Run the prepared historical retry first.
    rows.sort(key=lambda row: (row["version"] != "V2", int(row["source_id"])))
    g3.require(len(rows) == len({row["request_id"] for row in rows}) ==
               len({row["source_id"] for row in rows}), "Duplicate request in frozen scope")
    manifest = {"sources": sources, "historical_variants": variants, "requests": rows, "provenance": provenance}
    g3.write_json_exclusive(root / "scope.json", manifest)
    g3.write_exclusive(root / "selected_sources.csv", _csv_bytes([
        {**source, "classes": "|".join(source["classes"]),
         "selected_polygon_ids": "|".join(source["selected_polygon_ids"]),
         "all_polygon_ids": "|".join(source["all_polygon_ids"])} for source in sources.values()
    ], ["source_id", "split", "decision", "priority", "earlier_backup_decision", "classes", "selected_polygon_ids",
        "all_polygon_ids", "image_sha256", "annotation_sha256", "source_note", "hold"]))
    g3.write_json_exclusive(root / "historical_reuse.json", variants)
    g3.write_exclusive(root / "request_manifest.csv", _csv_bytes(rows, list(rows[0])))
    task_copy = root / "authorization_task.txt"
    g3.write_exclusive(task_copy, authorization_task.read_bytes())
    auth = {
        "record_type": "EXP14_LARGE_SCALE_REPAIR_EXPLICIT_USER_AUTHORIZATION",
        "authorization_basis": "User task attachment sections 0, 3, 4, 5 and 6",
        "authorization_attachment_sha256": hash_file(task_copy),
        "cost_status": "COST_UNKNOWN_USER_AUTHORIZED_WITHOUT_SPEND_CAP",
        "scope_sha256": hash_file(root / "scope.json"),
        "request_manifest_sha256": hash_file(root / "request_manifest.csv"),
        "protocol_sha256": protocol_sha, "source_csv_sha256": FINAL_SHA,
        "request_ids": [row["request_id"] for row in rows], "max_posts": len(rows), "n": 1,
    }
    g3.write_json_exclusive(root / "authorization.json", auth)
    print(json.dumps({"selected": len(sources), "historical_variants": len(variants),
                      "historical_sources": len(historical_sources), "requests": dict(Counter(row["version"] for row in rows)),
                      "holds": dict(Counter(s["hold"] for s in sources.values() if s.get("hold")))}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-attachment", type=Path, required=True)
    args = parser.parse_args()
    private = PRIVATE_ROOT.resolve()
    prepare(private, private / PHASE, args.task_attachment)


if __name__ == "__main__":
    main()
