"""Offline-only G3-B2A source statistics, proposal, and selected-mask dry run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tools.exp13_cloud.g3.build_candidate_review import CLASSES, HISTORY
from tools.exp13_cloud.g3.verify_train_export import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_TRAIN_INSTANCES,
    read_rows,
    sha256_file,
    validate_export_data,
)

PROMPT_DIR = Path("configs/exp13_cloud/g3/prompts")
PROTOCOL_PATH = Path("configs/exp13_cloud/g2/pilot_protocol_v2.json")
PROMPT_PATHS = {
    "Burn": PROMPT_DIR / "BURN_RESTORATION_V1.txt",
    "Crack": Path("configs/exp13_cloud/g2/CRACK_RESTORATION_V1.txt"),
    "Dent": PROMPT_DIR / "DENT_RESTORATION_V1.txt",
    "Material missing": PROMPT_DIR / "MATERIAL_MISSING_RESTORATION_V1.txt",
    "Tears": PROMPT_DIR / "TEARS_RESTORATION_V1.txt",
    "Tip curl": PROMPT_DIR / "TIP_CURL_RESTORATION_V1.txt",
    "corrosion": Path("configs/exp13_cloud/g2/CORROSION_RESTORATION_V1.txt"),
}
KNOWN_HISTORY = {"2", "23", "260", "470"}
PROPOSAL_CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "corrosion"]
PROPOSAL_PER_CLASS = 6
MANIFEST_COLUMNS = [
    "source_id", "split", "class", "group_id", "near_duplicate_group_id",
    "decision", "priority", "selected_polygon_ids", "all_polygon_ids",
    "selected_polygon_count", "all_polygon_count", "image_path", "json_path",
    "source_image_sha256", "source_json_sha256", "image_width", "image_height",
    "image_format", "roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi_policy",
    "mask_path", "mask_sha256", "mask_target_pixels", "mask_coverage_fraction",
    "target_area_fraction", "target_bbox_xyxy", "padding_risk", "ui_risk",
    "frame_edge_proximity_px",
    "prompt_id", "prompt_file", "prompt_file_sha256", "prompt_wire_sha256",
    "historical_cloud_exposure", "history_state", "request_count_proposed",
    "trusted_training_pair_status", "manually_approved", "authorized_for_paid_post", "status", "reason_codes",
    "provider", "model", "endpoint", "method", "n", "request_params_state",
]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mask_for_selection(size: tuple[int, int], shapes: list[dict[str, Any]], selected_indexes: set[int]) -> tuple[Image.Image, int]:
    if not selected_indexes or any(index < 0 or index >= len(shapes) for index in selected_indexes):
        raise ValueError("selected Polygon indexes are empty or outside the annotation")
    binary = Image.new("L", size, 0)
    draw = ImageDraw.Draw(binary)
    for index in sorted(selected_indexes):
        shape = shapes[index]
        points = shape.get("points")
        if shape.get("shape_type") != "polygon" or not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"invalid selected Polygon at index {index}")
        parsed = []
        for point in points:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError(f"malformed point in selected Polygon {index}")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= size[0] and 0 <= y <= size[1]):
                raise ValueError(f"selected Polygon {index} has an out-of-image point")
            parsed.append((x, y))
        draw.polygon(parsed, fill=255)
    pixels = sum(value > 0 for value in binary.get_flattened_data())
    if pixels <= 0 or pixels >= size[0] * size[1]:
        raise ValueError("selected mask is empty or covers the entire image")
    return binary, pixels


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def _manifest_hash_check(final_csv: Path, final_audit: Path) -> dict[str, Any]:
    audit = json.loads(final_audit.read_text(encoding="utf-8-sig"))
    if audit.get("status") != "G3B1_FINAL_SELECTION_AUDIT_PASS":
        raise RuntimeError("final selection audit is missing or not PASS")
    if sha256_file(final_csv) != audit.get("final_csv_sha256"):
        raise RuntimeError("formal final-selection CSV SHA256 differs from its audit report")
    if audit.get("source_rows") != EXPECTED_TRAIN_IMAGES or audit.get("train_images") != EXPECTED_TRAIN_IMAGES or audit.get("train_instances") != EXPECTED_TRAIN_INSTANCES:
        raise RuntimeError("final-selection audit does not bind the expected G3-A TRAIN counts")
    return audit


def _load_prompts(repo: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    text_by_class: dict[str, str] = {}
    for label, relative in PROMPT_PATHS.items():
        path = repo / relative
        data = path.read_bytes()
        if not data.endswith(b"\n") or data.endswith(b"\n\n") or data.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"Prompt must be UTF-8 without BOM and have exactly one terminal LF: {path}")
        wire = data[:-1]
        wire.decode("utf-8")
        prompt_id = f"{label.upper().replace(' ', '_')}_RESTORATION_V1"
        records[label] = {
            "prompt_id": prompt_id,
            "path": relative.as_posix(),
            "file_sha256": sha_bytes(data),
            "wire_sha256": sha_bytes(wire),
            "wire_rule": "UTF-8; remove exactly one final LF byte; no other normalization",
            "provenance": "REUSED_HISTORICAL_G2_V1" if label in {"Crack", "corrosion"} else "NEW_CLASS_SPECIFIC_V1_DRAFT",
        }
        text_by_class[label] = wire.decode("utf-8")
    return records, text_by_class


def _load_protocol(repo: Path) -> tuple[dict[str, Any], str]:
    path = repo / PROTOCOL_PATH
    data = path.read_bytes()
    bundle = json.loads((repo / "configs/exp13_cloud/g2/execution_bundle.json").read_text(encoding="utf-8"))
    frozen_entry = next((item for item in bundle.get("files", []) if item.get("path") == PROTOCOL_PATH.as_posix()), None)
    if frozen_entry is None or sha_bytes(data) != frozen_entry.get("sha256"):
        raise ValueError("G2 protocol bytes do not match the frozen executor bundle")
    protocol = json.loads(data.decode("utf-8-sig"))
    if protocol.get("record_type") != "G2_EXECUTION_PROTOCOL_V2_NOT_AUTHORIZATION":
        raise ValueError("historical G2 protocol is missing or has an unexpected record type")
    return protocol, sha_bytes(data)


def _proposal_candidates(sources: list[dict[str, str]], instances: dict[str, dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {label: [] for label in PROPOSAL_CLASSES}
    for source in sources:
        if source["user_decision"] != "SELECT" or source["repair_priority"] != "A":
            continue
        source_id = source["source_id"]
        polygon_ids = [value for value in source["polygon_ids"].split("|") if value]
        selected_ids = [value for value in source["selected_polygon_ids"].split("|") if value]
        if not polygon_ids or set(polygon_ids) != set(selected_ids):
            continue
        labels = [value for value in source["classes_present"].split("|") if value]
        if len(labels) != 1 or labels[0] not in candidates or source_id in KNOWN_HISTORY:
            continue
        selected = [instances[value] for value in selected_ids]
        if any(row["padding_risk"] != "LOW" for row in selected):
            continue
        group = source["group_id"]
        near = source["near_duplicate_group_id"]
        if near != "UNKNOWN":
            dedup_key = f"near:{near}"
        else:
            dedup_key = f"group:{group}"
        area = sum(float(row["area_fraction_polygon"]) for row in selected)
        candidates[labels[0]].append({"source": source, "target_rows": selected, "dedup_key": dedup_key, "area_fraction": area})
    for values in candidates.values():
        values.sort(key=lambda item: (item["area_fraction"], item["source"]["source_id"]))
    return candidates


def _select_proposals(candidates: dict[str, list[dict[str, Any]]], explicit_ids: list[str] | None = None) -> list[dict[str, Any]]:
    if explicit_ids is not None:
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("explicit proposal source IDs contain duplicates")
        eligible = {item["source"]["source_id"]: item for values in candidates.values() for item in values}
        missing = [source_id for source_id in explicit_ids if source_id not in eligible]
        if missing:
            raise ValueError(f"explicit proposal source is not eligible under frozen metadata rules: {missing}")
        selected = [eligible[source_id] for source_id in explicit_ids]
        dedup_keys = [item["dedup_key"] for item in selected]
        if len(dedup_keys) != len(set(dedup_keys)):
            raise ValueError("explicit proposal list repeats a source group or known near-duplicate group")
        return selected

    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for label in PROPOSAL_CLASSES:
        values = candidates[label]
        quota = min(PROPOSAL_PER_CLASS, len({item["dedup_key"] for item in values} - used_keys))
        chosen: list[dict[str, Any]] = []
        for rank in range(quota):
            target_quantile = (rank + 0.5) / max(1, quota)
            target_index = round(target_quantile * (len(values) - 1))
            order = sorted(range(len(values)), key=lambda index: (abs(index - target_index), values[index]["source"]["source_id"]))
            candidate = next((values[index] for index in order if values[index]["dedup_key"] not in used_keys), None)
            if candidate is None:
                raise RuntimeError(f"cannot fill a group-deduplicated proposal quota for {label}")
            chosen.append(candidate)
            used_keys.add(candidate["dedup_key"])
        selected.extend(chosen)
    return selected


def _draw_private_preview(image_path: Path, shapes: list[dict[str, Any]], target_indexes: set[int], mask: Image.Image, overlay_path: Path, mask_path: Path) -> int:
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    width, height = original.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_width = max(2, round(min(width, height) / 360))
    for index, shape in enumerate(shapes):
        points = [(float(point[0]), float(point[1])) for point in shape["points"]]
        target = index in target_indexes
        color = (240, 55, 40, 112) if target else (35, 155, 220, 100)
        edge = (255, 235, 30, 255) if target else (30, 210, 255, 255)
        draw.polygon(points, fill=color, outline=edge, width=line_width)
        label = f"{index:03d} {'TARGET' if target else 'UNSELECTED'}"
        draw.text(points[0], label, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 230))
    preview = Image.alpha_composite(original.convert("RGBA"), layer).convert("RGB")
    preview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(overlay_path, format="JPEG", quality=90, optimize=True)
    alpha = Image.eval(mask, lambda value: 255 - value)
    rgba = Image.new("RGBA", mask.size, (255, 255, 255, 255))
    rgba.putalpha(alpha)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(mask_path, format="PNG", optimize=True)
    return sum(value == 0 for value in alpha.get_flattened_data())


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing preparation directory: {args.output}")
    audit = _manifest_hash_check(args.selection_csv, args.selection_audit)
    gate = validate_export_data(args.manifest, args.export_manifest, args.train_root, args.class_mapping)
    if gate.get("status") != "PASS" or gate.get("issues"):
        raise RuntimeError("G3-A TRAIN export failed current offline revalidation")
    if gate.get("frozen_manifest_sha256") != EXPECTED_MANIFEST_SHA256 or gate.get("export_manifest_sha256") != audit.get("export_manifest_sha256"):
        raise RuntimeError("final-selection audit does not match the current G3-A export")

    sources = read_rows(args.selection_csv)
    instances_rows = read_rows(args.instance_review_csv)
    exports = read_rows(args.export_manifest)
    if len(sources) != EXPECTED_TRAIN_IMAGES or len(instances_rows) != EXPECTED_TRAIN_INSTANCES:
        raise RuntimeError("source or Polygon row count differs from the audited TRAIN set")
    source_by_id = {row["source_id"]: row for row in sources}
    export_by_id = {row["source_id"]: row for row in exports}
    instances = {row["candidate_id"]: row for row in instances_rows}
    if len(source_by_id) != len(sources) or len(instances) != len(instances_rows) or set(source_by_id) != set(export_by_id):
        raise RuntimeError("source/Polygon exact-set does not match G3-A export")

    prompt_records, _ = _load_prompts(args.repo_root)
    protocol, protocol_hash = _load_protocol(args.repo_root)
    proposal_candidates = _proposal_candidates(sources, instances)
    explicit_ids = None
    if args.proposal_source_list:
        payload = json.loads(args.proposal_source_list.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
            raise ValueError("proposal source list must be a JSON array of source_id strings")
        explicit_ids = payload
    proposals = _select_proposals(proposal_candidates, explicit_ids)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    class_source_sets: dict[str, set[str]] = {label: set() for label in CLASSES}
    class_present_source_sets: dict[str, set[str]] = {label: set() for label in CLASSES}
    class_target_counts: Counter[str] = Counter()
    decision_target_counts: Counter[str] = Counter()
    decision_class_target_counts: Counter[tuple[str, str]] = Counter()
    priority_sources: Counter[tuple[str, str]] = Counter()
    selected_source_rows = [row for row in sources if row["user_decision"] == "SELECT"]
    maybe_source_rows = [row for row in sources if row["user_decision"] == "MAYBE"]
    all_candidate_rows = selected_source_rows + maybe_source_rows
    total_polygon_counts: Counter[str] = Counter()
    risks: Counter[tuple[str, str, str]] = Counter()
    frame_edge_proximity: Counter[tuple[str, str]] = Counter()
    resolution: Counter[str] = Counter()
    multi_polygon_sources = multi_class_sources = partial_sources = multi_target_sources = 0
    known_history_rows = []
    selected_poly_index: dict[str, int] = {}
    for source in all_candidate_rows:
        source_id = source["source_id"]
        selected_ids = [value for value in source["selected_polygon_ids"].split("|") if value]
        all_ids = [value for value in source["polygon_ids"].split("|") if value]
        labels = [value for value in source["classes_present"].split("|") if value]
        if source["user_decision"] == "SELECT":
            for label in labels:
                class_present_source_sets[label].add(source_id)
        if len(all_ids) > 1:
            multi_polygon_sources += source["user_decision"] == "SELECT"
        if len(labels) > 1:
            multi_class_sources += source["user_decision"] == "SELECT"
        if 0 < len(selected_ids) < len(all_ids):
            partial_sources += source["user_decision"] == "SELECT"
        if len(selected_ids) > 1:
            multi_target_sources += source["user_decision"] == "SELECT"
        priority_sources[(source["user_decision"], source["repair_priority"])] += 1
        exposure = source["historical_cloud_exposure"]
        if exposure != "NONE_RECORDED_IN_G1_G2_AUDIT_MAP":
            known_history_rows.append((source_id, exposure))
        target_rows = []
        for polygon_id in selected_ids:
            instance = instances.get(polygon_id)
            if instance is None or instance["source_id"] != source_id or instance["split"] != "train":
                raise RuntimeError(f"selected Polygon identity/split mismatch: {polygon_id}")
            if source["user_decision"] == "SELECT":
                class_source_sets[instance["class"]].add(source_id)
                class_target_counts[instance["class"]] += 1
            decision_target_counts[source["user_decision"]] += 1
            decision_class_target_counts[(source["user_decision"], instance["class"])] += 1
            risks[(source["user_decision"], "padding", instance["padding_risk"])] += 1
            risks[(source["user_decision"], "ui", instance["ui_risk"])] += 1
            bbox = [float(value) for value in json.loads(instance["bbox_xyxy"])]
            image_width, image_height = int(instance["image_width"]), int(instance["image_height"])
            edge_distance = min(bbox[0], bbox[1], image_width - bbox[2], image_height - bbox[3])
            frame_edge_proximity[(source["user_decision"], "<=5px")] += edge_distance <= 5
            frame_edge_proximity[(source["user_decision"], "<=15px")] += edge_distance <= 15
            selected_poly_index[polygon_id] = int(instance["polygon_index"])
            target_rows.append(instance)
        for polygon_id in all_ids:
            if polygon_id not in instances or instances[polygon_id]["source_id"] != source_id:
                raise RuntimeError(f"all-Polygon source linkage failed: {polygon_id}")
            total_polygon_counts[instances[polygon_id]["class"]] += source["user_decision"] == "SELECT"
        if source["user_decision"] != "SELECT":
            continue
        exp = export_by_id[source_id]
        resolution[f"{exp['image_width']}x{exp['image_height']}"] += 1
        del target_rows

    manifest_rows: list[dict[str, Any]] = []
    preview_items = []
    for item in proposals:
        source = item["source"]
        source_id = source["source_id"]
        exp = export_by_id[source_id]
        image_path = args.train_root / exp["export_image_path"]
        json_path = args.train_root / exp["export_json_path"]
        if sha256_file(image_path) != exp["source_image_sha256"] or sha256_file(json_path) != exp["source_json_sha256"]:
            raise RuntimeError(f"source file hash mismatch before dry-run: {source_id}")
        annotation = json.loads(json_path.read_text(encoding="utf-8-sig"))
        width, height = int(exp["image_width"]), int(exp["image_height"])
        with Image.open(image_path) as image:
            image.load()
            if image.size != (width, height) or image.format not in {"JPEG", "PNG"}:
                raise RuntimeError(f"unsupported/changed source image metadata: {source_id}")
            image_format = image.format
        if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
            raise RuntimeError(f"source JSON dimensions mismatch: {source_id}")
        shapes = annotation.get("shapes")
        if not isinstance(shapes, list) or len(shapes) != len(source["polygon_ids"].split("|")):
            raise RuntimeError(f"source JSON Polygon count mismatch: {source_id}")
        selected_ids = [value for value in source["selected_polygon_ids"].split("|") if value]
        target_indexes = {selected_poly_index[value] for value in selected_ids}
        for polygon_id, index in zip(source["polygon_ids"].split("|"), range(len(shapes)), strict=True):
            if polygon_id != f"{source_id}_{index:03d}":
                raise RuntimeError(f"Polygon IDs do not follow the audited annotation index: {source_id}")
        binary, pixels = mask_for_selection((width, height), shapes, target_indexes)
        mask_path = output / "masks" / f"{source_id}_selected_rgba.png"
        overlay_path = output / "previews" / f"{source_id}_target_overlay.jpg"
        count = _draw_private_preview(image_path, shapes, target_indexes, binary, overlay_path, mask_path)
        if count != pixels:
            raise RuntimeError(f"generated RGBA target alpha differs from selected Polygon raster: {source_id}")
        single_class = item["source"]["classes_present"].split("|")
        prompt = prompt_records[single_class[0]]
        reason_codes = [
            "SELECT_A", "ALL_SOURCE_POLYGONS_SELECTED", "SINGLE_CLASS_SOURCE",
            "TARGET_PADDING_LOW", "NO_RECORDED_G1_G2_CALL_NOT_PROOF_OF_NO_PRIOR_CALL",
            "ONE_SOURCE_PER_GROUP", "AREA_RANK_STRATIFIED_WITHIN_CLASS",
            "UI_RISK_UNKNOWN_REQUIRES_HUMAN_REVIEW", "MECHANICAL_EDGE_SEMANTICS_UNVERIFIED",
        ]
        if explicit_ids is not None:
            reason_codes.extend(["CONTACT_SHEET_SCREEN_NO_OBVIOUS_CAMERA_UI", "FULL_RESOLUTION_HUMAN_CONFIRMATION_REQUIRED"])
        row = {
            "source_id": source_id, "split": "train", "class": single_class[0],
            "group_id": source["group_id"], "near_duplicate_group_id": source["near_duplicate_group_id"],
            "decision": source["user_decision"], "priority": source["repair_priority"],
            "selected_polygon_ids": "|".join(selected_ids), "all_polygon_ids": source["polygon_ids"],
            "selected_polygon_count": len(selected_ids), "all_polygon_count": len(source["polygon_ids"].split("|")),
            "image_path": exp["export_image_path"], "json_path": exp["export_json_path"],
            "source_image_sha256": exp["source_image_sha256"], "source_json_sha256": exp["source_json_sha256"],
            "image_width": width, "image_height": height, "image_format": image_format,
            "roi_x1": 0, "roi_y1": 0, "roi_x2": width, "roi_y2": height,
            "roi_policy": "FULL_NATIVE_FRAME_NO_CROP_RESIZE_PADDING_OR_COORDINATE_TRANSFORM",
            "mask_path": mask_path.relative_to(output).as_posix(), "mask_sha256": sha256_file(mask_path),
            "mask_target_pixels": pixels, "mask_coverage_fraction": round(pixels / (width * height), 10),
            "target_area_fraction": round(item["area_fraction"], 10),
            "target_bbox_xyxy": "|".join(str(value) for target in item["target_rows"] for value in json.loads(target["bbox_xyxy"])),
            "padding_risk": "|".join(target["padding_risk"] for target in item["target_rows"]),
            "ui_risk": "|".join(target["ui_risk"] for target in item["target_rows"]),
            "frame_edge_proximity_px": min(
                min(float(bbox[0]), float(bbox[1]), int(target["image_width"]) - float(bbox[2]), int(target["image_height"]) - float(bbox[3]))
                for target in item["target_rows"] for bbox in [json.loads(target["bbox_xyxy"])]
            ),
            "prompt_id": prompt["prompt_id"], "prompt_file": prompt["path"],
            "prompt_file_sha256": prompt["file_sha256"], "prompt_wire_sha256": prompt["wire_sha256"],
            "historical_cloud_exposure": source["historical_cloud_exposure"],
            "history_state": "NO_CALL_RECORDED_IN_CURRENT_MAP_NOT_PROOF_OF_NO_PRIOR_CALL",
            "trusted_training_pair_status": source["trusted_training_pair_status"],
            "request_count_proposed": 1, "manually_approved": "NO", "authorized_for_paid_post": "NO",
            "provider": protocol["request"]["base_url"], "model": protocol["request"]["model"],
            "endpoint": protocol["request"]["endpoint"], "method": protocol["request"]["method"],
            "n": protocol["request"]["n"],
            "request_params_state": "REUSED_G2_PROTOCOL;SIZE_QUALITY_FORMAT_UNSET;PROVIDER_COMPATIBILITY_UNVERIFIED",
            "status": "PROPOSED", "reason_codes": "|".join(reason_codes),
        }
        manifest_rows.append(row)
        preview_items.append((source_id, single_class[0], overlay_path.name, f"masks/{mask_path.name}"))

    manifest_bytes = _csv_bytes(manifest_rows, MANIFEST_COLUMNS)
    manifest_path = output / "proposed_batch_manifest.csv"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    prompt_manifest = {"record_type": "G3B2A_PROMPT_V1_DRY_RUN_NOT_AUTHORIZATION", "prompts": prompt_records}
    (output / "prompt_manifest.json").write_text(json.dumps(prompt_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    class_stats = {}
    for label in CLASSES:
        class_stats[label] = {
            "select_sources_with_class_present": len(class_present_source_sets[label]),
            "select_sources_with_selected_target": len(class_source_sets[label]),
            "select_target_polygons": class_target_counts[label],
            "maybe_target_polygons": decision_class_target_counts[("MAYBE", label)],
            "selected_annotation_polygon_count": total_polygon_counts[label],
        }
    source_group_stats = Counter(row["group_id"] for row in selected_source_rows)
    near_group_stats = Counter(row["near_duplicate_group_id"] for row in selected_source_rows)
    eligible_counts = {label: len(proposal_candidates[label]) for label in PROPOSAL_CLASSES}
    proposed_class_counts = Counter(row["class"] for row in manifest_rows)
    routine_shape = sum(
        1 for source in selected_source_rows
        if source["repair_priority"] == "A"
        and len(source["classes_present"].split("|")) == 1
        and set(source["polygon_ids"].split("|")) == set(source["selected_polygon_ids"].split("|"))
        and all(instances[pid]["padding_risk"] == "LOW" for pid in source["selected_polygon_ids"].split("|") if pid)
        and source["source_id"] not in KNOWN_HISTORY
        and source_group_stats[source["group_id"]] == 1
    )
    summary = {
        "status": "G3B2A_OFFLINE_DRY_RUN_READY_FOR_REVIEW",
        "input_evidence": {
            "selection_csv_sha256": sha256_file(args.selection_csv),
            "selection_audit_final_csv_sha256": audit["final_csv_sha256"],
            "selection_audit_status": audit["status"],
            "frozen_manifest_sha256": gate["frozen_manifest_sha256"],
            "export_manifest_sha256": gate["export_manifest_sha256"],
            "train_integrity_status": gate["status"], "train_integrity_issues": len(gate["issues"]),
        },
        "candidate_pool": {
            "train_images": EXPECTED_TRAIN_IMAGES, "train_instances": EXPECTED_TRAIN_INSTANCES,
            "select_sources": len(selected_source_rows), "maybe_sources": len(maybe_source_rows),
            "reject_sources": sum(row["user_decision"] == "REJECT" for row in sources),
            "select_target_polygons": decision_target_counts["SELECT"],
            "maybe_target_polygons": decision_target_counts["MAYBE"],
            "class_counts": class_stats,
            "priority_distribution": {f"{decision}:{priority or 'UNSET'}": count for (decision, priority), count in priority_sources.items()},
            "selected_sources_with_multiple_annotation_polygons": int(multi_polygon_sources),
            "selected_sources_with_multiple_target_polygons": int(multi_target_sources),
            "selected_sources_with_multiple_classes": int(multi_class_sources),
            "selected_sources_with_partially_selected_polygons": int(partial_sources),
            "select_group_distribution": dict(sorted(source_group_stats.items())),
            "select_group_count": len(source_group_stats),
            "select_multi_source_group_count": sum(count > 1 for count in source_group_stats.values()),
            "select_sources_in_multi_source_groups": sum(count for count in source_group_stats.values() if count > 1),
            "known_near_duplicate_distribution": dict(sorted(near_group_stats.items())),
            "known_near_duplicate_group_count": sum(count > 1 for key, count in near_group_stats.items() if key != "UNKNOWN"),
            "select_sources_with_known_near_duplicate_group": sum(count for key, count in near_group_stats.items() if key != "UNKNOWN"),
            "select_resolution_distribution": dict(sorted(resolution.items())),
            "candidate_target_frame_edge_proximity": {f"{decision}:{threshold}": count for (decision, threshold), count in sorted(frame_edge_proximity.items())},
            "selected_target_risk_distribution": {f"{decision}:{kind}:{risk}": count for (decision, kind, risk), count in sorted(risks.items())},
            "historical_exposure_sources": [{"source_id": source_id, "exposure": exposure} for source_id, exposure in known_history_rows if source_by_id[source_id]["user_decision"] == "SELECT"],
            "all_sources_ui_risk": "UNKNOWN; no source-specific UI map exists",
            "mechanical_edge_semantics": "UNKNOWN; no source-specific mechanical-edge labels exist",
            "routine_shape_candidates_by_metadata": routine_shape,
            "confirmed_safe_for_generation": 0,
            "special_review_required": len(selected_source_rows) - routine_shape,
        },
        "proposal": {
            "status_values": {"PROPOSED": len(manifest_rows), "MANUALLY_APPROVED": 0, "AUTHORIZED_FOR_PAID_POST": 0},
            "source_count": len(manifest_rows), "proposed_request_count": sum(int(row["request_count_proposed"]) for row in manifest_rows),
            "class_request_counts": dict(sorted(proposed_class_counts.items())),
            "eligible_after_metadata_filters": eligible_counts,
            "class_quota_shortfall": {label: max(0, PROPOSAL_PER_CLASS - proposed_class_counts[label]) for label in PROPOSAL_CLASSES},
            "selection_method": "explicit private visual-screen list" if explicit_ids is not None else "deterministic area-rank stratification",
            "tip_curl_select_sources": class_stats["Tip curl"]["select_sources_with_class_present"],
            "known_historical_sources_excluded": sorted(KNOWN_HISTORY & set(source_by_id)),
            "manifest_sha256": sha_bytes(manifest_bytes),
            "group_deduplication": "one source per group/known near-duplicate key in the proposed batch",
        },
        "request_protocol": {
            "record_type": protocol["record_type"],
            "protocol_sha256": protocol_hash,
            "provider": protocol["request"]["base_url"],
            "model": protocol["request"]["model"],
            "endpoint": protocol["request"]["endpoint"],
            "method": protocol["request"]["method"],
            "multipart_fields": protocol["request"]["multipart_fields"],
            "n": protocol["request"]["n"],
            "size_quality_output_format": "NOT_SPECIFIED_IN_G2_PROTOCOL; PROVIDER_DEFAULT_UNKNOWN",
            "provider_input_resolution_limits": "UNKNOWN; no network probe performed",
            "automatic_retries": protocol["request"]["automatic_retries"],
            "fallbacks": protocol["request"]["fallbacks"],
            "redirects": protocol["request"]["redirects"],
            "parallel_calls": protocol["request"]["concurrency"],
        },
        "dry_run": {
            "mask_policy": "only explicitly selected Polygon IDs; target alpha 0, all other pixels/Polygons alpha 255",
            "roi_policy": "full native frame for proposal set; no crop, resize, padding, or synthetic edge fill",
            "provider_input_dimension_limits": "UNKNOWN; no network probe is permitted",
            "historical_request_dedup": "four known G1/G2 sources excluded; unrecorded history is not treated as proof of no call",
            "one_request_per_source_proposed": True, "automatic_retries": 0, "parallel_requests": 0,
            "production_locks_created": 0, "paid_post_authorizations": 0,
            "pixel_registration_or_trusted_pair_claim": "NONE; output geometry remains a separate gate",
        },
        "cost": {"state": "COST_UNKNOWN", "price_evidence": "No current attributable per-image price evidence was found in frozen G2 records.", "cost_ceiling_determinable": False},
        "safety": {"api_posts": 0, "gpu_training": 0, "test_image_access": 0, "autodl_startup": 0, "thesis_repository_access": 0},
        "output_paths": {"directory": str(output), "proposal_manifest": str(manifest_path), "preview_index": str(output / "index.html")},
    }
    # The proposal contact page remains private and has no network dependencies.
    cards = []
    for source_id, label, overlay_name, mask_href in preview_items:
        cards.append(
            f'<article><h2>{html.escape(source_id)} | {html.escape(label)}</h2>'
            f'<img loading="lazy" src="previews/{html.escape(overlay_name)}" alt="Source and Polygon overlay">'
            f'<img loading="lazy" src="{html.escape(mask_href)}" alt="Selected Polygon RGBA mask">'
            f'<p>PROPOSED; approval=NO; paid POST authorization=NO</p></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>G3-B2A Offline Proposal Preview</title>'
        '<style>body{font:14px sans-serif;margin:16px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article{border:1px solid #999;padding:8px}img{max-width:100%;height:auto;display:block;margin:6px 0}</style>'
        '<h1>G3-B2A Offline Proposal Preview</h1><p>PROPOSED only. This page does not authorize paid requests.</p><main>' + "\n".join(cards) + '</main>\n',
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--instance-review-csv", type=Path, required=True)
    parser.add_argument("--proposal-source-list", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
