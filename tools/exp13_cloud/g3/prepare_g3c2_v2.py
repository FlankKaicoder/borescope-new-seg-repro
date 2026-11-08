"""Prepare a private, frozen G3-C2 Prompt V2 request batch; never sends requests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from tools.exp13_cloud.g3.prepare_g3b2b_batch import REVIEW_FIELDS, csv_bytes
from tools.exp13_cloud.g3.verify_train_export import sha256_file

BASE_MANIFEST_SHA256 = "2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa"
FROZEN_REVIEW_SHA256 = "a8db2795566d95b204e16edf7ab5759fdc122bb9ff9a5d58ad8837b6a4cc52ec"
FROZEN_SPLIT_SHA256 = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
ELIGIBLE = {"556": "BURN_RESTORATION_V2", "640": "MATERIAL_MISSING_RESTORATION_V2",
            "645": "MATERIAL_MISSING_RESTORATION_V2", "650": "MATERIAL_MISSING_RESTORATION_V2"}
HELD = {"271": "HOLD_MASK_OVERLAPS_GREEN_CAMERA_UI"}
PROMPTS = {
    "Burn": ("BURN_RESTORATION_V2", "configs/exp13_cloud/g3/prompts/BURN_RESTORATION_V2.txt"),
    "Material missing": ("MATERIAL_MISSING_RESTORATION_V2", "configs/exp13_cloud/g3/prompts/MATERIAL_MISSING_RESTORATION_V2.txt"),
    "corrosion": ("CORROSION_RESTORATION_V2", "configs/exp13_cloud/g3/prompts/CORROSION_RESTORATION_V2.txt"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def unique_index(rows: list[dict[str, str]], field: str, label: str) -> dict[str, dict[str, str]]:
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {label} identity")
    return result


def prompt_entry(repo_root: Path, class_name: str) -> dict[str, str]:
    prompt_id, relative = PROMPTS[class_name]
    data = (repo_root / relative).read_bytes()
    if not data.endswith(b"\n") or data.endswith(b"\n\n") or data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"Prompt must be UTF-8 without BOM and have exactly one final LF: {relative}")
    return {"prompt_id": prompt_id, "path": relative, "file_sha256": digest(data),
            "wire_sha256": digest(data[:-1]), "wire_rule": "UTF-8; remove exactly one final LF byte"}


def build_plan(source_rows: list[dict[str, str]], review_rows: list[dict[str, str]],
               geometry_rows: list[dict[str, str]], export_rows: list[dict[str, str]],
               prompts: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sources = unique_index(source_rows, "source_id", "G3-B2 source")
    reviews = unique_index(review_rows, "source_id", "G3-C1 review")
    geometries = unique_index(geometry_rows, "source_id", "G3-C0 geometry")
    exports = unique_index(export_rows, "source_id", "TRAIN export")
    if set(reviews) != set(sources) or set(geometries) != set(sources):
        raise ValueError("G3-C1/G3-C0 source exact-set differs from frozen G3-B2 manifest")
    if not set(ELIGIBLE | HELD) <= set(sources) or not set(ELIGIBLE | HELD) <= set(exports):
        raise ValueError("Required G3-C2 source or TRAIN export row is missing")
    prepared: list[dict[str, str]] = []
    holds: list[dict[str, str]] = []
    for sid in [*ELIGIBLE, *HELD]:
        src, review, geometry, export = sources[sid], reviews[sid], geometries[sid], exports[sid]
        if src.get("split") != "train" or export.get("split") != "train":
            raise ValueError(f"Non-TRAIN source rejected: {sid}")
        if src.get("source_image_sha256") != export.get("export_image_sha256") or src.get("source_json_sha256") != export.get("export_json_sha256"):
            raise ValueError(f"Frozen TRAIN image/JSON identity mismatch: {sid}")
        if review.get("visual_review_status") != "NEEDS_RETRY":
            raise ValueError(f"Source is not a frozen NEEDS_RETRY case: {sid}")
        expected_class = {"556": "Burn", "640": "Material missing", "645": "Material missing",
                          "650": "Material missing", "271": "corrosion"}[sid]
        if src.get("class") != expected_class or geometry.get("class") != expected_class:
            raise ValueError(f"Frozen class identity mismatch: {sid}")
        if sid in HELD:
            holds.append({"source_id": sid, "class": expected_class, "status": "HOLD",
                          "reason_code": HELD[sid], "geometry_status": geometry.get("geometry_status", "UNKNOWN")})
            continue
        prompt = prompts[expected_class]
        if src.get("mask_sha256") == "" or src.get("source_image_sha256") == "":
            raise ValueError(f"Missing source or original mask identity: {sid}")
        row = dict(src)
        row.update({
            "prompt_id": prompt["prompt_id"], "prompt_file": prompt["path"],
            "prompt_file_sha256": prompt["file_sha256"], "prompt_wire_sha256": prompt["wire_sha256"],
            "v1_mask_path": src["mask_path"], "mask_path": f"masks/{sid}_repair_v2_rgba.png", "mask_sha256": "PENDING_COPY",
            "attempt_id": "g3c2_v2_attempt1_20261109", "attempt_number": "2",
            "v1_prompt_id": src["prompt_id"], "v1_prompt_file_sha256": src["prompt_file_sha256"],
            "v1_mask_sha256": src["mask_sha256"], "v1_raw_output_sha256": "",
            "v1_geometry_status": geometry.get("geometry_status", "UNKNOWN"),
            "repair_mask_basis": "SELECTED_POLYGON_MASK_PRESERVED; NO_UNSUPPORTED_PHYSICAL_EXPANSION",
            "live_authorized": "NO", "status": "PROPOSED", "manually_approved": "NO",
            "authorized_for_paid_post": "NO", "request_count_proposed": "1", "n": "1",
        })
        prepared.append(row)
    if {row["source_id"] for row in prepared} != set(ELIGIBLE):
        raise ValueError("Executable request exact-set mismatch")
    return prepared, holds


def validate_annotation(row: dict[str, str], export: dict[str, str], image_path: Path, json_path: Path) -> None:
    image_sha = sha256_file(image_path)
    json_sha = sha256_file(json_path)
    if image_sha != row["source_image_sha256"] or image_sha != export["export_image_sha256"]:
        raise ValueError(f"TRAIN source image hash mismatch: {row['source_id']}")
    if json_sha != row["source_json_sha256"] or json_sha != export["export_json_sha256"]:
        raise ValueError(f"TRAIN source JSON hash mismatch: {row['source_id']}")
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    with Image.open(image_path) as image:
        size = image.size
    if Path(payload.get("imagePath", "")).name != image_path.name:
        raise ValueError(f"JSON imagePath identity mismatch: {row['source_id']}")
    if (payload.get("imageWidth"), payload.get("imageHeight")) != size:
        raise ValueError(f"JSON dimensions mismatch: {row['source_id']}")
    shapes = payload.get("shapes")
    if not isinstance(shapes, list) or len(shapes) != int(export["instance_count"]):
        raise ValueError(f"TRAIN instance count mismatch: {row['source_id']}")
    selected = row["selected_polygon_ids"].split("|")
    for polygon_id in selected:
        suffix = polygon_id.rsplit("_", 1)[-1]
        if not suffix.isdigit() or int(suffix) >= len(shapes):
            raise ValueError(f"Selected Polygon ID cannot map to frozen JSON shape: {polygon_id}")
        shape = shapes[int(suffix)]
        if shape.get("shape_type") != "polygon" or shape.get("label") != row["class"]:
            raise ValueError(f"Selected Polygon class/type differs from frozen JSON: {polygon_id}")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Selected Polygon geometry is incomplete: {polygon_id}")


def prepare(repo_root: Path, private_root: Path, batch_root: Path, closeout_csv: Path,
            geometry_csv: Path, export_root: Path, output_root: Path) -> dict[str, Any]:
    if sha256_file(batch_root / "expanded_batch_manifest.csv") != BASE_MANIFEST_SHA256:
        raise ValueError("G3-B2 frozen Manifest SHA256 mismatch")
    if sha256_file(closeout_csv) != FROZEN_REVIEW_SHA256:
        raise ValueError("G3-C1 frozen review CSV SHA256 mismatch")
    export_manifest = export_root / "export_manifest.csv"
    frozen_manifest = export_root / "frozen_split_manifest.csv"
    if sha256_file(frozen_manifest) != FROZEN_SPLIT_SHA256:
        raise ValueError("G3-A frozen split Manifest SHA256 mismatch")
    if output_root.exists():
        raise FileExistsError(f"Output path already exists; preserve it and choose a new attempt directory: {output_root}")
    if private_root.resolve() not in output_root.resolve().parents:
        raise ValueError("Output must be under the private asset root")
    _, sources = read_csv(batch_root / "expanded_batch_manifest.csv")
    _, reviews = read_csv(closeout_csv)
    _, geometries = read_csv(geometry_csv)
    _, exports = read_csv(export_manifest)
    exports_by_id = unique_index(exports, "source_id", "TRAIN export")
    prompts = {name: prompt_entry(repo_root, name) for name in PROMPTS}
    rows, holds = build_plan(sources, reviews, geometries, exports, prompts)

    verified_files: dict[str, tuple[Path, Path, Path]] = {}
    for row in rows:
        sid = row["source_id"]
        source_image = batch_root / row["batch_image_path"]
        source_mask = batch_root / row["v1_mask_path"]
        source_json = export_root / exports_by_id[sid]["export_json_path"]
        if not source_image.is_file() or not source_mask.is_file() or not source_json.is_file():
            raise FileNotFoundError(f"Missing frozen TRAIN image, JSON, or selected Mask for {sid}")
        validate_annotation(row, exports_by_id[sid], source_image, source_json)
        if sha256_file(source_mask) != row["v1_mask_sha256"]:
            raise ValueError(f"V1 selected Mask hash mismatch: {sid}")
        verified_files[sid] = (source_image, source_mask, source_json)

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "inputs").mkdir()
    (output_root / "masks").mkdir()
    prompt_manifest = {"record_type": "G3C2_V2_PROMPT_MANIFEST_NOT_AUTHORIZATION",
                       "prompts": prompts}
    (output_root / "prompt_manifest.json").write_text(json.dumps(prompt_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        sid = row["source_id"]
        source_image, source_mask, _source_json = verified_files[sid]
        target_image = output_root / "inputs" / source_image.name
        target_mask = output_root / "masks" / f"{sid}_repair_v2_rgba.png"
        shutil.copyfile(source_image, target_image)
        shutil.copyfile(source_mask, target_mask)
        with Image.open(target_image) as image, Image.open(target_mask) as mask:
            image.load(); mask.load()
            if mask.mode != "RGBA" or mask.size != image.size:
                raise ValueError(f"V2 repair Mask dimensions/mode mismatch: {sid}")
            alpha = mask.getchannel("A").histogram()
            if alpha[0] <= 0 or alpha[255] <= 0 or sum(alpha[1:255]) != 0:
                raise ValueError(f"V2 repair Mask must be nonempty binary-alpha RGBA: {sid}")
        row["batch_image_path"] = f"inputs/{target_image.name}"
        row["batch_mask_path"] = f"masks/{target_mask.name}"
        row["mask_sha256"] = sha256_file(target_mask)
        row["v1_raw_output_sha256"] = json.loads((private_root / "g3b2_batch_generation_20261108" / "live" / sid / "response_manifest.json").read_text(encoding="utf-8"))["raw_output_sha256"]
    manifest_fields = list(dict.fromkeys(key for row in rows for key in row))
    (output_root / "expanded_batch_manifest.csv").write_bytes(csv_bytes(rows, manifest_fields))
    review_rows = [{"source_id": row["source_id"], "source_image_sha256": row["source_image_sha256"],
                    "prompt_id": row["prompt_id"], "prompt_file_sha256": row["prompt_file_sha256"],
                    "review_status": "PENDING", "review_notes": ""} for row in rows]
    (output_root / "review.csv").write_bytes(csv_bytes(review_rows, REVIEW_FIELDS))
    (output_root / "held_sources.json").write_text(json.dumps(holds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics = {
        "record_type": "G3C2_V1_FAILURE_DIAGNOSTICS_PRIVATE",
        "sources": [
            {"source_id": "556", "cause": "PROMPT_SEMANTIC_CONFLICT", "geometry": "GEOMETRY_CANDIDATE",
             "mask_action": "V1_SELECTED_POLYGON_PRESERVED", "basis": "V1 says preserve edges; reviewer says burn-distorted contour was incorrectly treated as normal. Mask already covers the selected burn and nearby boundary context."},
            {"source_id": "640", "cause": "INCOMPLETE_MATERIAL_RECONSTRUCTION; PROMPT_AMBIGUITY_POSSIBLE", "geometry": "ALIGNMENT_UNVERIFIED",
             "mask_action": "V1_SELECTED_POLYGON_PRESERVED", "basis": "Target polygon covers the reviewed edge-loss region. The evidence does not define a defensible expansion boundary; V2 clarifies completion while preserving real gaps. Raw-output visual result can be reviewed, but pair registration remains unsupported."},
            {"source_id": "645", "cause": "MODEL_RETAINED_OR_SMOOTHED_NOTCH; PROMPT_SEMANTIC_CONFLICT", "geometry": "GEOMETRY_CANDIDATE",
             "mask_action": "V1_SELECTED_POLYGON_PRESERVED", "basis": "Target polygon spans the reviewed notch. V1 forbids bridging gaps/inventing hidden parts without distinguishing material loss from a true opening; no separate physical boundary supports enlargement."},
            {"source_id": "650", "cause": "INCOMPLETE_RECONSTRUCTION_AND_WRONG_EDGE; PROMPT_SEMANTIC_CONFLICT", "geometry": "GEOMETRY_CANDIDATE",
             "mask_action": "V1_SELECTED_POLYGON_PRESERVED", "basis": "Selected polygon covers the reviewed broken edge. V2 makes supported edge continuation explicit; expansion beyond labeled damage lacks independent physical evidence."},
            {"source_id": "271", "cause": "UI_MASK_OVERLAP_AND_RESIDUAL_CORROSION", "geometry": "GEOMETRY_CANDIDATE",
             "mask_action": "HOLD", "basis": "Visible green timestamp/status characters intersect the selected mask. Excluding UI pixels reliably would need source-specific segmentation not supported by the current evidence; Prompt alone cannot guarantee UI preservation."},
        ],
        "pair_eligibility": "NONE_GRANTED_BY_THIS_PREPARATION",
    }
    (output_root / "failure_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "record_type": "G3C2_V2_OFFLINE_PREPARATION_NOT_AUTHORIZATION",
        "attempt_id": "g3c2_v2_attempt1_20261109", "source_ids": [row["source_id"] for row in rows],
        "hold_sources": holds, "max_posts": len(rows), "api_post_count": 0,
        "base_manifest_sha256": BASE_MANIFEST_SHA256, "frozen_review_sha256": FROZEN_REVIEW_SHA256,
        "frozen_split_manifest_sha256": FROZEN_SPLIT_SHA256,
        "new_manifest_sha256": sha256_file(output_root / "expanded_batch_manifest.csv"),
        "prompt_manifest_sha256": sha256_file(output_root / "prompt_manifest.json"),
        "review_csv_sha256": sha256_file(output_root / "review.csv"),
        "cost_status": "COST_UNKNOWN", "paid_authorization_created": False,
        "trusted_training_pair_pass": 0,
    }
    (output_root / "preparation_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--closeout-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.repo_root, args.private_root, args.batch_root, args.closeout_csv,
                     args.geometry_csv, args.export_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
