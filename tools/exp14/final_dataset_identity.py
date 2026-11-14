"""Reconcile five pinned human exports with the actual Exp14 review files."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from stat import S_IREAD

import cv2
import numpy as np

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp13_cloud.g3.align_g3b2_batch import read_rgb, valid_support
from tools.exp13_cloud.g3.verify_train_export import CLASSES, EXPECTED_MANIFEST_SHA256, validate_annotation
from tools.exp14.final_v4 import ALL_INPUTS, A6, A7_ROOT, PHASE_A8
from tools.exp14.large_scale_review import FIELDS, IDENTITY, REASONS, STATES
from tools.exp14.postprocess_large_scale_repair import candidates, finished_candidates
from tools.exp14.prepare_large_scale_repair import PHASE, canonical_digest, hash_file, load_json, unique_index
from tools.exp14.targeted_v2 import target_raster
from tools.exp14.targeted_v2_review import reviewable
from tools.exp14.targeted_v3_remaining import BATCH

V4_CSV = "Exp14_A8_FINAL_V4_2026-11-14T06-51-11-506Z.csv"
V4_SHA = "57e94943882de94d0332a7714b5a732fbf5faafe07ee264263b244f00d73c7e8"
INPUTS = {**ALL_INPUTS, V4_CSV: V4_SHA}
A8_ROOT = "exp14_a8_final_v4_20261113"
A5_ROOT = "exp14_ai_full_review_20261112"
REVIEW_COUNTS = (530, 288, 23, 84, 31)


def checked(condition, message):
    g3.require(condition, message)


def archive(root, csv, audit):
    a8 = root.parent / A8_ROOT
    origins = {name: a8 / "input" / name for name in ALL_INPUTS}
    origins[V4_CSV] = csv
    for name, path in origins.items():
        hash_file(path, INPUTS[name])
    root.mkdir()
    data = {}
    for name, path in origins.items():
        dst = root / "inputs" / name
        g3.write_exclusive(dst, path.read_bytes())
        dst.chmod(S_IREAD)
        data[name] = {"sha256": hash_file(dst, INPUTS[name]), "original_path": path.as_posix()}
    external = root / "inputs" / audit.name
    g3.write_exclusive(external, audit.read_bytes())
    external.chmod(S_IREAD)
    g3.write_json_exclusive(root / "inputs" / "input_hashes.json", {"human_csvs": data,
        "external_precheck": {"file": audit.name, "sha256": hash_file(external, hash_file(audit))}})


def paths_for_records(private, scopes):
    old, a6, a7, a8 = [private / name for name in (PHASE, A6, A7_ROOT, A8_ROOT)]
    first = finished_candidates(old)
    second, third, fourth = (reviewable(path) for path in (a6, a7, a8))
    ai_roots = (private / A5_ROOT, a6, a7, a7 / BATCH, a8)
    ai_sets = [unique_index(load_json(path / "ai_review_results.json")["records"], "variant_id") for path in ai_roots]
    a7a, a7b = (set(ai_sets[i]) for i in (2, 3))
    checked(a7a.isdisjoint(a7b) and a7a | a7b == third.keys(), "Historical V3 batch sets differ")
    records = (first, second, {v: third[v] for v in a7a}, {v: third[v] for v in a7b}, fourth)
    generated = {}
    for path in (old, a6, a7, a8):
        scope = scopes[path.name]
        returned = candidates(path, {**scope, "historical_variants": scope.get("historical_variants", {})})
        for v, entry in returned.items():
            checked(v not in generated, "Duplicate generation identity across stages")
            generated[v] = entry
    return list(zip((old, a6, a7, a7, a8), records, ai_sets, strict=True)), generated


def read_historical_inputs(root):
    private = root.parent
    a8 = private / A8_ROOT
    receipt = load_json(a8 / "delivery_receipt.json")
    for path in ("scope.json", "authorization.json", "validated/final_candidate_registry.json"):
        hash_file(a8 / path, receipt["sha256"][path])
    scopes = {name: load_json(private / name / "scope.json") for name in (PHASE, A6, A7_ROOT, A8_ROOT)}
    for name, digest in scopes[A8_ROOT]["historical_scopes"].items():
        hash_file(private / name / "scope.json", digest)
    for path, digest in scopes[A8_ROOT]["historical_locks"].items():
        hash_file(private / path, digest)
    registry = load_json(a8 / "validated" / "final_candidate_registry.json")
    checked(registry["phase"] == PHASE_A8, "Unexpected A8 candidate registry phase")
    return scopes, unique_index(registry["sources"], "source_id")


def train_index(sources):
    path = Path(next(iter(sources.values()))["image"]).parents[2] / "frozen_split_manifest.csv"
    hash_file(path, EXPECTED_MANIFEST_SHA256)
    _, split = g3.read_csv(path)
    totals = Counter(r["split"].lower() for r in split)
    checked(totals == {"train": 668, "val": 154, "test": 147}, "Frozen split totals changed")
    train = unique_index([r for r in split if r["split"].lower() == "train"], "stem")
    checked(sources.keys() <= train.keys(), "Source absent from frozen TRAIN")
    return train, totals


def validate_source(source, frozen):
    image, annotation = (Path(source[k]) for k in ("image", "annotation"))
    checked(source["split"] == "train" and source["decision"] == "SELECT" and
            (source["group_id"], source["image_sha256"], source["annotation_sha256"]) ==
            (frozen["group_id"], frozen["sha256"], frozen["json_sha256"]), "Frozen TRAIN source identity differs")
    hash_file(image, source["image_sha256"])
    hash_file(annotation, source["annotation_sha256"])
    w, h = source["size"]
    labels, issues, data = validate_annotation(annotation, image.name, w, h, set(CLASSES))
    sid = source["source_id"]
    expected = [f"{sid}_{i:03d}" for i in range(len(labels))]
    checked(not issues and expected == source["all_polygon_ids"], "Professional annotation failed integrity checks")
    for pid in source["selected_polygon_ids"]:
        checked(labels[expected.index(pid)] == source["polygon_labels"][pid], "Frozen Polygon label differs")
    original = read_rgb(image)
    checked(original.shape[:2] == (h, w), "Original decode dimensions differ")
    return original, dict(zip(expected, labels, strict=True)), data["shapes"]


def reconcile_review(row, record, allowed_final=None):
    fields = FIELDS if allowed_final is None else (*FIELDS, "final_version_choice")
    checked(set(row) == set(fields) and all(isinstance(x, str) for x in row.values()) and
            all(row[k] == (record[k] or "") for k in IDENTITY), "CSV and asset review identities differ")
    checked(row["decision"] in STATES and row["reason"] in REASONS and row["whole_image_normal"] in {"", "YES"} and
            row["b_region_review"] in {"", "CONFIRMED"}, "Invalid human review state")
    explicit = json.loads(row["polygon_decisions"])
    target_ids = {p["polygon_id"] for p in record["polygons"]}
    checked(isinstance(explicit, dict) and explicit.keys() <= target_ids and
            all(v in STATES - {"UNREVIEWED"} for v in explicit.values()), "Invalid explicit Polygon identity or state")
    checked(not row["whole_image_normal"] or
            (row["decision"] == "PASS" and set(explicit.values()) <= {"PASS"}), "Whole-image and local review conflict")
    checked(not row["b_region_review"] or (record["aligned_sha256"] and
            (row["decision"] == "PASS" or "PASS" in explicit.values())), "Invalid human B confirmation")
    if allowed_final is not None:
        choice = row["final_version_choice"]
        checked(choice in {"", "EXCLUDE", *allowed_final} and (not choice or row["decision"] != "UNREVIEWED") and
                (choice != row["variant_id"] or row["decision"] == "PASS"), "Invalid final version choice")
    return explicit, {pid: explicit.get(pid, row["decision"]) for pid in sorted(target_ids)}


def selected_mapping(diagnostic, record):
    if diagnostic.get("selected_fit"):
        return diagnostic["selected_fit"]
    if diagnostic.get("reused_exact_aligned_sha256") != record["aligned_sha256"]:
        return None
    fits = [f for f in diagnostic.get("candidate_diagnostics", [])
            if f.get("method") == diagnostic["mapping_method"] and f.get("status") == "GEOMETRY_CANDIDATE"]
    return fits[0] if len(fits) == 1 else None


def audit_variant(base, source, source_data, record, entry, hashes):
    original, _, shapes = source_data
    folder = base / record["asset_directory"]
    row_id = canonical_digest({k: record[k] for k in IDENTITY if k not in {"review_id", "source_id", "version"}})
    checked(record["review_id"] == row_id and record["source_sha256"] == source["image_sha256"] and
            record["annotation_sha256"] == source["annotation_sha256"] and
            record["source_id"] == source["source_id"] == entry["source_id"] and record["version"] == entry["version"] and
            all(record[k] == entry[k] for k in ("request_sha256", "mask_sha256", "raw_sha256")), "Variant provenance differs")
    assets = {}
    for name, sha in record["file_sha256"].items():
        p = g3.safe_file(folder, name)
        hash_file(p, sha)
        assets[name] = {"path": p.as_posix(), "sha256": sha}
    for name, field in (("raw", "raw_sha256"), ("original", "source_sha256")):
        checked(assets[record[name]]["sha256"] == record[field], "Derived copy hash differs from manifest")
    for name, sha_field in (("request_path", "request_sha256"), ("mask", "mask_sha256"), ("raw", "raw_sha256")):
        p = Path(entry[name])
        sha = record[sha_field]
        if p not in hashes:
            hashes[p] = hash_file(p, sha)
        checked(hashes[p] == sha, "Origin path was claimed with a different hash")
    raw = read_rgb(folder / record["raw"])
    checked(raw.ndim == 3 and raw.size > 0, "Raw decode failed")
    geometry = load_json(folder / "geometry.json")
    checked(geometry["geometry_status"] == record["geometry_status"], "Geometry status differs")
    target_ids = source["selected_polygon_ids"]
    polygon = unique_index(record["polygons"], "polygon_id")
    checked(polygon.keys() == set(target_ids), "Variant Polygon set differs from frozen SELECT")
    valid, aligned, mapping_ok = None, None, False
    matrix, geometry_risks = None, []
    visible_loss, invalid = None, None
    if record["aligned_sha256"]:
        checked(assets["aligned.png"]["sha256"] == record["aligned_sha256"], "Aligned file hash differs")
        aligned = read_rgb(folder / "aligned.png")
        valid = read_rgb(folder / "valid.png")[..., 0]
        checked(aligned.shape == original.shape and valid.shape == original.shape[:2] and
                np.all((valid == 0) | (valid == 255)), "Aligned or Valid Mask has invalid dimensions or encoding")
        visible_loss = int(np.count_nonzero((valid == 0) & (cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) > 12)))
        invalid = int(np.count_nonzero(valid == 0))
        checked(visible_loss == record["unsupported_bright_original_pixels"] and
                np.isclose(1 - invalid / valid.size, record["valid_coverage"]), "Valid Mask support differs from record")
        fit = selected_mapping(geometry, record)
        if fit and fit["status"] == "GEOMETRY_CANDIDATE" and fit.get("matrix_output_to_source"):
            transform = np.asarray(fit["matrix_output_to_source"], dtype=np.float64)
            checked(transform.shape == (2, 3) and np.isfinite(transform).all() and
                    np.linalg.det(transform[:, :2]) > 0, "Stored registration matrix is invalid")
            matrix = transform.tolist()
            mapping_ok = True
        else:
            geometry_risks.append("STORED_MATRIX_UNAVAILABLE")
        if visible_loss:
            geometry_risks.append("VISIBLE_SOURCE_PIXELS_UNSUPPORTED")
    else:
        checked("aligned.png" not in assets, "Unexpected Aligned asset without review hash")
        geometry_risks.append("ALIGNMENT_UNVERIFIED")
    if "bounded.png" in assets:
        bounded = read_rgb(folder / "bounded.png")
        checked(bounded.shape == original.shape and "bounded_image" in geometry, "Bounded lacks source-grid identity")
    support = {}
    for pid in target_ids:
        p = polygon[pid]
        checked(p["class"] == source["polygon_labels"][pid], "Variant Polygon class differs")
        x0, y0, x1, y1 = p["box"]
        checked(0 <= x0 < x1 <= original.shape[1] and 0 <= y0 < y1 <= original.shape[0], "Invalid crop bounds")
        checked(np.array_equal(read_rgb(folder / p["before"]), original[y0:y1, x0:x1]), "Original crop pixels differ")
        raster, total = target_raster(source, shapes, [pid])
        mask = np.asarray(raster) > 0
        missing = int(np.count_nonzero((valid == 0) & mask)) if valid is not None else None
        checked(missing == p["unsupported_target_pixels"], "Polygon support differs from manifest")
        if aligned is not None:
            checked(np.array_equal(read_rgb(folder / p["after"]), aligned[y0:y1, x0:x1]), "Aligned crop pixels differ")
        support[pid] = {"polygon_id": pid, "class": p["class"], "pixels": total, "unsupported_pixels": missing,
                        "crop_box": p["box"], "before": assets[p["before"]], "after": assets.get(p["after"])}
    return {"assets": assets, "mapping_ok": mapping_ok, "matrix": matrix, "visible_loss": visible_loss,
            "invalid_pixels": invalid, "geometry_risks": geometry_risks, "support": support,
            "origin_paths": {k: Path(entry[k]).as_posix() for k in ("request_path", "mask", "raw")}}


def verify_existing_matrix_support(evidence):
    """Compare a fixed saved transform against the saved support without fitting."""
    cv2.setNumThreads(2)
    discrepancies, checked_count = [], 0
    for item in evidence["reviews"]:
        audit = item.get("audit", {})
        if not audit.get("matrix"):
            continue
        rec = item["record"]
        files = audit["assets"]
        raw = read_rgb(Path(files[rec["raw"]]["path"]))
        valid = read_rgb(Path(files["valid.png"]["path"]))[..., 0]
        predicted = valid_support(np.float64(audit["matrix"]), raw.shape[1::-1], valid.shape[1::-1])
        diff = predicted != valid
        checked_count += 1
        if not np.any(diff):
            continue
        sid = rec["source_id"]
        source = evidence["sources"][sid]
        raster, _ = target_raster(source, evidence["source_info"][sid]["shapes"], source["selected_polygon_ids"])
        local = int(np.count_nonzero((np.asarray(raster) > 0) & diff))
        discrepancies.append({"variant_id": rec["variant_id"], "review_id": rec["review_id"],
            "different_pixels": int(diff.sum()), "different_pixels_in_any_selected_polygon": local,
            "different_pixel_coordinates_yx": np.argwhere(diff).tolist(),
            "stored_valid_sha256": files["valid.png"]["sha256"]})
    return {"matrix_variants": checked_count, "exact_mask_matches": checked_count - len(discrepancies),
            "legacy_mask_boundary_differences": discrepancies, "training_approved": False}


def load_reviews(root):
    scopes, old_registry = read_historical_inputs(root)
    sources = scopes[PHASE]["sources"]
    train, split = train_index(sources)
    batches, generated = paths_for_records(root.parent, scopes)
    routes = {r["source_id"]: r for r in scopes[A8_ROOT]["routes"]}
    reviews, by_source, issues = [], defaultdict(list), []
    known, grouped = {}, defaultdict(list)
    for index, ((file, sha), (base, records, ai), expected) in enumerate(zip(INPUTS.items(), batches, REVIEW_COUNTS, strict=True)):
        path = root / "inputs" / file
        hash_file(path, sha)
        header, rows = g3.read_csv(path)
        checked(tuple(header) == (FIELDS if index != 4 else (*FIELDS, "final_version_choice")) and
                len(rows) == len(records) == len(ai) == expected, "Human batch schema or size differs")
        checked({r["variant_id"] for r in rows} == records.keys() == ai.keys(), "Human review variant set differs")
        for row in rows:
            record = records[row["variant_id"]]
            sid, vid = record["source_id"], record["variant_id"]
            checked(sid in sources and vid not in known, "Duplicate variant or out-of-scope historical source")
            item = {"record": record, "human": row, "batch": index + 1, "input_name": file, "base": base}
            known[vid] = item
            grouped[sid].append(item)
            try:
                older = [e["row"]["variant_id"] for e in routes[sid]["human_evidence"]]
                item["explicit"], item["resolved"] = reconcile_review(row, record, [vid, *older] if index == 4 else None)
                checked(all((ai[vid][k] or "") == (record[k] or "") for k in IDENTITY), "AI evidence identity differs")
                item["ai"] = ai[vid]
            except (KeyError, ValueError, TypeError) as exc:
                issues.append({"source_id": sid, "variant_id": vid, "step": "REVIEW", "error": str(exc)})
            by_source[sid].append(item)
            reviews.append(item)
    checked(len(reviews) == 956 and len({x["record"]["review_id"] for x in reviews}) == 956 and
            grouped.keys() == sources.keys() == old_registry.keys(), "Historical row or source set differs")
    last = [e for e in reviews if e["batch"] == 5]
    checked(Counter(e["human"]["decision"] for e in last) == {"PASS": 24, "HOLD": 5, "UNREVIEWED": 2} and
            sum(e["human"]["whole_image_normal"] == "YES" for e in last) == 24 and
            sum(e["human"]["b_region_review"] == "CONFIRMED" for e in last) == 24 and
            not any(e["human"]["final_version_choice"] for e in last), "V4 human CSV differs from the verified precheck")
    hashes = {}
    source_info = {}
    for i, (sid, entries) in enumerate(grouped.items(), 1):
        source = sources[sid]
        try:
            earlier = old_registry[sid]
            checked(earlier["original_sha256"] == source["image_sha256"] and
                    earlier["selected_polygon_ids"] == source["selected_polygon_ids"] and
                    set(earlier["all_review_ids"]) == {e["record"]["review_id"] for e in entries},
                    "A8 registry differs from persisted source and review identities")
            data = validate_source(source, train[sid])
            source_info[sid] = {"labels": data[1], "shapes": data[2]}
        except (KeyError, OSError, ValueError, TypeError) as exc:
            issues.append({"source_id": sid, "variant_id": "", "step": "SOURCE", "error": str(exc)})
            continue
        for item in entries:
            rec = item["record"]
            vid = rec["variant_id"]
            try:
                gen = rec.get("generation_variant_id", vid)
                entry = generated[gen]
                if rec.get("alignment_parent_review_id"):
                    parents = [e["record"] for e in entries if e["record"]["review_id"] == rec["alignment_parent_review_id"]]
                    checked(len(parents) == 1 and all(parents[0][k] == rec[k] for k in ("mask_sha256", "raw_sha256", "request_sha256")),
                            "Alignment parent provenance differs")
                item["audit"] = audit_variant(item["base"], source, data, rec, entry, hashes)
            except (AttributeError, KeyError, OSError, RuntimeError, ValueError, TypeError, cv2.error) as exc:
                issues.append({"source_id": sid, "variant_id": vid, "step": "ASSET", "error": str(exc)})
        if i % 100 == 0:
            print(f"Identity audit: {i}/{len(grouped)} TRAIN sources", flush=True)
    return sources, source_info, by_source, reviews, issues, scopes, old_registry, split
