"""Offline, identity-bound Exp17-E Polygon review. No model, registration or admission."""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from datetime import datetime, timezone

from PIL import Image

SCHEMA = "exp17e_a10_polygon_user_review_v1"
DECISIONS = ("PENDING", "ACCEPT_THIS_POLYGON_FOR_J2", "REJECT_THIS_POLYGON_FOR_J2",
             "HOLD_NEED_MORE_EVIDENCE", "ASSET_SELECTION_REQUIRED")
IDENTITY_KEYS = ("candidate_id", "source_id", "group_id", "polygon_id", "class_id", "class_name",
                 "source_image_sha256", "annotation_sha256", "n_asset_sha256", "asset_type",
                 "repair_version", "n_review_id", "n_variant_id", "matrix_identity_sha256",
                 "source_valid_mask_sha256", "n_valid_mask_sha256", "accepted_region_scope")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError("Non-finite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def identity(row):
    return {k: row[k] for k in IDENTITY_KEYS}


def fresh(row, package_id):
    return {"schema": SCHEMA, "package_id": package_id, **identity(row),
            "asset_identity": identity(row), "binding_sha256": row["binding_sha256"],
            "decision": "PENDING", "reviewer": "", "reviewed_at": None, "user_notes": "",
            "historical_rejection_override_reason": "", "review_role": "UNREVIEWED",
            "training_approved": False, "independent_data_gate": "NOT_GRANTED"}


def validate_review(row, value, package_id):
    require(isinstance(value, dict) and set(value) == set(fresh(row, package_id)), "Unexpected or missing review fields")
    require(value["schema"] == SCHEMA and value["package_id"] == package_id, "Wrong package/schema")
    require(value["binding_sha256"] == row["binding_sha256"], "Stale binding")
    require(digest({k: value[k] for k in IDENTITY_KEYS}) == digest(identity(row)), "Changed redundant asset identity")
    require(digest(value["asset_identity"]) == digest(identity(row)), "Changed asset/matrix/Valid Mask/region identity")
    require(value["decision"] in DECISIONS, "Unknown decision")
    require(value["training_approved"] is False and value["independent_data_gate"] == "NOT_GRANTED", "Review cannot grant Data Gate")
    require(value["review_role"] in ("UNREVIEWED", "USER_LOCAL_REVIEW_UNVERIFIED"), "Unknown reviewer role")
    for key in ("reviewer", "user_notes", "historical_rejection_override_reason"):
        require(isinstance(value[key], str), "Review text must be text")
    if value["decision"] != "PENDING":
        require(value["review_role"] == "USER_LOCAL_REVIEW_UNVERIFIED" and value["reviewer"].strip(), "Reviewer attribution required")
        require(isinstance(value["reviewed_at"], str), "Review time required")
        datetime.fromisoformat(value["reviewed_at"].replace("Z", "+00:00"))
        require(value["user_notes"].strip(), "Decision explanation required")
    if value["decision"] == "ACCEPT_THIS_POLYGON_FOR_J2":
        require(row["mapping_verified"] is True and row["support_complete"] is True, "ACCEPT blocked: no verified/full target mapping")
        require(row["identity_resolved"] is True, "ACCEPT blocked: unresolved identity")
        require(row["accepted_region_scope"]["scope"] == "THIS_POLYGON_ONLY", "Only this Polygon can be accepted")
        if row.get("historical_user_rejected"):
            require(value["historical_rejection_override_reason"].strip(), "New explicit reason required for historical user rejection")
    return True


def validate_export(bundle, payload):
    require(isinstance(payload, dict) and payload.get("schema") == SCHEMA and payload.get("package_id") == bundle["package_id"], "Wrong import package")
    require(payload.get("training_approved") is False and payload.get("independent_data_gate") == "NOT_GRANTED", "Import cannot grant Data Gate")
    require(isinstance(payload.get("records"), list), "Missing import records")
    rows = {r["candidate_id"]: r for r in bundle["records"]}
    seen = set()
    for value in payload["records"]:
        key = value.get("candidate_id")
        require(key in rows and key not in seen, "Unknown/duplicate candidate")
        validate_review(rows[key], value, bundle["package_id"])
        seen.add(key)
    return {"validated_records": len(seen), "new_user_accepts": sum(v["decision"] == DECISIONS[1] for v in payload["records"]),
            "formal_trusted_pairs": 0, "independent_data_gate": "NOT_GRANTED"}


def csv_text(records, columns):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(columns)
    for row in records:
        writer.writerow([json.dumps(row[k], ensure_ascii=False, separators=(",", ":"), allow_nan=False) for k in columns])
    return "\ufeff" + stream.getvalue()


def parse_csv(text, columns):
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff"), newline=""))
    header = next(reader, [])
    require(header == list(columns) and len(set(header)) == len(header), "CSV header identity mismatch")
    records = []
    for line in reader:
        require(len(line) == len(header), "CSV column count mismatch")
        records.append(dict(zip(header, (json.loads(cell) for cell in line))))
    return records


def transform(points, matrix):
    result = []
    for x, y in points:
        z = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        require(math.isfinite(z) and abs(z) > 1e-12, "Invalid historical projective mapping")
        result.append([(matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / z,
                       (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / z])
    return result


def context_roi(points, size):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    margin = max(12, round(max(max(xs) - min(xs), max(ys) - min(ys)) * .25))
    return [max(0, math.floor(min(xs)) - margin), max(0, math.floor(min(ys)) - margin),
            min(size[0], math.ceil(max(xs)) + margin + 1), min(size[1], math.ceil(max(ys)) + margin + 1)]


def separate_history_roles(history):
    """Display projection only; the complete raw history remains unchanged in the bundle."""
    ai = []
    def visit(value, path):
        if isinstance(value, dict):
            projected = {}
            for key, item in value.items():
                where = path + [key]
                if key.startswith("ai_") or "codex" in key.lower() or key in ("old_visual_note", "historical_d0_trace", "historical_d0_result"):
                    ai.append({"historical_key": ".".join(map(str, where)), "value": item})
                else:
                    projected[key] = visit(item, where)
            return projected
        if isinstance(value, list): return [visit(v, path + [i]) for i, v in enumerate(value)]
        return value
    return visit(history, []), ai


def prepare(queue):
    """Normalize the census/expansion schema without changing its classifications."""
    sources = {str(s["source_id"]): dict(s) for s in queue["sources"]}
    require(len(sources) == len(queue["sources"]), "Duplicate source")
    for source in sources.values():
        source.setdefault("original_path", source.get("source_image_path"))
        source["authoritative_n"] = dict(source["authoritative_n"])
        source["authoritative_n"].setdefault("type", source["authoritative_n"].get("asset_type"))
    rows = []
    for polygon in queue["polygons"]:
        p = dict(polygon)
        p["source_id"] = str(p["source_id"])
        source = sources[p["source_id"]]
        sm, pm = source["source_mapping"], p["polygon_mapping"]
        n = source["authoritative_n"]
        require(pm["n_asset_sha256"] == sm["n_asset_sha256"] == n["sha256"], "Different A10 N in mapping and census")
        for key, expected in (("n_asset_sha256", n["sha256"]), ("n_review_id", n["review_id"]),
                              ("n_variant_id", n["variant_id"]), ("repair_version", n["version"])):
            require(key not in p or p[key] == expected, "Polygon refers to another A10 N version")
        p["candidate_id"] = p.get("candidate_id", p["source_id"] + ":" + p["polygon_id"] + ":" + n["sha256"])
        p.update(source_image_sha256=source["source_image_sha256"], annotation_sha256=source["annotation_sha256"],
                 n_asset_sha256=n["sha256"], asset_type=n["type"], repair_version=n["version"],
                 n_review_id=n["review_id"], n_variant_id=n["variant_id"],
                 matrix_identity_sha256=sm.get("matrix_identity_sha256"),
                 source_valid_mask_sha256=(sm.get("source_valid_mask") or {}).get("sha256"),
                 n_valid_mask_sha256=(sm.get("n_valid_mask") or {}).get("sha256"),
                 mapping_verified=sm.get("mapping_verified") is True and pm.get("mapping_verified") is True,
                 support_complete=pm.get("support_complete") is True)
        p.setdefault("identity_resolved", p.get("primary_status") != "IDENTITY_UNRESOLVED")
        p.setdefault("historical_user_rejected", False)
        versions = p.get("historical_all_versions", p.get("history", {}).get("other_versions", []))
        p["historical_versions"] = sorted({v["version"] for v in versions if v.get("version")})
        p["historical_d_filter"] = {"INHERITED_REGION_PAIR_CANDIDATE": "D357", "ASSET_BINDING_ONLY_PENDING": "D176",
                                    "TARGETED_EVIDENCE_REVIEW": "D12"}.get(p.get("historical_d_category"), p.get("historical_d_category"))
        p["history_display"], p["historical_ai_display"] = separate_history_roles(p.get("history", {}))
        require(p["coordinates"] and p["group_id"] == source["group_id"], "Polygon identity or source group differs")
        if p["mapping_verified"]:
            require(pm.get("matrix_identity_sha256", sm["matrix_identity_sha256"]) == sm["matrix_identity_sha256"], "Polygon uses another matrix")
            projected = transform(p["coordinates"], sm["source_to_n"])
            require(len(projected) == len(pm["n_coordinates"]) and all(abs(a - b) <= 1e-6 for p1, p2 in zip(projected, pm["n_coordinates"]) for a, b in zip(p1, p2)), "N coordinates differ from pinned historical mapping")
        original_roi = pm.get("source_roi_xyxy") or context_roi(p["coordinates"], source["source_size"])
        n_roi = (pm.get("n_roi_xyxy") or context_roi(pm["n_coordinates"], source["n_size"])) if p["mapping_verified"] else None
        p["roi_available"] = bool(n_roi and n_roi[0] < n_roi[2] and n_roi[1] < n_roi[3])
        p["accepted_region_scope"] = {"scope": "THIS_POLYGON_ONLY", "source_coordinates": p["coordinates"],
                                      "n_coordinates": pm.get("n_coordinates") if p["mapping_verified"] else None,
                                      "source_roi_xyxy": original_roi if p["mapping_verified"] else None,
                                      "n_roi_xyxy": n_roi}
        p["binding_sha256"] = digest(identity(p))
        p["asset_identity"] = identity(p)
        rows.append(p)
    require(len({r["candidate_id"] for r in rows}) == len(rows), "Duplicate Polygon identity")
    return sources, rows


def build(queue_path, output):
    queue_sha256 = sha(queue_path)
    queue = load(queue_path)
    sources, rows = prepare(queue)
    output.mkdir()
    (output / "assets").mkdir(); (output / "roi").mkdir()
    manifest = {}

    def copy(path, expected):
        path = Path(path).resolve(strict=True)
        require(expected and sha(path) == expected, "Source asset identity changed")
        relative = "assets/" + expected + path.suffix.lower()
        target = output / relative
        if not target.exists():
            with path.open("rb") as src, target.open("xb") as out:
                shutil.copyfileobj(src, out)
        require(sha(target) == expected, "Copied bytes differ")
        manifest[relative] = {"sha256": expected, "bytes": target.stat().st_size, "kind": "EXACT_FROZEN_COPY"}
        return relative

    def crop(relative, roi, name):
        with Image.open(output / relative) as im:
            require(0 <= roi[0] < roi[2] <= im.width and 0 <= roi[1] < roi[3] <= im.height, "ROI beyond native image")
            im.crop(roi).save(output / name)
        manifest[name] = {"sha256": sha(output / name), "bytes": (output / name).stat().st_size, "kind": "DISPLAY_ONLY_NATIVE_CROP"}
        return name

    for sid, source in sources.items():
        n = source["authoritative_n"]
        files = {"original": copy(source["original_path"], source["source_image_sha256"]),
                 "normal": copy(n["path"], n["sha256"]), "annotation": copy(source["annotation_path"], source["annotation_sha256"])}
        for name, size in (("original", source["source_size"]), ("normal", source["n_size"])):
            with Image.open(output / files[name]) as im:
                require(list(im.size) == size, "Native image dimensions changed")
        sm = source["source_mapping"]
        for kind in ("source_valid_mask", "n_valid_mask"):
            entry = sm.get(kind)
            if sm.get("mapping_verified") and entry:
                files[kind] = copy(entry["path"], entry["sha256"])
        source["files"] = files
        source["all_polygons"] = [{"polygon_id": r["polygon_id"], "class_name": r["class_name"], "coordinates": r["coordinates"]} for r in rows if r["source_id"] == sid]
        if sm.get("mapping_verified"):
            source["n_all_polygons"] = [{**p, "coordinates": transform(p["coordinates"], sm["source_to_n"])} for p in source["all_polygons"]]
    for i, row in enumerate(rows):
        row["roi_files"] = {}
        if row["mapping_verified"] and row["roi_available"]:
            source = sources[row["source_id"]]
            for kind, roi_key in (("original", "source_roi_xyxy"), ("normal", "n_roi_xyxy"),
                                  ("source_valid_mask", "source_roi_xyxy"), ("n_valid_mask", "n_roi_xyxy")):
                if kind in source["files"]:
                    row["roi_files"][kind] = crop(source["files"][kind], row["accepted_region_scope"][roi_key], "roi/" + str(i).zfill(4) + "_" + kind + ".png")
    template_path = Path(__file__).with_suffix(".html")
    require(sha(queue_path) == queue_sha256, "Queue changed during package build")
    package_id = digest({"queue_sha256": queue_sha256, "bindings": [r["binding_sha256"] for r in rows],
                         "builder_sha256": sha(__file__), "html_sha256": sha(template_path), "schema": SCHEMA})
    bundle = {"schema": SCHEMA, "package_id": package_id, "queue_sha256": sha(queue_path), "sources": sources,
              "records": rows, "created_at": datetime.now(timezone.utc).isoformat(), "formal_trusted_pairs": 0,
              "decisions": DECISIONS, "identity_keys": IDENTITY_KEYS, "summary": queue.get("summary", {})}
    templates = [fresh(row, package_id) for row in rows]
    bundle["review_columns"] = list(templates[0])
    encoded = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    with (output / "index.html").open("x", encoding="utf-8", newline="\n") as f:
        f.write(template_path.read_text(encoding="utf-8").replace("__BUNDLE__", encoded))
    write(output / "review_bundle.json", bundle)
    write(output / "new_review_template.json", {"schema": SCHEMA, "package_id": package_id, "records": templates,
                                               "training_approved": False, "independent_data_gate": "NOT_GRANTED"})
    with (output / "new_review_template.csv").open("x", encoding="utf-8", newline="") as f:
        f.write(csv_text(templates, bundle["review_columns"]))
    write(output / "file_manifest.json", manifest)
    receipt = {"package_id": package_id, "queue_sha256": sha(queue_path), "index_sha256": sha(output / "index.html"),
               "bundle_sha256": sha(output / "review_bundle.json"), "sources": len(sources), "polygons": len(rows),
               "independent_source_groups": len({s["group_id"] for s in sources.values()}), "media_and_annotation_files": len(manifest),
               "verified_local_mapping_polygons": sum(r["mapping_verified"] for r in rows),
               "new_user_accepts": 0, "formal_trusted_pairs": 0, "assets_bytes": sum(v["bytes"] for v in manifest.values()),
               "status": "OFFLINE_PREPARED_PENDING_USER_AND_INDEPENDENT_DATA_GATE"}
    write(output / "package_receipt.json", receipt)
    with (output / "README.txt").open("x", encoding="utf-8") as f:
        f.write("Open index.html offline. Default shows items needing action, grouped by source. A10 N is unique per source. Unverified mappings have independent full images only. New decisions never overwrite history or grant Data Gate. Export before closing; JSON and typed CSV imports validate every asset identity atomically. CSV cells use JSON types to preserve Chinese multiline notes and identity values.\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.queue, args.output), indent=2))


if __name__ == "__main__":
    main()
