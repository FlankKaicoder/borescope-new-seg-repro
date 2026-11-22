"""Inherit hash-bound historical Polygon review evidence into a new Data Gate queue.

This module reads metadata only. It does not decode/rasterize images, fit a
registration, call a model, choose another version, or create human approvals.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

AUDIT_CANDIDATES_SHA = "11244f84b9fe0025a79b49e8c362c3c838d7d752a146071edc0a7712f0b84472"
AUDIT_FILES_SHA = "65398fafec214019839d10bab0f55d835955db1d52793ae0ee9aaecdd515e8fc"
SEMANTIC_EVIDENCE_SHA = "3b15d42ee7fa61e56b05cd31d164b9a60da055341cb6373088292aa9d2f4ed74"
CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
COMPLETE = "EXP17_D_HISTORICAL_RECONCILIATION_COMPLETE_PENDING_CHATGPT_DATA_GATE"
HOLD = "EXP17_D_RECONCILIATION_HOLD_WITH_EXPLICIT_BLOCKERS"
INHERITED = "INHERITED_REGION_PAIR_CANDIDATE"
ASSET_ONLY = "ASSET_BINDING_ONLY_PENDING"
TARGETED = "TARGETED_EVIDENCE_REVIEW"
EXCLUDED = "EXPLICITLY_EXCLUDED_FOR_J2"
UNKNOWN = "UNKNOWN_HOLD"
RULES = {
    "identity": "Join exact review_id, source, version, variant, original/JSON/raw/aligned/request/edit SHA, Polygon and historical affine/Valid support. Reuse prior 3998-file audit; do not rehash/decode images.",
    "polygon": "Reuse A9 explicit.get(pid, overall_decision) only for that review identity's frozen target IDs. b_region_review must also be CONFIRMED. Whole-image PASS alone does not establish a region Pair.",
    "scope": "Every inherited claim is limited to the hash-bound selected Polygon. Preserve region-only and explicit textual restrictions separately from coordinate-defined crop/FOV restrictions.",
    "selection": "Keep A9 recommended unchanged. Ten alternates and all other historical versions are evidence context, never silently substitute them.",
    "explicit_exclusion": "After identity checking, an explicit user EXCLUDE takes precedence over a same-row Polygon PASS and region confirmation.",
    "asset_binding": "A9 pending pixel asset selection remains pending. V4 final_version_choice blank does not negate existing same-version regional semantic acceptance.",
    "coarse_correspondence": "Reuse audited same-version source-grid affine, Valid asset and full target support; no Formal TRE, G2 or repeated independent-boundary review is required for inheritance.",
    "nonautomatic_rejection": "Frame-edge, modest texture/brightness differences, source-level version flags, other GT elsewhere in ROI and D0 ROI/context thresholds are not automatic target failures.",
    "conflicts": "Specific same-version unresolved target/part/serious UI evidence may create a targeted queue without revoking the historical user decision. Only explicit local evidence can exclude a selected version.",
    "independence": "Count selected Polygons and unique frozen group_id separately. Multiple versions or Polygons never create extra independent source groups.",
    "gate": "Inherited candidates are eligible for independent J2 Data Gate review, not formal Data Gate PASS or training approval. Current new human G1 fields remain a separate historical snapshot.",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read_csv(path):
    """Record actual physical line spans even for quoted multiline notes."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        previous = reader.line_num
        result = []
        for ordinal, fields in enumerate(reader, 1):
            need(len(fields) == len(header), "CSV field count differs")
            row = dict(zip(header, fields, strict=True))
            result.append({"row": row, "data_row_1based": ordinal,
                           "record_row_including_header_1based": ordinal + 1,
                           "physical_line_start": previous + 1, "physical_line_end": reader.line_num,
                           "row_sha256": canonical(row)})
            previous = reader.line_num
        return result


def indexed(rows, key):
    need(len(rows) == len({r[key] for r in rows}), f"Duplicate {key}")
    return {r[key]: r for r in rows}


def csv_values(path):
    return [r["row"] for r in read_csv(path)]


def resolve_polygon(human, resolved, pid):
    """A9's frozen target-set semantics, with explicit local decisions first."""
    need(pid in resolved, "Polygon absent from this exact review's frozen target set")
    explicit = json.loads(human["polygon_decisions"])
    state = explicit.get(pid, human["decision"])
    need(state == resolved[pid], "A9 resolved Polygon state differs from original CSV")
    return state, "EXPLICIT" if pid in explicit else "OVERALL_DEFAULT"


def historical_state(human, local_state, identity_ok=True, text_scope=False):
    if not identity_ok:
        return "ASSET_IDENTITY_MISMATCH"
    if human.get("final_version_choice") == "EXCLUDE":
        return "HISTORICAL_USER_REJECTED"
    if local_state == "PASS" and human["b_region_review"] == "CONFIRMED":
        return ("HISTORICAL_USER_ACCEPTED_WITH_SCOPE" if not human["whole_image_normal"] or text_scope
                else "HISTORICAL_USER_ACCEPTED")
    if local_state == "RETRY_REPAIR":
        return "HISTORICAL_USER_REJECTED"
    if local_state == "RETRY_ALIGNMENT":
        return "HISTORICAL_ALIGNMENT_UNRESOLVED"
    if local_state in {"HOLD", "UNREVIEWED", "PASS"}:
        return "HISTORICAL_USER_PENDING"
    return "NO_HISTORICAL_DECISION"


def require_semantic_rules(value):
    need(isinstance(value,dict) and isinstance(value.get("rules"),list) and value["rules"],
         "Exact semantic evidence rulebook is mandatory and cannot be empty")
    need(Counter(r.get("type") for r in value["rules"]) == {"ACCEPTED_WITH_SCOPE":37,"TARGETED_REVIEW":12},
         "Frozen 49-rule evidence population differs")
    return value


def disposition(history_status, pending_asset, rules, identity_errors):
    if identity_errors:
        return UNKNOWN
    if history_status == "HISTORICAL_USER_REJECTED" or any(r["type"] == "EXPLICIT_EXCLUDED" for r in rules):
        return EXCLUDED
    if history_status not in {"HISTORICAL_USER_ACCEPTED", "HISTORICAL_USER_ACCEPTED_WITH_SCOPE"}:
        return UNKNOWN
    if any(r["type"] == "TARGETED_REVIEW" for r in rules):
        return TARGETED
    return ASSET_ONLY if pending_asset else INHERITED


def count(rows):
    return {"polygons": len(rows), "source_ids": len({r["source_id"] for r in rows}),
            "independent_source_groups": len({r["group_id"] for r in rows})}


def csv_write(path, rows):
    need(rows, "Cannot publish empty private matrix")
    keys = list(rows[0])
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            need(set(row) == set(keys), "Private matrix schema differs")
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":")) if isinstance(v, (dict,list))
                             else v for k,v in row.items()})


class Metadata:
    def __init__(self):
        self.records = {}

    def check(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        key = path.as_posix()
        if key not in self.records:
            self.records[key] = {"path": key, "bytes": path.stat().st_size, "sha256": sha(path)}
        item = self.records[key]
        need(expected is None or item["sha256"] == expected, "Historical metadata SHA256 differs")
        return item

    def postcheck(self):
        for item in self.records.values():
            need(sha(item["path"]) == item["sha256"], "Historical metadata changed during reconciliation")


def audit(args):
    metadata = Metadata()
    need(args.semantic_evidence is not None, "Exact semantic evidence rulebook is required")
    metadata.check(args.semantic_evidence, SEMANTIC_EVIDENCE_SHA)
    semantic = require_semantic_rules(load(args.semantic_evidence))
    code_root = Path(__file__).resolve().parents[2]
    source_identity = {str(p.relative_to(code_root)).replace("\\", "/"): metadata.check(p)["sha256"] for p in
                       (Path(__file__), Path(__file__).with_name("test_reconcile.py"),
                        code_root/"tools/exp14/final_dataset_identity.py", code_root/"tools/exp14/final_dataset_selection.py")}
    a9 = args.a9_root
    receipt = load(a9/"handoff/delivery_receipt.json")
    for name, expected in receipt["private_sha256"].items():
        # These are manifests/CSV/JSON/HTML metadata, never image/model bytes.
        metadata.check(a9/name, expected)
    metadata.check(args.exp17_audit/"candidates.json", AUDIT_CANDIDATES_SHA)
    metadata.check(args.exp17_audit/"verified_files.json", AUDIT_FILES_SHA)
    candidates = load(args.exp17_audit/"candidates.json")["candidates"]
    prior_files = {Path(r["path"]).resolve().as_posix(): r for r in load(args.exp17_audit/"verified_files.json")["files"]}
    need(len(candidates) == 545 and len(prior_files) == 3998, "Accepted Exp17 audit population differs")
    source_rows = indexed(csv_values(a9/"registry/source_master_registry.csv"), "source_id")
    variants = indexed(csv_values(a9/"registry/variant_master_registry.csv"), "review_id")
    polygons = indexed(csv_values(a9/"registry/polygon_master_registry.csv"), "polygon_id")
    manifest = csv_values(a9/"registry/b_pair_candidate_manifest.csv")
    by_poly = indexed(candidates, "polygon_id")
    need(len(variants) == 956 and len(manifest) == 545, "A9 identity counts differ")
    inputs = load(a9/"inputs/input_hashes.json")["human_csvs"]
    human_rows = {}; batch_stats = []; all_human = []
    for name, expected in inputs.items():
        path = a9/"inputs"/name
        metadata.check(path, expected["sha256"])
        metadata.check(Path(expected["original_path"]), expected["sha256"])
        records = read_csv(path)
        batch_stats.append({"input":name,"sha256":expected["sha256"],"review_rows":len(records),
                            "physical_lines": max(r["physical_line_end"] for r in records)})
        for item in records:
            raw = item["row"]; rid = raw["review_id"]
            need(rid not in human_rows and rid in variants, "Historical review identity duplicated or unknown")
            v = variants[rid]
            for k, other in (("source_id","source_id"),("variant_id","variant_id"),("version","version"),
                             ("source_sha256","original_sha256"),("annotation_sha256","annotation_sha256"),
                             ("request_sha256","request_sha256"),("mask_sha256","mask_sha256"),
                             ("raw_sha256","raw_sha256"),("aligned_sha256","aligned_sha256")):
                need(raw[k] == v[other], "CSV and A9 version identity differ")
            need(v["review_input"] == name and raw["decision"] == v["human_decision"]
                 and raw["notes"] == v["human_notes"] and raw["reason"] == v["human_reason"]
                 and raw["b_region_review"] == v["b_region_review"] and raw["whole_image_normal"] == v["whole_image_normal"]
                 and json.loads(raw["polygon_decisions"]) == json.loads(v["human_polygon_decisions"])
                 and raw.get("final_version_choice", "") == v["final_version_choice"],
                 "A9 human decision differs from original export")
            trace = {k:item[k] for k in item if k != "row"}
            trace.update({"csv_path":path.resolve().as_posix(),"original_csv_path":expected["original_path"],
                          "csv_sha256":expected["sha256"],"review_input":name,"review_id":rid})
            human_rows[rid] = {"raw":raw,"trace":trace}
            all_human.append({"trace":trace,"raw":raw,"a9_variant_identity_status":v["identity_status"]})
    need(len(human_rows) == len(variants) == 956, "Five CSV review identity sets differ")
    for rule in semantic["rules"]:
        need(rule["type"] in {"TARGETED_REVIEW","EXPLICIT_EXCLUDED","ACCEPTED_WITH_SCOPE"}
             and rule["review_id"] in variants and rule.get("source_locator") and rule.get("reason_code"),
             "Invalid exact-version semantic evidence rule")
        if rule.get("source_document") and rule.get("source_sha256"):
            metadata.check(Path(rule["source_document"]), rule["source_sha256"])
            if Path(rule["source_document"]).suffix.lower() == ".csv":
                matches = [r for r in read_csv(rule["source_document"]) if r["row"].get("review_id") == rule["review_id"]
                           and (not r["row"].get("polygon_id") or r["row"]["polygon_id"] == rule.get("polygon_id"))]
                need(len(matches) == 1, "Semantic quote does not bind one exact source CSV row")
                loc = rule["source_locator"]
                need(all(loc.get(k, matches[0][k]) == matches[0][k] for k in ("physical_line_start","physical_line_end"))
                     and loc.get("data_row_ordinal",matches[0]["data_row_1based"]) == matches[0]["data_row_1based"],
                     "Semantic evidence source row/line locator differs")
        v = variants[rule["review_id"]]
        need(all(rule.get(k,v[k]) == v[k] for k in ("source_id","variant_id","version"))
             and (not rule.get("polygon_id") or rule["polygon_id"] in json.loads(v["resolved_polygon_decisions"])),
             "Semantic evidence version/Polygon differs")
    contexts = defaultdict(list)
    events = []
    for rid, v in variants.items():
        raw = human_rows[rid]["raw"]
        states = json.loads(v["resolved_polygon_decisions"])
        for pid in states:
            local, basis = resolve_polygon(raw, states, pid)
            state = historical_state(raw, local, v["identity_status"] == "VERIFIED")
            if raw["decision"] == "UNREVIEWED" and v["ai_decision"] == "PASS":
                state = "HISTORICAL_AI_ONLY_PASS"
            event = {"source_id":v["source_id"],"group_id":source_rows[v["source_id"]]["group_id"],"polygon_id":pid,
                     "review_id":rid,"variant_id":v["variant_id"],"version":v["version"],"polygon_state":local,
                     "polygon_state_basis":basis,"historical_status":state,"overall_user_decision":raw["decision"],
                     "b_region_review":raw["b_region_review"],"trace":human_rows[rid]["trace"]}
            contexts[pid].append(event); events.append(event)
    selected = []; alternates = []; identity_errors = []; records = {}; requests = {}; asset_stat_checks = {}
    multi_selected = defaultdict(set)
    for r in manifest:
        multi_selected[r["source_id"]].add(json.loads(r["recommended"])["review_id"])
    for row in manifest:
        c = by_poly[row["polygon_id"]]; src = source_rows[row["source_id"]]
        for role, choice in [("RECOMMENDED",json.loads(row["recommended"])), *[("ALTERNATE",v) for v in json.loads(row["alternates"])]]:
            rid = choice["review_id"]; v = variants[rid]; human = human_rows[rid]; raw = human["raw"]
            old_variant = next(q for q in c["variants"] if q["review_id"] == rid)
            errors = []
            def same(ok, code):
                if not ok:errors.append(code)
            same(v["source_id"] == row["source_id"] == c["source_id"] == raw["source_id"], "SOURCE_IDENTITY_MISMATCH")
            same(src["group_id"] == c["group_id"], "GROUP_IDENTITY_MISMATCH")
            same(v["variant_id"] == choice["variant_id"] == raw["variant_id"] == old_variant["variant_id"]
                 and v["version"] == choice["version"] == raw["version"] == old_variant["version"], "VERSION_IDENTITY_MISMATCH")
            same(v["original_sha256"] == row["original_sha256"] == c["original_sha256"] == src["original_sha256"]
                 and v["annotation_sha256"] == row["annotation_sha256"] == c["annotation_sha256"] == src["annotation_sha256"], "ORIGINAL_JSON_IDENTITY_MISMATCH")
            same(json.loads(row["polygon_coordinates"]) == c["polygon_coordinates"] and row["class"] == c["class"]
                 == polygons[row["polygon_id"]]["class"], "POLYGON_IDENTITY_MISMATCH")
            resolved = json.loads(v["resolved_polygon_decisions"])
            local, basis = resolve_polygon(raw, resolved, row["polygon_id"])
            same(local == choice["human_polygon_state"] == "PASS" and raw["b_region_review"] == choice["human_b_state"] == "CONFIRMED"
                 and basis == choice["polygon_state_basis"], "A9_REGION_DECISION_MISMATCH")
            same(v["identity_status"] == "VERIFIED", "A9_IDENTITY_UNRESOLVED")
            same(old_variant["historical_human"] == raw, "PRIOR_AUDIT_HUMAN_ROW_MISMATCH")
            matrix = choice["matrix_output_to_source"]
            same(matrix == json.loads(v["matrix_output_to_source"]) == old_variant["matrix_output_to_source"], "MATRIX_IDENTITY_MISMATCH")
            asset_refs = {"original":{"path":row["original_path"],"sha256":row["original_sha256"]},
                          "annotation":{"path":row["annotation_path"],"sha256":row["annotation_sha256"]},
                          "raw":old_variant["raw"],"aligned":choice["aligned"],"valid_mask":choice["valid_mask"],
                          "geometry":old_variant["geometry"],"edit_mask":old_variant["edit_mask"]}
            for name, ref in asset_refs.items():
                key = Path(ref["path"]).resolve().as_posix()
                same(key in prior_files and prior_files[key]["sha256"] == ref["sha256"], "PRIOR_BYTE_AUDIT_BINDING_MISMATCH_" + name)
                asset_path = Path(ref["path"])
                regular = asset_path.is_file() and not asset_path.is_symlink()
                current_bytes = asset_path.stat().st_size if regular else None
                expected_bytes = prior_files.get(key,{}).get("bytes")
                same(regular, "ASSET_MISSING_OR_SYMBOLIC_" + name)
                same(current_bytes is not None and current_bytes == expected_bytes, "ASSET_CURRENT_SIZE_MISMATCH_" + name)
                asset_stat_checks[key] = {"regular_file":regular,"current_bytes":current_bytes,"prior_bytes":expected_bytes}
                asset_refs[name] = {**ref,"prior_bytes":expected_bytes,"current_bytes":current_bytes,
                                   "identity_basis":"PRIOR_BYTE_SHA256_LEDGER_REUSED_PLUS_CURRENT_REGULAR_FILE_AND_SIZE_STAT"}
            same(v["aligned_sha256"] == raw["aligned_sha256"] == choice["aligned"]["sha256"] == old_variant["aligned"]["sha256"], "ALIGNED_IDENTITY_MISMATCH")
            same(v["raw_sha256"] == raw["raw_sha256"] == old_variant["raw"]["sha256"], "RAW_IDENTITY_MISMATCH")
            same(choice["valid_mask"]["sha256"] == old_variant["valid_mask"]["sha256"], "VALID_IDENTITY_MISMATCH")
            support = choice["support"]
            same(support == json.loads(v["valid_support"])[row["polygon_id"]] and support["unsupported_pixels"] == 0,
                 "TARGET_SUPPORT_MISMATCH")
            record_path = Path(choice["aligned"]["path"]).parent/"record.json"
            record_key = record_path.resolve().as_posix()
            if record_key not in records:
                record_ref = metadata.check(record_path)
                records[record_key] = {"record":load(record_path),"identity":record_ref}
            actual_record = records[record_key]["record"]
            same(all(actual_record[k] == raw[k] for k in ("review_id","source_id","variant_id","version","source_sha256",
                 "annotation_sha256","raw_sha256","aligned_sha256","mask_sha256","request_sha256")), "DERIVED_RECORD_IDENTITY_MISMATCH")
            same(actual_record["file_sha256"]["valid.png"] == choice["valid_mask"]["sha256"], "DERIVED_VALID_IDENTITY_MISMATCH")
            request_path = Path(json.loads(v["origin_paths"])["request_path"])
            request_key = request_path.resolve().as_posix()
            if request_key not in requests:
                request_ref = metadata.check(request_path, v["request_sha256"])
                requests[request_key] = {"record":load(request_path),"identity":request_ref}
            request_record = requests[request_key]["record"]
            applicable = [q for q in semantic["rules"] if q["review_id"] == rid and (not q.get("polygon_id") or q["polygon_id"] == row["polygon_id"])]
            text_scope = [q for q in applicable if q["type"] == "ACCEPTED_WITH_SCOPE"]
            note_scope = [q for q in text_scope if q["reason_code"] != "HISTORICAL_REGION_ONLY_CONFIRMATION"]
            historical = historical_state(raw, local, not errors, bool(text_scope))
            pending_asset = choice["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION"
            kind = disposition(historical,pending_asset,applicable,errors)
            flags = []
            if src["version_conflict"] == "true":flags.append("HISTORICAL_SOURCE_VERSION_FLAG_NOT_SELECTED_VERSION_REJECTION")
            if len(multi_selected[row["source_id"]]) > 1:flags.append("SOURCE_HAS_MULTIPLE_SELECTED_REGION_VERSIONS")
            if c["other_annotated_defects_in_roi"]:flags.append("OTHER_GT_IN_ROI_NOT_PROOF_OF_TARGET_CONTAMINATION")
            if c["metrics"]["frame_edge_touch"]:flags.append("ROI_FRAME_EDGE_NOT_AUTOMATIC_TARGET_REJECTION")
            if role == "RECOMMENDED" and not c["numeric_screen_pass"]:flags.append("D0_ROI_CONTEXT_THRESHOLD_NOT_TARGET_SUPPORT_FAILURE")
            if role == "RECOMMENDED" and c["historical_g1_result"] == "G1_HOLD_INDEPENDENT_BOUNDARY_EVIDENCE":flags.append("OLD_ADDITIONAL_BOUNDARY_FIELD_NOT_REQUIRED_TO_REPEAT_HUMAN_REVIEW")
            other_history = [e for e in contexts[row["polygon_id"]] if e["review_id"] != rid]
            out = {"record_id":f"{row['source_id']}:{row['polygon_id']}:{rid}","selection_role":role,"source_id":row["source_id"],"group_id":src["group_id"],
                "polygon_id":row["polygon_id"],"class":row["class"],"review_id":rid,"variant_id":choice["variant_id"],"version":choice["version"],
                "run_id":actual_record.get("run_id",request_record.get("run_id","UNKNOWN_NOT_RECORDED")),
                "run_id_status":"EXPLICIT_HISTORICAL_FIELD" if "run_id" in actual_record or "run_id" in request_record else "NOT_SEPARATELY_RECORDED",
                "run_id_basis":"EXPLICIT_HISTORICAL_RECORD_OR_REQUEST_FIELD_ONLY_NO_DIRECTORY_TIMESTAMP_INFERENCE",
                "alignment_asset_directory":actual_record["asset_directory"],"generation_variant_id":v["generation_variant_id"],
                "alignment_parent_review_id":v["alignment_parent_review_id"],"request_sha256":v["request_sha256"],
                "request_id":request_record.get("request_id","NOT_SEPARATELY_RECORDED"),"request_manifest":requests[request_key]["identity"],
                "polygon_coordinates":c["polygon_coordinates"],"polygon_coordinates_sha256":canonical(c["polygon_coordinates"]),
                "assets":asset_refs,"stored_matrix_output_to_source":matrix,"matrix_sha256":canonical(matrix),
                "geometry_provenance":old_variant["geometry_provenance"],"derived_record":records[record_key]["identity"],
                "same_version_support":support,"valid_mask_complete_target":support["unsupported_pixels"] == 0,
                "full_frame_valid":float(v["valid_coverage"]) == 1,
                "full_roi_valid":c["metrics"]["valid_roi_fraction"] == 1 if role == "RECOMMENDED" else None,
                "trace":human["trace"],"historical_user_decision":raw["decision"],"historical_polygon_state":local,
                "polygon_state_basis":basis,"b_region_review":raw["b_region_review"],"whole_image_normal":raw["whole_image_normal"],
                "historical_user_reason":raw["reason"],"historical_user_notes":raw["notes"],"historical_status":historical,
                "scope":{"inherited_scope":"EXACT_HASH_BOUND_POLYGON_ONLY_NOT_WHOLE_IMAGE","region_only_user_confirmation":not bool(raw["whole_image_normal"]),
                         "textual_restrictions":note_scope,"scope_evidence_rules":text_scope,
                         "coordinate_crop_fov_restricted":any(q.get("scope",{}).get("coordinate_crop_fov_restricted",False)
                         or q.get("scope",{}).get("crop_rectangle_proven_by_note",False) for q in text_scope)},
                "pixel_asset_state_a9":choice["pixel_asset_state"],"asset_binding_pending":pending_asset,
                "final_version_choice_a9":raw.get("final_version_choice",""),"selected_version_identity_conflict":bool(errors),
                "source_version_flag_a9":src["version_conflict"] == "true","same_source_multiple_selected_region_versions":len(multi_selected[row["source_id"]]) > 1,
                "other_version_history":other_history,"polygon_master_human_history":json.loads(polygons[row["polygon_id"]]["human_history"]),
                "other_version_rejected":any(e["historical_status"] == "HISTORICAL_USER_REJECTED" for e in other_history),
                "other_version_retry_alignment":any(e["polygon_state"] == "RETRY_ALIGNMENT" for e in other_history),
                "other_version_hold":any(e["polygon_state"] == "HOLD" for e in other_history),
                "other_annotated_defects_in_roi":[{**q,"same_variant_historical_state":resolved.get(q["polygon_id"])} for q in c["other_annotated_defects_in_roi"]],
                "actual_target_contamination":"NOT_ESTABLISHED_BY_ROI_PRESENCE",
                "historical_d0_result":c["historical_g1_result"] if role == "RECOMMENDED" else "NOT_INHERITED_FROM_OTHER_REVIEW_ID",
                "historical_d0_trace":c["history"]["pilot"] if role == "RECOMMENDED" else None,
                "flags":flags,"specific_evidence_rules":applicable,"identity_errors":errors,"data_gate_disposition":kind,
                "data_gate":"PENDING_INDEPENDENT_CHATGPT_DATA_GATE","formal_data_gate_pass":False,"training_approved":False,
                "new_human_g1_pass_created":False,"formal_trusted_pair":False}
            (selected if role == "RECOMMENDED" else alternates).append(out)
            identity_errors.extend({"record_id":out["record_id"],"role":role,"reason":e} for e in errors)
    need(len(selected) == 545 and len(alternates) == 10, "Recommended/alternate counts differ")
    metadata.postcheck()
    counts = {
        "1_historical_human_polygon_pass":count([r for r in selected if r["historical_polygon_state"] == "PASS" and r["b_region_review"] == "CONFIRMED"]),
        "2_same_version_aligned_identity":count([r for r in selected if not r["identity_errors"]]),
        "3_historical_valid_mask_file_identity":count([r for r in selected if not r["identity_errors"]]),
        "4_traceable_historical_matrix":count([r for r in selected if not r["identity_errors"]]),
        "5_coordinate_defined_crop_fov_limited_acceptance":count([r for r in selected if r["scope"]["coordinate_crop_fov_restricted"]]),
        "6_selected_version_identity_conflict":count([r for r in selected if r["selected_version_identity_conflict"]]),
        "7_selected_historical_user_rejected":count([r for r in selected if r["historical_status"] == "HISTORICAL_USER_REJECTED"]),
        "8_selected_ai_only_pass":count([r for r in selected if r["historical_status"] == "HISTORICAL_AI_ONLY_PASS"]),
        "9_selected_no_historical_human_decision":count([r for r in selected if r["historical_status"] == "NO_HISTORICAL_DECISION"]),
        "10_inherited_region_pair_candidate":count([r for r in selected if r["data_gate_disposition"] == INHERITED]),
        "11_asset_binding_only":count([r for r in selected if r["data_gate_disposition"] == ASSET_ONLY]),
        "12_targeted_review_required":count([r for r in selected if r["data_gate_disposition"] == TARGETED]),
        "13_explicitly_excluded_for_j2":count([r for r in selected if r["data_gate_disposition"] == EXCLUDED]),
    }
    extra = {
        "historical_full_target_valid":count([r for r in selected if r["valid_mask_complete_target"]]),
        "historical_full_frame_valid":count([r for r in selected if r["full_frame_valid"]]),
        "historical_full_roi_valid":count([r for r in selected if r["full_roi_valid"]]),
        "historical_region_only_user_confirmation":count([r for r in selected if r["scope"]["region_only_user_confirmation"]]),
        "historical_text_explicitly_region_scoped":count([r for r in selected if r["scope"]["textual_restrictions"]]),
        "source_version_flag_not_selected_conflict":count([r for r in selected if r["source_version_flag_a9"]]),
        "same_source_multiple_selected_region_versions":count([r for r in selected if r["same_source_multiple_selected_region_versions"]]),
        "other_version_retry_repair_context_only":count([r for r in selected if r["other_version_rejected"]]),
        "other_version_retry_alignment_context_only":count([r for r in selected if r["other_version_retry_alignment"]]),
        "other_version_retry_repair_or_alignment_context_only":count([r for r in selected if r["other_version_rejected"] or r["other_version_retry_alignment"]]),
        "other_version_user_hold_context_only":count([r for r in selected if r["other_version_hold"]]),
        "asset_binding_pending_including_targeted":count([r for r in selected if r["asset_binding_pending"]]),
        "unknown_hold":count([r for r in selected if r["data_gate_disposition"] == UNKNOWN]),
        "historical_acceptance_inherited_including_pending_issues":count([r for r in selected if r["historical_status"] in {"HISTORICAL_USER_ACCEPTED","HISTORICAL_USER_ACCEPTED_WITH_SCOPE"}]),
        "aligned_review_id_bound_including_targeted":count([r for r in selected if not r["asset_binding_pending"]]),
        "targeted_already_bound_overlap":count([r for r in selected if r["data_gate_disposition"] == TARGETED and not r["asset_binding_pending"]]),
        "targeted_asset_pending_overlap":count([r for r in selected if r["data_gate_disposition"] == TARGETED and r["asset_binding_pending"]]),
    }
    class_counts = {c:{"total":count([r for r in selected if r["class"] == c]),
                      **{kind:count([r for r in selected if r["class"] == c and r["data_gate_disposition"] == kind]) for kind in (INHERITED,ASSET_ONLY,TARGETED,EXCLUDED,UNKNOWN)}} for c in CLASSES}
    write(args.output/"historical_review_rows.json",{"rows":all_human})
    write(args.output/"historical_polygon_events.json",{"events":events})
    write(args.output/"reconciliation.json",{"schema":"exp17d_historical_reconciliation_v1","selected":selected,"alternates":alternates,"identity_errors":identity_errors})
    csv_write(args.output/"reconciliation.csv",selected)
    for filename,kind in (("inherited_region_pair_candidates",INHERITED),("asset_binding_queue",ASSET_ONLY),("targeted_review_queue",TARGETED),("excluded_queue",EXCLUDED)):
        write(args.output/(filename+".json"),{"disposition":kind,"records":[r for r in selected if r["data_gate_disposition"] == kind],"data_gate":"PENDING","training_approved":False})
    write(args.output/"input_metadata_identity.json",{"files":list(metadata.records.values()),"postchecked":True,
          "asset_current_existence_size_stat":[{"path":k,**v} for k,v in asset_stat_checks.items()],
          "prior_image_byte_audit_reused":True,"current_image_byte_rehashes":0,"current_image_decodes":0})
    summary = {"status":HOLD if identity_errors else COMPLETE,"population":count(selected),"recommended_rows":len(selected),"alternate_polygon_version_rows":len(alternates),
        "selected_unique_review_ids":len({r["review_id"] for r in selected}),"alternate_unique_review_ids":len({r["review_id"] for r in alternates}),
        "historical_review_rows":len(all_human),"historical_polygon_version_events":len(events),"five_human_inputs":batch_stats,
        "required_13_counts":counts,"supplemental_counts":extra,"class_coverage":class_counts,
        "historical_status_counts":{status:count([r for r in selected if r["historical_status"] == status]) for status in
            ("HISTORICAL_USER_ACCEPTED","HISTORICAL_USER_ACCEPTED_WITH_SCOPE","HISTORICAL_USER_PENDING","HISTORICAL_USER_REJECTED",
             "HISTORICAL_AI_ONLY_PASS","HISTORICAL_VERSION_UNRESOLVED","HISTORICAL_ALIGNMENT_UNRESOLVED","ASSET_IDENTITY_MISMATCH","NO_HISTORICAL_DECISION")},
        "all_historical_event_states":dict(Counter(r["historical_status"] for r in events)),"selected_disposition_counts":dict(Counter(r["data_gate_disposition"] for r in selected)),
        "all_historical_resolved_polygon_states":dict(Counter(r["polygon_state"] for r in events)),
        "all_historical_overall_user_decisions":dict(Counter(r["raw"]["decision"] for r in all_human)),
        "all_historical_semantically_accepted_polygon_versions":sum(e["polygon_state"] == "PASS" and e["b_region_review"] == "CONFIRMED" for e in events),
        "all_historical_semantically_accepted_distinct_polygons":len({e["polygon_id"] for e in events if e["polygon_state"] == "PASS" and e["b_region_review"] == "CONFIRMED"}),
        "historically_semantically_accepted_polygons_outside_current_545_not_promoted":len({e["polygon_id"] for e in events if e["polygon_state"] == "PASS" and e["b_region_review"] == "CONFIRMED"} - {r["polygon_id"] for r in selected}),
        "asset_and_action_axes":{"asset_axis":{"bound":extra["aligned_review_id_bound_including_targeted"],"pending":extra["asset_binding_pending_including_targeted"]},
             "action_axis":{kind:count([r for r in selected if r["data_gate_disposition"] == kind]) for kind in (INHERITED,ASSET_ONLY,TARGETED,EXCLUDED,UNKNOWN)},
             "targeted_scope":"Specific missing technical explanation/clarification, not a repeated five-round human G1 review"},
        "independent_group_counts_can_overlap_between_classes_or_queues":True,"identity_errors":len(identity_errors),
        "prior_audit":{"candidate_ledger_sha256":AUDIT_CANDIDATES_SHA,"file_ledger_sha256":AUDIT_FILES_SHA,"image_and_other_files_previously_byte_verified":3998,
                       "current_image_byte_rehashes":0,"current_image_decodes":0,"current_rasters_or_registrations":0},
        "current_metadata_and_asset_checks":{"metadata_files_sha256_checked_and_postchecked":len(metadata.records),
            "unique_asset_paths_existence_regular_type_size_statted":len(asset_stat_checks),
            "current_asset_size_mismatches":sum(not q["regular_file"] or q["current_bytes"] != q["prior_bytes"] for q in asset_stat_checks.values()),
            "image_byte_sha256_policy":"Reused accepted prior 3998-file SHA256/postcheck ledger; current pass checks file existence, regular type and size only; no image SHA256 rehash or decode"},
        "rules":RULES,"source_sha256":source_identity,"semantic_evidence_sha256":sha(args.semantic_evidence) if args.semantic_evidence else None,
        "private_output_sha256":{name:sha(args.output/name) for name in ("reconciliation.json","reconciliation.csv","historical_review_rows.json","historical_polygon_events.json","input_metadata_identity.json")},
        "formal_data_gate_pass":0,"new_human_g1_pass":0,"formal_trusted_pairs":0,"optimizer_updates":0,"gpu_calls":0,"val_test_image_reads":0,
        "new_generation":0,"new_alignment":0,"historical_files_modified":0,"completed_utc":datetime.now(timezone.utc).isoformat()}
    write(args.output/"summary.json",summary)
    if args.public_output:
        args.public_output.mkdir(parents=True,exist_ok=False)
        write(args.public_output/"summary.json",summary)
        write(args.public_output/"rules.json",RULES)
    print(json.dumps({"status":summary["status"],"required_13_counts":counts,"supplemental_counts":extra,"identity_errors":len(identity_errors)},sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("a9-root","exp17-audit","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--semantic-evidence",type=Path,required=True);p.add_argument("--public-output",type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        audit(args)
    except Exception as exc:
        write(args.output/"stopped.json",{"status":HOLD,"error_type":type(exc).__name__,"reason":str(exc),"image_reads":0,"gpu_calls":0,"optimizer_updates":0})
        raise


if __name__=="__main__":main()
