"""Import the exact Exp14 A9D1 final human asset choices and historical bindings."""

import csv
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.admission_diagnosis import verify_frozen_a9
from tools.exp14.final_dataset_identity import INPUTS
from tools.exp14.full392_recovery import COMPLETE, EXCLUDED, PROPOSED, REUSABLE, asset_id, csv_rows, source_digest
from tools.exp14.prepare_large_scale_repair import hash_file, load_json

SELECTION_SHA256 = "484a5d8d5d80fe1f92aeca9fdc8daf6f808b017f849f35028dc875094f4a76b4"
SELECTION_BYTES = 116965
D1_COMMIT = "f91d3071e83c31c594a8a44a6bfc637e7c4a0f49"
EXACT_SET_SHA256 = "bb7839d9b486f3853f5021ceef958b1f25fcfd7ed9e6487af82404226d461cf2"
BATCH_ID = f"exp14-a9d1-full392-{EXACT_SET_SHA256[:12]}"
ACCEPT = "ACCEPT_RECOMMENDED"
ALTERNATE = "CHOOSE_ALTERNATE"
HISTORICAL = "HISTORICAL_ALIGNED_HUMAN_ASSET_BINDING"
FINAL = "A10_FINAL_HUMAN_ASSET_SELECTION"
ASSET_TYPES = {"ALIGNED", "RAW", "BOUNDED", "CROP"}
FIELDS = ("batch_id", "exact_set_sha256", "source_id", "action", "asset_id", "review_id", "variant_id",
          "asset_type", "asset_sha256", "reason", "notes", "recorded_utc", "operation_id", "batch_scope")
EVIDENCE_PATHS = (
    "registry/exact_source_set.json", "registry/source_392_disposition.csv",
    "registry/recommended_asset_manifest.csv", "registry/alternate_assets.csv",
    "registry/derived_crop_provenance.json", "visual_inputs/cases.json", "review_full392/index.html",
)
HASH = re.compile(r"[0-9a-f]{64}")
CSV_TEXT_PREFIX = "'=@+-\t\r\n"


def index(records, key):
    result = {}
    for row in records:
        identifier = row[key]
        g3.require(identifier and identifier not in result, f"Missing or duplicate {key}: {identifier}")
        result[identifier] = row
    return result


def inline_bundle(path):
    html = path.read_text(encoding="utf-8")
    marker = '<script id="bundle" type="application/json">'
    g3.require(html.count(marker) == 1, "D1 page must contain one inline JSON bundle")
    raw = html.split(marker, 1)[1].partition("</script>")
    g3.require(bool(raw[1]), "D1 inline JSON bundle has no closing tag")
    return json.loads(raw[0])


def read_selection(path):
    data = path.read_bytes()
    g3.require(len(data) == SELECTION_BYTES and g3.digest(data) == SELECTION_SHA256,
               "Final human selection CSV does not match the requested original bytes")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    g3.require(tuple(reader.fieldnames or ()) == FIELDS, "Unrecognized final human selection CSV fields")
    rows = list(reader)
    for row in rows:
        g3.require(None not in row and all(isinstance(value, str) for value in row.values()),
                   "Incomplete final human selection CSV row")
        for key in ("reason", "notes"):
            value = row[key]
            if len(value) > 1 and value[0] == "'" and value[1] in CSV_TEXT_PREFIX:
                row[key] = value[1:]
    return data, rows


def asset_identity(row, hash_key="sha256"):
    rid, variant, kind, sha = (row[key] for key in ("review_id", "variant_id", "asset_type", hash_key))
    g3.require(bool(HASH.fullmatch(rid)) and bool(HASH.fullmatch(sha)) and variant and kind in ASSET_TYPES
               and row["asset_id"] == asset_id(rid, kind, sha), "Malformed selected asset identity")
    return (row["asset_id"], rid, variant, kind, sha)


def validate_choice(choice, bundle_rows, assets):
    src, action = choice["source_id"], choice["action"]
    g3.require(choice["batch_id"] == BATCH_ID and choice["exact_set_sha256"] == EXACT_SET_SHA256,
               "Final selection refers to a different review batch or complete source set")
    g3.require(action in (ACCEPT, ALTERNATE), f"Unexpected final whole-normal action for {src}")
    matching = assets.get(choice["asset_id"])
    g3.require(matching is not None and asset_identity(choice, "asset_sha256") == asset_identity(matching),
               f"Final selection asset tuple mismatch for {src}")
    wanted = bundle_rows[src]["recommended_asset_id"]
    g3.require((action == ACCEPT) == (choice["asset_id"] == wanted), f"Selection action and recommendation mismatch for {src}")
    recorded = datetime.fromisoformat(choice["recorded_utc"].replace("Z", "+00:00"))
    g3.require(recorded.utcoffset() == timedelta(0), f"Final selection time is not a UTC timestamp for {src}")
    if choice["operation_id"] or choice["batch_scope"]:
        scope = choice["batch_scope"].split("|")
        g3.require(choice["operation_id"] and action == ACCEPT and src in scope and len(set(scope)) == len(scope)
                   and all(bundle_rows.get(sid, {}).get("disposition") == PROPOSED for sid in scope),
                   f"Final selection batch scope invalid for {src}")
    return matching


def local_page_asset_path(page, url):
    parsed = urlparse(url)
    g3.require(parsed.scheme in ("", "file") and not (parsed.netloc or parsed.query or parsed.fragment),
               "D1 release bundle contains a nonlocal asset")
    display_path = Path(url2pathname(parsed.path))
    if parsed.scheme == "file":
        g3.require(display_path.is_absolute(), "D1 file URL is not absolute")
        return display_path.resolve()
    g3.require(not display_path.is_absolute(), "D1 local asset reference is not relative to the page")
    return (page.parent / display_path).resolve()


def same_asset(left, right):
    return left["sha256"] == right["sha256"] and Path(left["path"]).resolve() == Path(right["path"]).resolve()


def verify_selected_provenance(src, chosen, variant, verified, cases, crop_rows, pinned_assets):
    source, item = cases["sources"][src], cases["items"][chosen["review_id"]]
    record, audited = item["record"], item["audit"]
    g3.require(source["source_id"] == src and source["split"] == "train" and record["source_id"] == src
               and record["review_id"] == chosen["review_id"] and record["variant_id"] == chosen["variant_id"]
               and variant["variant_id"] == chosen["variant_id"], f"Selected asset does not have a frozen TRAIN review for {src}")
    g3.require(source["image_sha256"] == verified["original"]["sha256"] == record["source_sha256"]
               and source["annotation_sha256"] == verified["annotation"]["sha256"] == record["annotation_sha256"]
               and Path(source["image"]).resolve() == Path(verified["original"]["path"]).resolve()
               and Path(source["annotation"]).resolve() == Path(verified["annotation"]["path"]).resolve(),
               f"Original or annotation identity disagreement for {src}")
    kind = chosen["asset_type"]
    if kind == "CROP":
        crop = crop_rows[src]
        g3.require(crop["review_id"] == chosen["review_id"] and crop["variant_id"] == chosen["variant_id"]
                   and same_asset(crop, chosen) and crop["parent_sha256"] == variant["assets"]["ALIGNED"]["sha256"]
                   and same_asset(variant["assets"]["ALIGNED"], audited["assets"]["aligned.png"]),
                   f"D1 crop provenance mismatch for {src}")
        return crop
    ref = {"ALIGNED": "aligned.png", "BOUNDED": "bounded.png", "RAW": record["raw"]}[kind]
    original = audited["assets"].get(ref)
    g3.require(original and same_asset(chosen, original) and same_asset(chosen, variant["assets"][kind])
               and pinned_assets.get(original["path"]) == chosen["sha256"], f"D1 asset provenance mismatch for {src}")
    return None


def exact_visual_note(visual, selected):
    kind = selected["asset_type"]
    rid = selected["review_id"]
    if not visual:
        return ""
    if rid == visual["review_id"]:
        return visual["asset_observations"].get(kind, "")
    return visual.get("alternate_observations", {}).get(rid, {}).get(kind, "")


def import_selections(d1, csv_path, a9):
    d1, csv_path, a9 = (Path(p).resolve() for p in (d1, csv_path, a9))
    a9_head = verify_frozen_a9(a9)
    d1_receipt = load_json(d1 / "audits/delivery_receipt.json")
    g3.require(d1_receipt["status"] == COMPLETE and d1_receipt["commit"] == D1_COMMIT
               and d1_receipt["remote_head"] == D1_COMMIT, "D1 review receipt identity changed")
    inputs = d1_receipt["private_sha256"]
    for file in EVIDENCE_PATHS:
        hash_file(d1 / file, inputs[file])
    data, human_rows = read_selection(csv_path)
    choices = index(human_rows, "source_id")
    page = d1 / "review_full392/index.html"
    bundle = inline_bundle(page)
    exact = load_json(d1 / "registry/exact_source_set.json")
    case_file = load_json(d1 / "visual_inputs/cases.json")
    case_rows = index(case_file["rows"], "source_id")
    bundle_rows = index(bundle["rows"], "source_id")
    dispositions = index(csv_rows(d1 / "registry/source_392_disposition.csv"), "source_id")
    recommended = index(csv_rows(d1 / "registry/recommended_asset_manifest.csv"), "source_id")
    alternates = defaultdict(list)
    for row in csv_rows(d1 / "registry/alternate_assets.csv"):
        alternates[row["source_id"]].append(row)
    crop_rows = index(load_json(d1 / "registry/derived_crop_provenance.json"), "source_id")
    a9_evidence = load_json(a9 / "audits/verified_evidence_reconciled.json")
    historical = index(csv_rows(a9 / "registry/a_candidate_manifest.csv"), "source_id")
    pinned_assets = load_json(a9 / "registry/candidate_asset_sha256.json")["files"]
    a9_records = [{**item, "review_id": item["record"]["review_id"]} for item in a9_evidence["reviews"]]
    cases = {"sources": a9_evidence["sources"], "items": index(a9_records, "review_id")}
    source_ids = set(exact["source_ids"])
    g3.require(len(exact["source_ids"]) == len(source_ids) == 392
               and source_ids == bundle_rows.keys() == case_rows.keys() == dispositions.keys()
               and source_ids >= (recommended.keys() | alternates.keys() | crop_rows.keys())
               and bundle["batch_id"] == BATCH_ID and source_digest(source_ids) == exact["sha256"]
               == bundle["exact_set_sha256"] == case_file["source_set_sha256"] == EXACT_SET_SHA256,
               "Final import exact source set does not match the frozen 392-source bundle")
    reuse_ids = {sid for sid in source_ids if dispositions[sid]["disposition"] == REUSABLE}
    a9_reuse_candidates = {sid for sid, row in historical.items()
                           if json.loads(row["recommended"])["pixel_asset_state"] == "ALIGNED_REVIEW_ID_BOUND"}
    g3.require(len(reuse_ids) == 138 and len(choices) == 254 and choices.keys() == source_ids - reuse_ids
               and a9_reuse_candidates == reuse_ids, "Final export does not complement the historical 138 bindings")
    count_actions = dict(Counter(x["action"] for x in human_rows))
    count_types = dict(Counter(x["asset_type"] for x in human_rows))
    g3.require(count_actions == {ACCEPT: 220, ALTERNATE: 34}
               and count_types == {"ALIGNED": 221, "RAW": 29, "BOUNDED": 1, "CROP": 3}
               and len({x["review_id"] for x in human_rows}) == len({x["asset_sha256"] for x in human_rows}) == 254,
               "Frozen 254-row export action and asset population differs")
    operations = {}
    selected, exceptions, overrides = [], [], []
    history_receipt, selection_receipt = [], []
    for src in sorted(source_ids):
        case, web, disp = case_rows[src], bundle_rows[src], dispositions[src]
        choice = choices.get(src)
        try:
            g3.require(case["group_id"] == web["group_id"] == disp["group_id"] == cases["sources"][src]["group_id"]
                       and disp["disposition"] == web["disposition"]
                       and case["original"]["sha256"] == web["original"]["sha256"] == disp["original_sha256"]
                       and case["annotation"]["sha256"] == disp["annotation_sha256"]
                       and case["primary"]["review_id"] == disp["historical_review_id"], f"D1 source identity differs for {src}")
            assets = index(web["assets"], "asset_id")
            main = recommended.get(src)
            main_id = main["asset_id"] if main else None
            g3.require(web["recommended_asset_id"] == main_id, f"D1 recommended identity differs for {src}")
            registry = index([main, *alternates[src]] if main else alternates[src], "asset_id")
            g3.require(assets.keys() == registry.keys(), f"D1 registry and page asset sets differ for {src}")
            if choice:
                matched = validate_choice(choice, bundle_rows, assets)
                if choice["operation_id"]:
                    scope = operations.setdefault(choice["operation_id"], choice["batch_scope"])
                    g3.require(scope == choice["batch_scope"], f"Conflicting batch provenance for {src}")
                origin, input_sha = FINAL, SELECTION_SHA256
            else:
                matched = assets[main_id]
                origin = HISTORICAL
                old = json.loads(historical[src]["recommended"])
                item = cases["items"][matched["review_id"]]
                input_sha = INPUTS[item["input_name"]]
                g3.require(web["historical_asset_id"] == main_id and main_id == asset_id(old["review_id"], "ALIGNED", old["asset"]["sha256"])
                           and same_asset(old["asset"], registry[main_id])
                           and item["human"]["decision"] == "PASS" and item["human"]["whole_image_normal"] == "YES"
                           and item["human"]["aligned_sha256"] == matched["sha256"]
                           and "bounded.png" not in item["audit"]["assets"]
                           and disp["historical_binding_basis"] == "EXACT_HUMAN_REVIEW_ID_ALIGNED_SHA256_ONLY_REPAIR_MAIN_VIEW",
                           f"Historical Aligned binding is not exact for {src}")
            saved = registry[matched["asset_id"]]
            g3.require(asset_identity(matched) == asset_identity(saved)
                       and saved["group_id"] == case["group_id"]
                       and local_page_asset_path(page, matched["url"]) == Path(saved["path"]).resolve(),
                       f"D1 registry and page path or identity differ for {src}")
            variant = index(case["variants"], "review_id")[matched["review_id"]]
            crop = verify_selected_provenance(src, saved, variant, case, cases, crop_rows, pinned_assets)
            visual = None
            if choice:
                location = f"visual_notes/{src}.json"
                hash_file(d1 / location, inputs[location])
                visual = load_json(d1 / location)
                g3.require(visual["source_id"] == src and visual["review_id"] in {v["review_id"] for v in case["variants"]},
                           f"D1 visual evidence identity does not match for {src}")
            provenance = {"source_id": src, "group_id": case["group_id"], "review_id": saved["review_id"],
                          "variant_id": saved["variant_id"], "asset_type": saved["asset_type"], "asset_id": saved["asset_id"],
                          "sha256": saved["sha256"], "decision_origin": origin, "human_input_sha256": input_sha,
                          "recorded_utc": choice["recorded_utc"] if choice else "",
                          "action": choice["action"] if choice else "REUSE_EXACT_HISTORICAL_ASSET"}
            selected_note = exact_visual_note(visual, saved)
            row = {**provenance, "path": saved["path"], "original": case["original"], "annotation": case["annotation"],
                   "split": "train", "advisory_state": disp["disposition"], "selected_note": selected_note,
                   "original_reasons": json.loads(disp["original_reasons"]), "variant": variant, "crop": crop,
                   "original_classes": cases["sources"][src]["classes"], "existing_b_candidates": int(disp["existing_b_candidates"]),
                   "human_reason": choice["reason"] if choice else "", "human_notes": choice["notes"] if choice else "",
                   "historical_input_name": cases["items"][matched["review_id"]]["input_name"],
                   "historical_review_id": disp["historical_review_id"], "old_visual_note": disp["visual_note"]}
            (selection_receipt if choice else history_receipt).append(provenance)
            selected.append(row)
            if choice and (choice["action"] == ALTERNATE or disp["disposition"] == EXCLUDED):
                previous = next((v for v in case["variants"] if v["review_id"] == disp["historical_review_id"]), None)
                previous_asset = (previous or {}).get("assets", {}).get(saved["asset_type"], {})
                overrides.append({**provenance, "d1_disposition": disp["disposition"], "d1_recommended_asset_id": main_id or "",
                                  "historical_review_id": disp["historical_review_id"],
                                  "same_historical_review": disp["historical_review_id"] == saved["review_id"],
                                  "same_historical_pixels": previous_asset.get("sha256") == saved["sha256"],
                                  "formerly_proposed_exclusion": disp["disposition"] == EXCLUDED,
                                  "prior_risks": row["original_reasons"], "prior_visual_note": disp["visual_note"],
                                  "selected_asset_prior_note": selected_note})
        except (KeyError, ValueError, OSError) as error:
            exceptions.append({"source_id": src, "group_id": case["group_id"], "reason": "SOURCE_IDENTITY_MISMATCH",
                               "message": str(error), "selection": choice or recommended.get(src)})
    audit = {"a9_verified_commit": a9_head, "d1_verified_commit": D1_COMMIT, "batch_id": BATCH_ID,
             "exact_set_sha256": EXACT_SET_SHA256, "source_count": len(source_ids),
             "source_groups": len({row["group_id"] for row in case_rows.values()}), "input_csv_bytes": len(data),
             "input_csv_sha256": SELECTION_SHA256, "new_rows": len(human_rows), "historical_rows": len(reuse_ids),
             "new_action_counts": count_actions, "new_type_counts": count_types, "matched_selections": len(selected),
             "matched_type_counts": dict(Counter(row["asset_type"] for row in selected)), "identity_exception_count": len(exceptions),
             "matched_distinct_asset_hashes": len({row["sha256"] for row in selected}),
             "new_nonblank_optional_fields": {key: sum(bool(x[key]) for x in human_rows)
                                              for key in ("reason", "notes", "operation_id", "batch_scope")},
             "former_exclusion_overrides": sum(r["formerly_proposed_exclusion"] for r in overrides)}
    return {"selected": selected, "override_rows": overrides, "audit": audit, "history_receipt": history_receipt,
            "selection_receipt": selection_receipt, "input_bytes": data, "exceptions": exceptions}
