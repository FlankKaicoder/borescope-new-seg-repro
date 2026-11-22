"""Join immutable Exp17-E evidence and exact user Polygon decisions for Data Gate.

Metadata only: no image decode, registration, model, optimizer, sampler, or
historical-file mutation. Original Queue rows and original typed CSV records
remain separate from the coordinate-normalized compatibility view.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tools.exp17e import review_package as legacy

CSV_SHA = "b0a5f768040539a6c2daaab18b1166f99995d4fcf7d0f968e10bec4de2e476fd"
PACKAGE_ID = "a0d977ae3dae0a148f0c3f128c113954f18a282ac70a5371d0cf90b352560351"
QUEUE_SHA = "1bf154d3f0a565892fad12c5f7e80882424baeee34be289cf77911bb8b1ac363"
BUNDLE_SHA = "43115742e25c550475ff3f4d86e7c40f92b31079620faa7964ef085a18dcf1e7"
INDEX_SHA = "668c1a7b6848a462e10f14e80ecb4fb69dd6db962315e6ff4d28b2c74387c464"
ZIP_SHA = "29a4bee6c7fd40c20cf7fa22cc686c5a4631f368d8db56ca09faeb6984cd56fb"
SPLIT_SHA = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
ACCEPT = "ACCEPT_THIS_POLYGON_FOR_J2"
REJECT = "REJECT_THIS_POLYGON_FOR_J2"
HISTORICAL = "HISTORICAL_INHERITED_PENDING_DATA_GATE"
USER_ACCEPTED = "USER_ACCEPTED_PENDING_DATA_GATE"
USER_REJECTED = "USER_REJECTED_EXCLUDE"
TECHNICAL = "TECHNICAL_HOLD"
OUTSIDE = "NOT_IN_CURRENT_J2_POOL"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    return legacy.load(path)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def csv_write(path, rows, fields=None):
    require(rows or fields, "Empty CSV needs an explicit schema")
    fields = fields or list(rows[0])
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            require(set(row) == set(fields), "CSV output row schema differs")
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                             if isinstance(v, (dict,list)) else v for k,v in row.items()})


def indexed(rows, key):
    require(len(rows) == len({r[key] for r in rows}), "Duplicate identity: " + key)
    return {r[key]:r for r in rows}


def counts(rows):
    return {"polygons":len(rows), "source_images":len({r["source_id"] for r in rows}),
            "independent_source_groups":len({r["group_id"] for r in rows})}


class Inputs:
    def __init__(self):
        self.records = {}

    def check(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Expected a regular metadata file")
        path = path.resolve(strict=True)
        key = path.as_posix()
        if key not in self.records:
            self.records[key] = {"path":key,"bytes":path.stat().st_size,"sha256":sha(path)}
        value = self.records[key]
        require(expected is None or value["sha256"] == expected, "Frozen metadata SHA256 differs")
        return value

    def postcheck(self):
        for value in self.records.values():
            require(sha(value["path"]) == value["sha256"], "Read-only input changed during reconciliation")


def matrix_product(a,b):
    require(len(a) == len(b) == 3 and all(len(r)==3 for r in a+b), "Expected homogeneous matrices")
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def raw_direction_valid(source_mapping):
    h = source_mapping.get("historical_raw_to_source")
    if h is None:
        return False
    n_to_source = source_mapping["n_to_source"]; source_to_n = source_mapping["source_to_n"]
    return all(abs(h[i][j]-n_to_source[i][j]) <= 1e-9 for i in range(3) for j in range(3)) and all(
        abs(v-(1. if i==j else 0.)) <= 1e-9 for matrix in (matrix_product(source_to_n,n_to_source),matrix_product(n_to_source,source_to_n))
        for i,row in enumerate(matrix) for j,v in enumerate(row))


def final_state(original, review):
    if original["primary_status"] in {"TECHNICAL_SUPPORT_FAILURE","MAPPING_UNVERIFIED"}:
        require(review is None, "A technically locked Polygon received a user decision")
        return TECHNICAL
    if review:
        require(original["primary_status"] in {"CURRENT_TARGETED_REVIEW","NEW_HUMAN_REVIEW_CANDIDATE"},
                "A new review was applied outside the frozen human-review queue")
        if review["decision"] == ACCEPT:
            return USER_ACCEPTED
        if review["decision"] == REJECT:
            return USER_REJECTED
        return OUTSIDE
    return HISTORICAL if original["primary_status"] == "CURRENT_INHERITED" else OUTSIDE


def evidence_tier(original, state):
    if state == USER_ACCEPTED:
        require(original["asset_type"] in {"ALIGNED","RAW"}, "Unexpected newly accepted asset type")
        return "NEW_USER_ACCEPTED_CURRENT_" + original["asset_type"] + "_POLYGON"
    if state == HISTORICAL:
        require(original["evidence_route"] in {
            "EXACT_REVIEWED_ALIGNED_REGION","RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H",
            "CROP_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION","BOUNDED_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION"}, "Unknown historical inheritance route")
        return original["evidence_route"]
    return "USER_REJECTED_EXACT_POLYGON" if state == USER_REJECTED else "TECHNICAL_LOCKED" if state == TECHNICAL else "UNADMITTED"


def join_decisions(originals, records, raw_records, row_audit):
    reviews = indexed(records,"candidate_id"); raw = indexed(raw_records,"candidate_id")
    require(set(reviews) == set(raw), "Raw and normalized review sets differ")
    audits = {r["candidate_id"]:r for r in row_audit}
    require(set(reviews) == set(audits), "Review trace set differs")
    require(set(reviews) <= {p["candidate_id"] for p in originals}, "Review refers to an unknown Candidate ID")
    result=[]
    for original in originals:
        cid=original["candidate_id"]; review=reviews.get(cid); state=final_state(original,review)
        entry={"candidate_id":cid,"source_id":original["source_id"],"group_id":original["group_id"],
               "polygon_id":original["polygon_id"],"class_id":original["class_id"],"class_name":original["class_name"],
               "original":original,"new_user_review":None if review is None else {
                   "raw_typed_unchanged":raw[cid],"normalized_coordinate_view":review,"audit_trace":audits[cid]},
               "final_state":state,"evidence_tier":evidence_tier(original,state),
               "candidate_pool_pending_data_gate":state in {HISTORICAL,USER_ACCEPTED},
               "independent_data_gate":"NOT_GRANTED","formal_trusted_pair":False,"training_approved":False}
        result.append(entry)
    return result


def technical_evidence(original, displayed, census, source, source_map, polygon_map):
    sm=source_map; pm=polygon_map; n=source["authoritative_n"]
    require((original["n_asset_sha256"],original["n_review_id"],original["n_variant_id"],original["repair_version"],original["asset_type"]) ==
            (n["sha256"],n["review_id"],n["variant_id"],n["version"],n["asset_type"]), "Queue does not bind the frozen A10 N")
    require((displayed["source_valid_mask_sha256"],displayed["n_valid_mask_sha256"],displayed["matrix_identity_sha256"]) ==
            ((sm.get("source_valid_mask") or {}).get("sha256"),(sm.get("n_valid_mask") or {}).get("sha256"),sm["matrix_identity_sha256"]), "Display/CSV mapping or Valid identity differs")
    require(pm["matrix_identity_sha256"] == sm["matrix_identity_sha256"] and original["polygon_mapping"] == pm,
            "Queue and mapping proof differ")
    require(all(original[k] == census[k] for k in ("source_id","group_id","polygon_id","class_id","class_name","source_image_sha256","annotation_sha256","n_asset_sha256","repair_version","n_review_id","n_variant_id","coordinates")),
            "Census and Queue Polygon identity differ")
    full = bool(sm["mapping_verified"] and pm["mapping_verified"] and pm["support_complete"]
                and pm.get("source_polygon_pixels",0)>0 and pm.get("n_polygon_pixels",0)>0
                and pm.get("source_valid_target_pixels")==pm.get("source_polygon_pixels")
                and pm.get("n_valid_target_pixels")==pm.get("n_polygon_pixels") and pm.get("n_vertices_inside_frame") is True)
    scope = displayed["accepted_region_scope"]
    require(scope["scope"] == "THIS_POLYGON_ONLY" and scope["source_coordinates"] == original["coordinates"]
            and scope["n_coordinates"] == pm["n_coordinates"], "Review scope differs from this exact Polygon")
    raw_ok = None
    if original["asset_type"] == "RAW":
        raw_ok = bool(sm["evidence"].get("historical_raw_sha256") == original["n_asset_sha256"]
                      and source["variant_metadata"]["raw_sha256"] == original["n_asset_sha256"]
                      and sm["review_id"] == original["n_review_id"] and raw_direction_valid(sm))
    return {"identity_consistent":True,"a10_n_consistent":True,"mapping_verified":bool(sm["mapping_verified"] and pm["mapping_verified"]),
            "target_support_complete":full,"source_valid_mask_sha256":displayed["source_valid_mask_sha256"],
            "n_valid_mask_sha256":displayed["n_valid_mask_sha256"],"matrix_identity_sha256":sm["matrix_identity_sha256"],
            "raw_same_generation_and_inverse_direction_valid":raw_ok,"accepted_region_scope":scope,
            "source_polygon_pixels":pm.get("source_polygon_pixels"),"source_valid_target_pixels":pm.get("source_valid_target_pixels"),
            "n_polygon_pixels":pm.get("n_polygon_pixels"),"n_valid_target_pixels":pm.get("n_valid_target_pixels")}


def verify_historical_route(original, census, source_map, lineage):
    h=census["same_n_version_history"]; route=original["evidence_route"]
    require(h["same_review_user_polygon_pass_and_b_confirmed"] and not original["historical_user_rejected"], "Historical route lacks accepted regional evidence")
    proposal=h["a9_b_proposal_on_this_review"]
    require(not proposal or proposal["geometry_status"] != "INELIGIBLE", "A historical inherited route still has a current-N proposal conflict")
    require(not original["specific_d_question_applies_current_n"], "Historical candidate retains a current-N unresolved question")
    if route == "EXACT_REVIEWED_ALIGNED_REGION":
        require(original["asset_type"] == "ALIGNED" and h["historical_aligned_sha256"] == original["n_asset_sha256"], "Exact Aligned inheritance differs")
    elif route == "RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H":
        require(original["asset_type"] == "RAW" and source_map["evidence"]["historical_aligned_sha256"] == h["historical_aligned_sha256"], "Raw inferential parent differs")
    elif route in {"CROP_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION","BOUNDED_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION"}:
        proof=lineage[original["polygon_id"]]
        require(proof["review_id"] == original["n_review_id"] and proof["a10_n_sha256"] == original["n_asset_sha256"]
                and proof["matrix_identity_sha256"] == source_map["matrix_identity_sha256"]
                and proof["reviewed_aligned_parent"]["sha256"] == h["historical_aligned_sha256"]
                and proof["historical_same_polygon_state"] == "PASS" and proof["historical_b_region_review"] == "CONFIRMED"
                and proof["target_rgb_pixels_equal_parent"] and proof["different_target_rgb_pixels"] == 0,
                "Crop/Bounded target-pixel lineage differs")
    else:
        raise ValueError("Unknown inherited evidence route")


def run(args):
    from tools.exp17f.compat import parse_and_validate
    files=Inputs(); code_root=Path(__file__).resolve().parents[2]
    files.check(args.review_csv,CSV_SHA); files.check(args.e_delivery)
    delivery=load(args.e_delivery); require(delivery["status"] == "EXP17_E_A10_379_POLYGON_EXPANSION_COMPLETE_PENDING_HUMAN_AND_DATA_GATE", "Unexpected Exp17-E delivery")
    required=delivery["required_deliverables"]
    for spec in required.values():files.check(spec["path"],spec["sha256"])
    queue_path=Path(delivery["final_queue_original"]);files.check(queue_path,QUEUE_SHA)
    require(required["polygon_review_queue.json"]["sha256"] == QUEUE_SHA, "Delivered queue identity differs")
    queue=load(queue_path); original_digest=digest(queue)
    review_dir=Path(delivery["final_review_directory"])
    receipt=load(review_dir/"package_receipt.json")
    require(receipt["package_id"] == PACKAGE_ID and receipt["queue_sha256"] == QUEUE_SHA and receipt["bundle_sha256"] == BUNDLE_SHA
            and receipt["index_sha256"] == INDEX_SHA and receipt["polygons"] == 576, "Frozen review package differs")
    for name,expected in (("review_bundle.json",BUNDLE_SHA),("index.html",INDEX_SHA),("package_receipt.json",None),("file_manifest.json",None),("new_review_template.json",None),("new_review_template.csv",None)):
        files.check(review_dir/name,expected)
    bundle=load(review_dir/"review_bundle.json")
    require(bundle["package_id"] == PACKAGE_ID and bundle["queue_sha256"] == QUEUE_SHA, "Bundle package or queue differs")
    integrity=load(args.integrity_receipt);files.check(args.integrity_receipt)
    require(integrity["schema"] == "EXP17_F_INPUT_AND_PACKAGE_INTEGRITY_V1" and integrity["overall_status"] == "PASS", "External package/current-byte audit did not pass")
    ip=integrity["review_package"]
    require(ip["package_id"] == PACKAGE_ID and ip["queue_sha256"] == QUEUE_SHA and ip["bundle_sha256"] == BUNDLE_SHA
            and ip["index_sha256"] == INDEX_SHA and ip["zip_sha256"] == ZIP_SHA and ip["media_files_verified"] == 3710
            and integrity["user_csv"]["sha256"] == CSV_SHA, "External integrity receipt covers another input")
    asset_receipt=load(args.asset_receipt);files.check(args.asset_receipt)
    require(asset_receipt["status"] == "PASS_CURRENT_ASSET_AND_EVIDENCE_PRECONDITIONS_NOT_DATA_GATE" and asset_receipt["identity_errors"] == 0
            and asset_receipt["queue_sha256"] == QUEUE_SHA and asset_receipt["user_csv_sha256"] == CSV_SHA, "External current source asset audit did not pass")
    source_path=Path(required["a10_379_source_scope.json"]["path"]);census_path=Path(required["p379_polygon_census.json"]["path"])
    sources=indexed(load(source_path)["sources"],"source_id");census=indexed(load(census_path)["polygons"],"polygon_id")
    mapping_root=Path(delivery["mapping_proofs_directory"])
    for name in ("mapping_summary.json","source_mappings.json","polygon_mappings.json","region_lineage_summary.json","bounded_region_lineage.json","crop_region_lineage.json"):
        files.check(mapping_root/name)
    ms=load(mapping_root/"mapping_summary.json");ls=load(mapping_root/"region_lineage_summary.json")
    require(ms["scope_sha256"] == sha(source_path) and ms["polygon_census_sha256"] == sha(census_path)
            and ms["private_source_mappings_sha256"] == sha(mapping_root/"source_mappings.json")
            and ms["private_polygon_mappings_sha256"] == sha(mapping_root/"polygon_mappings.json"), "Mapping proof hashes differ")
    require(ls["bounded_private_sha256"] == sha(mapping_root/"bounded_region_lineage.json") and ls["crop_private_sha256"] == sha(mapping_root/"crop_region_lineage.json"), "Target-pixel lineage hashes differ")
    sm=indexed(load(mapping_root/"source_mappings.json")["sources"],"source_id");pm=indexed(load(mapping_root/"polygon_mappings.json")["polygons"],"polygon_id")
    lineage=indexed([r for name in ("bounded_region_lineage.json","crop_region_lineage.json") for r in load(mapping_root/name)["records"]],"polygon_id")
    a10_receipt=load(args.a10_root/"release/freeze_receipt.json");files.check(args.a10_root/"release/freeze_receipt.json")
    frozen_manifest=args.a10_root/"registry/a_frozen_manifest.csv"; files.check(frozen_manifest,a10_receipt["private_file_sha256"]["registry/a_frozen_manifest.csv"])
    with frozen_manifest.open(encoding="utf-8-sig",newline="") as stream:a10=indexed(list(csv.DictReader(stream)),"source_id")
    require(len(a10)==len(sources)==379 and len(census)==576,"S379/P379 populations differ")
    originals=queue["polygons"];q_by_cid=indexed(originals,"candidate_id");q_by_pid=indexed(originals,"polygon_id")
    require(len(originals)==576 and set(q_by_pid)==set(census),"Original Queue population differs")
    _,prepared=legacy.prepare(queue); expected=indexed(prepared,"candidate_id"); br=indexed(bundle["records"],"candidate_id")
    require(set(expected)==set(br)==set(q_by_cid),"Review Bundle and Queue Candidate ID sets differ")
    for cid,row in expected.items():
        require(legacy.digest(legacy.identity(row))==legacy.digest(legacy.identity(br[cid])) and row["binding_sha256"]==br[cid]["binding_sha256"],"Bundle identity differs from frozen Queue rendering metadata")
        require(row["mapping_verified"]==br[cid]["mapping_verified"] and row["support_complete"]==br[cid]["support_complete"]
                and row["identity_resolved"]==br[cid]["identity_resolved"],"Bundle technical state differs")
    parsed=parse_and_validate(args.review_csv,review_dir/"review_bundle.json",expected_csv_sha256=CSV_SHA)
    reviews=parsed["records"]; raw_reviews=parsed["raw_records"]
    required_review_ids={p["candidate_id"] for p in originals if p["primary_status"] in {"CURRENT_TARGETED_REVIEW","NEW_HUMAN_REVIEW_CANDIDATE"}}
    require({r["candidate_id"] for r in reviews}==required_review_ids and len(reviews)==69,"User CSV does not exactly cover the frozen 69 human-review records")
    require(len({(r["source_id"],r["polygon_id"]) for r in reviews})==69,"Duplicate source/Polygon review")
    require({r["reviewer"] for r in reviews}=={"kk"},"Unexpected reviewer attribution")
    require(Counter(r["decision"] for r in reviews)=={ACCEPT:65,REJECT:4},"Actual frozen user decision counts differ")
    technical={}; crosswalk=[]
    for original in originals:
        cid=original["candidate_id"];pid=original["polygon_id"];source=sources[original["source_id"]];f=a10[original["source_id"]]
        require((source["group_id"],source["selected_asset_sha256"],source["review_id"],source["variant_id"],source["asset_type"]) ==
                (f["group_id"],f["selected_asset_sha256"],f["review_id"],f["variant_id"],f["asset_type"]),"Census source does not bind actual A10 manifest")
        technical[cid]=technical_evidence(original,br[cid],census[pid],source,sm[original["source_id"]],pm[pid])
    reviews_by_id=indexed(reviews,"candidate_id")
    for review in reviews:
        cid=review["candidate_id"];old=q_by_cid[cid];t=technical[cid]
        if review["decision"]==ACCEPT:
            require(t["mapping_verified"] and t["target_support_complete"] and br[cid]["identity_resolved"] is True, "User ACCEPT failed technical prerequisites")
            require(not old["historical_user_rejected"] and not review["historical_rejection_override_reason"],"Unexpected unresolved historical rejection/override")
            if old["asset_type"]=="RAW":require(t["raw_same_generation_and_inverse_direction_valid"],"Raw ACCEPT does not bind its historical matrix/generation")
        crosswalk.append({"candidate_id":cid,"source_id":old["source_id"],"group_id":old["group_id"],"polygon_id":old["polygon_id"],"class_id":old["class_id"],"class_name":old["class_name"],
            "original_primary_status":old["primary_status"],"decision":review["decision"],"reviewer":review["reviewer"],"reviewed_at":review["reviewed_at"],
            "bundle_binding_sha256":br[cid]["binding_sha256"],"csv_binding_sha256":review["binding_sha256"],"exact_bundle_candidate_match":True,
            "technical_prerequisites":t,"accepted_scope":review["accepted_region_scope"],"user_notes":review["user_notes"]})
    merged=join_decisions(originals,reviews,raw_reviews,parsed["row_audit"])
    require(all(m["original"]==q_by_cid[m["candidate_id"]] for m in merged) and digest(queue)==original_digest,"Original Queue fields mutated")
    bypid=indexed(merged,"polygon_id")
    require({m["polygon_id"] for m in merged if m["final_state"]==USER_REJECTED}=={"13_002","185_005","616_001","628_000"},"Actual rejected Polygon scope differs")
    require(all(bypid[pid]["final_state"]==USER_ACCEPTED for pid in ("185_002","185_003")),"Source185 sibling ACCEPT was lost")
    states=Counter(m["final_state"] for m in merged)
    require(states=={HISTORICAL:499,USER_ACCEPTED:65,USER_REJECTED:4,TECHNICAL:8},"Actual 576-way disposition arithmetic differs")
    pool=[m for m in merged if m["candidate_pool_pending_data_gate"]]
    require(len(pool)==564,"Candidate pool arithmetic differs")
    for m in pool:
        old=m["original"];t=technical[m["candidate_id"]]
        require(t["mapping_verified"] and t["target_support_complete"],"Candidate lacks verified/full target support")
        if old["asset_type"]=="RAW":require(t["raw_same_generation_and_inverse_direction_valid"],"Raw candidate lacks a same-generation directional mapping")
        if m["final_state"]==HISTORICAL:verify_historical_route(old,census[m["polygon_id"]],sm[m["source_id"]],lineage)
    files.check(args.train_manifest,SPLIT_SHA)
    with args.train_manifest.open(encoding="utf-8-sig",newline="") as stream:train=[r for r in csv.DictReader(stream) if r["split"].lower()=="train"]
    trainidx=indexed(train,"stem");require(len(train)==668 and len({r["group_id"] for r in train})==632,"Full TRAIN B identity population differs")
    b_groups={c:{r["group_id"] for r in train if c in r["labels_present"].split("|")} for c in CLASSES}
    evidence=[]
    for m in pool:
        old=m["original"];require(trainidx[m["source_id"]]["group_id"]==m["group_id"] and m["class_name"] in trainidx[m["source_id"]]["labels_present"].split("|"),"Candidate source/group/class differs from original TRAIN")
        other=sorted(b_groups[m["class_name"]]-{m["group_id"]})
        evidence.append({"candidate_id":m["candidate_id"],"source_id":m["source_id"],"group_id":m["group_id"],"polygon_id":m["polygon_id"],"class_id":m["class_id"],"class_name":m["class_name"],
            "final_state":m["final_state"],"evidence_tier":m["evidence_tier"],"asset_type":old["asset_type"],"repair_version":old["repair_version"],
            "n_review_id":old["n_review_id"],"n_variant_id":old["n_variant_id"],"n_asset_sha256":old["n_asset_sha256"],"source_image_sha256":old["source_image_sha256"],"annotation_sha256":old["annotation_sha256"],
            "mapping_and_scope_evidence":technical[m["candidate_id"]],"historical_user_evidence":old["history"]["current_n_same_review_human"],
            "new_user_decision":(m["new_user_review"] or {}).get("normalized_coordinate_view"),"accepted_region_scope":br[m["candidate_id"]]["accepted_region_scope"],
            "full_train_same_class_b_group_count":len(b_groups[m["class_name"]]),"independent_same_class_b_groups":other,"independent_b_available":bool(other),
            "actual_native_batch_b_presence":"NOT_GUARANTEED_NO_NEW_SAMPLING","independent_data_gate":"NOT_GRANTED","formal_trusted_pair":False,"training_approved":False})
    require(all(e["independent_b_available"] for e in evidence),"A candidate lacks another frozen TRAIN same-class group")
    require(all(len({(m["original"]["n_asset_sha256"],m["original"]["repair_version"],m["original"]["n_review_id"],m["original"]["n_variant_id"]) for m in merged if m["source_id"]==sid})==1 for sid in sources),"One source mixed A10 N versions")
    by_original=[]
    for status in ("CURRENT_TARGETED_REVIEW","NEW_HUMAN_REVIEW_CANDIDATE"):
        items=[m for m in merged if m["original"]["primary_status"]==status]
        by_original.append({"original_primary_status":status,"reviewed":len(items),"ACCEPT":sum(m["final_state"]==USER_ACCEPTED for m in items),"REJECT":sum(m["final_state"]==USER_REJECTED for m in items),"HOLD":sum(m["final_state"]==OUTSIDE for m in items)})
    coverage=[]
    for c in CLASSES:
        items=[m for m in merged if m["class_name"]==c]; candidates=[m for m in items if m["candidate_pool_pending_data_gate"]]
        coverage.append({"class_name":c,"class_id":CLASSES.index(c),"all_original":counts(items),"historical_inherited":counts([m for m in items if m["final_state"]==HISTORICAL]),
            "new_user_ACCEPT":counts([m for m in items if m["final_state"]==USER_ACCEPTED]),"new_user_REJECT":counts([m for m in items if m["final_state"]==USER_REJECTED]),"technical_HOLD":counts([m for m in items if m["final_state"]==TECHNICAL]),
            "candidate_pool_pending_data_gate":counts(candidates),"full_train_B_groups":len(b_groups[c]),"candidate_A_groups_with_independent_B":len({m["group_id"] for m in candidates if b_groups[c]-{m["group_id"]}}),
            "evidence_tiers":{tier:counts([m for m in candidates if m["evidence_tier"]==tier]) for tier in sorted({m["evidence_tier"] for m in candidates})}})
    files.check(__file__);files.check(Path(__file__).with_name("compat.py"));files.postcheck()
    csv_audit={"status":"PASS_USER_REVIEW_CSV_AND_NORMALIZED_BINDING_IDENTITY","compatibility":parsed["summary"],"row_audit":parsed["row_audit"],"actual_decisions":dict(Counter(r["decision"] for r in reviews)),
               "user_csv_sha256":CSV_SHA,"reviewer_values":sorted({r["reviewer"] for r in reviews}),"exact_69_queue_review_scope":True,"raw_records_preserved":True,"legacy_inputs_modified":False}
    write(args.output/"user_review_csv_audit.json",csv_audit)
    csv_write(args.output/"review_bundle_crosswalk.csv",crosswalk)
    csv_write(args.output/"review_decision_by_original_status.csv",by_original)
    write(args.output/"p379_576_merged_decisions.json",{"schema":"exp17f_immutable_original_plus_user_review_v1","records":merged,"independent_data_gate":"NOT_GRANTED","formal_trusted_pairs":0})
    csv_write(args.output/"p379_576_merged_decisions.csv",merged)
    manifest=[{k:m[k] for k in ("candidate_id","source_id","group_id","polygon_id","class_id","class_name","final_state","evidence_tier")} |
              {"current_a10_n_identity":{k:m["original"][k] for k in ("asset_type","n_asset_sha256","repair_version","n_review_id","n_variant_id")},"accepted_region_scope":br[m["candidate_id"]]["accepted_region_scope"],"independent_data_gate":"NOT_GRANTED","formal_trusted_pair":False,"training_approved":False} for m in pool]
    csv_write(args.output/"accepted_564_candidate_manifest.csv",manifest)
    csv_write(args.output/"accepted_564_evidence_tiers.csv",[{k:m[k] for k in ("candidate_id","source_id","group_id","polygon_id","class_name","evidence_tier","final_state")} for m in pool])
    csv_write(args.output/"rejected_4_polygon_manifest.csv",[m for m in merged if m["final_state"]==USER_REJECTED])
    csv_write(args.output/"technical_hold_8_manifest.csv",[m for m in merged if m["final_state"]==TECHNICAL])
    csv_write(args.output/"class_group_coverage_after_review.csv",coverage)
    csv_write(args.output/"data_gate_evidence_matrix.csv",evidence)
    write(args.output/"data_gate_evidence_matrix.json",{"records":evidence,"independent_data_gate":"NOT_GRANTED","formal_trusted_pairs":0})
    write(args.output/"input_identity.json",{"files":list(files.records.values()),"post_run_sha256_rechecked":True,"external_integrity_receipt_sha256":sha(args.integrity_receipt),"external_asset_receipt_sha256":sha(args.asset_receipt)})
    summary={"status":"EXP17_F_USER_REVIEW_RECONCILIATION_PENDING_CHATGPT_DATA_GATE","population":counts(merged),"final_states":{s:counts([m for m in merged if m["final_state"]==s]) for s in states},
        "candidate_pool_pending_data_gate":counts(pool),"user_review_decisions":dict(Counter(r["decision"] for r in reviews)),"review_by_original_status":by_original,
        "historical_499_evidence_routes":{tier:counts([m for m in merged if m["final_state"]==HISTORICAL and m["evidence_tier"]==tier]) for tier in sorted({m["evidence_tier"] for m in merged if m["final_state"]==HISTORICAL})},
        "new_ACCEPT_asset_types":dict(Counter(m["original"]["asset_type"] for m in merged if m["final_state"]==USER_ACCEPTED)),"candidate_asset_types":dict(Counter(m["original"]["asset_type"] for m in pool)),
        "all_candidate_evidence_tiers":{tier:counts([m for m in pool if m["evidence_tier"]==tier]) for tier in sorted({m["evidence_tier"] for m in pool})},
        "class_group_coverage":coverage,"full_train_B_images":668,"full_train_B_groups":632,"B_scope":"FULL_ORIGINAL_TRAIN_NOT_S379","native_sampling_or_extra_B_sampling_changed":False,
        "compatibility":parsed["summary"],"new_accept_technical_prerequisites_passed":sum(m["final_state"]==USER_ACCEPTED for m in merged),"rejection_scope":"EXACT_REVIEWED_POLYGON_ONLY",
        "source185_other_acceptances_preserved":2,"old_technical_locks_retained":sum(m["final_state"]==TECHNICAL for m in merged),"original_queue_fields_preserved":True,"one_fixed_A10_N_per_source":True,
        "input_identity":{"package_id":PACKAGE_ID,"queue_sha256":QUEUE_SHA,"bundle_sha256":BUNDLE_SHA,"user_csv_sha256":CSV_SHA,"external_integrity_receipt_sha256":sha(args.integrity_receipt),"external_asset_receipt_sha256":sha(args.asset_receipt)},
        "source_sha256":{"reconcile.py":sha(__file__),"compat.py":sha(Path(__file__).with_name("compat.py"))},
        "output_sha256":{p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        "scientific_boundaries":["564 are candidates pending independent Data Gate, not admitted training Pairs.","35 inherited Raw records remain inferential same-generation regional evidence; three new Raw accepts are separately attributed to the user.","Five Crop/Bounded records preserve exact target pixels only; whole-frame context and Backbone receptive fields can differ.","Cross-group B availability is structural across all668 TRAIN; it is not guaranteed in any unchanged native batch.","No new model, sampler, SupCon, Pair-loss or training-parameter decisions are authorized here."],
        "new_human_acceptances":65,"independent_data_gate_granted":0,"formal_trusted_pairs":0,"training_approved":False,"model_forwards":0,"optimizer_updates":0,"new_registration":0,"new_generation":0,"val_test_image_reads":0}
    write(args.output/"summary.json",summary)
    args.public_output.mkdir(parents=True,exist_ok=True);write(args.public_output/"reconciliation_summary.json",summary)
    print(json.dumps({"status":summary["status"],"final_states":summary["final_states"],"candidate_pool":summary["candidate_pool_pending_data_gate"],"new_accept_asset_types":summary["new_ACCEPT_asset_types"]},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("review-csv","e-delivery","a10-root","train-manifest","integrity-receipt","asset-receipt","output","public-output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except Exception as error:
        write(args.output/"stopped.json",{"status":"EXP17_F_RECONCILIATION_HOLD_WITH_IDENTITY_EVIDENCE","error_type":type(error).__name__,"reason":str(error),"training_approved":False,"model_forwards":0,"optimizer_updates":0})
        raise


if __name__=="__main__":main()
