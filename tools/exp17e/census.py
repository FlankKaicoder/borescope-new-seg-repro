"""Census all original GT inside manifest-defined A10 S379; never select another N.

Only TRAIN source/JSON, selected frozen N and published N image bytes are read.
Geometry and final Polygon recovery states belong to the separate mapping and
expansion tools. Historical decisions always retain their exact review identity.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tools.exp17d.reconcile import canonical, csv_values, indexed, load, need, read_csv, sha, write, csv_write

CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
A10_MANIFEST_SHA = "1d98e7effbe9c5ec900afaa5cafeaa9994cedf0f839a0c21b2452d5a868819bd"
A10_EXCEPTIONS_SHA = "5856342f496ae3b03c12a21e5c7f65cd9079faa4340fd8c08ecc72e872e30e18"
A10_RELEASE_SHA = "b0674f782c6da97f34585a3ff4b4aeb1b4f53972fb6e533361e019749b6750d3"
D_MATRIX_SHA = "6d2e2a145e4264ba9ca046ba323b4cdcfef6f7d1ee92746920503decf61ddbdb"
TYPES = {"ALIGNED":346,"RAW":29,"BOUNDED":1,"CROP":3}
D_STATES = {"INHERITED_REGION_PAIR_CANDIDATE":357,"ASSET_BINDING_ONLY_PENDING":176,"TARGETED_EVIDENCE_REVIEW":12}


def counts(rows):
    return {"polygons":len(rows),"sources":len({r["source_id"] for r in rows}),"groups":len({r["group_id"] for r in rows})}


def image_size(path):
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return list(size)


class Files:
    """Current SHA reads are scoped and timed; hard-linked bytes are not reread."""
    def __init__(self):
        self.records = {}; self.inodes = {}; self.total_read_seconds = 0.; self.total_read_bytes = 0

    def check(self, path, expected, purpose):
        path = Path(path)
        need(path.is_file() and not path.is_symlink(), "Required frozen input missing or symbolic")
        path = path.resolve(strict=True); name = path.as_posix(); stat = path.stat()
        snapshot = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if name in self.records:
            item = self.records[name]
            need(tuple(item["stat_identity"]) == snapshot and item["sha256"] == expected, "Conflicting current input identity")
            if purpose not in item["purposes"]:item["purposes"].append(purpose)
            return item
        start = time.perf_counter()
        cached = self.inodes.get(snapshot) if stat.st_ino else None
        digest = cached if cached else sha(path)
        elapsed = time.perf_counter()-start
        need(digest == expected, "Current input SHA256 differs from frozen identity")
        if not cached:
            self.inodes[snapshot] = digest; self.total_read_seconds += elapsed; self.total_read_bytes += stat.st_size
        item = {"path":name,"bytes":stat.st_size,"sha256":digest,"purposes":[purpose],"stat_identity":snapshot,
                "sha256_read_seconds":elapsed if not cached else 0.,"hardlink_hash_reused":bool(cached)}
        self.records[name] = item
        return item

    def postcheck(self):
        for item in self.records.values():
            stat=Path(item["path"]).stat()
            need((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns) == tuple(item["stat_identity"]), "Read-only input changed during census")


def validate_scope(rows, exceptions):
    need(len(rows) == len({r["source_id"] for r in rows}) == 379, "A10 must define exactly 379 unique source IDs")
    need(len({r["group_id"] for r in rows}) == 359 and Counter(r["asset_type"] for r in rows) == TYPES, "A10 frozen group/type population differs")
    need(len(exceptions) == len({r["source_id"] for r in exceptions}) == 13
         and not {r["source_id"] for r in rows} & {r["source_id"] for r in exceptions}, "Excluded 13 sources leaked into S379")
    need(all(r["split"].lower() == "train" for r in rows), "A10 scope includes a non-TRAIN source")


def extract_polygons(source_id, group_id, annotation, size, a9):
    need([annotation["imageWidth"],annotation["imageHeight"]] == size, "Original image/JSON frame differs")
    selected = json.loads(a9["selected_polygon_ids"]); all_ids = json.loads(a9["all_gt_polygon_ids"])
    need(len(all_ids) == len(set(all_ids)) and set(selected) <= set(all_ids), "A9 original target identities are invalid")
    records = []
    for index, shape in enumerate(annotation["shapes"]):
        need(shape.get("shape_type") == "polygon" and shape.get("label") in CLASSES, "Original annotation shape/class is invalid")
        points = shape.get("points")
        need(isinstance(points,list) and len(points) >= 3 and all(isinstance(p,(list,tuple)) and len(p)==2
             and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in p)
             and 0 <= p[0] <= size[0] and 0 <= p[1] <= size[1] for p in points), "Original Polygon coordinates are invalid")
        derived = f"{source_id}_{index:03d}"
        original_id = shape.get("polygon_id")
        pid = str(original_id) if original_id is not None else derived
        if index < len(all_ids):
            need(pid == all_ids[index], "Original Polygon ordinal/ID differs from A9 on the same JSON identity")
        records.append({"source_id":source_id,"group_id":group_id,"polygon_id":pid,"annotation_index":index,
            "id_basis":"ORIGINAL_JSON_EXPLICIT_POLYGON_ID" if original_id is not None else "DERIVED_ID",
            "derived_id_basis":None if original_id is not None else "source_id + underscore + original zero-based shapes ordinal, three-digit minimum; not a native JSON ID",
            "historical_a9_id_present":pid in all_ids,"original_json_polygon_id":original_id,
            "class_id":CLASSES.index(shape["label"]),"class_name":shape["label"],"coordinates":points,
            "coordinates_sha256":canonical(points),"source_size":size,"a9_selected":pid in selected,"a9_all":pid in all_ids,
            "a9_selected_polygon_ids":selected,"a9_all_polygon_ids":all_ids,"original_shape_group_id":shape.get("group_id"),
            "extra_original_annotation_not_selected_by_a9":pid not in selected})
    need(len(records) == len({r["polygon_id"] for r in records}), "Original Polygon IDs collide")
    need(set(all_ids) <= {r["polygon_id"] for r in records}, "An A9 original Polygon disappeared")
    return records


def review_evidence(raw, variant, pid, trace, n_type, n_sha, specialty):
    resolved = json.loads(variant["resolved_polygon_decisions"])
    explicit = json.loads(raw["polygon_decisions"])
    state = resolved.get(pid, "NO_RECORDED_POLYGON_DECISION_FOR_THIS_REVIEW")
    if pid in resolved:
        need(state == explicit.get(pid,raw["decision"]), "Original human Polygon state differs from A9 semantics")
    local_pass = state == "PASS" and raw["b_region_review"] == "CONFIRMED"
    aligned_sha = variant["aligned_sha256"]
    exact_aligned = n_type == "ALIGNED" and bool(aligned_sha) and n_sha == aligned_sha
    crop_parent = n_type == "CROP" and (specialty.get("parent") or {}).get("sha256") == aligned_sha
    return {"review_id":variant["review_id"],"variant_id":variant["variant_id"],"version":variant["version"],
        "historical_polygon_state":state,"polygon_state_basis":"EXPLICIT" if pid in explicit else "OVERALL_DEFAULT_FROZEN_TARGET" if pid in resolved else "NOT_IN_REVIEW_TARGET_SET",
        "b_region_review":raw["b_region_review"],"overall_user_decision":raw["decision"],"whole_image_normal":raw["whole_image_normal"],
        "final_version_choice":raw.get("final_version_choice",""),"human_notes":raw["notes"],"human_reason":raw["reason"],
        "same_review_user_polygon_pass_and_b_confirmed":local_pass,"historical_aligned_sha256":aligned_sha,
        "current_n_is_exact_historically_reviewed_aligned_pixels":exact_aligned,
        "crop_parent_is_historical_aligned":crop_parent,
        "exact_current_n_region_acceptance_evidence":local_pass and exact_aligned,
        "crop_parent_region_evidence_requires_verified_crop_mapping":local_pass and crop_parent,
        "raw_or_bounded_same_review_is_not_automatic_aligned_pixel_acceptance":n_type in {"RAW","BOUNDED"},
        "trace":trace,"ai_decision_separate":variant["ai_decision"],"new_user_decision":False}


def run(args):
    started = time.perf_counter(); files = Files()
    a10 = args.a10_root; a9 = args.a9_root
    a10_receipt = load(a10/"release/freeze_receipt.json")
    need(a10_receipt["status"] == "EXP14_A10_A_DATASET_FROZEN_WITH_EXCEPTIONS"
         and a10_receipt["frozen_a"] == {"images":379,"source_groups":359,"asset_types":TYPES}, "Wrong A10 freeze receipt")
    files.check(a10/"release/freeze_receipt.json",sha(a10/"release/freeze_receipt.json"),"A10_FREEZE_RECEIPT")
    for name, expected in a10_receipt["private_file_sha256"].items():files.check(a10/name,expected,"A10_PINNED_METADATA")
    files.check(a10/"registry/a_frozen_manifest.csv",A10_MANIFEST_SHA,"A10_SCOPE_MANIFEST")
    files.check(a10/"registry/a_freeze_exceptions.csv",A10_EXCEPTIONS_SHA,"A10_EXCLUSIONS")
    files.check(a10/"release/sha256_manifest.json",A10_RELEASE_SHA,"A10_RELEASE_FILE_INDEX")
    release = load(a10/"release/sha256_manifest.json")
    need(release["file_count"] == len(release["files"]) == 2094 and canonical(release["files"]) == release["sha256"] == a10_receipt["freeze_content_sha256"],
         "A10 release file-index content differs")
    manifest = read_csv(a10/"registry/a_frozen_manifest.csv"); frozen = [x["row"] for x in manifest]
    exceptions = csv_values(a10/"registry/a_freeze_exceptions.csv"); validate_scope(frozen,exceptions)
    a9_receipt = load(a9/"handoff/delivery_receipt.json")
    for name, expected in a9_receipt["private_sha256"].items():files.check(a9/name,expected,"A9_PINNED_METADATA")
    sources = indexed(csv_values(a9/"registry/source_master_registry.csv"),"source_id")
    variants = indexed(csv_values(a9/"registry/variant_master_registry.csv"),"review_id")
    b545 = indexed(csv_values(a9/"registry/b_pair_candidate_manifest.csv"),"polygon_id")
    a9_poly = indexed(csv_values(a9/"registry/polygon_master_registry.csv"),"polygon_id")
    b_proposal_audit = indexed(load(a9/"audits/alignment_support_audit.json")["records"],"polygon_id")
    need(len(b545) == 545 and len(variants) == 956, "A9 historical denominators differ")
    human = {}; human_input_counts = []
    for name, spec in load(a9/"inputs/input_hashes.json")["human_csvs"].items():
        path = a9/"inputs"/name;files.check(path,spec["sha256"],"FIVE_HUMAN_CSV")
        files.check(spec["original_path"],spec["sha256"],"FIVE_HUMAN_CSV_ORIGINAL")
        rows = read_csv(path);human_input_counts.append({"name":name,"rows":len(rows),"sha256":spec["sha256"]})
        for item in rows:
            raw = item["row"];rid=raw["review_id"]; need(rid not in human and rid in variants,"Human review identity collision/absence")
            v = variants[rid]
            need(all(raw[k] == v[f] for k,f in (("source_id","source_id"),("variant_id","variant_id"),("version","version"),
                ("source_sha256","original_sha256"),("annotation_sha256","annotation_sha256"),("raw_sha256","raw_sha256"),
                ("aligned_sha256","aligned_sha256"),("request_sha256","request_sha256"),("mask_sha256","mask_sha256"))), "Original CSV and A9 asset identity differ")
            need(raw["decision"]==v["human_decision"] and raw["b_region_review"]==v["b_region_review"]
                 and raw["whole_image_normal"]==v["whole_image_normal"] and raw["notes"]==v["human_notes"]
                 and json.loads(raw["polygon_decisions"])==json.loads(v["human_polygon_decisions"]),"A9 human fields differ from original export")
            human[rid]={"raw":raw,"trace":{**{k:v for k,v in item.items() if k!="row"},"csv_path":path.as_posix(),"csv_sha256":spec["sha256"],"review_id":rid}}
    need(len(human) == 956,"Five human exports are incomplete")
    delivery = load(args.d_delivery);files.check(args.d_delivery,sha(args.d_delivery),"ACTUAL_EXP17_D_DELIVERY")
    need(delivery["all_545_and_10_alternates_json"]["sha256"] == D_MATRIX_SHA,"Exp17-D delivery points to an unaudited matrix")
    for key in ("all_545_and_10_alternates_json","inherited_candidates","asset_binding_queue","targeted_questions"):
        spec=delivery[key];files.check(spec["path"],spec["sha256"],"EXP17_D_PINNED_MATRIX_AND_QUEUES")
    d_matrix=load(delivery["all_545_and_10_alternates_json"]["path"])["selected"]; d_by_pid=indexed(d_matrix,"polygon_id")
    need(Counter(r["data_gate_disposition"] for r in d_matrix)==D_STATES and set(d_by_pid)==set(b545),"Exp17-D matrix population differs")
    for key,state in (("inherited_candidates","INHERITED_REGION_PAIR_CANDIDATE"),("asset_binding_queue","ASSET_BINDING_ONLY_PENDING"),("targeted_questions","TARGETED_EVIDENCE_REVIEW")):
        queue=load(delivery[key]["path"])["records"]
        need({r["record_id"] for r in queue}=={r["record_id"] for r in d_matrix if r["data_gate_disposition"]==state},"D action queue differs from matrix")
    by_source=defaultdict(list)
    for v in variants.values():by_source[v["source_id"]].append(v)
    scope=[]; polygons=[]; asset_rows=[]; version_rows=[]; crosswalk=[]
    hash_scope_started=time.perf_counter()
    for ordinal,entry in enumerate(manifest,1):
        f=entry["row"];sid=f["source_id"];a=sources[sid];v=variants[f["review_id"]];h=human[f["review_id"]]
        need(f["group_id"]==a["group_id"] and f["source_image_sha256"]==a["original_sha256"]==v["original_sha256"]
             and f["annotation_sha256"]==a["annotation_sha256"]==v["annotation_sha256"] and f["variant_id"]==v["variant_id"]
             and f["historical_review_id"]==f["review_id"],"A10 source/version provenance differs from A9")
        need(f["asset_id"]==f["selected_asset_id"]==f"{f['review_id']}:{f['asset_type']}:{f['selected_asset_sha256']}"
             and f["selected_asset_sha256"]==f["published_sha256"]==release["files"][f["published_path"]],"A10 selected asset identity differs")
        refs={"ORIGINAL_TRAIN_IMAGE":(f["source_image_path"],f["source_image_sha256"]),
              "ORIGINAL_PROFESSIONAL_JSON":(f["annotation_path"],f["annotation_sha256"]),
              "A10_SELECTED_N":(f["selected_asset_path"],f["selected_asset_sha256"]),
              "A10_PUBLISHED_N":(a10/f["published_path"],f["published_sha256"])}
        for purpose,(path,expected) in refs.items():
            checked=files.check(path,expected,purpose)
            asset_rows.append({"source_id":sid,"group_id":f["group_id"],"purpose":purpose,"path":checked["path"],
                               "bytes":checked["bytes"],"sha256":checked["sha256"],"sha256_read_seconds":checked["sha256_read_seconds"],"current_verified":True})
        source_size=image_size(f["source_image_path"]); n_size=image_size(f["selected_asset_path"])
        need(n_size==[int(f["width"]),int(f["height"])],"A10 selected N dimensions differ")
        annotation=load(f["annotation_path"])
        need(Path(str(annotation["imagePath"]).replace("\\","/")).name == Path(f["source_image_path"]).name,"Professional JSON names another source image")
        ps=extract_polygons(sid,f["group_id"],annotation,source_size,a)
        specialty=json.loads(f["specialty_provenance"])
        n={"path":f["selected_asset_path"],"sha256":f["selected_asset_sha256"],"asset_type":f["asset_type"],
           "review_id":f["review_id"],"variant_id":f["variant_id"],"version":v["version"],"size":n_size,"asset_id":f["asset_id"]}
        trace={k:entry[k] for k in entry if k!="row"};trace.update({"manifest_path":(a10/"registry/a_frozen_manifest.csv").as_posix(),"manifest_sha256":A10_MANIFEST_SHA})
        s={**f,"manifest_trace":trace,"source_size":source_size,"n_size":n_size,"repair_version":v["version"],
           "specialty":specialty,"authoritative_n":n,"variant_metadata":v,"historical_review":h,
           "a9_selected_polygon_ids":json.loads(a["selected_polygon_ids"]),"a9_all_polygon_ids":json.loads(a["all_gt_polygon_ids"]),
           "all_original_polygon_ids":[p["polygon_id"] for p in ps],"original_polygon_count":len(ps),
           "one_fixed_n_for_source":True,"source_and_json_current_sha256_verified":True,"selected_and_published_n_current_sha256_verified":True}
        scope.append(s)
        for p in ps:
            pid=p["polygon_id"];b=b545.get(pid);d=d_by_pid.get(pid)
            same=review_evidence(h["raw"],v,pid,h["trace"],f["asset_type"],f["selected_asset_sha256"],specialty)
            proposals = b_proposal_audit.get(pid,{}).get("semantic_candidates",[])
            proposal = next((q for q in proposals if q["review_id"]==f["review_id"]),None)
            same["a9_b_proposal_on_this_review"] = proposal
            same["a9_b_proposal_applies_to_exact_selected_n_pixels"] = bool(proposal and proposal["aligned"]["sha256"]==f["selected_asset_sha256"])
            histories=[review_evidence(human[q["review_id"]]["raw"],q,pid,human[q["review_id"]]["trace"],f["asset_type"],f["selected_asset_sha256"],specialty)
                       for q in by_source[sid] if pid in json.loads(q["resolved_polygon_decisions"])]
            rec={**p,"source_image_path":f["source_image_path"],"source_image_sha256":f["source_image_sha256"],"annotation_path":f["annotation_path"],"annotation_sha256":f["annotation_sha256"],
                "n_asset_path":f["selected_asset_path"],"n_asset_sha256":f["selected_asset_sha256"],"n_asset_type":f["asset_type"],"n_review_id":f["review_id"],"n_variant_id":f["variant_id"],"repair_version":v["version"],
                "n_size":n_size,"b545_record":b,"exp17d_record":d,"exp17d_historical_disposition":d["data_gate_disposition"] if d else None,
                "same_n_version_history":same,"historical_all_versions":histories,"a10_whole_image_acceptance_is_not_automatic_polygon_pair_acceptance":True,
                "historical_b_selected_n_unchanged":False if b and json.loads(b["recommended"])["aligned"]["sha256"] != f["selected_asset_sha256"] else bool(b),
                "new_user_approval":False,"formal_data_gate_pass":False,"formal_trusted_pair":False}
            if pid in a9_poly:need(a9_poly[pid]["class"]==p["class_name"],"A9 and original Polygon class differ")
            polygons.append(rec)
            old=json.loads(b["recommended"]) if b else None
            version_rows.append({"source_id":sid,"group_id":f["group_id"],"polygon_id":pid,"class_name":p["class_name"],"a10_review_id":f["review_id"],"a10_variant_id":f["variant_id"],"a10_version":v["version"],"a10_asset_type":f["asset_type"],"a10_n_sha256":f["selected_asset_sha256"],
                "a9_b_review_id":old["review_id"] if old else "","a9_b_variant_id":old["variant_id"] if old else "","a9_b_aligned_sha256":old["aligned"]["sha256"] if old else "",
                "same_b_review_id":bool(old and old["review_id"]==f["review_id"]),"same_b_asset_pixels":bool(old and old["aligned"]["sha256"]==f["selected_asset_sha256"]),
                "d_historical_action":d["data_gate_disposition"] if d else "NO_B545_RECORD","current_n_same_review_polygon_state":same["historical_polygon_state"],"current_n_historical_b_confirmed":same["b_region_review"],
                "exact_current_n_aligned_region_evidence":same["exact_current_n_region_acceptance_evidence"],"crop_parent_region_evidence_requires_mapping":same["crop_parent_region_evidence_requires_verified_crop_mapping"],
                "rule":"A10_SELECTED_N_FIXED; no cross-version/asset PASS transfer; history retained separately"})
        if ordinal%100==0:print(f"Census verified {ordinal}/379 sources",flush=True)
    hash_scope_elapsed=time.perf_counter()-hash_scope_started
    pids={r["polygon_id"] for r in polygons};sids={r["source_id"] for r in scope};scope_index=indexed(scope,"source_id")
    for pid,b in b545.items():
        d=d_by_pid[pid];n=scope_index.get(b["source_id"]);old=json.loads(b["recommended"])
        crosswalk.append({"source_id":b["source_id"],"group_id":sources[b["source_id"]]["group_id"],"polygon_id":pid,"class_name":b["class"],"in_s379":n is not None,"in_p379":pid in pids,
            "d_original_disposition":d["data_gate_disposition"],"b_selected_review_id":old["review_id"],"b_selected_variant_id":old["variant_id"],"b_selected_aligned_sha256":old["aligned"]["sha256"],
            "a10_n_review_id":n["review_id"] if n else "","a10_n_variant_id":n["variant_id"] if n else "","a10_n_asset_type":n["asset_type"] if n else "","a10_n_sha256":n["selected_asset_sha256"] if n else "",
            "a10_n_equals_b_selected_pixels":bool(n and n["selected_asset_sha256"]==old["aligned"]["sha256"]),"a10_review_equals_b_review":bool(n and n["review_id"]==old["review_id"])})
    files.postcheck()
    write(args.output/"source_scope.json",{"schema":"exp17e_s379_sources_v1","sources":scope,"exceptions_excluded":13,"scope_basis":"A10_FROZEN_MANIFEST_ONLY"})
    csv_write(args.output/"source_scope.csv",scope)
    write(args.output/"p379_polygon_census.json",{"schema":"exp17e_p379_polygon_census_v1","polygons":polygons,"formal_trusted_pairs":0})
    csv_write(args.output/"p379_polygon_census.csv",polygons)
    csv_write(args.output/"existing_b545_crosswalk.csv",crosswalk)
    csv_write(args.output/"version_consistency_report.csv",version_rows)
    csv_write(args.output/"asset_manifest.csv",asset_rows)
    write(args.output/"input_identity.json",{"files":list(files.records.values()),"postcheck":"unchanged stat identity after current SHA reads","d_delivery_sha256":sha(args.d_delivery)})
    in_b=[r for r in polygons if r["b545_record"]];new=[r for r in polygons if not r["b545_record"]]
    summary={"status":"PASS_EXP17E_A10_MANIFEST_SCOPE_AND_ORIGINAL_POLYGON_CENSUS","sources":len(scope),"groups":len({r["group_id"] for r in scope}),"asset_types":dict(Counter(r["asset_type"] for r in scope)),
        "excluded_original_13_sources":len(exceptions),"p379":counts(polygons),"a9_original_selected_polygons":counts([r for r in polygons if r["a9_selected"]]),
        "additional_original_annotations_not_a9_selected":counts([r for r in polygons if not r["a9_selected"]]),"original_annotations_not_in_a9_all_registry":counts([r for r in polygons if not r["a9_all"]]),
        "b545_total":545,"b545_inside_s379":counts(in_b),"b545_outside_s379":{"polygons":sum(not r["in_s379"] for r in crosswalk),"sources":len({r["source_id"] for r in crosswalk if not r["in_s379"]}),"groups":len({r["group_id"] for r in crosswalk if not r["in_s379"]})},
        "new_p379_not_in_b545":counts(new),"d545_historical_subsets_within_s379":{k:counts([r for r in in_b if r["exp17d_historical_disposition"]==k]) for k in D_STATES},
        "class_coverage":{c:counts([r for r in polygons if r["class_name"]==c]) for c in CLASSES},"id_basis_counts":dict(Counter(r["id_basis"] for r in polygons)),
        "b545_intersection_exact_a10_selected_pixels":counts([r for r in in_b if r["historical_b_selected_n_unchanged"]]),
        "same_review_historical_polygon_pass_and_b_confirmed":counts([r for r in polygons if r["same_n_version_history"]["same_review_user_polygon_pass_and_b_confirmed"]]),
        "exact_current_aligned_n_historical_region_evidence":counts([r for r in polygons if r["same_n_version_history"]["exact_current_n_region_acceptance_evidence"]]),
        "five_human_inputs":human_input_counts,"historical_human_review_rows":len(human),"current_sha256_audit":{"scope":"379 original TRAIN images,379 original professional JSON,379 selected N,379 published N; no VAL/TEST or empty label files read",
            "asset_reference_count":len(asset_rows),"unique_files_including_metadata":len(files.records),"physical_bytes_read_for_sha256":files.total_read_bytes,"sha256_read_seconds":files.total_read_seconds,
            "full_image_scope_hash_and_header_validation_seconds":hash_scope_elapsed,"total_census_seconds":time.perf_counter()-started,"hardlink_hash_reuse_references":sum(r["hardlink_hash_reused"] for r in files.records.values())},
        "input_sha256":{"a10_manifest":A10_MANIFEST_SHA,"a10_exceptions":A10_EXCEPTIONS_SHA,"a10_release_file_index":A10_RELEASE_SHA,"d_delivery":sha(args.d_delivery),"d_matrix":D_MATRIX_SHA},
        "output_sha256":{name:sha(args.output/name) for name in ("source_scope.json","p379_polygon_census.json","existing_b545_crosswalk.csv","version_consistency_report.csv","asset_manifest.csv","input_identity.json")},
        "source_sha256":sha(__file__),"one_fixed_n_per_source":True,"final_polygon_states_assigned_here":False,"new_user_approval":0,"formal_data_gate_pass":0,"formal_trusted_pairs":0,"model_forwards":0,"optimizer_updates":0,"registrations":0,"generation_calls":0,"val_test_image_reads":0}
    write(args.output/"summary.json",summary)
    if args.public_summary:
        args.public_summary.parent.mkdir(parents=True,exist_ok=True);write(args.public_summary,summary)
    print(json.dumps(summary,sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("a10-root","a9-root","d-delivery","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--public-summary",type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except Exception as error:
        write(args.output/"stopped.json",{"status":"EXP17E_CENSUS_HOLD_WITH_IDENTITY_EVIDENCE","error_type":type(error).__name__,"reason":str(error),"model_forwards":0,"optimizer_updates":0})
        raise


if __name__=="__main__":main()
