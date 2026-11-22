"""Reverify the frozen 545 TRAIN Pair candidates; never grant human/G1 admission."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops

from tools.exp13_cloud.g3.prepare_g3b2a_batch import mask_for_selection

SPLIT_SHA = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
B_SHA = "08042f6c05a0d823565394493bd9abdd8dec6cb94c27fa2df0cc794907264c6e"
ROI_SHA = "8b89916203c762a52beb535afa6248c894557f2ba9312243bac626e024802e4e"
PILOT_SHA = "4085986a1d2e96b1384ceba3c2a6f7c94f93465716cb67382c185b27007ed4e4"
CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
FEASIBLE = "G1_PILOT_FEASIBLE_PENDING_INDEPENDENT_REVIEW"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Files:
    def __init__(self):
        self.records = {}

    def check(self, path, expected=None):
        path = Path(path)
        need(path.is_file() and not path.is_symlink(), "Missing or symbolic candidate file")
        name = path.resolve().as_posix()
        if name not in self.records:
            self.records[name] = {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}
        result = self.records[name]
        need(expected is None or result["sha256"] == expected, "Candidate file SHA256 differs")
        return dict(result)

    def postcheck(self):
        for item in self.records.values():
            need(sha(item["path"]) == item["sha256"], "Read-only audit input changed during execution")


def count(mask):
    return mask.histogram()[255]


def binary(image):
    return image.convert("L").point(lambda value: 255 if value > 0 else 0)


def region_metrics(mask, valid):
    """Pillow equivalent of the pinned D0 integer-raster metric definition."""
    need(mask.size == valid.size, "Mask grid mismatch")
    left, top, right, bottom = mask.getbbox()
    w, h = mask.size
    dx, dy = right - left, bottom - top
    minimum = max(12, round(.055 * min(w, h)))
    px, py = max(minimum, math.ceil(dx * .45)), max(minimum, math.ceil(dy * .45))
    roi = [max(0, left-px), max(0, top-py), min(w, right+px), min(h, bottom+py)]
    m, v = mask.crop(roi), valid.crop(roi)
    context = ImageChops.invert(m)
    context_count = count(context)
    need(context_count > 0, "ROI has no non-target context")
    vw, vh = v.size
    perimeter = ([v.getpixel((x,0)) for x in range(vw)] + [v.getpixel((x,vh-1)) for x in range(vw)]
                 + [v.getpixel((0,y)) for y in range(1,vh-1)] + [v.getpixel((vw-1,y)) for y in range(1,vh-1)])
    vp, cp = count(v), count(ImageChops.multiply(context, v))
    return {"roi": roi, "target_bbox": [left,top,right,bottom], "saved_frame_size": [w,h],
            "target_pixels": count(m), "unsupported_target_pixels": count(ImageChops.multiply(m,ImageChops.invert(v))),
            "roi_pixels": vw*vh, "valid_roi_pixels": vp, "valid_roi_fraction": round(vp/(vw*vh),6),
            "context_pixels": context_count, "valid_context_pixels": cp,
            "valid_context_fraction": round(cp/context_count,6),
            "valid_perimeter_fraction": round(sum(x>0 for x in perimeter)/len(perimeter),6),
            "context_margin_pixels_ltrb": [left-roi[0],top-roi[1],roi[2]-right,roi[3]-bottom],
            "frame_edge_touch": roi[0]==0 or roi[1]==0 or roi[2]==w or roi[3]==h,
            "roi_short_side": min(vw,vh), "roi_diagonal": round(math.hypot(vw,vh),3),
            "target_short_side": min(dx,dy), "target_diagonal": round(math.hypot(dx,dy),3)}


def numeric_pass(metric):
    return (metric["unsupported_target_pixels"] == 0 and metric["valid_roi_pixels"]*100 >= metric["roi_pixels"]*95
            and metric["valid_context_pixels"]*100 >= metric["context_pixels"]*90)


def geometry(chosen, variant, asset_index, files):
    """Reuse D0's record-directory recovery without fitting or altering transforms."""
    matrix = chosen["matrix_output_to_source"]
    need(len(matrix)==2 and all(len(row)==3 and all(math.isfinite(x) for x in row) for row in matrix),
         "Malformed saved affine matrix")
    need(matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0] > 0, "Invalid affine orientation")
    descriptor = chosen["geometry"]
    provenance = "A9_MANIFEST_FILE_SHA"
    if not descriptor:
        directory = Path(chosen["aligned"]["path"]).parent
        record_file = files.check(directory/"record.json")
        record = load(directory/"record.json")
        for key, value in (("review_id",chosen["review_id"]),("aligned_sha256",chosen["aligned"]["sha256"]),
                           ("source_sha256",variant["original_sha256"]),("annotation_sha256",variant["annotation_sha256"])):
            need(record[key]==value, "Recovered legacy geometry record is not bound to this exact asset")
        path = directory/"geometry.json"
        expected = asset_index.get(path.as_posix()) or record["file_sha256"].get("geometry.json")
        descriptor = files.check(path, expected)
        provenance = "LEGACY_RECORD_AND_PINNED_MATRIX" if expected is None else "LEGACY_PINNED_RECORD_FILE_SHA"
    else:
        descriptor = files.check(descriptor["path"],descriptor["sha256"])
    g = load(descriptor["path"])
    fits = [g.get("selected_fit") or {}, *g.get("candidate_diagnostics",[])]
    need(any(fit.get("matrix_output_to_source")==matrix for fit in fits), "Saved matrix differs from frozen candidate matrix")
    return descriptor, provenance


def audit(a9, d0, output, public_output=None):
    output.mkdir(parents=True)
    files = Files()
    receipt, old = load(a9/"handoff/delivery_receipt.json"), load(d0/"delivery_receipt.json")
    need(receipt["commit"]=="78c5f2c8e81134a9cc270ef2262f1a202e961bc5" and old["commit"]=="b48c29abf0437fd73c5a36d45930d5d74d490e68", "Historical delivery identities differ")
    for root, rec in ((a9,receipt),(d0,old)):
        for relative, digest in rec["private_sha256"].items():
            files.check(root/relative,digest)
    files.check(a9/"registry/b_pair_candidate_manifest.csv",B_SHA)
    files.check(d0/"b_roi_integrity_population.json",ROI_SHA)
    files.check(d0/"b_g1_pilot_manifest.csv",PILOT_SHA)
    inputs = load(a9/"inputs/input_hashes.json")
    human = {}
    for name, descriptor in inputs["human_csvs"].items():
        files.check(a9/"inputs"/name,descriptor["sha256"])
        for h in rows(a9/"inputs"/name):
            if h["review_id"] in human:
                need(human[h["review_id"]]==h,"Conflicting duplicate human review ID")
            human[h["review_id"]] = h
    candidates = rows(a9/"registry/b_pair_candidate_manifest.csv")
    source = {r["source_id"]:r for r in rows(a9/"registry/source_master_registry.csv")}
    variants = {r["review_id"]:r for r in rows(a9/"registry/variant_master_registry.csv")}
    asset_index = load(a9/"registry/candidate_asset_sha256.json")["files"]
    population = {(r["polygon_id"],r["review_id"]):r for r in load(d0/"b_roi_integrity_population.json")["cases"]}
    pilot = {(r["polygon_id"],r["review_id"]):r for r in rows(d0/"b_g1_pilot_manifest.csv")}
    split_path = Path(candidates[0]["original_path"]).parents[2]/"frozen_split_manifest.csv"
    files.check(split_path,SPLIT_SHA)
    train = {r["stem"]:r for r in rows(split_path) if r["split"]=="train"}
    need(len(candidates)==len({r["polygon_id"] for r in candidates})==545 and len(train)==668,"Cohort count changed")
    annotations, valid_cache, result, errors = {}, {}, [], []
    for row in candidates:
        try:
            sid,pid = row["source_id"],row["polygon_id"]
            s,frozen = source[sid],train[sid]
            need(s["group_id"]==frozen["group_id"] and row["original_sha256"]==frozen["sha256"]
                 and row["annotation_sha256"]==frozen["json_sha256"], "Source is not the original frozen TRAIN member")
            original = files.check(row["original_path"],row["original_sha256"])
            annotation = files.check(row["annotation_path"],row["annotation_sha256"])
            if sid not in annotations:
                ann = load(row["annotation_path"])
                with Image.open(row["original_path"]) as image:
                    need(image.size==(ann["imageWidth"],ann["imageHeight"]),"Professional JSON/image dimensions differ")
                need(len(ann["shapes"])==int(frozen["instance_count"]),"Professional instance count differs")
                annotations[sid] = ann
            ann = annotations[sid];index = int(pid.rsplit("_",1)[1]);shape = ann["shapes"][index]
            need(shape["label"]==row["class"] in CLASSES and shape["points"]==json.loads(row["polygon_coordinates"])
                 and shape["shape_type"]=="polygon", "Professional Polygon identity/class differs")
            chosen = json.loads(row["recommended"])
            choices = [chosen,*json.loads(row["alternates"])]
            choice_reports = []
            for choice in choices:
                v = variants[choice["review_id"]];h = human[choice["review_id"]]
                need((v["source_id"],v["variant_id"],v["version"])==(sid,choice["variant_id"],choice["version"]),"Wrong version/source binding")
                need(h["source_id"]==sid and h["aligned_sha256"]==choice["aligned"]["sha256"]
                     and h["source_sha256"]==original["sha256"] and h["annotation_sha256"]==annotation["sha256"]
                     and h["b_region_review"]=="CONFIRMED" and json.loads(v["resolved_polygon_decisions"])[pid]=="PASS",
                     "Historical human semantic decision is not bound to this candidate")
                aligned=files.check(choice["aligned"]["path"],choice["aligned"]["sha256"])
                valid=files.check(choice["valid_mask"]["path"],choice["valid_mask"]["sha256"])
                raw=json.loads(v["raw"]);raw=files.check(raw["path"],raw["sha256"])
                geo,provenance=geometry(choice,v,asset_index,files)
                for key in ("before","after"):
                    files.check(choice["support"][key]["path"],choice["support"][key]["sha256"])
                origin=json.loads(v["origin_paths"])
                edit = None
                if origin.get("mask"):
                    edit=files.check(origin["mask"],v["mask_sha256"])
                choice_reports.append({"variant_id":v["variant_id"],"version":v["version"],"review_id":v["review_id"],
                    "aligned":aligned,"raw":raw,"valid_mask":valid,"geometry":geo,"geometry_provenance":provenance,
                    "matrix_output_to_source":choice["matrix_output_to_source"],"historical_human":h,
                    "resolved_polygon_decisions":json.loads(v["resolved_polygon_decisions"]),
                    "edit_mask":edit,"edit_mask_scope":"HASH_BOUND_HISTORICAL_REQUEST_MASK_SEMANTICS_REQUIRE_REVIEW" if edit else "UNKNOWN",
                    "risk_evidence":choice["risk_evidence"],"pixel_asset_state":choice["pixel_asset_state"]})
            c=choice_reports[0];key=(pid,c["review_id"])
            need(key in population,"Missing frozen D0 numerical evidence")
            with Image.open(c["aligned"]["path"]) as image:
                need(image.size==(ann["imageWidth"],ann["imageHeight"]),"Aligned is not in the source grid")
            vm=c["valid_mask"]["path"]
            if vm not in valid_cache:
                with Image.open(vm) as image:valid_cache[vm]=binary(image)
            target,pixels=mask_for_selection((ann["imageWidth"],ann["imageHeight"]),ann["shapes"],{index})
            metrics=region_metrics(target,valid_cache[vm])
            need(all(metrics[k]==population[key][k] for k in metrics),"Recomputed integer-raster evidence differs from D0")
            need(pixels==chosen["support"]["pixels"] and metrics["unsupported_target_pixels"]==chosen["support"]["unsupported_pixels"],"Target support differs")
            others=[]
            for i, other in enumerate(ann["shapes"]):
                if i==index:continue
                m,_=mask_for_selection(target.size,ann["shapes"],{i})
                if m.crop(metrics["roi"]).getbbox():others.append({"polygon_id":f"{sid}_{i:03d}","class":other["label"],"same_variant_historical_state":c["resolved_polygon_decisions"].get(f"{sid}_{i:03d}","UNKNOWN")})
            hist=pilot.get(key);historical=hist["g1_result"] if hist else "NOT_VISUALLY_SAMPLED_IN_D0"
            risk=[]
            if c["geometry_provenance"]=="LEGACY_RECORD_AND_PINNED_MATRIX":risk.append("LEGACY_GEOMETRY_FILE_NO_HISTORICAL_FILE_SHA_ONLY_MATRIX_AND_ASSET_BINDING")
            if c["pixel_asset_state"]=="NEEDS_HUMAN_ASSET_CONFIRMATION":risk.append("FINAL_PIXEL_ASSET_CONFIRMATION_PENDING")
            if others:risk.append("OTHER_ANNOTATED_DEFECT_IN_ROI_REQUIRE_SEMANTIC_REVIEW")
            if s["version_conflict"]=="true":risk.append("HISTORICAL_SOURCE_VERSION_CONFLICT")
            if len(choices)>1:risk.append("SAME_POLYGON_ALTERNATE_VARIANTS_NOT_INDEPENDENT")
            if metrics["frame_edge_touch"]:risk.append("ROI_TOUCHES_FRAME_EDGE")
            if not numeric_pass(metrics):risk.append("G1_NUMERIC_SUPPORT_NOT_MET")
            corrupt=bool("\ufffd" in c["historical_human"].get("notes","") or (hist and "\ufffd" in hist["visual_note"]))
            if corrupt:risk.append("HISTORICAL_TEXT_ENCODING_LOSS_DO_NOT_INFER_MISSING_WORDS")
            status = ("UNKNOWN_HOLD" if not numeric_pass(metrics) else "G1_REJECT" if historical=="G1_PILOT_FAIL"
                      else "G1_VISUAL_PENDING" if hist else "G1_NUMERIC_CANDIDATE")
            result.append({"record_id":f"{sid}:{pid}:{c['review_id']}","source_id":sid,"polygon_id":pid,"class":row["class"],"group_id":s["group_id"],
                "review_id":c["review_id"],"variant_id":c["variant_id"],"version":c["version"],"original_path":original["path"],"original_sha256":original["sha256"],
                "annotation_path":annotation["path"],"annotation_sha256":annotation["sha256"],"aligned_path":c["aligned"]["path"],"aligned_sha256":c["aligned"]["sha256"],
                "raw_path":c["raw"]["path"],"raw_sha256":c["raw"]["sha256"],"valid_mask_path":c["valid_mask"]["path"],"valid_mask_sha256":c["valid_mask"]["sha256"],
                "geometry_path":c["geometry"]["path"],"geometry_sha256":c["geometry"]["sha256"],"geometry_provenance":c["geometry_provenance"],
                "matrix_output_to_source":c["matrix_output_to_source"],"polygon_coordinates":shape["points"],"all_annotation_polygons":ann["shapes"],
                "frame_size":target.size,"roi":metrics["roi"],"metrics":metrics,"numeric_screen_pass":numeric_pass(metrics),"edit_mask":c["edit_mask"],
                "other_annotated_defects_in_roi":others,"other_defect_contamination":"PENDING_HUMAN_NOT_NORMAL_BY_OMISSION",
                "history":{"human":c["historical_human"],"pilot":hist},"human_polygon_state":chosen["human_polygon_state"],"human_b_state":chosen["human_b_state"],"polygon_state_basis":chosen["polygon_state_basis"],"historical_g1_result":historical,"historical_text_encoding_loss":corrupt,
                "priority_six":historical==FEASIBLE,"g0_status":"G0_PASS","g1_status":status,"technical_status":status,"risk_flags":risk,
                "variants":choice_reports,"human_current":"PENDING","trusted_pair":False,"training_approved":False})
        except Exception as error:
            errors.append({"source_id":row["source_id"],"polygon_id":row["polygon_id"],"error_type":type(error).__name__,"reason":str(error)})
            result.append({"source_id":row["source_id"],"polygon_id":row["polygon_id"],"class":row["class"],"g0_status":"UNKNOWN_HOLD","g1_status":"UNKNOWN_HOLD","technical_status":"UNKNOWN_HOLD","risk_flags":[str(error)],"trusted_pair":False,"training_approved":False,"human_current":"PENDING"})
    files.postcheck()
    write(output/"candidates.json",{"schema":"exp17_pair_identity_audit_v1","candidates":result,"errors":errors})
    selected=[r for r in result if r.get("priority_six")]
    diagnostics=[r for r in result if (r.get("history") or {}).get("pilot")]
    write(output/"priority_six.json",{"cases":selected,"human_pass":0,"trusted_pairs":0})
    write(output/"diagnostic_inputs.json",{"cases":diagnostics,"scope":"READ_ONLY_TRAIN_FEATURE_DIAGNOSTICS_NOT_ADMISSION","human_pass":0})
    write(output/"verified_files.json",{"files":list(files.records.values()),"post_run_sha256_rechecked":True})
    summary={"status":"PASS_EXP17_ASSET_IDENTITY_AUDIT" if not errors else "EXP17_ASSET_AUDIT_HOLD_WITH_IDENTITY_ERRORS",
        "candidate_count":len(result),"original_source_ids":len({r["source_id"] for r in result}),"source_groups":len({r.get("group_id") for r in result if r.get("group_id")}),
        "variant_polygon_combinations_audited":sum(len(r.get("variants",[])) for r in result),"file_bytes_sha256_verified":len(files.records),"identity_errors":len(errors),
        "g0_status":dict(Counter(r["g0_status"] for r in result)),"g1_technical_strata":dict(Counter(r["g1_status"] for r in result)),
        "g1_numeric_candidates":sum(r.get("numeric_screen_pass",False) for r in result),"complete_roi":sum(r.get("metrics",{}).get("valid_roi_fraction")==1 for r in result),
        "frame_edge_rois":sum(r.get("metrics",{}).get("frame_edge_touch",False) for r in result),"historical_pilot_cases":len(diagnostics),"historical_feasible_priority_cases":len(selected),
        "historical_human_pair_pass":0,"current_user_human_pass":0,"formal_trusted_pairs":0,"training_approved":0,
        "risk_flags":dict(Counter(flag for r in result for flag in r["risk_flags"])),
        "class_counts":{c:{"polygons":sum(r["class"]==c for r in result),"source_ids":len({r["source_id"] for r in result if r["class"]==c}),
                            "source_groups":len({r.get("group_id") for r in result if r["class"]==c and r.get("group_id")}),"human_pass":0} for c in CLASSES},
        "input_sha256":{"b_manifest":B_SHA,"d0_numeric_population":ROI_SHA,"d0_pilot":PILOT_SHA,"frozen_split":SPLIT_SHA},
        "private_ledger_sha256":sha(output/"candidates.json"),"verified_files_sha256":sha(output/"verified_files.json"),
        "historical_text_replacement_character_cases":sum(r.get("historical_text_encoding_loss",False) for r in result),
        "limitations":["Numeric support is not visual G1 approval.","Historical feasible trials are not current human PASS.","Legacy geometry file SHA provenance is distinguished from its pinned affine matrix.","Unannotated context is not assumed normal.","UTF-8 historical notes are preserved; console display garbling is not evidence loss. Inspect the explicit replacement-character case count."],
        "new_registration":0,"new_generation":0,"optimizer_updates":0,"val_test_image_reads":0,"completed_utc":datetime.now(timezone.utc).isoformat()}
    write(output/"summary.json",summary)
    if public_output:
        public_output.mkdir(parents=True)
        write(public_output/"asset_audit_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("a9-root","d0-root","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--public-output",type=Path)
    args=p.parse_args()
    audit(args.a9_root,args.d0_root,args.output,args.public_output)


if __name__=="__main__":
    main()
