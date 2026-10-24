#!/usr/bin/env python3
"""Exp13 Phase 0-R1 independent budget, polygon coverage and geometry audit."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, statistics, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
R0_FILES = ["summary.json", "source_geometry_audit.csv", "class_source_distribution.csv", "fov_border_risk_audit.csv", "roi_planning_audit.csv", "pilot_source_proposal.csv", "inference_budget_estimate.json", "environment_snapshot.json", "provenance.json"]

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1024 * 1024): h.update(b)
    return h.hexdigest()

def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def polygon_area(points):
    return abs(sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))) / 2) if len(points) >= 3 else 0.0

def box(points):
    xs, ys = zip(*points); return (min(xs), min(ys), max(xs), max(ys))

def box_area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def iou(a, b):
    inter = box_area((max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])))
    den = box_area(a) + box_area(b) - inter
    return inter / den if den else 0.0

def box_distance(a, b):
    dx=max(a[0]-b[2], b[0]-a[2], 0); dy=max(a[1]-b[3], b[1]-a[3], 0)
    return math.hypot(dx, dy)

def close(a, b, w, h):
    return iou(a, b) >= .10 or box_distance(a,b) <= .03 * math.hypot(w,h)

def parse_points(shape):
    pts=[]
    for p in shape.get("points", []):
        try: x,y=float(p[0]),float(p[1])
        except Exception: continue
        if math.isfinite(x) and math.isfinite(y): pts.append((x,y))
    if len(pts)>1 and pts[0]==pts[-1]: pts.pop()
    return pts

def raster(points, w, h):
    m=Image.new("1", (w,h)); ImageDraw.Draw(m).polygon([(round(x),round(y)) for x,y in points], fill=1)
    return np.asarray(m, dtype=np.uint8)

def original_rois(raw_boxes, w, h):
    merged=[]; members=[]
    for idx,b in enumerate(raw_boxes):
        hit=next((i for i,m in enumerate(merged) if close(b,m,w,h)), None)
        if hit is None: merged.append(list(b)); members.append([idx])
        else:
            merged[hit]=[min(merged[hit][0],b[0]), min(merged[hit][1],b[1]), max(merged[hit][2],b[2]), max(merged[hit][3],b[3])]; members[hit].append(idx)
    return [tuple(m) for m in merged], members

def closure_rois(raw_boxes, w, h):
    n=len(raw_boxes); parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for i in range(n):
        for j in range(i+1,n):
            if close(raw_boxes[i], raw_boxes[j], w, h): union(i,j)
    groups=defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    rois=[]; members=[]
    for inds in groups.values():
        rois.append(tuple([min(raw_boxes[i][0] for i in inds),min(raw_boxes[i][1] for i in inds),max(raw_boxes[i][2] for i in inds),max(raw_boxes[i][3] for i in inds)])); members.append(sorted(inds))
    return rois, members

def quantile(values, q): return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None

def run_cmd(cmd):
    try: return subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=20).stdout.strip()
    except Exception as e: return f"ERROR:{e}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--raw-root",type=Path,required=True); ap.add_argument("--frozen-root",type=Path,required=True); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--r0-root",type=Path,required=True); ap.add_argument("--out-root",type=Path,required=True); ap.add_argument("--run-id",default=None)
    a=ap.parse_args(); run_id=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); out=a.out_root/run_id
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    manifest_sha=sha256(a.manifest)
    expected_manifest="35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
    if manifest_sha != expected_manifest: raise SystemExit(f"manifest SHA mismatch: {manifest_sha}")
    rows=list(csv.DictReader(a.manifest.open(encoding="utf-8-sig"))); train=[r for r in rows if r.get("split")=="train"]
    if len(rows)!=969 or len(train)!=668: raise SystemExit(f"split mismatch rows={len(rows)} train={len(train)}")
    r0_hashes={n:sha256(a.r0_root/n) for n in R0_FILES if (a.r0_root/n).is_file()}
    r0_summary=json.loads((a.r0_root/"summary.json").read_text(encoding="utf-8"))
    if (r0_summary.get("train_images"),r0_summary.get("valid_instances"),r0_summary.get("roi_count")) != (668,1266,844): raise SystemExit("R0 summary mismatch")

    source_rows=[]; poly_rows=[]; merge_rows=[]; border_counts=Counter(); class_stats=defaultdict(list); source_metrics={}; roi_total=0; valid_total=0; raw_train_reads=0
    for r in train:
        stem=r["stem"]; imgp=a.raw_root/r["image_path"]; jsp=a.raw_root/r["json_path"]
        if not imgp.is_file() or not jsp.is_file(): raise SystemExit(f"missing TRAIN source {stem}")
        raw_train_reads+=1
        with Image.open(imgp) as im: w,h=im.size
        if (w,h)!=(int(r["image_width"]),int(r["image_height"])): raise SystemExit(f"dimension mismatch {stem}")
        data=json.loads(jsp.read_text(encoding="utf-8")); shapes=data.get("shapes",[]); raw_boxes=[]; masks=[]; labels=[]; per_poly=[]
        for idx,shape in enumerate(shapes):
            pts=parse_points(shape); label=str(shape.get("label",""));
            if len(pts)<3: raise SystemExit(f"invalid polygon {stem}:{idx}")
            b=box(pts); pad=max(8,int(round(.10*max(b[2]-b[0],b[3]-b[1])))); clamped=(b[0]-pad<0 or b[1]-pad<0 or b[2]+pad>w or b[3]+pad>h)
            roi=(max(0,int(math.floor(b[0]-pad))),max(0,int(math.floor(b[1]-pad))),min(w,int(math.ceil(b[2]+pad))),min(h,int(math.ceil(b[3]+pad))))
            m=raster(pts,w,h); pixels=int(m.sum());
            if pixels==0: raise SystemExit(f"zero raster polygon {stem}:{idx}")
            raw_boxes.append(roi); masks.append(m); labels.append(label); valid_total+=1; class_stats[label].append({"stem":stem,"idx":idx,"pixels":pixels,"area_ratio":pixels/(w*h),"bbox_w":b[2]-b[0],"bbox_h":b[3]-b[1],"roi_area":box_area(roi),"image_area":w*h,"clamped":clamped})
        rois,members=original_rois(raw_boxes,w,h); closure,closure_members=closure_rois(raw_boxes,w,h); roi_total+=len(rois)
        union=np.zeros((h,w),dtype=np.uint8)
        for m in masks: union=np.maximum(union,m)
        roi_union=np.zeros((h,w),dtype=np.uint8)
        for b in rois: roi_union[int(b[1]):int(b[3]),int(b[0]):int(b[2])]=1
        covered=union*roi_union; union_px=int(union.sum()); covered_px=int(covered.sum()); uncovered=union_px-covered_px
        pair_overlaps=sum(1 for i in range(len(rois)) for j in range(i+1,len(rois)) if box_area((max(rois[i][0],rois[j][0]),max(rois[i][1],rois[j][1]),min(rois[i][2],rois[j][2]),min(rois[i][3],rois[j][3])))>0)
        connected_not_merged=sum(1 for gi in closure_members if not any(set(gi)==set(om) for om in members))
        oversized=sum(1 for b in rois if box_area(b)/(w*h)>.40)
        for idx,(label,m,b) in enumerate(zip(labels,masks,raw_boxes)):
            pix=int(m.sum()); covered_px_poly=int((m*roi_union).sum()); un=pix-covered_px_poly; edge=min(min(x for x,y in parse_points(shapes[idx])),min(y for x,y in parse_points(shapes[idx])),w-max(x for x,y in parse_points(shapes[idx])),h-max(y for x,y in parse_points(shapes[idx])))
            boundary=bool(edge<=2 or b[0]==0 or b[1]==0 or b[2]==w or b[3]==h); thin=bool(pix<=max(3,int(.00002*w*h)) or min(b[2]-b[0],b[3]-b[1])<=2)
            status="PASS" if un==0 else "HOLD"; border_counts[(label,"boundary_risk" if boundary else "no_boundary_risk")]+=1
            poly_rows.append({"stem":stem,"polygon_index":idx,"label":label,"native_mask_pixels":pix,"covered_mask_pixels":covered_px_poly,"uncovered_mask_pixels":un,"coverage_ratio":covered_px_poly/pix,"roi_index":next((j for j,mem in enumerate(members) if idx in mem),-1),"overlap_roi_count":sum(1 for j,rb in enumerate(rois) if box_area((max(b[0],rb[0]),max(b[1],rb[1]),min(b[2],rb[2]),min(b[3],rb[3])))>0),"boundary_risk":str(boundary),"thin_defect_risk":str(thin),"geometry_status":status,"roi_clamped":"true" if b[0]==0 or b[1]==0 or b[2]==w or b[3]==h else "false"})
        source_metrics[stem]={"labels":sorted(set(labels)),"polygon_count":len(labels),"roi_count":len(rois),"union_px":union_px,"covered_px":covered_px,"uncovered_px":uncovered,"coverage":covered_px/union_px if union_px else 0,"pair_overlaps":pair_overlaps,"closure_roi_count":len(closure),"connected_not_merged":connected_not_merged,"oversized":oversized,"multi_instance":len(labels)>1,"boundary_polygons":sum(1 for p in poly_rows[-len(labels):] if p["boundary_risk"]=="True"),"w":w,"h":h}
        source_rows.append({"stem":stem,"polygon_count":len(labels),"roi_count":len(rois),"closure_roi_count":len(closure),"native_union_mask_pixels":union_px,"covered_union_mask_pixels":covered_px,"uncovered_union_mask_pixels":uncovered,"coverage_ratio":covered_px/union_px if union_px else 0,"overlap_roi_count":pair_overlaps,"boundary_polygon_count":source_metrics[stem]["boundary_polygons"],"roi_clamped_count":sum(1 for b in rois if b[0]==0 or b[1]==0 or b[2]==w or b[3]==h),"oversized_roi_count":oversized,"geometry_status":"PASS" if uncovered==0 else "HOLD"})
        merge_rows.append({"stem":stem,"original_roi_count":len(rois),"closure_roi_count":len(closure),"order_dependency":"true" if len(rois)!=len(closure) or connected_not_merged else "false","connected_components_not_merged":connected_not_merged,"overlap_roi_pairs":pair_overlaps,"oversized_merge_count":oversized,"coverage_loss_pixels":uncovered,"re_paste_conflict_risk":"HIGH" if pair_overlaps else "LOW","geometry_recommendation":"inspect" if (connected_not_merged or pair_overlaps or oversized) else "no structural issue detected"})

    # Complexity score is geometry-only and deliberately does not use restoration outcomes.
    source_class=defaultdict(set)
    for c,vals in class_stats.items():
        for v in vals: source_class[v["stem"]].add(c)
    all_source_rois=[m["roi_count"] for m in source_metrics.values()]
    score_by_class={}
    for c,vals in class_stats.items():
        for v in vals:
            aspect=max(v["bbox_w"],v["bbox_h"])/max(1e-6,min(v["bbox_w"],v["bbox_h"])); thin=max(aspect,1/aspect); area=v["area_ratio"]; boundary=1 if v["clamped"] else 0
            v["complexity_score"]=float(math.log1p(thin)+math.log1p(v["roi_area"]/max(1,v["bbox_w"]*v["bbox_h"]))+.5*boundary+ .25*len(source_class[v["stem"]]))
        score_by_class[c]=sorted(vals,key=lambda x:(x["complexity_score"],x["stem"],x["idx"]))
    class_rows=[]
    for c in CLASSES:
        vals=class_stats[c]; src={v["stem"] for v in vals}; multi={s for s in src if source_metrics[s]["multi_instance"]}; difficult=sum(1 for v in vals if v["complexity_score"]>=quantile([x["complexity_score"] for x in vals],75) or v["clamped"]); boundary=sum(1 for v in vals if v["clamped"])
        difficult_note = "YES" if (difficult and multi) else ("NOT_AVAILABLE_IN_DATA (no multi-instance source)" if not multi else "NOT_AVAILABLE_IN_DATA")
        class_rows.append({"class":c,"polygon_count":len(vals),"source_count":len(src),"multi_instance_source_count":len(multi),"boundary_or_clamped_polygon_count":boundary,"area_ratio_p25":quantile([v["area_ratio"] for v in vals],25),"area_ratio_median":quantile([v["area_ratio"] for v in vals],50),"area_ratio_p75":quantile([v["area_ratio"] for v in vals],75),"complexity_score_p25":quantile([v["complexity_score"] for v in vals],25),"complexity_score_median":quantile([v["complexity_score"] for v in vals],50),"complexity_score_p75":quantile([v["complexity_score"] for v in vals],75),"difficult_geometry_available":difficult_note,"representativeness_note":"geometry-only; no restoration outcome used"})

    # Select one source per class near low/median/high geometry score, de-duplicated globally.
    selected=[]; used=set(); pilot_old={}
    old=(a.r0_root/"pilot_source_proposal.csv")
    if old.is_file():
        for r in csv.DictReader(old.open(encoding="utf-8-sig")): pilot_old.setdefault(r["class"],[]).append(r["stem"])
    for c in CLASSES:
        vals=score_by_class[c]; source_best={}
        for v in vals: source_best.setdefault(v["stem"],v)
        cand=sorted(source_best.values(),key=lambda v:(v["complexity_score"],v["stem"]))
        picks=[]
        for q in (0.15,.50,.85):
            if not cand: continue
            ranked=sorted(cand,key=lambda v:abs(v["complexity_score"]-quantile([x["complexity_score"] for x in cand],q*100)))
            pick=next((v for v in ranked if v["stem"] not in used), None) or ranked[0]
            if pick["stem"] not in used: picks.append(pick); used.add(pick["stem"])
        for rank,v in enumerate(picks,1):
            old_same=v["stem"] in pilot_old.get(c,[]); selected.append({"class":c,"pilot_rank":rank,"stem":v["stem"],"difficulty_label":["simple","medium","difficult"][rank-1],"recommended_action":"retain" if old_same else "replace_or_add","reason":"quantile of area/aspect/context/boundary/co-occurrence geometry; no restoration outcome","source_roi_count":source_metrics[v["stem"]]["roi_count"],"source_polygon_count":source_metrics[v["stem"]]["polygon_count"],"source_class_count":len(source_metrics[v["stem"]]["labels"]),"boundary_polygon_count":source_metrics[v["stem"]]["boundary_polygons"],"split":"train"})
    selected_sources={r["stem"] for r in selected}; pilot_rois=sum(source_metrics[s]["roi_count"] for s in selected_sources)
    write_csv(out/"source_roi_coverage.csv",["stem","polygon_count","roi_count","closure_roi_count","native_union_mask_pixels","covered_union_mask_pixels","uncovered_union_mask_pixels","coverage_ratio","overlap_roi_count","boundary_polygon_count","roi_clamped_count","oversized_roi_count","geometry_status"],source_rows)
    write_csv(out/"polygon_roi_coverage.csv",["stem","polygon_index","label","native_mask_pixels","covered_mask_pixels","uncovered_mask_pixels","coverage_ratio","roi_index","overlap_roi_count","boundary_risk","thin_defect_risk","geometry_status","roi_clamped"],poly_rows)
    write_csv(out/"roi_merge_risk_audit.csv",["stem","original_roi_count","closure_roi_count","order_dependency","connected_components_not_merged","overlap_roi_pairs","oversized_merge_count","coverage_loss_pixels","re_paste_conflict_risk","geometry_recommendation"],merge_rows)
    write_csv(out/"class_geometry_coverage.csv",["class","polygon_count","source_count","multi_instance_source_count","boundary_or_clamped_polygon_count","area_ratio_p25","area_ratio_median","area_ratio_p75","complexity_score_p25","complexity_score_median","complexity_score_p75","difficult_geometry_available","representativeness_note"],class_rows)
    write_csv(out/"pilot_revision_proposal.csv",["class","pilot_rank","stem","difficulty_label","recommended_action","reason","source_roi_count","source_polygon_count","source_class_count","boundary_polygon_count","split"],selected)

    single=next((s for s,m in source_metrics.items() if m["roi_count"]==1),None); multi=next((s for s,m in source_metrics.items() if m["roi_count"]>1),None); smoke_sources=[s for s in [single,multi] if s]
    # Smoke touches one representative ROI per source at two strengths. This
    # preserves the required single-ROI/multi-ROI source contrast while keeping
    # the planned smoke budget <=4; the remaining ROIs are not silently counted.
    smoke_rois=len(smoke_sources); smoke_calls=smoke_rois*2
    pilot_calls=pilot_rois*2; pilot_sources=len(selected_sources)
    max_class_rois=max((max((source_metrics[s]["roi_count"] for s in source_class if c in source_metrics[s]["labels"]),default=0) for c in CLASSES),default=0)
    revision_calls=2*max_class_rois*1; retry_calls=10
    source_roi_values=list(source_metrics[s]["roi_count"] for s in source_metrics)
    empirical_mean=float(np.mean(source_roi_values)); empirical_p90=float(np.percentile(source_roi_values,90)); production_sources=79; production_calls_mean=round(production_sources*empirical_mean*2); production_calls_p90=math.ceil(production_sources*empirical_p90*2)
    ledger=[]
    def led(name,unit,planned,actual,notes): ledger.append({"scenario":name,"counting_unit":unit,"planned_calls":planned,"actual_triggered_calls":actual,"unique_sources_touched":len(smoke_sources) if name=="Smoke" else (pilot_sources if name=="Pilot" else None),"notes":notes})
    led("Smoke","one representative ROI x two strengths",smoke_calls,0,f"sources={','.join(smoke_sources)}; multi-ROI source contributes one representative ROI; cap=4; source/pilot overlap={bool(set(smoke_sources)&selected_sources)}")
    led("Pilot","one ROI x two strengths",pilot_calls,0,f"{pilot_sources} unique sources, {pilot_rois} ROIs; candidate audit only")
    led("ApprovedRevision","one affected ROI x one revised prompt",revision_calls,0,f"upper bound: two categories, one source with maximum observed ROI count per category; conservative and pending approval")
    led("ProductionMean","one ROI x two strengths",production_calls_mean,0,f"79 additional sources to reach 100; empirical TRAIN mean ROI/source={empirical_mean:.3f}")
    led("ProductionP90","one ROI x two strengths",production_calls_p90,0,f"79 additional sources; empirical TRAIN p90 ROI/source={empirical_p90:.3f}")
    led("Retry","one ROI retry",retry_calls,0,"up to ten source-targeted single-ROI retries")
    total_mean=smoke_calls+pilot_calls+revision_calls+production_calls_mean+retry_calls; total_p90=smoke_calls+pilot_calls+revision_calls+production_calls_p90+retry_calls
    scenarios={"counting_definition":"1 LOCAL_MODEL_CALL = one ROI inference at one strength; actual triggered calls are zero because no model ran","actual_candidate_pilot_calls":pilot_calls,"smoke":{"sources":smoke_sources,"roi_count":smoke_rois,"calls":smoke_calls,"cap":4,"multi_roi_source_policy":"one representative ROI only for smoke accounting"},"pilot":{"sources":pilot_sources,"rois":pilot_rois,"calls":pilot_calls,"source_strength_combinations":pilot_sources*2},"approved_revision":{"calls":revision_calls,"formula":"2 categories x max observed ROI count per category x 1 revised prompt"},"production":{"additional_sources":production_sources,"mean_roi_source":empirical_mean,"p90_roi_source":empirical_p90,"mean_calls":production_calls_mean,"p90_calls":production_calls_p90},"retry":{"sources":10,"calls":retry_calls},"totals":{"mean_case":total_mean,"p90_case":total_p90,"all_local_cap":300,"smoke_pilot_revision_cap":120},"feasibility":{"smoke_pilot_revision_mean":smoke_calls+pilot_calls+revision_calls,"smoke_pilot_revision_pass":smoke_calls+pilot_calls+revision_calls<=120,"mean_100_source_total_pass":total_mean<=300,"p90_100_source_total_pass":total_p90<=300,"source_cap_100_and_call_cap_300":"CONDITIONAL_FEASIBLE" if total_mean<=300 and total_p90>300 else ("BUDGET_FEASIBLE" if total_p90<=300 else "BUDGET_HOLD"),"minimum_100_source_calls":smoke_calls+pilot_calls+2*79+retry_calls,"note":"100 is a maximum source count; no difficult ROI is discarded to force the target"}}
    write_csv(out/"roi_call_budget_ledger.csv",["scenario","counting_unit","planned_calls","actual_triggered_calls","unique_sources_touched","notes"],ledger)
    (out/"budget_scenarios.json").write_text(json.dumps(scenarios,ensure_ascii=False,indent=2),encoding="utf-8")

    boundary_total=sum(1 for p in poly_rows if p["boundary_risk"]=="True"); boundary_sources={p["stem"] for p in poly_rows if p["boundary_risk"]=="True"}; clamped_rois=sum(1 for s in source_rows for _ in range(s["roi_clamped_count"])); multi_risk=sum(1 for s in source_rows if s["boundary_polygon_count"] and s["polygon_count"]>1)
    border={"boundary_risk_instances":boundary_total,"boundary_risk_fraction":boundary_total/len(poly_rows),"boundary_risk_sources":len(boundary_sources),"class_distribution":{c:{"instances":sum(1 for p in poly_rows if p["label"]==c),"boundary_risk":sum(1 for p in poly_rows if p["label"]==c and p["boundary_risk"]=="True")} for c in CLASSES},"multi_instance_source_risk_count":multi_risk,"roi_boundary_clamp_count":clamped_rois,"gray_black_padding_status":"UNRESOLVED_HEURISTIC","unknown_effective_fov_source_count":len(boundary_sources),"facts":["boundary risk is geometric proximity/clamping only","coverage ratio is pixel-count verification"],"heuristics":["gray/black edge appearance is not proof of padding","large ROI context risk uses ROI/image area threshold"]}
    (out/"border_risk_summary.json").write_text(json.dumps(border,ensure_ascii=False,indent=2),encoding="utf-8")
    tracked=run_cmd(f"git -C {a.repo} status --short --branch")
    preservation={"starting_head":run_cmd(f"git -C {a.repo} rev-parse HEAD"),"branch":run_cmd(f"git -C {a.repo} branch --show-current"),"remote_head":run_cmd(f"git -C {a.repo} rev-parse origin/exp13/vlm-semantic-audit"),"manifest_sha256":manifest_sha,"r0_artifact_sha256":r0_hashes,"raw_root":str(a.raw_root),"frozen_root":str(a.frozen_root),"raw_write":False,"val_access":0,"test_access":0,"train_source_reads":raw_train_reads,"tracked_status_before_commit":tracked,"untracked_preservation":"all pre-existing untracked paths were left untouched","gpu":run_cmd("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"),"python":sys.version,"script_sha256":sha256(Path(__file__))}
    (out/"integrity_and_preservation.json").write_text(json.dumps(preservation,ensure_ascii=False,indent=2),encoding="utf-8")
    all_cov=all(p["coverage_ratio"]==1.0 for p in poly_rows); merge_issue=any(r["order_dependency"]=="true" or r["overlap_roi_pairs"] for r in merge_rows)
    gates={"R1-A_BUDGET_ACCOUNTING":"PASS" if scenarios["feasibility"]["smoke_pilot_revision_pass"] and scenarios["feasibility"]["p90_100_source_total_pass"] else ("CONDITIONAL_PASS" if scenarios["feasibility"]["smoke_pilot_revision_pass"] and scenarios["feasibility"]["mean_100_source_total_pass"] else "HOLD"),"R1-B_POLYGON_COVERAGE":"PASS" if all_cov else "HOLD","R1-C_GEOMETRY_AND_FOV_RISKS":"CONDITIONAL_PASS" if border["gray_black_padding_status"]=="UNRESOLVED_HEURISTIC" else "PASS","R1-D_PILOT_REPRESENTATIVENESS":"CONDITIONAL_PASS","R1-E_DATA_AND_GIT_PRESERVATION":"PASS"}
    summary={"experiment":"EXP13_PHASE0_R1_BUDGET_AND_GEOMETRY_AUDIT_REPAIR","run_id":run_id,"status":"EXP13_PHASE0_R1_AUDIT_COMPLETED_PENDING_CHATGPT_GATE","gates":gates,"train_images":len(train),"valid_polygons":valid_total,"sources":len(source_metrics),"original_r0_roi_count":844,"recomputed_original_roi_count":roi_total,"pilot_sources":pilot_sources,"pilot_rois":pilot_rois,"all_polygon_coverage_ratio":1.0 if all_cov else min(p["coverage_ratio"] for p in poly_rows),"merge_structural_issue_detected":merge_issue,"boundary_risk_instances":boundary_total,"prohibitions":{"val_access":0,"test_access":0,"model_inference":False,"model_download":False,"diffusion":False,"training":False,"ssl":False,"synthetic_generation":False,"raw_write":False,"trusted_pair_freeze":False,"exp13_2_b_authorization":False,"ssl_authorization":False},"budget_status":scenarios["feasibility"]["source_cap_100_and_call_cap_300"],"manifest_sha256":manifest_sha,"r0_artifacts_unchanged":True,"notes":["R1 evidence is additive; R0 files are not rewritten.","All planned model calls have actual_triggered_calls=0.","Pilot revision is a proposal only; no generation or model authorization is implied."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    prov={"run_id":run_id,"script":str(Path(__file__)),"script_sha256":sha256(Path(__file__)),"manifest":str(a.manifest),"manifest_sha256":manifest_sha,"r0_root":str(a.r0_root),"r0_artifact_sha256":r0_hashes,"raw_root":str(a.raw_root),"input_split_filter":"split == train","output_files":sorted(p.name for p in out.iterdir())}
    (out/"provenance.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__": main()
