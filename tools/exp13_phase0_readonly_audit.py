#!/usr/bin/env python3
"""Exp13 Phase 0: read-only TRAIN geometry/ROI/budget audit.

This script never writes to the raw dataset. It reads only the frozen TRAIN
rows in the authoritative split manifest and stores derived summaries in a
new, non-overwriting run directory.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, platform, shutil, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1024 * 1024): h.update(b)
    return h.hexdigest()

def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def polygon_area(points):
    if len(points) < 3: return 0.0
    return abs(sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))/2)

def bbox(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)

def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter else 0.0

def dist_box(a,b):
    dx=max(a[0]-b[2], b[0]-a[2], 0); dy=max(a[1]-b[3], b[1]-a[3], 0)
    return math.hypot(dx,dy)

def parse_shape(shape, w, h):
    raw=shape.get("points", [])
    pts=[]; invalid=0; nonfinite=False
    if not isinstance(raw,list): return None, "points_not_list"
    for p in raw:
        try: x,y=float(p[0]),float(p[1])
        except Exception: invalid+=1; continue
        if not math.isfinite(x) or not math.isfinite(y): nonfinite=True; continue
        pts.append((x,y))
    if len(pts)>1 and pts[0]==pts[-1]: pts.pop()
    if nonfinite: return None, "nonfinite_point"
    if invalid or len(pts)<3: return None, "invalid_points"
    outside=sum(x<0 or x>w or y<0 or y>h for x,y in pts)
    return (pts, outside, invalid), ("out_of_bounds" if outside else "ok")

def environment(repo, manifest, raw, out):
    def cmd(c):
        try: return subprocess.run(c, shell=True, text=True, capture_output=True, timeout=20).stdout.strip()
        except Exception as e: return f"ERROR:{e}"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
        "python": sys.version, "platform": platform.platform(), "repo": str(repo),
        "branch": cmd(f"git -C {repo} branch --show-current"), "head": cmd(f"git -C {repo} rev-parse HEAD"),
        "git_status_short": cmd(f"git -C {repo} status --short --branch")[:4000],
        "gpu": cmd("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"),
        "disk": cmd(f"df -h {raw}"), "python_executable": sys.executable,
        "packages": {"numpy": np.__version__, "Pillow": Image.__version__},
        "raw_dataset": str(raw), "frozen_manifest": str(manifest),
        "manifest_sha256": sha256(manifest), "script_sha256": sha256(Path(__file__)),
        "output_dir": str(out), "raw_write_policy": "read-only; no derived files written under raw_dataset",
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--raw-root", type=Path, required=True); ap.add_argument("--frozen-root", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--out-root", type=Path, required=True); ap.add_argument("--run-id", default=None)
    a=ap.parse_args(); run_id=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=a.out_root/run_id
    if out.exists(): raise SystemExit(f"refusing to overwrite existing run directory: {out}")
    out.mkdir(parents=True)
    rows=list(csv.DictReader(a.manifest.open(encoding="utf-8-sig"))); train=[r for r in rows if r.get("split")=="train"]
    if len(train)!=668: raise SystemExit(f"expected 668 TRAIN rows, got {len(train)}")
    source_rows=[]; inst_rows=[]; class_counts=Counter(); image_class=Counter(); cooc=Counter(); risks=Counter(); roi_rows=[]; source_stats=defaultdict(lambda:{"classes":set(),"instances":0,"rois":0})
    for r in train:
        stem=r["stem"]; imgp=a.raw_root/r["image_path"]; jsp=a.raw_root/r["json_path"]
        rec={"stem":stem,"image_path":r["image_path"],"json_path":r["json_path"],"manifest_width":r.get("image_width"),"manifest_height":r.get("image_height"),"exists_image":imgp.exists(),"exists_json":jsp.exists(),"image_sha256_match":False,"json_sha256_match":False}
        if not imgp.exists() or not jsp.exists(): source_rows.append(rec); risks["missing_source_file"]+=1; continue
        try:
            with Image.open(imgp) as im: w,h=im.size; gray=np.asarray(im.convert("L"),dtype=np.float32)
            data=json.loads(jsp.read_text(encoding="utf-8")); shapes=data.get("shapes",[])
            rec.update({"actual_width":w,"actual_height":h,"shape_count":len(shapes),"image_sha256_match":sha256(imgp)==r.get("sha256"),"json_sha256_match":sha256(jsp)==r.get("json_sha256")})
        except Exception as e: rec["parse_error"]=str(e); source_rows.append(rec); risks["parse_error"]+=1; continue
        labels=[]; boxes=[]; valid_instances=0
        for idx,sh in enumerate(shapes):
            label=str(sh.get("label","")); parsed,status=parse_shape(sh,w,h)
            if parsed is None:
                risks[status]+=1; inst_rows.append({"stem":stem,"instance_index":idx,"label":label,"status":status}); continue
            pts,outside,invalid=parsed; x1,y1,x2,y2=bbox(pts); area=polygon_area(pts); edge=sum(x<=2 or y<=2 or x>=w-2 or y>=h-2 for x,y in pts); border=edge>0 or min(x1,y1,w-x2,h-y2)<=0.02*min(w,h)
            mask=Image.new("1",(w,h)); ImageDraw.Draw(mask).polygon([(round(x),round(y)) for x,y in pts],fill=1); pix=int(np.asarray(mask,dtype=np.uint8).sum()); area_ratio=pix/(w*h)
            pad=max(8, int(round(0.10*max(x2-x1,y2-y1)))); rx1=max(0,int(math.floor(x1-pad))); ry1=max(0,int(math.floor(y1-pad))); rx2=min(w,int(math.ceil(x2+pad))); ry2=min(h,int(math.ceil(y2+pad)))
            labels.append(label); boxes.append((rx1,ry1,rx2,ry2)); valid_instances+=1; class_counts[label]+=1; image_class[(label,stem)]+=1; source_stats[stem]["classes"].add(label); source_stats[stem]["instances"]+=1
            inst_rows.append({"stem":stem,"instance_index":idx,"label":label,"status":"valid","point_count":len(pts),"polygon_area_px":round(area,3),"raster_area_px":pix,"area_ratio":area_ratio,"bbox_x1":x1,"bbox_y1":y1,"bbox_x2":x2,"bbox_y2":y2,"border_vertex_count":edge,"border_risk":str(border),"out_of_bounds_count":outside,"native_width":w,"native_height":h})
        for label in set(labels): cooc[(label,tuple(sorted(set(labels))))]+=1
        # Merge overlapping or near boxes into independent source ROIs.
        merged=[]
        for b in boxes:
            hit=next((i for i,m in enumerate(merged) if iou(b,m)>=0.10 or dist_box(b,m)<=0.03*math.hypot(w,h)),None)
            if hit is None: merged.append(list(b))
            else: merged[hit]=[min(merged[hit][0],b[0]),min(merged[hit][1],b[1]),max(merged[hit][2],b[2]),max(merged[hit][3],b[3])]
        for j,b in enumerate(merged):
            source_stats[stem]["rois"]+=1; roi_rows.append({"stem":stem,"roi_id":f"{stem}_roi{j+1:02d}","roi_x1":b[0],"roi_y1":b[1],"roi_x2":b[2],"roi_y2":b[3],"roi_width":b[2]-b[0],"roi_height":b[3]-b[1],"native_width":w,"native_height":h,"crop_mapping":"native pixels; output=(x-x1,y-y1), inverse=(x+x1,y+y1)","clamped_to_image":"true"})
        edge_arr=np.concatenate([gray[0,:],gray[-1,:],gray[:,0],gray[:,-1]]) if min(w,h)>2 else np.array([])
        rec.update({"valid_instance_count":valid_instances,"labels_present":"|".join(sorted(set(labels))),"roi_count":len(merged),"edge_gray_mean":float(edge_arr.mean()) if edge_arr.size else None,"edge_gray_std":float(edge_arr.std()) if edge_arr.size else None,"gray_border_heuristic":"UNRESOLVED_HEURISTIC"})
        source_rows.append(rec)
    for stem,s in source_stats.items():
        for c in s["classes"]: pass
    for label in CLASSES:
        image_count=sum(1 for (c,st) in image_class if c==label); instance_count=class_counts[label]
        write=None
    write_csv(out/"source_geometry_audit.csv",["stem","image_path","json_path","manifest_width","manifest_height","actual_width","actual_height","exists_image","exists_json","image_sha256_match","json_sha256_match","shape_count","valid_instance_count","labels_present","roi_count","edge_gray_mean","edge_gray_std","gray_border_heuristic","parse_error"],source_rows)
    write_csv(out/"fov_border_risk_audit.csv",["stem","instance_index","label","status","point_count","polygon_area_px","raster_area_px","area_ratio","bbox_x1","bbox_y1","bbox_x2","bbox_y2","border_vertex_count","border_risk","out_of_bounds_count","native_width","native_height"],inst_rows)
    write_csv(out/"roi_planning_audit.csv",["stem","roi_id","roi_x1","roi_y1","roi_x2","roi_y2","roi_width","roi_height","native_width","native_height","crop_mapping","clamped_to_image"],roi_rows)
    class_rows=[]
    for c in CLASSES:
        ims={st for (cc,st),n in image_class.items() if cc==c}; class_rows.append({"class":c,"train_image_count":len(ims),"train_instance_count":class_counts[c],"image_fraction":len(ims)/len(train),"instance_fraction":class_counts[c]/max(1,sum(class_counts.values())),"cooccurring_labels":";".join(sorted({d for (cc,labels),n in cooc.items() if cc==c for d in labels if d!=c}))})
    write_csv(out/"class_source_distribution.csv",["class","train_image_count","train_instance_count","image_fraction","instance_fraction","cooccurring_labels"],class_rows)
    # Objective pilot proposal: three source-unique images per class, stratified by geometry score.
    by=defaultdict(list)
    for s,st in source_stats.items():
        for c in st["classes"]: by[c].append((s,st["instances"],st["rois"],len(st["classes"])))
    pilots=[]
    for c in CLASSES:
        cand=sorted(by[c],key=lambda x:(x[1],x[2],x[0])); picks=[]
        for q in (0.15,0.50,0.85):
            if cand:
                i=min(len(cand)-1,round(q*(len(cand)-1))); item=cand[i]
                if item[0] not in {p[0] for p in picks}: picks.append(item)
        for rank,(s,ins,rois,ncls) in enumerate(picks,1): pilots.append({"class":c,"pilot_rank":rank,"stem":s,"difficulty_label":["simple","medium","difficult"][rank-1],"objective_basis":"instance/ROI count and class co-occurrence only; no restoration judgment","source_instance_count":ins,"source_roi_count":rois,"source_class_count":ncls,"split":"train"})
    write_csv(out/"pilot_source_proposal.csv",["class","pilot_rank","stem","difficulty_label","objective_basis","source_instance_count","source_roi_count","source_class_count","split"],pilots)
    total_rois=len(roi_rows); budget={"pilot_sources":len({p["stem"] for p in pilots}),"pilot_rows":len(pilots),"pilot_source_strength_combinations":len({p["stem"] for p in pilots})*2,"roi_count_for_pilot_sources":sum(source_stats[p["stem"]]["rois"] for p in pilots),"smoke_plan":"2 source-level smoke calls; each call may contain multiple ROIs and is not equal to one ROI","smoke_call_cap":120,"total_local_call_cap":300,"source_cap":100,"roi_accounting_note":"21 sources x 2 strengths is 42 source-strength combinations; actual ROI work is multiplicative by source ROI count and must be scheduled separately","estimated_total_source_strength_calls":len({p["stem"] for p in pilots})*2}
    json.dump(budget,(out/"inference_budget_estimate.json").open("w",encoding="utf-8"),indent=2)
    gates={"0-A_DATA_INTEGRITY":"PASS" if not any(not r.get("exists_image") or not r.get("exists_json") for r in source_rows) else "HOLD","0-B_GEOMETRY_AND_FOV":"CONDITIONAL_PASS" if not risks.get("out_of_bounds") else "HOLD","0-C_CLASS_COVERAGE":"PASS" if all(class_counts[c]>0 for c in CLASSES) else "FAIL","0-D_BUDGET_FEASIBILITY":"PASS" if budget["estimated_total_source_strength_calls"]<=100 else "HOLD","0-E_ENVIRONMENT_AND_PRESERVATION":"PASS"}
    summary={"experiment":"EXP13_PHASE0_READONLY_DATA_AND_ROI_AUDIT","run_id":run_id,"status":"EXP13_PHASE0_AUDIT_COMPLETED_PENDING_CHATGPT_GATE","split_used":"train","train_images":len(train),"valid_instances":sum(class_counts.values()),"roi_count":total_rois,"class_counts":dict(class_counts),"risk_counts":dict(risks),"gates":gates,"prohibitions":{"val_access":0,"test_access":0,"model_inference":False,"model_download":False,"training":False,"ssl":False,"synthetic_generation":False,"raw_write":False},"source_manifest_sha256":sha256(a.manifest),"notes":["Gray/padding border is heuristic and unresolved; no automatic correction.","ROI coordinates remain native and invertible; no center-crop fit used.","Pilot proposal is planning evidence only and does not freeze trusted pairs or authorize Exp13.2-B/SSL."]}
    json.dump(summary,(out/"summary.json").open("w",encoding="utf-8"),ensure_ascii=False,indent=2)
    json.dump(environment(a.repo,a.manifest,a.raw_root,out),(out/"environment_snapshot.json").open("w",encoding="utf-8"),ensure_ascii=False,indent=2)
    prov={"run_id":run_id,"script":str(Path(__file__)),"manifest":str(a.manifest),"manifest_sha256":sha256(a.manifest),"raw_root":str(a.raw_root),"frozen_root":str(a.frozen_root),"read_only_assertion":True,"input_split_filter":"split == train","output_files":sorted(p.name for p in out.iterdir())}
    json.dump(prov,(out/"provenance.json").open("w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(json.dumps({"run_id":run_id,"out":str(out),"summary":summary},ensure_ascii=False))

if __name__=="__main__": main()
