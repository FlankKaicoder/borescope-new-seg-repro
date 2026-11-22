"""Read-only TRAIN Pair artifact/representation proxies, never an admission gate.

prepare copies the 36 previously frozen pilot cases into a new private packet.
precheck verifies identities without a forward. run executes exactly one paired
FP32 eval forward per case using the frozen seed42 J1 epoch100 master.
"""
import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

CHECKPOINT_SHA = "902fbc1efa4693e2601c162c76fb3736b38036d7eb0473438fe84317bb7895f5"
NATIVE_SHA = "1ffe60884850dd454bb7c5ffdddcd9e5c5d9163f4db5d3b5b40026d10a24c107"
PROTOCOL_SHA = "b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca"
DILATION = 5
DEFINITIONS = {
    "population": "36 previously selected historical D0 pilot cases; 36 independent frozen source groups; not a random sample",
    "model": "Frozen J1 seed42 epoch100 FP32 master including its original projection; eval/inference_mode, no optimizer",
    "input": "Original and historical aligned repair in the same full 640 letterbox; RGB /255; no random augmentation",
    "target": "Historical target Polygon intersected with Valid, shared by both domains",
    "context": "ROI intersect Valid minus 5 native-pixel square dilation of union(all annotated GT polygons, historical request edit support)",
    "edit_support": "Same-frame historical RGBA request mask alpha==0; request support does not prove edits were confined there",
    "pooling": "Nearest-neighbor native /4 binary raster, average /2 fractional P3 weights, same project_regions as Exp15",
    "distance": "1-cosine of paired L2-normalized pooled P3-64 and original learned projection-128",
    "domain_probe": "Leave-one-independent-source-group-out balanced nearest normalized-centroid; two domain examples per group; ties receive 0.5; no fitted model or optimizer",
    "pixels": "Same valid proxy context: RGB absolute error /255, signed mean RGB change, grayscale Laplacian variance ratio, RGB perimeter contrast change",
    "interpretation": "All are descriptive artifact/representation PROXIES, not pure style causal effects, semantic restoration, normal GT, or G1 PASS thresholds",
}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(source, output):
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
    from tools.exp13_cloud.g3.prepare_g3b2a_batch import mask_for_selection

    require(not output.exists(), "New private packet directory required")
    cases = load(source)["cases"]
    require(len(cases) == len({c["group_id"] for c in cases}) == 36, "Expected frozen 36 distinct source groups")
    output.mkdir(parents=True)
    records = []
    for index, c in enumerate(cases):
        dest = output / f"case_{index:02d}"
        dest.mkdir()
        entry = {k: c[k] for k in ("record_id", "source_id", "polygon_id", "review_id", "group_id", "class", "frame_size", "roi", "g1_status", "priority_six")}
        entry["historical_case_id"] = c["history"]["pilot"]["case_id"]
        entry["files"] = {}
        sources = {"original": (c["original_path"], c["original_sha256"]), "aligned": (c["aligned_path"], c["aligned_sha256"]),
                   "valid": (c["valid_mask_path"], c["valid_mask_sha256"]), "edit": (c["edit_mask"]["path"], c["edit_mask"]["sha256"]),
                   "annotation": (c["annotation_path"], c["annotation_sha256"])}
        for kind, (path, expected) in sources.items():
            require(sha(path) == expected, f"Frozen local {kind} identity differs")
            target = dest / (kind + Path(path).suffix.lower())
            shutil.copy2(path, target)
            require(sha(target) == expected, "Packet copy identity differs")
            entry["files"][kind] = {"path": target.relative_to(output).as_posix(), "bytes": target.stat().st_size, "sha256": expected}
        frame = tuple(c["frame_size"])
        original = Image.open(sources["original"][0]); aligned = Image.open(sources["aligned"][0])
        edit = Image.open(sources["edit"][0]); valid = Image.open(sources["valid"][0]).convert("L").point(lambda x: 255 if x > 0 else 0)
        ann = load(sources["annotation"][0])
        require(original.size == aligned.size == valid.size == edit.size == frame, "Diagnostic images/masks have different frames")
        require((ann["imageWidth"], ann["imageHeight"]) == frame and edit.mode == "RGBA", "Original annotation/request mask frame differs")
        require(set(edit.getchannel("A").getdata()) == {0, 255}, "Historical edit alpha semantics are not binary")
        target, _ = mask_for_selection(frame, [{"shape_type":"polygon", "points":c["polygon_coordinates"]}], {0})
        union, _ = mask_for_selection(frame, c["all_annotation_polygons"], set(range(len(c["all_annotation_polygons"]))))
        edit_support = edit.getchannel("A").point(lambda x: 255 if x == 0 else 0)
        blocked = ImageChops.lighter(union, edit_support).filter(ImageFilter.MaxFilter(DILATION * 2 + 1))
        roi = Image.new("L", frame)
        x0,y0,x1,y1 = c["roi"]
        ImageDraw.Draw(roi).rectangle((x0,y0,x1-1,y1-1), fill=255)
        context = ImageChops.multiply(ImageChops.multiply(roi, valid), ImageChops.invert(blocked))
        target = ImageChops.multiply(target, valid)
        inner = target.filter(ImageFilter.MinFilter(DILATION * 2 + 1))
        outer = target.filter(ImageFilter.MaxFilter(DILATION * 2 + 1))
        seam = ImageChops.multiply(ImageChops.subtract(outer, inner), valid)
        for kind, im in (("target_mask", target), ("context_mask", context), ("seam_mask", seam)):
            path = dest / f"{kind}.png"; im.save(path)
            entry["files"][kind] = {"path":path.relative_to(output).as_posix(), "bytes":path.stat().st_size, "sha256":sha(path)}
            entry[kind + "_pixels"] = im.histogram()[255]
        require(entry["target_mask_pixels"] > 0, "No supported target pixels")
        records.append(entry)
    write(output / "manifest.json", {"schema":"exp17_train_readonly_diagnostics_v1", "source_ledger_sha256":sha(source),
          "definitions": DEFINITIONS, "checkpoint_file_sha256":CHECKPOINT_SHA, "native_fp32_state_sha256":NATIVE_SHA,
          "cases": records, "optimizer_updates":0, "human_pass":0, "trusted_pairs":0})
    print(json.dumps({"cases":len(records), "groups":len({c["group_id"] for c in records}), "manifest_sha256":sha(output/"manifest.json"),
                      "packet_bytes":sum(f["bytes"] for c in records for f in c["files"].values()), "empty_proxy_contexts":sum(c["context_mask_pixels"] == 0 for c in records)}))


def verify_files(packet, record):
    observed = {}
    for c in record["cases"]:
        for f in c["files"].values():
            path = (packet/f["path"]).resolve(strict=True)
            require(path.is_relative_to(packet.resolve()) and path.is_file() and not (packet/f["path"]).is_symlink(), "Packet escapes root or is not regular")
            require(path.stat().st_size == f["bytes"] and sha(path) == f["sha256"], "Diagnostic input bytes differ")
            observed[f["path"]] = f["sha256"]
    return observed


def precheck(packet, checkpoint, output):
    import torch, ultralytics, cv2, numpy
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp14.label_plan import CLASSES
    record = load(packet/"manifest.json")
    require(record["definitions"] == DEFINITIONS and len(record["cases"]) == len({c["group_id"] for c in record["cases"]}) == 36, "Frozen diagnostic population or definitions differ")
    require(sha(checkpoint) == CHECKPOINT_SHA, "Frozen J1 checkpoint bytes differ")
    ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
    native = {k:v for k,v in ckpt["state"].items() if not k.startswith("region_projection.")}
    projection = {k for k in ckpt["state"] if k.startswith("region_projection.")}
    require(projection == {f"region_projection.{layer}.{p}" for layer in (0,2) for p in ("weight","bias")}, "Frozen original projection is incomplete")
    require(hash_state(native) == ckpt["native_state_sha256"] == NATIVE_SHA and hash_state(ckpt["state"]) == ckpt["model_state_sha256"], "Frozen native/full master hash differs")
    require(ckpt["arm"] == "j1" and ckpt["effective_updates"] == ckpt["ema_updates"] == 2100 and ckpt["scheduler"]["last_epoch"] == 99
            and ckpt["class_names"] == dict(enumerate(CLASSES)) and ckpt["ema_state_sha256"] != ckpt["model_state_sha256"], "Not accepted epoch100 FP32 master")
    require(torch.cuda.is_available() and ultralytics.__version__ == "8.4.117", "Pinned CUDA/Ultralytics runtime differs")
    source_root = Path(__file__).resolve().parents[2]
    protocol = source_root/"results/exp15_jb/formal_protocol.json"
    require(sha(protocol) == PROTOCOL_SHA, "Frozen training protocol differs")
    runtime={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"ultralytics":ultralytics.__version__,
             "numpy":numpy.__version__,"opencv":cv2.__version__,"gpu":torch.cuda.get_device_name(0)}
    expected = load(protocol)["runtime"]
    require(all(runtime[k] == v for k,v in expected.items() if k in runtime), "Runtime differs from frozen seed42 runtime")
    files = verify_files(packet, record)
    receipt = {"status":"PASS_TRAIN_READONLY_DIAGNOSTIC_PRECHECK", "checkpoint_file_sha256":CHECKPOINT_SHA,"native_fp32_state_sha256":NATIVE_SHA,
               "full_fp32_state_sha256":ckpt["model_state_sha256"],"ema_state_sha256_not_used":ckpt["ema_state_sha256"],"protocol_sha256":PROTOCOL_SHA,
               "packet_manifest_sha256":sha(packet/"manifest.json"),"input_files":len(files),"cases":36,"independent_groups":36,"runtime":runtime,
               "source_sha256":{str(p.relative_to(source_root)):sha(p) for p in (Path(__file__),source_root/"tools/exp15/regions.py")},
               "disk_free_bytes":shutil.disk_usage(packet).free,"definitions":DEFINITIONS,"optimizer_updates":0,"val_test_forwards":0}
    output.mkdir(parents=True,exist_ok=False)
    write(output/"precheck.json",receipt)
    return record,ckpt,receipt


def centroid_probe(pairs):
    """Balanced leave-group-out domain discrimination, no learned parameters."""
    import numpy as np
    x=np.asarray(pairs,dtype=np.float64)
    require(x.ndim==3 and x.shape[1]==2 and len(x)>=3 and np.isfinite(x).all(), "Invalid probe features")
    n=len(x); scores=[]
    for hold in range(n):
        means=np.delete(x,hold,axis=0).mean(axis=0)
        norms=np.linalg.norm(means,axis=1,keepdims=True)
        require((norms>0).all(),"Zero centroid")
        means/=norms
        for domain in range(2):
            sim=means@x[hold,domain]
            scores.append(0.5 if abs(sim[0]-sim[1])<=1e-12 else float(int(np.argmax(sim))==domain))
    return {"balanced_accuracy":float(np.mean(scores)),"correct_weight":float(sum(scores)),"samples":len(scores),"groups":n,"chance":0.5,"tie_credit":0.5}


def run(packet,checkpoint,output,precheck_path):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.tasks import SegmentationModel
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp15.regions import add_projection, project_regions, Regions, P3_LAYER

    require(not output.exists(),"New diagnostic run directory required")
    record=load(packet/"manifest.json"); prep=load(precheck_path)
    require(prep["status"]=="PASS_TRAIN_READONLY_DIAGNOSTIC_PRECHECK" and prep["packet_manifest_sha256"]==sha(packet/"manifest.json")
            and prep["source_sha256"]["tools/exp17/diagnostics.py"]==sha(__file__) and sha(checkpoint)==CHECKPOINT_SHA,"Precheck identity changed")
    files=verify_files(packet,record)
    ckpt=torch.load(checkpoint,weights_only=True,map_location="cpu")
    with torch.random.fork_rng(devices=[]):
        model=SegmentationModel(ckpt["model_config"],ch=3,nc=7,verbose=False); add_projection(model)
    model.load_state_dict(ckpt["state"],strict=True)
    require(hash_state(model.state_dict())==prep["full_fp32_state_sha256"],"Strict original J1 reload changed state")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=model.cuda().float().eval().requires_grad_(False)
    before=hash_state(model.state_dict())
    bn_before=hash_state({k:v for k,v in model.named_buffers() if k.endswith(("running_mean","running_var","num_batches_tracked"))})
    output.mkdir(parents=True)
    write(output/"started.json",{"checkpoint_sha256":CHECKPOINT_SHA,"precheck_sha256":sha(precheck_path),"cases":36,"optimizer_updates":0})
    results=[]; feature_pairs={"target_raw64":[],"target_projection128":[],"context_raw64":[],"context_projection128":[]}
    groups={k:[] for k in feature_pairs}; start=time.time(); torch.cuda.reset_peak_memory_stats()
    transform=LetterBox(new_shape=(640,640),auto=False,scale_fill=False,scaleup=True,center=True)
    for c in record["cases"]:
        paths={k:packet/f["path"] for k,f in c["files"].items()}
        imgs=[cv2.imread(str(paths[k]),cv2.IMREAD_COLOR) for k in ("original","aligned")]
        require(all(im is not None for im in imgs) and imgs[0].shape==imgs[1].shape,"Unreadable/mismatched paired image")
        h,w=imgs[0].shape[:2]; r=min(640/h,640/w); nw,nh=round(w*r),round(h*r)
        left=round((640-nw)/2-0.1); top=round((640-nh)/2-0.1)
        masks=[cv2.imread(str(paths[k]),cv2.IMREAD_GRAYSCALE)>0 for k in ("target_mask","context_mask")]
        labels=np.zeros((160,160),np.uint8)
        for i,m in enumerate(masks,1):
            canvas=np.zeros((640,640),np.uint8)
            canvas[top:top+nh,left:left+nw]=cv2.resize(m.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)
            low=cv2.resize(canvas,(160,160),interpolation=cv2.INTER_NEAREST)
            require(not ((labels>0)&(low>0)).any(),"Target/context overlap after shared raster")
            labels[low>0]=i
        raster=np.stack([labels,labels]); cells=np.array([int((labels==i).sum()) for _ in range(2) for i in (1,2)])
        regions=Regions(slots=np.array([0,0,1,1]),ordinals=np.array([0,1,0,1]),classes=np.array([0,0,0,0]),groups=np.array([0,0,0,0]),cells=cells,foreground_fraction=np.ones(4),max_instances=2)
        tensors=np.stack([transform(image=im)[:,:,::-1].transpose(2,0,1) for im in imgs]).copy()
        tensor=torch.from_numpy(tensors).cuda().float()/255
        captured=[]; hook=model.model[P3_LAYER].register_forward_hook(lambda _m,_i,out:captured.append(out))
        try:
            with torch.inference_mode():
                model(tensor)
                require(len(captured)==1 and tuple(captured[0].shape)==(2,64,80,80) and torch.isfinite(captured[0]).all(),"Nonfinite/missing frozen P3")
                raster_t=torch.from_numpy(raster).cuda()
                raw=project_regions(captured[0],raster_t,regions,torch.nn.Identity()).cpu().numpy()
                projected=project_regions(captured[0],raster_t,regions,model.region_projection).cpu().numpy()
        finally:
            hook.remove()
        require(np.isfinite(raw).all() and np.isfinite(projected).all(),"Nonfinite representation")
        kept=[(slot,kind) for slot in range(2) for kind in range(2) if cells[slot*2+kind]>0]
        reps={"raw64":{key:raw[i] for i,key in enumerate(kept)},"projection128":{key:projected[i] for i,key in enumerate(kept)}}
        row={k:c[k] for k in ("record_id","group_id","class","historical_case_id","g1_status","priority_six")}
        row["pooled"]={}; row["shared_raster_pixels"]={"target":int((labels==1).sum()),"context":int((labels==2).sum())}
        for region,name in enumerate(("target","context")):
            for space,d in reps.items():
                if (0,region) not in d:
                    row["pooled"][f"{name}_{space}"]={"status":"MISSING_SHARED_NATIVE_RASTER"};continue
                pair=[d[(0,region)],d[(1,region)]]
                require(all(np.linalg.norm(v)>0 for v in pair),"Zero pooled vector")
                key=f"{name}_{space}"; feature_pairs[key].append(pair); groups[key].append(c["group_id"])
                row["pooled"][key]={"cosine_distance":float(1-np.dot(*pair)),"original":pair[0].tolist(),"repair":pair[1].tolist()}
        a,n=[im[:,:,::-1].astype(np.float64)/255 for im in imgs]; context=masks[1]
        pixels={"context_pixels":int(context.sum()),"scope":"PROXY_NOT_NORMAL_GT"}
        if context.any():
            pixels.update({"outside_context_mae_rgb_0_1":float(np.abs(n-a)[context].mean()),"signed_mean_rgb_change":(n-a)[context].mean(axis=0).tolist()})
            laps=[cv2.Laplacian(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float64)/255,cv2.CV_64F) for im in imgs]
            va,vn=[float(v[context].var()) for v in laps]
            pixels.update({"original_laplacian_variance":va,"repair_laplacian_variance":vn,"repair_over_original_laplacian_variance":vn/max(va,1e-12)})
        seam=cv2.imread(str(paths["seam_mask"]),cv2.IMREAD_GRAYSCALE)>0
        pixels["target_perimeter_change_mae_0_1"]=float(np.abs(n-a)[seam].mean()) if seam.any() else None
        row["pixels"]=pixels; results.append(row)
        write(output/f"case_{len(results)-1:02d}.json",row)
    after=hash_state(model.state_dict()); bn_after=hash_state({k:v for k,v in model.named_buffers() if k.endswith(("running_mean","running_var","num_batches_tracked"))})
    require(after==before==prep["full_fp32_state_sha256"] and bn_after==bn_before and sha(checkpoint)==CHECKPOINT_SHA,"Frozen model/BN/checkpoint mutated")
    require(verify_files(packet,record)==files,"Diagnostic inputs changed")
    require(all(p.grad is None for p in model.parameters()),"Unexpected gradient")
    summaries={}
    for key,pairs in feature_pairs.items():
        require(len(groups[key])==len(set(groups[key])),"Independent group leakage")
        distances=[1-float(np.dot(*p)) for p in pairs]
        summaries[key]={"paired_cosine_distance":{"n":len(distances),"mean":float(np.mean(distances)),"median":float(np.median(distances)),"min":min(distances),"max":max(distances)},
                        "domain_probe":centroid_probe(pairs) if len(pairs)>=3 else {"status":"INSUFFICIENT_SHARED_SUPPORT"}}
    for metric in ("outside_context_mae_rgb_0_1","repair_over_original_laplacian_variance","target_perimeter_change_mae_0_1"):
        values=[r["pixels"][metric] for r in results if r["pixels"].get(metric) is not None]
        summaries[metric]={"n":len(values),"mean":float(np.mean(values)),"median":float(np.median(values)),"min":min(values),"max":max(values)}
    summary={"status":"PASS_TRAIN_READONLY_DIAGNOSTICS_NOT_PAIR_ADMISSION","cases":len(results),"independent_groups":len({r["group_id"] for r in results}),
             "forward_calls":len(results),"images_forwarded":len(results)*2,"optimizer_updates":0,"backward_calls":0,"val_test_forwards":0,"human_pass":0,"trusted_pairs":0,
             "model_full_state_sha256_before":before,"model_full_state_sha256_after":after,"native_fp32_state_sha256":NATIVE_SHA,
             "checkpoint_file_sha256_before":CHECKPOINT_SHA,"checkpoint_file_sha256_after":sha(checkpoint),"bn_sha256_before":bn_before,"bn_sha256_after":bn_after,
             "definitions":DEFINITIONS,"precheck_sha256":sha(precheck_path),"packet_manifest_sha256":sha(packet/"manifest.json"),"source_sha256":prep["source_sha256"],"runtime":prep["runtime"],
             "summaries":summaries,"elapsed_seconds":time.time()-start,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"tf32":False,"fp32":True,
             "input_sha256_postchecked":True,"limitations":["Historical hand-selected population, no population-level estimate.","Content/registration and generation effects confound domain separation; this is not pure style identification.","Context excludes all annotated GT and requested edit support but may contain unannotated defects or edits outside request support.","No thresholds, filtering, semantic approvals, or trusted Pairs are produced."]}
    write(output/"summary.json",summary)
    print(json.dumps(summary,sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="mode",required=True)
    prepare_p=sub.add_parser("prepare");prepare_p.add_argument("--source",type=Path,required=True);prepare_p.add_argument("--output",type=Path,required=True)
    for name in ("precheck","run"):
        s=sub.add_parser(name);s.add_argument("--packet",type=Path,required=True);s.add_argument("--checkpoint",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
        if name=="run":s.add_argument("--precheck",type=Path,required=True)
    args=p.parse_args()
    if args.mode=="prepare":prepare(args.source,args.output)
    elif args.mode=="precheck":precheck(args.packet,args.checkpoint,args.output)
    else:run(args.packet,args.checkpoint,args.output,args.precheck)


if __name__=="__main__":main()
