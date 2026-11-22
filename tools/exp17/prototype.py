"""At most three synthetic-only J2 updates; no real-data or formal training entry point."""

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import ModelEMA, autocast

from tools.exp14.a11_full_numerics import capture_rng_state, restore_rng_state
from tools.exp14.a11_full_protocol import source_sha256
from tools.exp14.a11_smoke_numerics import NumericalAudit, exact_tree, finite_tree, optimizer_group_names
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp15 import regions as j1
from tools.exp17 import pair_loss

CAP = 3
SCHEMA = "exp17_j2_synthetic_only_v1"
PAIR_WEIGHT = 1.0  # Engineering unit coefficient. Formal lambda_p remains unset.
SIZE, BATCH = 128, 14


def write(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def code_identity():
    paths = [Path(__file__), Path(pair_loss.__file__), Path(j1.__file__)]
    return {p.name: source_sha256(p) for p in paths}


def reserve(budget):
    with (budget / "synthetic.attempts").open("a+", encoding="ascii") as stream:
        stream.seek(0)
        prior = stream.read().splitlines()
        require(prior == [str(n) for n in range(1, len(prior)+1)] and len(prior) < CAP,
                "Synthetic cumulative optimizer-attempt cap exhausted or malformed")
        stream.write(f"{len(prior)+1}\n")
        stream.flush()
        os.fsync(stream.fileno())
        return len(prior)+1


def synthetic_batch(device):
    generator = torch.Generator().manual_seed(17001)
    images = torch.rand(BATCH, 3, SIZE, SIZE, generator=generator)
    masks = torch.zeros(BATCH, SIZE//4, SIZE//4, dtype=torch.int64)
    masks[:, SIZE//16:3*SIZE//16, SIZE//16:3*SIZE//16] = 1
    classes = np.arange(BATCH) % 7
    semantic = masks * torch.as_tensor(classes[:,None,None])
    # All coordinates, pixels and source labels are artificial, never read from TRAIN assets.
    batch = {"img": images.to(device), "masks": masks.to(device), "sem_masks": semantic.to(device),
             "cls": torch.as_tensor(classes[:,None], dtype=torch.float32, device=device),
             "batch_idx": torch.arange(BATCH, dtype=torch.float32, device=device)}
    groups = np.array([f"synthetic-group-{i}" for i in range(BATCH)])
    # Native stride4 raster occupies [8,24) of32: source pixels[32,96), center.5/size.5.
    # The mask and boxes are explicitly cross-checked below, preventing an accidental coordinate mismatch.
    y,x = np.nonzero(masks[0].numpy())
    box = [(x.min()+x.max()+1)/(2*(SIZE//4)),(y.min()+y.max()+1)/(2*(SIZE//4)),
           (x.max()-x.min()+1)/(SIZE//4),(y.max()-y.min()+1)/(SIZE//4)]
    batch["bboxes"] = torch.tensor([box]*BATCH,dtype=torch.float32,device=device)
    cells = len(np.unique((y//2)*(SIZE//8)+x//2))
    regions = j1.Regions(np.arange(BATCH),np.zeros(BATCH,dtype=np.int64),classes,groups,
                         np.full(BATCH,cells),np.ones(BATCH),1)
    normal = images.clone()
    # Synthetic counterfactual edit at exactly the synthetic Polygon; outside pixels stay equal.
    full_mask = masks.repeat_interleave(4,1).repeat_interleave(4,2).bool()
    normal = torch.where(full_mask[:,None],torch.full_like(normal,.5),normal)
    bindings = []
    for i in range(BATCH):
        names = [pair_loss.digest(["synthetic",i,key]) for key in ("A","N","H","valid","trace")]
        bindings.append(pair_loss.PairBinding(i,i,str(groups[i]),f"synthetic-polygon-{i}",int(classes[i]),
            "synthetic-v1",*names,a_tensor_sha256=pair_loss.tensor_identity(images[i]),
            n_tensor_sha256=pair_loss.tensor_identity(normal[i]),synthetic=True,
            support_complete=True,contamination_clear=True,unambiguous_region=True))
    return batch,regions,normal.to(device),bindings


def native_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith("region_projection.")}


def backbone(model):
    return {n:p for n,p in model.named_parameters() if p.requires_grad and n.startswith("model.") and int(n.split('.')[1])<=10}


def head(model):
    return {n:p for n,p in model.named_parameters() if n.startswith("region_projection.")}


def bn_counters(model):
    return {n:int(v) for n,v in model.named_buffers() if n.endswith("num_batches_tracked")}


def norm(vectors):
    return sum(float(v.float().square().sum()) for v in vectors if v is not None)**.5


def checkpoint(trainer,path):
    model=trainer.model
    record={"schema":SCHEMA,"synthetic_only":True,"formal_training_allowed":False,
        "source_identity":code_identity(),"model_config":model.yaml,"class_names":dict(enumerate(CLASSES)),
        "updates":trainer.audit.effective_updates,"state":cpu_state(model.state_dict()),
        "ema":cpu_state(trainer.ema.ema.state_dict()),"ema_updates":trainer.ema.updates,
        "optimizer":cpu_state(trainer.optimizer.state_dict()),"optimizer_names":optimizer_group_names(model,trainer.optimizer),
        "scaler":trainer.scaler.state_dict(),"native_sha256":hash_state(native_state(model))}
    with path.open("xb") as stream:torch.save(record,stream)
    saved=torch.load(path,map_location="cpu",weights_only=True)
    require(exact_tree(saved,record) and finite_tree(saved),"Synthetic checkpoint serialization differs")
    with torch.random.fork_rng(devices=[]):
        reloaded=SegmentationModel(saved["model_config"],nc=7,verbose=False)
        j1.add_projection(reloaded)
    for key in ("ema","state"):
        reloaded.load_state_dict(saved[key],strict=True)
        require(exact_tree(dict(reloaded.state_dict()),saved[key]),"Strict synthetic model/EMA reload differs")
    params=dict(reloaded.named_parameters())
    optimizer=torch.optim.AdamW([{"params":[params[n] for n in names]} for names in saved["optimizer_names"]])
    optimizer.load_state_dict(saved["optimizer"])
    scaler=torch.amp.GradScaler("cpu",init_scale=1.0);scaler.load_state_dict(saved["scaler"])
    require(exact_tree(optimizer.state_dict(),saved["optimizer"]) and scaler.state_dict()==saved["scaler"]
            and saved["ema_updates"]==saved["updates"] and saved["scaler"]["_growth_tracker"]==saved["updates"],
            "Strict synthetic AdamW, EMA or GradScaler reload differs")
    counts=Counter(int(v["step"]) for v in saved["optimizer"]["state"].values())
    require(set(counts)=={saved["updates"]},"Synthetic AdamW step counts differ")
    del reloaded.region_projection
    require(hash_state(reloaded.state_dict())==saved["native_sha256"],"Head stripping changed native state")
    reloaded.names=saved["class_names"];reloaded.eval()
    with torch.no_grad():prediction=reloaded(torch.zeros(1,3,64,64))
    require(finite_tree(prediction),"Stripped native synthetic prediction is nonfinite")
    return {"file_sha256":sha256_file(path),"native_sha256":saved["native_sha256"],
            "model_state_sha256":hash_state(saved["state"]),"ema_sha256":hash_state(saved["ema"]),
            "strict_reload":True,"stripped_native_synthetic_prediction":True,"optimizer_slots":sum(counts.values())}


def run(args):
    import fcntl
    require(torch.cuda.is_available(),"CUDA synthetic Smoke requires the audited GPU environment")
    require(ultralytics.__version__=="8.4.117" and 1<=args.steps<=CAP,"Pinned runtime or bounded step count differs")
    out=args.output.resolve();out.mkdir()
    budget=args.budget.resolve();budget.mkdir(exist_ok=True)
    require(not (budget/"synthetic.attempts").exists(),"A previous synthetic run started; no automatic restart")
    trainer=None
    try:
        with (budget/"lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            model=SegmentationModel("yolo11n-seg.yaml",nc=7,verbose=False)
            j1.add_projection(model);model.to("cuda").train();model.args=get_cfg();model.names=dict(enumerate(CLASSES))
            for name,param in model.named_parameters():
                if name.endswith("dfl.conv.weight"):param.requires_grad_(False)
            optimizer=BaseTrainer.build_optimizer(SimpleNamespace(),model,name="AdamW",lr=.001,momentum=.937,decay=.0005)
            scaler=torch.amp.GradScaler("cuda",init_scale=1.0)
            trainer=SimpleNamespace(model=model,optimizer=optimizer,scaler=scaler,ema=ModelEMA(model))
            trainer.audit=NumericalAudit(trainer,update_limit=args.steps)
            batch,regions,normal,bindings=synthetic_batch("cuda")
            initial={k:v.detach().clone() for k,v in model.state_dict().items()};rng=capture_rng_state()
            # Counterfactual identity is a route/BN test, not a semantic normal Pair.
            before=bn_counters(model)
            with autocast(True,device="cuda"):
                identity_bindings=[pair_loss.PairBinding(**{**vars(b),"n_tensor_sha256":b.a_tensor_sha256}) for b in bindings]
                result=pair_loss.forward_joint(model,batch,regions,batch["img"],identity_bindings)
            a_feature,n_feature,a,n=result[-1]
            require(torch.equal(a_feature,n_feature) and torch.equal(a,n),"N=A acquired a branch/mode feature shortcut")
            require(all(bn_counters(model)[k]==v+1 for k,v in before.items()),"N=A route added a second BN update")
            del result,a_feature,n_feature,a,n
            model.load_state_dict(initial,strict=True);restore_rng_state(rng)
            require(hash_state(model.state_dict())==hash_state(initial),"Identity probe did not restore model/BN")
            original_backbone,original_head=hash_state(backbone(model)),hash_state(head(model))
            trace=[];torch.cuda.reset_peak_memory_stats();started=time.monotonic();gradient_proof=None
            for step in range(1,args.steps+1):
                optimizer.zero_grad(set_to_none=True);before=bn_counters(model)
                with autocast(True,device="cuda"):
                    native,parts,supcon,pair,coverage,features=pair_loss.forward_joint(model,batch,regions,normal,bindings)
                    base=native+pair_loss.SUPCON_WEIGHT*supcon
                    total=base+PAIR_WEIGHT*pair
                    evidence=trainer.audit.check_forward(total,parts)
                require(all(bn_counters(model)[k]==v+1 for k,v in before.items()),"Auxiliary N polluted native BN counters")
                if step==1:
                    names,params=zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad])
                    gp=torch.autograd.grad(pair*1024,params,retain_graph=True,allow_unused=True)
                    gp=[None if x is None else x.float()/1024 for x in gp]
                    gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
                    gj=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
                    gn=torch.autograd.grad(pair*1024,features[1],retain_graph=True)[0].float()/1024
                    bb=[i for i,n in enumerate(names) if n in backbone(model)]
                    change=[gj[i]-gb[i] for i in bb if gj[i] is not None and gb[i] is not None]
                    gradient_proof={"pair_backbone_norm":norm([gp[i] for i in bb]),
                        "pair_backbone_nonzero_tensors":sum(gp[i] is not None and bool(gp[i].count_nonzero()) for i in bb),
                        "N_P3_gradient_norm":float(gn.norm()),"actual_joint_minus_J1_backbone_norm":norm(change),
                        "diagnostic_loss_scale_cancelled":1024,"pair_grad_finite":finite_tree(gp)}
                    require(gradient_proof["pair_backbone_norm"]>0 and gradient_proof["N_P3_gradient_norm"]>0
                            and gradient_proof["actual_joint_minus_J1_backbone_norm"]>0 and finite_tree((gp,gb,gj,gn)),
                            "Pair objective did not affect the shared Backbone/N route or yielded nonfinite derivatives")
                    del gp,gb,gj,gn
                scaler.scale(total).backward()
                require(all(p.grad is not None and bool(p.grad.count_nonzero()) for p in head(model).values()),
                        "Shared projection gradient absent")
                ticket=reserve(budget);numeric=trainer.audit.optimizer_step()
                require(trainer.audit.native_steps==trainer.audit.effective_updates==trainer.ema.updates==step,
                        "Expected exactly one joint optimizer/EMA update")
                row={"step":step,"ticket":ticket,"native_loss":float(native.detach()),"supcon_loss":float(supcon.detach()),
                    "pair_loss":float(pair.detach()),"total_loss":evidence["loss"],"pair_coverage":coverage,"numeric":numeric}
                with (out/"updates.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
                trace.append(row)
                del native,parts,supcon,pair,base,total,features
            require(hash_state(backbone(model))!=original_backbone and hash_state(head(model))!=original_head,
                    "Shared Backbone or projection did not update")
            saved=checkpoint(trainer,out/"synthetic_last.pt")
            write(out/"summary.json",{"status":"PASS_EXP17_SYNTHETIC_J2_SMOKE","schema":SCHEMA,
                "identity":code_identity(),"initialization":"RANDOM_NATIVE_ARCHITECTURE_SYNTHETIC_ENGINEERING_ONLY",
                "device":torch.cuda.get_device_name(0),"python":platform.python_version(),"torch":torch.__version__,
                "ultralytics":ultralytics.__version__,"synthetic_images":BATCH,"synthetic_resolution":SIZE,
                "optimizer_attempts":trainer.audit.optimizer_attempts,"effective_updates":trainer.audit.effective_updates,
                "native_optimizer_calls":trainer.audit.native_steps,"amp_skips":trainer.audit.amp_skips,
                "attempt_cap":CAP,"lambda_s":8.0,"temperature":.1,"lambda_pair_synthetic":PAIR_WEIGHT,
                "formal_lambda_pair":None,"margin_candidate":pair_loss.MARGIN,"real_pair_updates":0,"formal_updates":0,
                "val_inference":0,"test_inference":0,"trusted_real_pairs":0,"formal_training_authorized":False,
                "identical_A_N_features_and_embeddings":True,"native_BN_updates_per_joint_step":1,
                "extra_N_running_stats_updates":0,"gradient_proof":gradient_proof,"checkpoint":saved,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "elapsed_seconds":time.monotonic()-started,"trace":trace})
    except Exception as failure:
        (out/"failure.log").write_text(traceback.format_exc(),encoding="utf-8")
        audit=getattr(trainer,"audit",None)
        write(out/"failed.json",{"status":"EXP17_SYNTHETIC_HARD_GATE","optimizer_attempts":getattr(audit,"optimizer_attempts",0),
            "effective_updates":getattr(audit,"effective_updates",0),"error":str(failure),
            "numerical_evidence":getattr(failure,"evidence",{})})
        raise
    finally:
        if trainer is not None:trainer.audit.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--budget",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=CAP)
    run(parser.parse_args())


if __name__=="__main__":main()
