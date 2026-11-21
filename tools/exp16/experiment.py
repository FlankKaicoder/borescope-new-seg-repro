"""TRAIN-only preparation and sequential, explicitly bound Exp16 J3 execution."""

import argparse
import gc
import json
import math
import shutil
import statistics
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a11_full_protocol import source_sha256
from tools.exp14.a11_smoke_numerics import finite_tree
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.formal_e0_e1 import hash_state
from tools.exp15 import audit_jc, ja, jb, je0
from tools.exp15.jb_binding import check_pool_tags, install_format_audit
from tools.exp15.regions import inspect_regions
from tools.exp16 import ce, train

BASE = Path(__file__).resolve().parents[2]
PARENT = "401210427a61f3430f18bb4fe93261c911f66080"
JE0_SHA = "1e114398c58a87337a1addbc4392b4485e651f928e95ca9d8c189af52972a305"
SEEDS = (42, 142, 242)
PROBES = (1, 7, 11, 21, 32, 42, 56, 70)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj):
    je0.write(Path(p), obj)


def sources():
    files = [*sorted((BASE / "tools/exp16").glob("*.py"))]
    files += [BASE / p for p in load(BASE / "results/exp15_je0_multiseed/formal_protocol.json")["source_sha256_lf"]]
    return {p.relative_to(BASE).as_posix(): source_sha256(p) for p in files}


def environment(config):
    oldpath = BASE / "results/exp15_je0_multiseed/formal_protocol.json"
    require(sha256_file(oldpath) == JE0_SHA, "Frozen Exp15 protocol changed")
    old = load(oldpath)
    for name, expected in old["source_sha256_lf"].items():
        require(source_sha256(BASE / name) == expected, "Inherited audited source changed")
    release, weight = Path(config["release"]), Path(config["weight"])
    groups, runtime, framework = je0.verify_inputs(release, weight, old)
    return {"release": release, "weight": weight, "groups": groups, "old": old,
            "runtime": runtime, "framework": framework}


def baseline(config, seed):
    folder = Path(config["baselines"][str(seed)])
    pubs = load(BASE / "results/exp15_je1_formal_seed142_242/summary.json")
    result, traces = {}, {}
    for arm in ("j0", "j1"):
        public = load(BASE / f"results/exp15_jc_formal_seed42/{arm}_summary.json") if seed == 42 else next(
            x["audit"] for x in pubs["arms"] if x["seed"] == seed and x["arm"] == arm)
        for name, key in (("summary.json", "summary_sha256"), ("events.jsonl", "events_sha256")):
            require(sha256_file(folder / arm / name) == public[key], "Historical evidence byte identity differs")
        result[arm] = load(folder / arm / "summary.json")
        traces[arm] = audit_jc.by_event(folder / arm / "events.jsonl")
        require(result[arm]["effective_updates"] == 2100 and result[arm]["amp_skips"] == 0,
                "Historical arm incomplete")
    audit_jc.compare_batches(traces["j0"]["batch_read"], traces["j1"]["batch_read"], 2100)
    a, b = result["j0"], result["j1"]
    require(a["initial_native_sha256"] == b["initial_native_sha256"] and a["native_groups"] == b["native_groups"],
            "Historical initial state/groups differ")
    first = torch.load(folder / "j0/step_0001.pt", map_location="cpu", weights_only=True)
    # Published initial BN is independently reproduced from a fresh setup below.
    return {"native_initial_sha256": a["initial_native_sha256"], "native_groups": a["native_groups"],
            "batch_tensor_sha256": [x["batch_sha256"] for x in a["updates"]],
            "lr": [x["lr"] for x in a["updates"]],
            "per_batch_regions": [x["coverage"] for x in traces["j1"]["jb_update"]],
            "baseline_summaries_sha256": {arm: sha256_file(folder / arm / "summary.json") for arm in result},
            "coco_sha256": first["weight_transfer"]["source_weight_sha256"]}


def use_trainer(env, seed, output, identity, action):
    output.mkdir()
    trainer = None
    try:
        trainer = train.make_trainer(env["release"], env["weight"], output, env["groups"], identity, seed)
        return action(trainer)
    except Exception:
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if trainer:
            trainer.close()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def gradients(model, native, auxiliary):
    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    y = torch.autograd.grad(native, params, retain_graph=True, allow_unused=True)
    s = torch.autograd.grad(auxiliary, params, allow_unused=True)
    require(finite_tree((y, s)), "Calibration nonfinite gradients")
    def stats(indexes):
        n, a = (jb.grad_norm([g[i] for i in indexes]) for g in (y, s))
        dot = jb.dot([y[i] for i in indexes], [s[i] for i in indexes])
        return {"native_norm": n, "auxiliary_norm": a, "ratio": a / n if n else None,
                "cosine": dot / (n * a) if n and a else None, "dot": dot,
                "nonzero_aux_tensors": sum(s[i] is not None and bool(s[i].count_nonzero()) for i in indexes)}
    back = [i for i,n in enumerate(names) if n.startswith("model.") and int(jb.native_layer(n)) <= 10]
    result = {"backbone": stats(back), "global": stats(list(range(len(names)))),
              "layers": {layer: stats([i for i,n in enumerate(names) if jb.native_layer(n) == layer])
                         for layer in dict.fromkeys(map(jb.native_layer,names))}}
    require(result["backbone"]["nonzero_aux_tensors"] == 120
            and result["layers"]["projection"]["nonzero_aux_tensors"] == 6
            and not any(row["nonzero_aux_tensors"] for layer,row in result["layers"].items()
                        if layer != "projection" and int(layer) > 16),
            "CE auxiliary gradient endpoint mismatch")
    return result


def weighted(row, weight):
    n, s, d = row["native_norm"], row["auxiliary_norm"], row["dot"]
    joint = math.sqrt(max(0, n*n + weight*weight*s*s + 2*weight*d))
    return {**row, "weighted_ratio": weight*s/n if n else None,
            "native_joint_cosine": (n*n+weight*d)/(n*joint) if n and joint else None,
            "joint_norm": joint, "clip_native": min(1,10/(n+1e-6)), "clip_joint": min(1,10/(joint+1e-6))}


def calibrate(trainer, prior, config):
    trainer.setup()
    require(trainer.initial_native == prior["native_initial_sha256"], "CE calibration native init mismatch")
    initial = {k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    head = {k:v for k,v in initial.items() if k.startswith(ce.HEAD_PREFIX)}
    ja_public = load(BASE / "results/exp15_ja/summary.json")
    _, previous = jb.read_reference(Path(config["ja_root"]), ja_public)
    snapshots = {"coco_init": initial, **{f"ja_{arm}_step70_fresh_ce_head":
                  {**{k:v for k,v in state.items() if not k.startswith(ce.HEAD_PREFIX)}, **head}
                  for arm,state in previous.items()}}
    install_format_audit(trainer.train_loader.dataset)
    private = trainer.output / "private_binding"
    private.mkdir()
    counts, featured, probes = Counter(), set(), []
    disabled_equivalence = None
    for step, batch in jb.matched_batches(trainer, 70):
        require(ja.batch_digest(batch) == prior["batch_tensor_sha256"][step-1], "TRAIN calibration trajectory mismatch")
        region = inspect_regions(batch, trainer.groups)
        require(region.report() == prior["per_batch_regions"][step-1], "CE/J1 region eligibility differs")
        counts.update(check_pool_tags(batch, trainer.groups, private, featured))
        if step == 1:
            disabled_equivalence = ja.probe_disabled(trainer,batch)
            trainer.model.load_state_dict(initial,strict=True)
            require(ja.bn_fingerprint(trainer.model) == trainer.initial_bn,
                    "Native equivalence diagnostic did not restore initial BN")
        if step not in PROBES:
            continue
        payload = trainer.preprocess_batch({k:v for k,v in batch.items() if k != "jb_binding"})
        for label, state in snapshots.items():
            trainer.model.load_state_dict(state, strict=True)
            with autocast(trainer.amp, device="cuda"):
                y, parts, aux = ce.forward(trainer.model, payload, region, lambda: ja.bn_fingerprint(trainer.model))
            require(finite_tree((y,parts,aux)), "Calibration nonfinite loss")
            probes.append({"batch":step, "snapshot":label, "native_loss":float(y.detach()),
                           "ce_loss":float(aux.detach()), "coverage":region.report(), **gradients(trainer.model,y,aux)})
    medians = {label:statistics.median(p["backbone"]["ratio"] for p in probes if p["snapshot"]==label) for label in snapshots}
    raw = min(.05/max(medians.values()), .19/max(p["backbone"]["ratio"] for p in probes))
    quantum = 10**(math.floor(math.log10(raw))-1)
    weight = math.floor(raw/quantum)*quantum
    for p in probes:
        p["backbone"] = weighted(p["backbone"],weight)
        p["global"] = weighted(p["global"],weight)
        p["layers"] = {k:weighted(v,weight) for k,v in p["layers"].items()}
        p["joint_loss"] = p["native_loss"]+weight*p["ce_loss"]
    require(all(.01 <= weight*x <= .10 for x in medians.values())
            and min(p["backbone"]["native_joint_cosine"] for p in probes) >= .98,
            "Conservative TRAIN gradient strength screen did not pass")
    require(featured == {"high_overlap","small_polygon","thin_crack","mixed_class"}
            and trainer.audit.optimizer_attempts == 0, "Binding coverage or zero-update calibration failed")
    return {"status":"PASS_TRAIN_CE_CALIBRATION", "lambda_ce":weight, "optimizer_attempts":0,
            "disabled_equivalence":disabled_equivalence,
            "joint_gradient_computation":"Exact linear-combination identity reconstructed from independent native and CE gradients; no third backward",
            "selection":"two_significant_digit_floor(min(0.05/max_snapshot_median,0.19/max_probe_ratio))",
            "snapshot_raw_medians":medians, "probes":probes, "binding":dict(counts), "featured":sorted(featured),
            "initial_bn_sha256":trainer.initial_bn, "initial_native_sha256":trainer.initial_native,
            "limitation":"Early frozen native snapshots with a fresh CE head; later trained auxiliary strength may differ"}


def prepare(env, config, root):
    root.mkdir()
    priors = {}
    for seed in SEEDS:
        prior = baseline(config,seed)
        def replay(t):
            t.setup()
            require(t.initial_native == prior["native_initial_sha256"] and
                    [[n for n in g if not n.startswith(ce.HEAD_PREFIX)] for g in t.optimizer_names] == prior["native_groups"],
                    "Fresh CE initialization/optimizer groups do not match J0")
            for step,batch in jb.matched_batches(t,70):
                require(ja.batch_digest(batch)==prior["batch_tensor_sha256"][step-1]
                        and inspect_regions(batch,t.groups).report()==prior["per_batch_regions"][step-1],
                        "Zero-update CE augmentation/region replay mismatch")
            require(t.audit.optimizer_attempts==0,"Replay performed an update")
            return t.initial_bn
        prior["initial_bn_sha256"] = use_trainer(env,seed,root/f"replay_{seed}",{"seed":seed,"phase":"replay"},replay)
        priors[str(seed)] = prior
    calibration = use_trainer(env,42,root/"calibration",{"seed":42,"phase":"calibration"},
                               lambda t:calibrate(t,priors["42"],config))
    write(root/"calibration.json",calibration)
    smokes = {}
    for seed in SEEDS:
        budget = root/f"smoke_budget_{seed}"
        budget.mkdir()
        result = use_trainer(env,seed,root/f"smoke_{seed}",{"seed":seed,"phase":"smoke"},
                             lambda t:train.run_arm(t,priors[str(seed)],priors[str(seed)],budget,calibration["lambda_ce"]))
        write(root/f"smoke_{seed}"/"summary.json",result)
        smokes[str(seed)] = result
    estimate = sum((root/f"smoke_{s}/step_0001.pt").stat().st_size*21 for s in SEEDS)
    require(shutil.disk_usage(root).free >= estimate+2*1024**3,"Insufficient three-arm capacity plus 2GiB")
    write(root/"preflight.json",{"status":"PASS_EXP16_PREFLIGHT","runtime":env["runtime"],
        "framework":env["framework"],"code_source_sha256":sources(),"priors":priors,
        "calibration_sha256":sha256_file(root/"calibration.json"),"lambda_ce":calibration["lambda_ce"],
        "smoke":{s:{k:r[k] for k in ("optimizer_attempts","effective_updates","amp_skips","checkpoints")} for s,r in smokes.items()},
        "estimated_three_arm_checkpoint_bytes":estimate,"free_bytes":shutil.disk_usage(root).free,
        "formal_updates":0,"val_inference":0,"test_inference":0})


def formal(env,config,root,protocol_path,preflight_path,authorization,seed):
    import fcntl
    from tools.exp16.audit import audit_arm
    spec, pre = load(protocol_path), load(preflight_path)
    require(pre["status"]=="PASS_EXP16_PREFLIGHT","The bound preflight did not pass")
    require(spec["code_source_sha256"] == pre["code_source_sha256"] == sources(), "Frozen Exp16 code changed")
    require(spec["preflight_sha256"] == sha256_file(preflight_path) and spec["lambda_ce"]==pre["lambda_ce"],"Frozen prep/weight changed")
    require(load(authorization)=={"stage":"EXP16_J3_THREE_SEED_FORMAL_AUTHORIZATION","decision":"PASS",
            "protocol_sha256":sha256_file(protocol_path),"preflight_sha256":sha256_file(preflight_path),"seeds":list(SEEDS)},"Formal Gate mismatch")
    require(seed in SEEDS,"Unexpected J3 seed")
    root.mkdir(exist_ok=True)
    with (root/"formal.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for previous in SEEDS[:SEEDS.index(seed)]:
            accepted = load(root/str(previous)/"accepted.json")
            require(accepted["status"]=="PASS_EXP16_J3_FORMAL_ARM"
                    and accepted["protocol_sha256"]==sha256_file(protocol_path),
                    "Previous arm not accepted under the current frozen protocol")
        remaining = len(SEEDS)-SEEDS.index(seed)
        require(shutil.disk_usage(root).free >= remaining*pre["estimated_three_arm_checkpoint_bytes"]//3+2*1024**3,"Remaining disk budget insufficient")
        folder=root/str(seed)
        folder.mkdir()
        budget=folder/"formal_budget"
        budget.mkdir()
        identity={"seed":seed,"exp16_protocol_sha256":sha256_file(protocol_path),"coco_sha256":sha256_file(env["weight"])}
        status=1
        try:
            result=use_trainer(env,seed,folder/"j3",identity,
                lambda t:train.run_arm(t,pre["priors"][str(seed)],pre["priors"][str(seed)],budget,spec["lambda_ce"],formal=True))
            require(je0.verify_inputs(env["release"],env["weight"],env["old"])[0]==env["groups"],"Post-training TRAIN identity changed")
            write(folder/"j3/summary.json",result)
            status=0
        finally:
            (folder/"j3.launch.exit").write_text(f"{datetime.now().astimezone().isoformat()}\t{status}\n",encoding="utf-8")
        acceptance,rows=audit_arm(folder,seed,Path(config["baselines"][str(seed)]),spec,identity)
        acceptance.update(protocol_sha256=sha256_file(protocol_path),matched_native_batches=2100)
        import csv
        with (folder/"matched_batch_ledger.csv").open("x",encoding="utf-8",newline="") as out:
            writer=csv.writer(out,lineterminator="\n")
            writer.writerow(("step","epoch","images","raw_targets","augmented_targets",
                             "ordered_sources_sha256","augmented_tensors_sha256",
                             "lr_no_decay","lr_decay","lr_bias"))
            writer.writerows(rows)
        acceptance["matched_batch_ledger_sha256"]=sha256_file(folder/"matched_batch_ledger.csv")
        write(folder/"accepted.json",acceptance)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","formal"))
    for name in ("config","output","protocol","preflight","authorization"):
        p.add_argument("--"+name,type=Path,required=name in ("config","output"))
    p.add_argument("--seed",type=int)
    a=p.parse_args()
    torch.set_num_threads(4)
    config=load(a.config)
    env=environment(config)
    if a.phase=="prepare":prepare(env,config,a.output)
    else:formal(env,config,a.output,a.protocol,a.preflight,a.authorization,a.seed)


if __name__=="__main__":
    main()
