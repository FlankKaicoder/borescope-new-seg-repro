"""Prepare new Exp15 paired seeds using the frozen J-B training loop."""

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from tools.exp14.audit_admission import require, sha256_file

NEW_SEEDS = (142, 242)
ARMS = ("j0", "j1")
FORMAL_STAGE = "EXP15_JE0_MULTISEED_FORMAL_AUTHORIZATION"
READY = "EXP15_JE0_MULTISEED_READY_FOR_FORMAL_GATE"
HOLD = "EXP15_JE0_HOLD_WITH_BLOCKERS"
BASE_HEAD = "bca1d48f2a3004fe843db4a259f52e3ee360d0ff"
JB_PROTOCOL_SHA = "b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca"
BASE = Path(__file__).resolve().parents[2]
JB_PROTOCOL = BASE / "results/exp15_jb/formal_protocol.json"
JD_SUMMARY = BASE / "results/exp15_jd_val_seed42/summary.json"
JC_SUMMARY = BASE / "results/exp15_jc_formal_seed42/pair_summary.json"
PROBE_STEPS, SMOKE_STEPS, SMOKE_CAP = 70, 1, 5


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_seed(seed):
    require(type(seed) is int and seed in NEW_SEEDS, "Only seed142 and seed242 are new formal seeds")
    return seed


def seeded_copy(spec, seed):
    return {**spec, "seed": validate_seed(seed)}


def gate_record(protocol_sha, preflight_sha):
    return {"stage": FORMAL_STAGE, "decision": "PASS", "protocol_sha256": protocol_sha,
            "preflight_sha256": preflight_sha, "authorized_seeds": list(NEW_SEEDS), "authorized_arms": list(ARMS)}


def validate_gate(approval, protocol_sha, preflight_sha):
    require(approval == gate_record(protocol_sha, preflight_sha),
            "Separate authorization must bind the exact JE0 protocol and preflight")


def protocol_spec():
    from tools.exp14.a11_full_protocol import source_sha256

    require(sha256_file(JB_PROTOCOL) == JB_PROTOCOL_SHA, "Audited J-B protocol changed")
    old, jd = load(JB_PROTOCOL), load(JD_SUMMARY)
    require(jd["status"] == "EXP15_JD_FROZEN_VAL_COMPLETE_PENDING_RESULT_GATE" and jd["seed"] == 42,
            "Audited seed42 result identity differs")
    code = {name: source_sha256(BASE / name) for name in (
        "tools/exp15/je0.py", "tools/exp15/ja.py", "tools/exp15/jb.py", "tools/exp15/regions.py",
        "tools/exp15/audit_jc.py", "tools/exp14/a11_smoke_train.py", "tools/exp14/a11_full_train.py",
        "tools/exp14/a11_smoke_numerics.py", "tools/exp14/a11_full_numerics.py", "tools/exp14/a11_full_eval.py")}
    require(all(code[f"tools/exp15/{name}"] == old["identity"]["files"][name] for name in ("ja.py", "jb.py", "regions.py")),
            "Audited J-A/J-B training sources changed")
    train = {k: v for k, v in old["train"].items() if k != "seed"}
    return {"schema": "exp15_je0_multiseed_v1", "audited_base_head": BASE_HEAD,
            "authorization": "PENDING_INDEPENDENT_FORMAL_GATE", "source_sha256_lf": code,
            "jb_protocol_sha256": JB_PROTOCOL_SHA, "jd_seed42_summary_sha256": sha256_file(JD_SUMMARY),
            "jc_seed42_pair_sha256": sha256_file(JC_SUMMARY), "seed42_role": "ALREADY_COMPLETED_DISCOVERY_PAIR",
            "seed42_primary": jd["primary_metric"], "seeds": [42, *NEW_SEEDS], "new_seeds": list(NEW_SEEDS),
            "run_order": [[seed, arm] for seed in NEW_SEEDS for arm in ARMS],
            "train_common": train, "identity": old["identity"], "runtime": old["runtime"],
            "arms": {"j0": {"lambda_s": 0.0}, "j1": {"lambda_s": 8.0, "temperature": 0.10}},
            "budget": {"original_train_images": 668, "raw_instances": 1266, "batches_per_epoch": 21,
                       "epochs": 100, "effective_updates": 2100, "warmup_iterations": 63,
                       "original_image_exposures_per_arm": 66800, "raw_instance_exposures_per_arm": 126600},
            "checkpoint": old["checkpoint"], "evaluation": {**old["evaluation"], "batch": 32,
                           "val_images": 154, "val_instances": 296, "precision": "CUDA_FP32_TF32_DISABLED",
                           "val_files_sha256": jd["preflight"]["val_files_sha256"],
                           "passes_per_pair": ["j0_standard", "j0_fixed", "j1_standard", "j1_fixed"],
                           "independent_val_gate_required": True},
            "reporting": {"primary": "Mask mAP50-95", "individual_delta": "unrounded_J1_minus_J0",
                          "aggregate": "unweighted_mean_and_sample_SD_of_three_paired_deltas_ddof_1",
                          "positive_rule": "strictly_greater_than_zero",
                          "by_positive_count": {"3": "stronger_preliminary_stability", "2": "mixed_evidence",
                                                "1": "no_stable_gain_claim", "0": "no_stable_gain_claim"},
                          "incomplete_pair": "HOLD_NO_THREE_SEED_CONCLUSION",
                          "secondary": ["Mask Precision and Recall", "fixed Mask/Box FP/FN",
                                        "per-class Mask AP", "size Recall", "Tears AP50-95", "Dent Tiny Mask Recall",
                                        "Crack fixed Mask FP", "Burn/corrosion AP and fixed counts"]},
            "preflight": {"replay_first_batches": PROBE_STEPS, "additional_replay_epoch": 100,
                          "smoke_attempts_per_seed_arm": SMOKE_STEPS, "hard_smoke_cap_per_seed_arm": SMOKE_CAP},
            "hard_stop": old["hard_gate"]}


def read_protocol(path):
    obtained = load(path)
    require(obtained == protocol_spec(), "Frozen multiseed protocol or source identity differs")
    return obtained


def make_trainer(release, weight, output, arm, groups, identity, seed):
    from tools.exp14 import a11_smoke_train
    from tools.exp15 import ja

    base_training_protocol = a11_smoke_train.training_protocol
    base_formal_spec = ja.protocol_spec

    def seed_formal():
        cfg = base_formal_spec()
        return {**cfg, "training": seeded_copy(cfg["training"], seed)}

    # Scope the seed to both inherited constructor configurations, before BaseTrainer seeds the model.
    with patch.object(ja, "protocol_spec", seed_formal), patch.object(
        a11_smoke_train, "training_protocol", lambda *a: seeded_copy(base_training_protocol(*a), seed)
    ):
        trainer = ja.JATrainer(release, weight, output, arm, groups, identity)
    require(trainer.args.seed == trainer.formal["seed"] == seed, "Native constructor and epoch loader seeds differ")
    return trainer


def native_groups(trainer):
    from tools.exp15 import ja
    return [[n for n in names if not n.startswith(ja.HEAD_PREFIX)] for names in trainer.optimizer_names]


def verify_inputs(release, weight, spec):
    import platform
    import torch
    import torchvision
    import ultralytics

    from tools.exp15 import ja

    runtime = {"gpu": torch.cuda.get_device_name(0), "cuda_devices": torch.cuda.device_count(),
               "python": platform.python_version(), "cuda": torch.version.cuda, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__}
    require(runtime == spec["runtime"] and sha256_file(weight) == spec["identity"]["coco_sha256"],
            "Runtime or COCO weight differs")
    root = Path(ultralytics.__file__).resolve().parent.parent
    jc = load(JC_SUMMARY)
    pinned = jc["input_preflight"]["framework_source_sha256"]
    require(all(sha256_file(root / name) == sha for name, sha in pinned.items()), "Pinned framework source differs")
    return ja.checked_train(release), runtime, pinned


def state_identity(spec, protocol_sha, seed):
    return {**spec["identity"], "exp15_multiseed_protocol_sha256": protocol_sha, "seed": seed,
            "archived_seed42_result_sha256": spec["jd_seed42_summary_sha256"]}


def trace(trainer):
    import torch
    from tools.exp14.a11_full_train import check_epoch_counts
    from tools.exp15 import ja, jb
    from tools.exp15.regions import inspect_regions

    def record(batch, include_lr):
        item = {"sources_sha256": digest(batch["im_file"]),
                "tensor_sha256": ja.batch_digest(batch), "images": len(batch["im_file"])}
        if include_lr:
            item["lr"] = [g["lr"] for g in trainer.optimizer.param_groups]
        return item, inspect_regions(batch, trainer.groups).report() if trainer.variant == "j1" else None

    trainer.setup()
    require(trainer.amp and trainer.scaler.is_enabled() and len(trainer.train_loader) == 21
            and trainer._get_warmup_iterations(21) == 63 and trainer.test_loader is None,
            "Sealed startup, AMP or TRAIN scope differs")
    with torch.no_grad():
        early, regions, late = [], [], []
        for _, batch in jb.matched_batches(trainer, PROBE_STEPS):
            item, coverage = record(batch, True)
            early.append(item)
            if coverage is not None:
                regions.append(coverage)
        trainer.epoch = 99
        counters = trainer.counters.copy()
        seen = set()
        for batch in trainer.new_epoch_loader(trainer.train_loader.dataset, 99):
            paths = trainer.audit_batch(batch)
            require(not seen.intersection(paths), "Repeated image in the final-epoch replay")
            seen.update(paths)
            item, _ = record(batch, False)
            late.append(item)
        check_epoch_counts(trainer.counters - counters, trainer.expected_arm, len(late),
                           late[-1]["images"], trainer.expected.keys(), seen)
    require(trainer.audit.native_steps == trainer.audit.optimizer_attempts == 0, "Replay consumed an optimization step")
    return {"native_initial_sha256": trainer.initial_native, "initial_bn_sha256": trainer.initial_bn,
            "native_groups_sha256": digest(native_groups(trainer)), "transfer": trainer.weight_transfer,
            "early": early, "late": late, "regions": regions}


def with_trainer(env, seed, arm, output, action):
    import gc
    import torch

    output.mkdir()
    trainer = None
    try:
        trainer = make_trainer(env["release"], env["weight"], output, arm, env["groups"],
                               state_identity(env["spec"], env["protocol_sha256"], seed), seed)
        return action(trainer)
    except Exception:
        with (output / "failure.log").open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        raise
    finally:
        if trainer:
            trainer.close()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def preflight(env, root):
    import fcntl
    from tools.exp15 import jb

    root.mkdir()
    probe_dir, smoke_dir = root / "replay", root / "smoke"
    probe_dir.mkdir()
    smoke_dir.mkdir()
    ready = {}
    for seed in NEW_SEEDS:
        folder = probe_dir / str(seed)
        folder.mkdir()
        traces = {arm: with_trainer(env, seed, arm, folder / arm, trace) for arm in ARMS}
        a, b = map(traces.get, ARMS)
        shared = ("native_initial_sha256", "initial_bn_sha256", "native_groups_sha256", "early", "late")
        require(all(a[k] == b[k] for k in shared), f"Within-seed native J0/J1 identity differs for {seed}")
        require(a["transfer"]["source_weight_sha256"] == b["transfer"]["source_weight_sha256"]
                == env["spec"]["identity"]["coco_sha256"], "Paired COCO transfer differs")
        require(len(b["regions"]) == PROBE_STEPS and all(r["eligible_anchors"] > 0 for r in b["regions"]),
                "Replay produced a batch without active SupCon anchors")
        ready[str(seed)] = {**{k: b[k] for k in shared},
                             "batch_tensor_sha256": [r["tensor_sha256"] for r in b["early"]],
                             "per_batch_regions": b["regions"], "transfer": {arm: t["transfer"] for arm, t in traces.items()}}
    a, b = [ready[str(seed)] for seed in NEW_SEEDS]
    require(a["native_initial_sha256"] != b["native_initial_sha256"]
            and a["early"][0]["sources_sha256"] != b["early"][0]["sources_sha256"]
            and a["late"][0]["sources_sha256"] != b["late"][0]["sources_sha256"],
            "New seeds did not vary native initialization and sampled image order")
    smoke = {}
    for seed in NEW_SEEDS:
        prior = ready[str(seed)]
        folder = smoke_dir / str(seed)
        folder.mkdir()
        budget = folder / "budget"
        budget.mkdir()
        with (budget / "lock").open("x") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for arm in ARMS:
                # Reuse the J-B loop exactly, including its joint loss and checkpoint reload.
                with patch.object(jb, "SMOKE_STEPS", SMOKE_STEPS), patch.object(jb, "SMOKE_CAP", SMOKE_CAP):
                    receipt = with_trainer(env, seed, arm, folder / arm,
                                           lambda t: jb.run_arm(t, prior, prior, budget, formal=False))
                record = {**receipt, "post_warmup_steps": 0}
                write(folder / arm / "summary.json", record)
                require(record["effective_updates"] == record["optimizer_attempts"] == record["native_optimizer_calls"]
                        == SMOKE_STEPS and record["amp_skips"] == 0, "Bounded seed smoke did not pass")
                last = record["checkpoints"]["1"]
                require(last["strict_reload"] and last["stripped_native_forward"], "Strict checkpoint replay missing")
                smoke[f"{seed}/{arm}"] = {**{k: record[k] for k in ("optimizer_attempts", "effective_updates",
                                          "amp_skips", "peak_allocated_bytes", "peak_reserved_bytes",
                                          "lambda_s")},
                                          "checkpoint": last, "checkpoint_bytes": (folder / arm / "step_0001.pt").stat().st_size,
                                          "gradient_norm": record["updates"][0]["global_gradient_norm"],
                                          "native_loss": record["updates"][0]["native_loss"],
                                          "supcon_loss": record["updates"][0]["supcon_loss"]}
    require(verify_inputs(env["release"], env["weight"], env["spec"])[0] == env["groups"], "Original TRAIN changed")
    expected_checkpoint_bytes = sum(s["checkpoint_bytes"] * 21 for s in smoke.values())
    free = shutil.disk_usage(root).free
    require(free >= expected_checkpoint_bytes + 1024**3, "Insufficient measured disk for four formal arms and audit headroom")
    write(root / "preflight.json", {"status": READY, "protocol_sha256": env["protocol_sha256"],
          "runtime": env["runtime"], "framework_sha256": env["framework"], "train_images": len(env["groups"]),
          "train_groups": len(set(env["groups"].values())), "train_images_labels_sha_verified": 1336,
          "seeds": ready, "smoke": smoke, "replay_optimizer_attempts": 0,
          "free_disk_bytes_after_smoke": free, "estimated_four_arm_checkpoint_bytes": expected_checkpoint_bytes,
          "formal_updates": 0, "val_inference": 0, "test_inference": 0})


def formal(env, root, seed, arm, evidence, approval):
    import fcntl
    from tools.exp15 import jb

    validate_seed(seed)
    require(arm in ARMS, "Unexpected formal arm")
    expect = load(evidence)
    require(expect["status"] == READY and expect["protocol_sha256"] == env["protocol_sha256"]
            and expect["runtime"] == env["runtime"] and expect["framework_sha256"] == env["framework"],
            "Formal run requires the exact approved JE0 preflight")
    validate_gate(load(approval), env["protocol_sha256"], sha256_file(evidence))
    root.mkdir(exist_ok=True)
    with (root / "formal.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = [(s, ar) for s in NEW_SEEDS for ar in ARMS].index((seed, arm))
        for s, ar in [(s, ar) for s in NEW_SEEDS for ar in ARMS][:index]:
            require(load(root / str(s) / f"{ar}_accepted.json")["status"] == "PASS_JE0_FORMAL_ARM",
                    "An earlier arm in the preregistered sequence has not passed")
        folder = root / str(seed)
        folder.mkdir(exist_ok=True)
        budget = folder / "formal_budget"
        budget.mkdir(exist_ok=True)
        require(not (budget / f"{arm}.attempts").exists(), "Formal arm already started")
        require(shutil.disk_usage(root).free >= expect["smoke"][f"{seed}/{arm}"]["checkpoint_bytes"] * 21 + 1024**3,
                "Insufficient disk for the arm and its audit")
        phase_start = datetime.now().astimezone().isoformat()
        output = folder / arm
        prior = expect["seeds"][str(seed)]
        status = 1
        started = time.monotonic()
        try:
            receipt = with_trainer(env, seed, arm, output, lambda t: jb.run_arm(t, prior, prior, budget, formal=True))
            require(verify_inputs(env["release"], env["weight"], env["spec"])[0] == env["groups"], "Original TRAIN changed")
            receipt["elapsed_seconds"] = time.monotonic() - started
            write(output / "summary.json", receipt)
            status = 0
        finally:
            if output.is_dir():
                with (folder / f"{arm}.launch.exit").open("x", encoding="utf-8") as stream:
                    stream.write(f"{datetime.now().astimezone().isoformat()}\t{status}\n")
        verify_formal_arm(env, folder, seed, arm, prior, phase_start)


def verify_formal_arm(env, folder, seed, arm, prior, phase_start):
    from tools.exp15 import audit_jc

    training = seeded_copy(env["spec"]["train_common"], seed)
    p = {"train": training, "lambda_s": {"j0": 0.0, "j1": 8.0}}
    identity = {"identity": state_identity(env["spec"], env["protocol_sha256"], seed),
                "runtime": env["runtime"], "native_initial_sha256": prior["native_initial_sha256"]}
    inv = load(env["release"] / "configs/train_entry_files.json")["baseline"]
    expected = {r["image_path"]: r["instance_count"] for r in inv}
    trace = audit_jc.by_event(folder / arm / "events.jsonl")
    require(trace["setup"][0]["config"]["seed"] == seed, "Formal optimizer path was not initialized with the declared seed")
    result = audit_jc.check_arm(folder, arm, p, identity, expected)
    paired = 0
    if arm == "j1":
        left = audit_jc.check_arm(folder, "j0", p, identity, expected)
        audit_jc.compare_batches(left[0], result[0], 2100)
        require(left[2] == result[2], "Full paired augment or scheduling trace differs")
        groups = [load(folder / v / "summary.json")["native_groups"] for v in ARMS]
        require(groups[0] == groups[1] and digest(groups[0]) == prior["native_groups_sha256"],
                "Full paired optimizer groups differ")
        paired = 2100
    write(folder / f"{arm}_accepted.json", {"status": "PASS_JE0_FORMAL_ARM", "seed": seed,
          "arm": arm, "formal_started": phase_start, "protocol_sha256": env["protocol_sha256"],
          "matched_native_batches": paired, "audit": result[3]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("protocol", "preflight", "formal"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("protocol", "release", "weight", "preflight", "apply-gate"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    a = parser.parse_args()
    if a.phase == "protocol":
        write(a.output, protocol_spec())
        return
    require(a.protocol and a.release and a.weight, "Frozen protocol, TRAIN release and COCO weight are required")
    a.output = a.output.resolve()
    release, weight = a.release.resolve(strict=True), a.weight.resolve(strict=True)
    require(a.output.parent == release.parent.parent and not a.output.is_relative_to(release.parent),
            "An independent AutoDL experiment directory is required")
    if a.phase == "formal":
        require(a.preflight and a.apply_gate, "Formal updates require separate explicit approval")
        validate_seed(a.seed)
        require(a.arm in ARMS, "A declared arm is required")
    spec = read_protocol(a.protocol)
    groups, runtime, framework = verify_inputs(release, weight, spec)
    env = {"spec": spec, "release": release, "weight": weight, "groups": groups, "runtime": runtime,
           "framework": framework, "protocol_sha256": sha256_file(a.protocol)}
    import torch
    torch.set_num_threads(4)
    if a.phase == "preflight":
        preflight(env, a.output)
    else:
        formal(env, a.output, a.seed, a.arm, a.preflight, a.apply_gate)


if __name__ == "__main__":
    main()
