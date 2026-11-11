"""Compare AMP and FP32 on a few reconstructed TRAIN batches at the A4 stop state."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion
from ultralytics.models.yolo.segment.train import SegmentationTrainer

from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.diagnose_amp_first_batch import (
    NeverUpdate, configure_fp32, parameters_sha, restore_rng, rng_state, tensor_digest,
)
from tools.exp14.foreground_audit import HEADS, STRIDES, ForegroundAudit
from tools.exp14.formal_e0_e1 import INITIAL_STATE_SHA, FormalTrainer, hash_state
from tools.exp14.revised_amp_smoke import gradient_population
from tools.exp14.smoke_trainer import TaggedBatchSampler, finite, write_json


STOP_SHA = "a8eee8628a1ec66f8205f6f1babd88d64ad996c17c770eb970bcc142fdd019c8"
POINTS = (("e0", 42, 22), ("e0", 43, 1), ("e0", 43, 2), ("e0", 4, 19),
          ("e0", 1, 1), ("e1", 2, 13))
LOSS_RTOL = 0.10
MIN_COSINE = 0.90


def reconstruct(trainer, epoch, batch):
    seed = 42 + epoch * 22 + batch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = trainer.train_loader
    loader.frozen.set_epoch(epoch)
    loader.dataset.epoch = epoch
    requests = list(TaggedBatchSampler(loader.frozen))[batch - 1]
    raw = loader.dataset.collate([loader.dataset[request] for request in requests])
    slots = [row["slot_id"] for row in raw["_audit"]]
    require(slots == trainer.plan[epoch - 1]["batches"][batch - 1], "Diagnostic slots differ")
    return SegmentationTrainer.preprocess_batch(trainer, raw), seed, slots


def gradient_vector(module):
    params = [p for p in module.parameters() if p.requires_grad]
    return torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                      for p in params])


def compare_gradients(amp, fp32):
    require(amp.shape == fp32.shape, "Gradient vector shapes differ")
    mask = (amp == 0) & (fp32 != 0)
    norm_a, norm_f = float(amp.norm()), float(fp32.norm())
    return {"elements": amp.numel(), "amp_nonzero": int(amp.count_nonzero()),
            "fp32_nonzero": int(fp32.count_nonzero()), "amp_norm": norm_a, "fp32_norm": norm_f,
            "amp_zero_fp32_nonzero": int(mask.sum()),
            "amp_zero_fp32_norm_fraction": float(fp32[mask].norm()) / norm_f if norm_f else None,
            "cosine": float(torch.nn.functional.cosine_similarity(amp, fp32, dim=0, eps=1e-30))
            if norm_a and norm_f else None}


def backward(model, checkpoint, observer, batch, random_state, amp):
    restore_rng(random_state)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    observer.counts = None
    with torch.autocast("cuda", enabled=amp):
        loss, parts = model.loss(batch)
    loss = loss.sum()
    require(finite(loss) and finite(parts), "Diagnostic loss became non-finite")
    if amp:
        scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(NeverUpdate(model.parameters()))
    else:
        loss.backward()
    population = gradient_population(model.parameters())
    require(not population["nonfinite"], "Diagnostic backward became non-finite")
    head = model.model[-1]
    vectors = {f"{name}.{i}": gradient_vector(branch)
               for name in HEADS for i, branch in enumerate(getattr(head, name))}
    vectors["all"] = gradient_vector(model)
    row = {"loss": float(loss.detach()), "components": {key: float(value) for key, value in parts.items()},
           "foreground": observer.read(), "gradient_population": population,
           "gradient_norm": float(vectors["all"].norm()), "amp_scale": 1 if amp else None}
    require(row["gradient_norm"] > 0 and parameters_sha(model) == checkpoint["parameter_sha256"],
            "Diagnostic has zero global gradient or changed parameters")
    return row, vectors


def assess(rows):
    absent_coarse = positive_coarse = 0
    problems = []
    for row in rows:
        point = f'{row["arm"]}:{row["epoch"]}/{row["batch"]}'
        a, f = row["amp"], row["fp32"]
        if abs(a["loss"] - f["loss"]) > LOSS_RTOL * abs(f["loss"]):
            problems.append(f"{point} loss divergence")
        for i, stride in enumerate(map(str, STRIDES)):
            na, nf = a["foreground"][stride], f["foreground"][stride]
            absent_coarse += i == 2 and na == nf == 0
            positive_coarse += i == 2 and na > 0 and nf > 0
            for name in ("cv2", "cv4"):
                branch = f"{name}.{i}"
                g = row["gradients"][branch]
                if (na > 0) != (g["amp_nonzero"] > 0) or (nf > 0) != (g["fp32_nonzero"] > 0):
                    problems.append(f"{point} {branch} foreground and gradient disagree")
                if na and nf and (g["cosine"] is None or g["cosine"] < MIN_COSINE):
                    problems.append(f"{point} {branch} gradient divergence")
        if row["gradients"]["all"]["cosine"] < MIN_COSINE:
            problems.append(f"{point} global gradient divergence")
    if not absent_coarse or not positive_coarse:
        problems.append("Missing paired stride-32 foreground-positive or foreground-empty coverage")
    return {"decision": "HOLD" if problems else "PASS_CONDITIONAL_RESTART",
            "stride32_zero_both": absent_coarse, "stride32_positive_both": positive_coarse,
            "problems": problems}


def run(args):
    require(sha256_file(args.checkpoint) == STOP_SHA, "Historical checkpoint SHA256 differs")
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(not checkpoint["complete"] and checkpoint["batches"] == checkpoint["effective_updates"] == 926
            and checkpoint["model_state_sha256"] == hash_state(checkpoint["model_state"]),
            "Historical stop identity differs")
    rows = []
    for arm in ("e0", "e1"):
        folder = args.output / f"loader_{arm}"
        folder.mkdir()
        trainer = FormalTrainer(args.payload, args.weight, folder, arm)
        trainer.expected_init_sha = INITIAL_STATE_SHA
        trainer._setup_train()
        configure_fp32()
        model = trainer.model
        observer = ForegroundAudit(model)
        for _, epoch, batch in (p for p in POINTS if p[0] == arm):
            inputs, seed, slots = reconstruct(trainer, epoch, batch)
            state = rng_state()
            hashes = {k: tensor_digest(v) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            a, av = backward(model, checkpoint, observer, inputs, state, True)
            f, fv = backward(model, checkpoint, observer, inputs, state, False)
            require(hashes == {k: tensor_digest(inputs[k]) for k in hashes}, "Batch tensors mutated")
            comparison = {k: compare_gradients(av[k], fv[k]) for k in av}
            row = {"arm": arm, "epoch": epoch, "batch": batch, "seed": seed, "slots": slots,
                   "batch_hashes": hashes, "amp": a, "fp32": f, "gradients": comparison}
            rows.append(row)
            write_json(args.output / f"{arm}_{epoch}_{batch}.json", row)
        del trainer, model, observer
    result = {**assess(rows), "historical_replay": False, "optimizer_updates": 0,
              "thresholds": {"loss_rtol": LOSS_RTOL, "gradient_cosine_min": MIN_COSINE},
              "checkpoint_sha256": STOP_SHA, "batches": rows}
    write_json(args.output / "diagnosis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}))
    if result["decision"] == "HOLD":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("payload", "weight", "output", "checkpoint"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    run(parser.parse_args())
