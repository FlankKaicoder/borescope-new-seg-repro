"""Replay predeclared TRAIN probes on isolated copies of four formal J1 checkpoints."""

import argparse
from pathlib import Path

import torch
from ultralytics.utils.torch_utils import autocast

from tools.exp14.audit_admission import require, sha256_file
from tools.exp15 import ja, jb
from tools.exp15.regions import forward, inspect_regions

EPOCHS = (25, 50, 75, 100)
FIXED_TRAIN_BATCHES = (1, 21)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "release", "weight", "ja-root", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir()
    root = a.run_root
    controlled = jb.load(root / "j1" / "summary.json")
    require(controlled["status"] == "PASS_FORMAL_ARM" and controlled["effective_updates"] == 2100,
            "Gradient diagnosis requires the completed J1 training")
    early = jb.load(a.ja_root / "preflight_r5" / "summary.json")
    release = a.release.resolve(strict=True)
    groups = ja.checked_train(release)
    trainer = ja.JATrainer(release, a.weight.resolve(strict=True), a.output, "j1", groups, controlled["identity"])
    report = []
    torch.set_num_threads(4)
    try:
        trainer.setup()
        require(trainer.initial_native == controlled["initial_native_sha256"], "Gradient replay initialization differs")
        iterator = iter(trainer.new_epoch_loader(trainer.train_loader.dataset, 0))
        kept = {}
        try:
            for i, batch in enumerate(iterator, 1):
                if i in FIXED_TRAIN_BATCHES:
                    require(ja.batch_digest(batch) == early["batch_tensor_sha256"][i - 1],
                            "Fixed TRAIN diagnostic batch differs from the verified replay")
                    kept[i] = batch
                if i == max(FIXED_TRAIN_BATCHES):
                    break
        finally:
            del iterator
        require(set(kept) == set(FIXED_TRAIN_BATCHES), "Missing fixed TRAIN diagnostic batch")
        for epoch in EPOCHS:
            step = epoch * 21
            path = root / "j1" / ("last.pt" if epoch == 100 else f"step_{step:04}.pt")
            sha = sha256_file(path)
            require(sha == controlled["checkpoints"][str(step)]["sha256"], "Diagnostic checkpoint identity differs")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for batch_index, batch in kept.items():
                trainer.model.load_state_dict(saved["state"], strict=True)
                region = inspect_regions(batch, groups)
                before = ja.bn_fingerprint(trainer.model)
                with autocast(trainer.amp, device="cuda"):
                    native, _, auxiliary = forward(trainer.model, trainer.preprocess_batch(dict(batch)), region,
                                                  lambda: ja.bn_fingerprint(trainer.model))
                require(ja.bn_fingerprint(trainer.model) != before, "Expected one private replay BN update")
                probes = jb.gradient_probe(trainer.model, native, auxiliary)
                report.append({"epoch": epoch, "fixed_train_batch": batch_index, "checkpoint_sha256": sha,
                               "backbone": probes["backbone"], "clip_coef": probes["clip_coef"],
                               "native_loss": float(native.detach()), "supcon_loss": float(auxiliary.detach())})
            require(sha256_file(path) == sha, "Gradient diagnostic modified a formal checkpoint")
        require(trainer.audit.native_steps == trainer.audit.optimizer_attempts == 0 and ja.checked_train(release) == groups,
                "Gradient diagnostic changed official state")
        ja.record_json(a.output / "summary.json", {"status": "PASS_ISOLATED_TRAIN_GRADIENT_REPLAY",
                       "epochs": list(EPOCHS), "fixed_batches": list(FIXED_TRAIN_BATCHES), "optimizer_attempts": 0,
                       "lambda_s": jb.WEIGHT, "probes": report})
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
