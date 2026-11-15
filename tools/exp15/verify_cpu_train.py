"""Independently exercise one authentic TRAIN batch on CPU without any optimizer update."""

import argparse
import json
from pathlib import Path

import torch

from tools.exp14.audit_admission import require
from tools.exp15.ja import (
    JATrainer, batch_digest, bn_fingerprint, checked_train, code_identity, probe_gradients, record_json,
)
from tools.exp15.regions import forward, inspect_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "weight", "preflight", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    output = args.output.resolve()
    require(not output.is_relative_to(release.parent), "Use an independent experiment directory")
    output.mkdir()
    torch.set_num_threads(8)
    groups = checked_train(release)
    gate = json.loads(args.preflight.read_text())
    require(gate["status"] == "PASS_TRAIN_PREFLIGHT" and gate["identity"]["code"] == code_identity(),
            "CPU replay requires the completed preflight")
    trainer = JATrainer(release, args.weight, output, "j1", groups, gate["identity"])
    try:
        trainer.setup()
        require(trainer.initial_native == gate["native_initial_state_sha256"], "Native initialization differs")
        iterator = iter(trainer.new_epoch_loader(trainer.train_loader.dataset, 0))
        try:
            batch = next(iterator)
        finally:
            del iterator
        require(batch_digest(batch) == gate["batch_tensor_sha256"][0], "Real CPU replay batch differs")
        region = inspect_regions(batch, groups)
        model = trainer.model.cpu()
        model.criterion = None
        batch["img"] = batch["img"].float().div_(255)
        before = bn_fingerprint(model)
        native, _, supcon = forward(model, batch, region, lambda: bn_fingerprint(model))
        gradients = probe_gradients(model, native, supcon)
        require(bn_fingerprint(model) != before and trainer.audit.native_steps == 0, "CPU BN or update count differs")
        raw_groups = [set() for _ in range(7)]
        for path, (classes, _) in trainer.expected.items():
            for class_id in classes:
                raw_groups[class_id].add(groups[str(path)])
        report = {"status": "PASS_REAL_TRAIN_CPU_FORWARD_BACKWARD", "batch_index": 1, "images": len(batch["img"]),
                  "native_loss": float(native.detach()), "supcon_loss": float(supcon.detach()),
                  "same_augmented_tensor_sha256": True, "native_only_bn_update": True, "effective_updates": 0,
                  "native_train_source_groups": len(set(groups.values())),
                  "raw_source_groups_by_class": list(map(len, raw_groups)), "gradient_probe": gradients}
        record_json(output / "summary.json", report)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

