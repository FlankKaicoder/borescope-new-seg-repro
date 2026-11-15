"""Independently reconcile JE0 replay and one-step smoke evidence."""

import argparse
import math
from pathlib import Path

import torch

from tools.exp14.audit_admission import require, sha256_file
from tools.exp15 import audit_jc, je0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "protocol", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    source = a.run / "preflight.json"
    preflight = je0.load(source)
    protocol = je0.read_protocol(a.protocol)
    require(preflight["status"] == je0.READY and preflight["protocol_sha256"] == sha256_file(a.protocol)
            and preflight["formal_updates"] == preflight["val_inference"] == preflight["test_inference"]
            == preflight["replay_optimizer_attempts"] == 0, "Preflight scope or protocol identity differs")
    summaries = {}
    for seed in je0.NEW_SEEDS:
        key = str(seed)
        paths = [a.run / "replay" / key / arm / "events.jsonl" for arm in je0.ARMS]
        paired = [audit_jc.by_event(path) for path in paths]
        expected = preflight["seeds"][key]
        count = je0.PROBE_STEPS + 21
        for events in paired:
            require(events["setup"][0]["config"]["seed"] == seed and len(events["batch_read"]) == count
                    and set(events) == {"setup", "batch_read"}, "Replay seed, event scope or count differs")
        audit_jc.compare_batches(paired[0]["batch_read"], paired[1]["batch_read"], count)
        require(all(row["epoch"] == 100 for row in paired[0]["batch_read"][-21:])
                and expected["late"][-1]["images"] == 28, "Final replay lacks the last epoch or its tail")
        max_lr_error = max(abs(x - y) for i, row in enumerate(expected["early"], 1)
                           for x, y in zip(row["lr"], audit_jc.expected_lr(i, protocol["train_common"]), strict=True))
        require(max_lr_error <= 1e-14, "Replayed warmup or post-warmup schedule differs")
        proofs = {}
        for arm in je0.ARMS:
            root = a.run / "smoke" / key
            path = root / arm / "step_0001.pt"
            receipt = je0.load(root / arm / "summary.json")
            frozen = preflight["smoke"][f"{seed}/{arm}"]
            require((root / "budget" / f"{arm}.attempts").read_text().splitlines() == ["1"]
                    and receipt["initial_native_sha256"] == expected["native_initial_sha256"]
                    and je0.digest(receipt["native_groups"]) == expected["native_groups_sha256"]
                    and sha256_file(path) == frozen["checkpoint"]["sha256"]
                    and receipt["attempt_cap"] == 5 and receipt["effective_updates"] == 1
                    and receipt["native_optimizer_calls"] == receipt["optimizer_attempts"] == 1 and receipt["amp_skips"] == 0,
                    "Smoke independent budget or native identity differs")
            record = torch.load(path, map_location="cpu", weights_only=True)
            require(record["identity"] == je0.state_identity(protocol, preflight["protocol_sha256"], seed)
                    and record["arm"] == arm and record["effective_updates"] == record["ema_updates"] == 1
                    and record["scaler"]["scale"] == 1.0 and record["scaler"]["_growth_tracker"] == 1,
                    "Checkpoint bound seed, arm, optimizer count or scaler differs")
            step = receipt["updates"][0]
            require(step["batch_sha256"] == expected["batch_tensor_sha256"][0]
                    and math.isclose(step["total_loss"], step["native_loss"] + receipt["lambda_s"] * (step["supcon_loss"] or 0),
                                     rel_tol=3e-7, abs_tol=2e-5)
                    and step["projection_nonzero_gradients"] == (4 if arm == "j1" else 0),
                    "One-step joint loss, projection or data audit differs")
            proofs[arm] = frozen
        require(proofs["j0"]["native_loss"] == proofs["j1"]["native_loss"],
                "Paired initial native loss differs")
        summaries[key] = {"paired_batches_first": je0.PROBE_STEPS, "paired_batches_epoch100": 21,
                          "native_initial_sha256": expected["native_initial_sha256"],
                          "initial_bn_sha256": expected["initial_bn_sha256"],
                          "native_optimizer_groups_sha256": expected["native_groups_sha256"],
                          "first_order_sha256": expected["early"][0]["sources_sha256"],
                          "first_augmented_sha256": expected["early"][0]["tensor_sha256"],
                          "final_epoch_first_order_sha256": expected["late"][0]["sources_sha256"],
                          "min_active_anchors_first_70": min(r["eligible_anchors"] for r in expected["per_batch_regions"]),
                          "max_analytic_lr_error_first_70": max_lr_error, "smoke": proofs}
    je0.write(a.output, {**{k: preflight[k] for k in ("status", "protocol_sha256", "runtime", "framework_sha256",
                      "train_images", "train_groups", "train_images_labels_sha_verified", "free_disk_bytes_after_smoke",
                      "estimated_four_arm_checkpoint_bytes", "formal_updates", "val_inference", "test_inference")},
                      "seed_results": summaries, "preflight_sha256": sha256_file(source),
                      "total_gpu_smoke_attempts": 4, "seed42_updates": 0})


if __name__ == "__main__":
    main()
