"""Zero-update shared-prefix equivalence, sparse N=A and memory/time evidence."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a11_full_numerics import capture_rng_state, restore_rng_state
from tools.exp14.a11_smoke_numerics import exact_tree, finite_tree
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp15 import regions as j1
from tools.exp17.engineering.bn import auxiliary_bn, buffer_snapshot, buffers_equal
from tools.exp17.engineering.calibration_rules import RULES
from tools.exp17.engineering.data import load, require, sha, write
from tools.exp17.engineering.model import forward_a, forward_n


def prefix_benchmark(model, prepared, *, amp):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    before, rng = cpu_state(model.state_dict()), capture_rng_state()
    before_hash = hash_state(before)
    try:
        with autocast(amp, device="cuda"):
            a = forward_a(model, prepared.batch, prepared.a_regions)
        statistics_a = a.bn_statistics
        a_p3, a_embedding = a.p3.detach(), a.embedding.detach()
        del a
        bn_after_a = buffer_snapshot(model)
        slots = {binding.n_slot: binding.a_slot for binding in prepared.bindings}
        require(set(slots) == set(range(len(prepared.normal_images))), "Sparse N slot binding is incomplete")
        selected_a = torch.stack([prepared.batch["img"][slots[i]] for i in range(len(slots))])
        n_keep, a_keep = prepared.normal_regions.cells > 0, prepared.a_regions.cells > 0
        a_rows = []
        for ns, ordinal in zip(prepared.normal_regions.slots[n_keep], prepared.normal_regions.ordinals[n_keep], strict=True):
            candidates = np.flatnonzero((prepared.a_regions.slots[a_keep] == slots[int(ns)]) &
                                       (prepared.a_regions.ordinals[a_keep] == ordinal))
            require(len(candidates) == 1, "N=A region identity has no unique A row")
            a_rows.append(int(candidates[0]))
        invariance = {}
        for mode in ("independent", "replay_a"):
            with autocast(amp, device="cuda"):
                p3 = forward_n(model, selected_a, bn_mode=mode, a_statistics=statistics_a)
                embedding = j1.project_regions(p3, prepared.normal_masks, prepared.normal_regions, model.region_projection)
            target_p3 = a_p3[torch.tensor([slots[i] for i in range(len(slots))], device=a_p3.device)]
            target_embedding = a_embedding[torch.tensor(a_rows, device=a_embedding.device)]
            relative = float((p3.float() - target_p3.float()).norm() / target_p3.float().norm().clamp_min(1e-20))
            distance = float((1 - (embedding.float() * target_embedding.float()).sum(1)).max())
            invariance[mode] = {"relative_P3_L2_error": relative, "max_embedding_cosine_distance": distance,
                                "same_input_sparse_slots": len(slots), "extra_bn_updates": 0}
            del p3, embedding
        prefix = "AMP" if amp else "FP32"
        require(invariance["replay_a"]["relative_P3_L2_error"] <= RULES[prefix + "_N_equals_A_relative_P3_L2_tolerance"]
                and invariance["replay_a"]["max_embedding_cosine_distance"] <= RULES[prefix + "_N_equals_A_max_embedding_cosine_distance"],
                "Sparse N=A exceeds the prespecified counterfactual feature tolerance")
        del a_p3, a_embedding, selected_a
        torch.cuda.empty_cache()
        variants, reference = {}, None
        for variant in ("full_N_reference", "prefix", "checkpointed_prefix"):
            timings, peaks = [], []
            for repeat in range(3):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started = time.perf_counter()
                with autocast(amp, device="cuda"):
                    if variant == "full_N_reference":
                        features = []
                        handle = model.model[j1.P3_LAYER].register_forward_hook(lambda _m, _i, v: features.append(v))
                        try:
                            with auxiliary_bn(model, "replay_a", statistics_a):
                                unused_native_heads = model(prepared.normal_images)
                        finally:
                            handle.remove()
                        p3 = features[0]
                    else:
                        p3 = forward_n(model, prepared.normal_images, bn_mode="replay_a", a_statistics=statistics_a,
                                       checkpoint_prefix=variant == "checkpointed_prefix")
                    embedding = j1.project_regions(p3, prepared.normal_masks, prepared.normal_regions, model.region_projection)
                    scalar = embedding.float().sum()
                gradients = torch.autograd.grad(scalar, [p for _, p in parameters], allow_unused=True)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                peaks.append(torch.cuda.max_memory_allocated())
                require(finite_tree((p3, embedding, gradients)), "Nonfinite prefix benchmark")
                current = (p3.detach().cpu(), embedding.detach().cpu(), tuple(None if g is None else g.detach().cpu() for g in gradients))
                if reference is None:
                    reference = current
                else:
                    torch.testing.assert_close(current[0], reference[0], rtol=0, atol=0)
                    torch.testing.assert_close(current[1], reference[1], rtol=0, atol=0)
                    require(len(current[2]) == len(reference[2]), "Gradient list changed")
                    for actual, expected in zip(current[2], reference[2], strict=True):
                        require((actual is None) == (expected is None), "Prefix changed gradient endpoints")
                        if actual is not None:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                del gradients, scalar, p3, embedding, current
                if variant == "full_N_reference":del unused_native_heads, features
            variants[variant] = {"seconds_each": timings, "median_seconds": statistics.median(timings),
                                 "peak_allocated_bytes": max(peaks), "equivalent_P3_embedding_and_gradients": True,
                                 "timing_scope": "N_ONLY_FORWARD_AND_BACKWARD_THREE_REPEATS_INCLUDES_FIRST_CALL"}
        require(buffers_equal(model, bn_after_a), "Prefix benchmark wrote extra BN running buffers")
        return {"status": "PASS_PREFIX_EQUIVALENCE_AND_SPARSE_COUNTERFACTUAL", "amp": amp,
                "invariance": invariance, "variants": variants, "N_images": len(slots),
                "checkpoint_prefix_permitted": True,
                "note": "Full N heads are an isolated equivalence reference only; production N stops at16.",
                "optimizer_updates": 0}
    finally:
        model.load_state_dict(before, strict=True)
        restore_rng_state(rng)
        require(hash_state(model.state_dict()) == before_hash, "Benchmark polluted model state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from tools.exp17.engineering.runtime import setup_trainer, prepare_batch
    trainer = None
    try:
        config = load(args.config)
        fixed = [r for r in config["fixed_batches"] if r["batch_position"] == 1]
        require(len(fixed) == 1 and Path(fixed[0]["path"]).resolve(strict=True) == args.batch.resolve(strict=True)
                and sha(args.batch) == fixed[0]["sha256"], "Benchmark must use the byte-bound preselected first TRAIN batch")
        trainer = setup_trainer(config, args.output / "native_trainer")
        saved = torch.load(args.batch, map_location="cpu", weights_only=False)
        require(saved["schema"] == "EXP17_J2_NATIVE_CPU_BATCH_CACHE_V1" and saved["epoch"] == 0
                and saved["batch_position"] == 1, "Benchmark batch cache identity differs")
        prepared = prepare_batch(saved["batch"], trainer.index, trainer.groups, decode_N=True, device=trainer.device)
        require(prepared.bindings, "Fixed benchmark batch has no complete A/B/N")
        report = prefix_benchmark(trainer.model, prepared, amp=trainer.amp)
        report["identity"] = {"batch_sha256": sha(args.batch), "batch_position": 1,
                              "config_sha256": sha(args.config), "manifest_sha256": config["manifest_sha256"],
                              "source_sha256": sha(__file__),
                              "rules_sha256": sha(Path(__file__).with_name("calibration_rules.py")),
                              "model_sources_sha256": {name: sha(Path(__file__).with_name(name))
                                                       for name in ("model.py", "bn.py", "losses.py")}}
        require(trainer.audit.effective_updates == trainer.audit.native_steps == 0, "Benchmark performed an optimizer update")
        write(args.output / "summary.json", report)
        print(json.dumps(report, sort_keys=True))
    except Exception as error:
        write(args.output / "stopped.json", {"status": "ENGINEERING_BENCHMARK_HARD_GATE", "error": str(error), "optimizer_updates": 0})
        raise
    finally:
        if trainer is not None:trainer.close()


if __name__ == "__main__":main()
