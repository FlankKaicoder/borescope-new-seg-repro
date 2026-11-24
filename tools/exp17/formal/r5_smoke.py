"""Single-use R5 AMP validation, bounded to 3A/3N/6 VJPs/1 disposable update."""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch

from tools.exp17.formal.authorization import exclusive_json, file_sha, load, protocol_bindings, require
from tools.exp17.formal.repaired_seed42 import MODE, adapter_type


def forward_fingerprints(result):
    from tools.exp17.formal.checkpoint import state_hash
    return {key: state_hash(getattr(result, key).detach().cpu())
            for key in ("yolo", "parts", "supcon", "pair", "total", "a_p3", "n_p3", "a_embedding", "n_embedding")}


def run(gate):
    from tools.exp17.engineering.bn import buffer_snapshot, buffers_equal
    from tools.exp17.engineering.calibration import gradient_basis, cosine, norm
    from tools.exp17.engineering.model import forward_joint
    from tools.exp17.engineering.runtime import DecodedAssetCache, prepare_batch
    from tools.exp17.formal.checkpoint import capture_training_state, restore_training_state, state_hash
    protocol, config = load(gate / "protocol.json"), load(gate / "runtime_config.json")
    bindings = protocol_bindings(protocol, file_sha(gate / "protocol.json"))
    exclusive_json(gate / "smoke_claim.json", {"code_sha256": protocol["code"]["sha256"],
                                              "scope": protocol["engineering_repair"]["smoke_budget"]})
    started, adapter = time.perf_counter(), None
    counts = {"A": 0, "N": 0, "component_autograd": 0, "joint_backward": 0, "disposable_updates": 0, "VAL": 0}
    try:
        adapter = adapter_type()(config, protocol, gate / "smoke_output", bindings)
        trainer = adapter.trainer
        initial = capture_training_state(trainer, bindings=bindings,
            position={"epoch": 0, "batch_position": 0, "global_update": 0},
            sampler_state={"schema": "BEFORE_FIRST_EPOCH", "epoch": 0, "next_batch": 0})
        parameters = [p for p in trainer.model.parameters() if p.requires_grad]
        spec = adapter.fixed_batches[1]
        require(file_sha(spec["path"]) == spec["sha256"], "R5 smoke batch identity differs")
        vectors, fingerprints, buffers = {}, {}, {}
        torch.cuda.reset_peak_memory_stats()
        for mode in ("replay_a", MODE):
            restore_training_state(trainer, initial, bindings=bindings)
            packet = torch.load(spec["path"], map_location="cpu", weights_only=False)
            prepared = prepare_batch(packet["batch"], adapter.index, adapter.groups, decode_N=True,
                                     device=trainer.device, asset_cache=DecodedAssetCache(max_bytes=0))
            with torch.autocast("cuda", dtype=torch.float16):
                counts["A"] += 1
                counts["N"] += 1
                result = forward_joint(trainer.model, prepared.batch, prepared.a_regions, prepared.normal_images,
                    prepared.normal_masks, prepared.normal_regions, prepared.bindings,
                    bn_mode=mode, lambda_pair=4., checkpoint_prefix=False)
            require(float(result.pair.detach()) > 0 and prepared.bindings, "Smoke requires an active Pair hinge")
            fingerprints[mode] = forward_fingerprints(result)
            buffers[mode] = buffer_snapshot(trainer.model)
            vectors[mode] = []
            for loss in (result.yolo, 8 * result.supcon, 4 * result.pair):
                counts["component_autograd"] += 1
                vectors[mode].append(gradient_basis(loss, parameters, scale=trainer.scaler.get_scale()))
            require(buffers_equal(trainer.model, buffers[mode]), "R5 smoke backward changed BN buffers")
            require(all(p.grad is None for p in parameters), "Component gradients polluted param.grad")
            del loss, result, prepared, packet
            gc.collect()
        require(fingerprints["replay_a"] == fingerprints[MODE], "R5 forward differs under AMP")
        require(state_hash(buffers["replay_a"]) == state_hash(buffers[MODE]), "A running BN differs")
        require(torch.equal(vectors["replay_a"][0], vectors[MODE][0])
                and torch.equal(vectors["replay_a"][1], vectors[MODE][1]), "YOLO/SupCon8 gradients changed")
        pair = {"detached_L2": norm(vectors["replay_a"][2]), "connected_L2": norm(vectors[MODE][2]),
                "cosine": cosine(vectors["replay_a"][2], vectors[MODE][2])}
        del vectors
        component_peak = torch.cuda.max_memory_allocated()
        component_reserved = torch.cuda.max_memory_reserved()
        restore_training_state(trainer, initial, bindings=bindings)
        adapter.start_epoch(0)
        packet = torch.load(spec["path"], map_location="cpu", weights_only=False)
        counts["A"] += 1
        counts["N"] += 1
        counts["joint_backward"] += 1
        counts["disposable_updates"] += 1
        row = adapter.train_batch(packet["batch"], 1)
        require(row["effective_updates"] == 1 and row["amp_skipped"] == 0 and row["N_extra_BN_updates"] == 0,
                "R5 disposable native update failed")
        final = capture_training_state(trainer, bindings=bindings,
            position={"epoch": 0, "batch_position": 1, "global_update": 1},
            sampler_state={"schema": "SMOKE_ONLY_NO_FORMAL_RESUME", "epoch": 0, "next_batch": 1})
        restore_training_state(trainer, final, bindings=bindings)
        report = {"status": "PASS", "code_sha256": protocol["code"]["sha256"], "counts": counts,
            "seconds": time.perf_counter() - started, "forward_bitwise_equal": True,
            "YOLO_SupCon8_gradients_bitwise_equal": True, "pair": pair,
            "peak_allocated_bytes": max(component_peak, torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": max(component_reserved, torch.cuda.max_memory_reserved()), "disposable_update": row,
            "capacity_limitation": "N14 fixture; formal historical maxN25 is monitored without changing batch",
            "constructor_stride_probes": "Native setup only, outside counted A/N training traversals"}
        exclusive_json(gate / "smoke.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "disposable_update"}), flush=True)
    except BaseException as error:
        exclusive_json(gate / "smoke_failure.json", {"status": "HOLD", "counts": counts,
            "error": str(error), "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started})
        raise
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    run(parser.parse_args().gate)
