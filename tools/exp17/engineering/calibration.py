"""Isolated TRAIN-only BN diagnostics and fixed-lambda gradient calibration.

No optimizer steps. Frozen CPU batches, FP32 masters and all state/RNG identities
are restored between diagnostic forwards. Seven candidates use one gradient
basis algebraically; neither VAL nor a loss-minimizing BN selection is present.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.exp14.formal_e0_e1 import hash_state
from tools.exp15 import regions as j1
from tools.exp17.engineering.bn import buffer_snapshot, buffers_equal, rng_equal, rng_snapshot
from tools.exp17.engineering.calibration_rules import CHECKPOINT_STATES, FIXED_BATCH_POSITIONS, LAMBDA_CANDIDATES, RULES, select_fixed_lambda
from tools.exp17.engineering.model import forward_joint, forward_n, native_state_dict


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); out.write("\n")


def reject_optimizer_step(_optimizer, _args, _kwargs):
    raise ValueError("Calibration is read-only: optimizer.step is forbidden before any parameter update")


def frozen_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def bn_digest(snapshot):
    return hash_state({name + "." + key: value for name, state in snapshot.items()
                       for key, value in state.items() if value is not None})


def rng_digest(snapshot):
    h = hashlib.sha256()
    numpy_state = snapshot["numpy"]
    h.update(json.dumps({"python": snapshot["python"], "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}, sort_keys=True).encode())
    for value in [snapshot["torch_cpu"], *snapshot["torch_cuda"]]: h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def restore_rng(state):
    # Whole diagnostic isolation, not a hidden N-branch RNG repair. auxiliary_bn
    # separately rejects every consumed RNG stream before this outer restoration.
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]: torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def isolated_model(model):
    state, rng = frozen_state(model), rng_snapshot()
    before = hash_state(state); flags = [(m, m.training) for m in model.modules()]
    gradients = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    try: yield {"before_model_sha256": before, "before_bn": buffer_snapshot(model)}
    finally:
        model.load_state_dict(state, strict=True)
        for module, training in flags: module.training = training
        for parameter, old in gradients:
            require((old is None and parameter.grad is None) or (old is not None and parameter.grad is not None
                    and torch.equal(old, parameter.grad)), "Diagnostic changed preexisting .grad buffers")
        restore_rng(rng)
        require(hash_state(model.state_dict()) == before and rng_equal(rng), "Diagnostic isolation failed to restore model or RNG")


def parameter_layout(model):
    names, sizes, domains = [], [], {}
    offset = 0
    for name, value in model.named_parameters():
        if not value.requires_grad: continue
        require(value.dtype == torch.float32, "Calibration requires FP32 master parameters")
        names.append(name); sizes.append(value.numel())
        layer = ".".join(name.split(".")[:2]) if name.startswith("model.") else "region_projection"
        indexes = torch.arange(offset, offset + value.numel(), dtype=torch.int64)
        domains.setdefault(layer, []).append(indexes)
        offset += value.numel()
    domains = {key: torch.cat(parts) for key, parts in domains.items()}
    domains["all_trainable"] = torch.arange(offset, dtype=torch.int64)
    backbone = [value for key, value in domains.items() if key.startswith("model.") and int(key.split(".")[1]) <= 10]
    require(backbone, "Missing shared backbone parameter domain")
    domains["backbone"] = torch.cat(backbone)
    return names, sizes, domains


def gradient_basis(loss, parameters, scale=1.0):
    require(loss.ndim == 0 and bool(torch.isfinite(loss.detach())) and math.isfinite(scale) and scale > 0, "Invalid diagnostic objective or AMP scale")
    values = torch.autograd.grad(loss * scale, parameters, retain_graph=True, allow_unused=True)
    flat = []
    for parameter, value in zip(parameters, values, strict=True):
        if value is None: flat.append(torch.zeros(parameter.numel(), dtype=torch.float64))
        else:
            require(bool(torch.isfinite(value).all()), "Nonfinite diagnostic gradient")
            flat.append(value.detach().to(device="cpu", dtype=torch.float64).reshape(-1) / scale)
    return torch.cat(flat)


def norm(value): return float(torch.linalg.vector_norm(value))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return float(torch.dot(a, b)) / denominator if denominator else 0.0


def candidate_algebra(yolo, supcon8, pair, domains, clip=10.0):
    require(clip == 10.0, "Frozen native gradient clip changed")
    main = yolo + supcon8
    rows = []
    for weight in LAMBDA_CANDIDATES:
        joint = main + weight * pair
        total = norm(joint); coefficient = min(1.0, clip / (total + 1e-6))
        bb = domains["backbone"]
        main_bb, pair_bb, joint_bb = norm(main[bb]), norm(pair[bb]), norm(joint[bb])
        rows.append({"lambda_pair": weight, "joint_all_trainable_norm": total,
                     "clip_coefficient": coefficient, "clipped_joint_all_trainable_norm": total * coefficient,
                     "weighted_pair_all_trainable_norm": weight * norm(pair),
                     "clipped_weighted_pair_all_trainable_norm": coefficient * weight * norm(pair),
                     "joint_backbone_norm": joint_bb, "main_joint_backbone_norm": main_bb,
                     "weighted_pair_backbone_norm": weight * pair_bb,
                     "weighted_pair_to_main_joint_backbone_ratio": weight * pair_bb / main_bb if main_bb else None,
                     "weighted_pair_to_resulting_joint_backbone_ratio": weight * pair_bb / joint_bb if joint_bb else None})
        rows[-1]["domain_joint_norms"] = {name: {"pre_clip": norm(joint[indexes]),
             "post_clip": coefficient * norm(joint[indexes]), "weighted_pair_norm": weight * norm(pair[indexes])}
             for name, indexes in domains.items()}
    return rows


def gradient_report(yolo, supcon8, pair, domains):
    output = {}
    for name, positions in domains.items():
        y, s, p = yolo[positions], supcon8[positions], pair[positions]
        main = y + s
        output[name] = {"yolo_norm": norm(y), "supcon8_norm": norm(s), "pair_norm": norm(p),
                        "main_joint_norm": norm(main), "pair_to_main_joint_ratio": norm(p) / norm(main) if norm(main) else None,
                        "pair_yolo_cosine": cosine(p, y), "pair_supcon8_cosine": cosine(p, s),
                        "basis_gram": [[float(torch.dot(a, b)) for b in (y, s, p)] for a in (y, s, p)]}
    return output


def normal_equals_a(model, prepared, output, mode, checkpoint_prefix, amp):
    if output.n_p3 is None:
        return {"status": "NO_ACTIVE_N", "pass": None}
    a_slots = prepared.a_regions.slots[prepared.a_regions.cells > 0]
    n_slots = prepared.normal_regions.slots[prepared.normal_regions.cells > 0]
    by_n = {}
    for binding in prepared.bindings:
        by_n[int(n_slots[binding.n_row])] = int(a_slots[binding.a_row])
    positions = torch.as_tensor([by_n[i] for i in range(len(prepared.normal_images))], device=prepared.batch["img"].device)
    images = prepared.batch["img"].index_select(0, positions)
    device_type = images.device.type
    with torch.autocast(device_type, enabled=amp, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16):
        feature = forward_n(model, images, bn_mode=mode, a_statistics=output.bn_statistics, checkpoint_prefix=checkpoint_prefix)
        embedding = j1.project_regions(feature, prepared.normal_masks, prepared.normal_regions, model.region_projection)
    reference = output.a_p3.index_select(0, positions)
    relative = norm((feature.detach().float() - reference.detach().float()).cpu().double().flatten()) / max(1e-30, norm(reference.detach().cpu().double().flatten()))
    ar = torch.as_tensor([b.a_row for b in prepared.bindings], device=embedding.device)
    nr = torch.as_tensor([b.n_row for b in prepared.bindings], device=embedding.device)
    distance = (1 - (output.a_embedding[ar].detach() * embedding[nr].detach()).sum(1)).float()
    max_distance = max(0.0, float(distance.max()))
    precision = "AMP" if amp else "FP32"
    passed = relative <= RULES[precision + "_N_equals_A_relative_P3_L2_tolerance"] and max_distance <= RULES[precision + "_N_equals_A_max_embedding_cosine_distance"]
    return {"status": "MEASURED_NATIVE_A_SUBSET_WITH_IDENTICAL_IMAGES", "pass": passed,
            "relative_P3_L2": relative, "max_P3_absolute_error": float((feature.detach().float() - reference.detach().float()).abs().max()),
            "max_embedding_cosine_distance": max_distance, "N_image_slots": len(images), "precision": precision,
            "comparison": "full native A statistics versus same A-image subset under the selected N BN mode"}


def compute_probe(model, prepared, *, state, batch_position, amp=False, checkpoint_prefix=False, gradient_scale=1.0):
    require(state in CHECKPOINT_STATES and batch_position in FIXED_BATCH_POSITIONS, "Unregistered calibration state/batch")
    require(type(amp) is bool and type(checkpoint_prefix) is bool, "Diagnostic execution flags must be explicit bool")
    names, sizes, domains = parameter_layout(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    modes, cached_n, basis_b, class_norms = {}, {}, None, {}
    start_model, start_bn, start_rng = hash_state(model.state_dict()), buffer_snapshot(model), rng_snapshot()
    started = time.perf_counter()
    for mode in ("independent", "replay_a"):
        with isolated_model(model):
            dtype = torch.float16 if prepared.batch["img"].device.type == "cuda" else torch.bfloat16
            with torch.autocast(prepared.batch["img"].device.type, enabled=amp, dtype=dtype):
                output = forward_joint(model, prepared.batch, prepared.a_regions, prepared.normal_images,
                    prepared.normal_masks, prepared.normal_regions, prepared.bindings, bn_mode=mode,
                    lambda_pair=None, checkpoint_prefix=checkpoint_prefix)
            y = gradient_basis(output.yolo, parameters, gradient_scale)
            s = gradient_basis(8 * output.supcon, parameters, gradient_scale)
            p = gradient_basis(output.pair, parameters, gradient_scale)
            report = gradient_report(y, s, p, domains)
            if mode == "replay_a":
                basis_b = (y, s, p)
                represented = len(output.per_class_pair_tensors)
                for cls, loss in output.per_class_pair_tensors.items():
                    component = gradient_basis(loss / represented, parameters, gradient_scale)
                    class_norms[str(cls)] = {"backbone_norm": norm(component[domains["backbone"]]),
                                             "all_trainable_norm": norm(component), "class_mean_divisor": represented,
                                             "per_class_pair_loss": float(loss.detach())}
            details = output.report.get("per_anchor", [])
            average = lambda key: sum(d[key] for d in details) / len(details) if details else None
            n_path_gradient = None
            if output.n_p3 is not None:
                value = torch.autograd.grad(output.pair * gradient_scale, output.n_p3, retain_graph=True, allow_unused=True)[0]
                if value is not None:
                    require(bool(torch.isfinite(value).all()), "Nonfinite N branch gradient")
                    n_path_gradient = norm(value.detach().cpu().double().flatten() / gradient_scale)
                cached_n[mode] = (output.n_p3.detach().cpu().float(), output.n_embedding.detach().cpu().float())
            # Invariance needs values and detached A statistics, not the already
            # consumed main loss graph. Release its activations before another N
            # graph, retaining a genuinely differentiable prefix for this check.
            invariant_reference = SimpleNamespace(n_p3=None if output.n_p3 is None else True,
                a_p3=output.a_p3.detach(), a_embedding=output.a_embedding.detach(), bn_statistics=output.bn_statistics)
            modes[mode] = {"yolo_loss": float(output.yolo.detach()), "supcon_loss": float(output.supcon.detach()),
                           "supcon8_loss": 8 * float(output.supcon.detach()), "pair_loss": float(output.pair.detach()),
                           "pair_report": output.report, "gradient_domains": report,
                           "A_N_mean_cosine_distance": average("d_an"), "A_B_mean_cosine_distance": average("d_ab"),
                           "active_hinge_fraction": sum(d["hinge"] > 0 for d in details) / len(details) if details else 0.0,
                           "zero_hinge_fraction": sum(d["hinge"] == 0 for d in details) / len(details) if details else 0.0,
                           "N_P3_pair_gradient_norm": n_path_gradient}
            del output
            if mode == "replay_a" and represented: del loss
            modes[mode]["N_equals_A_invariance"] = normal_equals_a(model, prepared, invariant_reference, mode, checkpoint_prefix, amp)
            del invariant_reference
    require(hash_state(model.state_dict()) == start_model and buffers_equal(model, start_bn) and rng_equal(start_rng), "Probe polluted model/BN/RNG")
    y, s, p = basis_b; backbone = domains["backbone"]
    denominator = norm((y + s)[backbone]); require(denominator > 0, "Zero native+SupCon reference gradient")
    cross = {}
    if set(cached_n) == {"independent", "replay_a"}:
        fa, ea = cached_n["independent"]; fb, eb = cached_n["replay_a"]
        cross = {"N_P3_relative_L2": norm((fa - fb).double().flatten()) / max(1e-30, norm(fb.double().flatten())),
                 "N_embedding_mean_cosine_distance": float((1 - (ea * eb).sum(1)).mean())}
    candidates = candidate_algebra(y, s, p, domains)
    for row in candidates:
        row["total_loss"] = modes["replay_a"]["yolo_loss"] + modes["replay_a"]["supcon8_loss"] + row["lambda_pair"] * modes["replay_a"]["pair_loss"]
    result = {"state": state, "batch_position": batch_position, "effective_updates_in_reference": CHECKPOINT_STATES[state],
              "pair_to_main_joint_backbone_ratio": norm(p[backbone]) / denominator,
              "pair_yolo_backbone_cosine": cosine(p[backbone], y[backbone]),
              "pair_supcon_backbone_cosine": cosine(p[backbone], s[backbone]),
              "global_pair_class_component_norms": {cls: value["backbone_norm"] for cls, value in class_norms.items()},
              "per_class_global_pair_gradient_components": class_norms,
              "bn_modes": modes, "between_bn_modes": cross,
              "lambda_candidates_algebra": candidates,
              "model_state_sha256_before": start_model, "model_state_sha256_after": hash_state(model.state_dict()),
              "BN_state_sha256_before": bn_digest(start_bn), "BN_state_sha256_after": bn_digest(buffer_snapshot(model)),
              "RNG_state_sha256_before": rng_digest(start_rng), "RNG_state_sha256_after": rng_digest(rng_snapshot()),
              "BN_and_RNG_restored": True, "parameters_with_grad_basis": len(names),
              "gradient_basis_precision": "UNSCALED_FP32_MASTER_GRADS_CPU_FLOAT64_ALGEBRA", "autograd_loss_scale": gradient_scale,
              "calibration_amp": amp, "checkpoint_prefix": checkpoint_prefix,
              "zero_norm_cosine_convention": "ZERO_REPORTED_WHEN_EITHER_VECTOR_NORM_IS_ZERO", "elapsed_seconds": time.perf_counter() - started,
              "optimizer_updates": 0, "formal_training_authorized": False}
    return result


def load_reference(model, spec, state):
    require(state in CHECKPOINT_STATES and state != "coco_initial", "Unknown reference checkpoint")
    path = Path(spec["path"]); before = sha(path); require(before == spec["sha256"], "Reference checkpoint file SHA256 differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved["arm"] == "j1" and saved["effective_updates"] == spec["step"] == CHECKPOINT_STATES[state], "Wrong J1 checkpoint update identity")
    master = saved["state"]
    require(all(not v.is_floating_point() or (v.dtype == torch.float32 and bool(torch.isfinite(v).all())) for v in master.values()), "Reference is not a finite FP32 master")
    require(hash_state(master) == saved["model_state_sha256"] and hash_state(master) != saved["ema_state_sha256"], "FP32 master/EMA identity differs")
    model.load_state_dict(master, strict=True)
    require(hash_state(native_state_dict(model)) == spec["expected_native_state_sha256"] == saved["native_state_sha256"], "Native checkpoint state does not match frozen reference")
    require(sha(path) == before, "Reference checkpoint bytes changed")
    return {"state": state, "checkpoint_sha256": before, "native_state_sha256": saved["native_state_sha256"],
            "model_state_sha256": saved["model_state_sha256"], "effective_updates": saved["effective_updates"],
            "master_selected_not_ema": True, "gradient_scale": float(saved["scaler"]["scale"])}


def verify_checkpoint_benchmark(config, fixed):
    require(config.get("checkpoint_prefix_benchmark_approved") is True, "N checkpoint optimization lacks the preceding GPU equivalence benchmark gate")
    benchmark = config["checkpoint_prefix_benchmark_receipt"]
    require(sha(benchmark["path"]) == benchmark["sha256"], "Prefix benchmark receipt bytes differ")
    checked = json.loads(Path(benchmark["path"]).read_text(encoding="utf-8"))
    first = next(r for r in fixed if r["batch_position"] == 1)
    require(checked["status"] == "PASS_PREFIX_EQUIVALENCE_AND_SPARSE_COUNTERFACTUAL"
            and checked["checkpoint_prefix_permitted"] is True and checked["optimizer_updates"] == 0
            and checked["identity"]["manifest_sha256"] == config["manifest_sha256"]
            and checked["identity"]["rules_sha256"] == sha(Path(__file__).with_name("calibration_rules.py"))
            and checked["identity"]["batch_sha256"] == first["sha256"], "Prefix benchmark covers another graph, manifest or batch")
    expected = {name: sha(Path(__file__).with_name(name)) for name in ("model.py", "bn.py", "losses.py")}
    require(checked["identity"]["model_sources_sha256"] == expected, "Model/BN/loss source changed after prefix benchmark")
    return benchmark["sha256"]


def run_calibration(config, output):
    from tools.exp17.engineering.runtime import prepare_batch, setup_trainer
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    trainer, optimizer_guard = None, None
    try:
        require(config.get("formal_training_authorized") is False, "Engineering calibration cannot authorize formal training")
        trainer = setup_trainer(config, output / "setup")
        optimizer_guard = trainer.optimizer.register_step_pre_hook(reject_optimizer_step)
        fresh = frozen_state(trainer.model); initial_rng = rng_snapshot()
        specs = config["reference_checkpoints"]
        require(set(specs) == set(CHECKPOINT_STATES) - {"coco_initial"}, "All three accepted J1 checkpoints are required")
        fixed = config["fixed_batches"]
        require({r["batch_position"] for r in fixed} == set(FIXED_BATCH_POSITIONS) and len(fixed) == 4, "Frozen CPU batch positions differ")
        checkpoint_prefix = config.get("checkpoint_prefix", False)
        if checkpoint_prefix:
            verify_checkpoint_benchmark(config, fixed)
        amp = config.get("calibration_amp", bool(trainer.amp))
        probes, references, input_hashes = [], [], {}
        for state in CHECKPOINT_STATES:
            trainer.model.load_state_dict(fresh, strict=True); restore_rng(initial_rng)
            if state == "coco_initial":
                reference = {"state": state, "native_state_sha256": trainer.initial_native,
                             "model_state_sha256": hash_state(trainer.model.state_dict()), "effective_updates": 0,
                             "fresh_coco_initialization": True, "gradient_scale": float(trainer.scaler.get_scale())}
            else:
                reference = load_reference(trainer.model, specs[state], state)
                input_hashes[specs[state]["path"]] = specs[state]["sha256"]
            references.append(reference)
            for spec in sorted(fixed, key=lambda x: x["batch_position"]):
                path = Path(spec["path"]); require(sha(path) == spec["sha256"], "Fixed CPU native batch SHA256 differs")
                input_hashes[str(path)] = spec["sha256"]
                cache = torch.load(path, map_location="cpu", weights_only=False)
                require(cache["schema"] == "EXP17_J2_NATIVE_CPU_BATCH_CACHE_V1" and cache["epoch"] == 0
                        and cache["batch_position"] == spec["batch_position"], "Fixed CPU batch cache identity differs")
                prepared = prepare_batch(copy.deepcopy(cache["batch"]), trainer.index, trainer.groups, decode_N=True,
                                         device=next(trainer.model.parameters()).device)
                probe = compute_probe(trainer.model, prepared, state=state, batch_position=spec["batch_position"],
                                      amp=amp, checkpoint_prefix=checkpoint_prefix, gradient_scale=reference["gradient_scale"])
                probe["reference"] = reference; probe["fixed_batch_sha256"] = spec["sha256"]
                probe["replay_coverage"] = prepared.coverage
                write(output / (state + "_batch_" + str(spec["batch_position"]).zfill(2) + ".json"), probe)
                probes.append(probe); del prepared
        decision = select_fixed_lambda(probes)
        invariant = [p["bn_modes"]["replay_a"]["N_equals_A_invariance"] for p in probes]
        failed = [v for v in invariant if v["pass"] is False]
        if failed:
            decision.update(status="HOLD_A_STATISTICS_REPLAY_INVARIANCE", recommended_lambda_pair=None)
        require(all(sha(Path(path)) == identity for path, identity in input_hashes.items()), "Read-only diagnostic input changed")
        require(trainer.audit.native_steps == trainer.audit.optimizer_attempts == 0, "Calibration attempted an optimizer update")
        trainer.model.load_state_dict(fresh, strict=True); restore_rng(initial_rng)
        result = {"status": decision["status"], "lambda_screen": decision,
                  "bn_recommendation": "replay_a" if not failed else None,
                  "bn_recommendation_basis": RULES["bn_recommendation_basis"],
                  "A_statistics_gradient_on_N": RULES["A_statistics_gradient_on_N"],
                  "bn_invariance_failures": len(failed), "references": references, "probes": probes,
                  "input_sha256": input_hashes, "calibration_source_sha256": sha(__file__),
                  "rules_source_sha256": sha(Path(__file__).with_name("calibration_rules.py")),
                  "optimizer_attempts": 0, "optimizer_updates": 0, "formal_training_authorized": False}
        write(output / "summary.json", result)
        if decision["status"] == "PASS_FIXED_TRAIN_LAMBDA_SCREEN" and not failed:
            receipt = output / "lambda_receipt.json"
            write(receipt, {"schema": "EXP17_J2_TRAIN_CALIBRATED_FIXED_LAMBDA_V1", "status": "PASS",
                            "lambda_pair": decision["recommended_lambda_pair"], "bn_mode": "replay_a",
                            "summary_sha256": sha(output / "summary.json"), "manifest_sha256": config["manifest_sha256"],
                            "calibration_source_sha256": sha(__file__), "rules_source_sha256": result["rules_source_sha256"],
                            "formal_training_authorized": False})
            receipt.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return result
    except Exception as error:
        write(output / "stopped.json", {"status": "HARD_GATE_CALIBRATION_STOPPED", "error_type": type(error).__name__,
                                       "reason": str(error), "optimizer_updates": 0, "formal_training_authorized": False})
        raise
    finally:
        if optimizer_guard is not None: optimizer_guard.remove()
        if trainer is not None: trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_calibration(config, args.output)
    print(json.dumps({"status": result["status"], "bn_mode": result["bn_recommendation"],
                      "lambda_pair": result["lambda_screen"]["recommended_lambda_pair"], "optimizer_updates": 0}))
    if result["status"] != "PASS_FIXED_TRAIN_LAMBDA_SCREEN": raise SystemExit(2)


if __name__ == "__main__": main()
