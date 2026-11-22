"""One independently budgeted real TRAIN smoke; never a formal training entry."""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a11_smoke_numerics import finite_tree, optimizer_group_names
from tools.exp14.formal_e0_e1 import cpu_state, hash_state
from tools.exp15.jb import matched_batches
from tools.exp17.engineering.bn import bn_modules, buffer_snapshot, buffers_equal, rng_equal, rng_snapshot
from tools.exp17.engineering.checkpoint import save_engineering_checkpoint
from tools.exp17.engineering.calibration import verify_checkpoint_benchmark
from tools.exp17.engineering.calibration_extension import PASS as EXTENSION_PASS, SCHEMA as EXTENSION_SCHEMA, analyze as analyze_extension
from tools.exp17.engineering.coverage import PASS as COVERAGE_PASS, compare_reference, native_evidence, read_reference
from tools.exp17.engineering.data import load, require, sha, write
from tools.exp17.engineering.model import forward_joint

AUTHORIZATION = "EXP17_J2_ENGINEERING_SMOKE_AUTHORIZATION"
CAP = 50


def reserve_attempt(path):
    """The separate exclusive run claim permits only one writer of this ledger."""
    with Path(path).open("a+", encoding="ascii") as stream:
        stream.seek(0); tokens = stream.read().splitlines()
        require(tokens == [str(i) for i in range(1, len(tokens) + 1)] and len(tokens) < CAP,
                "Cumulative J2 engineering optimizer-attempt budget exhausted or malformed")
        number = len(tokens) + 1
        stream.write(str(number) + "\n"); stream.flush(); os.fsync(stream.fileno())
    return number


def validate_gates(config, authorization):
    require(config.get("formal_training_authorized") is False, "Formal training is forbidden")
    require(authorization["stage"] == AUTHORIZATION and authorization["seed"] == 42
            and authorization["maximum_optimizer_attempts"] == CAP
            and authorization["manifest_sha256"] == config["manifest_sha256"], "Smoke authorization scope differs")
    coverage = load(config["coverage_summary"])
    require(sha(config["coverage_summary"]) == authorization["coverage_summary_sha256"]
            and coverage["status"] == COVERAGE_PASS and coverage["all_2100_native_batches_exact"],
            "Full read-only native trajectory has not passed")
    require(coverage["data_identity"]["trusted_manifest_sha256"] == config["manifest_sha256"],
            "Read-only coverage belongs to another admitted manifest")
    receipt = load(config["lambda_receipt"])
    require(sha(config["lambda_receipt"]) == authorization["lambda_receipt_sha256"]
            and receipt["status"] == "PASS" and receipt["manifest_sha256"] == config["manifest_sha256"]
            and receipt["bn_mode"] == "replay_a" and receipt["formal_training_authorized"] is False,
            "A fixed TRAIN-calibrated Pair weight and BN recommendation are required")
    require(math.isfinite(receipt["lambda_pair"]) and receipt["lambda_pair"] > 0,
            "Invalid calibrated Pair weight")
    calibration = load(config["calibration_summary"])
    require(sha(config["calibration_summary"]) == receipt["summary_sha256"]
            and calibration["bn_recommendation"] == "replay_a"
            and receipt["calibration_source_sha256"] == sha(Path(__file__).with_name("calibration.py"))
            and receipt["rules_source_sha256"] == sha(Path(__file__).with_name("calibration_rules.py")),
            "Calibrated coefficient does not match its frozen source and full decision evidence")
    if receipt["schema"] == EXTENSION_SCHEMA:
        original = load(config["original_gradient_summary"])
        require(sha(config["original_gradient_summary"]) == receipt["original_gradient_summary_sha256"]
                and sha(config["candidate_extension_plan"]) == receipt["extension_plan_sha256"]
                and receipt["extension_source_sha256"] == sha(Path(__file__).with_name("calibration_extension.py"))
                and receipt["candidate_origin"] == "ADDITIONAL_TRAIN_ONLY_CANDIDATE_NOT_PREREGISTERED",
                "Additional candidate is missing its unchanged original HOLD, plan or source identity")
        recomputed = analyze_extension(original, receipt["lambda_pair"])
        require(calibration["status"] == EXTENSION_PASS and recomputed["status"] == EXTENSION_PASS
                and calibration["lambda_pair"] == receipt["lambda_pair"]
                and calibration["snapshot_median_weighted_ratios"] == recomputed["snapshot_median_weighted_ratios"],
                "Additional candidate does not pass the original strength/conflict/invariance evidence")
    else:
        require(calibration["status"] == "PASS_FIXED_TRAIN_LAMBDA_SCREEN"
                and calibration["lambda_screen"]["recommended_lambda_pair"] == receipt["lambda_pair"],
                "Original suggested-grid calibration did not pass")
    for name, identity in authorization["engineering_source_sha256"].items():
        require(sha(Path(__file__).parent / name) == identity, "Authorized engineering source changed")
    # The original prefix also needs its graph/input identity proof, even when
    # the optional checkpoint optimization was measured and deliberately unused.
    verify_checkpoint_benchmark(config, config["fixed_batches"])
    require(sha(config["frozen_j1_protocol"]) == config["frozen_j1_protocol_sha256"], "Frozen J1 protocol changed")
    return receipt


def parameter_identity(model, domain):
    return hash_state({n: p for n, p in model.named_parameters() if p.requires_grad and
                       ((n.startswith("region_projection.")) if domain == "projection" else
                        (n.startswith("model.") and int(n.split(".")[1]) <= 10))})


def run(config, output, authorization):
    receipt = validate_gates(config, authorization)
    limit = config.get("smoke_steps", config.get("smoke_update_limit", CAP))
    require(type(limit) is int and 1 <= limit <= min(CAP, config.get("smoke_update_limit", CAP)),
            "Configured smoke budget exceeds the fixed authorization ceiling")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    trainer = None
    summary = {"status": "EXP17_J2_ENGINEERING_SMOKE_HARD_GATE", "effective_updates": 0,
               "optimizer_attempts": 0, "formal_training_authorized": False}
    started = time.perf_counter()
    try:
        from tools.exp17.engineering.runtime import setup_trainer, prepare_batch
        trainer = setup_trainer(config, output / "native_trainer")
        require(trainer.batch_size == 32 and trainer.args.imgsz == 640 and trainer.args.seed == 42,
                "Smoke must preserve native seed42/640/batch32")
        require(not Path(config["smoke_budget_file"]).exists(), "Existing smoke attempt ledger cannot be reset or reused")
        claim = Path(config["smoke_budget_file"]).with_name("SMOKE_RUN_CLAIM.json")
        write(claim, {"stage": AUTHORIZATION, "output": str(output), "manifest_sha256": config["manifest_sha256"],
                      "authorization_identity": authorization, "formal_training_authorized": False})
        references = read_reference(config["baseline_ledger"])
        weight = float(receipt["lambda_pair"])
        before_native, before_projection = trainer.initial_native, trainer.projection_initial
        optimizer_groups = copy.deepcopy(optimizer_group_names(trainer.model, trainer.optimizer))
        require(optimizer_groups == trainer.optimizer_names and not trainer.optimizer.state,
                "Smoke must begin with the original parameter groups and fresh AdamW slots")
        end_of_previous_iteration = time.perf_counter()
        training_loop_started = end_of_previous_iteration
        rows = []
        with (output / "step_metrics.jsonl").open("x", encoding="utf-8") as metrics, \
             (output / "explicit_augmentation_traces.jsonl").open("x", encoding="utf-8") as traces:
            for step, batch in matched_batches(trainer, limit):
                compare_reference(native_evidence(trainer, batch, step), references[step - 1], trainer.formal)
                rng = rng_snapshot()
                prepared = prepare_batch(batch, trainer.index, trainer.groups, decode_N=True, device=trainer.device)
                require(rng_equal(rng), "N replay consumed a native RNG stream")
                trainer.optimizer.zero_grad(set_to_none=True)
                before = {name: int(module.num_batches_tracked) for name, module in bn_modules(trainer.model).items()}
                backbone_before = parameter_identity(trainer.model, "backbone")
                projection_before = parameter_identity(trainer.model, "projection")
                require(reserve_attempt(config["smoke_budget_file"]) == step, "Budget ledger and native batch position differ")
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
                with autocast(trainer.amp, device="cuda"):
                    result = forward_joint(trainer.model, prepared.batch, prepared.a_regions, prepared.normal_images,
                                           prepared.normal_masks, prepared.normal_regions, prepared.bindings,
                                           bn_mode="replay_a", lambda_pair=weight,
                                           checkpoint_prefix=bool(config.get("checkpoint_prefix", False)))
                require(finite_tree((result.total, result.yolo, result.supcon, result.pair, result.parts)),
                        "Nonfinite J2 loss or native loss component")
                trainer.loss = result.total
                trainer.audit.check_forward(result.total, {"yolo": result.yolo, "supcon": result.supcon,
                                                          "pair": result.pair, "total": result.total})
                after_forward_bn = buffer_snapshot(trainer.model)
                if result.n_p3 is not None:result.n_p3.retain_grad()
                trainer.scaler.scale(result.total).backward()
                require(buffers_equal(trainer.model, after_forward_bn), "N backward recomputation changed BN running buffers")
                n_norm = 0.0
                if result.n_p3 is not None:
                    require(result.n_p3.grad is not None and bool(torch.isfinite(result.n_p3.grad).all()),
                            "Sparse N lost its finite gradient path")
                    n_norm = float(result.n_p3.grad.float().norm()) / trainer.scaler.get_scale()
                    require(float(result.pair.detach()) == 0 or n_norm > 0, "Active Pair hinge has no N gradient")
                numerical = trainer.audit.optimizer_step()
                after = {name: int(module.num_batches_tracked) for name, module in bn_modules(trainer.model).items()}
                require(all(after[name] - before[name] == 1 for name in result.bn_statistics),
                        "Native A BN did not advance exactly once")
                require(parameter_identity(trainer.model, "backbone") != backbone_before
                        and parameter_identity(trainer.model, "projection") != projection_before,
                        "Backbone or shared projection did not really update")
                require(trainer.audit.optimizer_attempts == trainer.audit.native_steps == trainer.audit.effective_updates == step
                        and trainer.audit.amp_skips == 0, "Unexpected optimizer update or AMP skip")
                torch.cuda.synchronize()
                norm = numerical["gradient_norm"]
                row = {"step": step, "epoch": trainer.epoch + 1, "lambda_pair": weight,
                       "yolo_loss": float(result.yolo.detach()), "supcon_loss": float(result.supcon.detach()),
                       "pair_loss": float(result.pair.detach()), "total_loss": float(result.total.detach()),
                       "gradient_norm": norm, "clip_coefficient": min(1.0, 10.0 / (norm + 1e-6)),
                       "amp_scale_before": numerical["scale_before"], "amp_scale_after": numerical["scale_after"],
                       "effective_updates": step, "N_P3_unscaled_gradient_norm": n_norm,
                       "bn_buffers_sha256": hash_state({n: v for n, v in trainer.model.named_buffers() if n.endswith(("running_mean", "running_var", "num_batches_tracked"))}),
                       "BN_advances_per_A": 1, "N_extra_BN_updates": 0, "native_RNG_unchanged_by_N": True,
                       "coverage": prepared.coverage, "pair": result.report,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "seconds": time.perf_counter() - tick,
                       "native_optimizer_parameter_groups_unchanged": optimizer_group_names(trainer.model, trainer.optimizer) == optimizer_groups,
                       "numeric_audit": numerical}
                require(row["native_optimizer_parameter_groups_unchanged"], "Actual optimizer parameter groups changed")
                row["end_to_end_iteration_seconds"] = time.perf_counter() - end_of_previous_iteration
                metrics.write(json.dumps(row, allow_nan=False) + "\n");metrics.flush()
                traces.write(json.dumps({"step": step, "traces": [t.compact_record() for t in prepared.trace]}, allow_nan=False) + "\n")
                rows.append(row); summary["effective_updates"] = summary["optimizer_attempts"] = step
                print(json.dumps({k: row[k] for k in ("step", "total_loss", "pair_loss", "gradient_norm", "seconds", "peak_allocated_bytes")}), flush=True)
                del result, prepared
                end_of_previous_iteration = time.perf_counter()
        training_loop_elapsed = time.perf_counter() - training_loop_started
        checkpoint = save_engineering_checkpoint(trainer, output / "engineering_last.pt",
            {"manifest_sha256": config["manifest_sha256"], "lambda_pair": weight, "bn_mode": "replay_a",
             "formal_training_authorized": False}, {"attempts_cap": CAP, "effective_updates": limit})
        summary.update(status="PASS_521_J2_ENGINEERING_SMOKE", lambda_pair=weight, bn_mode="replay_a",
                       amp_skips=trainer.audit.amp_skips, checkpoint=checkpoint, steps=rows,
                       initial_native_sha256=before_native, initial_projection_sha256=before_projection,
                       peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows),
                       peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in rows),
                       mean_step_seconds=sum(r["seconds"] for r in rows)/len(rows),
                       mean_end_to_end_iteration_seconds=sum(r["end_to_end_iteration_seconds"] for r in rows)/len(rows),
                       training_loop_wall_seconds=training_loop_elapsed,
                       mean_full_training_loop_seconds_per_update=training_loop_elapsed/len(rows),
                       step_timing_scope="JOINT_FORWARD_BACKWARD_NUMERIC_AUDIT_EXCLUDES_DATA_REPLAY_AND_BINDING",
                       inference_uses="FP32_NATIVE_MASTER_ONLY_AUXILIARY_HEAD_REMOVED",
                       val_test_image_reads=0, native_trajectory_matched_batches=limit)
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error), "evidence": getattr(error, "evidence", None)}
        if trainer is not None:
            summary.update(effective_updates=trainer.audit.effective_updates, optimizer_attempts=trainer.audit.optimizer_attempts,
                           amp_skips=trainer.audit.amp_skips)
            try:
                stopped = output / "stopped_engineering_state.pt"
                with stopped.open("xb") as stream:
                    torch.save({"schema": "exp17_j2_failed_engineering_evidence_only_v1", "formal_model": False,
                                "state": cpu_state(trainer.model.state_dict()),
                                "optimizer": cpu_state(trainer.optimizer.state_dict()),
                                "scaler": trainer.scaler.state_dict(),
                                "effective_updates": trainer.audit.effective_updates}, stream)
                summary["failure_state_evidence_sha256"] = sha(stopped)
            except Exception as evidence_error:
                summary["failure_state_save_error"] = str(evidence_error)
    finally:
        if trainer is not None:trainer.close()
        summary["elapsed_seconds"] = time.perf_counter() - started
        ledger = Path(config["smoke_budget_file"])
        summary["budget_reserved_attempts"] = len(ledger.read_text(encoding="ascii").splitlines()) if ledger.exists() else 0
        summary["budget_ledger_sha256"] = sha(ledger) if ledger.exists() else None
        write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True);parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize", type=Path, required=True)
    args = parser.parse_args();result = run(load(args.config), args.output, load(args.authorize))
    print(json.dumps({k: result[k] for k in ("status", "effective_updates", "optimizer_attempts")}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS_521_J2_ENGINEERING_SMOKE" else 2)


if __name__ == "__main__":main()
