"""Frozen scientific contract. All real-network operations remain unauthorized."""
from __future__ import annotations

from copy import deepcopy
from .common import digest, require

BASE_HEAD = "3d58386d969a56279660d535bb63ce87e2255292"
ARMS = ("j1", "a1_true", "a1_shuffled")
FORBIDDEN_CPU_OPERATIONS = ("calibrate", "gpu-smoke", "train", "evaluate", "export")


def make_protocol(epochs=200):
    require(type(epochs) is int and epochs in (100, 200), "Only independent H100/H200 profiles exist")
    return {
        "schema": "EXP18_PROTOCOL_V1", "phase": "EXP18_CPU", "seed": 42, "arms": list(ARMS),
        "base_head": BASE_HEAD, "lambda_state": None, "formal_training_authorized": False,
        "budgets": {"GPU_compute": 0, "CUDA_initializations": 0, "real_model_constructions": 0,
                    "real_checkpoint_tensor_loads": 0, "real_model_forwards": 0,
                    "real_model_backwards": 0, "VAL_passes": 0, "TEST_accesses": 0},
        "train": {"epochs": epochs, "batches_per_epoch": 21, "updates": 21 * epochs,
                  "images": 668, "instances": 1266, "batch": 32, "tail": 28, "nbs": 32,
                  "imgsz": 640, "workers": 2, "optimizer": "AdamW", "lr0": .01, "lrf": .01,
                  "betas": [.937, .999], "eps": 1e-8, "weight_decay": .0005,
                  "clip_global_norm": 10., "warmup_iterations": 63, "warmup_bias_lr": .1,
                  "drop_last": False, "accumulate": 1, "cos_lr": False},
        "method": {"supcon_coefficient": 8., "supcon_temperature": .1,
                   "state_loss": "softplus(-(score_A-score_N))", "state_temperature": 1.,
                   "state_head": {"in_features": 64, "out_features": 1, "bias": False,
                                  "initializer": "torch_Linear_default_uniform_nonzero_independent_CPU_generator",
                                  "seed_namespace": "exp18:state-head:v1:42"},
                   "p3_layer": 16, "p3_stride": 8, "mask_stride": 4,
                   "raw_pooling": "native_overlap_2x2_coverage_weighted_then_L2_eps1e-12",
                   "bn_mode": "replay_a_connected", "bn_statistics": "whole_native_A_batch_and_spatial_biased_variance",
                   "bn_derivative": "R5_StatisticsGradient_unchanged", "N_prefix": [0, 16],
                   "checkpoint_prefix": False, "reduction": "mean_active_class_mean_retained_polygon",
                   "empty_head_policy": "grad_None_skip_AdamW_and_EMA_head", "head_ema_decay": "global_update_decay",
                   "A0": "synthetic_mechanism_reference_only", "B": "excluded", "C": "excluded"},
        "matching": {"same_batch": True, "same_class": True, "different_group": True,
                     "same_repair_family": True, "area_field": "native_polygon_area_divided_by_frame_area",
                     "area_ratio_max": 2., "p3_buckets": [1, 4], "cells_ratio_max": 2.,
                     "closed_permutation": True, "no_fixed_points": True, "unique_N": True,
                     "objective": "maximum_non_diagonal_assignment_then_seeded_integer_cost",
                     "seed": 42, "permutation_after_N_pooling": True},
        "future_validation": {"interval_epochs": 5, "events_per_arm": epochs // 5,
                              "native_passes_per_arm": 4 * epochs // 5,
                              "roles": ["MASTER", "EMA"], "modes": ["standard", "fixed_conf_0.25"],
                              "primary": "Last_FP32_MASTER_running_statistics_macro7_MaskAP50_95",
                              "prediction_dump": "same_pass_lossless_mask_and_matching_stream",
                              "executed": False},
        "storage": {"immutable_artifacts": True, "artifact_delete_allowed": False,
                    "artifact_overwrite_allowed": False, "formal_storage_gate": "PENDING"},
    }


def validate_protocol(value):
    require(isinstance(value, dict), "Protocol must be an object")
    expected = make_protocol(value.get("train", {}).get("epochs"))
    require(value == expected, "Frozen EXP18 scientific/authorization contract differs")
    return deepcopy(value)


def require_operation(operation, protocol):
    validate_protocol(protocol)
    require(operation not in FORBIDDEN_CPU_OPERATIONS,
            f"EXP18 CPU phase forbids {operation}: GPU Gate, lambda and formal authority are pending")
    require(operation in ("audit", "replay", "cpu-test"), "Unknown EXP18 operation")


def protocol_sha(value):
    validate_protocol(value)
    return digest(value)
