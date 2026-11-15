"""Validate the preregistered E0/E1-FULL protocol and its execution identity."""

import hashlib
import json
from pathlib import Path

from tools.exp14.a11_release_audit import FROZEN_SHA256, verify
from tools.exp14.audit_admission import EXPECTED_MANIFEST_SHA256, require, sha256_file
from tools.exp14.freeze_core import COCO_SHA256, training_protocol
from tools.exp14.label_plan import CLASSES


STAGE = "EXP14_E0_E1_FULL_FORMAL_V1"
STATUS = "EXP14_FORMAL_PROTOCOL_READY_PENDING_CHATGPT_GATE"
DEPLOY_SHA = "219a018ecb6b0c21a75b1e869dba172371a91b429ec3b7246ae40cf87af6bb57"
BASE_COMMIT = "5442a0cb28548b76a64dcdc568476e3660ce91c5"
SIZES = (0.0020196759259259256, 0.007364969135802469, 0.029209280303030303)

RUNTIME_CODE = (
    "tools/exp14/a11_full_protocol.py", "tools/exp14/a11_full_train.py", "tools/exp14/a11_full_numerics.py",
    "tools/exp14/a11_full_eval.py", "tools/exp14/a11_smoke_train.py", "tools/exp14/a11_smoke_numerics.py",
    "tools/exp14/foreground_audit.py", "tools/exp14/evaluate_formal_e0_e1.py", "tools/exp14/a10_dataset_entries.py",
    "tools/exp14/a11_release_audit.py", "tools/exp14/a11_loader.py", "tools/exp14/freeze_core.py",
    "tools/exp14/audit_admission.py", "tools/exp14/label_plan.py", "tools/exp14/formal_e0_e1.py",
    "tools/exp14/smoke_trainer.py", "tools/exp14/revised_amp_smoke.py", "tools/exp14/diagnose_amp_first_batch.py",
    "tools/exp13_cloud/g3/g3_batch_executor.py", "tools/exp13_cloud/g3/verify_train_export.py",
)


def arm_budget(original, repair, raw_instances, epochs, batch, warmup_epochs):
    images = original + repair
    batches = (images + batch - 1) // batch
    return {"images": images, "original_images": original, "repair_images": repair, "raw_instances": raw_instances,
            "batches_per_epoch": batches, "warmup_iterations": round(warmup_epochs * batches),
            "effective_updates": epochs * batches, "tail_batch": images % batch or batch,
            "original_exposures": epochs * original, "repair_exposures": epochs * repair,
            "raw_instance_exposures": epochs * raw_instances}


def protocol_spec():
    prior = training_protocol(EXPECTED_MANIFEST_SHA256, None)
    hyperkeys = ("seed", "epochs", "imgsz", "batch", "nbs", "accumulate", "optimizer", "lr0", "lrf", "cos_lr",
                 "warmup_epochs", "warmup_bias_lr", "warmup_momentum", "momentum", "weight_decay", "amp", "deterministic")
    augment = ("mosaic", "close_mosaic", "mixup", "cutmix", "copy_paste", "rect", "multi_scale", "degrees",
               "translate", "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "bgr")
    train = {**{key: prior[key] for key in hyperkeys}, "augmentation": {key: prior[key] for key in augment},
             "drop_last": False, "workers": 2, "amp_initial_scale": 1.0, "amp_growth_interval": 2000,
             "amp_growth_factor": 2.0, "amp_backoff_factor": 0.5, "gradient_clip_l2_max": 10.0,
             "checkpoint_interval_epochs": 5, "patience": 0, "compile": False,
             "sampling": "SEEDED_NATIVE_RANDOM_SAMPLER_WITHOUT_REPLACEMENT_PER_EPOCH",
             "lr_schedule": "NATIVE_LINEAR_PER_EPOCH_WITH_NATIVE_ITERATION_WARMUP",
             "skip_policy": "STOP_ON_FIRST_NONFINITE_OR_NATIVE_SKIP",
             "validation_during_training": False}
    args = (train["epochs"], train["batch"], train["warmup_epochs"])
    return {
        "schema": STAGE, "status": STATUS, "base_commit": BASE_COMMIT,
        "classes": list(CLASSES),
        "identities": {"dataset_content_sha256": FROZEN_SHA256, "frozen_split_sha256": EXPECTED_MANIFEST_SHA256,
                       "deployment_manifest_sha256": DEPLOY_SHA, "coco_weight_sha256": COCO_SHA256,
                       "coco_weight_name": "yolo11n-seg.pt", "train_image_label_files": 2094,
                       "runtime_release_files": 2101},
        "runtime": {"gpu": "NVIDIA GeForce RTX 3080 Ti", "cuda_devices": 1, "torch": "2.8.0+cu128",
                    "torchvision": "0.23.0+cu128", "cuda": "12.8", "ultralytics": "8.4.117", "python": "3.12.3"},
        "training": train,
        "arms": {"e0": arm_budget(668, 0, 1266, *args), "e1": arm_budget(668, 379, 1266, *args)},
        "evaluation": {"split": "val", "checkpoint": "epoch100_last_master_fp32", "imgsz": 640,
                       "conf": 0.001, "nms_iou": 0.70, "max_det": 300, "fixed_conf": 0.25, "match_iou": 0.50,
                       "size_thresholds": list(SIZES), "gt_size_definition": "native_rasterized_polygon_area_over_full_image_area",
                       "matching": "NATIVE_ULTRALYTICS_CLASS_AWARE_IOU_GREEDY_ONE_TO_ONE",
                       "test_model_inference": False},
        "optional_controls": {"e0_update": {"status": "DESIGN_ONLY", "effective_updates": 3300, "full_original_epochs": 157,
                                         "partial_batches": 3, "original_exposures": 104972,
                                         "lr_schedule_basis": "E1_EQUIVALENT_UPDATE_INDEX",
                                         "warmup_iterations": 99},
                              "paired_seed_plan": [42, 142, 242], "additional_seeds": "SEPARATE_GATE"},
    }


def load_protocol(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    require(doc == protocol_spec(), "Formal protocol differs from the preregistered contract")
    return doc


def protocol_sha256(path):
    return sha256_file(Path(path))


def source_sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def identity_spec(protocol_path, root):
    load_protocol(protocol_path)
    return {"schema": "exp14_full_execution_identity_v1", "protocol_file_sha256": protocol_sha256(protocol_path),
            "source_encoding": "SHA256_WITH_CRLF_NORMALIZED_TO_LF",
            "files": {name: source_sha256(root / name) for name in RUNTIME_CODE}}


def validate_identity(identity_path, protocol_path):
    identity = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    require(identity == identity_spec(protocol_path, root), "Formal runtime source/task identity differs")
    return identity


def authorize_training(gate_path, protocol_path, identity_path, arm):
    require(arm in {"e0", "e1"}, "Only the two primary arms have a formal execution path")
    validate_identity(identity_path, protocol_path)
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    required = {"stage": "EXP14_E0_E1_FULL_TRAIN_AUTHORIZATION", "decision": "PASS",
                "protocol_sha256": protocol_sha256(protocol_path), "identity_sha256": sha256_file(Path(identity_path)),
                "authorized_arms": ["e0", "e1"]}
    require(gate == required, "Separate formal execution authorization does not match this paired protocol")
    return gate


def verify_inputs(protocol, release, weight, deployment_manifest):
    identities = protocol["identities"]
    release = Path(release).resolve(strict=True)
    require(sha256_file(Path(weight)) == identities["coco_weight_sha256"], "Frozen COCO initialization weights differ")
    require(sha256_file(release / "configs/frozen_split_manifest.csv") == identities["frozen_split_sha256"],
            "Frozen membership manifest differs")
    report = verify(release, Path(deployment_manifest), identities["deployment_manifest_sha256"])
    require(report["dataset_content_sha256"] == identities["dataset_content_sha256"]
            and report["dataset_file_count"] == identities["train_image_label_files"]
            and report["deployment_file_count"] == report["readonly_files"] == identities["runtime_release_files"]
            and report["multiple_hardlink_files"] == 0,
            "Formal TRAIN release is not the independent read-only deployment")
    return {"dataset_content_sha256": report["dataset_content_sha256"],
            "deployment_manifest_sha256": report["deployment_manifest_sha256"],
            "frozen_split_sha256": identities["frozen_split_sha256"],
            "coco_weight_sha256": identities["coco_weight_sha256"], "train_image_label_files": report["dataset_file_count"],
            "runtime_release_files": report["deployment_file_count"], "checked_train_arms": report["arms"],
            "val_test_file_bytes_read": False}
