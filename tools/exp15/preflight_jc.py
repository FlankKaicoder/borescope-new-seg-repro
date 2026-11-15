"""Verify the audited Exp15 J-B identities before issuing the private J-C authorization."""

import argparse
import importlib.util
import platform
import shutil
from pathlib import Path

import torch
import torchvision
import ultralytics

from tools.exp14.audit_admission import require, sha256_file
from tools.exp15 import ja, jb

APPROVED_HEAD = "2aba71a929d2cb593d0e32377ee24811e0e4dc8d"
APPROVED_PROTOCOL_SHA = "b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca"
STAGE = "EXP15_J0_J1_FORMAL_AUTHORIZATION"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "code-archive", "archive-sha", "transport-head", "ja-root", "ja-public",
                 "calibration", "smoke", "release", "weight", "framework-pin", "output", "authorization"):
        parser.add_argument(f"--{name}", required=True,
                            type=str if name in ("archive-sha", "transport-head") else Path)
    a = parser.parse_args()
    require(a.transport_head == APPROVED_HEAD and sha256_file(a.code_archive) == a.archive_sha
            and sha256_file(a.protocol) == APPROVED_PROTOCOL_SHA, "Audited Git transfer or protocol identity differs")
    spec = jb.load(a.protocol)
    identity = {"files": jb.source_identity(), "coco_sha256": sha256_file(a.weight),
                "frozen_split_sha256": sha256_file(a.release / "configs" / "frozen_split_manifest.csv"),
                "ja_audited_summary_sha256": sha256_file(a.ja_public)}
    require(spec == jb.formal_protocol(identity, sha256_file(a.calibration), sha256_file(a.smoke)),
            "Code, data, initialization or J-B evidence differs")
    gate, smoke = jb.load(a.calibration), jb.load(a.smoke)
    require(gate["status"] == "PASS_TRAIN_CALIBRATION" and gate["optimizer_attempts"] == 0
            and smoke["status"] == "PASS_JB_BOUNDED_SMOKE" and smoke["effective_updates"] == 70
            and smoke["amp_skips"] == 0 and gate["identity"] == smoke["identity"] == identity,
            "J-B authorization evidence differs")
    ja_history = jb.load(a.ja_public)
    _, checkpoints = jb.read_reference(a.ja_root, ja_history)
    require(len(checkpoints) == 2, "Prior run evidence differs")
    runtime = {"gpu": torch.cuda.get_device_name(0), "cuda_devices": torch.cuda.device_count(),
               "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
               "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda}
    require(runtime == spec["runtime"], "Live AutoDL environment differs from the audited protocol")
    package_root = Path(ultralytics.__file__).resolve().parent.parent
    pinned = jb.load(a.framework_pin)
    require(pinned and all(sha256_file(package_root / path) == expected for path, expected in pinned.items()),
            "Installed Ultralytics source differs from the prior verified pin")
    observed = dict(pinned)
    for name in ("ultralytics.data.augment", "ultralytics.engine.trainer", "ultralytics.utils.loss",
                 "ultralytics.utils.torch_utils"):
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        observed[origin.relative_to(package_root).as_posix()] = sha256_file(origin)
    train_groups = ja.checked_train(a.release.resolve(strict=True))
    aj = {"stage": STAGE, "decision": "PASS", "protocol_sha256": APPROVED_PROTOCOL_SHA,
          "authorized_arms": ["j0", "j1"]}
    ja.record_json(a.authorization, aj)
    jb.verify_authorization(a.protocol, a.authorization, identity)
    ja.record_json(a.output, {
        "status": "PASS_JC_ZERO_UPDATE_PREFLIGHT", "approved_head": APPROVED_HEAD, "protocol_sha256": APPROVED_PROTOCOL_SHA,
        "code_archive_sha256": a.archive_sha, "authorization_sha256": sha256_file(a.authorization),
        "calibration_report_sha256": spec["calibration_report_sha256"], "smoke_report_sha256": spec["smoke_report_sha256"],
        "native_initial_sha256": gate["native_initial_sha256"], "identity": identity, "runtime": runtime,
        "framework_source_sha256": observed, "train_images": len(train_groups), "train_groups": len(set(train_groups.values())),
        "train_images_labels_hash_verified": 2 * len(train_groups),
        "working_disk_available_bytes": shutil.disk_usage(a.output.parent).free, "updates": 0,
        "val_inference": 0, "test_inference": 0,
    })


if __name__ == "__main__":
    main()
