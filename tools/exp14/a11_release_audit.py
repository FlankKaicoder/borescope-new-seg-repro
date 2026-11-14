"""Create and verify the private A11-0 transfer inventory for the A10 runtime subset."""

import argparse
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from tools.exp13_cloud.g3.g3_batch_executor import require, write_json_exclusive
from tools.exp14.a10_dataset_entries import (
    ARM_CONFIGS, EVAL_ROOT, arm_counts, load_frozen_split, reference_descriptor,
    verify_inventory, verify_paths, verify_references,
)
from tools.exp14.a10_freeze import content_receipt
from tools.exp14.full392_recovery import csv_rows
from tools.exp14.prepare_large_scale_repair import hash_file, load_json


FROZEN_SHA256 = "6c899ab76fab35832ae38cf8fd67bd0a8a5439ee5bafd48670ce2c37df3321f9"
SOURCE_COMMIT = "e5337cf1d26990913d5fd00aede69d7ec5d0163f"
CONFIGS = {"configs/frozen_split_manifest.csv", "configs/train_entry_files.json"}
CONFIGS.update(f"configs/{name}" for paths in ARM_CONFIGS.values() for name in paths)
MIRRORED_RELEASE = "release/sha256_manifest.json"
ARMS = {
    "e0": {"train_images": 668, "train_labels": 668, "original_images": 668,
           "explicit_empty_labels": 0, "segmentation_instances": 1266},
    "e1": {"train_images": 1047, "train_labels": 1047, "original_images": 668,
           "explicit_empty_labels": 379, "segmentation_instances": 1266},
}


def frozen_members(release, exclusions):
    configs = release / "configs"
    split = load_frozen_split(configs / "frozen_split_manifest.csv")
    data = load_json(configs / "train_entry_files.json")
    verify_inventory(data, split)
    verify_paths(configs, release, data, PurePosixPath(EVAL_ROOT))
    base, derived = data["baseline"], data["derived"]
    expected = load_json(release / MIRRORED_RELEASE)
    recomputed = content_receipt(release)
    require(recomputed == expected and expected["sha256"] == FROZEN_SHA256
            and expected["file_count"] == 2094, "A10 frozen dataset content or member count differs")
    pairs = {"e0": arm_counts(base), "e1": arm_counts(base + derived)}
    require(pairs == ARMS and len(exclusions) == len(set(exclusions)) == 13
            and set(exclusions) <= {row["source_id"] for row in base}
            and not (set(exclusions) & {row["source_id"] for row in derived}),
            "TRAIN arm counts or quarantined-source membership differs")
    require(len({row["group_id"] for row in derived}) == 359, "Frozen derived source groups differ")
    require({p.relative_to(release).as_posix() for p in configs.iterdir() if p.is_file()} == CONFIGS,
            "Frozen config membership differs")
    return expected["files"], split, pairs


def build(source, manifest):
    receipt = load_json(source / "release/freeze_receipt.json")
    require(receipt["commit"] == receipt["remote_head"] == SOURCE_COMMIT
            and receipt["freeze_content_sha256"] == FROZEN_SHA256, "A10 freeze receipt differs")
    for name in CONFIGS | {MIRRORED_RELEASE, "registry/a_freeze_exceptions.csv"}:
        hash_file(source / name, receipt["private_file_sha256"][name])
    raw_exceptions = csv_rows(source / "registry/a_freeze_exceptions.csv")
    exclusions = sorted(row["source_id"] for row in raw_exceptions)
    members, _, arms = frozen_members(source, exclusions)
    members.update({path: receipt["private_file_sha256"][path] for path in CONFIGS | {MIRRORED_RELEASE}})
    inventory = {name: {"sha256": sha, "bytes": (source / name).stat().st_size} for name, sha in members.items()}
    report = {"source_commit": SOURCE_COMMIT, "dataset_content_sha256": FROZEN_SHA256, "source_arms": arms,
              "quarantined_original_source_ids": exclusions, "files": inventory,
              "outside_runtime_subset": "A10 history, human evidence and source paths remain at the Windows source."}
    write_json_exclusive(manifest, report)
    return {"deployment_manifest_sha256": hash_file(manifest), "file_count": len(inventory),
            "files_bytes": sum(row["bytes"] for row in inventory.values())}


def copy_original(source, target, expected, release):
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.is_symlink() and target.resolve().is_relative_to(release), "Baseline target escapes release")
    if target.exists():
        require(target.stat().st_size == expected["bytes"], f"Existing baseline size differs: {target.name}")
        hash_file(target, expected["sha256"])
        return False
    require(source.stat().st_size == expected["bytes"], f"Original baseline size differs: {source.name}")
    # A verified temporary inode is published exclusively, so an interrupted copy never
    # becomes a finished target and concurrent retries cannot overwrite an existing file.
    with TemporaryDirectory(dir=release.parent, prefix=".a11-baseline-") as scratch:
        staged_path = Path(scratch) / "asset"
        with source.open("rb") as incoming, staged_path.open("xb") as staged:
            shutil.copyfileobj(incoming, staged)
            staged.flush()
            os.fsync(staged.fileno())
        hash_file(staged_path, expected["sha256"])
        if os.name != "nt":
            os.chmod(staged_path, 0o444)
        try:
            os.link(staged_path, target)
        except FileExistsError:
            require(target.stat().st_size == expected["bytes"], f"Concurrent baseline size differs: {target.name}")
            hash_file(target, expected["sha256"])
            return False
    return True


def hydrate_base(original, release, manifest, manifest_sha):
    hash_file(manifest, manifest_sha)
    deployment = load_json(manifest)
    require(deployment["source_commit"] == SOURCE_COMMIT and deployment["dataset_content_sha256"] == FROZEN_SHA256,
            "Deployment manifest refers to another frozen release")
    files = deployment["files"]
    for name in CONFIGS:
        hash_file(release / name, files[name]["sha256"])
    manifest_name = "configs/frozen_split_manifest.csv"
    hash_file(original / "split_manifest.csv", files[manifest_name]["sha256"])
    split = load_frozen_split(release / manifest_name)
    inventory = load_json(release / "configs/train_entry_files.json")
    verify_inventory(inventory, split)
    copied, reused, copied_bytes = 0, 0, 0
    for row in inventory["baseline"]:
        frozen = split["train"][row["source_id"]]
        filenames = (("images", frozen["stem"] + frozen["image_suffix"], "image"),
                     ("labels", frozen["stem"] + ".txt", "label"))
        for folder, basename, kind in filenames:
            name = f"dataset/{row[kind + '_path']}"
            expected = files[name]
            require(expected["sha256"] == row[kind + "_sha256"], "Deployment baseline member hash differs")
            if copy_original(original / folder / "train" / basename, release / name, expected, release):
                copied += 1
                copied_bytes += expected["bytes"]
            else:
                reused += 1
    return {"copied_files": copied, "copied_bytes": copied_bytes, "already_verified_files": reused}


def verify(release, manifest, manifest_sha, eval_root=None):
    hash_file(manifest, manifest_sha)
    deployment = load_json(manifest)
    require(deployment["source_commit"] == SOURCE_COMMIT and deployment["dataset_content_sha256"] == FROZEN_SHA256,
            "Deployment manifest refers to another frozen release")
    dataset, sealed, arms = frozen_members(release, deployment["quarantined_original_source_ids"])
    files = deployment["files"]
    require(files.keys() == dataset.keys() | CONFIGS | {MIRRORED_RELEASE}, "Deployment manifest member list differs")
    actual = {p.relative_to(release).as_posix() for folder in ("dataset", "configs", "release")
              for p in (release / folder).rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == files.keys(), "Runtime trees contain an unexpected file or symlink")
    readonly_files, multiple_hardlink_files = 0, 0
    for name, expected in files.items():
        path = release / name
        metadata = path.stat()
        require(stat.S_ISREG(metadata.st_mode) and metadata.st_size == expected["bytes"],
                f"Deployed file size or type differs for {name}")
        require(not path.is_symlink() and path.resolve().is_relative_to(release), f"Deployed file escapes release for {name}")
        if name in dataset:
            require(dataset[name] == expected["sha256"], f"Deployment manifest content hash differs for {name}")
        else:
            hash_file(path, expected["sha256"])
        readonly_files += not bool(metadata.st_mode & 0o222)
        multiple_hardlink_files += metadata.st_nlink > 1
    evaluation = reference_descriptor(sealed, PurePosixPath(EVAL_ROOT))
    if eval_root:
        require(Path(EVAL_ROOT).is_dir() and eval_root.resolve(strict=True) == Path(EVAL_ROOT).resolve(strict=True),
                "Resolved evaluation files differ from the immutable YAML references")
        for split, count in verify_references(eval_root, sealed).items():
            records = sealed[split].values()
            evaluation[split].update(
                file_hashes_verified_this_run=count,
                image_hashes_verified_this_run=len(records),
                label_hashes_verified_this_run=len(records),
                image_bytes=sum((eval_root / "images" / split / (r["stem"] + r["image_suffix"])).stat().st_size for r in records),
                label_bytes=sum((eval_root / "labels" / split / (r["stem"] + ".txt")).stat().st_size for r in records),
            )
    return {"dataset_content_sha256": FROZEN_SHA256, "dataset_file_count": len(dataset),
            "deployment_manifest_sha256": manifest_sha, "deployment_file_count": len(files),
            "deployment_file_bytes": sum(row["bytes"] for row in files.values()), "readonly_files": readonly_files,
            "multiple_hardlink_files": multiple_hardlink_files, "arms": arms, "derived_source_groups": 359,
            "quarantined_originals_retained": 13, "quarantined_repairs_included": 0, "held_out": evaluation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("build")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    copy = actions.add_parser("hydrate-base")
    copy.add_argument("--original-root", type=Path, required=True)
    copy.add_argument("--release", type=Path, required=True)
    copy.add_argument("--manifest", type=Path, required=True)
    copy.add_argument("--manifest-sha256", required=True)
    check = actions.add_parser("verify")
    check.add_argument("--release", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--manifest-sha256", required=True)
    check.add_argument("--eval-root", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        result = build(args.source.resolve(strict=True), args.manifest)
    elif args.action == "hydrate-base":
        result = hydrate_base(args.original_root.resolve(strict=True), args.release.resolve(strict=True),
                              args.manifest, args.manifest_sha256)
    else:
        result = verify(args.release.resolve(strict=True), args.manifest, args.manifest_sha256, args.eval_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
