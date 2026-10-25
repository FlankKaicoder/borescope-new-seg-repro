"""Verify an offline SDXL FP16 snapshot without loading model weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import struct
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_FILES = {
    "model_index.json",
    "README.md",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
}

REQUIRED_JSON = {
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "vae/config.json",
}

REQUIRED_SAFETENSORS = {
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
}

EXPECTED_CLASSES = {
    "model_index.json": "StableDiffusionXLInpaintPipeline",
    "scheduler/scheduler_config.json": "EulerDiscreteScheduler",
    "unet/config.json": "UNet2DConditionModel",
    "vae/config.json": "AutoencoderKL",
}

EXPECTED_ARCHITECTURES = {
    "text_encoder/config.json": "CLIPTextModel",
    "text_encoder_2/config.json": "CLIPTextModelWithProjection",
}

EXPECTED_MODEL_COMPONENTS = {
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> bool:
    posix = PurePosixPath(value.replace("\\", "/"))
    return not posix.is_absolute() and ".." not in posix.parts and bool(posix.parts)


def inspect_tar(path: Path) -> dict:
    members = []
    unsafe = []
    roots = set()
    expanded_bytes = 0
    with tarfile.open(path, "r:") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.parts:
                roots.add(pure.parts[0])
            reason = None
            if not safe_relative(member.name):
                reason = "absolute_or_traversal"
            elif member.issym() or member.islnk():
                reason = "link_member"
            elif not (member.isfile() or member.isdir()):
                reason = "unsupported_member_type"
            if reason:
                unsafe.append({"name": member.name, "reason": reason})
            if member.isfile():
                expanded_bytes += member.size
            members.append({"name": member.name, "type": "file" if member.isfile() else "directory" if member.isdir() else "other", "bytes": member.size})
    return {
        "tar_path": str(path),
        "tar_bytes": path.stat().st_size,
        "member_count": len(members),
        "file_count": sum(item["type"] == "file" for item in members),
        "directory_count": sum(item["type"] == "directory" for item in members),
        "expanded_file_bytes": expanded_bytes,
        "top_level_entries": sorted(roots),
        "unsafe_members": unsafe,
        "safe_to_extract": not unsafe and len(roots) == 1,
        "members": members,
    }


def read_manifest(path: Path) -> tuple[list[dict], list[str]]:
    errors = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Path", "Bytes", "SHA256"]:
            errors.append(f"unexpected_columns:{reader.fieldnames}")
        rows = list(reader)
    paths = [row.get("Path", "") for row in rows]
    duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate_paths:{duplicates}")
    for value in paths:
        if not safe_relative(value):
            errors.append(f"unsafe_path:{value}")
    return rows, errors


def inspect_safetensors(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("missing 8-byte safetensors header length")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 1 or header_bytes > min(size - 8, 100 << 20):
            raise ValueError(f"invalid header length {header_bytes}")
        header_raw = handle.read(header_bytes)
    header = json.loads(header_raw.decode("utf-8"))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError("no tensor entries")
    dtype_counts = Counter()
    offsets = []
    invalid = []
    for name, value in tensors.items():
        if not isinstance(value, dict):
            invalid.append(f"{name}:not_object")
            continue
        dtype = value.get("dtype")
        shape = value.get("shape")
        data_offsets = value.get("data_offsets")
        dtype_counts[str(dtype)] += 1
        if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            invalid.append(f"{name}:invalid_shape")
        if not isinstance(data_offsets, list) or len(data_offsets) != 2 or not all(isinstance(offset, int) for offset in data_offsets):
            invalid.append(f"{name}:invalid_offsets")
            continue
        start, end = data_offsets
        if start < 0 or end < start:
            invalid.append(f"{name}:offset_order")
        offsets.append((start, end, name))
    offsets.sort()
    for previous, current in zip(offsets, offsets[1:]):
        if current[0] < previous[1]:
            invalid.append(f"overlap:{previous[2]}:{current[2]}")
    payload_bytes = size - 8 - header_bytes
    max_end = max((end for _, end, _ in offsets), default=-1)
    if max_end != payload_bytes:
        invalid.append(f"payload_size:{max_end}!={payload_bytes}")
    with path.open("rb") as handle:
        pointer_prefix = handle.read(128)
    is_lfs_pointer = pointer_prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
    return {
        "path": str(path),
        "bytes": size,
        "header_bytes": header_bytes,
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "metadata": header.get("__metadata__"),
        "invalid_entries": invalid,
        "git_lfs_pointer": is_lfs_pointer,
        "valid": not invalid and not is_lfs_pointer,
    }


def verify_snapshot(root: Path, manifest: Path, output: Path) -> int:
    rows, manifest_errors = read_manifest(manifest)
    expected_paths = [row["Path"] for row in rows if row.get("Path")]
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    unsafe_symlinks = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink())
    comparison = []
    for row in rows:
        relative = row.get("Path", "")
        target = root / relative
        exists = target.is_file() and not target.is_symlink()
        actual_bytes = target.stat().st_size if exists else None
        actual_sha = sha256(target) if exists else None
        expected_bytes = int(row["Bytes"]) if row.get("Bytes", "").isdigit() else None
        expected_sha = row.get("SHA256", "").lower()
        size_match = exists and actual_bytes == expected_bytes
        sha_match = exists and actual_sha == expected_sha
        comparison.append({
            "Path": relative,
            "ExpectedBytes": expected_bytes,
            "ActualBytes": actual_bytes,
            "SizeMatch": size_match,
            "ExpectedSHA256": expected_sha,
            "ActualSHA256": actual_sha,
            "SHA256Match": sha_match,
            "Status": "PASS" if size_match and sha_match else "HOLD",
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sha256_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    missing = sorted(set(expected_paths) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(expected_paths))
    size_mismatches = [row["Path"] for row in comparison if not row["SizeMatch"]]
    sha_mismatches = [row["Path"] for row in comparison if not row["SHA256Match"]]
    file_report = {
        "status": "PASS" if not manifest_errors and len(rows) == 19 and set(expected_paths) == EXPECTED_FILES and not missing and not unexpected and not size_mismatches and not sha_mismatches and not unsafe_symlinks else "HOLD",
        "manifest_path": str(manifest),
        "manifest_sha256": sha256(manifest),
        "model_root": str(root),
        "manifest_errors": manifest_errors,
        "manifest_matches_registered_19_file_set": set(expected_paths) == EXPECTED_FILES,
        "files_expected": len(rows),
        "files_found": len(set(actual_paths) & set(expected_paths)),
        "files_verified": sum(row["Status"] == "PASS" for row in comparison),
        "size_mismatch": len(size_mismatches),
        "sha256_mismatch": len(sha_mismatches),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "unsafe_symlinks": unsafe_symlinks,
        "total_verified_bytes": sum(row["ActualBytes"] or 0 for row in comparison if row["Status"] == "PASS"),
    }
    dump(output / "model_file_verification.json", file_report)

    json_results = {}
    json_errors = []
    for relative in sorted(REQUIRED_JSON):
        try:
            payload = json.loads((root / relative).read_text(encoding="utf-8"))
            class_name = payload.get("_class_name") if isinstance(payload, dict) else None
            architecture = payload.get("architectures") if isinstance(payload, dict) else None
            expected_class = EXPECTED_CLASSES.get(relative)
            expected_architecture = EXPECTED_ARCHITECTURES.get(relative)
            class_match = expected_class is None or class_name == expected_class
            architecture_match = expected_architecture is None or (isinstance(architecture, list) and expected_architecture in architecture)
            json_results[relative] = {"valid_json": True, "class_name": class_name, "expected_class": expected_class, "class_match": class_match, "architectures": architecture, "expected_architecture": expected_architecture, "architecture_match": architecture_match}
            if not class_match:
                json_errors.append(f"{relative}:class:{class_name}")
            if not architecture_match:
                json_errors.append(f"{relative}:architecture:{architecture}")
        except Exception as error:
            json_results[relative] = {"valid_json": False, "error": f"{type(error).__name__}: {error}"}
            json_errors.append(relative)
    tensor_results = {}
    tensor_errors = []
    for relative in sorted(REQUIRED_SAFETENSORS):
        try:
            result = inspect_safetensors(root / relative)
            tensor_results[relative] = result
            if not result["valid"]:
                tensor_errors.append(relative)
        except Exception as error:
            tensor_results[relative] = {"path": str(root / relative), "valid": False, "error": f"{type(error).__name__}: {error}"}
            tensor_errors.append(relative)
    model_index_payload = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    component_checks = {
        name: {"expected": expected, "actual": model_index_payload.get(name), "match": model_index_payload.get(name) == expected}
        for name, expected in EXPECTED_MODEL_COMPONENTS.items()
    }
    component_errors = [name for name, check in component_checks.items() if not check["match"]]
    model_index = json_results.get("model_index.json", {})
    structure = {
        "status": "PASS" if file_report["status"] == "PASS" and not json_errors and not tensor_errors and not component_errors else "HOLD",
        "required_file_set_complete": set(expected_paths) == EXPECTED_FILES and not missing,
        "model_index_pipeline_class": model_index.get("class_name"),
        "expected_pipeline_class": "StableDiffusionXLInpaintPipeline",
        "model_index_components": component_checks,
        "json_configs": json_results,
        "safetensors": tensor_results,
        "json_errors": json_errors,
        "safetensors_errors": tensor_errors,
        "component_errors": component_errors,
        "from_pretrained_called": False,
        "gpu_model_load": False,
        "inference_calls": 0,
    }
    dump(output / "model_structure_validation.json", structure)
    return 0 if file_report["status"] == "PASS" and structure["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tar:
        report = inspect_tar(args.tar)
        dump(args.output / "tar_safety_validation.json", report)
        print(json.dumps({key: report[key] for key in ("member_count", "file_count", "expanded_file_bytes", "top_level_entries", "unsafe_members", "safe_to_extract")}, indent=2))
        if not report["safe_to_extract"]:
            return 2
    if args.model_root or args.manifest:
        if not args.model_root or not args.manifest:
            parser.error("--model-root and --manifest must be used together")
        return verify_snapshot(args.model_root.resolve(), args.manifest.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
