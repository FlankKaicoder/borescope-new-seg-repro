#!/usr/bin/env python3
"""Create the final G1C manifest after direct visual-review fields are recorded."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--g1b-raw", required=True, type=Path)
    parser.add_argument("--expected-raw-sha256", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "manifest.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    raw_after = sha256(args.g1b_raw)
    if raw_after != args.expected_raw_sha256:
        raise ValueError("G1B raw output changed during G1C")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != target and "__pycache__" not in path.parts:
            files.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "status": "G1C_POSTPROCESS_COMPLETE / PENDING_HUMAN_REVIEW",
        "scope": "corrosion23 CPU-only local postprocess and engineering audit",
        "result_roles": {"raw_output.png": "FIRST_CALL_RAW (external G1B input)", "outputs/aligned_output_640x480.png": "RESIZED_FOR_ALIGNMENT_ONLY", "outputs/composite_hard.png": "POSTPROCESS_ONLY", "outputs/composite_feather_inward_3px.png": "POSTPROCESS_ONLY"},
        "g1c_api_post_requests": 0,
        "gpu_calls": 0,
        "training_runs": 0,
        "g1b_raw_sha256_after": raw_after,
        "file_count_excluding_manifest": len(files),
        "files": files,
        "gate": {"train_pair_eligible": "NOT_GRANTED", "restoration_method_pass": "NOT_GRANTED", "human_review_required": True},
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "file_count": len(files), "manifest": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
