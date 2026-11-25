"""Freeze authorized local scientific context without editing the old worktree."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from .common import digest, sha256, write_json_exclusive
from .protocol import BASE_HEAD, make_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    args.public_output.mkdir(parents=True, exist_ok=False)
    old = args.historical_worktree.resolve()
    selected = ("AGENTS.md", "docs/PROJECT_STATE.md", "docs/DECISION_LOG.md", "docs/latest_handoff.md",
                "docs/HISTORICAL_METHODS.md", "EXP18_NEXT_METHOD_DESIGN_REVIEW.md",
                "EXP18_P1_PREREGISTRATION_DRAFT.md", "STORAGE_MIGRATION_HANDOFF_20261125.md",
                "EXP17_R5_POSTHOC_ANALYSIS_HANDOFF.md",
                "results/exp17_r5_posthoc/20261125T055351Z_v1/EXECUTIVE_FINDINGS.md",
                "results/exp17_r5_posthoc/20261125T055351Z_v1/POSTHOC_REPORT_COMPLETE.md")
    raw_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=old).decode("utf-8")
    rows = []
    archive = args.output / "inherited_context.zip"
    with archive.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for name in selected:
                path = old / name
                data = path.read_bytes()
                bundle.writestr(name, data)
                rows.append({"path": name, "source": str(path), "bytes": len(data), "sha256": sha256(path),
                             "git_status": subprocess.check_output(["git", "status", "--porcelain=v1", "--", name], cwd=old).decode().strip()})
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            import hashlib
            assert hashlib.sha256(bundle.read(row["path"])).hexdigest() == row["sha256"]
    inherited = ["tools/exp17/engineering/bn.py", "tools/exp17/engineering/augmentation.py",
                 "tools/exp17/engineering/model.py", "tools/exp15/regions.py"]
    manifest = {"base_head": BASE_HEAD, "branch": "codex/exp18-cpu-engineering",
                "historical_worktree": str(old), "historical_status_before": raw_status,
                "files": rows, "context_archive_sha256": sha256(archive),
                "inherited_execution": [{"path": p, "sha256": sha256(root / p),
                                          "historical_sha256": sha256(old / p)} for p in inherited]}
    assert all(r["sha256"] == r["historical_sha256"] for r in manifest["inherited_execution"])
    write_json_exclusive(args.output / "context_manifest.json", manifest)
    write_json_exclusive(args.public_output / "inheritance.json", manifest)
    config_dir = root / "configs/exp18"
    config_dir.mkdir(parents=True, exist_ok=True)
    for horizon in (100, 200):
        protocol = make_protocol(horizon)
        protocol_path = config_dir / f"protocol_h{horizon}.json"
        write_json_exclusive(protocol_path, protocol)
        runtime = {"schema": "EXP18_CPU_RUNTIME_V1", "protocol": protocol_path.relative_to(root).as_posix(),
                   "protocol_sha256": digest(protocol), "protocol_file_sha256": sha256(protocol_path),
                   "horizon": horizon, "seed": 42, "device": "cpu", "formal_training_authorized": False,
                   "lambda_state": None, "historical_runtime_config": "/root/exp17_r5_gate_20261124_v5/runtime_config.json",
                   "historical_runtime_sha256": "1d90672b5ae86d2d2c6a64bf7326bdfefe35ae473b86c6ef579ee66f67b9b763",
                   "python": "/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python",
                   "data_memory_policy": {"workers": 2, "prefetch_factor": 1, "worker_buffer_images": 32,
                                          "decoded_asset_cache_MiB": 32, "materialize_N_batch": False,
                                          "parent_CPU_threads": 1, "native_tensor_hashes_must_match": True},
                   "output_policy": "new_EXP18_roots_only_immutable_artifacts_no_cleanup",
                   "GPU_calls": 0, "real_models": 0, "real_checkpoint_tensor_loads": 0,
                   "VAL_passes": 0, "TEST_accesses": 0}
        write_json_exclusive(config_dir / f"runtime_h{horizon}.json", runtime)
    assert subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=old).decode("utf-8") == raw_status
    print(json.dumps({"context": str(archive), "sha256": sha256(archive), "files": len(rows)}))


if __name__ == "__main__": main()
