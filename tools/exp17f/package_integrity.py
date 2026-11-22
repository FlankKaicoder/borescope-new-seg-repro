"""Read-only byte audit of the frozen offline package; no image decoding.

All detailed paths and manifests are private outputs. Existing input files are
opened read-only. New outputs use exclusive creation and cannot replace history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

CSV_SHA = "b0a5f768040539a6c2daaab18b1166f99995d4fcf7d0f968e10bec4de2e476fd"
QUEUE_SHA = "1bf154d3f0a565892fad12c5f7e80882424baeee34be289cf77911bb8b1ac363"
INDEX_SHA = "668c1a7b6848a462e10f14e80ecb4fb69dd6db962315e6ff4d28b2c74387c464"
ZIP_SHA = "29a4bee6c7fd40c20cf7fa22cc686c5a4631f368d8db56ca09faeb6984cd56fb"
PACKAGE_ID = "a0d977ae3dae0a148f0c3f128c113954f18a282ac70a5371d0cf90b352560351"
BASE = "a5d86a3d92b1cd6e5a974435c653e2fdd32f66b9"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def snapshot(path: Path, expected: str | None = None) -> dict:
    require(not path.is_symlink(), f"Symlink input forbidden: {path}")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f"Changed input: {path}")
    actual = digest.hexdigest()
    require(expected is None or actual == expected, f"SHA mismatch: {path}")
    return {"path": str(path), "bytes": after.st_size,
            "mtime_ns": str(after.st_mtime_ns), "sha256": actual}


def child(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    require(not parsed.is_absolute() and not PureWindowsPath(relative).drive and
            ":" not in relative and ".." not in parsed.parts and "\\" not in relative,
            "Unsafe package member")
    result = root.joinpath(*parsed.parts)
    require(result.resolve().is_relative_to(root.resolve()), "Package path escapes root")
    current = result
    while current != root:
        require(not current.is_symlink(), "Symlink package member")
        current = current.parent
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def audit(e_root: Path, csv_path: Path, output: Path, repo: Path) -> dict:
    started = time.perf_counter()
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip() == BASE, "Unexpected starting HEAD")
    protected: dict[str, dict] = {}

    def record(path: Path, expected: str | None = None) -> dict:
        item = snapshot(path, expected)
        require(str(path) not in protected or protected[str(path)] == item,
                f"Repeated input changed during audit: {path}")
        protected[str(path)] = item
        return item

    user = record(csv_path, CSV_SHA)
    copy = output / "inputs" / csv_path.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("rb") as source, copy.open("xb") as target:
        shutil.copyfileobj(source, target)
    record(copy, CSV_SHA)
    delivery_path = e_root / "EXP17_E_DELIVERY.json"
    record(delivery_path)
    delivery = read_json(delivery_path)
    for item in delivery["required_deliverables"].values():
        record(Path(item["path"]), item["sha256"])
        source = Path(item["source"])
        record(source if source.is_absolute() else repo / source, item["sha256"])
    record(Path(delivery["final_queue_original"]), QUEUE_SHA)
    review = Path(delivery["final_review_directory"])
    for name in ("review_bundle.json", "new_review_template.json", "new_review_template.csv",
                 "package_receipt.json", "file_manifest.json", "index.html"):
        record(review / name)
    receipt = read_json(review / "package_receipt.json")
    bundle = read_json(review / "review_bundle.json")
    require(receipt["package_id"] == bundle["package_id"] == PACKAGE_ID, "Package ID mismatch")
    require(receipt["queue_sha256"] == bundle["queue_sha256"] == QUEUE_SHA, "Queue mismatch")
    record(review / "review_bundle.json", receipt["bundle_sha256"])
    record(review / "index.html", INDEX_SHA)
    files = read_json(review / "file_manifest.json")
    total = 0
    media_stats = {}
    for relative, item in files.items():
        actual = snapshot(child(review, relative), item["sha256"])
        require(actual["bytes"] == item["bytes"], "Package member size mismatch")
        total += actual["bytes"]
        media_stats[relative] = actual
    require(len(files) == receipt["media_and_annotation_files"] == 3710, "Package file count")
    require(total == receipt["assets_bytes"], "Package byte count")
    archive = record(Path(delivery["portable_archive"]["path"]), ZIP_SHA)
    require(archive["bytes"] == delivery["portable_archive"]["bytes"], "ZIP size mismatch")
    with zipfile.ZipFile(archive["path"]) as zipped:
        members = zipped.infolist()
        require(len(members) == len({member.filename for member in members}), "Duplicate ZIP member")
        require(len(members) == 3717, "ZIP member count")
        # Whole ZIP SHA proves bytes; only metadata is read here, no extraction.
        for member in members:
            child(review, member.filename.rstrip("/"))
    mapping = Path(delivery["mapping_proofs_directory"])
    public_e = read_json(repo / "results/exp17e_a10_379_polygon_expansion/summary.json")
    for name in ("source_mappings.json", "polygon_mappings.json", "region_lineage_summary.json",
                 "bounded_region_lineage.json", "crop_region_lineage.json",
                 "mapping_summary.json", "verified_mapping_inputs.json"):
        record(mapping / name)
    mapping_summary = read_json(mapping / "mapping_summary.json")
    record(mapping / "source_mappings.json", mapping_summary["private_source_mappings_sha256"])
    record(mapping / "polygon_mappings.json", mapping_summary["private_polygon_mappings_sha256"])
    lineage = read_json(mapping / "region_lineage_summary.json")
    record(mapping / "bounded_region_lineage.json", lineage["bounded_private_sha256"])
    record(mapping / "crop_region_lineage.json", lineage["crop_private_sha256"])
    # Preserve exact SHA/stat identities of the previously frozen metadata.
    prior = read_json(e_root / "execution/start_identity.json")
    for path, item in prior["files"].items():
        record(Path(path), item["sha256"])
    for key in ("mapping_summary.json", "region_lineage_summary.json"):
        record(mapping / key, public_e["evidence_sha256"][key])
    record(repo / "results/exp17e_a10_379_polygon_expansion/polygon_recovery_summary_final.json",
           public_e["evidence_sha256"]["polygon_recovery_summary_final.json"])
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    excluded = {"docs/PROJECT_STATE.md", "docs/DECISION_LOG.md", "docs/experiment_index.md",
                "results/experiment_registry.csv", ".gitattributes"}
    for relative in tracked:
        if relative and relative not in excluded:
            record(repo / relative)
    result = {
        "schema": "EXP17_F_INPUT_AND_PACKAGE_INTEGRITY_V1", "overall_status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "audited_base_commit": BASE,
        "user_csv": {"original_path": user["path"], "copied_path": str(copy),
                     "sha256": CSV_SHA, "bytes": user["bytes"]},
        "review_package": {"package_id": PACKAGE_ID, "queue_sha256": QUEUE_SHA,
                           "index_sha256": INDEX_SHA, "bundle_sha256": receipt["bundle_sha256"],
                           "media_files_verified": len(files), "media_bytes_verified": total,
                           "zip_sha256": ZIP_SHA, "zip_bytes": archive["bytes"],
                           "zip_members": len(members)},
        "protected_inputs": list(protected.values()), "media_input_stats": media_stats,
        "scope": "Frozen S379 private review package and prior metadata; whole-file bytes only",
        "elapsed_seconds": time.perf_counter() - started,
        "image_decodes": 0, "model_forwards": 0, "new_registration": 0,
        "optimizer_updates": 0, "val_test_image_reads": 0,
    }
    write_new(output / "execution/input_and_package_integrity.json", result)
    return result


def postcheck(receipt: Path, output: Path) -> dict:
    old = read_json(receipt)
    for item in old["protected_inputs"]:
        actual = snapshot(Path(item["path"]), item["sha256"])
        require(actual == item, f"Protected input stat changed: {item['path']}")
    # The media were already hashed in the audit. Check their immutable size/time
    # identities after reconciliation rather than unnecessarily rehashing all images.
    for item in old["media_input_stats"].values():
        path = Path(item["path"])
        info = path.stat()
        require(not path.is_symlink() and info.st_size == item["bytes"] and
                str(info.st_mtime_ns) == item["mtime_ns"], "Review media stat changed")
    result = {"schema": "EXP17_F_PROTECTION_POSTCHECK_V1", "status": "PASS",
              "integrity_receipt_sha256": snapshot(receipt)["sha256"],
              "protected_files_unchanged": len(old["protected_inputs"]),
              "review_media_stat_unchanged": len(old["media_input_stats"]),
              "mtime_ns_compared_as_exact_decimal_strings": True}
    write_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-root", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--postcheck", type=Path)
    args = parser.parse_args()
    if args.postcheck:
        result = postcheck(args.postcheck, args.out)
    else:
        parser.error("--e-root and --csv required") if not (args.e_root and args.csv) else None
        result = audit(args.e_root, args.csv, args.out, args.repo)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"protected_inputs", "media_input_stats"}}, ensure_ascii=True))
