"""Create a source-only immutable deployment; never include data or weights."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from .common import require, sha256, write_json_exclusive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(["git", "ls-files", "tools", "configs", "AGENTS.md"], cwd=root).decode().splitlines()
    paths = {root / p for p in tracked if (root / p).is_file()}
    paths.update(p for d in (root / "tools/exp18", root / "configs/exp18") if d.exists()
                 for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    require(all(p.suffix not in (".pt", ".pth", ".safetensors") for p in paths), "Weight in source deployment")
    members = [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
               for p in sorted(paths)]
    archive = args.output / "source.tar"
    with archive.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w") as tar:
            for item in members:
                tar.add(root / item["path"], arcname=item["path"], recursive=False)
    receipt = write_json_exclusive(args.output / "source_manifest.json", {"files": members, "archive_sha256": sha256(archive)})
    print(json.dumps({"archive": str(archive), "manifest": receipt}))


if __name__ == "__main__": main()
