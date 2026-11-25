"""Small standard-library-only identity and immutable output helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == raw, "Output verification failed: " + str(path))
    return {"path": str(path), "bytes": len(raw), "sha256": sha256(path)}


def source_manifest(root):
    root = Path(root)
    names = [p for directory in (root / "tools/exp18", root / "configs/exp18")
             if directory.exists() for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".json")]
    # Actual inherited execution dependencies, not only their containing Git commit.
    names += [root / name for name in ("tools/exp15/regions.py", "tools/exp17/engineering/bn.py", "tools/exp17/engineering/model.py",
              "tools/exp17/engineering/augmentation.py", "tools/exp17/engineering/data.py",
              "tools/exp14/a11_smoke_train.py", "tools/exp14/a11_full_train.py")]
    rows = [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(set(names))]
    return {"files": rows, "sha256": digest(rows)}
