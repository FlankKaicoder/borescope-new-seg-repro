"""Create a backed-up, repaired review HTML at a new private path."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tools.exp13_cloud.g3.source_review_ui import repair_html_document


def repair_page(source: Path, backup: Path, output: Path, source_count: int) -> dict[str, str]:
    if source.resolve() == backup.resolve() or source.resolve() == output.resolve() or backup.resolve() == output.resolve():
        raise ValueError("source, backup, and repaired output must be separate paths")
    if backup.exists() or output.exists():
        raise FileExistsError("refusing to overwrite an existing HTML backup or repaired page")
    original = source.read_bytes()
    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("xb") as handle:
        handle.write(original)
        handle.flush()
    if backup.read_bytes() != original:
        raise RuntimeError("original HTML backup is not byte-identical")
    repaired_text = repair_html_document(original.decode("utf-8"), source_count)
    repaired = repaired_text.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(repaired)
        handle.flush()
    return {
        "backup": str(backup), "backup_sha256": hashlib.sha256(original).hexdigest(),
        "repaired_page": str(output), "repaired_sha256": hashlib.sha256(repaired).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, required=True)
    args = parser.parse_args()
    print(repair_page(args.source, args.backup, args.output, args.source_count))


if __name__ == "__main__":
    main()
