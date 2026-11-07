"""Restore a G3-B1 emergency JSON backup into a new validated review CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.exp13_cloud.g3.review_decisions import restore_backup_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-json", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore_backup_file(args.backup_json, args.baseline_csv, args.output_csv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
