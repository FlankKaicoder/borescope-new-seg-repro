#!/usr/bin/env python3
"""Load one or more Ultralytics checkpoints and report their hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("checkpoints", type=Path, nargs="+")
args = p.parse_args()
rows = []
for path in args.checkpoints:
    YOLO(str(path))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": h, "load_status": "PASS"})
print(json.dumps(rows, indent=2))
