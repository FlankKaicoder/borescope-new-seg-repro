#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    nvidia = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout.strip()
    lines = [
        "# Training environment report",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "This is the isolated project training environment. The original Exp00 audit environment remains in `environment_report.md` and `pip_freeze.txt`.",
        "",
        f"- Virtual environment: `{sys.prefix}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- GPU visibility: `{nvidia}`",
        f"- Overall smoke status: **{smoke.get('status')}**",
        "",
        "## Imports",
        "",
        "| Package/import | Version |",
        "|---|---|",
    ]
    for name, version in smoke["imports"].items():
        lines.append(f"| `{name}` | `{version}` |")
    lines.extend(["", "## CUDA smoke", "", "```json", json.dumps(smoke["cuda"], ensure_ascii=False, indent=2), "```", "", "## Ultralytics model-load smoke", "", "```json", json.dumps(smoke["ultralytics_model_load"], ensure_ascii=False, indent=2), "```", "", "## OpenCV decode smoke", "", "```json", json.dumps(smoke["opencv_decode"], ensure_ascii=False, indent=2), "```", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
