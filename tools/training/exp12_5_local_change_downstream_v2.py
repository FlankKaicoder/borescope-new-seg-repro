#!/usr/bin/env python3
"""Exp12.5 launcher correction: normalize equivalent relative/absolute YAML paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.training import exp12_5_local_change_downstream as implementation


def fair_args(requested: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in requested.items() if key != "project"}
    data_path = Path(normalized["data"]).resolve()
    normalized["data"] = {
        "resolved": str(data_path),
        "sha256": implementation.base.file_sha256(data_path),
    }
    return normalized


implementation.fair_args = fair_args


if __name__ == "__main__":
    raise SystemExit(implementation.run(implementation.parse_args()))
