#!/usr/bin/env python3
"""Run the frozen read-only C1 audit against the V2 repair family names."""
from __future__ import annotations

import phaseC1_distribution_audit as audit


audit.FAMILIES = [
    "A_small_low_contrast",
    "B_diffuse_texture",
    "C_crack_like_structural_evolution",
    "D_boundary_evolution",
]


if __name__ == "__main__":
    raise SystemExit(audit.main())
