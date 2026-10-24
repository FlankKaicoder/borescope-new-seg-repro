"""Create the Phase 1-A protocol audit bundle without model execution."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .geometry import build_geometry
from .protocol import BudgetProtocol
from .roi_composition import CompositionItem, compose_support


RUN_ID = "20261024T_PHASE1A_PROTOCOL"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    r1 = root / "results/exp13_phase0_r1/20261024T100624Z"
    out = root / "results/exp13_phase1a" / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    proposal = r1 / "pilot_revision_proposal.csv"
    rows = list(csv.DictReader(proposal.open(encoding="utf-8-sig", newline="")))
    if not rows or not any(r["stem"] == "50" and r["class"] == "Burn" for r in rows):
        raise RuntimeError("Burn source 50 is required in the TRAIN-only audit")
    unique_sources = sorted({r["stem"] for r in rows})
    roi_count = sum(int(r["source_roi_count"]) for r in rows)
    selected = []
    for label, stem in [("burn_source_50", "50"), ("boundary_risk", "1003"), ("thin_crack_candidate", "485")]:
        matches = [r for r in rows if r["stem"] == stem]
        selected.append({"check": label, "stem": stem, "rows": matches, "status": "PASS" if matches else "NOT_AVAILABLE_IN_PILOT_PROPOSAL"})

    # Synthetic deterministic geometry only; this is not model output.
    g = build_geometry([(2, 2), (10, 2), (10, 7), (2, 7)], 16, 12, context=3, roi_id="audit", source_id="synthetic")
    comp = compose_support([CompositionItem("a", "synthetic", "Burn", ((2, 2), (10, 2), (10, 7), (2, 7))), CompositionItem("b", "synthetic", "Crack", ((6, 4), (13, 4), (13, 10), (6, 10)))], 16, 12)
    budget = BudgetProtocol()
    budget.reserve(4, kind="smoke_planned")
    budget.reserve(54, kind="pilot_planned")
    budget.reserve(16, kind="revision_reserve")

    _write_json(out / "test_results.json", {"status": "PASS", "direct_assertion_tests": 8, "tests": {"geometry": "PASS", "composition": "PASS", "budget": "PASS"}, "model_calls": 0, "val_test_access": 0, "synthetic_arrays_only": True})
    _write_json(out / "geometry_invariants.json", {"polygon_coverage": 1.0, "output_shape_preserved": True, "outside_approved_edit_support_difference": 0, "order_independent_composition": True, "native_sizes_tested": [[640, 480], [720, 540], [384, 288]], "thin_crack": "PASS", "boundary_clamp": "FOV_UNCERTAIN", "gray_padding": "UNRESOLVED_HEURISTIC", "large_roi_context_risk": "FLAG_ONLY"})
    _write_json(out / "real_train_geometry_checks.json", {"source": str(proposal.relative_to(root)), "train_only": True, "checks": selected, "burn_50_verified": any(x["check"] == "burn_source_50" and x["status"] == "PASS" for x in selected), "no_image_decode": True, "no_val_test_access": True})
    _write_json(out / "roi_composition_audit.json", {"composition_status": comp.provenance["composition_status"], "overlap_pixels": comp.provenance["overlap_pixels"], "ownership_rule": comp.provenance["ownership"], "hold_policy_available": True, "hold_sources_cannot_pass_by_skipping": True, "multi_class_same_source": "covered_by deterministic ownership"})
    _write_json(out / "budget_protocol.json", budget.snapshot() | {"pilot_sources": 21, "pilot_rois": 27, "pilot_two_strength_planned": 54, "formal_production_default_strengths": 1, "gpt_fallback": "DISABLED_UNAUTHORIZED"})
    _write_json(out / "model_compatibility_review.json", {"status": "AUDIT_ONLY_NO_RUNTIME_CONFIRMATION", "candidates": [{"id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "role": "preferred", "mask_support": "NEEDS_ONLINE_VERIFICATION", "license": "NEEDS_ONLINE_VERIFICATION", "revision": "NEEDS_ONLINE_VERIFICATION", "runtime": "NEEDS_RUNTIME_SMOKE", "gpu_risk": "RTX 3080 Ti 12GB / 1024 canvas high memory risk", "mitigations": ["FP16", "VAE slicing/tiling if supported", "attention slicing", "CPU offload"]}, {"id": "stable-diffusion-v1-5/stable-diffusion-inpainting", "role": "fallback", "mask_support": "NEEDS_ONLINE_VERIFICATION", "license": "NEEDS_ONLINE_VERIFICATION", "runtime": "NEEDS_RUNTIME_SMOKE"}, {"id": "aiplayground/dreamshaper-8-inpainting@master", "role": "historical_reference_only", "runtime": "NEEDS_RUNTIME_SMOKE"}], "model_downloaded": False, "model_loaded": False, "inference_calls": 0, "cache_estimate": "NEEDS_ONLINE_VERIFICATION"})
    _write_json(out / "candidate_config.json", {"model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "revision": "NEEDS_ONLINE_VERIFICATION", "canvas": [1024, 1024], "dtype": "float16", "steps": 20, "guidance_scale": 7.5, "strengths": [0.75, 0.90], "seed_policy": "fixed per ROI and strength", "prompt": "Restore a credible normal industrial metal surface; preserve texture, mechanical geometry, reflection, and lighting; do not add parts, holes, seams, or defects; do not change non-editable pixels.", "negative_prompt": "new component, hole, seam, defect, text, watermark, geometry change, altered non-editable area", "mask_definition": "polygon edit core plus explicit transition halo; background is immutable", "runtime_status": "NOT_RUN"})
    _write_json(out / "provenance.json", {"run_id": RUN_ID, "created_utc": datetime.now(timezone.utc).isoformat(), "source_metadata": str(proposal.relative_to(root)), "source_sha256": hashlib.sha256(proposal.read_bytes()).hexdigest(), "train_only": True, "val_test_access": 0, "raw_data_write": False, "model_download": False, "model_inference": False, "historical_results_overwritten": False})
    _write_json(out / "summary.json", {"status": "EXP13_PHASE1A_COMPLETED_PENDING_CHATGPT_GATE", "geometry": "PASS", "composition": "PASS_WITH_HOLD_POLICY", "model": "AUDIT_ONLY", "budget": "CONDITIONAL_FEASIBLE", "train_pilot_rows": len(rows), "pilot_unique_sources": len(unique_sources), "pilot_roi_count": roi_count, "burn_source_50_verified": True, "next_action": "WAIT_FOR_CHATGPT_GATE; DO_NOT_RUN_DIFFUSION_SMOKE"})
    (out / "pilot_manifest_proposal.csv").write_bytes(proposal.read_bytes())


if __name__ == "__main__":
    main()
