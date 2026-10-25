#!/usr/bin/env python3
"""CPU-only finalization for persisted Exp13 Phase 1-B-S5 artifacts."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(path: Path, source: np.ndarray, mask: np.ndarray) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS), dtype=np.int16)
    diff = np.max(np.abs(arr - source.astype(np.int16)), axis=2).astype(np.uint8)
    changed = diff > 5
    result = {
        "path": str(path),
        "size": list(Image.open(path).size),
        "changed_pixels_gt5": int(changed.sum()),
        "changed_pixels_gt5_inside_mask": int(np.count_nonzero(changed & mask)),
        "changed_pixels_gt5_outside_mask": int(np.count_nonzero(changed & ~mask)),
        "mean_abs_rgb_delta": float(np.abs(arr - source.astype(np.int16)).mean()),
        "max_abs_rgb_delta": int(np.abs(arr - source.astype(np.int16)).max()),
    }
    Image.fromarray(diff, "L").save(path.parent / (path.stem + "_difference_512.png"))
    return result


def contact_sheet(run: Path, panels: list[tuple[Image.Image, str]]) -> None:
    cards = []
    for image, title in panels:
        thumb = image.convert("RGB").copy(); thumb.thumbnail((440, 300), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (460, 350), "white")
        card.paste(thumb, ((460 - thumb.width) // 2, 34 + (300 - thumb.height) // 2))
        ImageDraw.Draw(card).text((8, 8), title, fill="black")
        cards.append(card)
    cols = 3; rows = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows * 350), (232, 232, 232))
    for i, card in enumerate(cards): sheet.paste(card, ((i % cols) * 460, (i // cols) * 350))
    sheet.save(run / "matched_input_comparison_contact_sheet.png")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); args = ap.parse_args()
    run = Path(args.run_dir)
    source = np.asarray(Image.open(run / "common_input_512.png").convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(run / "common_mask_512.png").convert("L"), dtype=np.uint8) > 0
    aligned = Image.open(run / "sdxl_aligned_512.png").convert("RGB")
    outputs = {
        "dreamshaper_v1": run / "dreamshaper_v1_512.png",
        "dreamshaper_v2": run / "dreamshaper_v2_512.png",
        "sdxl_aligned_512": run / "sdxl_aligned_512.png",
    }
    # Historical files are copied into the controlled review directory; the source files remain untouched.
    hist = run / "historical"; hist.mkdir(exist_ok=True)
    repo_root = run.parent.parent.parent
    for name, src in [("dreamshaper_v1_512.png", repo_root / "results/exp13_diffusion_pilot/run_20260921_124440/crack/outputs/crack_260_v1.png"), ("dreamshaper_v2_512.png", repo_root / "results/exp13_diffusion_pilot/run_20260921_124440/crack/outputs/crack_260_v2.png")]:
        # The persisted run is repository/results/exp13_phase1b_s5/run_id.
        if src.is_file(): Image.open(src).convert("RGB").save(run / name)
        else: raise FileNotFoundError(src)
    ds1 = Image.open(run / "dreamshaper_v1_512.png").convert("RGB")
    ds2 = Image.open(run / "dreamshaper_v2_512.png").convert("RGB")
    comparison = {name: metrics(path, source, mask) for name, path in outputs.items()}
    comparison["protocol"] = {
        "label": "BOUNDED_SINGLE_SOURCE_INPUT_ALIGNMENT_STUDY",
        "same_fov_and_mask_content_not_identical_native_resolution": True,
        "postprocess_label": "POSTPROCESS_COMPARISON_ONLY",
        "no_model_ranking": True,
    }
    existing = json.loads((run / "cross_model_comparison_metrics.json").read_text())
    existing["per_output_metrics"] = comparison
    dump(run / "cross_model_comparison_metrics.json", existing)
    panels = [
        (Image.open(run / "historical_native_input.png"), "Native source 720x540"),
        (Image.open(run / "common_input_512.png"), "Common input 512"),
        (Image.open(run / "common_mask_512.png"), "Common mask 512"),
        (ds1, "DreamShaper v1 512"),
        (Image.open(run / "dreamshaper_v1_512_difference_512.png"), "DreamShaper v1 difference"),
        (ds2, "DreamShaper v2 512"),
        (Image.open(run / "dreamshaper_v2_512_difference_512.png"), "DreamShaper v2 difference"),
        (Image.open(run / "sdxl_raw_output_1024.png"), "SDXL raw output 1024"),
        (aligned, "SDXL aligned 512"),
        (Image.open(run / "sdxl_difference_512.png"), "SDXL difference"),
        (Image.open(run / "sdxl_postprocess_comparison_only_512.png"), "SDXL protected (comparison only)"),
    ]
    contact_sheet(run, panels)
    ledger = json.loads((run / "budget_successor_ledger.json").read_text())
    runtime = json.loads((run / "single_call_runtime.json").read_text())
    summary = {
        "status": "EXP13_PHASE1B_S5_COMPLETED_PENDING_CHATGPT_GATE",
        "label": "BOUNDED_SINGLE_SOURCE_INPUT_ALIGNMENT_STUDY",
        "scope": "TRAIN-only; one Crack:260 source; one real SDXL call; no retry",
        "source_id": "260",
        "source_sha256": json.loads((run / "input_alignment_audit.json").read_text())["native_sha256"],
        "model_revision": runtime["model_revision"],
        "model_load": "SUCCESS_OFFLINE",
        "actual_new_model_calls": 1,
        "budget": {"occupied_calls": ledger["occupied_calls"], "actual_calls": ledger["actual_calls"], "smoke_remaining": ledger["remaining"]["smoke"], "retry_calls": ledger["retry_calls"], "gpt_fallback_calls": ledger["gpt_fallback_calls"]},
        "runtime": {"elapsed_seconds": runtime["elapsed_seconds"], "gpu_peak_allocated_bytes": runtime["gpu_peak_allocated_bytes"], "gpu_peak_reserved_bytes": runtime["gpu_peak_reserved_bytes"]},
        "input_alignment": {"fov": "DreamShaper center fit 512 then SDXL LANCZOS 2x", "mask": "same 512 mask then NEAREST 2x", "reflect_padding": False, "mask_2x_nearest_exact": True},
        "preservation": {"raw_output_preserved": True, "aligned_changed_gt5_outside_mask": runtime["changed_pixels_gt5_outside_mask_512"], "protected_comparison_only": True, "val_access": 0, "test_access": 0, "raw_data_write": 0},
        "quality": {"semantic_quality": "PENDING_USER_VISUAL_REVIEW", "formal_quality_pass": 0, "training_pair_admission": "NOT_YET_FROZEN"},
        "interpretation_limit": "One source/call tests input-protocol compatibility only; it cannot rank models or establish general restoration success.",
    }
    dump(run / "summary.json", summary)
    zip_path = run / "s5_controlled_review_images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.glob("*.png")):
            z.write(path, path.name)
    print(json.dumps({"status": summary["status"], "summary": str(run / "summary.json"), "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
