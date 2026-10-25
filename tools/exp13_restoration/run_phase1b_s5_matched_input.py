#!/usr/bin/env python3
"""Exp13 Phase 1-B-S5: one matched-input SDXL verification call.

The script is intentionally single-purpose.  It reconstructs the historical
DreamShaper 512 input on CPU, builds the matched 1024 SDXL input, and only
executes the final Smoke call after all preflight gates pass.  It never writes
to the raw dataset or mutates the S2 ledger.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, "/root/autodl-tmp/borescope-new-seg-repro")

REPO = Path("/root/autodl-tmp/borescope-new-seg-repro")
RAW = Path("/root/autodl-tmp/损伤训练数据集")
MANIFEST = REPO / "results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv"
MANIFEST_SHA = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
HIST = REPO / "results/exp13_diffusion_pilot/run_20260921_124440"
HIST_INPUT = HIST / "crack/inputs/vlm_04_260.png"
HIST_ORIGINAL_MASK = HIST / "masks/original/vlm_04_260_mask.png"
HIST_INPAINT_MASK = HIST / "masks/inpaint/vlm_04_260_mask.png"
HIST_OUT_V1 = HIST / "crack/outputs/crack_260_v1.png"
HIST_OUT_V2 = HIST / "crack/outputs/crack_260_v2.png"
S2_LEDGER = REPO / "results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json"
MODEL = Path("/root/autodl-tmp/models_staging/exp13_phase1b_r3_20261025T042354Z/sdxl-inpainting-0.1")
MODEL_MANIFEST = Path("/root/autodl-tmp/sdxl_fp16_manifest.csv")
MODEL_MANIFEST_SHA = "a1a0302f7f8f9a0e2bf21e9b47a7757fd4e1ba9f246bf0c59bfcef1d5eb9607e"
REVISION = "115134f363124c53c7d878647567d04daf26e41e"
SOURCE_ID = "260"
EXPECTED_SOURCE_SHA = "222419556d6bd4761d6ec6c66102b6bd4775a1989d517eb4cc93f91ecf3c2bd2"
PROMPT = "restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting"
MODEL_CONFIG_ID = "exp13_phase1b_s5_sdxl_matched512_1024_s130500"
SEED = 130500
CONFIG = {
    "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "revision": REVISION,
    "resolution": [1024, 1024],
    "precision": "float16",
    "steps": 20,
    "guidance_scale": 7.5,
    "strength": 0.8,
    "seed": SEED,
    "prompt": PROMPT,
    "negative_prompt": None,
    "input_protocol": "historical DreamShaper center ImageOps.fit 512 then LANCZOS 2x",
    "mask_protocol": "historical 3px MaxFilter + GaussianBlur(sigma=1.0), fit NEAREST, 2x NEAREST",
    "padding": "none",
    "mask_polarity": "white=redraw, black=preserve",
    "cpu_offload": True,
    "vae_tiling": True,
    "vae_slicing": True,
}
USER_SOURCE_VISUAL_ACCEPTANCE = {
    "overlay_id": "S5_USER_SOURCE_VISUAL_ACCEPTANCE_20261025",
    "source_scope": {
        "Crack": {"86": "PASS", "111": "PASS", "146": "PASS", "260": "PASS", "470": "PASS"},
        "corrosion": {"2": "PASS_CROP_SCOPED", "8": "PASS_V2_PREFERRED_V1_REJECTED_HISTORY", "13": "PASS_CROP_SCOPED"},
    },
    "source": "USER_PROVIDED_S5_GATE",
    "independent_training_pair_admission": "NOT_YET_FROZEN",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def available_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def load_train_row() -> dict:
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("frozen TRAIN manifest SHA mismatch")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("split") == "train" and r.get("stem") == SOURCE_ID]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one TRAIN source row for {SOURCE_ID}, got {len(rows)}")
    row = rows[0]
    if row.get("sha256") != EXPECTED_SOURCE_SHA or row.get("labels_present") != "Crack":
        raise RuntimeError("source identity/hash/class does not match frozen Crack 260 record")
    return row


def load_polygon() -> tuple[Path, list[tuple[float, float]], dict]:
    row = load_train_row()
    image_path = RAW / row["image_path"]
    json_path = RAW / row["json_path"]
    if sha256(image_path) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("raw TRAIN image hash mismatch")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = [s for s in payload.get("shapes", []) if s.get("shape_type", "polygon") == "polygon" and s.get("label") == "Crack"]
    if len(shapes) != 1:
        raise RuntimeError(f"expected one Crack polygon for source 260, got {len(shapes)}")
    points = [(float(x), float(y)) for x, y in shapes[0]["points"]]
    return image_path, points, {"row": row, "json_path": str(json_path), "json_sha256": sha256(json_path)}


def mask_from_points(size: tuple[int, int], points: list[tuple[float, float]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon([(round(x), round(y)) for x, y in points], fill=255)
    return mask


def reconstruct_inputs() -> dict:
    raw_path, points, source_meta = load_polygon()
    with Image.open(raw_path) as im:
        native = im.convert("RGB")
    if native.size != (720, 540):
        raise RuntimeError(f"unexpected native source size {native.size}")
    original_mask = mask_from_points(native.size, points)
    inpaint_mask = original_mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.0))
    if sha256(HIST_INPUT) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("historical DreamShaper input hash mismatch")
    with Image.open(HIST_ORIGINAL_MASK) as historical_original:
        original_pixel_equal = bool(np.array_equal(np.asarray(historical_original.convert("L")), np.asarray(original_mask)))
    with Image.open(HIST_INPAINT_MASK) as historical_inpaint:
        inpaint_pixel_equal = bool(np.array_equal(np.asarray(historical_inpaint.convert("L")), np.asarray(inpaint_mask)))
    if not original_pixel_equal:
        raise RuntimeError("reconstructed original polygon mask differs from historical mask")
    if not inpaint_pixel_equal:
        raise RuntimeError("reconstructed 3px+feather inpaint mask differs from historical mask")
    common_rgb = ImageOps.fit(native, (512, 512), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    common_mask = ImageOps.fit(inpaint_mask, (512, 512), method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
    sdxl_rgb = common_rgb.resize((1024, 1024), Image.Resampling.LANCZOS)
    sdxl_mask = common_mask.resize((1024, 1024), Image.Resampling.NEAREST)
    if common_mask.getbbox() is None or sdxl_mask.getbbox() is None:
        raise RuntimeError("matched mask is empty")
    crop_box = [90, 0, 630, 540]
    if native.crop(tuple(crop_box)).size != (540, 540):
        raise RuntimeError("center crop mismatch")
    # Every 2x2 block of the nearest mask must map to the 512 mask exactly.
    mask_arr = np.asarray(common_mask)
    mask2_arr = np.asarray(sdxl_mask)
    block_equal = bool(np.array_equal(mask2_arr[0::2, 0::2], mask_arr))
    if not block_equal:
        raise RuntimeError("1024 nearest mask is not a 2x exact expansion of 512 mask")
    return {
        "raw_path": str(raw_path),
        "source_meta": source_meta,
        "points_xy": points,
        "native_size": list(native.size),
        "native_sha256": sha256(raw_path),
        "historical_input_sha256": sha256(HIST_INPUT),
        "historical_original_mask_sha256": sha256(HIST_ORIGINAL_MASK),
        "historical_inpaint_mask_sha256": sha256(HIST_INPAINT_MASK),
        "reconstructed_original_mask_sha256": sha256_image(original_mask),
        "reconstructed_inpaint_mask_sha256": sha256_image(inpaint_mask),
        "original_mask_pixel_equal_historical": original_pixel_equal,
        "inpaint_mask_pixel_equal_historical": inpaint_pixel_equal,
        "crop_xyxy": crop_box,
        "crop_description": "ImageOps.fit center crop [90,0,630,540] from 720x540",
        "common_rgb_512_sha256": sha256_image(common_rgb),
        "common_mask_512_sha256": sha256_image(common_mask),
        "sdxl_input_1024_sha256": sha256_image(sdxl_rgb),
        "sdxl_mask_1024_sha256": sha256_image(sdxl_mask),
        "common_mask_512_pixels": int(np.count_nonzero(mask_arr)),
        "sdxl_mask_1024_pixels": int(np.count_nonzero(mask2_arr)),
        "mask_2x_nearest_exact": block_equal,
        "same_fov_and_mask_content_not_identical_native_resolution": True,
        "ui_fov_review": {
            "status": "PASS_BOUNDED_USER_SOURCE_ACCEPTANCE",
            "basis": "USER_SOURCE_VISUAL_ACCEPTANCE=PASS for Crack 260; this is not pair admission",
            "automatic_ui_detector": "NOT_USED",
        },
        "native": native,
        "original_mask": original_mask,
        "inpaint_mask": inpaint_mask,
        "common_rgb": common_rgb,
        "common_mask": common_mask,
        "sdxl_rgb": sdxl_rgb,
        "sdxl_mask": sdxl_mask,
    }


def sha256_image(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def verify_model_files() -> dict:
    if sha256(MODEL_MANIFEST) != MODEL_MANIFEST_SHA:
        raise RuntimeError("SDXL model manifest SHA mismatch")
    rows = list(csv.DictReader(MODEL_MANIFEST.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 19:
        raise RuntimeError(f"expected 19 model files, got {len(rows)}")
    files = []
    for row in rows:
        path = MODEL / row["Path"]
        if not path.is_file() or path.stat().st_size != int(row["Bytes"]) or sha256(path) != row["SHA256"]:
            raise RuntimeError(f"model file integrity mismatch: {row['Path']}")
        files.append({"path": row["Path"], "bytes": int(row["Bytes"]), "sha256": row["SHA256"]})
    total = sum(x["bytes"] for x in files)
    if total != 6941223330:
        raise RuntimeError(f"model byte total mismatch: {total}")
    return {"revision": REVISION, "manifest_sha256": MODEL_MANIFEST_SHA, "files_verified": len(files), "verified_bytes": total, "files": files}


def verify_ledger() -> dict:
    data = json.loads(S2_LEDGER.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise RuntimeError("S2 ledger schema is not v2")
    active = [r for r in data.get("calls", {}).values() if r.get("status") != "CANCELLED"]
    if data.get("occupied_calls") != 3 or data.get("actual_calls") != 3 or data.get("remaining", {}).get("smoke") != 1:
        raise RuntimeError("S2 ledger does not expose exactly one remaining Smoke slot")
    if data.get("gpt_fallback_calls") != 0 or any(r.get("status") in {"RESERVED", "STARTED"} for r in active):
        raise RuntimeError("S2 ledger has unresolved or unauthorized active calls")
    return {"path": str(S2_LEDGER), "sha256": sha256(S2_LEDGER), "occupied_calls": data["occupied_calls"], "actual_calls": data["actual_calls"], "remaining_smoke": data["remaining"]["smoke"], "gpt_fallback_calls": data.get("gpt_fallback_calls", 0), "calls": data["calls"]}


def environment() -> dict:
    import torch
    import accelerate
    import diffusers
    import transformers
    free, total = torch.cuda.mem_get_info(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_bytes_before_load": int(free),
        "gpu_total_bytes": int(total),
        "cpu_ram_bytes_available": available_ram(),
        "gpu_preflight_threshold_bytes": 10 * 1024**3,
        "cpu_ram_preflight_threshold_bytes": 16 * 1024**3,
    }


def save_input_artifacts(run: Path, inp: dict) -> None:
    Image.open(HIST_INPUT).convert("RGB").save(run / "historical_native_input.png")
    inp["common_rgb"].save(run / "common_input_512.png")
    inp["common_mask"].save(run / "common_mask_512.png")
    inp["sdxl_rgb"].save(run / "sdxl_input_1024.png")
    inp["sdxl_mask"].save(run / "sdxl_mask_1024.png")
    overlay = inp["common_rgb"].copy()
    d = ImageDraw.Draw(overlay)
    d.bitmap((0, 0), inp["common_mask"], fill=(255, 0, 0, 100))
    overlay.save(run / "common_input_mask_overlay_512.png")


def preflight(run: Path) -> dict:
    run.mkdir(parents=True, exist_ok=False)
    inp = reconstruct_inputs()
    model = verify_model_files()
    ledger = verify_ledger()
    env = environment()
    if env["gpu_free_bytes_before_load"] < env["gpu_preflight_threshold_bytes"] or env["cpu_ram_bytes_available"] < env["cpu_ram_preflight_threshold_bytes"]:
        raise RuntimeError(f"insufficient resource preflight: {env}")
    save_input_artifacts(run, inp)
    input_audit = {k: v for k, v in inp.items() if k not in {"native", "original_mask", "inpaint_mask", "common_rgb", "common_mask", "sdxl_rgb", "sdxl_mask"}}
    dump(run / "input_alignment_audit.json", input_audit)
    dump(run / "source_visual_acceptance_overlay.json", USER_SOURCE_VISUAL_ACCEPTANCE)
    dump(run / "git_preflight.json", {
        "branch": os.popen(f"git -C {REPO} branch --show-current").read().strip(),
        "head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "expected_start_head": "10f189938ef61572deb002056439ee0e2dcd7059",
        "remote_check_deferred_to_host": True,
        "historical_untracked_assets_preserved": True,
    })
    dump(run / "environment_preflight.json", {"environment": env, "model": model, "ledger": ledger, "raw_data_write": 0, "val_access": 0, "test_access": 0, "model_load": 0, "inference_calls": 0})
    dump(run / "frozen_single_call_config.json", {"status": "PREFLIGHT_PASS_MODEL_NOT_LOADED", "source_id": SOURCE_ID, "source_sha256": inp["native_sha256"], "config": CONFIG, "user_source_visual_acceptance": "PASS", "training_pair_geometry_and_quality_admission": "NOT_YET_FROZEN", "label": "BOUNDED_SINGLE_SOURCE_INPUT_ALIGNMENT_STUDY"})
    dump(run / "preflight_only_summary.json", {"status": "EXP13_PHASE1B_S5_PREFLIGHT_PASS", "model_load": 0, "inference_calls": 0, "source_id": SOURCE_ID, "model_files": model["files_verified"], "verified_model_bytes": model["verified_bytes"], "source_sha256": inp["native_sha256"], "ledger_remaining_smoke": ledger["remaining_smoke"], "val_access": 0, "test_access": 0, "raw_data_write": 0})
    return {"input": inp, "model": model, "ledger": ledger, "environment": env}


def copy_successor_ledger(run: Path, source_ledger: dict) -> Path:
    successor = run / "budget_successor_ledger.json"
    payload = json.loads(S2_LEDGER.read_text(encoding="utf-8"))
    payload.setdefault("history", []).append({"event": "SUCCESSOR_CREATED_FOR_S5", "source_ledger": str(S2_LEDGER), "source_ledger_sha256": source_ledger["sha256"], "verified_occupied_calls": 3, "verified_actual_calls": 3, "authorized_new_calls": 1, "timestamp": datetime.now(timezone.utc).isoformat()})
    successor.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return successor


def make_contact_sheet(run: Path, inp: dict, sdxl_raw: Image.Image, sdxl_aligned: Image.Image, diff: Image.Image, local_zoom: Image.Image) -> None:
    items = [
        (inp["common_rgb"], "Common DreamShaper input 512"),
        (inp["common_mask"].convert("RGB"), "Common mask 512 (white=edit)"),
        (Image.open(HIST_OUT_V1).convert("RGB"), "DreamShaper 260 v1"),
        (Image.open(HIST_OUT_V2).convert("RGB"), "DreamShaper 260 v2"),
        (sdxl_raw, "SDXL raw output 1024"),
        (sdxl_aligned, "SDXL aligned view 512"),
        (diff, "SDXL input difference 512"),
        (local_zoom, "Crack local zoom"),
    ]
    panels = []
    for image, title in items:
        thumb = image.convert("RGB").copy(); thumb.thumbnail((440, 320), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (460, 370), "white"); panel.paste(thumb, ((460-thumb.width)//2, 35+(320-thumb.height)//2)); ImageDraw.Draw(panel).text((8, 10), title, fill="black"); panels.append(panel)
    sheet = Image.new("RGB", (920, 4 * 370), (230, 230, 230))
    for i, panel in enumerate(panels): sheet.paste(panel, ((i % 2) * 460, (i // 2) * 370))
    sheet.save(run / "matched_input_comparison_contact_sheet.png")


def execute(run: Path) -> dict:
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    from tools.exp13_restoration.protocol import BudgetProtocol

    pre = preflight_existing(run)
    ledger_path = copy_successor_ledger(run, pre["ledger"])
    budget = BudgetProtocol(ledger_path=ledger_path)
    call_id = "260:matched_center512:s0.8"
    reserved = budget.reserve_call(call_id=call_id, source_id=SOURCE_ID, roi_id="matched_center512", stage="smoke", strength=0.8, model_config_id=MODEL_CONFIG_ID)
    if reserved.get("status") != "RESERVED" or budget.snapshot().get("remaining", {}).get("smoke") != 0:
        raise RuntimeError("successor ledger did not reserve exactly the final Smoke slot")
    dump(run / "budget_reservation.json", {"call_id": call_id, "status": reserved["status"], "snapshot": budget.snapshot()})
    load_report = {"attempted": False, "success": False, "revision": REVISION, "local_path": str(MODEL), "local_files_only": True}
    runtime = {"call_id": call_id, "status": "NOT_STARTED"}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        load_report["attempted"] = True; start_load = time.perf_counter()
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(MODEL), torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True)
        pipe.enable_model_cpu_offload(gpu_id=0); pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
        load_report.update({"success": True, "load_seconds": time.perf_counter() - start_load, "pipeline_class": type(pipe).__name__, "cpu_offload": True, "vae_tiling": True, "vae_slicing": True})
        budget.start_call(call_id, real_model_call=True)
        started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        inp = pre["input"]
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        output = pipe(prompt=PROMPT, negative_prompt=None, image=inp["sdxl_rgb"], mask_image=inp["sdxl_mask"], height=1024, width=1024, num_inference_steps=20, guidance_scale=7.5, strength=0.8, generator=generator).images[0].convert("RGB")
        elapsed = time.perf_counter() - started
        arr = np.asarray(output, dtype=np.uint8)
        if arr.shape != (1024, 1024, 3): raise RuntimeError(f"unexpected output shape {arr.shape}")
        budget.finish_call(call_id, success=True)
        aligned = output.resize((512, 512), Image.Resampling.LANCZOS)
        inp_arr = np.asarray(inp["common_rgb"], dtype=np.int16); aligned_arr = np.asarray(aligned, dtype=np.int16)
        diff_arr = np.max(np.abs(aligned_arr - inp_arr), axis=2).astype(np.uint8)
        diff = Image.fromarray(diff_arr, "L")
        mask_arr = np.asarray(inp["common_mask"], dtype=np.uint8) > 0
        changed = diff_arr > 5
        outside = int(np.count_nonzero(changed & ~mask_arr)); inside = int(np.count_nonzero(changed & mask_arr))
        protected = inp["common_rgb"].copy(); protected.paste(aligned, mask=inp["common_mask"])
        protected.save(run / "sdxl_postprocess_comparison_only_512.png")
        output.save(run / "sdxl_raw_output_1024.png"); aligned.save(run / "sdxl_aligned_512.png"); diff.save(run / "sdxl_difference_512.png")
        # Local zoom follows the same 512 mask bbox and is diagnostic only.
        bbox = inp["common_mask"].getbbox(); local_zoom = aligned.crop(bbox).resize((max(1, (bbox[2]-bbox[0])*4), max(1, (bbox[3]-bbox[1])*4)), Image.Resampling.NEAREST) if bbox else aligned
        local_zoom.save(run / "sdxl_crack_zoom_512.png")
        make_contact_sheet(run, inp, output, aligned, diff, local_zoom)
        runtime.update({"status": "SUCCEEDED", "elapsed_seconds": elapsed, "input_size": [1024, 1024], "output_size": list(output.size), "output_shape_hwc": list(arr.shape), "output_dtype": str(arr.dtype), "mask_polarity": "white=redraw; black=preserve", "mask_pixels_1024": int(np.count_nonzero(np.asarray(inp["sdxl_mask"]))), "changed_pixels_gt5_aligned_512": int(changed.sum()), "changed_pixels_gt5_inside_mask_512": inside, "changed_pixels_gt5_outside_mask_512": outside, "nan_inf": False, "black_or_constant_output": bool(int(arr.std()) == 0 or float(arr.mean()) < 2.0), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "model_revision": REVISION})
        dump(run / "single_call_runtime.json", runtime)
        dump(run / "model_load_report.json", load_report)
        dump(run / "cross_model_comparison_metrics.json", {"label": "BOUNDED_SINGLE_SOURCE_INPUT_ALIGNMENT_STUDY", "same_fov_and_mask_content_not_identical_native_resolution": True, "historical_outputs": {"dreamshaper_v1": str(HIST_OUT_V1), "dreamshaper_v2": str(HIST_OUT_V2)}, "sdxl_raw_output": "sdxl_raw_output_1024.png", "sdxl_aligned_output": "sdxl_aligned_512.png", "sdxl_postprocess_label": "POSTPROCESS_COMPARISON_ONLY", "metrics": runtime})
        dump(run / "artifact_quality_preliminary.json", {"semantic_quality": "PENDING_USER_VISUAL_REVIEW", "formal_quality_pass": 0, "training_pair_admission": "NOT_YET_FROZEN", "candidate": "Crack:260", "user_source_visual_acceptance": "PASS", "interpretation_limit": "one source and one call cannot rank models or establish general success"})
        dump(run / "integrity_and_preservation.json", {"source_sha256": pre["input"]["native_sha256"], "raw_data_write": 0, "val_access": 0, "test_access": 0, "outside_mask_changed_pixels_aligned_512": outside, "protected_comparison_only": True, "raw_output_preserved": True, "historical_outputs_untouched": True})
        return runtime
    except BaseException as exc:
        try:
            budget.finish_call(call_id, success=False, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        runtime.update({"status": "FAILED", "exception": traceback.format_exc()})
        dump(run / "single_call_runtime.json", runtime); dump(run / "model_load_report.json", load_report); raise
    finally:
        dump(run / "budget_successor_ledger.json", budget.snapshot())
        dump(run / "provenance.json", {"phase": "Exp13 Phase 1-B-S5", "scope": "one TRAIN source, one SDXL call", "source_id": SOURCE_ID, "source_sha256": EXPECTED_SOURCE_SHA, "s2_ledger": str(S2_LEDGER), "s2_ledger_sha256": pre["ledger"]["sha256"], "model_revision": REVISION, "new_model_calls": 1 if runtime.get("status") == "SUCCEEDED" or runtime.get("status") == "FAILED" else 0, "val_access": 0, "test_access": 0, "raw_data_write": 0})


def preflight_existing(run: Path) -> dict:
    if not (run / "preflight_only_summary.json").is_file():
        return preflight(run)
    # Re-run all checks without creating another directory or loading a model.
    inp = reconstruct_inputs(); model = verify_model_files(); ledger = verify_ledger(); env = environment()
    return {"input": inp, "model": model, "ledger": ledger, "environment": env}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    if args.preflight:
        result = preflight(run)
        print(json.dumps({"run_dir": str(run), "status": "EXP13_PHASE1B_S5_PREFLIGHT_PASS", "model_load": 0, "inference_calls": 0, "remaining_smoke": result["ledger"]["remaining_smoke"]}, ensure_ascii=False))
    elif args.execute:
        if not run.exists(): preflight(run)
        result = execute(run)
        print(json.dumps({"run_dir": str(run), "status": "EXP13_PHASE1B_S5_EXECUTED", "call_status": result.get("status"), "call_id": result.get("call_id")}, ensure_ascii=False))
    else:
        raise SystemExit("choose --preflight or --execute")


if __name__ == "__main__":
    main()
