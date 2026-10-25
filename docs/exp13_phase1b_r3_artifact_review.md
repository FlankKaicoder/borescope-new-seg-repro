# Exp13 Phase 1-B-R3 SDXL Offline Artifact Review

Run: `20261025T042354Z`
Status: `EXP13_PHASE1B_R3_ARTIFACT_VERIFIED_PENDING_CHATGPT_GATE`
Starting repository HEAD: `9c4f0601af10edbcab3ce3fc12ca24f2da91d0b6`

## Scope and provenance

This was an offline artifact-verification run only. The user-provided Windows evidence identifies `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, revision `115134f363124c53c7d878647567d04daf26e41e`, and license label `openrail++`. AutoDL did not independently query Hugging Face or fetch the license text. The server independently rehashed the extracted files against the uploaded Windows manifest; this establishes transfer/content agreement with that manifest, not independent official provenance.

The uploaded archive was `/root/autodl-tmp/sdxl-inpainting-fp16.tar` (6,941,242,368 bytes), with manifest `/root/autodl-tmp/sdxl_fp16_manifest.csv`. The manifest SHA256 is `a1a0302f7f8f9a0e2bf21e9b47a7757fd4e1ba9f246bf0c59bfcef1d5eb9607e`. The archive contained 27 members, of which 19 were files; path and link safety checks passed. Extraction went only to the new isolated directory `/root/autodl-tmp/models_staging/exp13_phase1b_r3_20261025T042354Z/sdxl-inpainting-0.1`. The earlier incomplete `/root/autodl-tmp/models/sdxl-inpainting-0.1` directory was not modified, merged, or deleted. The archive was retained.

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| R3-A artifact discovery | PASS | `artifact_discovery.json`; archive, manifest, and isolated model path located |
| R3-B file integrity | PASS | 19 expected/found/verified; 0 size or SHA256 mismatches; 0 missing/unexpected; 6,941,223,330 verified bytes |
| R3-C structure and FP16 static checks | PASS | Required Diffusers components/configs present; JSON parsed; four safetensors headers/layouts inspected; no LFS pointers; 2,643 tensor entries |
| R3-D offline static prerequisites | PASS (static only) | Python/dependency imports and local structure checks passed; `from_pretrained` and GPU load were not attempted |
| R3-E preservation | PASS | No model/TAR added to Git; no VAL/TEST, raw-data write, training, inference, or budget reservation |

After extraction, the recorded free space was 17,858,826,240 bytes. Environment probe: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1, Safetensors 0.6.2; RTX 3080 Ti, 12,288 MiB total and 11,912 MiB free at probe time.

## Limits and disposition

Static compatibility is not a successful Diffusers load. No `from_pretrained`, GPU model load, inference, image generation, visual-quality assessment, or Smoke reservation occurred. The 12 GiB GPU's ability to load or run this model is unknown. The source revision/license claims are recorded as user-provided Windows evidence and are not server-side authenticated. No VAL/TEST access or raw-data write occurred; Diffusion inference, GPU model load, fallback, Smoke reservation/STARTED, SSL, and YOLO training all remained zero.

R3-A through R3-E pass within the stated evidence boundaries. This does not authorize GPU Smoke, Pilot, trusted-pair freezing, or SSL. Stop for ChatGPT Gate.

Machine-readable evidence: `results/exp13_phase1b_r3/20261025T042354Z/`. The verifier is `tools/exp13_restoration/verify_phase1b_r3_artifact.py`.
