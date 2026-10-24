# Exp13 Phase 1-B SDXL Inpainting Smoke review

Status: `EXP13_PHASE1B_SMOKE_HOLD` (`PREFLIGHT_HOLD`)

Run: `results/exp13_phase1b_smoke/20261024T130500Z/`

The authorized smoke did not start. Read-only preflight found the RTX 3080 Ti/12 GiB environment and installed runtime (`torch 2.8.0+cu128`, CUDA 12.8, Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1), with about 30 GiB free on `/root/autodl-tmp`. The required Hugging Face metadata request for `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` timed out; IPv6 was unreachable. Exact revision, license, file manifest and download size therefore could not be verified.

No checkpoint was downloaded, no model was loaded, no GPU inference or API call occurred, and no fallback model was used. No call was reserved or started. R2 evidence identifies TRAIN source `2` as a one-ROI candidate and source `17` as a two-ROI candidate with zero edit-support overlap, but final source/mask hashes were not frozen because model preflight failed. No contact sheet exists because there are no model outputs to review.

Gates: B-S1 `HOLD`; B-S2 `CONDITIONAL_PASS_NO_CALLS`; B-S3 `CONDITIONAL_PASS_R2_CANDIDATE_ONLY`; B-S4/B-S5 `NOT_TESTED`; B-S6 `PASS`. Do not retry download, run Smoke, use a fallback checkpoint, start Pilot, freeze pairs, or run SSL without a new authorized continuation after preflight succeeds.
