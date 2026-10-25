# Exp13 Phase 1-B-S5 matched-input review

Status: `EXP13_PHASE1B_S5_COMPLETED_PENDING_CHATGPT_GATE`

This bounded study used exactly one TRAIN source (`Crack:260`) and one real SDXL inpainting call. The purpose was to test whether using the historical DreamShaper center-crop and mask construction removes the compact-ROI/reflect-padding confound. It is an input-protocol compatibility study, not a model ranking or a general restoration result.

## Scope and provenance

- Branch: `exp13/vlm-semantic-audit`; reporting start `10f189938ef61572deb002056439ee0e2dcd7059`.
- Source: TRAIN-only `260`, native 720×540, image SHA256 `222419556d6bd4761d6ec6c66102b6bd4775a1989d517eb4cc93f91ecf3c2bd2`.
- Historical center crop: `[90, 0, 630, 540]`, then RGB LANCZOS to 512×512.
- Historical polygon mask and 3 px dilation plus Gaussian feather were reproduced pixel-exactly; the same 512 mask was enlarged to 1024 with NEAREST.
- No reflect padding, VAL/TEST access, raw-data writes, training, or fallback model was used.
- Model: offline `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, revision `115134f363124c53c7d878647567d04daf26e41e`, FP16, local-only load.

The source-level user decision for Crack 260 is `USER_SOURCE_VISUAL_ACCEPTANCE=PASS`. This does not freeze a generated training pair: `TRAINING_PAIR_GEOMETRY_AND_QUALITY_ADMISSION=NOT_YET_FROZEN`.

## Fixed call

The call ID was `260:matched_center512:s0.8` with 1024×1024 input/output, 20 steps, guidance 7.5, strength 0.8, seed 130500, and the registered normal-metal prompt. Runtime was 12.756242 s; peak allocated/reserved GPU memory was 5,764,813,824 / 6,247,415,808 bytes. The output is finite, non-constant, and `uint8` RGB.

The successor ledger inherits the immutable S2 ledger and records `occupied=4`, `actual=4`, `smoke_remaining=0`, `retry=0`, and `GPT fallback=0`. The first failed process launch was an import-path failure before model loading, reservation, or inference; it is not a model retry or an additional call.

## Evidence and limits

The raw SDXL 1024 output and the 512 aligned view are preserved. In the aligned diagnostic comparison, 75,253 pixels changed by more than 5 intensity levels: 8,518 inside the mask and 66,735 outside it. This is an engineering observation about the raw output and is not converted into a semantic quality verdict. The protected-background composite is explicitly `POSTPROCESS_COMPARISON_ONLY` and cannot replace the raw output.

DreamShaper v1/v2 were copied CPU-only from the existing historical outputs and compared under the common 512 view. Their diagnostic outside-mask changes are also nonzero. Because historical DreamShaper and this SDXL call differ in resolution, strength, seed, model, and execution protocol, the contact sheet is labeled `CROSS_PROTOCOL_CASE_STUDY_NOT_MODEL_RANKING`.

Semantic quality remains `PENDING_USER_VISUAL_REVIEW`; formal quality pass is 0 and trusted-pair admission is 0. One source and one call cannot establish model superiority, general restoration success, or downstream SSL value. No further call is authorized by this record.

Evidence directory on AutoDL:

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s5/20261025T230000Z_s5/`

The controlled review ZIP is `s5_controlled_review_images.zip` (6,632,554 bytes; SHA256 `2b45632cdd1622e680c22c5ef983e8d8d9d4e56dc7889232805c9dd208b39387`). Industrial PNGs and the ZIP remain outside Git.

