# Exp13.1-R2 Controlled Generation Diagnosis

## Purpose

Exp13.1-R1 established a working diffusion/inpainting chain but failed the quality Gate: Crack outputs often restored away the defect, while corrosion outputs showed clean-surface restoration, semantic drift, or hallucinated mechanical structures. R2 is a controlled diagnosis of whether those failures are attributable to mask, strength, or prompt settings, versus a generic text-only SD1.5-family inpainting limitation for borescope defects.

## Fixed scope

- Server repository: `/root/autodl-tmp/borescope-new-seg-repro`
- Branch: `exp13/vlm-semantic-audit`
- Model: `aiplayground/dreamshaper-8-inpainting`, revision `master`
- Local snapshot: `/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting`
- Resolution: 512x512; steps 20; guidance 7.5; seed 17001
- Sources: Crack `260`, Crack `469`, corrosion `9`, corrosion `23`
- No VAL or TEST image access; no dataset or label changes
- No SSL, YOLO training, downstream training, ControlNet, LoRA, or model replacement

The four sources are fixed because Crack 260 represents the recurring “crack repaired away” failure, Crack 469 retains a crack intermittently and therefore exposes seed/parameter sensitivity, corrosion 9 exhibits hallucinated mechanical objects, and corrosion 23 exhibits semantic drift.

## Diagnosis matrix

For each source, the matrix is:

- mask_A: R1 inpaint mask (3 px dilation, feathering retained)
- mask_B: fixed 5 px dilation plus the same feathering
- strength: `0.55`, `0.75`, `0.95`
- prompt: `v1` R1 prompt and `v2` shorter defect-preserving prompt
- fixed seed: `17001`

This yields `2 x 3 x 2 = 12` outputs per source and `48` outputs total. Only mask strategy, strength, and prompt variant vary; all other inference settings remain fixed.

## Artifacts

The unique run is `results/exp13_diffusion_diagnosis/r2_20260921_135903/` and contains the environment record, prompt configuration, original and derived masks, copied source inputs, manifest, outputs, logs, and review assets. The manifest is `manifests/diagnosis_manifest.csv`.

Review assets:

- `review_assets/crack_260_diagnosis_sheet.png`
- `review_assets/crack_469_diagnosis_sheet.png`
- `review_assets/corrosion_9_diagnosis_sheet.png`
- `review_assets/corrosion_23_diagnosis_sheet.png`
- `review_assets/diagnosis_overview_contact_sheet.png`
- `review_assets/diagnosis_review_template.csv`

The review CSV is intentionally blank apart from identifiers and fixed condition fields; no quality scores were assigned automatically.

## Gate

All 48 planned outputs completed successfully and are available for human/ChatGPT review. The current Gate is `DIAGNOSIS_COMPLETED_WAITING_REVIEW`. This does not authorize SSL, seven-class expansion, or any downstream training. Possible review conclusions are: parameters can repair the failure; only some sources can be repaired; or generic text-only inpainting is unsuitable for these defects.
