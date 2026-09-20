# Exp13.0-B Visual Review Assets Handoff

## Artifact path

`results/exp13_vlm_semantic_validation/review_assets/`

The directory contains class-separated raw samples and polygon-overlay copies, plus:

- `contact_sheet_crack.png`
- `contact_sheet_corrosion.png`
- `review_manifest.csv`

## Counts and split scope

- Total images: 40
- Crack: 20
- corrosion: 20
- All records are from TRAIN only.
- VAL/TEST were not accessed.
- No resampling was performed; the existing Exp13.0 manifest was reused.

## Restrictions

No VLM inference, API call, Diffusion, ControlNet, synthetic image generation, SSL, or model training was performed.

## Git

- Branch: `exp13/vlm-semantic-audit`
- Visual asset export commit: `7e73da79c75df59e341cf7771e3aac1bd02c72a1`

## Gate

`VISUAL_REVIEW_ARTIFACTS_READY_WAITING_FOR_REVIEW`

Stop here and wait for visual review. Do not proceed to VLM inference or Diffusion without explicit authorization.
