# Exp13.0 Server Handoff

## Environment

- Hostname: autodl-container-8e424bb370-61e7c45a
- Path: /root/autodl-tmp/borescope-new-seg-repro
- Branch: exp13/vlm-semantic-audit
- Preparation base HEAD: 7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732

## TRAIN sampling

- Dataset root: /root/autodl-tmp/borescope-new-seg-data/v1
- Verified TRAIN label files: 668
- Unique source images: 40
- Crack: 20 images
- corrosion: 20 images
- Instances recorded in manifest per selected target class
- VAL accessed: no
- TEST accessed: no

## Generated files

- results/exp13_vlm_semantic_audit/sample_manifest.csv
- results/exp13_vlm_semantic_audit/samples/ (40 copied TRAIN samples)
- results/exp13_vlm_semantic_audit/overlays/ (40 polygon overlays)
- results/exp13_vlm_semantic_audit/contact_sheets/contact_sheet_crack.png
- results/exp13_vlm_semantic_audit/contact_sheets/contact_sheet_corrosion.png
- results/exp13_vlm_semantic_audit/metadata/polygon_context.json
- results/exp13_vlm_semantic_audit/metadata/preparation_audit.json
- results/exp13_vlm_semantic_audit/metadata/vlm_schema.json
- results/exp13_vlm_semantic_audit/prompts/vlm_prompt_template.md
- tools/exp13_vlm_semantic_audit.py
- docs/exp13_0_vlm_semantic_audit.md

## Restrictions verified

No VLM/API call, diffusion, synthetic image generation, SSL training, YOLO training, model download, or GPU training was performed. Original dataset, labels, and splits were not modified. Existing untracked Phase C1 content was left untouched and excluded from the Exp13 commit.

## Gate

PASS_PREPARATION_ONLY

This is not a semantic-model performance result. Exp13.1 Diffusion is blocked pending ChatGPT review and explicit authorization.

## Next step

Review the manifest, overlays, schema, and prompt template. Do not enter Diffusion or any model-training stage until the next Gate is authorized.

## Git commit

Preparation commit: e76bab6cbf5d41a235732947641f29ca2c9d2b31

