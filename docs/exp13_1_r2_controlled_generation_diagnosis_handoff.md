# Exp13.1-R2 Handoff

- Hostname: `autodl-container-8e424bb370-61e7c45a`
- Path: `/root/autodl-tmp/borescope-new-seg-repro`
- Branch: `exp13/vlm-semantic-audit`
- Starting HEAD: `7010864e1f3eadb6a98405b02aa7f7cd671293b3`
- Final HEAD: to be filled after commit
- Model: `aiplayground/dreamshaper-8-inpainting`, revision `master`
- Sources: Crack 260, Crack 469, corrosion 9, corrosion 23
- Masks: mask_A = R1 3 px dilation; mask_B = fixed 5 px dilation; both feathered
- Prompt variants: v1 = R1 prompt; v2 = shorter defect-preserving prompt
- Strengths: 0.55, 0.75, 0.95
- Outputs: 48 planned, 48 successful, 0 failed; all verified as 512x512 PNGs
- Review sheets: `results/exp13_diffusion_diagnosis/r2_20260921_135903/review_assets/`
- VAL access: 0
- TEST access: 0
- Training: false
- SSL: false
- Protected R1 runs were preserved and not overwritten
- Git commit: to be filled after commit
- Push: to be filled after push
- Current Gate: `DIAGNOSIS_COMPLETED_WAITING_REVIEW`

The run is complete for artifact generation only. Wait for review before any Exp13.1 continuation; do not enter SSL or broaden the source set.
