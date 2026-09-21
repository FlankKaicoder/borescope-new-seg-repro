# Exp13.1-R1 Recovery Handoff

- Hostname: `autodl-container-8e424bb370-61e7c45a`
- Repository: `/root/autodl-tmp/borescope-new-seg-repro`
- Branch: `exp13/vlm-semantic-audit`
- Starting HEAD: `8e8983b2bc956af4a8a0aa39cafa889250c123b3`
- Recovery run: `results/exp13_diffusion_pilot/run_20260921_124440/`
- Model: `aiplayground/dreamshaper-8-inpainting@master` from ModelScope
- Snapshot: `/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting` (about 7.7G)
- Model identity: changed from the failed Hugging Face candidate; recorded as `MODEL_CHANGED_FOR_ACQUISITION_RECOVERY`
- GPU: NVIDIA GeForce RTX 2080 Ti, 22528 MiB
- Crack: 10 source images, 20 outputs
- corrosion: 10 source images, 20 outputs
- Failures: 0
- Smoke: Crack PASS, corrosion PASS, seed 13001
- Contact sheets: `review_assets/crack_generated_contact_sheet.png`, `review_assets/corrosion_generated_contact_sheet.png`
- VAL access: 0
- TEST access: 0
- training_performed: false
- SSL_performed: false
- Quality scores: not filled; awaiting review
- Final commit and push: `ca14f92aa9d32743f64d6624b71f3488a1e63c3a` pushed to origin/exp13/vlm-semantic-audit

## Gate

`MODEL_ACQUISITION_RECOVERED_PILOT_COMPLETED_WAITING_QUALITY_REVIEW`

Next step: wait for ChatGPT/human quality review only.
