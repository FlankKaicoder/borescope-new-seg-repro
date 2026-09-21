# Exp13.1 Diffusion Pilot Handoff

- Hostname: `autodl-container-8e424bb370-61e7c45a`
- Repository: `/root/autodl-tmp/borescope-new-seg-repro`
- Branch: `exp13/vlm-semantic-audit`
- Starting HEAD: `25230a662989488f2c5a8b74a157af4ad4e80992`
- Final HEAD: `d87ce55ab09757d9a8a42713d0451bddd5303856`
- GPU: NVIDIA GeForce RTX 2080 Ti, 22528 MiB
- Diffusion model: `runwayml/stable-diffusion-inpainting`, revision `main`
- diffusers: `0.35.1`; torch: `2.8.0+cu128`
- Crack sources: 10; outputs: 0
- corrosion sources: 10; outputs: 0
- Failure count: model load failed before inference; no output rows were generated
- Smoke: FAILED for `crack_86_v1` and `corrosion_1_v1` before inference
- Failure: Hugging Face Hub metadata request to `huggingface.co` timed out; model was not cached
- Planned output directory: `results/exp13_diffusion_pilot/run_20260921_120819/`
- Contact sheets: not generated because no synthetic outputs exist
- VAL access: 0
- TEST access: 0
- training_performed: false
- SSL_performed: false
- Git commit: `01ee2b6c58c0c16132459016eaa78d5aed18379c`
- Push: completed to `origin/exp13/vlm-semantic-audit`

## Gate

`HARD_STOP_ENVIRONMENT_OR_MODEL_FAILURE`

Next step: failure report only; do not enter SSL, downstream training, or seven-class expansion.
