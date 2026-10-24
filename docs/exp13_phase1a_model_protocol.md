# Exp13 Phase 1-A model and geometry protocol

Status: `EXP13_PHASE1A_COMPLETED_PENDING_CHATGPT_GATE`. This is a protocol and deterministic geometry audit only. No Diffusion checkpoint was downloaded, loaded, or run.

The preferred candidate is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`; the v1.5 inpainting checkpoint is a fallback and DreamShaper is historical reference only. License, revision, exact mask behavior, cache size, and runtime memory remain `NEEDS_ONLINE_VERIFICATION` or `NEEDS_RUNTIME_SMOKE`. A 1024x1024 canvas on a 12 GB RTX 3080 Ti is a high-memory-risk configuration; FP16, attention slicing, VAE slicing/tiling, and CPU offload are mitigations, not evidence of success.

The candidate configuration fixes seed policy, prompt, negative prompt, mask definition, 20 steps, guidance 7.5, and exploratory strengths 0.75/0.90. Formal production would freeze one strength after a separately authorized smoke/pilot decision. A local call is exactly one ROI, one strength, one actual inference. Planned and actual calls are separate counters; retry calls are separately identified and GPT fallback is disabled without authorization.

The current result bundle contains no generated images. It records a TRAIN-only metadata audit, synthetic-array geometry tests, and explicit stop conditions. The next action is ChatGPT gate review; do not run Diffusion smoke automatically.
