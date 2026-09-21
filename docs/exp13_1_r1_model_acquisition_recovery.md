# Exp13.1-R1 Model Acquisition Recovery

## Recovery reason

The original Exp13.1 run `run_20260921_120819` stopped before inference because the Hugging Face Hub request for `runwayml/stable-diffusion-inpainting` timed out. That failure is preserved and is not a conclusion about generated-image quality.

## Acquisition path

ModelScope metadata connectivity was verified with the official Python API. Candidate A, `xichen/cleansd`, downloaded successfully to `/root/autodl-tmp/model_cache/exp13/xichen_cleansd` (about 6.9G), but its `stable-diffusion-inpainting` subdirectory declared a VAE in `model_index.json` while lacking a `vae/` directory and VAE weights. It was therefore not treated as a self-contained local inpainting pipeline.

Under the authorized fallback rule, Candidate B was selected: `aiplayground/dreamshaper-8-inpainting` from ModelScope, revision `master`, license `creativeml-openrail-m`. This is an SD1.5-family inpainting checkpoint but a changed model identity, recorded as `MODEL_CHANGED_FOR_ACQUISITION_RECOVERY`. Its local snapshot is `/root/autodl-tmp/model_cache/exp13/dreamshaper-8-inpainting`, approximately 7.7G, and contains model index, tokenizer, text encoder, VAE, UNet, scheduler, and safety checker. Key hashes are in the run `model_info.json`.

## Smoke and pilot

The local pipeline was loaded offline with diffusers `0.35.1`, transformers `4.55.4`, accelerate `1.10.1`, safetensors `0.6.2`, and torch `2.8.0+cu128` in fp16 on an RTX 2080 Ti. Crack source 86 and corrosion source 1 each passed one-variant smoke with seed 13001 and 512x512 output validation.

The formal recovery run reuses the frozen TRAIN-only selection and prompts: Crack 10 sources and corrosion 10 sources, two variants per source with seeds 13001 and 13002, for 40 successful outputs. No VAL or TEST data was accessed. The previous failed run was not overwritten.

## Boundaries

Prompts are human-reviewed VLM-oriented class-conditioned templates, not automatically generated per-image VLM outputs. No model training, fine-tuning, SSL, YOLO, downstream evaluation, seven-class expansion, or dataset modification was performed.

## Gate

`MODEL_ACQUISITION_RECOVERED_PILOT_COMPLETED_WAITING_QUALITY_REVIEW`

The next action is quality review only. Do not enter SSL or downstream training before review.
