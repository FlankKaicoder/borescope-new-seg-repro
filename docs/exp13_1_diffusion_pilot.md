# Exp13.1 Diffusion/Inpainting Pilot

## Scope

This run is a controlled inference pilot using a pretrained Stable Diffusion inpainting pipeline. It is not generative-model training or fine-tuning. The run is restricted to Crack and corrosion because the preceding visual review authorized only those two classes.

## Inputs and selection

Inputs are existing Exp13.0 TRAIN-only samples and polygon annotations. No VAL or TEST content was accessed. The deterministic selection rule is the first 10 unique source_id values per class after stable numeric source_id sorting from `results/exp13_vlm_semantic_audit/sample_manifest.csv`; each source has two fixed variants with seeds 13001 and 13002.

Masks are generated from the existing polygon annotations. Original binary masks and separately derived inpaint masks are retained. White denotes editable/inpaint pixels and black denotes preserved pixels. The fixed derivation uses a 3-pixel dilation and Gaussian feather sigma 1.0.

## Prompt provenance

The prompts are human-reviewed, VLM-oriented, class-conditioned semantic templates. They are not automatically generated per-image VLM outputs. The Crack and corrosion templates are stored under the run `prompts/` directory.

## Model and execution

The selected conservative checkpoint was `runwayml/stable-diffusion-inpainting` with diffusers `0.35.1`, transformers `4.55.4`, accelerate `1.10.1`, safetensors `0.6.2`, and torch `2.8.0+cu128`. The intended configuration is fp16, 512x512, 20 inference steps, guidance scale 7.5, and strength 0.95.

The minimum smoke test attempted one Crack and one corrosion output. Model loading failed before inference because the Hugging Face Hub request timed out and the checkpoint was not cached locally. No image was generated, and no retry or alternate unauthorized model was used.

## Constraints and next Gate

No dataset, labels, or splits were modified. No SSL, YOLO training, downstream evaluation, or seven-class expansion was performed. The run is a model-runtime hard stop and cannot proceed to quality review without a successful, separately authorized rerun.

Gate: `HARD_STOP_ENVIRONMENT_OR_MODEL_FAILURE`
