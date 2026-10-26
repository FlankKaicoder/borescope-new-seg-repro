# Exp13.2-M1 execution review

Status: `EXP13_2_M1_COMPLETED_PENDING_CHATGPT_GATE`. Twelve authorized calls succeeded, zero failed, zero retries, six complete source pairs. Generated restoration quality and model choice remain pending user blind review; trusted pairs frozen = 0.

## Authority and frozen inputs

The user supplied `EXP13_2_M0_R1_USER_INPUT_GATE = PASS / BOUNDED` and `EXP13_2_M1_AUTHORIZATION = YES / BOUNDED`, authorizing exactly the six selections below and at most twelve calls. A new `user_input_decision_overlay.json` records this explicit source; historical PENDING files were not edited. Corrosion 1 and 2 retain `HIGH_RISK_BUT_USER_ACCEPTED` without revoking the user's input decision.

| Class | Source | Crop | Native half-open box [x0,y0,x1,y1) |
|---|---|---|---|
| Crack | 469 | alt_2 | [240,0,960,720] |
| Crack | 260 | default | [90,0,630,540] |
| Crack | 470 | default | [120,0,840,720] |
| corrosion | 8 | default | [90,0,630,540] |
| corrosion | 1 | alt_1 | [48,0,588,540] |
| corrosion | 2 | alt_1 | [0,0,480,480] |

Existing M0-R1 input assets were reused after file/pixel hash, polygon coverage, TRAIN membership, common FOV and mask checks. No crop search, target movement, padding or mask-rule changes occurred. `frozen_source_crop_manifest.json` includes native image/annotation, RGB/mask hashes, dimensions, crop coordinates and decision provenance. Its SHA256 is `c5d44a226a5bba6b7222fa7dcba406b8def6eda31b537ceac409af55b8958ec0`.

## Executed protocol and environment

Common prompt: `restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting`. Steps 20, guidance 7.5, strength 0.8, seed 130500, negative prompt None, FP16. Frozen config SHA256: `a3fb76cd12797a012c3e7149571a1e50e6e4dd992fa72dd663c181c1610ff81e`.

DreamShaper used the existing `aiplayground/dreamshaper-8-inpainting` local fingerprint `9598058d591b43726c5bf5c7db7f5183911415ca18e5c123c1a63bd3ee39d927`, 19 files / 8,223,354,902 bytes, 512 input and DEISMultistepScheduler. Its immutable upstream revision remains UNKNOWN; historical ModelScope/master provenance and locally verified files are retained rather than inventing a revision. Reuse of this fingerprint was explicitly authorized.

SDXL used `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, revision `115134f363124c53c7d878647567d04daf26e41e`, fingerprint `d7fb8e6a3d8c4ac6538b13f2f2575a902662364c051716bd3961bf94a465fd14`, 19 files / 6,941,223,330 bytes, fp16 variant and EulerDiscreteScheduler. Both loaded local-only; model downloads = 0. DreamShaper used attention slicing; SDXL used CPU offload and VAE slicing/tiling as frozen. Both model loads succeeded, taking 2.774 and 0.756 seconds respectively; offload means loading time alone is not a full GPU warmup measurement.

Actual preflight hardware was one NVIDIA GeForce RTX 3080 Ti, 12,288 MiB total / 11,912 MiB free, superseding older hardware notes for this run. Python 3.12.3, PyTorch 2.8.0+cu128, Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1, NumPy 2.3.2, Pillow 11.3.0. Preflight free disk was 17,487,650,816 bytes; finalizer measurement 17,307,291,648 bytes preceded additional small native diagnostic files. No environment upgrade occurred.

This is a paired practical comparison with shared content and editing objective. Architecture, scheduler, VAE, native resolution and internal noise trajectories differ; equal numerical seed does not give identical latent noise. Six distinct image/source IDs do not prove six independent physical parts. No statistical model ranking is claimed.

## Runtime and budget

| Source | DreamShaper seconds | SDXL seconds | Pair |
|---|---:|---:|---|
| 469 | 1.714 | 12.488 | complete |
| 260 | 1.134 | 9.088 | complete |
| 470 | 1.140 | 9.310 | complete |
| 8 | 1.136 | 9.051 | complete |
| 1 | 1.135 | 8.850 | complete |
| 2 | 1.134 | 8.639 | complete |

DreamShaper peak allocated/reserved = 2,805,598,720 / 3,103,784,960 bytes. SDXL = 5,764,813,824 / 6,247,415,808 bytes. These are measured execution configurations, not equal-throughput configurations. No OOM, nonfinite decode, abnormal all-black/constant output or inference exception was detected. `execution_exit.json` records actual supervisor exit code 0 and stdout/stderr hashes. Preserved stderr includes normal loading progress, deprecation/config warnings and the frozen DreamShaper safety-checker warning.

The authoritative successor is `results/exp13_2_m1_matched_restoration_benchmark/20261026T064000Z_m1/budget_successor_ledger.json`, inheriting all S5 calls/history and original caps from `results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json` (SHA256 `3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94`). The unchanged BudgetProtocol uses cross-process flock, atomic replacement/fsync and durable STARTED. Both models were reserved atomically per source before its first call; no GPU call preceded STARTED. The internal `pilot` budget category is the specifically authorized M1 allocation, not authorization for another Pilot.

New occupied/actual/succeeded = 12/12/12; failed = 0; reserved-unstarted = 0. Cumulative occupied/actual = 16/16. Smoke stays 4/4, remaining 0; aggregate remaining 104; all-local remaining 284. Retry and GPT = 0. A CPU-only competing-process simulation separately verified whole-pair reservation under a cap, preserved inherited history, failure charging and terminal restart rejection; it used a temporary simulation ledger and never replaced either authority.

## Output checks and blind review

All twelve raw floating decoded arrays and original PNG outputs remain controlled on AutoDL under `outputs/<source>/<model>/`. Independent CPU verification checked finite arrays, shape/dtype, exact rounded float-to-PNG conversion, 512 alignment and immutable raw hashes. SDXL raw output is 1024; DreamShaper raw output is 512. Common input is 512, with SDXL RGB LANCZOS 2x and mask NEAREST 2x. Actual Diffusers mask processing was checked: white edits, effective uint8 threshold >=128; grayscale feathering is not interpreted as a soft inference mask.

Raw model outputs may change pixels outside the edit mask. They remain intact. Separately labelled `POSTPROCESS_COMPARISON_ONLY` common-512 and native composites have exactly zero outside-support changes; this is a compositing guarantee, not proof of model background preservation or restoration quality. Difference maps, local pixel zooms, polygon/input views and both raw/protected outputs are in the review package. None is frozen as a trusted pseudo-normal pair.

The A/B mapping was randomized independently per source, frozen before model loads/inference and kept under `private/` (directory mode 700, key mode 600). Public hash commitment: `cd7b165539c9446e904ee259b74df71d32484389cb86afe492658dd8e0f32a5d`. The key is excluded from Git and the review ZIP. Both slots use 512x512 RGB PNG with the same encoding and no PNG metadata; filenames/labels do not name models. Model-linked output hashes/pixel metrics remain controlled until blind review. Independent validation checked ZIP CRC, file manifest hashes, slot contents and 24 PENDING human cells. One anonymous sheet was visually checked for readable layout and correct panel labelling without semantic scoring or choosing a winner.

ZIP absolute path:
`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_m1_matched_restoration_benchmark/20261026T064000Z_m1/anonymous_blind_review_package.zip`

Size 33,967,690 bytes; 69 entries; six contact sheets / six sources. SHA256 `773e9a8942e336b6d96c10d253177f1639dff0dac9b2c4a3564a81f4f8841399`.

User fields are A_QUALITY / B_QUALITY (PASS, FAIL, UNCERTAIN), PREFERENCE (A_WIN, B_WIN, TIE), REVIEWER_COMMENT. All remain PENDING. Review each source independently; A/B assignment is not shared across sources. Evaluate residual defects, texture, hallucinated structures, physical boundaries and raw background changes separately from protected composites. Only the user can select a model; no automatic aggregate winner was produced.

## Protection, Git and stopping point

Report branch started at verified local/remote `01b730cd0471f982d5d17263cc254b04bb937da8`. Execution used an isolated minimal code snapshot at `/root/autodl-tmp/exp13_m1_workspace/20261026T064000Z_m1/code/`; executed script SHA256 `92a79029e4e5086cf1fc6c3c2e7ce0119e70a6fc58b58ae95c2ff0b8c81fee1d`. Server historical checkout stayed at `787a3be0190dfa9b11c5416a5b686a6bae4598d9`; 11,974 historical status entries and 293 prior M0/M0-R1 artifact hashes were preserved. No server pull, reset or cleanup occurred.

VAL/TEST accesses, original/split writes, new model downloads, GPT calls, SSL/YOLO training and trusted-pair freezes all 0. Only scripts, metadata, logs and reports are submitted; industrial PNG/NPY/ZIP, weights and private mapping stay controlled. This stage is technically complete, with semantic quality/model selection PENDING. Stop for ChatGPT Gate and user blind review. No further generation, model selection, trusted-pair freeze, Local Change or training follows automatically.
