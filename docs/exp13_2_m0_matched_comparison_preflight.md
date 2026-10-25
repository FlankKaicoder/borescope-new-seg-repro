# Exp13.2-M0 matched restoration benchmark preflight

Status: `EXP13_2_M0_SOURCE_OR_PROTOCOL_HOLD` / `SOURCE_SELECTION_HOLD`.

The fixed six-source pool cannot meet the required two corrosion sources under the registered all-target, inside-native square and context rules. Do not approve M1 with this pool. M0 completed CPU evidence preparation with zero model loads, inference, downloads, VAL/TEST image access, raw writes, or training. No trusted pair was frozen.

Evidence: `results/exp13_2_m0_matched_model_benchmark/20261025T115110Z_m0/`. CPU execution ran 2026-10-25 11:54:01–11:54:50 UTC, followed by independent sampling/topology checks at 11:56 UTC. The run identifier was allocated before execution. Main stdout/stderr and exit code 0 are retained.

## Candidate findings

All six source image and annotation hashes match the frozen TRAIN manifest (`35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`). They have six distinct image hashes and six distinct split group IDs, but physical-part and capture-scene independence remains unproven. No VAL/TEST image or annotation was opened. All 15 polygons have finite, in-bounds, nondegenerate geometry and no nonadjacent edge intersections.

Coordinates below are native integer half-open `[x0,y0,x1,y1]`. Geometry-only candidates remain pending whole-context semantic review; none is RESTORATION_ELIGIBLE.

| Source | Class / polygons | Registered crop | Target retention | Decision |
|---|---|---|---|---|
| 469 | Crack / 1 | `[120,0,840,720]` | 100% | Geometry only; semantic HOLD |
| 260 | Crack / 1 | `[90,0,630,540]` | 100% | Geometry only; semantic HOLD |
| 470 | Crack / 1 | `[120,0,840,720]` | 100% | Geometry only; semantic HOLD |
| 9 | corrosion / 9 | None | Center diagnostic 93.583803% | Pairwise excluded: union width 643 > maximum square 540 |
| 23 | corrosion / 2 | None | Center diagnostic 90.937864% | Pairwise excluded: target reaches x=0, no left context |
| 8 | corrosion / 1 | `[90,0,630,540]` | 100% | Geometry only; semantic HOLD |

Source 9 has raster union bbox `[72,55,715,489]`. No translation of an inside-native square can preserve all nine targets. Source 23 has bbox `[0,38,454,286]`; shifting the center window left could contain the polygons, but cannot create real normal-metal context left of x=0. Its minimum margin is 16 pixels; source 9/260/8 require 17 and source 469/470 require 22. Exclusion follows the registered rule, not historical model quality. No replacement source, reflect padding, reduced target union, or relaxed margin was used.

The class gate fails even if every geometry-only candidate later passes visual review: **Crack 3 / corrosion 1**, requiring at least 2 in each class. The four remaining sources are not an approved four-source batch. The previews of 9/23 deliberately retain center-crop diagnostics and are labeled `DIAGNOSTIC_ONLY / NOT_FROZEN_FOR_INFERENCE`.

## Frozen input construction and hashes

Before building inputs, `preregistered_input_rules.json` froze `L=min(W,H)`, margin `max(16,ceil(0.03*L))`, and deterministic clamping of the center origin to the feasible all-polygon-plus-margin interval. The margin is an engineering rule, not a scientifically validated definition of adequate metal context. Full-square UI/FOV/gray border, normal-metal continuity, structural boundary crossing, unannotated defects, and restoration-target clarity are still pending review. Unmasked fraction is recorded only as a proxy.

The native Pillow `round(x),round(y)` polygon union uses MaxFilter(7), then GaussianBlur(1.0). RGB and edit mask share the exact crop; RGB uses LANCZOS to512, mask NEAREST to512. DreamShaper would consume that unique512 baseline. SDXL would consume its LANCZOS1024 RGB / NEAREST1024 mask expansion. There is no padding or implicit second crop. Pixel-center forward/inverse matrices and raw file hashes are in `shared_crop_manifest.json`.

| Source | Common RGB512 pixel SHA256 | Common mask512 pixel SHA256 |
|---|---|---|
| 469 | `de6e31b7776e3afc175f23b60745f2741ff40fad59d993b7cccc2c2923520a7e` | `6747e93608205addfd25429c894539d6debee3797847132c58723e51a8b1c724` |
| 260 | `0b7d36965e7b2f8cdb48487eb3f68a34175c65a0040727ffb2417196ede7d657` | `862fd09af676925a2843c5b8ff05f438e3bd2e048e5de0748450edcd7ed237d5` |
| 470 | `2f709ba7013194b2b11c2537b1b08ab2e5a4c9368e1d08c57812e4521023ebd5` | `5b4109a0e6f44ccbbb9f59e15aa232c400c56a67e24c3294009bf443edca7e15` |
| 8 | `e9e32051445fef268f83000725dd59bfc579fbe656c4639c1fccccc278265b93` | `885045fef8612a23ef6e23df4896220b55633a7b5f611e0446919f385fb5ab90` |

Complete native image/annotation, PNG-file, pixel, and1024 hashes, modes, shapes, per-polygon retention and diagnostic hashes for rejected inputs are in the manifest. These are image pixel-byte hashes, distinct from PNG encoding hashes. Native images/annotations were hashed again after input preparation and remained unchanged.

All four geometry candidates retain 100% native polygon and edit support. Coordinate roundtrip error is below 1e-9. Every1024 mask equals the exact2x2 repetition of the512 mask. Independent checks of the installed Diffusers0.35.1 sources confirm both inpainting pipelines binarize at0.5 (uint8 threshold128), with default `padding_mask_crop=None`. All retained polygon pixels remain covered after thresholding and nearest inverse mapping. These checks establish input alignment, not semantic defect removal. Rejected diagnostics only test the retained part, and cannot pass the full-target gate.

## Static model assets and protocol

Both existing local asset sets are complete for the registered loaders; M0 read file bytes and safetensors headers without loading tensor values or importing torch/diffusers/transformers. The installed package versions were read using distribution metadata. No network model acquisition occurred.

| Model | Pipeline / scheduler | Verified local files / bytes | Provenance |
|---|---|---|---|
| DreamShaper | StableDiffusionInpaintPipeline / DEISMultistepScheduler | 19 / 8,223,354,902 | Historical ModelScope acquisition, ID `aiplayground/dreamshaper-8-inpainting`, mutable `master`; local hash freeze only |
| SDXL | StableDiffusionXLInpaintPipeline / EulerDiscreteScheduler | 19 / 6,941,223,330 | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, full revision `115134f363124c53c7d878647567d04daf26e41e`; matches accepted19-file manifest |

DreamShaper audit includes both full and FP16 weights, plus safety-checker assets. Its registered historical loader keeps `variant=None`, full safetensors cast to FP16, attention slicing, and `safety_checker=None`. SDXL keeps `variant=fp16`, safetensors, CPU offload, VAE tiling/slicing. Both use local-only loading in any separately authorized future phase. No runtime compatibility claim is added by M0; DreamShaper's historical pipeline was used for defect variants, while future restoration is a different objective.

DreamShaper fingerprint: `9598058d591b43726c5bf5c7db7f5183911415ca18e5c123c1a63bd3ee39d927`.
SDXL fingerprint: `d7fb8e6a3d8c4ac6538b13f2f2575a902662364c051716bd3961bf94a465fd14`.
Each fingerprint covers the recorded ordered file/hash/header inventory. DreamShaper has no authenticated immutable upstream revision/hash baseline; an embedded config path is not revision proof. Its local model card declares `creativeml-openrail-m` (card hash and attribution retained in `cpu_validation.json`); this is local evidence, not renewed upstream license verification. Existing SDXL provenance/permissions remain unchanged. No new download was requested. Available disk after M0: 17,620,918,272 bytes.

The preregistered comparison task is `RESTORE_TO_PSEUDO_NORMAL_METAL`. Exact shared prompt:

`restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting`

Negative prompt None;20 steps; guidance7.5; strength0.8; seed130500; FP16; single GPU sequential execution. DreamShaper512 versus SDXL1024 and their native schedulers are fixed. All component configs, scheduler settings and VAE configuration are retained in `model_asset_preflight.json`. Different architecture, scheduler, native resolution and VAE are residual comparison factors; equal seed/strength/guidance do not imply identical internal noise. This is a paired practical adaptation study, not a pure weight-causal experiment or seven-class statistical ranking. No VAE numerical fault is inferred from static config values.

## Future batch and blinded review

`planned_call_matrix.csv` lists the12 possible source/model slots, all blocked. Requested future ceiling is12; geometry-conditional maximum is8; **currently executable count is0**. No batch or budget successor has been opened. Changing source pool or eligibility rules requires a new Gate, not silent continuation to M1.

After separate authorization only, use one persistent cross-process-locked successor inheriting every S5 call/history/counter, with a hashed parent. Atomically reserve each source's two model calls as a pair and persist STARTED before each actual inference. Failure charges budget, with no retry/search/replacement; incomplete pairs do not count in wins. Pair-reservation integration with `tools/exp13_restoration/protocol.py` must be verified before execution. The historical88 DreamShaper outputs/20 groups are separate provenance, not S5 calls or new charges.

`blind_review_protocol.json` fixes two views: unchanged raw outputs and identically protected composites marked `POSTPROCESS_COMPARISON_ONLY`. Both outputs align to512 using LANCZOS. Both composites use the same common grayscale alpha/support and original RGB, preserving support-exterior pixels exactly. Difference threshold is maximum RGB-channel change >5, with both positive-support and threshold128 regions reported. Review panels use identical fields of view, zoom and scale; runtime data is separate.

After freezing complete future output inventory, randomly assign balanced anonymous A/B labels by source, store the private mapping outside review assets/Git, and publish its hash commitment before review. No mapping exists in M0. The user separately records each model's PASS/FAIL/UNCERTAIN, A_WIN/B_WIN/TIE, reason, defect removal, metal texture, mechanical hallucination, structural preservation, seams, raw-background changes and composite acceptability. Final model selection belongs to the user. Count sources, not pixels or two outputs as independent sources; no statistical superiority claim.

## Latest decisions, budget and preservation

`latest_user_decision_overlay.json` records the M0 user prompt as its source: historical eight DreamShaper sources are visually accepted at source level, preserving crop scopes and distinct historical variant decisions; SDXL S5 Crack260 visual effect is now accepted. This supersedes the previous *pending visual feedback* prospectively without editing S5. It does not approve all variants, formal quality or training pairs. S2 failures remain recorded. A5 pairing engineering and Local Change training are paused pending a future user-selected single model.

Authoritative S5 ledger SHA256 before/after: `3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94`. Occupied/actual4/4, Smoke remaining0, aggregate remaining116, all-local remaining296, retries0, GPT0. No ledger was created or mutated. This phase added0 calls,0 model loads,0 downloads,0 VAL/TEST file accesses,0 raw writes,0 SSL/YOLO training,0 trusted-pair freezes.

Reporting branch began at `6d515c9dfaf6b624a7da727257b67cbe369b7791`, also verified as the GitHub remote starting HEAD after transient network timeouts recovered. CPU scripts ran in the server's preserved old checkout at `787a3be0190dfa9b11c5416a5b686a6bae4598d9`; there was no pull, reset or cleanup. Historical server status entries are preserved. Reports are assembled in the existing isolated Windows worktree on `exp13/vlm-semantic-audit`; primary checkout and historical untracked assets are retained.

Six per-source review PNGs, a six-source overview and input files remain in the controlled AutoDL result directory. Git contains only CPU scripts, metadata/logs and reports, with no industrial PNG/ZIP, weights or cache. No paper repository was accessed. Stop for ChatGPT audit. **M1 is not recommended under the current fixed pool.**
