# Exp13.2-M0-R1 source reselection and human input protocol review

`EXP13_2_M0_R1_PREPARED_AWAITING_HUMAN_INPUT_REVIEW`

The revised pool meets the preferred **Crack3 + corrosion3** geometry condition. It does not establish six semantically eligible inputs: all six user input decisions and selected crop IDs remain PENDING. No M1 execution is authorized. This report incorporates the user's supplemental human input-protocol Gate within M0-R1, without opening a new experiment.

Evidence: `results/exp13_2_m0_r1/20261025T122430Z_m0r1/`. CPU preparation actually ran **2026-10-25 12:24:27–12:25:35 UTC**, exit0, empty stderr; the run identifier was allocated before execution. Independent existing-ZIP validation ran **12:29:52–12:29:55 UTC**, with `PASS_PACKAGE_TECHNICAL_ONLY`. Industrial PNGs and ZIP remain controlled on AutoDL; Git contains scripts, audit metadata, logs and reports only.

## Candidate order and retained M0 facts

The first-tier order was frozen before opening new sources: **1,2,6,7,12,13,27**. A numeric ascending list of other pure-corrosion TRAIN IDs and a maximum30 actual supplemental inspections were also preregistered. Sources1 and2 were the first two inspected and both passed objective checks, so expansion stopped immediately. Priority sources6/7/12/13/27 and every supplemental source were not opened. Selection did not inspect any new or historical model output for ranking.

All four M0 geometry candidates **469,260,470,8** retain their exact default files. Actual PNG file/pixel hashes, shapes and dtype, original image/annotation hashes, deterministic crop coordinates, and the original review sheet hashes were verified. Recomputing the protocol in memory reproduced the seven original input pixel hashes per source; no original PNG was rewritten. The complete M0 result-file inventory is unchanged before/after. A mismatch would have produced PROTOCOL_HOLD rather than rebuilding an old input.

Source9 remains excluded for `ALL_TARGET_SPAN_EXCEEDS_MAX_SQUARE`: all-target horizontal span643 exceeds540. Source23 remains excluded for `NO_ALL_SIDES_NORMAL_CONTEXT_MARGIN`: target reaches native x=0. Their original M0 records, diagnostic retention and exclusions are preserved in `m0_original_hold_preservation.json`; neither was reconsidered, padded, reduced to fewer polygons or allowed a weaker margin.

| Source / class | Polygons | Default crop `[x0,y0,x1,y1)` | Default effective edit pixels512 | Qualification |
|---|---:|---|---:|---|
| 469 / Crack | 1 | `[120,0,840,720]` | 1637 | M0 default retained; geometry only |
| 260 / Crack | 1 | `[90,0,630,540]` | 8405 | M0 default retained; geometry only |
| 470 / Crack | 1 | `[120,0,840,720]` | 2997 | M0 default retained; geometry only |
| 8 / corrosion | 1 | `[90,0,630,540]` | 1323 | M0 default retained; geometry only |
| 1 / corrosion | 3 | `[67,0,607,540]` | 4860 | Newly qualified geometry; semantic pending |
| 2 / corrosion | 1 | `[24,0,504,480]` | 4005 | Newly qualified geometry; semantic pending |

Source1 native720×540 has union bbox `[84,259,571,410]`, x-origin interval48–67 and margin17. Source2 native640×480 has bbox `[40,64,160,113]`, x-origin interval0–24 and margin16. Both default crops translate the historical center window under the original rule. They retain **all** polygons and positive edit support. Source1's actual left/top/right/bottom target margins are17/259/36/130; source2's are16/64/344/367. These bounding-box margins do not prove normal-metal context.

New source image SHA256:

- 1: `4cacb0d2c49d471fada2c05f0fc006d87367e0bdd6568c623e677ed0d382d904`
- 2: `514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03`

All six image hashes and frozen TRAIN group IDs are distinct (`singleton_469/260/470/8/1/2`). This does not establish six independent physical parts or capture scenes. Each new source's complete annotation hash, polygon coordinates, topology result, pixel-center transform, native/edit mask and shared-input hashes are recorded in the audit JSONs. Raw image/JSON hashes were checked again after all exports; the split-manifest hash remains `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`.

The preferred six-source plan proposes **12 future model calls**, two per source, with **zero executable now**. If future human decisions leave only2+2, the frozen Crack preference is469 then260;470 is held. Corrosion preference is8,1,2, subject to separate acceptance. If469/260 or two corrosion sources are not approved, that fallback is HOLD; no failed source is silently replaced. The5-source/10-call3+2 plan remains available only if the approved input count and a later Gate support it. This exploratory pool cannot support a two-class statistical superiority claim.

## Unchanged input protocol and technical validation

Every option uses the original native `L=min(W,H)` square, margin `max(16,ceil(.03*L))`, all target polygons rasterized with Pillow rounded coordinates, MaxFilter7 then GaussianBlur1. RGB uses LANCZOS to512; grayscale mask uses NEAREST to512. SDXL RGB uses that exact512 image LANCZOS to1024; the mask uses exact2× NEAREST. No reflect padding, whole-image stretching, second crop, target omission or new mask rule was introduced.

The installed Diffusers0.35.1 image-processor and both pipeline source hashes still match M0. Both pipelines binarize the mask at0.5, equivalent to uint8>=128. Every option includes the **actual effective binary repair support** at512 and1024, and its red overlay on the planned input. Grayscale feather is shown separately; it must not be misrepresented as continuously blended diffusion conditioning after binarization.

For all16 options, every native polygon pixel is retained; all positive native edit support is retained; per-polygon retention is100%; no target pixel is missing after thresholded-mask inverse mapping; coordinate roundtrip error is below1e-9;1024 masks and effective supports are exact2×2 repetitions of512. Crop/mask cross-model FOVs are identical by construction. Changing a crop can change nearest-sampling phase and pixel counts, which are separately hashed and disclosed, not concealed as the default input.

The independent validator opened the existing ZIP and checked **230 unique entries**: disk/ZIP file SHA256, decoded pixel SHA256, mode, shape and dtype; exact thresholded masks and2× correspondence; manifests/templates; all16 sheets; six source rows;48 PENDING human cells; and12 blocked future slots. ZIP CRC checks pass, with no missing or unexpected entries. This is package/geometry validation, not semantic approval.

## Human input protocol Gate and alternatives

Each review sheet contains Native Original RGB; red Polygon and cyan crop box; Native Crop; original native/cropped Polygon Mask; native/cropped dilated and feathered Edit Mask; effective thresholded Mask; red repair-support overlay; planned DreamShaper512 RGB/Mask; planned SDXL1024 RGB/Mask and effective1024 Mask; hashes, crop rule, selection reason and whole-context risk prompts. The exact PNGs are also in the ZIP, so inspection is not limited to scaled contact sheets. No generated output is included.

Up to two alternative crops were proposed from the legal interval endpoints in fixed lexicographic order, excluding the default. They share the native source, square size, margin, all targets and mask rule. They are **proposals only**; the original frozen default is unchanged.

| Source | Default x-origin | alt_1 x-origin | alt_2 x-origin | y / square side |
|---|---:|---:|---:|---|
| 469 | 120 | 0 | 240 | 0 /720 |
| 260 | 90 | 0 | 175 | 0 /540 |
| 470 | 120 | 0 | 178 | 0 /720 |
| 8 | 90 | 0 | 180 | 0 /540 |
| 1 | 67 | 48 | None | 0 /540 |
| 2 | 24 | 0 | None | 0 /480 |

All six have at least one legal alternative. There are6 defaults and10 alternatives,16 sheets total. No `NO_LEGAL_ALTERNATIVE` case occurred. Every option's coordinates, full file/pixel hashes, technical checks and construction rule are in `input_crop_alternative_audit.json` and `human_input_protocol_review_manifest.json`.

Assistant layout/context-risk spot checks inspected only `1_default`, `2_default` and `260_alt_2`; they are not user decisions. Source1 retains a UI crosshair near one target and logo in the crop, plus multiple metal boundaries. Source2 retains GE/status UI, dark background/cavity and a real metal edge near the target. Source260's alternative changes the available context while retaining a dark-cavity/metal boundary. **These visible risks explain why geometry cannot grant eligibility**; they do not automatically reject every repair method or approve a crop. All human fields remain PENDING, including UI/FOV acceptability and normal texture sufficiency. The package deliberately exposes whole-crop contamination, not merely UI intersecting the white mask.

The user should compare options, set `SELECTED_CROP_ID` to `default`, `alt_1` or `alt_2` where listed, and complete `CROP_CONTEXT_ACCEPTABLE`, `MASK_COVERAGE_ACCEPTABLE`, `METAL_STRUCTURE_PRESERVED_BY_INPUT`, `UI_FOV_CONTAMINATION_ACCEPTABLE`, `NORMAL_METAL_REFERENCE_SUFFICIENT`, `FINAL_INPUT_DECISION` and `REVIEWER_COMMENT`. All options may be rejected. User semantics have highest decision authority, but coverage/bounds/hash/cross-model mismatch cannot be waived by a visual PASS. A later review record must preserve this initial PENDING template, freeze the chosen option hashes and reassess the class minimum.

Scientific context is bounded: S2 compact ROI/reflect padding produced artifacts; the user accepted S5 Crack260 using the historical center-crop protocol. Inputs and other experimental factors differ, so crop-only causality is unproven. This phase prepares an input decision; it neither selects a winning model nor grants M1.

## Controlled review ZIP

Server absolute path:

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_m0_r1/20261025T122430Z_m0r1/input_protocol_review_package.zip`

SHA256: `6f683f6521737437ef37a5f24e24581865fd59467694dca779102c15a3bf53fc`

Size: **62,181,247 bytes** (about59.30 MiB). Review sheet count: **16**, covering all **6** sources. The ZIP contains exact input assets for pixel review as well as sheets and machine-readable records. It is retained on AutoDL, excluded from public Git. The48 user decision cells remain PENDING. No image/ZIP is staged in this commit.

## Models, budget, Git and stop

Both existing model inventories were rehashed CPU-only: DreamShaper19 files /8,223,354,902 bytes and SDXL19 files /6,941,223,330 bytes match M0. No torch/diffusers/transformers/accelerate module was imported and no model/tensor was loaded. DreamShaper fingerprint remains `9598058d591b43726c5bf5c7db7f5183911415ca18e5c123c1a63bd3ee39d927`; SDXL remains `d7fb8e6a3d8c4ac6538b13f2f2575a902662364c051716bd3961bf94a465fd14`, revision `115134f363124c53c7d878647567d04daf26e41e`. DreamShaper's ModelScope/master origin still lacks an independently established immutable upstream revision; no provenance was invented.

The exact M0 protocol SHA256 remains `949c863df202ce15e39cc86495098727f669f4d3779110fcac7b8d61ea4de8e7`: prompt `restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting`, negative None,20 steps, guidance7.5, strength0.8, seed130500,FP16, DreamShaper512/DEIS and SDXL1024/Euler. Loader and component configs remain the M0 records. Model architecture, scheduler, VAE and native resolution remain confounding factors in any future practical comparison.

The authoritative S5 ledger is unchanged, SHA256 `3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94`: occupied4 / actual4, Smoke remaining0, aggregate remaining116, all-local remaining296. New inference, model loads, downloads, retry/GPT calls and reservations are all0. No successor ledger was created. A separately authorized future M1 must inherit the complete history in a single locked successor shared by both models, atomically reserve source pairs, persist STARTED before each inference, charge failures, and prohibit retries/replacement/incomplete-pair wins. New calls cannot reopen Smoke.

Raw writes, split writes, VAL/TEST file access, trusted-pair freezes, SSL/YOLO/Pilot and downstream training are all0. Only the six explicitly TRAIN-filtered image/JSON pairs were opened; the split CSV was read as metadata for filtering. Python3.12.3, NumPy2.3.2, Pillow11.3.0; no system dependency was changed. Available server disk after preparation:17,498,406,912 bytes. Final protection verification confirms all68 M0 files retain their hashes and11,754 historical server Git-status entries remain unchanged after excluding only this stage's new result directory and two new scripts. Old M0/S2/S5 evidence and historical untracked assets are preserved.

Report branch and verified remote start: `exp13/vlm-semantic-audit`, `7e859bfc20e75f47e65fbfa22e25f017e3f8879f`. Work was assembled in the existing isolated Windows worktree `.r2tmp/r3worktree`. AutoDL stays at historical `787a3be0190dfa9b11c5416a5b686a6bae4598d9`; no pull/reset/clean/force synchronization occurred. Only two uniquely named CPU scripts and this new result directory were added there. No paper repository was accessed. Large server Git-status inventory remains controlled/untracked rather than staged.

**Stop for ChatGPT and user input eligibility review.** All6 sources need a crop/semantic decision; approved inputs0, currently executable calls0, trusted pairs0. A5, M1, Pilot, SSL and training remain paused; user acceptance of the unified inputs is necessary before separately requesting M1 authorization.
