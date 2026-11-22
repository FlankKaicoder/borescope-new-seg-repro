# Exp17 J2 real-TRAIN engineering handoff — 2026-11-23

**EXP17_J2_ENGINEERING_COMPLETE_PENDING_FORMAL_TRAINING_GATE**

The 521-Aligned data release, native augmentation replay, shared differentiable N prefix, BN comparison, four-state TRAIN gradient calibration and bounded real-data Smoke have passed their engineering checks. The sole real-data Smoke completed **50 attempts / 50 effective joint AdamW updates**, with zero AMP skips. This exhausts this stage's optimization budget. Formal training, VAL and TEST remain disabled. This delivery establishes engineering readiness for an independent Training Protocol Gate; it makes no J2 performance claim.

## Identity and preserved baselines

The isolated branch is `phase/exp17-j2-pseudonormal-pair`, starting from audited Exp17-F commit `0ce2225e224e49f816675f7097692650d4f2c4b1`. The accepted Exp15 branch remains at `401210427a61f3430f18bb4fe93261c911f66080` and Exp16 at `2f1d3b8d1bc10d635099867fab4a935d6cdb88cb`. Their training code, protocols, checkpoints and results are unchanged. Historical handoffs describe the evidence available at their own stage; the user's explicit 521-Aligned Data Gate now supersedes the earlier zero-trusted-Pair status without rewriting it.

J1 is imported and wrapped under the independent `tools/exp17/engineering/` package. The frozen protocol's canonical Git bytes retain SHA256 `b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca`. The Windows working copies have historical CRLF bytes; both canonical and executed identities are recorded separately. The four reused J1 source files match the actual historical training snapshot, and their Git LF identities match the frozen Exp15 branch. No historical file was normalized. The existing server repository is an older research checkout and was not used as an execution identity or checked out to another branch; execution used independently named, byte-verified source snapshots.

The frozen COCO file SHA256 is `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`. Fresh seed42 construction reproduced native state `8e0479ecdae6dc60558ed44d5d62c01f8b61075514bc34c109295950603fc886` and shared projection state `83664f93724017b43a2861826141fc41074a0ae58729d170829f787395d1fc6e`.

## Immutable 521-Aligned release

The release was built from the exact private Exp17-F delivery pointers, not inferred filenames. It preserves 459 historical exact-Aligned regional inheritances and 62 new user acceptances as different evidence tiers. Each row retains original Polygon, source/group/class identity, one frozen A10 N per source, A/N hashes, matrix and Valid identities, original/N coordinates, review/version provenance and `THIS_POLYGON_ONLY` scope. The immutable JSON and CSV are private.

| Release object | SHA256 |
|---|---|
| 521-Polygon JSON | `fe6a36601c6c8d90cfddfc19ab58335552c18653b66f3535e84c97b4c222dc22` |
| 521-Polygon CSV | `b5bf196b735101f28cf02c51a3297c2a1043c101bbce4473136bc67ebf4b462c` |
| Original native TRAIN identity map | `289848daad87bbb8601251a88305d58dd48f41963f7d6adff0c4777603b14c4e` |

The release comprises **521 Polygons / 345 source images / 327 independent groups**. All 1,266 native label rows were independently reconstructed from professional JSON and matched the frozen labels exactly. The original B population remains all **668 TRAIN images / 1,266 instances / 632 groups**. Population-level other-group B availability does not imply availability in a particular batch.

| Class | Admitted Polygons | Independent A/N groups | Full TRAIN B groups | Valid A/B/N exposures in read-only trajectory |
|---|---:|---:|---:|---:|
| Burn | 54 | 32 | 139 | 5,300 |
| Crack | 52 | 39 | 75 | 5,027 |
| Dent | 94 | 84 | 112 | 9,199 |
| Material missing | 80 | 77 | 150 | 7,788 |
| Tears | 40 | 38 | 43 | 3,385 |
| Tip curl | 0 | 0 | 18 | 0 |
| corrosion | 201 | 72 | 193 | 19,063 |

Groups overlap across classes and must not be summed. Tip curl still has native YOLO/SupCon training, but **no Pair auxiliary coverage**. The 35 historical Raw, three newly accepted Raw, four Crop and one Bounded records remain outside this main release; four rejects and eight technical HOLDs are also excluded. No held record was promoted to satisfy coverage.

The builder checked and postchecked 3,056 unique files / 462,597,844 bytes. All 345 bound N files already existed in the frozen server A10 release and were reused after hashing. The new small deployment contains manifests and 208 Valid assets, not another N image copy. Original images, labels, N, matrices and review files remain unchanged. See [public release summary](../../results/exp17_j2_engineering/data_manifest_summary.json); per-asset paths, identities and human notes remain private.

## Real native augmentation and sparse computation

`augmentation.py` observes the actual Ultralytics 8.4.117 operation sequence: source resize/Letterbox geometry, perspective/affine matrix, target filtering/order, padding, flips, channel order, HSV draws and their actual lookup tables. Mosaic is explicitly recorded as disabled under this protocol. N reuses these recorded operations and interpolation; it does not draw a second augmentation. Polygon identity is bound through the native label ordinal to the original professional-JSON Polygon. Actual transformed Valid support is checked before Pair admission. Missing N, missing cross-group same-class B or incomplete transformed support skips only Pair; the original YOLO/SupCon target remains intact.

Four fixed actual CPU batches cover 79 eligible anchors in 65 active native image slots, including 10 slots with multiple instances, four small P3 support cases and five frame-touching targets. Replaying each A image as N produced **bitwise equal native image, mask, class and box tensors**, while leaving all input identities and RNG states unchanged. No real N image decoding, model forward or optimizer update was needed for this replay proof. Synthetic CPU tests additionally cover identity, flips, scale, translation, combined affine, Letterbox/padding, Polygon/Valid support and image boundaries.

A runs the unchanged complete native model. B is drawn only from real regions already in that same native batch, with the same class and a different frozen independent group. N occupies only eligible sparse slots and executes shared layers **0 through 16**, ending at the existing fused P3; production N never runs the native detection/segmentation head. N is attached to autograd and shares the original 64→128→128 projection with A/B. There is no Normal class, N YOLO loss, extra B sampler, second optimizer or auxiliary CE head.

The ranking remains `hinge(mean_B_group_mean_distance - d(A,N) + 0.20)`, reduced by Polygon within A group, then A group within class, then class. Duplicate Polygons from one B group do not become independent B sources. The unchanged J1 SupCon is weighted by 8 at temperature 0.10. The calibrated engineering joint objective is `L_YOLO + 8 L_SupCon + 4 L_Pair`.

## Full read-only Batch32 coverage and fairness

The native seed42 100-epoch data trajectory was replayed without model-batch forwards or optimization. All **2,100 batches** match the sealed J1 reference for original image order, six augmented/native tensor identities, instance exposure, learning rate and 63-position warmup. Forward/update guards and before/after model, BN, optimizer, scaler and EMA identities passed. The inherited model constructor still performs its native synthetic stride initialization; the zero-forward statement concerns real traversal batches. Coverage consumed **zero N image decodes**; it determines potential support-eligible N exposure rather than claiming an N model run.

| Read-only trajectory quantity | Count |
|---|---:|
| Original image exposures | 66,800 |
| Pre-augmentation GT exposures | 126,600 |
| Post-augmentation GT exposures | 123,019 |
| Native region exposures with P3 cells | 123,009 |
| Native targets without P3 cells | 10 |
| Admitted A exposures surviving native augmentation | 50,875 |
| A/B/N candidates before transformed Valid check | 50,239 |
| Valid A/B/N anchor exposures | 49,762 |
| Potential sparse N image-slot exposures | 33,284 |
| No cross-group same-class B skips | 636 |
| Incomplete transformed Valid support skips | 477 |
| No admitted N skips | 72,134 |
| Batches with zero valid A/B/N | 0 |

Every admitted Polygon and all 327 A/N groups participate at least once. There are 7–61 valid anchors per batch; the batchwise counts are not independent research samples. Of raw admitted Polygon exposures, 49,762 / 52,100 = 95.51% remain Pair eligible; among admitted survivors the rate is 97.81%. Tear B scarcity accounts for 495 of the 636 no-B skips. Detailed per-batch/per-epoch and private group-frequency ledgers are retained. No sampler was changed to improve these figures.

## BN comparison and prefix equivalence

On the fixed sparse N=A control with 14 eligible slots, independent N statistics produce relative P3 L2 error **0.2321533** and maximum region cosine distance **0.04813135**. Replaying the actual A statistics gives **0.00142914** and **5.96046e-7**, within the predeclared AMP limits of 0.01 and 1e-4. Both modes add zero BN running-buffer updates. The recommendation **`replay_a`** was recorded before gradient calibration and chosen to preserve A/N normalization comparability, not to minimize Pair loss.

A statistics are detached within-step references. Native A normalization and its running updates remain unchanged; N neither adds a long-term BN state nor differentiates through A's reference statistics. This is an explicit statistical/gradient assumption requiring independent method review, not a claim that all possible counterfactual normalization choices are equivalent.

Full N is retained only as a zero-update reference for prefix equivalence. Original prefix and checkpointed prefix matched full N P3, embeddings and all parameter gradients **bitwise** under the same input/model/BN state.

| N-only forward/backward reference | Peak allocated bytes | Median seconds, 3 repetitions |
|---|---:|---:|
| Complete N reference | 2,343,268,864 | 0.059778 |
| Original shared P3 prefix | 1,700,038,656 | 0.048172 |
| Checkpointed shared prefix | 1,778,528,768 | 0.083226 |

The original prefix was selected, with **`checkpoint_prefix=false`**. Activation checkpointing remains a tested unselected option: it used more measured memory and time here. These isolated timings include the first call in the recorded repetitions and do not estimate whole-training speedup.

The bounded CPU decode cache preserves exact tensors, masks, bindings and RNG. On the same fixed batch, uncached preparation median was 1.475142 seconds, warm-cache median 1.288103 seconds, and cache cold preparation 1.763728 seconds. The retained decoded bytes were 19,784,704, below the 512 MiB cap. This measures application-cache preparation only, without a model or OS cache flush; cold and warm behavior differ.

## Four-state gradient calibration and the explicit λp extension

The calibration uses fixed native batch positions **1, 7, 11 and 21** at fresh COCO initialization and unchanged J1 seed42 checkpoints at epochs **5 / 50 / 100** (updates 105 / 1,050 / 2,100). Both BN modes were measured. All 16 state/batch probes retained the original model, BN, RNG and optimizer identity; optimizer attempts and updates were zero. Gradient bases use native AMP and unscaled gradients, retaining loss, layer/class norms, cosine conflicts, hinge activation and joint clipping evidence.

Before calibration, the rules set a 5% median Backbone Pair-to-J1-joint norm reference, state-median bounds **0.5%–10%**, a maximum individual probe of **15%**, and persistent strong-conflict/class-dominance screens. The denominator is `norm(grad(L_YOLO) + grad(8 L_SupCon))` on shared Backbone layers 0–10; loss scalar sizes do not select λp.

The suggested seven candidates **0.05 / 0.10 / 0.20 / 0.30 / 0.50 / 0.75 / 1.00 all failed only the minimum-strength screen**. That original result remains `HOLD_NO_FIXED_LAMBDA_PASSES`. At λp=1, the four state-median percentages are 0.1505%, 0.4016%, 1.8702% and 2.1738%. The original engineering plan and result were not edited to pretend the suggested search passed.

The user's candidate list was a suggestion, not an upper-bound authorization. A separate disclosed plan evaluated **one additional TRAIN-only fixed candidate, λp=4**, analytically using the same 16 recorded gradient Gram bases. It is explicitly `ADDITIONAL_TRAIN_ONLY_CANDIDATE_NOT_PREREGISTERED`. No new probe, model forward, optimizer update, VAL access or screen relaxation was added. The unchanged strength bounds yield the feasible interval **[3.3227706543, 4.6003000519]**; λp=4 lies inside it. The supplementary receipt is `PASS_SINGLE_ADDITIONAL_TRAIN_LAMBDA_CANDIDATE` and is independently bound to the original HOLD, unchanged rule/source hashes, 521 manifest and extension plan.

| Frozen model state | Median norm(4 gPair) / norm(gYOLO + gSupCon8), Backbone |
|---|---:|
| COCO initialization | 0.601907% |
| J1 epoch5 | 1.606429% |
| J1 epoch50 | 7.480641% |
| J1 epoch100 | 8.695085% |

Maximum individual-probe ratio is **11.776838%**; maximum recorded model-layer ratio is **12.361182%**. Pair's projection norm is at most **28.931281%** of the original weighted SupCon projection norm. All pooled class-gradient norm shares pass the unchanged 65% screen; the largest is Dent at 21.1885%. These are norm-share summaries rather than an additive causal decomposition of the joint gradient.

The screening does **not** mean conflict-free gradients. Epoch50 batch7 has Backbone Pair/YOLO cosine **−0.743903** and a layer0 cosine near −0.830848. Epoch100 batch11 has layer0 Pair/SupCon cosine near −0.537149. No frozen state-median persistent-conflict condition is triggered. Individual class dominance is also heterogeneous: a recorded Material missing class share reaches about 60.7% in one end-state batch and Dent about 58.2% in another.

Global clipping is substantial. In the λp=4 algebra, clip coefficients range **0.0027844–0.1233642**; all probes are clipped at 10. Relative to the same J1 gradient, adding Pair changes the coefficient by about 0.9727–1.0543×. This fixed λ is not approximately 5% throughout training: early and late intervention differ materially. Gradient evidence on historical J1 states does not predict the future J2 trajectory or AdamW displacement. The following real Smoke provides numerical/update evidence, not performance validation.

## Bounded real-data GPU Smoke

The single independently accounted run used fresh COCO seed42, the immutable 521 release, original 668 TRAIN, **640 / batch32**, fixed SupCon 8, Pair 4, BN `replay_a`, native AdamW/AMP and global clip 10. It completed **50 attempts, 50 native optimizer calls, 50 effective updates and zero AMP skips**. The permanent ledger reserves attempts before computation; the global claim and ledger cannot be reset by opening a new directory. Its 50-attempt cap is exhausted. No additional Smoke may be launched under this authorization.

All 50 native batches matched the historical J1 trajectory. Each step had finite loss, scaled/unscaled gradients, FP32 parameters and AdamW state; the shared Backbone and projection really changed, and N P3 had nonzero unscaled gradients. The first native warmup step changed only the bias optimizer group, as prescribed by the original LR trajectory; this does not mean every parameter changes on every step. Actual native optimizer parameter names/groups were rechecked, with no added optimizer or update. A BN counters advanced once and N added zero running-stat updates. N processing left Python, NumPy, Torch CPU and CUDA RNG unchanged. EMA advanced through the same existing numerical-audit optimizer path, once per effective update.

| Actual Smoke measurement | Result |
|---|---|
| YOLO loss range | 391.5436–493.1584 |
| Unweighted SupCon loss range | 3.46633–4.44092 |
| Unweighted Pair loss range | 0.08890–0.21814 |
| Total loss range | 421.3509–527.2703 |
| Pre-clip global gradient norm range | 234.3698–3,620.6663 |
| Clip coefficient range | 0.0027619–0.0426676 |
| Nonzero N P3 gradient norm range | 0.0129688–0.0726596 |
| Peak allocated GPU memory | 7,111,184,896 bytes |
| Peak reserved GPU memory | 7,379,877,888 bytes |
| Mean audited joint forward/backward step | 1.694623 seconds |
| Median audited joint forward/backward step | 1.024205 seconds |
| Mean end-to-end iteration, including replay/binding | 2.906841 seconds |
| Training-loop wall time / 50 updates | 147.062982 seconds / 2.941260 seconds per update |
| Full run including checkpoint audit | 151.892587 seconds |

These measurements include startup/outliers; the step timer excludes data replay and binding, while full-loop timing includes loop overhead. The objective need not decrease monotonically in this short native-augmentation run. **50 updates do not reach the complete 63-position warmup**, which is a remaining coverage limit imposed by this stage's cap.

The engineering checkpoint is tagged `exp17_j2_engineering_smoke_only_v1`, with file SHA256 `5ce54eb3ecb6e39b8d335dc74064b965d229da8000adc7e23b5a4f86839fc2fb`. Its native FP32 state SHA256 is `7165da16f3ca1dadf320e10fe0ca77cc9757fa82e93e92d14137b54155d3f881`. Strict master, EMA, AdamW and GradScaler reloading passed. Master and EMA are distinct. Removing the projection restores a strict native seven-class YOLO model and passes a synthetic CPU structural forward. This checkpoint is **not a formal endpoint and must never initialize or resume formal J2 training**.

## Tests, preserved engineering stops and remaining scientific limits

The final deployed source passed **70 CPU tests: 49 engineering/component tests and 21 replay/coverage tests**. Tests cover release identity, source versions, grouping, same-source/wrong-class B rejection, missing B/N, native-ordinal Polygon binding, Valid masks, RNG, BN, projection/ranking/margin, gradient screens, checkpoint identity, strict reload and durable authorization/budget guards. Real CPU-replay, GPU equivalence, four-state gradients and numerical Smoke provide complementary evidence; synthetic CPU tests alone are not claimed as real AMP validation.

Two ordinary zero-update engineering issues are retained rather than erased. The r2 test packet initially omitted an already-existing historical ledger fixture; the byte-identical fixture was added to the isolated packet and those tests then passed. The first read-only coverage observer stopped before batch0 because actual frozen Polygon metadata used flat coordinates; a strict finite even-length reshape adapter fixed only the observer's in-memory representation. Input manifests and labels were unchanged. The stopped outputs remain preserved. A generic interpreter probe lacking the pinned environment was likewise retained; execution used the verified original environment. None involved an attempted optimizer update, NaN/Inf recovery or hyperparameter retry.

Important remaining limits are six-class Pair coverage, sparse batchwise B availability, heterogeneous class/gradient exposure, historical J1 rather than future J2 calibration states, and detached A-statistics normalization as an explicit modeling assumption. Aligned regional acceptance does not prove artifact neutrality or eliminate changes in surrounding receptive-field context. Pair/SupCon loss behavior does not establish segmentation quality. No mAP, VAL model forward, TEST read, new image generation, new registration, auxiliary CE or extra seed was performed in this stage.

## Postchecks, resources and publication

The final read-only postcheck rehashed all **553 runtime N/Valid assets**, the immutable manifest/native identity, the engineering checkpoint and budget ledger, and all **374 deployed code files**. No identity mismatch was found. Original COCO, native TRAIN and the three historical J1 calibration snapshots are separately rechecked in the final runtime identity receipt. No optimizer update or model forward occurred during these postchecks; no active GPU compute process remained.

The actual environment is Python **3.12.3**, PyTorch **2.8.0+cu128**, CUDA **12.8**, Ultralytics **8.4.117**, OpenCV **5.0.0**, single **RTX 3080 Ti**. This current observation supersedes older hardware notes. The 50 GiB data disk has **7,156,047,872 bytes available** at the postcheck, about **7.156 GB / 6.665 GiB**. No deletion, environment replacement, original-file move or heavyweight backup was performed. Formal-run checkpoint/log capacity must be budgeted and checked again at its future Gate; the 50-step measurements are not a guarantee of a full-run peak or time.

Private delivery retains exact original assets, source packets, inputs, source/ref hash ledgers, all stopped and successful execution receipts, per-batch/group traces, calibration gradients and the engineering checkpoint location. Public Git contains only reusable engineering code/tests, aggregate statistics, source/weight identities, protocol proposals and this handoff. Private source/group IDs, exact asset paths, images and original human notes are not published.

Publication identity is the Git commit containing this handoff and the evidence manifest. Its exact commit and independently matching GitHub API / `git ls-remote` HEAD are recorded in the final delivery message and separate private publication receipt, avoiding a circular self-hash. Both frozen reference branches and the clean worktree are checked at publication; no historical branch is advanced.

## Proposed next Training Protocol Gate

The engineering recommendation is fixed **λp=4**, SupCon **8**, temperature **0.10**, margin **0.20**, BN **`replay_a`**, original sparse prefix and no activation checkpointing. Independent ChatGPT review must explicitly assess the single-candidate extension, early/late strength, local conflicts, coverage and A-statistics assumption before granting formal execution.

For historical comparability, propose fresh frozen COCO seed42, original TRAIN668/1,266, 640/batch32, native AdamW groups/augmentation/sampling/LR/warmup/clip10, **100 epochs / 2,100 effective updates**, and the fixed **epoch100 FP32 master**, with the historical five-epoch checkpoint cadence. These remain a **proposal**, not an executable authorization; formal epoch/update/schedule/endpoint fields remain pending the independent Training Protocol Gate. Do not resume the engineering checkpoint or historical J1 checkpoints, choose `best.pt`, add N to the YOLO sampler, alter λ dynamically or extend training length automatically.

A future authorized implementation must retain all-batch native trajectory identities, actual A/B/N exposure and skip counts, per-class coverage, loss/gradient/clipping/parameter updates, finite FP32/AdamW/AMP states, strict checkpoints, and durable hard-stop evidence. Any numeric, data/source, native trajectory or checkpoint identity failure stops execution rather than changing weights, seeds or hyperparameters. Formal evaluation requires a later separate VAL Gate; TEST remains excluded. Stop after this engineering delivery and publication verification.

## Evidence navigation

[Delivery summary](../../results/exp17_j2_engineering/summary.json) · [Data release](../../results/exp17_j2_engineering/data_manifest_summary.json) · [All-batch coverage](../../results/exp17_j2_engineering/coverage_summary.json) · [Replay](../../results/exp17_j2_engineering/replay_summary.json) · [BN and prefix](../../results/exp17_j2_engineering/bn_prefix_benchmark.json) · [CPU cache](../../results/exp17_j2_engineering/cpu_cache_benchmark.json) · [Calibration](../../results/exp17_j2_engineering/calibration_summary.json) · [Smoke](../../results/exp17_j2_engineering/smoke_summary.json) · [Runtime/source identities](../../results/exp17_j2_engineering/runtime_preflight.json) · [Protocol draft](../../results/exp17_j2_engineering/formal_protocol_draft.json) · [Evidence manifest](../../results/exp17_j2_engineering/evidence_manifest.json).
