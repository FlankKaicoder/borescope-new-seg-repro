# Exp17 J2 synthetic prototype: execution and independent boundary review

The bounded engineering prototype passed. This is evidence that the proposed shared A/B/N graph can be differentiated, jointly updated and serialized on synthetic inputs. It is **not** a real-Pair Data Gate or formal-training readiness result. Trusted real Pairs remain **0**, formal `lambda_p` remains **null**, and real-Pair training remains **HOLD**.

This review used the actual GPU receipt and preserved attempt ledger, then checked the implementation against those claims. No additional GPU optimization was run during the review. The registered synthetic budget is exhausted at **3 attempts / 3 effective updates**.

## Evidence identity

| Item | Frozen identity |
|---|---|
| Executed source commit | `8b4986a8571acebf4709af8a8d772d3d9f051256` |
| Executed source archive SHA256 | `9e72991307d75a8f78cbcb9e2869741b74e38ade8df1c20f0f738489727b19db` |
| Full private GPU summary SHA256 | `853a9e23188d466c18925521e2ebef8946335fd91f7761bf2f020d7efa190ccf` |
| Private launch receipt SHA256 | `f05b0029023598b33e1912b68c237060fc7e496f64cdf6a3d002d4d1a724e903` |
| Persistent attempt ledger SHA256 | `14c5e74c4b96ccef41cd94db73a9ec3348038ac094feca4fd897cecffa07cdae` |
| CPU unit-test log SHA256 | `a3a96b7731b11f20a7424e62c7e8b0c862366e90803d250d22052ee56062edda` |

Remote archive, GPU receipt and ledger bytes were rehashed read-only. The downloaded receipt and ledger matched those identities. The ledger contains exactly `1`, `2`, `3`. Canonical LF source hashes match both the receipt and Git objects at the execution commit:

| Source | SHA256 |
|---|---|
| `tools/exp17/pair_loss.py` | `d9dba2e3c1ecd46f72a00950215f20737f5bd1a039eeafbe7668e6d00050aaf4` |
| `tools/exp17/prototype.py` | `95fbba9106e84df02d7e39fed5a7726138023509aed2d9ebc7811c7781bbf7f7` |
| Unchanged `tools/exp15/regions.py` | `a3c9f78e855348327b8a707afc879056f2d9ae89666481b59f5b72af865b687e` |
| `tools/exp17/test_pair.py` | `8da6bc5ed0132d6061d84b480a62728d5a03557cbde06db564db7d12ae93e46b` |

There is no diff under `tools/exp15` from the audited starting commit `401210427a61f3430f18bb4fe93261c911f66080`. The compact public machine receipt is [prototype_summary.json](prototype_summary.json). Private hostnames, file locations and asset inventories are excluded from this report.

## Implemented objective and grouping

The prototype imports the existing fused P3 layer 16, Polygon coverage pooling, normalized shared 64→128→128 projection and J1 supervised contrastive objective directly from Exp15. It retains `lambda_s=8` and temperature `0.10`; it does not add a second representation head or classifier.

For a selected synthetic A/N binding, B consists of same-class native main-forward regions from **other source groups in that same batch**. It does not mine a hard B, add positive images, use a memory bank or treat a second repair of A as another source. The A/B cosine distances are averaged within each B group, then across B groups. With normalized embeddings and `d(x,y)=1-x·y`:

`h(A,N) = max(0, mean_B_group(mean_regions_in_group(d(A,B))) - d(A,N) + 0.20)`.

Anchor loss is averaged across Polygons within each represented A group, then A groups within each represented class, then represented classes. Repeated regions from a B group do not multiply that group's weight. If B is unavailable, the Pair anchor is skipped with an explicit reason. Class/group averaging prevents count inflation; it does not create independent evidence or solve sparse-class variance. The synthetic GPU batch has two independent artificial sources per class, so each anchor has only one other B group. Coverage in real TRAIN batches is unproven.

Margin `0.20` is an engineering candidate. The synthetic joint objective is `L_YOLO + 8 L_SupCon + 1 L_pair`. The unit Pair coefficient is an engineering input only. It is neither a frozen research coefficient nor a calibrated recommendation.

## Pair identity and version checks: implemented versus remaining

[PairBinding](../../tools/exp17/pair_loss.py) binds filtered region row, batch slot, class, source group, Polygon ID, repair version, source/repair SHA256 tokens, matrix and ValidMask identity, augmentation trace and actual A/N augmented tensor identities. Each provenance token must be a 64-character lowercase hexadecimal hash. Duplicate binding identities, row/class/group mismatches and different A/N row assignments fail closed. Incomplete target support, unknown contamination or ambiguous regions are skipped.

Version selection uses a private `random.Random(seed)` over a sorted list, choosing one version per `(source_group, polygon_id)` without consuming the native sampler's Python RNG. The default selection is deterministic across calls; an epoch-dependent version exposure schedule is **not** implemented. If different Polygons in one counterfactual image slot select conflicting repair/image/geometry identities, **all** selected bindings from that slot are skipped. This avoids silently assembling one image slot from incompatible repair versions. It does not establish that multiple simultaneously repaired Polygons are semantically safe.

The N input must be a full batch with A's shape, dtype, device and slot order. Every unpaired or unadmitted slot must retain A exactly. For admitted slots, the actual augmented A and N tensor bytes must match the binding hashes. The same A Polygon mask is pooled on the aligned N feature map.

These are **synthetic contract checks**. Hash-shaped tokens and declared quality flags do not prove historical Polygon identity, independent source grouping, a correct registration, valid support or human acceptance. The prototype does not open real repair assets, establish those semantics or create trusted Pairs. `real_pair_training_gate()` unconditionally rejects real training. The CLI exposes no real-data, real-weight or real-Pair manifest argument. This is an engineering scope boundary, not a general security mechanism for arbitrary edited Python code.

## BatchNorm and RNG protection

The native A forward remains in TRAIN mode and advances each native BatchNorm counter once. The N forward uses the complete counterfactual batch. Its scoped context leaves BatchNorm `training=True` while setting `track_running_stats=False`, so it uses TRAIN batch statistics without updating `running_mean`, `running_var` or `num_batches_tracked`. Affine parameters retain gradients. Flags and Python, NumPy, CPU and CUDA RNG states are restored in `finally`, including an exception path. Native running buffers are checked for equality; they are never restored by an in-place buffer write that would invalidate autograd versions.

Using `BN.eval()` only on N was rejected because A would then use current batch statistics and N stored running statistics. That route difference can itself become a feature cue. The actual native GPU test with a complete **N=A** batch found bitwise-equal P3 features and projected normalized embeddings, with exactly one native BN counter update and zero N updates. The identity probe restores model/BN and RNG before the three actual updates. The CPU test also checks matching batch-stat outputs, RNG restoration, exception flag restoration and a nonzero backward path.

This establishes buffer protection and removes the tested mode cue. It does **not** establish that repaired image content cannot change N's current batch statistics, that an edited whole-image domain cannot become a shortcut, or that real A/N augmentation has been matched. The implementation assumes the audited native architecture without a new stochastic auxiliary module; it does not claim exact replay for arbitrary future dropout-based models.

## Augmentation adapter boundary

`ReplayTrace` and `replay_aligned_pair()` provide a small synthetic contract: already aligned A/N arrays share one frozen 3×3 transform, output frame and affine RGB transform; target and ValidMask use nearest-neighbor mask transforms. CPU tests exercise a horizontal flip plus padding, exact A=N agreement, transformed target/ValidMask support, invalid support rejection at admission and singular-transform rejection. No new registration is estimated and no historical transform is changed.

**A real native YOLO augmentation replay adapter is not implemented.** The helper does not capture or replay the audited main loader's full resize, mosaic, perspective, crop, mixup, HSV and other choices. The synthetic helper's RGB transform is not a claim of equivalence to Ultralytics HSV. A future real loader must prove original-space A/N/Polygon/ValidMask binding, frame and pixel conventions, the actual main augmentation trace, compatible per-slot version assembly and preservation of the native main sampling/augmentation stream. A matrix token alone is insufficient.

The current smoke therefore cannot substitute for the future native real-Pair replay and main-trajectory parity Gate. It also cannot establish that the historical ValidMask covers the whole target; the GPU synthetic quality flags are constructed as true for artificial examples.

## Actual execution and numerical evidence

The actual runtime was Python 3.12.3, PyTorch `2.8.0+cu128`, Ultralytics `8.4.117`, on one NVIDIA GeForce RTX 3080 Ti. The model is a newly random-initialized native seven-class YOLO11n-seg architecture, not the frozen COCO initialization. All pixels, classes, groups and Polygons are artificial. The same 14-image, 128×128 synthetic batch is reused for the three engineering updates. No real TRAIN image pixels or real Pair entered an optimizer update.

The prototype uses one native AdamW with the existing native parameter grouping, FP32 master parameters and CUDA AMP, GradScaler initial scale 1, global clip 10, synthetic learning rate 0.001, momentum 0.937 and weight decay 0.0005. The auxiliary graph participates in that same optimizer update. These small smoke settings are not a formal 640/batch32 protocol or a full warmup test.

| Update | Native loss | SupCon loss | Pair loss | Joint loss | Global gradient norm before clip | Derived clip coefficient |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 317.200897 | 2.741294 | 0.135146 | 339.266388 | 8893.180664 | 0.00112446 |
| 2 | 274.124329 | 2.785354 | 0.221603 | 296.628754 | 1787.062622 | 0.00559577 |
| 3 | 243.271423 | 2.026778 | 0.193031 | 259.678680 | 1092.628052 | 0.00915225 |

The clip coefficients are derived from `min(1,10/(norm+1e-6))`; they are not separately measured post-AdamW effect ratios. Clipping is severe in this random small-image experiment. The results cannot calibrate a real TRAIN Pair weight or predict formal relative updates. AdamW, clipping, AMP rounding and the combined objective also prevent interpreting a loss scalar as an effective parameter contribution.

The receipt records three native optimizer calls and three effective updates; the registered ledger confirms three attempts. There were no AMP skips or detected overflows, no nonfinite scaled/unscaled gradients, no missing trainable gradient tensors, and finite FP32 model and AdamW states after every update. All three optimizer steps changed parameters. The shared projection's four trainable tensors had nonzero gradients before each step; final Backbone and head state hashes differ from initialization. Some high-stride localization/mask branches have zero gradients on these same-sized artificial targets. The claim is not that every one of the 294 trainable tensors has a nonzero gradient.

The first synthetic batch has independent no-update derivative checks:

- Pair derivative reaches all **120** audited Backbone parameter tensors with nonzero gradients; aggregate norm **26.57050761032345**.
- The N P3 tensor receives nonzero Pair gradient, norm **0.010236268863081932**; N is not detached.
- The actually computed joint-minus-J1 Backbone gradient has norm **33.21715759859944**, proving a nonzero auxiliary influence on that shared graph.

The Pair-only derivative uses a diagnostic multiplier 1024 and then cancels it in FP32 to reduce tiny AMP derivative loss. This does not alter the joint backward or its unit Pair coefficient. The separately computed joint-minus-J1 derivative is subject to finite-precision AMP addition and subtraction. The two reported norms are not an exact algebraic decomposition, a clipping-adjusted update ratio or a real-data strength estimate. No optimizer step occurs in these derivative checks.

Each update has 14 eligible synthetic Pair anchors from 14 artificial A groups and all seven classes. This proves a seven-class plumbing path, not independent real seven-class Pair availability. Peak allocated/reserved CUDA bytes were **305,948,160 / 350,224,384**; timed smoke-loop plus checkpoint validation elapsed **5.796000860631466 s**. This is not an estimate of full-resolution real-Pair training cost or end-to-end launch time.

## CPU optimization and bounded-budget interpretation

The preserved CPU log records **9 tests passed** in 0.166 s. They cover hinge arithmetic and missing B, B-group replica invariance, version/identity/quality rejection, an isolated embedding optimization, wrong N bytes and unpaired changes, same-slot version conflicts, BN/RNG/exception semantics, the synthetic geometric/ValidMask contract, and durable cap exhaustion.

The isolated embedding test uses normalized synthetic vectors and 50 CPU SGD steps; its final hinge is less than 10% of its initial value. It records an assertion, not a saved numerical learning curve. It demonstrates that the ranking term can be optimized. It does not assert monotonic loss reduction in the native joint training loop: the actual Pair sequence is **0.135146 → 0.221603 → 0.193031**, ending above the first value. No real representation or segmentation gain is inferred from either result.

The synthetic runner has cap 3, fsynced attempt reservation, exclusive run locking, exclusive new output creation and refusal to automatically restart an existing started budget. The actual registered budget is exhausted. The CLI's budget path must continue to be bound by the external authorization; creating a new budget path would not create new experimental permission. No further GPU optimization is authorized in this preparation stage.

## Checkpoint and stripped native model

| Checkpoint identity | SHA256 |
|---|---|
| Synthetic checkpoint file | `0b0476299783decc050cff72b8eb8542780c072f2f7f144bb4d181b7054ed989` |
| Full FP32 main state | `5b551f37e0f0e3ce7063b5ca46e69a3b0f0b68090cb32f91fd5a023816bd4f0c` |
| EMA state | `1f15330b34a9dbb54eecd7caeaf18f690c63254671a3d74e83304c54d25bc3d5` |
| Native FP32 state after projection removal | `1b1a83a833cd9c73e8ebd4c5efcdb5d45a980408390ffd1926f98c88f55db3f0` |

The runner strictly reloads the full main state, EMA, native AdamW groups/states and GradScaler. All 294 optimizer slots have step 3; EMA updates and scaler growth tracker equal 3. It loads the main state after the EMA check, removes only the shared projection module, and verifies the stripped native state identity. A CPU native forward on a synthetic zero tensor is finite. This is synthetic model-structure inference, with **0 VAL and 0 TEST inferences**, not a dataset evaluation.

## Findings and next Gate

No hard failure was found in the recorded synthetic run. Source, budget and receipt consistency checks pass. The implemented prototype supports the intended shared representation and same-update optimizer mechanism, with protected running statistics and a strict reload path.

The material gaps remain:

1. Human-reviewed trusted Pair admission, independent source groups, exact Polygon/version/registration/ValidMask semantics and contamination checks are absent from this synthetic-only entry.
2. Real native main-augmentation replay and the real full-batch counterfactual loader are unimplemented. The synthetic transform contract does not close this gap.
3. Real class/group coverage, small/elongated/overlapping-Polygon behavior, context/domain shortcuts, per-class gradients and independent B coverage remain unmeasured.
4. Formal `lambda_p` is unset. Calibration on admitted TRAIN Pairs, including joint gradient/clipping behavior at relevant states, requires a later independent authorization; the severe random-smoke clipping is no basis for selecting it.
5. The prototype does not prove formal COCO initializer identity, full 640/batch32 resource behavior, warmup stability, main-task paired trajectory equality or segmentation efficacy.

The defensible status is **synthetic engineering PASS; real-Pair Data/Formal Gate HOLD**. Future human G1 admission, native real-loader/replay validation, an independent Data Gate and an independent Formal Gate remain necessary. No real-Pair optimization, formal training, additional seed, held-out evaluation or automatic parameter choice follows from this report.
