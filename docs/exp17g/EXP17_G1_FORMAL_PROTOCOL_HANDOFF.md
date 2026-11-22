# Exp17-G1 formal protocol and recovery executor handoff — 2026-11-23

**EXP17_G1_FORMAL_PROTOCOL_AND_EXECUTOR_READY_PENDING_CHATGPT_GATE**

G1 prepares the fixed J2 protocol, independent formal entry, exact native schedule checks, full-state recovery, scheduled validation, checkpoint selection, monitoring and resource controls. This is a protocol/software delivery. **Formal training, real TRAIN optimizer updates, GPU model forwards/backwards, VAL inference and TEST access in G1 are all zero.** The previous engineering Smoke remains sealed at 50/50 attempts; G1 did not extend or reset it. Synthetic CPU optimizer/validation callbacks test control flow only and are not J2 training or performance evidence.

The machine protocol is [formal_training_protocol_v1.json](../../results/exp17g_formal_protocol/formal_training_protocol_v1.json), SHA256 **`613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44`**. It retains `formal_training_authorized: false` and `status: PENDING_CHATGPT_FORMAL_TRAINING_GATE`. A separate exact-identity ChatGPT authorization is required before any future formal construction or run. The prepared entry cannot grant itself that authorization.

## 1. Git baseline and preserved evidence

The work is isolated on `phase/exp17-j2-pseudonormal-pair`, based on audited engineering commit `855ea91d3b181625ab802fd1879b1722c4e6db79`. Local branch/HEAD/worktree, GitHub API and `git ls-remote` were checked. The independent Exp15 branch remains `401210427a61f3430f18bb4fe93261c911f66080`; Exp16 remains `2f1d3b8d1bc10d635099867fab4a935d6cdb88cb`. Original training/evaluation source, checkpoints, data and protocols remain unchanged. New code is under `tools/exp17/formal/`; new receipts are under `results/exp17g_formal_protocol/`.

The frozen J1 protocol's canonical Git SHA256 is `b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca`. Historical Windows CRLF working bytes and canonical LF Git identities are recorded separately. This is a provenance distinction; no historical file was rewritten or normalized. Private inputs are resolved through the prior engineering delivery receipt rather than inferred paths. The [prior engineering handoff](../exp17/EXP17_J2_ENGINEERING_HANDOFF.md) and its raw/stopped receipts remain intact.

Publication identity is the Git commit containing this handoff and its [evidence manifest](../../results/exp17g_formal_protocol/evidence_manifest.json). The exact commit and independently matching GitHub API / `git ls-remote` HEAD are supplied in the final delivery message and private publication receipt, avoiding circularly embedding a commit hash in its own bytes. The final publication checks also require a clean worktree and unchanged reference branches.

## 2. Fixed data, method and protocol audit

The admitted release is unchanged: **521 Aligned Polygons / 345 source images / 327 independent source groups**, comprising 459 historical exact-Aligned inheritances and 62 new current-Aligned user acceptances. The 43 Raw/Crop/Bounded candidates and 12 excluded records remain outside the main release. Polygon-scoped acceptance, A/N hashes, source groups, original/N geometry, matrix/Valid identities, review/version provenance and one frozen A10 N per source remain binding.

| Identity | SHA256 |
|---|---|
| 521-Aligned JSON | `fe6a36601c6c8d90cfddfc19ab58335552c18653b66f3535e84c97b4c222dc22` |
| 521-Aligned CSV | `b5bf196b735101f28cf02c51a3297c2a1043c101bbce4473136bc67ebf4b462c` |
| Native TRAIN identity | `289848daad87bbb8601251a88305d58dd48f41963f7d6adff0c4777603b14c4e` |
| Frozen split manifest | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Frozen COCO | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |
| Initial native seed42 state | `8e0479ecdae6dc60558ed44d5d62c01f8b61075514bc34c109295950603fc886` |
| Initial shared projection | `83664f93724017b43a2861826141fc41074a0ae58729d170829f787395d1fc6e` |

| Class | Admitted Polygons | Independent A/N groups |
|---|---:|---:|
| Burn | 54 | 32 |
| Crack | 52 | 39 |
| Dent | 94 | 84 |
| Material missing | 80 | 77 |
| Tears | 40 | 38 |
| Tip curl | 0 | 0 |
| corrosion | 201 | 72 |

The main task remains original TRAIN668 / 1,266 instances, YOLO11n-seg, fresh frozen COCO seed42, 640, batch32, 100 epochs and 2,100 effective joint updates. It retains original augmentation, sampling, seven-class YOLO losses, AdamW grouping, AMP and clip10. No normal class, empty-label N sample, second optimizer, extra B sampler or pseudo-normal YOLO loss is introduced.

The fixed objective is `L_YOLO + 8*L_SupCon + 4*L_Pair`, temperature 0.10, margin 0.20, detached `replay_a` normalization, shared native N layers 0–16 and `checkpoint_prefix=false`. B is from another independent group of the same class in the same native A batch. The frozen source-group and class reductions, valid-support skips, native Polygon pooling and shared projection are imported unchanged. This stage audits their execution controls; it does not recalibrate or retune them.

## 3. Full schedule and all 63 warmup positions

[Schedule audit](../../results/exp17g_formal_protocol/lr_schedule_audit.json) and [2,100-row trace](../../results/exp17g_formal_protocol/lr_schedule_2100.csv) compare an independent mathematical derivation, the actual native scheduling/optimizer code path and historical J1 training receipts. J1 and the J2 candidate agree bit-for-bit on all 2,100 native LR rows and parameter groups. Independent Decimal mathematics agrees within the predeclared `2e-16` absolute tolerance; binary operation ordering accounts for a maximum difference of about `1.39e-17`. Mathematical agreement is not falsely described as identical binary evaluation of independently ordered formulas.

The native settings are `lr0=.01`, `lrf=.01`, linear LambdaLR, AdamW `betas=(.937,.999)`, epsilon `1e-8`, `amsgrad=false`, `nbs=32`, accumulation1 and weight decay `.0005`. Authoritative native group order is **weight / BN / bias**; weight decay is `.0005 / 0 / 0`. Historical LR ledger column names alone do not identify weight-decay assignment, because the first two groups have equal LR trajectories.

Scheduler construction performs its native initial call; setup sets `last_epoch=-1`; each epoch starts with native model training mode and one scheduler call, followed by the original epoch loader. Before each update the original warmup receives the zero-based global iteration. There is no per-batch scheduler call and no extra call due to N, validation or recovery.

Warmup uses **`iteration < 63`**, exactly zero-based positions 0–62 / one-based updates **1–63**. All 13 positions 51–63 are checked by CPU mathematics/code, without another GPU update. First-update LRs are `[0, 0, .1]`; the bias group has its independent `.1` warmup start. The last warmup bias LR is approximately `.011233714285714291`. Update64 is the first non-warmup update, LR `.009703` in all groups. AdamW groups have `betas`, not a literal `momentum` key, so native `warmup_momentum=.8` does **not** change beta1: it remains `.937`. No new momentum ramp was invented.

At the actual epoch100 updates, LR is approximately **`.0001990000000000001`**, not `.0001`. The latter is the theoretical schedule value at unused epoch index100; the actual final trained index is99. Each epoch retains twenty full batches of32 and one final batch of28, with `drop_last=false`: 668 images per epoch, 66,800 original image exposures and 126,600 pre-augmentation instance exposures across the fixed budget.

This closes the **schedule-audit** gap at positions51–63. It does not retrospectively turn the historical 50-update GPU Smoke into a full-warmup numerical test.

## 4. VAL cadence, native fitness and selection bias

The future protocol proposes validation only after complete epochs **5,10,…,100**. It specifies **20 validation events**, with MASTER and EMA evaluated separately: 40 model evaluations and **80 native passes** because every model state receives standard `conf=.001` and fixed `conf=.25` passes. This explicit scope is larger than interpreting “20 VAL” as20 individual forward passes. A future Gate must approve the full named scope; no validation has occurred in G1.

The adapter reuses unchanged Exp14 `a11_full_eval.py` and J-D's sealed VAL-copy reader. It retains 154 frozen images / 296 instances / seven classes, Ultralytics8.4.117, 640, batch32, CUDA FP32, TF32 disabled, NMS IoU.70, max_det300, original class-aware one-to-one matching at IoU≥.50 and the existing TRAIN-derived size thresholds. It outputs the full native overall/per-class Box and Mask metrics, fixed TP/FP/FN and tiny/small/medium/large Mask Recall. No threshold search or alternate metric implementation is added. G1 reads only approved manifest/protocol metadata, not VAL image bytes.

[Fitness audit](../../results/exp17g_formal_protocol/fitness_audit.json) records actual source identities and locations: `ultralytics/utils/metrics.py` has Metric fitness near line1007, DetMetrics near1205 and SegmentMetrics near1368; `ultralytics/engine/trainer.py` has validation near856 and model saving near709. Native Metric weights are `[0,0,0,1]` over Precision, Recall, AP50 and AP50–95. **Segmentation fitness is Mask AP50–95 + Box AP50–95**, not Mask AP alone. All experiment metrics must be finite; native `nan_to_num` is not used to conceal a numerical failure.

Best begins as `None`. Native validation changes the numeric best on strict improvement; native saving writes Best when current fitness equals best fitness, including ties. The prepared policy therefore selects the **latest equal-fitness epoch**, separately for MASTER and EMA. Selection records bind epoch, role, native state identities, fitness and complete metrics. A synthetic test proves Mask AP can improve while native fitness decreases; the selector retains its frozen rule in that case.

The primary method comparison remains **fixed epoch100 Last FP32 MASTER**. Best is a VAL-selected supplement and is not an unbiased TEST estimate. EMA is independently named and reported. Historical J0/J1 Last scores are preserved; a J0/J1/J2 Best comparison would need matching validation cadence, model role, fitness and tie rules under another explicit Gate. There is no current matched historical Best result to claim.

## 5. Validation and gradient-diagnostic state isolation

Evaluation and diagnostics operate on disposable model copies. The isolation guard protects live FP32 parameters, BN buffers, every module's train/eval flags, existing gradients, EMA weights and attributes, optimizer state/groups/defaults, Scheduler, GradScaler including internal per-optimizer stage, sampler cursor/generator, epoch and numerical observers. It restores Python, NumPy, Torch CPU, already-initialized CUDA RNG and relevant backend settings even when the callback raises. An unexpected live mutation is restored where possible **and still triggers a hard Gate**; it is not accepted as a successful validation. A live optimizer pre-hook rejects a callback's attempted step before parameters change.

The candidate performs fixed TRAIN gradient diagnostics at the same five-epoch boundaries using the engineering phase's four hash-bound CPU batches at positions1,7,11,21. Diagnostics explain Pair/YOLO and Pair/SupCon cosine, shared-Backbone/per-layer norms, class component shares and clip10 changes for fixed λp4. They execute no optimizer update, preserve live state/RNG and cannot select a new weight or checkpoint. Disposable copies may legitimately update their own temporary BN statistics; live training BN and EMA remain unchanged.

## 6. Full checkpoints, Last, Best and final endpoints

The full recovery payload binds FP32 MASTER, EMA and update count, AdamW parameter names/state, GradScaler, Scheduler, epoch/batch/global update, training modes/BN identity, all RNG, the sealed sampler cursor, original model configuration/class mapping, counters, monitoring/selection metadata and protocol/code/data/COCO/environment identities. It is saved after gradient clearing at a committed boundary and must strictly reload before continuation.

The prepared retention policy is deliberately bounded and must be approved with the formal Gate:

| Object | Proposed retention and purpose |
|---|---|
| Latest full recovery Last | Two atomic alternating slots, saved every effective update; therefore also current at every epoch end |
| Periodic full recovery | Twenty immutable complete checkpoints at epochs5–100 |
| Best MASTER | Two atomic FP32 tensor slots containing both states at the selected MASTER epoch, plus immutable lightweight selection receipts |
| Best EMA | Two separate slots with its own selection pointer/history |
| Final native MASTER | Immutable `epoch100_last_master_fp32.pt`, sole primary evaluation endpoint |
| Final native EMA | Immutable `epoch100_ema_fp32.pt`, independent supplementary role |

Best slots omit AdamW; full recovery states remain in Last/periodic checkpoints. An old selection JSON preserves its identity, fitness and metrics, but does not claim overwritten rotating-slot bytes remain archived. Current Best is independent of Last. The native Ultralytics default EMA-half/optimizer-half checkpoint format and nonfinite repair/clamping are **not** reused: this experiment requires explicit FP32 roles and hard numerical stops.

Final native exports strip only the unchanged training projection, strictly reload the selected seven-class FP32 state, retain class mapping and use the pinned native weight loader without model inference. Their `model` field explicitly holds the selected MASTER or EMA state; implicit EMA substitution cannot change the primary endpoint. Epoch100 full optimizer/Scheduler/Scaler/RNG/sampler recovery state is retained separately. Synthetic CPU export tests inject a tiny seven-class module and explicitly report that real native-loader execution was not tested in G1.

## 7. Exact native batches and safe recovery semantics

The original J1 loader uses two workers. A sampler index alone cannot reconstruct prefetched worker RNG or guarantee the same partially augmented batch after interruption. The prepared adapter therefore drains the **same native epoch loader** once into21 sealed CPU batch packets, binding tensor bytes, native source/exposure fingerprints, ReplayTrace and manifest/code/protocol identity before optimization of that epoch. Consumption uses these exact bytes, and recovery reopens the next sealed batch. It does not reread prior original images, rerun completed random transforms or redraw N augmentation.

This changes loader/optimizer **overlap and temporary storage**, an explicit engineering difference. It does not intentionally change native sampling or augmentation mathematics. G1 did not run the real native stream again to empirically establish full-size prefetch equivalence. Future execution must match all2,100 known native fingerprints; each epoch's cache is checked before its updates, and every consumed batch is checked again. A mismatch stops rather than changing the sampler or silently resealing different evidence. Cache construction must leave the main RNG unchanged.

The permanent update journal reserves a physical attempt before optimizer work and commits it only after the exact full checkpoint pointer is durable. Recovery requires the journal and latest checkpoint hash/counters to agree, the same protocol/code/data/environment, an exact sealed cursor and an independent resume authorization. Epoch boundary, epoch-middle and warmup-middle tests compare continuous versus interrupted synthetic CPU trajectories, including AdamW, EMA, GradScaler, BN, RNG, future augmentation/batch identities and Scheduler.

The supported guarantee is **safe continuation at any durably committed update boundary**. Physical process death between attempt reservation and durable commit is an ambiguous attempt and hard-stops. An interrupted validation/diagnostic epoch event also hard-stops. The executor does not automatically replay that optimizer update, repeat VAL, infer which side of a crash occurred or advertise crash-proof recovery inside an uncommitted hardware operation. Additional investigation and a recovery Gate are required. The simulator tests these refusal paths as well as successful committed-boundary recovery.

## 8. Atomic storage and corruption refusal

Storage uses an exclusive temporary file, complete serialization, file fsync, SHA256, strict reload, atomic rename, directory fsync where supported, and an atomic pointer/final receipt. The inactive Last/Best slot is prepared while the prior pointer remains valid; an interrupted pre-commit write cannot invalidate the prior selected state. Windows directory-fsync limitations are recorded; the future Linux runtime supports that operation.

Missing, truncated or changed checkpoint bytes, pointer/hash disagreement, manifest/code/protocol changes, wrong counters, wrong sampler position and an incompatible environment are refused. Orphan or stopped evidence is retained. Renaming the output, copying authorization, deleting a ledger or restarting Python does not create a fresh budget. Immutable five-epoch/final artifacts cannot be overwritten.

## 9. Monitoring, hard stops and authorization

Per-update observations include native/SupCon/Pair/total losses and fixed weights, valid A/B/N and anchor counts, class coverage/skips, global gradient norm/clipping, AMP scale/skips, attempts/effective updates, exact LR/Scheduler point, epoch/batch, BN hash, N gradient path, parameter changes, memory/time and explicit augmentation/native fingerprints. Epoch reports reconcile21 updates,668 images,1,266 raw instances and the28-image tail; they retain loss/gradient quantiles, class/group coverage, per-class Pair loss, clipping, LR, BN, EMA count and parameter identity.

Nonfinite losses/gradients/parameters/AdamW, AMP skips, unexpected update counts, wrong batch/augmentation/Pair identity, source/N SHA mismatch, abnormal BN/RNG, corrupted recovery, validation mutation, memory/disk failure or leakage stops the run with append-only evidence. There is no automatic batch reduction, seed/weight/LR substitution, sample skip or hyperparameter retry.

The independent entry requires `EXP17_G1_FORMAL_TRAIN_AUTHORIZATION`, an explicit ChatGPT Gate PASS, exact formal-protocol SHA, source catalog SHA, frozen inputs/COCO, seed42/100epochs/2,100updates, private runtime-config identity, exact output and an experiment-specific permanent control anchor. Authorized actions must expressly include scheduled MASTER/EMA VAL, isolated diagnostics and retirement of only this future run's generated epoch-cache packets. Validation also has its exact protocol/path/epoch/role/80-pass scope. Current public protocol and this handoff grant none of these actions.

A process lease prevents simultaneous executors; the permanent hash-chained journal and execution seal survive output-directory changes. Missing private authorization rejects before heavy model/data imports, output creation or attempt reservation. G1's refusal tests exercise missing/changed gates, source/protocol/data/seed/budget/output changes, copied seals, missing ledgers, concurrent processes and excess/repeated/skipped attempts.

## 10. Current resources and prospective retention policy

The environment was freshly inspected through the established pinned connection: one RTX3080Ti, nominal12GiB, Python3.12.3, PyTorch2.8.0+cu128, CUDA12.8, Ultralytics8.4.117. The read-only probe records driver, memory and current compute processes without launching a model. This supersedes the older repository GPU note; it is not permission to start a GPU task.

The fresh data-disk availability used in this forecast is **7,155,695,616 bytes**. The conservative candidate forecast is **6,977,789,880 bytes**, leaving only **177,905,736 bytes beyond the included safety reserve**. The forecast accounts for recovery/EMA/optimizer states, twenty periodic full checkpoints, bounded Best slots, final native exports, logs/VAL/diagnostics, atomic temporary writes, a1GiB reserve and the transient **third epoch cache** while current/previous caches still exist. See [resource budget](../../results/exp17g_formal_protocol/resource_budget.json).

This is **tight estimated capacity**, not robust measured long-run headroom. Capacity expansion before formal launch is recommended. Recheck actual free space, native packet/checkpoint sizes and temporary peaks immediately before a future start; insufficient capacity blocks execution. Full native caching, GPU validation and end-to-end I/O performance were not measured in G1. Neither the old50-step peak nor this disk arithmetic guarantees the future full run.

The proposed lifecycle retires only exact generated cache members inside the future authorized run after a newer committed checkpoint no longer needs them, retaining current and previous caches. It does not delete historical experiments, frozen data, model weights, Git history or original evidence. No historical cleanup, source movement or environment removal occurred in G1. The future cache-retention action itself needs explicit independent authorization.

## 11. Tests, preserved stops and unresolved limits

Final CPU verification: **129 CPU tests passed: 59 new formal-protocol/recovery tests and 70 unchanged engineering/replay regression tests; zero failures/skips, CUDA never initialized**. The report is [cpu_tests.json](../../results/exp17g_formal_protocol/cpu_tests.json); detailed recovery evidence is [recovery_audit.json](../../results/exp17g_formal_protocol/recovery_audit.json). The common executor is exercised with a complete deterministic tiny CPU synthetic2,100-update/20-validation-event schedule, not a real J2 model or real VAL. Dedicated tests cover schedule/group/warmup behavior, checkpoint atomicity/corruption, all state/RNG recovery, cursor identity, selection/fitness, validation isolation, explicit final roles, resource arithmetic and authorization refusal.

Three ordinary source-inspection probe failures involving property/function access during environment discovery are retained; they occurred before any model or image call. An early common-executor CPU synthetic run stopped at its first Best selection because it mixed full recovery-state and canonical native-state hash vocabularies. The code now distinguishes them explicitly, preserves that synthetic failure and verifies the corrected path. These were new temporary protocol tests, not retries of real training or the exhausted engineering Smoke.

Remaining limits are explicit: the native GPU adapter is statically audited and uses validated prior engineering components, but has not been executed on GPU in G1; native full-size epoch prefetch and actual80-pass scheduling remain future runtime checks; recovery does not conceal ambiguous physical attempts or interrupted auxiliary events; disk margin is narrow; and prior scientific limitations are unchanged. In particular, λp4 was one additional TRAIN-only calibration candidate, state/layer/class gradient conflicts vary, `replay_a` uses detached A statistics, Tip curl has no trusted Pair, and repaired-context artifacts can confound the scientific interpretation. No new Mask mAP or efficacy claim is made.

## 12. Independent Gate and stop point

The candidate machine protocol gives definite training, schedule, method, validation, selection, state, checkpoint, resource and hard-stop fields while remaining non-authorizing. ChatGPT must independently review the protocol SHA,100epochs/2,100updates, Last/Best/EMA roles, native fitness,20events/80passes and VAL selection bias, committed-boundary recovery, the sealed native worker-stream strategy, prospective cache/Best retention, actual resource margin, λp4 and detached `replay_a` before enabling a new run.

Stop after commit, push, independent remote-HEAD verification and handoff. **No formal J2 training, GPU Smoke, VAL/TEST, generation, registration, new seed, new loss or data-pool change follows automatically.**

[Delivery summary](../../results/exp17g_formal_protocol/summary.json) · [Schedule audit](../../results/exp17g_formal_protocol/lr_schedule_audit.json) · [Full schedule trace](../../results/exp17g_formal_protocol/lr_schedule_2100.csv) · [Fitness](../../results/exp17g_formal_protocol/fitness_audit.json) · [Recovery](../../results/exp17g_formal_protocol/recovery_audit.json) · [Resource budget](../../results/exp17g_formal_protocol/resource_budget.json) · [Frozen protocol](../../results/exp17g_formal_protocol/formal_training_protocol_v1.json) · [CPU tests](../../results/exp17g_formal_protocol/cpu_tests.json) · [Evidence manifest](../../results/exp17g_formal_protocol/evidence_manifest.json).

## Final identity and resource postcheck

The final readonly check reverified all 256 production source files against the frozen protocol, the unchanged 50-attempt engineering ledger and checkpoint, COCO and admitted-data identities. It rechecked the exact 154-member VAL manifest and original split metadata without opening VAL images or labels; the 296-instance count is inherited frozen metadata. No formal output or permanent experiment claim exists. After CPU deployment/tests the data disk has **7,145,230,336 bytes free** (7.145 GB / 6.654 GiB); the conservative forecast remains 6,977,789,880 bytes including a 1 GiB reserve, leaving **167,440,456 bytes** beyond that forecast. This is tight. Recheck at the future Gate and launch, and prefer more storage rather than reducing evidence retention.

The final frozen CPU run uses an explicit code path and spawn-safe guard wrapper. The earlier wrapper import failure and all source-probe/synthetic integration failures remain preserved. No production code was changed between the frozen source packet, the final 129-test run and this postcheck. The initial update-zero recovery state is committed before the first epoch. A separately authorized resume at committed update2100 can complete pending epoch100 events/endpoints without another optimizer update; completed experiments and ambiguous attempts/events remain rejected. Fully committed matching endpoints are reused without rewriting; incomplete/orphan endpoints stop for review.
