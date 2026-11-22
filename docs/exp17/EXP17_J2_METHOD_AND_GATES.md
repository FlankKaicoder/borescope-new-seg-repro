# Exp17 J2: source-aware defect–pseudo-normal preparation

This is a preparation study on frozen TRAIN assets. It does not authorize real Pair optimization, 100-epoch training or held-out inference. The accepted Exp15 three-seed SupCon result is the starting point. The user accepted Exp16's fixed-weight CE negative result as a boundary; this does not establish that all regional CE methods fail. Neither historical branch is changed.

## Small candidate comparison

| Candidate | Useful property | Main limitation | Decision |
|---|---|---|---|
| Repel A from N only | Few operations, explicit state contrast | No same-class reference to constrain domain separation, despite a bounded cosine distance | Do not use |
| Binary defect/generated-normal CE | Easy classification objective | Generated domain can become the label; adds a classifier and normal-label assumptions | Do not extend CE |
| Shared-head relative A/B/N ranking | B anchors the same-class defect geometry; hinge stops once relative margin holds | Still vulnerable to repair artifacts and insufficient B coverage | Prototype first |

The base [SupCon paper](https://arxiv.org/abs/2004.11362) supplies the supervised contrastive foundation; [FaceNet](https://arxiv.org/abs/1503.03832) is a reference for relative triplet embeddings. The source-group reduction and pseudo-normal eligibility rules below are this project's proposed adaptation, not a claim that either paper validates these industrial repairs.

## First prototype

A and B are native augmented real-defect Polygon features from the main YOLO forward. B must have the same frozen class and a different independent source group. N is the already aligned repaired counterpart at A's same mechanical location. Use the existing fused P3 (layer 16), Polygon coverage pooling and the existing 64→128→128 J1 projection. Normalize its output; use cosine distance `d(x,y)=1-x·y`. A/B/N share both Backbone and projection parameters. Do not add a second head or classifier.

For each eligible A, average A/B distances within each represented B group, then average those group means. This avoids counting multiple Polygons or repeated views from one group as independent positive sources. Use every qualifying B in the existing main batch, without mining, a memory bank or extra positive-image sampling. If no cross-source same-class B is present, skip that Pair and record why.

`h_A = max(0, mean_B_group(d(A,B)) - d(A,N) + m)`

Average h first across Polygons within an A group, then across A groups within a represented class, then across represented classes. Report the actual denominator, class/group coverage and skip counts on every update. Equal represented-class averaging avoids raw Polygon-count dominance, but can give a sparsely represented class substantial weight. It is not proof of adequate class balance; per-class gradient diagnostics remain a calibration requirement.

`L = L_YOLO + 8 L_SupCon + λp L_pair`

SupCon's temperature remains 0.10 and its definition is unchanged. A single global margin 0.20 is an **engineering candidate**, not a validated scientific choice. No class-specific margins are justified. Formal λp is **unset**. The synthetic unit coefficient 1.0 is not a recommendation for real training.

Normal context is retained for diagnosis and human review, not given normal GT labels and not assigned a background consistency loss. The initial hypothesis concerns defect-location state differences; unannotated mechanical context does not establish defect absence.

## Identity, versions and contamination

Bind source image, professional annotation, Polygon, class, independent group, repair version, raw/aligned bytes, historical matrix, Valid Mask, and actual augmented A/N tensor identities. Multiple repair versions of a Polygon are alternatives, never additional independent examples. One counterfactual image slot cannot mix repair versions or matrices. Conflicting per-slot bindings are skipped in the prototype, with a recorded reason.

Incomplete target support, uncertain contamination, unknown normal semantics or ambiguous regions are excluded. A shared ROI containing another annotated defect requires explicit semantic review. Other GT within the target pooling support or state-inconsistent repair invalidates that normal reference. A10 whole-image admission never substitutes for B Pair admission. No labels are inferred from missing JSON or unannotated regions. Do not fabricate Tip curl coverage.

The synthetic binding selector uses a local RNG and never changes the native sampler. A future real selector must operate on a separately frozen human/Data-Gate manifest and preserve the full native image order, augmentation tensors and label exposure. Current code rejects all real bindings, including a caller-supplied claim of human PASS.

## Auxiliary forward and BN

N uses a **full same-slot counterfactual batch**: eligible slots contain their bound repair; unpaired slots are byte-identical to A. Both routes use TRAIN batch statistics. During the N forward only, BN tracking is disabled; running mean, running variance and counters cannot change. Restore original module flags and Python/NumPy/CPU/CUDA RNG streams on normal and exceptional exit. Leave autograd attached. Restoring BN tensors in place after a forward is avoided because it can invalidate saved autograd versions.

Using eval BN for N and train BN for A would create a route-dependent feature cue. The implemented design instead verifies exact P3/embedding equality when synthetic N=A. Different repaired content can still change batch statistics for every slot; this is a remaining composition risk, not a solved artifact problem. A future admitted-data smoke must test sparse counterfactual batches and class/group composition before formal use.

One joint backward, one AdamW step, one GradScaler update and one EMA update are permitted per main-task update. There is no independent auxiliary optimizer. N adds compute, not main-task updates. The prototype exercises the existing numerical audit, clipping, strict full-state reload and projection stripping, but uses random initialization and 128-pixel synthetic images. It is not a 640-pixel, batch32 resource or scientific performance result.

## What is implemented versus still gated

Implemented: identity guards, source-aware loss/reduction, synthetic replay contract, shared A/B/N graph, BN/RNG protection, durable three-attempt budget, numerical audit, actual GPU updates, strict model/EMA/AdamW/GradScaler reload and native synthetic inference after head removal. CPU embedding optimization tests sensible loss behavior; the three-step joint YOLO loss need not be monotone.

Not implemented or authorized: admitted-real-Pair native augmentation replay, an executable real-data trainer, calibrated λp, a frozen trusted-Pair release, or any J2 formal authorization. The generic geometric/RGB replay test does not prove equivalence to native YOLO affine/HSV/flip augmentation. A future adapter must capture and replay the exact native trace, pixel conventions, masks and instance IDs without consuming the main RNG stream or changing the 2,100 native batches.

## TRAIN-only artifact diagnostics

Use the 36 historically defined pilot cases, one per independent source group; keep the six historical feasible cases as an explicit priority subset. This cohort is diagnostic and selection-biased, not a new quality-approved training set. Pin the accepted J1 seed42 epoch100 FP32 master and its file/state hashes. Forward only, both routes eval/FP32/TF32 off, no optimizer or parameter gradients. Verify BN and checkpoint identities before/after.

The historical RGBA request mask's alpha-zero pixels are the requested editing support. They do not prove that generation changed nothing outside that support. Context is the historical ROI intersected with Valid Mask, excluding a fixed 5-pixel dilation of all annotated defects and editing support. Call it **candidate unedited mechanical context**, not verified normal. Preserve unsupported or empty regions as diagnostic skips.

Compare A/N target and context cosine distances in raw normalized P3 and the frozen projected embedding. Report outside-support pixel changes, color and Laplacian/edge proxies, and a leave-one-independent-group-out nearest-centroid original/repair domain diagnostic. Its labels describe image origin, not defect status. Domain separability can reflect style, changed content, camera overlays or residual alignment; it cannot isolate a causal generator-style mechanism. No diagnostic thresholds select λp or admit G1. Human review addresses repeated texture, seams, blur, implausible edges and defect removal separately.

## Required gates before a first real J2 run

1. User independently reviews exact asset-bound cases; exports new decisions without changing historical records. PASS requires all positive G1 criteria and no unresolved negative flags. Codex observations remain separate and cannot create human PASS.
2. Independent Data Gate verifies those exports, exact version/Polygon identities, support, semantics, contamination, near-duplicate groups and coverage. Freeze an admitted manifest with its hashes. Fifteen to thirty groups is a planning target, not a quality quota. Trusted Pair count is currently zero.
3. Implement the native paired-augmentation adapter against that frozen release. Prove native main-task trajectory equality, sparse N=A invariance, one-update/BN behavior and 640/batch32 resource capacity in a separately bounded admitted-data smoke.
4. TRAIN-only λp calibration checks per-layer and per-class gradient norms/cosines, clipping effects and parameter updates at prespecified early/mid/late states. Exp15/16 showed why initialization-only ratios are inadequate. Use independent replay that cannot alter training state/RNG. Freeze a conservative fixed coefficient; no VAL choice or large search. Stop for design review if no safe, meaningful intervention is supported.
5. Submit the complete machine protocol, source identities, admitted manifest, calibration, smoke and capacity evidence for independent Formal Gate. Start fresh from seed42 COCO transfer only if separately authorized. No checkpoint continuation from J1 or an engineering smoke.

The first future comparison is fixed J0/J1 versus fresh seed42 J2, 668 TRAIN images / 1,266 raw instances, 100 epochs / 2,100 effective main updates, matched AdamW/AMP/schedule/augmentation, and fixed epoch100 FP32 master. Existing J0/J1 remain read-only references. Evaluation requires a later independent frozen VAL Gate; TEST remains prohibited. This draft is not execution permission.
