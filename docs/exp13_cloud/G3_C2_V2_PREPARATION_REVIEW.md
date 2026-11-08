# Exp13-G3-C2 Prompt V2 Preparation

Status: `G3C2_V2_LIVE_READY` (paid authorization is still absent).

## Scope and diagnosis

The frozen G3-C1 decision set contains five `NEEDS_RETRY` sources. The failures
are separated into: one burn case where the generated result changed a metal
boundary; three material-missing cases where the output retained or smoothed a
missing-material notch instead of completing the component edge; and one
corrosion case where the target polygon overlaps green camera UI while
corrosion remains. Existing C0 geometry is a candidate for four cases and
unverified for one. Geometry candidacy does not resolve the visual failure.

The V1 material prompt combined a request to preserve component boundaries
with a prohibition on bridging gaps or inventing hidden parts, without clearly
distinguishing a material-loss edge from a true opening. V2 explicitly asks
for the smallest edge continuation supported by visible adjacent structure,
preserves true gaps, and labels hidden reconstruction as low-confidence. The
burn V2 treats a damage-distorted edge as unreliable while avoiding an
assumption that every edge is straight. The corrosion V2 explicitly protects
UI, but the one case whose mask covers UI remains HOLD because prompt text
cannot guarantee pixel protection inside an edit mask.

The four proposed repair masks are separate private copies of the selected
V1 Polygon masks. No mask expansion was justified: the frozen visual evidence
does not identify a defensible physical completion boundary outside the
selected target. This keeps the edit region bounded and avoids presenting an
invented Repair Mask as a new defect label. The held corrosion source is not
included in the executable manifest.

## Frozen preparation

The private request Manifest SHA256 is
`2689a94ead5f8e395c20a0c38d807acec0af958d65d9001b2598f2e3cc5c0f52` and binds
four TRAIN sources, one request per source, Prompt V2, source image/JSON
identity, copied RGBA masks, and V1 references. The frozen G3-B2 Manifest,
G3-C1 review, and split Manifest were rechecked against their expected hashes.
The V2 prompt-manifest SHA256 is
`8363b9db639cec2bc0e69afc4fefafcbf8b049a6724f34e6ff74e92c46f33d8c`.

Prompt file hashes:

- Burn V2: `3f26e4d41c1b39aaf607970e09ede9ffc139140a22f3586ae4babe956ac20f03`
- Material missing V2: `e2add9041fbdb4e0cd5a88d920a6821265f60a9b405058fb776484a59b8ae344`
- Corrosion V2: `56a1ee99eff9673311c5ceb7e460ae687c684cd0e94de8d96b50b204b36f10de`

Executor Dry-run completed with four request previews, zero POSTs, no
authorization record, and `NOT_APPLICABLE_NO_POST`. The executor continues to
use the G2-verified multipart endpoint and model configuration, with serial
execution, one attempt per source, and no retries. Historical G3-B2 responses
did not report price; the gateway's current price schedule and provider hard
spend limit remain unverified. Cost is `COST_UNKNOWN`, so no monetary estimate
or ceiling is asserted.

No V2 output exists yet. Alignment, composites, and the V1/V2 comparison atlas
will be produced only after the separately authorized batch runs. No visual
pass or geometry candidate becomes a Trusted Pair automatically.

## Validation and safety

- G3-C1 review SHA, 36-source exact-set, class/status links, G3-C0 geometry,
  frozen split SHA, TRAIN image/JSON hashes, JSON dimensions and image path,
  Polygon type/class/index, and copied mask mode/size/nonempty binary alpha:
  PASS for the executable sources.
- Synthetic G3-C2 plan, identity, and failure-closed tests: 3/3 PASS.
- G3-B2 executor Mock/Dry-run regressions: 10/10 PASS.
- Broader G3 suite excluding the unavailable alignment module: 39/39 PASS; the existing alignment test module
  could not import because OpenCV (`cv2`) is not installed in this Windows
  Python environment.
- API POST=0; GPU training=0; TEST image access=0; AutoDL startup=0; thesis
  repository access=0; Trusted Pair PASS=0.
- Historical images, annotations, outputs, permanent locks, and review CSVs
  remain unchanged. The protected untracked G2 audit script remains unstaged.

The legacy publication auditor is pinned to an obsolete historical base and
cannot serve as a full publication audit on this HEAD. Release review is
therefore limited to the exact staged files: text-only code/prompts/docs,
secret-pattern scan, private-path/data scan, staged diff review, and
protected-file exclusion. This scoped result must not be described as a full
historical publication-auditor pass.

## Next step

Wait for the independent code/data Gate and one explicit batch-level paid
authorization bound to this exact Manifest SHA, the four-source exact-set,
four-POST maximum, Prompt manifest SHA, model/protocol, and accepted cost
uncertainty. Source-specific authorization remains NO; the held corrosion
case is not eligible. After authorization, execute only those four sources,
then run the saved-output C0 alignment/composite pipeline and create the
private V1/V2 comparison atlas. Stop there for human review; do not expand to
the 60-100 source batch or train a model.
