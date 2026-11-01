# Reproducibility and source provenance

The 23 experiment source files are byte-identical snapshots of existing verified
Windows code. This sync does not run them or assert that copying them into a new
layout is a new validation. `code_provenance.csv` records both source and public
SHA256; `.gitattributes` prevents Git newline conversion for these snapshots and
the fixed prompts. Only report-root paths and report line endings are sanitized.

## Code layout and prerequisites

| Code directory | Role | Private runtime layout expected by historical code |
|---|---|---|
| tools/exp13_cloud/g1b | Original/JSON preparation, Gate A dry mode/probe, single live call, audit | Original G1B configs, code imports and runs layout |
| tools/exp13_cloud/g1c | Resize diagnosis, edge/patch metrics, hard/inward-feather compositing | Private G1B inputs plus independent G1C output root |
| tools/exp13_cloud/g1d | Static mask protocol and independent regional metric checks | Existing G1B/G1C roots and private request/response metadata |
| tools/exp13_cloud/g1e | Two synthetic rectangles, independent alpha masks, bounded A/B execution/audit | inputs/configs/reports/runs; user-approved two-call history is already terminal |
| tools/exp13_cloud/g1f0 | Frozen source/TRAIN checks and A/B polygon crops | Pilot ZIP, corrosion2 source_snapshot and original private assets |
| tools/exp13_cloud/g1f0b | Direct 4:3 crop, mask/pixel checks, DPAPI Dry Run loader | Original G1F tree and a new private g1f_0b_4x3 runtime directory |
| tools/exp13_cloud/g1f1a | Exact multipart, default dry mode, permanent lock, Gate receipt, future audit | Private inputs, prompt_v1.txt, matching bundle lock and request manifest; no approved receipt is supplied |

Python/Pillow and, for DPAPI, Windows PowerShell under the owning Windows user
are needed. The snapshot retains historical Windows paths and interpreter path
defaults to preserve evidence. These are not portable installed packages. Do
not run them against an existing historical output directory or inside the
public checkout. Any future authorized reproduction must provision a separate
private workspace and explicitly review paths before execution. Path changes
create a new code/bundle hash and need fresh preflight; an old Gate receipt
cannot approve changed code.

Runtime JSON configs, request manifests, original JSON annotations, private
source-mapping rows, bundle locks and approval/STARTED files are intentionally
not copied as executable runtime state. Public fixed protocol metadata is in
`configs/exp13_cloud/protocol_summary.json`. These omissions are private state,
not missing implementation code. The source includes the preparation routines
that document the expected schemas. Do not fabricate missing provenance or
re-run historic calls to fill a documentation gap.

## Prompt bytes

Three exact prompt files are in `configs/exp13_cloud/`. The older G1B/G1E
executors transmit stripped text while their files may end in a newline;
both hashes are recorded in `prompt_hashes.csv`. G1F-1A's new prompt has UTF-8,
no BOM and no terminal newline, and file/request hashes are identical. Do not
substitute the older restoration text for the new corrosion2 prompt.

## Preserved limitations

- G1D found no local mask pipeline polarity/size/field error. It cannot prove
  relay forwarding or upstream model identity.
- G1B's original manifest records the prompt-file hash rather than the stripped
  transmitted text. This is preserved and explained, not retroactively fixed.
- G1C's historical `max_channel_difference` takes the global maximum, even for
  regional fields. For hard/feather outside regions the correct maximum is 0,
  while the historical field says 139. Counts, fractions and regional means
  remain valid; the pair-fidelity failure is unchanged. G1D contains independent
  regional checks. Historical source/output metrics are not overwritten.
- G1E-1 shows a mask-conditioned spatial response, not strict shape adherence,
  direct proof of upstream mask forwarding, or statistically stable success.
- G1F-1A diagnostics distinguish raw visual quality, raw pair geometry and any
  postprocessed-pair usability; resizing is not registration. A 4:3 ROI does
  not guarantee returned dimensions or preserve the nearby metal/cavity edge.
- `seal_preflight.py` is a historical command-style script with top-level
  validation/writes. It must not be imported by a code-inspection tool.
- `relay_probe.py` contains network probes; it is archived but not executed here.
- G1B/G1E live executors are terminal historical implementations, not a rerun
  authorization. G1F-1 remains LIVE_NOT_STARTED and on HOLD.

Static syntax, hashes, index contents and publication safety may be inspected
without importing experiment modules or running their CLI, tests, loaders,
image-generation/preparation, audits, or live branches. This sync performs only
those static/package-management checks.
