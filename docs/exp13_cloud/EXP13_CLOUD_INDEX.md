# Exp13 cloud restoration experiment index

This is an independent post-project extension. Exp11's final frozen TEST and
Exp12's conclusions stay frozen. This Git synchronization runs no experiment,
image generation, image reconstruction, API probe, training, GPU or TEST evaluation.

Branch base: `origin/exp13/vlm-semantic-audit` at
`3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`.
Work branch: `phase/exp13-cloud-repair`. Local main is not the parent of this
sync commit; no merge or history rewrite is performed.

## Stage ledger

POST counts below are historical **image-generation POSTs**, not this sync's
Git network operations or output-download GETs. The stages account for 3
historical image-generation POSTs in total; this sync adds **0**.

| Stage | Goal and current recorded status | Git code entry | Prompt | Local asset ID | Important hash reference | Historical POSTs | Gate / next authorization |
|---|---|---|---|---|---|---:|---|
| G1B | corrosion23 first real edit; FIRST_CALL_RAW / PENDING_HUMAN_REVIEW | [executor](../../tools/exp13_cloud/g1b/gate_b_live_once.py), [input preparation](../../tools/exp13_cloud/g1b/prepare_inputs.py), [audit](../../tools/exp13_cloud/g1b/audit_live_output.py) | CLOUD_RESTORATION_V1 | cloud_pilot_g1b_relay | g1b.corrosion23.raw | 1 | Later G1C/G1D reject pair fidelity; no repeat |
| G1C | Geometry audit plus Hard/Feather; GEOMETRIC_ALIGNMENT_UNVERIFIED | [postprocess](../../tools/exp13_cloud/g1c/run_g1c_postprocess.py) | Inherits G1B | cloud_pilot_g1c_postprocess | g1c.hard / g1c.feather3 | 0 | FAIL_PAIR_FIDELITY / HOLD_EXPANSION; no trusted Pair |
| G1D | Request/Mask pipeline and G1C independent audit | [protocol audit](../../tools/exp13_cloud/g1d/offline_protocol_audit.py) | Inherits G1B | cloud_pilot_g1d_protocol_audit | report/code provenance rows | 0 | Independent local mask confirmed; relay/upstream use unproven |
| G1E-0 | Non-industrial synthetic left/right Mask control prepared; DRY_RUN_PASS | [synthetic preparation](../../tools/exp13_cloud/g1e/prepare_g1e.py) | MASK_CONTROL_TEST_V1 | cloud_pilot_g1e_mask_control | prompt/code provenance rows | 0 | Historical preparation Gate followed by separately authorized G1E-1 |
| G1E-1 | Both requests succeeded; MASK_CONDITIONED_SPATIAL_RESPONSE_OBSERVED | [pair executor](../../tools/exp13_cloud/g1e/g1e_live_pair.py), [audit](../../tools/exp13_cloud/g1e/audit_g1e_pair.py) | MASK_CONTROL_TEST_V1 | cloud_pilot_g1e_mask_control | g1e.A.raw / g1e.B.raw | 2 | Spatial response observed; strict compliance/direct forwarding not proven; no rerun |
| G1F-0 | Frozen TRAIN corrosion2 ROI A/B prepared; PENDING_REVIEW historical snapshot | [ROI preparation](../../tools/exp13_cloud/g1f0/prepare_g1f0.py) | CLOUD_RESTORATION_V1 | cloud_pilot_g1f_roi_prepare | g1f.A.roi / g1f.B.roi | 0 | Original A/B unchanged; later B4:3 selected by user |
| G1F-0B | B4:3 ROI/Mask and CurrentUser DPAPI loader prepared; DPAPI_LOADER_DRY_RUN_PASS | [ROI correction](../../tools/exp13_cloud/g1f0b/prepare_roi_4x3.py), [loader](../../tools/exp13_cloud/g1f0b/Start-BorescopeRoiDryRun.ps1) | CLOUD_RESTORATION_V1 inherited, no call | cloud_pilot_g1f_roi_prepare/g1f_0b_4x3 | g1f.B4x3.roi / g1f.B4x3.api_mask | 0 | Subsequent user gate G1F_0B_PREPARATION_PASS / ROI_B4X3_SELECTED |
| G1F-1A | G1F_1A_PREFLIGHT_PASS / LIVE_NOT_STARTED / STOP | [protocol](../../tools/exp13_cloud/g1f1a/protocol.py), [executor](../../tools/exp13_cloud/g1f1a/single_repair.py), [DPAPI launcher](../../tools/exp13_cloud/g1f1a/Start-Corrosion2Repair.ps1), [audit](../../tools/exp13_cloud/g1f1a/audit_saved_output.py) | CORROSION2_ROI_REPAIR_V1, user exact text | cloud_pilot_g1f_1a_preflight | g1f.B4x3 inputs + prompt | 0 | G1F-1 LIVE HOLD; independent ChatGPT GitHub audit and separate explicit paid-call approval required |

No Preflight PASS is a successful real corrosion2 repair. No stage admits a
trusted training pair, authorizes batch generation, or demonstrates long-run stability.

## Frozen key evidence

| Asset ID | SHA256 |
|---|---|
| g1b.corrosion23.raw | `492ad7779bd026d767e93ff86e059b02f09fbbb5e3be7880bde828acb54903f6` |
| g1e.A.raw | `4deeee9c25fd911833536a12013b4e7200bbabf4a03bc929ef786ac184f9635b` |
| g1e.B.raw | `15d5e03c3728138e4934225d90ec3c289f26d4fd26ba59fcef1457bb3a3d2f4f` |
| g1f.B4x3.roi | `887a206bb4c550f9b8ba006684a93d1089619f06bec57c3d1e6e16648dabd74a` |
| g1f.B4x3.api_mask | `da23f8bb1533a83f107bf8b2ef5d5d3f072bc0a9f5ba827393b60af107845dfc` |
| CORROSION2_ROI_REPAIR_V1 file/request text | `9ee1b2971ced4ccc81d21bbbe0cb433c8d5b57007d9159d70e559842f259476c` |

B4:3 is `[0,0,320,240)`, 320×240. The exact original polygon has 2394
transparent pixels (3.1171875%), alpha 0 inside and 255 outside. UI cropping
and metal-edge/cavity-fill risks remain. Source is frozen TRAIN corrosion2;
there is no M1 private unblinding-map access or TEST-data dependency.

Full hash-only linkage: [private evidence](evidence_sha256.csv).
Exact code/prompt source and public-copy hashes: [provenance](code_provenance.csv).
Reports are selected text-only historical summaries, with private root replaced
by `${PRIVATE_ASSET_ROOT}`. The summaries retain their stage-time PENDING/HOLD
wording; the table above records later user selections separately.

## Documents and version units

- [Eight stage summaries](reports/): G1B, G1C, G1D, G1E-0, G1E-1, G1F-0, G1F-0B, G1F-1A.
- [Publication and asset policy](PUBLICATION_POLICY.md).
- [Reproducibility and code limitations](REPRODUCIBILITY.md).
- [GitHub audit handoff](GITHUB_AUDIT_HANDOFF.md).
- [Sanitized registry](experiment_registry.csv), also registered in the repository-wide registry.

This initial synchronization is one historical-registration commit, not a
scientific-stage completion or authority grant. Future complete G1F, G2 Pilot,
dataset freeze, and downstream-training stages should each use one reviewed
version unit, rather than committing every small Dry Run. None is authorized
by this synchronization.
