# Exp13 Phase 1-B-S4-R2 Handoff

Final state: `EXP13_PHASE1B_S4_R2_COMPLETED_PENDING_CHATGPT_GATE`

This was a TRAIN-only, CPU-only, zero-inference archival and independence
audit. The S4-R1 strict-26 package was bridged to the preliminary visual review
source `CHATGPT_PRELIMINARY_VISUAL_REVIEW`; the bridge remains preliminary and
does not create user acceptance, industrial-expert sign-off, restoration
eligibility or trusted pairs. The review records 24 candidates as unsuitable
for the current compact ROI text-only SDXL protocol and retains `285:roi_0`
and `378:roi_0` for method-design discussion only. Formal eligible count and
trusted-pair count remain zero.

The audit verified 26 unique source IDs, 26 unique image SHA256 values and 26
unique annotation SHA256 values, with no exact collisions. It computed 325
pairwise CPU similarity records. The exploratory near-duplicate proxy found
zero pairs, but capture metadata, physical-part identity and sequence/viewpoint
information are absent; independent physical scenes are therefore unproven.
The leaf-notch group has the strongest observed pair (`639:roi_0` and
`701:roi_0`, gray cosine 0.9037), which remains only a related-scene hypothesis.
All user-specified risk groups and the `367:roi_0` normal-structure Mask risk
are preserved.

Route decisions remain: current compact SDXL `HOLD_NO_GO_FOR_CURRENT_PROTOCOL`;
historical DreamShaper `HOLD_AUDIT_ONLY_NO_DIRECT_TRAINING_REUSE`; classical
CPU texture repair `CONDITIONALLY_WORTH_A_SEPARATE_CPU_CONTROL_ONLY`. No new
experiment is authorized. A future `RESTORATION_ELIGIBILITY_GATE` must check
UI/FOV contamination, normal-metal context, physical boundaries, native
information/scale, padding ratio, raw output artifacts, native paste-back and
machine-readable selection provenance.

The authoritative S2 ledger remains
`results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json`: 3 occupied
and actual calls of 4, one remaining. This run consumed 0 model calls, loaded
0 models, accessed VAL/TEST 0/0, wrote raw data 0 and ran no training.

AutoDL evidence is at
`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s4_r2/20261025T092025Z/`.
Only small JSON evidence was copied into Git; industrial review PNGs and ZIPs
remain on the controlled server. Historical S2/S3/S4/S4-R1 assets and all
untracked server assets are preserved. Stop here for ChatGPT Gate; do not spend
the remaining Smoke call or enter DreamShaper, SDXL, Pilot, SSL or training.
