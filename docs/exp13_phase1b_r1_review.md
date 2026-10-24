# Exp13 Phase 1-B-R1 SDXL preflight recovery review

Status: `EXP13_PHASE1B_SMOKE_HOLD` / `NETWORK_HOLD`

Evidence: `results/exp13_phase1b_r1/20261024T132018Z/`.

The bounded recovery attempt did not reach model acquisition. AutoDL DNS resolved Hugging Face IPv4 addresses, but IPv4 TCP 443 failed; IPv6 had no usable route; therefore TLS/HTTPS and the Hugging Face API were not reached. The Windows controlled host resolved Hugging Face but its bounded HTTPS request timed out. No proxy was configured, TLS verification was not disabled, no token or certificate was saved, and no third-party mirror was used.

Because the official endpoint was unavailable from both trusted environments, the exact commit revision, openrail++ license text, component list, required file sizes, LFS/SHA256 metadata and FP16 completeness could not be verified. No model file was downloaded or transferred. The 20.8 GiB repository size was not incurred; no temporary artifact was created.

The installed AutoDL runtime and disk/GPU facts were recorded. R2 geometry identifies source `2` as a single-ROI candidate and `17` as a double-ROI candidate, but final boundary/hash freeze was deferred. No budget reservation, model loading, inference, output, contact sheet, VAL/TEST access or raw-data write occurred.

Gates: B-S1 `HOLD`; B-S2 `CONDITIONAL_PASS_NO_CALLS`; B-S3 `CONDITIONAL_PASS_R2_CANDIDATE_ONLY`; B-S4/B-S5 `NOT_TESTED`; B-S6 `PASS`. This is a network/artifact hold, not evidence of model or restoration quality. Do not retry indefinitely, use an unapproved mirror, switch models, run Pilot, freeze pairs or start SSL.
