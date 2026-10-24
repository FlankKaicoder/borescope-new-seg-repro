# Exp13 Phase 1-B-R2 handoff

Status: `NETWORK_HOLD` / `MODEL_ARTIFACT_HOLD`

Evidence: `results/exp13_phase1b_r2/20261024T134647Z/summary.json`.

The official acquisition channel remains untrusted and unavailable. DNS answers changed between resolvers and could not be cross-checked through Google or Cloudflare DoH. Both trusted environments timed out on Hugging Face metadata and the official LFS CDN. The supplied revision reference is not a local verification result.

Model calls and budget use remain zero. No source or mask was frozen, and no Diffusion Smoke artifact exists. Resume only after official API metadata and weight files are reachable, the complete revision-pinned manifest and hashes are captured, and the transfer path is proven. Preserve all previous HOLD evidence and do not use a third-party mirror or fallback model.
