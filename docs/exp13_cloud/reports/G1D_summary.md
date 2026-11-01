# Exp13-G1D cloud repair mask protocol audit

Status: `FAIL_PAIR_FIDELITY / HOLD_EXPANSION`

This audit was static and offline. API POST requests, generation requests, GPU operations, and training runs were all zero. Existing G1B/G1C artifacts were read only.

## Findings

1. **Was the mask sent as an independent local field?** Yes. The executed code constructed one multipart POST to `https://direct-api.ccccapi.cc/v1/images/edits` with `model`, `prompt`, `n`, `image[]`, and `mask`. `api_mask.png` was the independent `mask` file.
2. **Is relay-to-upstream mask forwarding proven?** No. `RELAY_MASK_FORWARDING_UNKNOWN` remains true. There is no relay trace, upstream echo, or provider attestation.
3. **Is upstream use/adherence proven?** No. `MODEL_MASK_ADHERENCE_UNVERIFIED` remains true. The official API treats masks as model guidance, and no controlled masked/unmasked comparison exists.
4. **Was a code error found?** No mask pipeline error was found. Image/mask size and format, alpha polarity, multipart field name, and source hashes are correct. Two audit-metadata issues were found: the request manifest hashes the prompt file rather than the stripped transmitted text; G1C's regional `max_channel_difference` reports the global maximum. Neither explains or changes the primary pair-fidelity failure.
5. **Is current result still `FAIL_PAIR_FIDELITY`?** Yes. The independently reproduced G1C evidence remains: 222,154 outside-mask pixels changed by more than five levels (81.92%), edge Dice 0.1938, and 0/30 accepted reliable patches. Hard/feather outside-mask change counts independently reproduce as zero because they are postprocess composites, not model preservation.

## Response interpretation

The requested model was `gpt-image-2.5-sunburst`, but the response contains no model identifier, so `UPSTREAM_MODEL_IDENTITY_UNVERIFIED` remains true. The relay returned a third-party HTTPS image URL rather than base64. G1D did not revisit or disclose that URL. The returned `revised_prompt` exactly equals the stripped fixed prompt, so prompt rewriting does not explain the output. No `size`, `quality`, or `output_format` was sent; output selection remained service-side automatic and produced 1448x1086.

## G1C consistency

An independent Pillow implementation reproduced the saved LANCZOS alignment, Hard Composite, and 3px inward-feather image pixel-for-pixel. It also reproduced the 81.92% raw outside-mask change. G1C used the correct frozen original, Polygon mask, and dimensions. Its conservative `GEOMETRIC_ALIGNMENT_UNVERIFIED` conclusion remains valid.

The one calculation defect is limited to `max_channel_difference`: the implementation takes the full-image maximum before applying the region mask. Consequently the outside field shows 139 for Hard/Feather even though all regional counts, fractions, and mean are zero; the correct regional outside maximum is zero. This is non-material to the Gate decision.

## Next minimum

The first step should be zero-generation provider evidence for multipart mask forwarding and upstream model identity. If unavailable, the most informative minimal causal experiment is a separately authorized two-request synthetic mask/control-mask pair. It would require paid calls to distinguish mask-dependent behavior from stochastic full-image editing. A one-call fixed ROI experiment is the smallest project-specific attempt, but cannot isolate relay forwarding from model adherence.

Do not expand to the remaining three Pilot sources. Until forwarding and adherence are resolved, use this relay only as an exploratory visual restoration channel, not a formal paired-training-data generator.

Detailed evidence is in `request_protocol_audit.md`, `mask_pipeline_audit.json`, `g1c_consistency_check.json`, and `next_experiment_options.md`.
