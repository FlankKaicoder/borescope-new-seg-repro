# G2 Result Audit Protocol

This protocol applies independently to each future G2 source result. It does not authorize generation or training.

## Preserve request evidence

- Keep the exact API-returned image bytes as `raw_output.bin` and an extension-named byte-identical copy. Never resize, recompress, or overwrite the original result.
- Record requested source/role, input and Mask hashes, Prompt file and transmitted-string hashes, execution-bundle hash, request count, HTTP status, elapsed time, response model identifier, and usage/cost fields when supplied.
- Do not persist API credentials, signed output URLs, or full sensitive transport logs. If an output URL is returned, retain only a sanitized host/path digest and fetch the image once without forwarding authorization or following redirects.
- If delivery, status, output decoding, or billing is uncertain, leave the source slot occupied and stop. Do not retry.

## Geometry and visual audit

- Record actual output format, channels, and dimensions before creating derived views. Any scale, crop, padding, or coordinate transform must be explicit; diagnostic resizing is not registration.
- Use the fixed G1F-2 correspondence method: SIFT features, Lowe ratio 0.75, RANSAC threshold 3 pixels in the original ROI coordinate system, with the prespecified Similarity and Affine models. Do not search parameters per source.
- Report full-frame and UI-excluded matches; exclude the target defect and its neighborhood when assessing stable geometry. Record inliers/outliers, reprojection errors, spatial distribution, valid-overlap mask and structural coverage. UI-only matches cannot establish mechanical correspondence.
- Only if evidence is sufficient, create global aligned and local masked-composite Pair candidates, each with a separate validity mask and explicit transformation metadata. Keep the raw result unchanged. Record seam/blending artifacts and changes outside the target region.
- Prepare whole-source, defect, metal-edge/cavity, alignment, and Pair comparison views. Do not infer success from pixel differences alone.

## Human gates

Record `DEFECT_REMOVAL`, `MECHANICAL_GEOMETRY`, `ALIGNMENT`, and `PAIR_USABILITY` separately. The user makes the final visual-quality decision. Geometry evidence does not by itself establish defect quality, and a plausible-looking output does not establish pixel correspondence. No result is automatically marked training-pair eligible.

## Separate evaluation groups

Crack260 and Crack470 are the only `NEW_CLOUD_SOURCE` outcomes used for the new Crack-source result count. corrosion23 is `HISTORICAL_FAILURE_TARGETED_RETEST`; report it separately from new-source generalization. The old G1B result contributes no new G2 outcome. All three new calls, statuses and usage/cost records are tracked; they do not constitute three independent new-source generalization cases. G1F corrosion2 is excluded from this cohort and all G2 denominators.

The exploratory mixed-class Gate requires at least one credible Crack candidate and a credible corrosion23 targeted-retest candidate, with all three outcomes documented. Failure to meet either class condition means the Gate does not pass. This is exploratory evidence only: no significance claim and no training authorization.
