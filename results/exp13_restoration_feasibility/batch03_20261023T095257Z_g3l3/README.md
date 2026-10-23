# Exp13.2-A2 Batch03 historical review packet

Prepared UTC: 2026-10-23T09:56:38.836772+00:00. `BATCH03_REVIEW_PACKET_PREPARED_AWAITING_HUMAN_REVIEW`. Evidence HEAD: `75dda701c054fb9ea6efb4ffbf4332fbed797f4e`.

Start with `contact_sheets/batch03_overview.png`, then the four source sheets and eight variant sheets. All are review composites of existing TRAIN-only R1 files. No model, generation, inference or quality assessment occurred. The derived aligned input uses the recorded deterministic ImageOps.fit transform; this is a display reconstruction, not a newly generated candidate or a claim that serialized historical input pixels were recovered.

## Selection

`source_selection_audit.json` lists all20 source keys,8 previously covered and12 remaining (Crack5/corrosion7), derived from the original88-row A0 index, historical source ledger and Batch01/02 manifest/overlay joins. Quota:3corrosion+1Crack. At each step prefer a historical screening status not yet represented in this batch, then numeric source ID ascending. Class corrosion first. This selects corrosion1,9,6 and Crack469; every ID and both R1variants exist in A0. Selection uses no current visual rating or generated output comparison. R2 is excluded. Four selected sources leave8historical sources uncovered.

## Review navigation and image semantics

- `contact_sheets/*_review.png`: native RAW copy in a panel, original mask/crop overlay, aligned input, aligned edit mask, v1/v2 at equal512display size.
- `variant_review_sheets/`: annotation/crop, input, aligned original defect overlay, edit mask, historical candidate and unamplified absolute RGB difference. Differences are descriptive, not quality scores.
- `batch03_manifest.csv`:8unique candidates; exact historical relative/absolute paths, original SHA256/dimensions, run/prompt/seed/mask provenance, crop/native counts and source-level historical context. Source opinions are never candidate decisions. Missing prior decisions remain blank with a separate no-record indicator.
- `batch03_human_review_template.csv`: independent blank template. All user decisions, scope/comments/reviewer fields and9quality fields are empty. Both freeze/quality booleans are false. Do not populate from screening labels or pixel differences.
- `historical_screening_context.md`: historical subjective source notes, separate from neutral image panels. Includes corrosion9 hallucination warning and Crack469 uncertainty. No new judgment is made.
- `package_integrity.json`: file inventory, crop cross-checks, image dimensions, joins and preservation validation.
- `SHA256SUMS.txt`: every package file except itself, including integrity JSON. The external ZIP checksum is in the delivery receipt.

Red tint marks the original native defect mask; cyan marks crop bounds. Panels preserve aspect ratio with letterboxing, RGB uses LANCZOS and masks NEAREST. Output v1/v2 are shown at equal size and without preference. Gray letterbox is a display-only margin. Edit mask grayscale includes historical feathering; no mask/prompt modification occurs. Original historical bytes remain in place and are linked rather than duplicated. Full paths/hashes are in manifest; display-only resizing/recompression affects composite pixels only.

## Spatial scope

Mask retained fraction = native nonzero original annotation area within historical center crop / all native annotated area. It is not repair success or a mask annotation completeness guarantee. Cropped-away defect pixels are outside the reviewed output. A fraction1.0 does not imply the entire native image is represented. Existing R1 rendering and Diffusion pixel changes, dilation/feather boundary artifacts and gray input borders require human review; the recorded deterministic grid does not certify semantic geometry.

- corrosion 1: crop[90.0, 0.0, 630.0, 540.0]; 2996/3074 retained=0.9746258945998699; outside=78 annotation pixels.
- corrosion 9: crop[90.0, 0.0, 630.0, 540.0]; 72709/77694 retained=0.9358380312508044; outside=4985 annotation pixels.
- corrosion 6: crop[90.0, 0.0, 630.0, 540.0]; 2789/4757 retained=0.5862938826991801; outside=1968 annotation pixels.
- Crack 469: crop[120.0, 0.0, 840.0, 720.0]; 1562/1562 retained=1.0; outside=0 annotation pixels.

## Human review and authority

Inspect residual defect, normal metal texture, mechanical hallucinations, geometry, edit boundary/halo/smoothing, and changes outside the edit. Smoothness alone is not normality. Low-strength retained cracks are not successful repair. Each source's two variants are correlated and count as one independent source. User supplies final semantic decisions; ChatGPT may assist later and determine Gate under separate review. Polygon masks originate from annotations; future learning use must disclose supervision.

Current sufficient formal admission evidence remains0for previously reviewed sources; adding review assets creates no admissions. A1 proposed10trusted sources,5per class is unfrozen and not a statistical guarantee. User acceptance alone cannot certify Restoration Quality PASS. All historical templates, overlays, masks, outputs and Phase C1 files are protected. No VAL/TEST images, dataset image reads or paper repository access.

```text
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

Stop for human/ChatGPT review. No next batch or experiments are authorized by this packet.
