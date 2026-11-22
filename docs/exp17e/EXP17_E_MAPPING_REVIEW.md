# Exp17-E frozen N mapping and target-support audit

This CPU-only audit starts from the A10-authoritative 379 selected assets and the new original-annotation census. It does not select a replacement N, fit a transform, change a historical matrix, load a model or infer new normal labels. All Polygons for a source retain that source's single selected A10 N SHA256.

## Actual result

| A10 selected N type | Sources | Sources with historical mapping | Original Polygons | Polygons with complete target support |
|---|---:|---:|---:|---:|
| Aligned | 346 | 346 | 528 | 525 |
| Raw | 29 | 28 | 43 | 38 |
| Bounded | 1 | 1 | 1 | 1 |
| Crop | 3 | 3 | 4 | 4 |
| Total | **379** | **378** | **576** | **568** |

The 575 mapped Polygons span 378 source images / 358 independent groups. The 568 fully supported Polygons span 373 images / 353 groups. These are technical coverage counts, not human acceptance or Data Gate counts. Group counts for different asset types or classes can overlap.

Eight Polygons in seven sources / seven groups have exceptions: one lacks a reliable historical mapping; seven have incomplete original-grid Valid support and incomplete mapped-N support. Four of the latter also have a mapped vertex outside the selected N frame. No transform, image or threshold was changed to remove these exceptions. The exception CSV preserves exact identities and original/N pixel counts.

Current SHA256 checks cover 956 unique mapping input files, 253,132,592 bytes, and recheck them after the operation. The mapping run took 25.21 seconds. The census separately verified all 379 originals, professional JSONs, selected N files and published N copies. New derived Valid Mask files are stored only in this phase's private output; historical masks remain unchanged.

## Coordinate direction and asset identity

The saved A9 `matrix_output_to_source` maps generated **Raw pixels to original source coordinates**. Each reused transform must be an already accepted historical affine candidate, tied to the same source/annotation/review/Raw identities. An inverse is calculated algebraically only when the selected N is Raw. Inverting a stored affine is not new registration.

For **Aligned**, the selected bytes must equal the exact same-review Aligned hash and its dimensions must equal the original source grid. Its source-to-N mapping is identity because alignment was already applied. Applying the Raw transform a second time would be wrong. The historical matrix and Valid Mask remain part of the binding evidence.

For **Raw**, the selected bytes must equal the same-review Raw hash. Source-to-N is the inverse saved Raw-to-source matrix. The original-grid Valid Mask is projected into native N coordinates using nearest-neighbor sampling with explicit integer-pixel-center conventions. Image dimensions alone never establish correspondence. One Raw has no reliable historical matrix and remains `MAPPING_UNVERIFIED`; its N coordinates, N ROI and local Valid Mask are null. Only independent original/N full views are appropriate there.

For **Crop**, the same-review Aligned parent hash, exact integer crop bounds, parent-to-crop translation, selected dimensions and every selected crop pixel are checked. Source-to-N is the recorded integer translation. Original and N target support are checked separately, so accepting a crop cannot extend to source pixels outside it.

For **Bounded**, selected bytes remain Bounded. Its saved base image, request, edit mask, Aligned parent and Valid Mask are bound to the same historical record. CPU replay proves Bounded equals Aligned on `edit_alpha_zero AND valid`, and equals the frozen base elsewhere. The original source grid is established by this composite provenance, not by equal dimensions. Editing support and geometric support are separate fields; an inherited base region outside the current edit would require its own provenance rather than being mislabeled invalid geometry.

The current Bounded case has a base equal to the original source bytes and one 377-pixel GT target fully inside the effective edit. The record preserves that exact fact rather than assuming every Bounded region has this property.

## Pixel-preserving regional lineage supplement

An additional read-only supplement verifies that the selected Bounded target's **377 RGB pixels exactly equal** the same-review previously accepted Aligned target. The three selected Crops similarly preserve all four original target rasters under integer translation and exactly preserve their parent target RGB pixels. All five target regions have same-review historical Polygon PASS and B CONFIRMED.

This supports retaining historical semantic evidence for those exact target supports while keeping A10's selected Bounded/Crop bytes as N. It does not substitute the parent image, extend acceptance to other Polygons or assert equality of larger contexts or Backbone receptive fields. The public [region-lineage summary](../../results/exp17e_a10_379_polygon_expansion/region_lineage_summary.json) binds the separate private proof files; the original mapping receipts are retained unchanged.

Reliable Raw mapping likewise preserves source/version lineage, but the selected Raw is not the exact Aligned display pixel grid. Keep its historical regional semantics and A10 whole-image acceptance as separate evidence. The independent Data Gate must determine whether any additional current-N scope/appearance confirmation is needed; whole-image acceptance alone cannot fabricate local mapping or regional user PASS.

## Mask and ROI conventions

Original GT rasters use the historical floating-Polygon Pillow fill convention on the original image grid. Transformed Polygon coordinates and native-N rasters are stored separately. Raw native areas can differ because N resolution and affine scale differ; their pixel counts must not be interpreted as additional instances or directly compared as equal-area exposures.

Validity uses binary nearest-neighbor interpolation only. Full support requires a nonempty original target with every source target pixel valid, a nonempty native-N target with every native target pixel valid, and the mapped Polygon inside the selected N frame. The audit retains both sides' pixel counts and every reason. It does not relax full target support because a missing fraction is small.

ROI windows are new display-only derivatives around the appropriate Polygon in each coordinate system. A verified matrix permits a correctly labeled mapped ROI; a support failure remains visibly flagged and does not become approval. No verified matrix means no paired N ROI at all. A source's other Polygons may be overlaid using the same bound transform, never a matrix from another version.

## Provenance limits and verification

For 186 selected sources, the historical evidence contains the audited matrix and pinned image/Valid identities but lacks a historical whole-geometry-file SHA descriptor. The new digest binds that saved matrix to exact source/N/Valid identities and does not invent an old whole-file hash. The immutable saved A9 evidence hash is published. These weaker file-provenance cases remain distinguishable from missing or unverified mapping.

Six synthetic CPU tests passed: inverse-direction correctness, nearest-mask pixel-center/binary preservation, integer crop translation and unsupported target handling, no fabricated ROI without H, invalid affine / incomplete source Valid handling, and separation of Bounded edit provenance from geometric validity. The actual Bounded/Crop RGB lineage checks also passed. No new registration, generation, model forward, optimizer update, held-out image access or automatic trusted Pair was performed.

The root expansion report combines these technical results with historical and current-N semantic evidence. [Mapping summary](../../results/exp17e_a10_379_polygon_expansion/mapping_summary.json) is not a Data Gate authorization. The real augmentation adapter, joint-forward verification, λp calibration and formal training remain separate future work.
