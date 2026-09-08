# Phase C1-R1 Semantic Quality Review Protocol

## Purpose

Phase C1-R1 repaired the C and D Local Change morphologies, and its formal generation gate passed. The read-only repair audit remains `CONDITIONAL_PASS`; it shows improved geometric acceptance but does not establish that a generated change resembles the evolution of a real borescope defect. This protocol prepares a human semantic review of the saved V2 output and does not make a semantic-validity decision automatically.

## Scope and evidence boundary

The package reads only the completed R1 formal artifacts in `results/phaseC1_repair_generation/phaseC1_r1_formal_20260908T084500Z/`: `generation_manifest.json`, `final_status.json`, `edit_metrics.csv`, and saved preview pairs. It does not change the generator, V2 logic, preview, morphology definitions, split, model, or prior results.

Each tile contains the full source panel, source and edited crops around the stored edit bbox, and a thresholded final-preview difference crop. The final preview may contain multiple sequential edits. The displayed change mask is therefore the combined source/final preview difference, not an isolated mask for the named edit. A reviewer must leave the label blank and explain the ambiguity when that evidence cannot support a reliable edit-level judgment.

The allowed R1 formal artifacts contain no `source_id`-to-class mapping. Every `class` entry is therefore `CLASS_PROVENANCE_UNAVAILABLE`; class names are not inferred from source IDs. The requested seven-class coverage cannot be verified in this package. Each family sample is deterministic with seed `42` and has unique source IDs within that family.

## V2 repair targets

- Family C, `C_crack_like_structural_evolution`, replaced the V1 elongated ribbon with a curved centerline and variable-width ribbon. The generation acceptance checks require one component, adequate skeleton length, and elongation. Its target is crack and tear evolution.
- Family D, `D_boundary_evolution`, replaced the V1 irregular random mask with a deformed primary region and boundary evolution. The generation acceptance checks require one component, an adequate largest-component ratio, and a boundary score. Its target is material-missing, tear, and corrosion boundary change.
- Families A and B were sampled only as V2 sanity checks. They were not changed by the R1 repair.

## Annotation rules

Enter one label and an evidence-based comment only when the named edit can be assessed from the saved panels. Do not edit the sample CSVs, previews, masks, generator, or threshold definitions. Use `semantic_annotation_template.csv` for the annotation record.

| Family | Allowed label | Meaning |
|---|---|---|
| C | `CRACK_LIKE_VALID` | A coherent, localized crack or tear-like structural evolution follows the stored bbox. |
| C | `LINE_ARTIFACT` | The result reads as a drawn line or graphic trace rather than defect evolution. |
| C | `TOO_WEAK` | The intended structure is not visible enough to assess as a defect change. |
| C | `UNNATURAL` | The structure is visible but implausible for crack or tear evolution. |
| D | `BOUNDARY_EVOLUTION_VALID` | A coherent main region changes principally at its boundary in a plausible way. |
| D | `MASK_EXPANSION_ONLY` | The result is a simple expansion or contraction without credible boundary evolution. |
| D | `FRAGMENTED` | The visible region is materially broken into disconnected fragments. |
| D | `UNNATURAL` | The boundary change is visible but not plausible for material loss, tears, or corrosion. |
| A or B | `VALID` | The stored edit is coherent for its original family. |
| A or B | `WEAK` | The named change is visible but too weak for reliable review. |
| A or B | `UNNATURAL` | The change is visually implausible for the family. |

## Gate follow-up

This preparation package has status `PREPARED_AWAITING_HUMAN_ANNOTATION`. Completion of the package does not promote Gate C. After human annotation, a separately recorded read-only Gate review must report label counts by family, unresolved evidence, and the class-provenance limitation.

Gate C stays `CONDITIONAL_PASS` until that later decision explicitly records `PASS`. Proxy Training, SSL pretraining, and downstream evaluation remain disallowed while the semantic labels are blank or the Gate is not `PASS`.
