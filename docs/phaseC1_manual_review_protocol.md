# Phase C1 Manual Review Protocol

## Purpose

Phase C1 Distribution Audit Gate C remains `CONDITIONAL_PASS`, and the completed Quality Review concluded `INSUFFICIENT_EVIDENCE_NEEDS_MANUAL_REVIEW`. The quantitative evidence is sufficient to identify the candidate groups, but it is not sufficient to confirm that the Local Change generator has a design defect.

This protocol governs annotation of the additive manual-review preparation artifacts. It does not change the generator, preview run, audit thresholds, preview data, model state, or any existing review label.

## Current evidence

- The completed quality-review CSV has 876 candidate rows for 860 unique edits, and all existing `review_label` values remain blank.
- Weak-change candidates total 96. Of these, 86 are family C, 8 are family D, and 2 are family A.
- Noise-pattern candidates total 780. Family C contributes 324 and family D contributes 439.
- Family C has 41.74% visible-change ratios below 0.50 and mean 19.84 connected components.
- Family D has 42.41% visible-change ratios below 0.50 and mean 80.23 connected components.

The prepared package includes every weak-change candidate plus deterministic fixed-seed samples of 100 C-noise and 100 D-noise candidates. It is an annotation preparation set, not a replacement for the full 860-unique-edit disposition recommended by the Quality Review.

## Review materials

Each contact-sheet tile contains source context, edited context, a thresholded change-mask context, and the required edit metadata. The red rectangle is the stored bounding box for the named edit.

The edited preview can contain one to three sequential edits. No per-edit rendered masks were retained by the completed preview run. Therefore the displayed change mask is the combined difference of the final preview pair, cropped around the named edit's stored bbox. It must not be treated as an edit-isolated mask or used to attribute unrelated changes to the named edit.

`manual_annotation_template.csv` intentionally adds `source_id` to the requested fields because `edit_id` is only unique within a source image. All `human_label` and `comment` values are blank when the package is created.

## Annotation rules

Review the source and edited contexts together. Use the saved bbox and metadata to identify the named edit. If the composite preview cannot support a reliable conclusion for the named edit, leave `human_label` blank and explain the limitation in `comment`; do not guess.

Use exactly one of the following labels for a reviewable item:

| Label | Use when |
|---|---|
| `VALID_SMALL_CHANGE` | The named edit is a coherent, appropriately subtle localized change. |
| `VALID_STRUCTURE_CHANGE` | The named edit is a coherent morphology-consistent structural change, including a thin or irregular but plausible local change. |
| `INVALID_NOISE` | The named edit has no coherent local change and is dominated by unrelated speckle or noise. |
| `INVALID_FRAGMENT` | The named edit is materially fragmented without a coherent intended local region. |
| `OVER_STRONG` | The named edit is visibly too broad, strong, or destructive for a local change. |

Do not alter the source CSVs, samples, generator, masks, preview images, thresholds, or candidate definitions while annotating. Record observations only in the additive annotation template or an explicitly versioned successor.

## Gate C follow-up

After annotation, a read-only review must report completion and label counts by candidate type and morphology family, retaining missing labels and comments. It must separately account for edits with more than one candidate reason.

No label distribution automatically promotes Gate C to `PASS`. A subsequent audit decision may reconsider Gate C only when the manual evidence is complete enough to support the decision. Systematic `INVALID_NOISE`, `INVALID_FRAGMENT`, or `OVER_STRONG` findings for C/D require a separately authorized, separately numbered generator-repair design; they do not authorize an in-place generator change. Predominantly valid findings may support a documented reassessment of the frozen Gate C evidence.

Proxy Training remains disallowed until a later recorded Gate C decision is `PASS`.
