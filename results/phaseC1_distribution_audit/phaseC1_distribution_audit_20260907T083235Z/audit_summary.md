# Phase C-1 Distribution Audit Summary

- Audit run: `phaseC1_distribution_audit_20260907T083235Z`
- Preview run: `phaseC1_preview_20260907T075012Z_seed42_1_0`
- Split manifest SHA256: `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`
- Preview pairs: **2004**
- Edits: **3991**
- Family counts: `{'A_small_low_contrast': 1146, 'B_diffuse_texture': 1195, 'C_elongated_ribbon': 1011, 'D_irregular_boundary': 639}`
- Class edit counts: `{'Burn': 862, 'Crack': 481, 'Dent': 712, 'Material missing': 930, 'Tears': 269, 'Tip curl': 160, 'corrosion': 1132}`
- Class family coverage: `{'Burn': 4, 'Crack': 4, 'Dent': 4, 'Material missing': 4, 'Tears': 4, 'Tip curl': 4, 'corrosion': 4}`
- Outlier candidate counts: `{'NOISE_PATTERN_CANDIDATE': 780, 'WEAK_CHANGE_CANDIDATE': 96}`

## Gate Decision

**CONDITIONAL_PASS**

### Conditional reasons
- VISIBLE_RATIO_BELOW_0_5_FRACTION_OVER_5_PERCENT:C_elongated_ribbon:0.417409
- VISIBLE_RATIO_BELOW_0_5_FRACTION_OVER_5_PERCENT:D_irregular_boundary:0.424100
- OUTLIER_CANDIDATES_PRESENT:NOISE_PATTERN_CANDIDATE=780,WEAK_CHANGE_CANDIDATE=96
- IMAGE_LEVEL_CHANGED_RATIO_OVER_30_PERCENT:0.389822

This audit is read-only. It does not authorize proxy training when the decision is not PASS.
