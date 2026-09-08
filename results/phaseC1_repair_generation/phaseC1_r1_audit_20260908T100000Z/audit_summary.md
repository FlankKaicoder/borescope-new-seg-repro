# Phase C-1 Distribution Audit Summary

- Audit run: `phaseC1_r1_audit_20260908T100000Z`
- Preview run: `phaseC1_r1_formal_20260908T084500Z`
- Split manifest SHA256: `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`
- Preview pairs: **2004**
- Edits: **4052**
- Family counts: `{'A_small_low_contrast': 1216, 'B_diffuse_texture': 1225, 'C_crack_like_structural_evolution': 1011, 'D_boundary_evolution': 600}`
- Class edit counts: `{'Burn': 846, 'Crack': 474, 'Dent': 705, 'Material missing': 947, 'Tears': 263, 'Tip curl': 169, 'corrosion': 1196}`
- Class family coverage: `{'Burn': 4, 'Crack': 4, 'Dent': 4, 'Material missing': 4, 'Tears': 4, 'Tip curl': 4, 'corrosion': 4}`
- Outlier candidate counts: `{'WEAK_CHANGE_CANDIDATE': 2, 'NOISE_PATTERN_CANDIDATE': 13}`

## Gate Decision

**CONDITIONAL_PASS**

### Conditional reasons
- OUTLIER_CANDIDATES_PRESENT:NOISE_PATTERN_CANDIDATE=13,WEAK_CHANGE_CANDIDATE=2
- IMAGE_LEVEL_CHANGED_RATIO_OVER_30_PERCENT:0.349211

This audit is read-only. It does not authorize proxy training when the decision is not PASS.
