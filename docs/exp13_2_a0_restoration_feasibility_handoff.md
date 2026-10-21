# Exp13.2-A0 handoff

Current Gate: `RESTORATION_REVIEW_ASSETS_PREPARED_AWAITING_HUMAN_REVIEW`.

- Host: `autodl-container-8e424bb370-61e7c45a`
- Repository: `/root/autodl-tmp/borescope-new-seg-repro`
- Branch: `exp13/vlm-semantic-audit`
- Evidence-base HEAD: `1d85844362862993e358dfdf943be4951f95eb65`, equal to GitHub branch at preflight.
- Result root: `results/exp13_restoration_feasibility/a0_20261021_081812/`
- Method/results: [preparation report](exp13_2_a0_restoration_feasibility_preparation.md).

R1: 40 outputs / 20 sources. R2: 48 outputs / 4 sources, all shared with R1. Combined: 88 candidate files / 20 source groups (10 Crack, 10 corrosion); 16 groups have 2 variants, four have 14. Multiple variants are correlated; distinct output file hashes do not establish independent information.

All 352 references to 164 unique image/mask files are present and decodable. Source and masks share native dimensions, with the documented historical central-crop/fit transform to 512x512 reproduced. Ten source masks are partly cropped; corrosion 13 retains only 44.1285% of its native defect mask. No cropped-away region can be called restored. Full-frame pseudo-normal targets have not been established.

Contact sheets: `contact_sheets/crack_overview.png`, `contact_sheets/corrosion_overview.png`, and 20 `<class>_<source>_review.png` sheets in the result root. Each source sheet links RAW/annotation/edit mask/candidate/diff/paired ROI at recorded conditions. Review navigation: `review_navigation.md`.

Review form: `human_quality_review_template.csv`, 88 rows with all 792 human judgment cells blank. Reviewers must inspect actual files for residual defects, realistic texture/geometry, new holes or components, mask-boundary artifacts, smoothing and grey-border changes. No quality pass rate, accepted pair list or human scores exist yet.

Metrics: `quantitative_differences.csv` and `descriptive_metric_ranges.csv`. Pixel MAE is diagnostic only; mask feathering/dilation, crop, source JPEG decoding, resampling and historical model reconstruction affect interpretation. See the preparation report for numerical summaries and coordinate definitions. Four independent metric spotchecks matched; all 22 PNGs decode.

Protection: all 195 historical R1/R2 files retained identical SHA256 before/after. Existing Phase C1 files remain untracked and excluded; historical templates/Gates, model cache, datasets and paper repository are preserved.

New tracked scope: audit script, this handoff, preparation report, independent result package, and append-only PROJECT_STATE/DECISION_LOG/experiment_index/registry records. Commit identity and push result are reported after the commit in the delivery message; this file is part of that commit and does not claim a self-referential hash. Verify with `git log -1 --format=fuller` and `git ls-remote --heads origin exp13/vlm-semantic-audit`.

```text
CURRENT_TEXT_ONLY_INPAINTING_SSL_AUTHORIZATION = NO
SEVEN_CLASS_GENERATION_AUTHORIZATION = NO
FURTHER_R3_PARAMETER_SWEEP = NO
EXP13_2_GENERATION_AUTHORIZATION = NO
EXP13_2_B_AUTHORIZATION = NO
```

This is preparation for human review only. Polygon-derived masks are annotated supervision; any future method must disclose that source and cannot claim pure self-supervision without qualification. No new generation, downloads, model inference/training, VAL/TEST image access or paper-repository modification is authorized. Stop and wait for ChatGPT and user visual review.
