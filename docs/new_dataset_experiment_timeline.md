# New dataset experiment timeline

| Stage | Outcome |
|---|---|
| Exp00 | Environment, pairing/schema, polygon/scale, duplicate/leakage audits completed; Gates documented. |
| Exp01 | 969-image Dataset v1 built; 668/154/147 group-aware split; zero near-duplicate cross-split leakage. |
| Exp02 | YOLO11n-seg baseline completed; AMP NaN root cause isolated; numerical waiver accepted. |
| Exp03–04 | Low-confidence positive diagnostic; one-class no clear gain. |
| Exp05 | Preliminary seed42 Hard Mining positive candidate. |
| Exp06–07 | ROI CE/SupCon completed; Stage2 negative. |
| Exp08 | KD skipped by engineering Gate; not evaluated. |
| Exp09 | SimSiam reconstruction invalidated by 0/120 trainable backbone updates; downstream not evaluated. |
| Exp10 | Controlled three-seed verification completed; Hard Mining not confirmed. |
| Review10.5 | Full evidence review; Option A direct finalization selected. |
| Candidate Freeze | Baseline seed44 frozen before TEST; commit `9991fcfcb9cf6c0ab8920ad7deadeed579ce5585`. |
| Exp11 initial | Client interruption before metrics; evidence preserved; invalidating Gate raised. |
| Exp11 authorized retry | Exactly one unchanged retry authorized; final TEST and qualitative audit PASS. |
| Project Complete | Results commit `fb584621b816de8f344d49daeba2656caee46e92`; no training or selection after TEST. |
