# Exp13.2-M0-R1 handoff

`EXP13_2_M0_R1_PREPARED_AWAITING_HUMAN_INPUT_REVIEW`

Read [review](exp13_2_m0_r1_source_reselection_review.md) and `results/exp13_2_m0_r1/20261025T122430Z_m0r1/`. Six geometry-only TRAIN sources are frozen in order **Crack469,260,470 / corrosion8,1,2**. Both new corrosion candidates passed in the first two preregistered inspections; later priority candidates and supplemental pool were not opened. Six distinct image hashes/TRAIN group IDs are not proof of six physical parts.

The four M0 defaults are reused with actual file/pixel hashes verified, without rewriting any historical input. Source9 span exclusion and23 context exclusion are unchanged. All16 proposed crops retain every polygon and positive edit-support pixel, meet native-square/margin/no-padding rules, have no threshold-mask coverage omission, and use identical per-model FOVs. Human context/semantic eligibility is **PENDING for all6**. Default files remain frozen; no alternative is silently selected.

Human input review ZIP, controlled AutoDL only:

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_m0_r1/20261025T122430Z_m0r1/input_protocol_review_package.zip`

SHA256 `6f683f6521737437ef37a5f24e24581865fd59467694dca779102c15a3bf53fc`;62,181,247 bytes;16 sheets (6 defaults +10 alternatives);230 entries checked against file/pixel hashes, shapes and effective masks. Exact RGB/mask PNGs, native/cropped polygon and edit masks, effective binary masks and support overlays are inside. `input_protocol_review_contact_sheets/` is the sheet folder. Public Git contains no industrial PNG or ZIP.

Start with `human_input_review_template.csv` (6 rows,48 PENDING cells) and `human_input_protocol_review_manifest.json`. Each source needs an explicit chosen crop and answers for crop context, mask coverage, structural preservation, UI/FOV acceptability, normal metal sufficiency, final decision and comment. Options per source:

- 469/260/470/8: `default`, `alt_1`, `alt_2`.
- 1/2: `default`, `alt_1`.

Every source may be rejected. UI and structural-boundary risks remain in the package: source1 contains an overlay near a target; source2 includes GE/UI, cavity/dark background and a real edge; even260 alternatives retain a metal boundary. Assistant spot checks are evidence preparation, not user semantic decisions. S2 versus S5 supports investigating input construction, not a crop-only causal conclusion. Do not overwrite the initial PENDING template when archiving future decisions. No objective geometry/hash error can be waived by human PASS.

Preferred proposed future ceiling12 calls (two per source). If later approval leaves3+2, ceiling10; minimum2+2 requires Crack469 and260, with470 held, plus the first two user-accepted corrosion in order8,1,2. If those conditions fail, HOLD. No replacement after observing new outputs. No call or reservation is authorized now.

M0 model assets/parameters are unchanged and CPU-rehashed: DreamShaper ModelScope/master local fingerprint with no authenticated immutable upstream revision; SDXL fixed115134f363124c53c7d878647567d04daf26e41e. Prompt, negative None,20 steps, guidance7.5, strength0.8, seed130500,FP16, DreamShaper512/DEIS and SDXL1024/Euler remain frozen. No model was imported, loaded or downloaded.

Authority remains S5 `results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json`, SHA256 `3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94`:4 occupied/4 actual, Smoke0, aggregate116, all-local296. This phase adds0 inference/load/download/reservation/VAL/TEST/raw write/split write/SSL/YOLO/trusted pairs. Any future M1 requires separate authorization and one shared persistent locked successor retaining all historical counters, pairwise atomic reservations and STARTED persistence; failures charged, no retries or replacement.

Report/remote start7e859bfc20e75f47e65fbfa22e25f017e3f8879f; AutoDL execution checkout remains787a3be0190dfa9b11c5416a5b686a6bae4598d9. Do not synchronize or clean the server. It retains all historical untracked assets and old evidence. Git commit/push and exact final remote verification are provided in the delivery response; the containing commit is the report version.

Next action: ChatGPT audits this package and the user explicitly reviews the unified input schemes. Only after input acceptance may M1 dual-model inference authorization be requested. Do not load models, run M1/Pilot/SSL/YOLO, resume A5, freeze trusted pairs or expand budgets automatically.
