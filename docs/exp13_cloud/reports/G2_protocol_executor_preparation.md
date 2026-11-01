# G2 Evaluation Revision and Single-Source Executors

**Status:** `G2_PROTOCOL_CORRECTED / THREE_SOURCE_EXECUTOR_PREPARED / PENDING_CHATGPT_CODE_GATE_AND_USER_PAID_AUTHORIZATION`
**Date:** 2026-11-02
**Starting audited HEAD:** `6044d78bab74369bd6283f4bb6a0bff60296a7d9`

## Evaluation correction

The original [G2 input manifest](../g2/G2_THREE_SOURCE_MANIFEST.json) and all frozen inputs remain unchanged. The versioned [evaluation revision 2](../g2/G2_EVALUATION_REVISION_2.json) corrects only the interpretation:

| Source | Evaluation role | Counted as a new-source result? |
|---|---|---:|
| Crack260 | `NEW_CLOUD_SOURCE` | Yes; new Crack group |
| Crack470 | `NEW_CLOUD_SOURCE` | Yes; new Crack group |
| corrosion23 | `HISTORICAL_FAILURE_TARGETED_RETEST` | No; report separately |

G2 therefore has two new-source Crack outcomes, one historical corrosion targeted retest, and up to three separate new-request execution/usage/cost records. corrosion23 was used in G1B/G1C/G1D and its prior Pair Fidelity result was FAIL; G1B is not a G2 outcome. The completed corrosion2 G1F case is outside this cohort and every denominator. These three executions are not three independent new-source generalization examples.

The mixed-class exploratory Gate requires all three new request outcomes to be documented, at least one credible Crack result, and a credible corrosion23 targeted-retest result. Otherwise the Gate does not pass. This is not a significance test and it does not authorize training or automatically admit a Pair.

## Frozen inputs and prompts

The Pilot-4 ZIP SHA256 is `d9f049bda223f6485644da71839aed670a78b169b7411f722593c32fabe04f18`; the frozen Split Manifest SHA256 is `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`. All native full-frame ROIs, original professional JSON Polygons, and prior masks remain frozen. The execution pack revalidates source identity, TRAIN row/group, original image/JSON/ROI/mask hashes, Polygon completeness and alpha direction before request construction.

| Source | Original image / ROI SHA256 | Original JSON SHA256 | RGBA Mask SHA256 |
|---|---|---|---|
| Crack260 | `222419556d6bd4761d6ec6c66102b6bd4775a1989d517eb4cc93f91ecf3c2bd2` | `cc1147fadda327791890229b515b5b8aa1456729ba3dd9b52c42dbe767474344` | `0015bb114886c6c7f9bf3f0b78bbfd6452e788547b76695cad200ebd77294348` |
| Crack470 | `6d1537bb91157f82a679581d82fa3e20d72dde3b29cc14d2f47974931c3d0ec1` | `69b3741e1e0f626879d649881661cf4b0746208c3508b6204a236ba1e08551f4` | `43d80782e541d9e0c0687dd5440ffe0d6df2774be3c6cecfc3557eeb175eaab3` |
| corrosion23 | `c57384471d743fc179add88589fa0d7c8d59e59d14c6947ab9f6d4fd692dbddb` | `2cb23a813ddd4bf2f90a59a688ad22f6c9c5f1e27351f74d06875cb945e5e4df` | `14ed941a17735e1247abb7c478597a92712f03cabea4797dbe2b6c0bddce2984` |

Prompt file hashes are Crack `f00610b50a4bebb1f0dcee08eabe116420658b968daf0e791897cf40e7cedcbc` and corrosion `d08632c7d3ba2eb9c4b70a745521d39a6de50be0537d684f7e5dab8c0e802013`. The files end in one LF. The transmitted string is UTF-8 after removing exactly that terminal LF byte, with no other trim or rewrite. Transmitted-string hashes are Crack `61f1964b74f55673263d9629aac193e457fe057d2366c79649e3cb0c583dae0a` and corrosion `3d9bc700f3dadcbcd2ca21c5378050bd86ebbdb48e6c485e4f9195d4d519c3cb`.

## Executor and Dry Run

The frozen execution bundle at [execution_bundle.json](../../../configs/exp13_cloud/g2/execution_bundle.json) has SHA256 `cee4f8efacab9027dcbcfa8b97e17d1edfb5ed72947bb3fc459fcd317ddfa115`. It binds runtime code, fixed protocols/prompts, original G2 manifest, source roles and private input hashes. Model is `gpt-image-2.5-sunburst`; endpoint is `/v1/images/edits`; each request is `n=1` with independent multipart `image[]` and RGBA `mask`.

Three fixed wrappers each dispatch only its named source: [Crack260](../../../tools/exp13_cloud/g2/Start-G2Crack260.ps1), [Crack470](../../../tools/exp13_cloud/g2/Start-G2Crack470.ps1), and [corrosion23](../../../tools/exp13_cloud/g2/Start-G2Corrosion23.ps1). The shared launcher loads the Windows User Base URL/model and first line of the CurrentUser DPAPI file; it exposes the decrypted credential only in the Python child environment, clears temporary references, and records only allowlisted status fields. The Python executor defaults to Dry Run. Live additionally requires a source-specific explicit confirmation and a local receipt whose source role, model, one-call cap and execution-bundle SHA all match.

Each source has a separate permanent `locks/<source_id>/STARTED.json`, created exclusively after validation and immediately before the one POST. The lock is outside the output directory, cannot be overwritten, and remains occupied if delivery or billing is uncertain. A cross-process mutex prevents concurrent calls; any existing source lock directory or live output blocks reuse. The fixed three-source allowlist and lock count enforce the total ceiling of three. Retries, fallbacks, prompt search, redirects and parallel calls are disabled. No production lock or authorization receipt was created in this stage.

The offline multipart test passed for all three frozen inputs. After the final bundle was computed, each PowerShell DPAPI Dry Run was rerun against SHA256 `cee4f8efacab9027dcbcfa8b97e17d1edfb5ed72947bb3fc459fcd317ddfa115`: key decryption and ciphertext-file stability were verified; the child saw a credential and removed it from its environment; input/Prompt hashes, alpha semantics, model, endpoint and request structure were validated. The supplied API credential was not tested against the relay. The three local Dry Run manifests are under `E:\borescope_review_assets\cloud_pilot_g2_execution_20261102\dry_run\{260,470,23}\`.

Static code review caught and corrected an undefined `source_lock_dir` reference in the future Live path before freezing this final bundle. The correction ensures the permanent per-source lock directory is created before the sole POST; no input, Prompt or prior experiment asset changed. Python compilation, PowerShell parser validation and the offline executor contract tests pass on the final bundle.

| Source | Role | DPAPI / input Dry Run | API POST | Network requests |
|---|---|---|---:|---:|
| 260 | `NEW_CLOUD_SOURCE` | PASS | 0 | 0 |
| 470 | `NEW_CLOUD_SOURCE` | PASS | 0 | 0 |
| 23 | `HISTORICAL_FAILURE_TARGETED_RETEST` | PASS | 0 | 0 |

`live/`, `locks/`, and `authorizations/` do not exist under the new execution root. No real approval or STARTED marker exists.

## Future independent commands

After independent ChatGPT code Gate and explicit user paid-call authorization, run each source separately from the repository root. Each command requires that source's local receipt at `E:\borescope_review_assets\cloud_pilot_g2_execution_20261102\authorizations\<source_id>\gate_approval.json` and will fail closed without it.

```powershell
.\tools\exp13_cloud\g2\Start-G2Crack260.ps1 -Live -ConfirmOnePaidPost G2_CRACK260_ONE_PAID_POST -AssetRoot 'E:\borescope_review_assets\cloud_pilot_g2_three_source_20261101_retry1'
.\tools\exp13_cloud\g2\Start-G2Crack470.ps1 -Live -ConfirmOnePaidPost G2_CRACK470_ONE_PAID_POST -AssetRoot 'E:\borescope_review_assets\cloud_pilot_g2_three_source_20261101_retry1'
.\tools\exp13_cloud\g2\Start-G2Corrosion23.ps1 -Live -ConfirmOnePaidPost G2_CORROSION23_ONE_PAID_POST -AssetRoot 'E:\borescope_review_assets\cloud_pilot_g2_three_source_20261101_retry1'
```

These are future command forms only; they were not executed. Current status remains pending independent code Gate and paid-call authorization. The fixed result-audit protocol is [G2_RESULT_AUDIT_PROTOCOL.md](../g2/G2_RESULT_AUDIT_PROTOCOL.md). It requires the raw output and response/usage record, fixed G1F-2 geometry method, validity-aware global/local Pair candidates when geometry is supported, and four separate human gates. It never automatically grants training-pair eligibility.
