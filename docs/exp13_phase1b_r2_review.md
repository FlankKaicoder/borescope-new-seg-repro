# Exp13 Phase 1-B-R2 trusted model channel validation

Status: `NETWORK_HOLD` / secondary `MODEL_ARTIFACT_HOLD`

Evidence: `results/exp13_phase1b_r2/20261024T134647Z/`.

This run was limited to read-only network and official model acquisition checks. Windows and AutoDL system DNS returned different, unverified answers. Google DoH was unreachable from both paths and Cloudflare DoH was reset, so the system answers could not be authenticated. Hugging Face root HTTPS, the fixed-revision API, the fixed-revision `resolve` endpoint and the official `cdn-lfs.huggingface.co` endpoint were all unreachable within bounded timeouts.

The user-supplied public reference revision `115134f363124c53c7d878647567d04daf26e41e` and `openrail++` label are recorded as references only; neither was locally verified in this run. No model metadata, component list, file size, LFS hash, download, transfer, cache, load or inference exists. No third-party mirror, DNS/hosts change, proxy change, TLS bypass or token was used.

No data, VAL/TEST, training, SSL, Pilot or budget ledger was touched. The next valid path requires a user-controlled environment that can reach both the official Hugging Face metadata API and weights CDN, followed by per-file hash verification before transfer. Do not retry this same network path or enter Smoke.
