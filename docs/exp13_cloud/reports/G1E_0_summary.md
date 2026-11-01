# Exp13-G1E-0 mask spatial-control preparation and dry run

Date: 2026-11-01 (Asia/Shanghai)

Status: `G1E_0_DRY_RUN_PASS / G1E_1_NOT_STARTED / STOP`

## Scope

This stage generated one non-industrial synthetic image, two location-controlled RGBA masks, review-only overlays, a frozen prompt/configuration, and the future sequential two-condition execution/audit code. It sent zero API POST requests and zero image-generation requests.

No industrial image, frozen dataset, AutoDL resource, GPU, TEST data, thesis repository, or historical G1B/G1C/G1D artifact was modified.

## Synthetic input

- Canvas: 640x480 RGB PNG.
- Background: uniform RGB `(211,211,211)`.
- Rectangles: identical pure blue RGB `(0,0,255)`.
- Coordinate convention: left/top inclusive; right/bottom exclusive.
- Left rectangle: `[100,160,220,320)`, 120x160, 19,200 pixels.
- Right rectangle: `[420,160,540,320)`, 120x160, 19,200 pixels.
- Original SHA256: `d1b5a1b303bc3ab293282ba0b0b882e691cd3c347790b281b1164be1a000a0d0`.
- The original contains no target label, text, arrow, border, or overlay.

## Masks

Both masks are 640x480 RGBA PNGs with white RGB channels and alpha values restricted to 0/255.

| Condition | Transparent region | Transparent pixels | SHA256 |
|---|---|---:|---|
| A | Left rectangle only | 19,200 | `bba4e012f62149fe834b9bac628ce76ad5ff8a3f1e6da96e1cd663afcc496438` |
| B | Right rectangle only | 19,200 | `cd86dfd2b7df262b8df93e148f092d4b108e9573eff1bf3c709a5016f504f48b` |

The transparent areas have equal size and shape, different positions, and zero overlap. `overlay_left.png` and `overlay_right.png` are review-only files and are not referenced by the request code.

## Frozen request

- Prompt ID: `MASK_CONTROL_TEST_V1`.
- Prompt file SHA256: `25af3b08202ca91952574f9791555d1702e3874ba1c50ad3b166373246f58b96`.
- Actual stripped multipart prompt SHA256: `ba87e316b3c7cd119f144599bede1ddeb1534962365db00cbb6a18ac5e79ee58`.
- Endpoint: `https://direct-api.ccccapi.cc/v1/images/edits`, frozen from the successful G1B protocol.
- Model: `gpt-image-2.5-sunburst`.
- Fields: `model`, `prompt`, `n`, `image[]`, `mask`.
- `n=1` per condition.
- Condition order: A, then B.
- Both conditions read the exact same original file bytes and prompt text; only the mask file/location differs semantically.

The multipart structure follows the official OpenAI image-edit guide. The guide also states that mask use is prompt-based guidance and is not guaranteed to follow the exact shape. Reference: https://platform.openai.com/docs/guides/image-generation

## Safety controls

- Future generation POST cap: 2.
- Automatic retries: 0.
- Model fallback: 0.
- Prompt search/modification: 0.
- Concurrency/background tasks: 0.
- Redirect following for POST: disabled.
- Condition B is called only after condition A returns one valid decodable image.
- Any A HTTP, transport, parse, download, or image error terminates the session before B.
- Any uncertain send creates/retains a `STARTED` marker and is charged as occupied; no automatic retry is possible.
- `g1e_live_session_started.json` uses exclusive creation and blocks every rerun.
- Each condition has separate exclusive STARTED and terminal markers.
- The safety self-test verified duplicate marker refusal and all frozen stop conditions without network access.

The current Codex child process does not inherit `OPENAI_API_KEY` or `OPENAI_BASE_URL`. A live-mode safety check confirmed that missing environment variables stop execution before creating any session marker or run directory.

## Dry-run outcome

- `G1E_0_DRY_RUN_PASS`.
- API POST requests: 0.
- Image generation requests: 0.
- `runs/` live artifacts: none.
- `g1e_review_sheet.png`: intentionally not created; requires real A/B outputs.
- `G1E_FINAL_REPORT.md`: intentionally not created; requires real A/B outputs.

## Future G1E-1 command

Only after user and ChatGPT review/authorization, run this once in the already configured PowerShell session:

```powershell
Set-Location -LiteralPath '${PRIVATE_ASSET_ROOT}\cloud_pilot_g1e_mask_control'; py .\code\g1e_live_pair.py --live --confirm-max-two-paid-requests AUTHORIZE_G1E_TWO_CONDITION_MAX2_POSTS
```

Do not run the command twice. A failed or ambiguous execution consumes the exclusive session and must not be retried.

## Remaining risks

- Relay-to-upstream mask forwarding remains unverified.
- Upstream model identity and mask adherence remain unverified.
- Generated output may have different dimensions or geometric drift; observation resizing cannot establish pixel alignment.
- Two conditions can show an observable mask-conditioned spatial response but cannot establish strict compliance or statistical significance.
- If both conditions redraw broadly or behave inconsistently, the result must be `MASK_SPATIAL_CONTROL_NOT_DEMONSTRATED` or `MASK_SPATIAL_CONTROL_INCONCLUSIVE`, not proof that the relay discarded the mask.
