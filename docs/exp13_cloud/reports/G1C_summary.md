# Exp13-G1C corrosion23 local composite and geometry audit

Status: `G1C_POSTPROCESS_COMPLETE / PENDING_HUMAN_REVIEW`

## Frozen inputs

- Original: 640x480, `c57384471d743fc179add88589fa0d7c8d59e59d14c6947ab9f6d4fd692dbddb`.
- FIRST_CALL_RAW: 1448x1086, `492ad7779bd026d767e93ff86e059b02f09fbbb5e3be7880bde828acb54903f6`.
- Both frozen corrosion polygons are represented by the 36,018-pixel mask.
- G1C API POST requests: `0`.

## Alignment

The exact aspect ratio is 4:3 for both inputs. The API output was resized once to 640x480 with Pillow LANCZOS and labelled `RESIZED_FOR_ALIGNMENT_ONLY`. No geometric warp was applied.

Geometry status: `GEOMETRIC_ALIGNMENT_UNVERIFIED`. Lightweight edge/patch evidence is recorded in `geometry_alignment_report.json`; resizing is not treated as proof of registration.

The resized raw output changes 222,154 non-mask pixels by more than five intensity levels (81.92% of the non-mask region; mean absolute channel difference 12.00). The best fixed edge search remains at `(0,0)`, but its Dice overlap is only 0.194. None of 30 high-texture non-mask candidate patches passed the preregistered correlation and uniqueness thresholds. This is insufficient correspondence evidence for pixel-local compositing.

## Composites

- Hard composite: `e1976e6bbd8eeae892099686145050bb2a13e0bc4e7fa74aface3e9f72158307`; outside-mask changed pixels: `0`.
- Fixed 3px inward feather: `9ed913febb4283624ac8340b705f1a1b2fb665af84cca35a2808678bc66d7614`; outside-mask changed pixels: `0`.
- Both are `POSTPROCESS_ONLY`, not direct model outputs and not trusted training pairs.

## Human review

Direct preliminary inspection of `review_sheet_g1c.png` and `local_boundary_review.png` found:

- The hard composite has strong polygon-shaped seams, brightness discontinuities, and generated texture-scale changes.
- The fixed 3px inward feather reduces the immediate seam but does not correct generated content geometry inside the mask.
- Generated edge/layer content is not continuous with the original blade boundaries, creating a high risk of reconstructed or newly introduced mechanical structure.
- Bright white damaged-looking material remains between and adjacent to the two frozen polygons.
- The result therefore retains residual-defect, texture, color, geometry, and mechanical-boundary risks.

Further user/ChatGPT review is recommended for documenting the failure mode. These two postprocess artifacts do **not** currently meet the engineering conditions to enter trusted training-Pair candidate review.

`TRAIN_PAIR_ELIGIBLE` is not granted. `RESTORATION_METHOD_PASS` is not granted.
