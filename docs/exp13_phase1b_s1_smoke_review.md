# Exp13 Phase 1-B-S1 SDXL GPU Smoke Review

Run: `20261025T044024Z`
Status: `EXP13_PHASE1B_S1_HOLD`
Repository starting/final pre-report HEAD: `e64436185f5d5b36aba79c7643aa1fc66f955af3`

## Decision

Stop before model loading or inference. Source `2` is a valid single-ROI TRAIN candidate, but source `17` has the required two independent ROIs and one cannot be safely approved: its corrosion polygon reaches x=718.008 in a 720-pixel image, its ROI clamps at x=720, and visual inspection shows the polygon extends into the bright right frame/FOV and annotation-overlay region. The required source is not silently replaced or reduced. The task also prohibits proceeding with source 2 alone when source 17 fails.

The contact sheet is retained on the controlled AutoDL result path at `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s1/20261025T044024Z/source_fov_review_contact.png`. It was used only for input boundary assessment; it contains no generated output.

## Preflight evidence

- AutoDL branch synchronized to `e64436185f5d5b36aba79c7643aa1fc66f955af3`; this is based on GitHub R3 delivery.
- Frozen TRAIN manifest SHA256 matches `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`.
- Source `2`: image `640x480 RGB uint8`, one corrosion polygon, non-clamped ROI `[24,48,176,129]`; image SHA256 `514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03`; native mask SHA256 `ff94041bce6b68941bb1ebc0cc6d36d243bb30cb1b6d2767b912f4edc519bdc2`.
- Source `17`: image `720x540 RGB uint8`, two polygons in two ROI groups; image SHA256 `84faf142b5177e0f212c8e8d5ac0cccdd16153d1c667720b30cd49d2a5cd1dcc`. Crack `roi_0` is non-clamped; corrosion `roi_1` native mask SHA256 `4e37c30975dfea788befc16117f0207e8e8abe17e37ecc5ba0c156908fa88b15`, ROI `[419,63,720,460]`, FOV status `FOV_UNCERTAIN`.
- The local model path was rechecked against the supplied R3 manifest: 19/19 files and 6,941,223,330 bytes match, with no mismatch. This is manifest agreement, not a new official API query.
- GPU at probe: RTX 3080 Ti, 12,288 MiB total, 11,912 MiB free. Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1, Safetensors 0.6.2, Pillow 11.3.0, NumPy 2.3.2, psutil 7.0.0.

## Gate status

| Gate | Result | Reason |
| --- | --- | --- |
| S1-A model artifact preflight | PASS | Pinned local model files agree with R3 manifest |
| S1-B memory/runtime | NOT RUN | Stop occurred before model load |
| S1-C mask/canvas alignment | NOT RUN | No model canvas was submitted |
| S1-D source 2 inference | NOT RUN | Source plan is atomic across both required sources |
| S1-E source 17 inference | HOLD | `roi_1` intersects unresolved image boundary/FOV risk |
| S1-F output/quality evidence | NOT RUN | No generated output exists |
| S1-G budget/data protection | PASS | Persistent schema-v2 ledger exists; zero reservations and calls; no VAL/TEST or raw writes |

No `from_pretrained`, GPU model load, inference, automatic retry, fallback model, VAL/TEST access, raw dataset write, SSL, or YOLO training occurred. Actual ROI calls: 0. The S1 parameter set was not used because inference did not start. Restoration quality cannot be assessed. The initial two command attempts that failed to import the repository package and the successful command/exit code are recorded in `results/exp13_phase1b_s1/execution_record.json`; the successful stdout/stderr remains on AutoDL at `/tmp/exp13_phase1b_s1_20261025T044024Z_stdout_stderr.log`.

This is a bounded `HOLD`, not a model failure. Do not infer a Smoke PASS or proceed to Pilot, batch generation, trusted-pair freezing, SSL, or YOLO training. Resolving source 17's FOV eligibility or changing the frozen source plan requires a new explicit decision.
