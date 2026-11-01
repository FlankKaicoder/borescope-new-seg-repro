# G2 Three-Source Preparation

**Status:** `G2_THREE_SOURCE_PREPARATION_COMPLETE / PENDING_CHATGPT_GATE`
**Preparation date:** 2026-11-01
**Private asset ID:** `cloud_pilot_g2_three_source_20261101_retry1`

## Frozen cohort

| Source | Class | TRAIN group | Native size | Full-frame ROI | Original Polygon(s) | Mask pixels | Prompt |
|---|---|---|---:|---|---:|---:|---|
| 260 | Crack | `singleton_260` | 720×540 | `[0,0,720,540)` | 1 (20 points) | 6,575 (1.6911%) | `CRACK_RESTORATION_V1` |
| 470 | Crack | `near_0007` | 960×720 | `[0,0,960,720)` | 1 (199 points) | 3,112 (0.4502%) | `CRACK_RESTORATION_V1` |
| 23 | corrosion | `singleton_23` | 640×480 | `[0,0,640,480)` | 2 (18/22 points) | 35,994 (11.7168%) | `CORROSION_RESTORATION_V1` |

The Pilot-4 ZIP SHA256 matches `d9f049bda223f6485644da71839aed670a78b169b7411f722593c32fabe04f18`. The frozen Split Manifest SHA256 matches `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`. Package integrity checks pass original image decoding, professional JSON parsing, image/annotation dimensions, Polygon bounds, frozen TRAIN membership, copied-file hashes and manifest consistency. All three image and JSON hashes match the package source mapping and frozen TRAIN rows. Each source has a complete original image and original professional Polygon JSON; no derived mask was used as annotation evidence.

Each ROI is the complete native frame. No crop, resize, padding or coordinate transform is applied. Binary and API RGBA masks were generated from all original JSON Polygons. Target alpha is 0 (transparent), and non-target alpha is 255 (opaque). The [sanitized Git manifest](../g2/G2_THREE_SOURCE_MANIFEST.json) includes hashes and metadata only.

Source 23 is corrosion23 and has prior G1B/G1C/G1D exposure. It is retained because the verified Pilot-4 package contains no other corrosion TRAIN source besides excluded corrosion2. This history is disclosed and does not count as a G2 outcome or denominator. Completed G1F corrosion2 is historical context only and is also excluded from the G2 denominator.

## Frozen protocol and limits

The two fixed prompt files and hashes are recorded in the manifest. The proposed request contract is model `gpt-image-2.5-sunburst`, endpoint `/v1/images/edits`, independent `image[]` and RGBA `mask`, and `n=1`. The ceiling is three first-generation POSTs total and one per source. Retries, fallback, prompt search, concurrency and redirects are disabled. This freezes a future protocol only; it does not authorize execution.

The exploratory Gate requires all three new outcomes to be recorded and at least one credible candidate per class (minimum two). It is not a significance test and does not authorize training. G1F is not part of the success denominator. Sources may not be replaced after generation begins.

## Safety and execution state

- API image-generation POSTs: **0**.
- GPU, training, VAL/TEST access, AutoDL: **0 / not used**.
- Paid authorization or per-source STARTED markers: **not created**.
- M1 private unblinding map and thesis repository: **not accessed**.
- Private images, professional JSON, masks, overlays and outputs: **excluded from Git**.

corrosion23's two Polygons cover a relatively broad area; one reaches the left image boundary and lies near a metal/cavity transition. This is a later visual-review risk, not grounds to alter the original Polygon.

**Next step:** independent ChatGPT review of source identity, provenance, prompts and protocol. A later explicit user authorization is required before any of the three single-shot requests. No call is authorized by this preparation commit.
