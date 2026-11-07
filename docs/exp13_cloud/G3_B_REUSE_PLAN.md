# Exp13 G3-B minimal reuse plan

This is a design and reuse checklist only. No G3-B request, model load, image
generation, or training is performed by this handoff.

## Reuse map

| Requirement | Reuse | Required hardening |
|---|---|---|
| Source selection | `g2_single_executor.py` bundle validation and source allowlist | Replace the three fixed source wrappers with one manifest row per source; preserve source role and frozen image/JSON/Mask hashes. |
| Seven-class Prompt | G2 `CRACK_RESTORATION_V1` and `CORROSION_RESTORATION_V1` files | Add five explicit class prompts in a versioned prompt index; each row stores prompt file and transmitted-string SHA256. No prompt search or implicit fallback. |
| Polygon to RGBA Mask | G2 preparation path and alpha checks | Rasterize every original professional Polygon at native size with the frozen Pillow rule; target alpha 0, outside alpha 255; record binary and RGBA hashes. |
| ROI | G2 full-frame native ROI policy | Keep `[0,0,width,height)` unless a separately reviewed manifest row defines another fixed ROI. No resize, padding, letterbox, or crop during POST. |
| Request lock | G2 `STARTED.json` and cross-process mutex | Use a permanent exclusive lock per source row, created immediately before the only POST. A timeout or uncertain delivery consumes the row permanently. |
| Execution order | G2 sequential wrappers | Iterate manifest order in one process, one source at a time, with total and per-source caps. Retries, fallback models, redirects, concurrency, and prompt search remain zero. |
| Evidence ledger | G2 request/response/terminal records | Record input/prompt/bundle hashes, HTTP status, response format and dimensions, raw output SHA256, usage and reported cost or explicit UNKNOWN. Never store credentials or signed URLs. |
| Geometry | G1F-2 fixed CPU audit and G2 protected single-result audit | Run native grayscale SIFT, BF/L2 k=2, Lowe 0.75, 2-pixel deduplication, Similarity and Affine RANSAC at 3 ROI pixels. Use a source-specific UI exclusion map only when frozen; otherwise report GEOMETRY_UNVERIFIED. |
| Resolution changes | G2 output dimensions plus fixed validity mask | Estimate registration in native ROI coordinates; make a conservative inverse-mapped validity mask for each output resolution. Invalid pixels are never treated as restored metal. |

## Required gates

Every result keeps separate `VISUAL_ACCEPTED`, execution/evidence, geometry, and
Pair fields. Human review must independently decide defect removal, mechanical
geometry, alignment, and Pair usability. A numeric correspondence pass yields
`GEOMETRY_CANDIDATE` only; it never writes `TRUSTED_PAIR_PASS` or authorizes
training. A missing cost, upstream identity, or UI map remains `UNKNOWN`.

## Planned manifest fields

`source_id`, `class`, `split`, `group_id`, `image_path`, `json_path`, native
dimensions, ROI half-open coordinates, Polygon count and point counts, image/JSON
Mask hashes, prompt ID and file/wire hashes, prior-call history, maximum POSTs,
and review status. The manifest is the only source of execution order and budget
eligibility.
