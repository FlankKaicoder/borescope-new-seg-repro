# Exp13 Phase 1-A geometry design

Native polygon masks are the edit core. A context ROI box is clamped to the image boundary and mapped to a square model canvas with explicit aspect-preserving resize and padding. Image resize and mask resize policies are recorded separately; masks use nearest-neighbor mapping and have a deterministic inverse map.

The transition halo is a narrow dilation outside the polygon. Pixels outside approved edit support remain immutable. Gray/black/padding appearance is not inferred from geometry: it is marked `UNRESOLVED_HEURISTIC` and, where relevant, `FOV_UNCERTAIN`. A clamped ROI is a risk flag, never automatic evidence that the border is padding.

ROI grouping uses sorted Union-Find connected components over deterministic bounding-box proximity. Multi-ROI composition uses stable ownership by ROI key, independent of input or execution order. An explicit `hold_on_overlap` policy returns `ROI_COMPOSITION_HOLD`; skipping a risky region is not allowed to manufacture a PASS. Large ROI context risk, thin Crack, non-square crops, and multi-class sources remain auditable fields.
