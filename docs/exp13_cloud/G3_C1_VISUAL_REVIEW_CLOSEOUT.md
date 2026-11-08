# Exp13-G3-C1 visual review closeout

Status: `G3C1_VISUAL_REVIEW_CLOSEOUT_READY`.

The recovered review file matched its pinned SHA256 and all 36 unique G3-C0
sources. The corresponding frozen Geometry values also matched exactly.

| Class | VISUAL_PASS | NEEDS_RETRY | UNREVIEWED |
|---|---:|---:|---:|
| Burn | 5 | 1 | 0 |
| Crack | 6 | 0 | 0 |
| Dent | 6 | 0 | 0 |
| Material missing | 3 | 3 | 0 |
| Tears | 6 | 0 | 0 |
| corrosion | 5 | 1 | 0 |
| **Total** | **31** | **5** | **0** |

The visual-by-geometry counts are: `VISUAL_PASS` with
`GEOMETRY_CANDIDATE` 24, `VISUAL_PASS` with `ALIGNMENT_UNVERIFIED` 7,
`NEEDS_RETRY` with `GEOMETRY_CANDIDATE` 4, and `NEEDS_RETRY` with
`ALIGNMENT_UNVERIFIED` 1. Thus 24 outputs merit bounded Pair-quality review
from the geometry screen; one has a user note that explicitly rejects its use
as a training Pair despite a recorded visual pass, leaving 23 immediate Pair
review proposals pending clarification. None is a Trusted Pair. A geometry
candidate is a feature-screen result, and a hard/feather composite is a local
post-process; neither establishes pixel-level Pair validity.

The five `NEEDS_RETRY` causes are a changed mechanical boundary, three
incomplete/incorrect material-missing reconstructions, and one corrosion
case with residual anomaly plus a Polygon/UI overlap risk. One other corrosion
record is `VISUAL_PASS`, while its note describes visible residual corrosion
and explicitly says it is unsuitable as a training Pair. That decision/note
conflict is preserved for user clarification; no status was edited.

Two further records need separate interpretation. A visually accepted crack
output has `ALIGNMENT_UNVERIFIED`, so no source-registered Pair follows. A
corrosion raw provider output removed green camera UI visible in the source;
the source-based hard composite retains that UI outside the selected target
Mask. Review the raw output and the composite as distinct artifacts.

## Next work

1. Perform a bounded CPU Pair-fidelity review for the 23 unambiguous visual and geometry candidates, keeping raw, aligned and composite evidence separate. Review UI preservation explicitly for the identified corrosion case. Do not promote any Pair automatically.
2. Draft a frozen Prompt V2 for the boundary-preservation and material-completion failures. The corrosion item whose final visual state conflicts with its note needs clarification first. Keep the Polygon/UI-overlap case on hold until its Mask semantics are reviewed.
3. Defer increasing paid batch size until Pair-fidelity results and the V2 protocol have been reviewed. This closeout grants no paid requests.

The full source-linked statistics and complete user notes remain in the private
closeout package. API POST=0, GPU training=0, TEST image access=0. The review
CSV was frozen byte-for-byte in the private asset directory; the source CSV
was not modified.
