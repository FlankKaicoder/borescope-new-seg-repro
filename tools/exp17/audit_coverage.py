"""Count frozen candidate availability by source and near-duplicate group."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tools.exp17.audit_assets import CLASSES, load, need, sha, write


def coverage(candidates):
    sources, groups = defaultdict(list), defaultdict(list)
    for row in candidates:
        need(row.get("g0_status")=="G0_PASS", "Coverage requires resolved G0 identities")
        sources[row["source_id"]].append(row)
        groups[row["group_id"]].append(row)
    def summarize(items, key):
        return [{key: identity, "source_ids": sorted({r["source_id"] for r in rows}),
                 "polygon_count": len(rows), "variant_polygon_combinations": sum(len(r["variants"]) for r in rows),
                 "classes": sorted({r["class"] for r in rows}), "risk_flags": sorted({f for r in rows for f in r["risk_flags"]}),
                 "g1_strata": dict(Counter(r["g1_status"] for r in rows)),
                 "numeric_candidates": sum(r["numeric_screen_pass"] for r in rows),
                 "current_human_pass":0, "trusted_pairs":0} for identity, rows in sorted(items.items())]
    source_rows, group_rows = summarize(sources,"source_id"), summarize(groups,"group_id")
    summary={"source_images":len(sources),"independent_source_groups":len(groups),"candidate_polygons":len(candidates),
        "polygons_per_source_image_histogram":dict(Counter(len(v) for v in sources.values())),
        "polygons_per_source_group_histogram":dict(Counter(len(v) for v in groups.values())),
        "multiple_images_in_same_group":sum(len({r['source_id'] for r in v})>1 for v in groups.values()),
        "strata_source_groups_overlapping":{status:len({r['group_id'] for r in candidates if r['g1_status']==status}) for status in sorted({r['g1_status'] for r in candidates})},
        "risk_source_groups_overlapping":{flag:len({r['group_id'] for r in candidates if flag in r['risk_flags']}) for flag in sorted({f for r in candidates for f in r['risk_flags']})},
        "class_source_groups_overlapping":{c:len({r['group_id'] for r in candidates if r['class']==c}) for c in CLASSES},
        "warning":"Source-group counts for different classes, risks or strata overlap; do not sum them as independent groups.",
        "current_human_pass":0,"formal_trusted_pairs":0}
    return source_rows,group_rows,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidates','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--public-output',type=Path)
    a=p.parse_args();a.output.mkdir()
    source,group,summary=coverage(load(a.candidates)['candidates'])
    summary['audited_candidates_sha256']=sha(a.candidates)
    write(a.output/'source_coverage.json',{'sources':source})
    write(a.output/'group_coverage.json',{'groups':group})
    write(a.output/'summary.json',summary)
    if a.public_output:write(a.public_output,summary)
    print('PASS_SOURCE_GROUP_COVERAGE',summary['source_images'],summary['independent_source_groups'])


if __name__=='__main__':main()
