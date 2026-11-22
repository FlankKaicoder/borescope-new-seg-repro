"""CPU byte-lineage supplement for pixel-preserving selected Bounded/Crop regions."""
import argparse
from pathlib import Path

from PIL import ImageChops

from tools.exp17e.mapping import Files, count, load, raster, sha, transform_points, write


def run(scope_path, census_path, mappings_path, a9_root, output, public):
    sources=load(scope_path)["sources"];polygons=load(census_path)["polygons"]
    mappings={r["source_id"]:r for r in load(mappings_path)["sources"]}
    entries={e["record"]["review_id"]:e for e in load(a9_root/"audits/verified_evidence_reconciled.json")["reviews"]}
    files=Files();results=[]
    for source in sources:
        kind=source["asset_type"]
        if kind not in {"BOUNDED","CROP"}:
            continue
        sid=source["source_id"];entry=entries[source["review_id"]];mapping=mappings[sid]
        assert mapping["mapping_verified"]
        normal=files.image({"path":source["selected_asset_path"],"sha256":source["selected_asset_sha256"]})
        aligned_desc=entry["audit"]["assets"]["aligned.png"];aligned=files.image(aligned_desc)
        human=entry["human"]
        for poly in (p for p in polygons if p["source_id"]==sid):
            pid=poly["polygon_id"];a_mask=raster(source["source_size"],poly["coordinates"])
            if kind=="CROP":
                box=source["specialty"]["crop_box_xyxy"]
                parent_view=aligned.crop(tuple(box));mask=a_mask.crop(tuple(box))
                nmask=raster(source["n_size"],transform_points(poly["coordinates"],mapping["source_to_n"]))
                assert mask.tobytes()==nmask.tobytes(),"Integer crop target raster correspondence differs"
                assert count(mask)==count(a_mask)>0,"Crop target not fully retained"
            else:
                parent_view=aligned;mask=a_mask
            difference=ImageChops.difference(normal,parent_view).convert("L")
            nonzero=difference.point(lambda x:255 if x else 0)
            mismatched=count(ImageChops.multiply(nonzero,mask))
            # RGB luma conversion can round a single-channel delta to zero, so use
            # an exact RGB pixel tuple comparison on the selected target instead.
            flags=mask.tobytes();a_bytes=normal.tobytes();b_bytes=parent_view.tobytes()
            mismatched=sum(a_bytes[i*3:i*3+3]!=b_bytes[i*3:i*3+3] for i,v in enumerate(flags) if v)
            assert mismatched==0,"Pixel-preserving region lineage does not hold"
            explicit=entry.get("explicit",{})
            results.append({"source_id":sid,"polygon_id":pid,"group_id":source["group_id"],"asset_type":kind,
                "review_id":source["review_id"],"variant_id":source["variant_id"],"a10_n_sha256":source["selected_asset_sha256"],
                "reviewed_aligned_parent":aligned_desc,"source_polygon_pixels":count(a_mask),"n_target_pixels":count(mask),
                "target_rgb_pixels_equal_parent":True,"different_target_rgb_pixels":mismatched,
                "historical_human_decision":human["decision"],"historical_b_region_review":human["b_region_review"],
                "historical_same_polygon_state":entry.get("resolved",{}).get(pid),
                "historical_polygon_basis":"EXPLICIT" if pid in explicit else "OVERALL_DEFAULT",
                "scope":"EXACT_POLYGON_PIXELS_ONLY_NOT_FULLFRAME_CONTEXT_EQUIVALENCE",
                "matrix_identity_sha256":mapping["matrix_identity_sha256"]})
    files.postcheck()
    bounded=[r for r in results if r["asset_type"]=="BOUNDED"];crops=[r for r in results if r["asset_type"]=="CROP"]
    write(output/"bounded_region_lineage.json",{"schema":"exp17e_region_pixel_lineage_v1","records":bounded,"formal_gate_pass":False})
    write(output/"crop_region_lineage.json",{"schema":"exp17e_region_pixel_lineage_v1","records":crops,"formal_gate_pass":False})
    summary={"status":"PASS_PIXEL_PRESERVING_REGION_LINEAGE_NOT_SEMANTIC_OR_DATA_GATE",
       "bounded_sources":len({r["source_id"] for r in bounded}),"bounded_polygons":len(bounded),
       "bounded_target_pixels":sum(r["source_polygon_pixels"] for r in bounded),
       "crop_sources":len({r["source_id"] for r in crops}),"crop_polygons":len(crops),
       "all_target_rgb_pixels_equal_reviewed_parent":all(r["target_rgb_pixels_equal_parent"] for r in results),
       "region_historical_pass_and_b_confirmed":sum(r["historical_same_polygon_state"]=="PASS" and r["historical_b_region_review"]=="CONFIRMED" for r in results),
       "source_sha256":sha(__file__),"bounded_private_sha256":sha(output/"bounded_region_lineage.json"),
       "crop_private_sha256":sha(output/"crop_region_lineage.json"),"verified_files":len(files.verified),
       "new_image_files":0,"registration":0,"model_forwards":0,"optimizer_updates":0,"formal_trusted_pairs":0,
       "limitations":["Pixel equivalence is limited to each exact target support; larger feature receptive fields/context can differ.",
                       "Preserves the selected A10 asset; no substitution of Aligned for Bounded or Crop."]}
    write(output/"region_lineage_summary.json",summary);write(public,summary)
    print(summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("scope","census","mappings","a9","output","public"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.scope,a.census,a.mappings,a.a9,a.output,a.public)
