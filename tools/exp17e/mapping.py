"""CPU-only reuse of frozen A10/A9 mappings; never estimates a transform."""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class Files:
    def __init__(self):
        self.verified = {}

    def check(self, descriptor):
        path = Path(descriptor["path"])
        expected = descriptor["sha256"]
        if str(path) not in self.verified:
            require(path.is_file() and sha(path) == expected, "Missing or changed frozen mapping input")
            self.verified[str(path)] = {"path":str(path), "sha256":expected, "bytes":path.stat().st_size}
        require(self.verified[str(path)]["sha256"] == expected, "One file has conflicting identities")
        return path

    def image(self, descriptor, mode="RGB"):
        path = self.check(descriptor)
        with Image.open(path) as im:
            return im.convert(mode)

    def postcheck(self):
        require(all(Path(p).stat().st_size == d["bytes"] and sha(p) == d["sha256"]
                    for p,d in self.verified.items()), "Mapping input changed during audit")


def affine(matrix):
    require(isinstance(matrix, list) and len(matrix) in (2,3), "No historical affine matrix")
    result = [[float(x) for x in row] for row in matrix]
    require(all(len(row)==3 and all(math.isfinite(v) for v in row) for row in result), "Malformed historical matrix")
    if len(result)==2:
        result.append([0.,0.,1.])
    require(result[2] == [0.,0.,1.], "Only audited historical affine matrices are supported")
    require(result[0][0]*result[1][1]-result[0][1]*result[1][0] > 1e-12, "Singular or reflected historical affine")
    return result


def inverse(matrix):
    h=affine(matrix); a,b,c=h[0];d,e,f=h[1];det=a*e-b*d
    return [[e/det,-b/det,(b*f-e*c)/det],[-d/det,a/det,(d*c-a*f)/det],[0.,0.,1.]]


def transform_points(points, matrix):
    h=affine(matrix)
    return [[h[0][0]*x+h[0][1]*y+h[0][2],h[1][0]*x+h[1][1]*y+h[1][2]] for x,y in points]


def raster(size, points):
    mask=Image.new("L",tuple(size),0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in points],fill=255)
    return mask


def count(mask):
    return mask.histogram()[255]


def binary(mask):
    return mask.convert("L").point(lambda v:255 if v else 0)


def warp_valid_nearest(source_valid, n_size, n_to_source):
    """Existing affine only; account for Pillow's half-pixel corner convention.

    Integer pixel centers obey the stored OpenCV-style H. The adjustment maps
    Pillow output centers back to the same source centers; no bilinear validity.
    """
    h=affine(n_to_source);a,b,c=h[0];d,e,f=h[1]
    coefficients=(a,b,c+.5-.5*a-.5*b,d,e,f+.5-.5*d-.5*e)
    return binary(source_valid.transform(tuple(n_size),Image.Transform.AFFINE,coefficients,
                                         resample=Image.Resampling.NEAREST,fillcolor=0))


def roi(points, size):
    xs,ys=zip(*points);w,h=size
    margin=max(12., .25*max(max(xs)-min(xs),max(ys)-min(ys)))
    return [max(0,math.floor(min(xs)-margin)),max(0,math.floor(min(ys)-margin)),
            min(w,math.ceil(max(xs)+margin+1)),min(h,math.ceil(max(ys)+margin+1))]


def prove_bounded(files, row, entry, source_size):
    special=row["specialty"]
    audit,record=entry["audit"],entry["record"]
    require(special["all_gt_accounted"] and special["preserves_base_outside_effective_edit"], "Missing frozen Bounded accounting")
    require(special["bounded"]["sha256"]==row["selected_asset_sha256"]==audit["assets"]["bounded.png"]["sha256"],"Bounded version differs")
    request_path=files.check({"path":audit["origin_paths"]["request_path"],"sha256":record["request_sha256"]})
    request=load(request_path)
    require(request["source_image_sha256"]==special["base"]["sha256"],"Bounded base request identity differs")
    base=files.image(special["base"]);aligned=files.image(audit["assets"]["aligned.png"])
    normal=files.image({"path":row["selected_asset_path"],"sha256":row["selected_asset_sha256"]})
    valid=binary(files.image(special["valid_mask"],"L"))
    edit=files.image(special["edit_mask"],"RGBA").getchannel("A").point(lambda x:255 if x==0 else 0)
    require(base.size==aligned.size==normal.size==valid.size==edit.size==tuple(source_size),"Bounded source grid differs")
    effective=ImageChops.multiply(edit,valid)
    expected=Image.composite(aligned,base,effective)
    require(ImageChops.difference(normal,expected).getbbox() is None,"Bounded bytes do not replay from frozen parents")
    return {"base":special["base"],"edit_mask":special["edit_mask"],"request_sha256":record["request_sha256"],
            "composite_pixel_replay":True,"effective_edit_pixels":count(effective),"all_gt_accounted_by_frozen_proof":True},effective


def source_mapping(row, entry, files, output):
    record,audit=entry["record"],entry["audit"]
    sid=row["source_id"];a_size=list(row["source_size"]);n_size=list(row["n_size"]);kind=row["asset_type"]
    n_sha=row["selected_asset_sha256"]
    require(record["source_id"]==sid and record["review_id"]==row["review_id"] and record["variant_id"]==row["variant_id"],"A10/A9 mapping version differs")
    require(record["source_sha256"]==row["source_image_sha256"] and record["annotation_sha256"]==row["annotation_sha256"],"Mapping source/JSON identity differs")
    n=files.image({"path":row["selected_asset_path"],"sha256":n_sha})
    require(list(n.size)==n_size,"Selected N dimensions differ")
    result={"source_id":sid,"group_id":row["group_id"],"asset_type":kind,"n_asset_sha256":n_sha,
            "n_path":row["selected_asset_path"],"source_size":a_size,"n_size":n_size,
            "review_id":row["review_id"],"variant_id":row["variant_id"],"repair_version":row["repair_version"],
            "mapping_status":"MAPPING_UNVERIFIED","mapping_verified":False,"source_to_n":None,"n_to_source":None,
            "matrix_identity_sha256":None,"historical_raw_to_source":None,"source_valid_mask":None,"n_valid_mask":None,
            "geometry_provenance":None,"failure_reasons":[],"evidence":{}}
    if not audit.get("mapping_ok") or not audit.get("matrix"):
        result["failure_reasons"]=["NO_RELIABLE_FROZEN_HISTORICAL_MAPPING"]
        return result,None,None
    require(record["geometry_status"]=="GEOMETRY_CANDIDATE", "Saved mapping was not an accepted historical geometric candidate")
    raw_to_source=affine(audit["matrix"])
    valid_desc=audit["assets"]["valid.png"];valid=binary(files.image(valid_desc,"L"))
    require(list(valid.size)==a_size,"Frozen Valid Mask is not on original grid")
    geometry=audit["assets"].get("geometry.json")
    if geometry:
        files.check(geometry)
    result.update(historical_raw_to_source=raw_to_source,source_valid_mask=valid_desc,
                  geometry_provenance="FROZEN_GEOMETRY_FILE_AND_AUDITED_MATRIX" if geometry else "LEGACY_AUDITED_RECORD_MATRIX_AND_ASSET_HASHES")
    result["evidence"]={"historical_raw_sha256":record["raw_sha256"],"historical_aligned_sha256":record["aligned_sha256"],
                        "geometry_file":geometry,"historical_mapping_method":record["mapping_method"]}
    effective=None
    if kind=="ALIGNED":
        require(n_sha==record["aligned_sha256"]==audit["assets"]["aligned.png"]["sha256"] and a_size==n_size,"Aligned is not frozen source grid")
        h=IDENTITY;nv=valid
        result["evidence"]["already_on_source_grid_do_not_apply_raw_transform_again"]=True
    elif kind=="RAW":
        require(n_sha==record["raw_sha256"] and audit["assets"][record["raw"]]["sha256"]==n_sha,"Raw mapping concerns other generated bytes")
        h=inverse(raw_to_source);nv=warp_valid_nearest(valid,n_size,raw_to_source)
    elif kind=="CROP":
        special=row["specialty"];parent=special["parent"]
        require(parent["sha256"]==record["aligned_sha256"]==audit["assets"]["aligned.png"]["sha256"],"Crop parent version differs")
        parent_image=files.image(parent)
        box=special["crop_box_xyxy"]
        require(len(box)==4 and all(type(x) is int for x in box),"Crop bounds must be exact integers")
        x0,y0,x1,y1=box
        require(list(parent_image.size)==a_size and 0<=x0<x1<=a_size[0] and 0<=y0<y1<=a_size[1]
                and special["parent_to_crop_xy"]==[-x0,-y0] and n_size==[x1-x0,y1-y0],"Crop translation/frame differs")
        require(n.tobytes()==parent_image.crop(tuple(box)).tobytes(),"Crop bytes differ from frozen parent bounds")
        h=[[1.,0.,-x0],[0.,1.,-y0],[0.,0.,1.]];nv=valid.crop(tuple(box))
        result["evidence"].update(parent=parent,crop_box_xyxy=box,parent_to_crop_xy=special["parent_to_crop_xy"],crop_pixels_equal_parent=True)
    elif kind=="BOUNDED":
        result["evidence"]["bounded"],effective=prove_bounded(files,row,entry,a_size)
        h=IDENTITY;nv=valid
    else:
        raise ValueError("Unknown frozen N asset type")
    target=output/"valid_masks"/f"{sid}_n_valid.png";nv.save(target)
    nv_desc={"path":str(target.resolve()),"sha256":sha(target),"bytes":target.stat().st_size,
             "derivative":"NEW_DISPLAY_AND_CPU_SUPPORT_ONLY_NEAREST_NO_HISTORICAL_ASSET_MODIFIED"}
    result.update(mapping_status="VERIFIED_HISTORICAL_MAPPING",mapping_verified=True,source_to_n=h,n_to_source=inverse(h),n_valid_mask=nv_desc)
    result["matrix_identity_basis"]="NEW_BINDING_DIGEST_OF_SAVED_MATRIX_AND_EXACT_SOURCE_A10_N_VALID_ASSETS_NOT_A_NEW_FIT"
    result["matrix_identity_sha256"]=digest({"source_sha256":record["source_sha256"],"n_sha256":n_sha,
         "review_id":record["review_id"],"raw_to_source":raw_to_source,"source_to_n":h,"valid_sha256":valid_desc["sha256"],"geometry":geometry})
    return result,valid,effective


def polygon_mapping(poly, source, mapping, valid, effective=None):
    points=poly["coordinates"];size=source["source_size"]
    am=raster(size,points);total=count(am)
    result={"source_id":poly["source_id"],"group_id":poly["group_id"],"polygon_id":poly["polygon_id"],
            "class_id":poly["class_id"],"class_name":poly["class_name"],"n_asset_sha256":mapping["n_asset_sha256"],
            "mapping_verified":mapping["mapping_verified"],"matrix_identity_sha256":mapping["matrix_identity_sha256"],
            "source_roi_xyxy":roi(points,size),"source_polygon_pixels":total,"source_valid_target_pixels":None,
            "n_coordinates":None,"n_roi_xyxy":None,"n_polygon_pixels":None,"n_valid_target_pixels":None,
            "source_support_complete":False,"n_support_complete":False,"support_complete":False,"failure_reasons":list(mapping["failure_reasons"])}
    if not mapping["mapping_verified"]:
        return result
    mapped=transform_points(points,mapping["source_to_n"]);n_size=source["n_size"]
    full_bounds=all(0<=x<n_size[0] and 0<=y<n_size[1] for x,y in mapped)
    nm=raster(n_size,mapped)
    with Image.open(mapping["n_valid_mask"]["path"]) as im:nv=binary(im)
    valid_source=count(ImageChops.multiply(am,valid));n_total=count(nm);n_valid=count(ImageChops.multiply(nm,nv))
    source_ok=total>0 and valid_source==total
    n_ok=n_total>0 and n_valid==n_total and full_bounds
    result.update(n_coordinates=mapped,n_roi_xyxy=roi(mapped,n_size),source_valid_target_pixels=valid_source,
                  n_polygon_pixels=n_total,n_valid_target_pixels=n_valid,source_support_complete=source_ok,
                  n_support_complete=n_ok,support_complete=source_ok and n_ok,n_vertices_inside_frame=full_bounds)
    if not source_ok:result["failure_reasons"].append("TARGET_SOURCE_VALID_SUPPORT_INCOMPLETE")
    if not full_bounds:result["failure_reasons"].append("TARGET_NOT_FULLY_INSIDE_FROZEN_N_FRAME")
    if n_valid!=n_total or n_total==0:result["failure_reasons"].append("TARGET_NATIVE_N_NEAREST_VALID_SUPPORT_INCOMPLETE")
    if effective is not None:
        edited=count(ImageChops.multiply(am,effective));result["bounded_effective_edit_target_pixels"]=edited
        result["bounded_target_edit_provenance_status"]=("FULLY_IN_EFFECTIVE_EDIT" if edited==total
            else "OUTSIDE_CURRENT_EDIT_REQUIRES_PRIOR_BASE_PROVENANCE_OR_SEMANTIC_SCOPE")
    return result


def run(scope_path, polygons_path, a9, output, public):
    started=time.monotonic();sources=load(scope_path)["sources"];polygons=load(polygons_path)["polygons"]
    require(len(sources)==len({s["source_id"] for s in sources})==379,"Expected unique A10 S379")
    require(Counter(s["asset_type"] for s in sources)==Counter(ALIGNED=346,RAW=29,BOUNDED=1,CROP=3),"Frozen asset-type identity changed")
    require(not output.exists(),"New mapping output required")
    output.mkdir(parents=True);(output/"valid_masks").mkdir()
    evidence_path=a9/"audits/verified_evidence_reconciled.json";evidence=load(evidence_path)
    entries={e["record"]["review_id"]:e for e in evidence["reviews"]};files=Files();mapped=[];mapped_polygons=[]
    grouped={s["source_id"]:[p for p in polygons if p["source_id"]==s["source_id"]] for s in sources}
    for source in sources:
        row,valid,effective=source_mapping(source,entries[source["review_id"]],files,output)
        mapped.append(row)
        mapped_polygons.extend(polygon_mapping(p,source,row,valid,effective) for p in grouped[source["source_id"]])
    files.postcheck()
    write(output/"source_mappings.json",{"schema":"exp17e_frozen_source_mappings_v1","sources":mapped})
    write(output/"polygon_mappings.json",{"schema":"exp17e_frozen_polygon_mappings_v1","polygons":mapped_polygons})
    exceptions=[p for p in mapped_polygons if not p["support_complete"]]
    with (output/"asset_mapping_exceptions.csv").open("x",encoding="utf-8-sig",newline="") as f:
        fields=["source_id","group_id","polygon_id","class_name","n_asset_sha256","mapping_verified","source_valid_target_pixels","source_polygon_pixels","n_valid_target_pixels","n_polygon_pixels","failure_reasons"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(p[k],ensure_ascii=False) if isinstance(p[k],list) else p[k] for k in fields} for p in exceptions)
    write(output/"verified_mapping_inputs.json",{"files":files.verified,"postchecked":True})
    summary={"status":"FROZEN_MAPPING_AUDIT_COMPLETE_NOT_DATA_GATE","source_assets":len(mapped),"polygons":len(mapped_polygons),
       "mapping_verified_sources":sum(m["mapping_verified"] for m in mapped),"mapping_unverified_sources":sum(not m["mapping_verified"] for m in mapped),
       "mapping_verified_polygons":sum(p["mapping_verified"] for p in mapped_polygons),"full_target_support_polygons":sum(p["support_complete"] for p in mapped_polygons),
       "exception_polygons":len(exceptions),"exceptions_by_reason":dict(Counter(r for p in exceptions for r in p["failure_reasons"])),
       "asset_type_counts":dict(Counter(s["asset_type"] for s in sources)),
       "by_asset_type":{kind:{"sources":sum(m["asset_type"]==kind for m in mapped),"mapping_verified_sources":sum(m["asset_type"]==kind and m["mapping_verified"] for m in mapped),
          "polygons":sum(p["source_id"] in {s["source_id"] for s in sources if s["asset_type"]==kind} for p in mapped_polygons),
          "full_support_polygons":sum(p["support_complete"] and p["source_id"] in {s["source_id"] for s in sources if s["asset_type"]==kind} for p in mapped_polygons)} for kind in ("ALIGNED","RAW","BOUNDED","CROP")},
       "verified_unique_input_files":len(files.verified),"verified_input_bytes":sum(x["bytes"] for x in files.verified.values()),"elapsed_seconds":time.monotonic()-started,
       "scope_sha256":sha(scope_path),"polygon_census_sha256":sha(polygons_path),"a9_saved_evidence_sha256":sha(evidence_path),
       "source_sha256":sha(__file__),"private_source_mappings_sha256":sha(output/"source_mappings.json"),"private_polygon_mappings_sha256":sha(output/"polygon_mappings.json"),
       "new_registration":0,"model_forward":0,"optimizer_updates":0,"formal_trusted_pairs":0,
       "limitations":["Historical affine reuse is region correspondence, not Formal TRE or new registration.","Raw uses inverse frozen Raw-to-source transform, never same-size identity.","Aligned is already source-grid; Raw transform is not applied twice.","Native N support conservatively checks mapped polygon raster and nearest validity; it does not prove normal semantics."]}
    write(output/"mapping_summary.json",summary)
    public.parent.mkdir(parents=True,exist_ok=True);write(public,summary)
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("scope","polygons","a9","output","public"):
        p.add_argument("--"+arg,type=Path,required=True)
    args=p.parse_args();run(args.scope,args.polygons,args.a9,args.output,args.public)
