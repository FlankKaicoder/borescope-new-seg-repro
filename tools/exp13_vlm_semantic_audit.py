#!/usr/bin/env python3
import csv, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
REPO=Path(__file__).resolve().parents[1]; DATA=Path("/root/autodl-tmp/borescope-new-seg-data/v1")
TRAIN_IMAGES=DATA/"images/train"; TRAIN_LABELS=DATA/"labels/train"; OUT=REPO/"results/exp13_vlm_semantic_audit"
NAMES={0:"Burn",1:"Crack",2:"Dent",3:"Material missing",4:"Tears",5:"Tip curl",6:"corrosion"}; TARGETS=[(1,"Crack"),(6,"corrosion")]
def key(p):
    try:return(0,int(p.stem))
    except ValueError:return(1,p.stem)
def img(s):
    x=sorted(TRAIN_IMAGES.glob(s+".*"))
    if len(x)!=1:raise RuntimeError("missing TRAIN image "+s)
    return x[0]
def parse(p):
    z=[]
    for n,r in enumerate(p.read_text().splitlines(),1):
        f=r.split()
        if not f:continue
        if len(f)<7 or (len(f)-1)%2:raise RuntimeError("bad polygon")
        c=int(f[0]); q=[float(x) for x in f[1:]]
        z.append({"class_id":c,"class":NAMES[c],"points":[[q[i],q[i+1]] for i in range(0,len(q),2)]})
    return z
def main():
    ps=sorted(TRAIN_LABELS.glob("*.txt"),key=key)
    if len(ps)!=668:raise RuntimeError("expected 668 TRAIN labels")
    rec={p.stem:parse(p) for p in ps}
    for s in rec:img(s)
    chosen=[]; used=set()
    for cid,cname in TARGETS:
        cand=[]
        for s,a in rec.items():
            ids={x["class_id"] for x in a}
            if cid in ids and s not in used:cand.append((ids!={cid},key(Path(s)),s))
        cand.sort()
        for s in [x[2] for x in cand[:20]]:
            used.add(s); chosen.append({"source_id":s,"class":cname,"class_id":cid,"split":"train","instance_count":sum(x["class_id"]==cid for x in rec[s]),"source_path":str(img(s)),"annotations":rec[s]})
    if len(chosen)!=40 or len(used)!=40:raise RuntimeError("not 40 unique")
    for n in ["samples","overlays","contact_sheets","metadata","prompts"]:(OUT/n).mkdir(parents=True,exist_ok=True)
    rows=[]; ctx=[]; groups=defaultdict(list)
    for i,it in enumerate(chosen,1):
        src=Path(it["source_path"]); sample=OUT/"samples"/("vlm_%02d_%s%s"%(i,it["source_id"],src.suffix.lower())); shutil.copyfile(src,sample)
        ov=OUT/"overlays"/("vlm_%02d_%s_overlay.png"%(i,it["source_id"])); base=Image.open(src).convert("RGBA"); lay=Image.new("RGBA",base.size,(0,0,0,0)); d=ImageDraw.Draw(lay)
        for a in it["annotations"]:
            pts=[(round(x*base.width),round(y*base.height)) for x,y in a["points"]]; hit=a["class_id"]==it["class_id"]
            d.polygon(pts,fill=(40,220,90,90) if hit else (255,170,0,55),outline=(0,255,70,255) if hit else (255,150,0,220))
        Image.alpha_composite(base,lay).convert("RGB").save(ov,quality=95)
        it["sample_path"]=str(sample);it["overlay_path"]=str(ov);ctx.append({k:v for k,v in it.items() if k!="class_id"});groups[it["class"]].append(it)
        rows.append({"image_path":it["source_path"],"source_id":it["source_id"],"class":it["class"],"split":"train","instance_count":it["instance_count"]})
    with (OUT/"sample_manifest.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=["image_path","source_id","class","split","instance_count"]);w.writeheader();w.writerows(rows)
    (OUT/"metadata/polygon_context.json").write_text(json.dumps(ctx,indent=2)+"\n")
    audit={"experiment":"Exp13.0","scope":"VLM semantic audit preparation only","data_root":str(DATA),"split_accessed":["train"],"val_accessed":False,"test_accessed":False,"synthetic_images_generated":False,"gpu_or_model_called":False,"total_images":40,"unique_source_ids":40,"class_counts":dict(Counter(x["class"] for x in chosen)),"train_label_files_verified":len(rec)}
    (OUT/"metadata/preparation_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    fields=["class","shape","boundary","texture","color","scale","orientation","surface","lighting","industrial_context","generation_hint","confidence"]
    (OUT/"metadata/vlm_schema.json").write_text(json.dumps({"type":"object","additionalProperties":False,"required":fields,"properties":{x:{"type":"string"} for x in fields}},indent=2)+"\n")
    (OUT/"prompts/vlm_prompt_template.md").write_text("# Exp13.0 VLM semantic-audit prompt template\n\nInput: one real industrial borescope TRAIN image, polygon/mask overlay, and known class. Use only visible evidence; do not invent causes, materials, depth, measurements, or unobserved structure. Describe geometry, boundary, texture, color, scale, orientation, surface, lighting, and industrial context. Return exactly one JSON object matching metadata/vlm_schema.json with the twelve fixed string fields and no free text outside. generation_hint is future prompt-building only; do not generate images or invoke diffusion. Record uncertainty in confidence.\n")
    for cname,items in groups.items():
        tw,th=320,270; sheet=Image.new("RGB",(tw*4,th*5),"white");d=ImageDraw.Draw(sheet)
        for i,it in enumerate(items):
            im=Image.open(it["sample_path"]).convert("RGB");im.thumbnail((tw-12,th-42));x=(i%4)*tw+(tw-im.width)//2;y=(i//4)*th+6;sheet.paste(im,(x,y));d.text(((i%4)*tw+6,(i//4)*th+th-28),"%s | %s | n=%d"%(it["source_id"],it["class"],it["instance_count"]),fill="black")
        sheet.save(OUT/"contact_sheets"/("contact_sheet_%s.png"%cname.lower()))
if __name__=="__main__":main()
