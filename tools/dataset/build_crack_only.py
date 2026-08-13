#!/usr/bin/env python3
"""Derive a Crack-only train/val dataset without touching frozen Dataset v1."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
import yaml
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 for s in ("train","val"):(a.output/"images"/s).mkdir(parents=True);(a.output/"labels"/s).mkdir(parents=True)
 with a.manifest.open(encoding="utf-8-sig",newline="") as f:rs=[r for r in csv.DictReader(f) if r["split"] in ("train","val")]
 stats={s:{"images":0,"positive_images":0,"background_images":0,"crack_instances":0} for s in ("train","val")}
 for r in rs:
  s=r["split"];src=a.source/"images"/s/(r["stem"]+r["image_suffix"]);dst=a.output/"images"/s/src.name;os.symlink(src,dst)
  lines=[]
  for raw in (a.source/"labels"/s/(r["stem"]+".txt")).read_text().splitlines():
   v=raw.split()
   if int(v[0])==1:lines.append("0 "+" ".join(v[1:]))
  (a.output/"labels"/s/(r["stem"]+".txt")).write_text(("\n".join(lines)+"\n") if lines else "")
  st=stats[s];st["images"]+=1;st["crack_instances"]+=len(lines);st["positive_images"]+=bool(lines);st["background_images"]+=not bool(lines)
 (a.output/"data.yaml").write_text(yaml.safe_dump({"path":str(a.output.resolve()),"train":"images/train","val":"images/val","names":{0:"Crack"}},sort_keys=False))
 (a.output/"derivation_summary.json").write_text(json.dumps({"status":"PASS","source":str(a.source),"test_accessed":False,"test_materialized":False,"split_preserved":True,"rule":"source class 1 -> class 0; all other polygons omitted",**stats},indent=2)+"\n")
 print(json.dumps(stats,indent=2))
if __name__=="__main__":main()
