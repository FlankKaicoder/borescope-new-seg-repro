#!/usr/bin/env python3
"""Re-audit original Exp02.1 epochs 1-10 directly from results.csv."""
import argparse, csv, json, math
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("--results-csv",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
mapping={"epoch":"epoch","train/box_loss":"train_box_loss","train/seg_loss":"train_seg_loss","train/cls_loss":"train_cls_loss","train/dfl_loss":"train_dfl_loss","val/box_loss":"val_box_loss","val/seg_loss":"val_seg_loss","val/cls_loss":"val_cls_loss","val/dfl_loss":"val_dfl_loss","metrics/mAP50(B)":"box_mAP50","metrics/mAP50-95(B)":"box_mAP50_95","metrics/mAP50(M)":"mask_mAP50","metrics/mAP50-95(M)":"mask_mAP50_95"}
with a.results_csv.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))[:10]
out=[]
for row in rows:
    clean={k.strip():v.strip() for k,v in row.items()}; out.append({dst:clean[src] for src,dst in mapping.items()})
with (a.output/"early_epoch_loss_audit.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
loss_fields=[x for x in mapping.values() if x.startswith(("train_","val_"))]
nan_rows=[]
for row in out:
    bad=[k for k in loss_fields if not math.isfinite(float(row[k]))]
    if bad:nan_rows.append({"epoch":int(row["epoch"]),"fields":bad})
metrics_finite=all(math.isfinite(float(r[k])) for r in out for k in ("box_mAP50","box_mAP50_95","mask_mAP50","mask_mAP50_95"))
train_finite=all(math.isfinite(float(r[k])) for r in out for k in ("train_box_loss","train_seg_loss","train_cls_loss","train_dfl_loss"))
summary={"source":str(a.results_csv),"audited_epochs":"1-10","nonfinite":nan_rows,"first_recovered_epoch":next(int(r["epoch"]) for r in out if all(math.isfinite(float(r[k])) for k in ("val_box_loss","val_seg_loss","val_cls_loss","val_dfl_loss"))),"metrics_all_finite":metrics_finite,"train_losses_all_finite":train_finite}
(a.output/"early_epoch_loss_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");print(json.dumps(summary,indent=2))
