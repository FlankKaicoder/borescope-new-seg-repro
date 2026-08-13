#!/usr/bin/env python3
"""Create FastTrack-A comparisons, figure links, and decoded artifact manifest."""
from __future__ import annotations
import argparse,csv,json,os,shutil
from pathlib import Path
import cv2,matplotlib.pyplot as plt,numpy as np
from fasttrack_common import dump,rows,write_csv

BASE={"Mask_P":.66292,"Mask_R":.52302,"Mask_mAP50":.53192,"Mask_mAP50_95":.29898123412927496,"Crack_R":.27883,"Crack_AP50":.17325,"Crack_AP50_95":.04675,"Burn_R":.25926,"Burn_AP50_95":.11245,"corrosion_R":.32110,"corrosion_AP50_95":.19783,"FP":99,"FN":173}
def metrics(root:Path):
 o=json.loads((root/"overall_metrics.json").read_text());pc=rows(root/"per_class_metrics.csv")
 by={x["class_name"]:x for x in pc};return {"Mask_P":o["metrics/precision(M)"],"Mask_R":o["metrics/recall(M)"],"Mask_mAP50":o["metrics/mAP50(M)"],"Mask_mAP50_95":o["metrics/mAP50-95(M)"],"classes":by}
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();r=a.root;fig=r/"figures";fig.mkdir(exist_ok=True)
 exp03=json.loads((r/"exp03_low_conf/summary.json").read_text());e4=metrics(r/"exp04_crack_oneclass/eval");c=metrics(r/"exp05_hard_mining/control_eval");t=metrics(r/"exp05_hard_mining/treatment_eval")
 crack=e4["classes"]["Crack"];crackvals={"R":float(crack["mask_recall"]),"AP50":float(crack["mask_ap50"]),"AP95":float(crack["mask_ap50_95"])}
 plt.figure(figsize=(7,4));x=np.arange(3);plt.bar(x-.18,[BASE["Crack_R"],BASE["Crack_AP50"],BASE["Crack_AP50_95"]],.36,label="multiclass");plt.bar(x+.18,[crackvals["R"],crackvals["AP50"],crackvals["AP95"]],.36,label="oneclass");plt.xticks(x,["Recall","AP50","AP50-95"]);plt.legend();plt.tight_layout();plt.savefig(r/"exp04_crack_oneclass/exp04_crack_multiclass_vs_oneclass.png",dpi=160);plt.close()
 hard=json.loads((r/"exp05_hard_mining/hard_pool/summary.json").read_text());cs=json.loads((r/"exp05_hard_mining/control/summary.json").read_text());ts=json.loads((r/"exp05_hard_mining/treatment/summary.json").read_text())
 if cs["total_sampled_images"]!=ts["total_sampled_images"] or cs["optimizer_steps"]!=ts["optimizer_steps"]:raise RuntimeError("EXP05_FAIRNESS_HARD_GATE")
 keys=["Mask_mAP50_95","Mask_R","Mask_P"]
 plt.figure(figsize=(7,4));x=np.arange(len(keys));plt.bar(x-.18,[c[k] for k in keys],.36,label="Control");plt.bar(x+.18,[t[k] for k in keys],.36,label="Hard");plt.xticks(x,keys);plt.legend();plt.tight_layout();plt.savefig(r/"exp05_hard_mining/exp05_control_vs_hard_mining_metrics.png",dpi=160);plt.close()
 def fixed_errors(evaldir):
  # fixed conf=.25 class-aware mask matching via the same Exp03 evaluator is saved separately by wrapper
  return json.loads((evaldir/"fixed_errors.json").read_text())
 ce=fixed_errors(r/"exp05_hard_mining/control_eval");te=fixed_errors(r/"exp05_hard_mining/treatment_eval")
 plt.figure(figsize=(6,4));x=np.arange(2);plt.bar(x-.18,[ce["FP"],ce["FN"]],.36,label="Control");plt.bar(x+.18,[te["FP"],te["FN"]],.36,label="Hard");plt.xticks(x,["FP","FN"]);plt.legend();plt.tight_layout();plt.savefig(r/"exp05_hard_mining/exp05_control_vs_hard_mining_errors.png",dpi=160);plt.close()
 diff=t["Mask_mAP50_95"]-c["Mask_mAP50_95"];conclusion="POSITIVE_CANDIDATE" if diff>=.005 and t["Mask_R"]>=c["Mask_R"]-.02 else "NEGATIVE" if diff<=-.005 else "NO_CLEAR_GAIN"
 summary=[]
 def row(method,typ,budget,m,conc,fp="N/A",fn="N/A",one=False):
  cl=m.get("classes",{});get=lambda n,k:float(cl[n][k]) if n in cl else "N/A"
  return {"Method":method,"Type":typ,"Training_budget":budget,**{k:m.get(k,"N/A") for k in ("Mask_P","Mask_R","Mask_mAP50","Mask_mAP50_95")},"Crack_R":m.get("Crack_R",get("Crack","mask_recall")),"Crack_AP50_95":m.get("Crack_AP50_95",get("Crack","mask_ap50_95")),"Burn_R":m.get("Burn_R",get("Burn","mask_recall")),"Burn_AP50_95":m.get("Burn_AP50_95",get("Burn","mask_ap50_95")),"corrosion_R":m.get("corrosion_R",get("corrosion","mask_recall")),"corrosion_AP50_95":m.get("corrosion_AP50_95",get("corrosion","mask_ap50_95")),"FP":fp,"FN":fn,"Conclusion":conc}
 summary.append(row("Exp02 Baseline","model","100 epochs",BASE,"PASS_WITH_NUMERICAL_WAIVER",99,173));summary.append(row("Exp03 F1-best","operating_point","N/A",{"Mask_P":exp03["fixed_point"]["Precision"],"Mask_R":exp03["fixed_point"]["Recall"]},exp03["conclusion"],exp03["fixed_point"]["FP"],exp03["fixed_point"]["FN"]));rm=exp03["recall95_metrics"];summary.append(row("Exp03 Recall95","operating_point","N/A",{"Mask_P":rm["Precision"],"Mask_R":rm["Recall"]},exp03["conclusion"],rm["FP"],rm["FN"]));summary.append(row("Exp04 Crack-only","oneclass_model","100 epochs",{"Crack_R":crackvals["R"],"Crack_AP50_95":crackvals["AP95"]},"POSITIVE" if crackvals["AP95"]>BASE["Crack_AP50_95"]+.005 and crackvals["R"]>=BASE["Crack_R"] else "NEGATIVE" if crackvals["AP95"]<BASE["Crack_AP50_95"]-.005 else "NO_CLEAR_GAIN"));summary.append(row("Exp05 Control","model","30 epochs / 668 samples per epoch",c,"CONTROL",ce["FP"],ce["FN"]));summary.append(row("Exp05 Hard Mining","model","30 epochs / 668 samples per epoch",t,conclusion,te["FP"],te["FN"]));write_csv(r/"fasttrack_a_summary.csv",summary)
 plt.figure(figsize=(8,4));labels=["Baseline","Control","Hard"];plt.bar(labels,[BASE["Mask_mAP50_95"],c["Mask_mAP50_95"],t["Mask_mAP50_95"]]);plt.ylabel("Mask mAP50-95");plt.tight_layout();plt.savefig(r/"fasttrack_a_main_comparison.png",dpi=160);plt.close()
 sources={"exp02_baseline":Path("results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline"),"exp03_low_conf":r/"exp03_low_conf","exp04_crack_oneclass":r/"exp04_crack_oneclass","exp05_hard_mining/control":r/"exp05_hard_mining/control","exp05_hard_mining/treatment":r/"exp05_hard_mining/treatment","exp05_hard_mining/comparison":r/"exp05_hard_mining"}
 for name,src in sources.items():
  d=fig/name;d.mkdir(exist_ok=True,parents=True)
  for q in src.rglob("*"):
   if q.is_file() and q.suffix.lower() in (".png",".jpg",".jpeg"):
    dest=d/q.name
    if not dest.exists():os.symlink(q.resolve(),dest)
 arts=[]
 for q in sorted(fig.rglob("*")):
  if q.is_file():im=cv2.imread(str(q));arts.append({"experiment":q.parent.name,"artifact_type":q.suffix.lower().lstrip("."),"file_path":str(q),"file_size":q.stat().st_size,"decode_pass":im is not None})
 write_csv(r/"artifact_manifest.csv",arts);dump(r/"finalize_summary.json",{"status":"PASS" if all(x["decode_pass"] for x in arts) else "FAIL","test_accessed":False,"artifact_count":len(arts),"all_decode_pass":all(x["decode_pass"] for x in arts),"exp04_conclusion":summary[3]["Conclusion"],"exp05_conclusion":conclusion,"exp05_fair":True,"hard_pool":hard})
if __name__=="__main__":main()
