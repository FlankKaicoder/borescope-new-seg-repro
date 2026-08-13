#!/usr/bin/env python3
from pathlib import Path
import csv,json,cv2

R=Path('/root/autodl-tmp/borescope-new-seg-repro')
F=R/'results/fast_repro'
def J(p):return json.loads(Path(p).read_text())
def writecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ds=J(F/'exp06_roi/dataset_summary.json');ce=J(F/'exp06_roi/ce/metrics.json');su=J(F/'exp06_roi/supcon/metrics.json');cmp=J(F/'exp06_roi/ce_vs_supcon.json');st=J(F/'exp07_stage2/summary.json');stce=J(F/'exp07_stage2_ce/summary.json')
pc=lambda m:'\n'.join(f"| {x['class_name']} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} | {x['support']} |" for x in m['per_class'])
counts=ds['counts']; cls=[]
for n in ['Burn','Crack','Dent','Material missing','Tears','Tip curl','corrosion','background']:
 cls.append((n,sum(v for k,v in counts.items() if k.startswith('train|'+n+'|')),sum(v for k,v in counts.items() if k.startswith('val|'+n+'|'))))
clines='\n'.join(f'| {n} | {a} | {b} |' for n,a,b in cls)
(R/'docs/exp06_roi_resnet_fast_repro.md').write_text(f'''# Exp06 ROI ResNet fast repro

状态：**COMPLETE / VAL ONLY / test_accessed=false**

统一 patch manifest：`results/fast_repro/exp06_roi/patch_manifest.csv`，SHA256 `{ce['manifest_sha256']}`。GT bbox 固定 1.2× crop，224×224、ImageNet normalization。CE 与 SupCon 使用同一 manifest、batch=64、WeightedRandomSampler、seed=42。

| Class | Train | Val |
|---|---:|---:|
{clines}

总计 train={ds['train_count']}、val={ds['val_count']}。background：train random/hard-FP=633/633，val=148/148；train/val source-image leakage=0；未生成或读取 test patch。

CE ResNet18 ImageNet 完成 50/50 epoch，loss finite，best checkpoint `{ce['checkpoint_sha256']}`。VAL accuracy={ce['accuracy']:.6f}、macro F1={ce['macro_f1']:.6f}、weighted F1={ce['weighted_f1']:.6f}。

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{pc(ce)}

局部 ROI 分类对多数缺陷类别明显可行，但 corrosion F1={ce['per_class'][6]['f1']:.4f}，且 background/defect 混淆仍明显；因此支持“ROI 分类比全图联合定位分类更易”的部分证据，不等价于已证明两阶段系统有效。
''',encoding='utf-8')
(R/'docs/exp06_supcon_fast_repro.md').write_text(f'''# Exp06 SupCon fast repro

状态：**COMPLETE / SUPCON_POSITIVE / VAL ONLY / test_accessed=false**

ResNet18 encoder，CE + 0.1×SupCon，temperature=0.07，双视图；与 CE 完全相同 manifest、batch、sampler、epoch、optimizer。50/50 epoch 完成，loss finite，embedding std 全程非零，best checkpoint `{su['checkpoint_sha256']}`。

VAL accuracy={su['accuracy']:.6f}、macro F1={su['macro_f1']:.6f}、weighted F1={su['weighted_f1']:.6f}；macro F1 相对 CE Δ={cmp['macro_f1_delta']:+.6f}。

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{pc(su)}

困难类 recall Δ（SupCon−CE）：Burn {cmp['difficult_recall_delta']['Burn']:+.4f}、Crack {cmp['difficult_recall_delta']['Crack']:+.4f}、corrosion {cmp['difficult_recall_delta']['corrosion']:+.4f}、background {cmp['difficult_recall_delta']['background']:+.4f}。满足 Gate，进入 Stage2 对照评估；不再调 SupCon。
''',encoding='utf-8')
b=st['baseline'];a=st['best']['A'];m=st['best']['B'];aa=st['analysis']['A'];ab=st['analysis']['B'];lat=st['latency']
(R/'docs/exp07_stage2_fast_repro.md').write_text(f'''# Exp07 Stage2 fast repro

状态：**COMPLETE / NEGATIVE / VAL ONLY / test_accessed=false**

Stage1 固定 Exp02 baseline；仅评估 conf=0.05/0.10/0.15 与 Stage2 threshold=0.3/0.5/0.7。crop=1.2×、224×224；Stage2 只过滤/改类，mask 永远来自 Stage1。Mode A score 定义 `stage1_conf × p_defect`，Mode B 定义 `stage1_conf × classifier_pred_class_probability`。主结果采用 SUPCON_POSITIVE classifier；CE 结果保存在 `exp07_stage2_ce/`。

Stage1-only：conf .05 TP/FP/FN=170/628/126，.10=157/323/139，.15=144/195/152。Baseline .25 P/R/F1={b['Precision']:.4f}/{b['Recall']:.4f}/{b['F1']:.4f}，FP/FN={b['FP']}/{b['FN']}。

最佳 Mode A：Stage1 .15、Stage2 .3，P/R/F1={a['Precision']:.4f}/{a['Recall']:.4f}/{a['F1']:.4f}，FP/FN={a['FP']}/{a['FN']}。相对 baseline：Recall {a['Recall']-b['Recall']:+.4f}、F1 {a['F1']-b['F1']:+.4f}、FP {a['FP']-b['FP']:+d}、FN {a['FN']-b['FN']:+d}。

最佳 Mode B：Stage1 .15、Stage2 .7，P/R/F1={m['Precision']:.4f}/{m['Recall']:.4f}/{m['F1']:.4f}，FP/FN={m['FP']}/{m['FN']}，wrong-class={m['WrongClass']}。相对 baseline：Recall {m['Recall']-b['Recall']:+.4f}、F1 {m['F1']-b['F1']:+.4f}、FP {m['FP']-b['FP']:+d}、FN {m['FN']-b['FN']:+d}。

91 个历史 low-conf recoverable GT：在选定 Stage1 conf=.15 下仅 21 个仍有 class-aware candidate，70 个在进入 Stage2 前已不可用；Mode A 保留 {aa['recoverable_retained']}、误滤 {aa['recoverable_filtered_by_stage2']}；Mode B 保留 {ab['recoverable_retained']}、误滤 {ab['recoverable_filtered_by_stage2']}。Mode A 删除/保留 FP={aa['FP_removed']}/{aa['FP_retained']}（{aa['FP_removal_rate']:.2%}）；Mode B={ab['FP_removed']}/{ab['FP_retained']}（{ab['FP_removal_rate']:.2%}）。Mode B correct/harmful/net reclassification={ab['correct_reclassification']}/{ab['harmful_reclassification']}/{ab['net_correction']:+d}。

Latency（warmup 后 VAL）：Stage1 mean/median={lat['stage1_mean_ms_image']:.3f}/{lat['stage1_median_ms_image']:.3f} ms/image；classifier={lat['classifier_mean_ms_candidate']:.3f}/{lat['classifier_median_ms_candidate']:.3f} ms/candidate；end-to-end={lat['end_to_end_mean_ms_image']:.3f}/{lat['end_to_end_median_ms_image']:.3f} ms/image。

AP=N/A：现有 fixed-point post-processing evaluator 无可靠 arbitrary-prediction COCO AP adapter，按 Fast Repro 规则不新写大型 AP 框架。Mode A/B 均未优于 YOLO conf=.25，Stage2=`NEGATIVE`；HardMining+FrozenStage2 probe=`NOT_RUN_BY_GATE`。
''',encoding='utf-8')
summary=[
 {'Method':'Exp06 CE ROI','Type':'Classifier','Classifier_accuracy':ce['accuracy'],'Classifier_macro_F1':ce['macro_f1'],'Stage1_conf':'N/A','Stage2_threshold':'N/A','Mask_P':'N/A','Mask_R':'N/A','Mask_F1':'N/A','Mask_AP50':'N/A','Mask_AP50_95':'N/A','FP':'N/A','FN':'N/A','WrongClass':'N/A','Latency_ms_per_image':'N/A','Conclusion':'COMPLETE'},
 {'Method':'Exp06 SupCon ROI','Type':'Classifier','Classifier_accuracy':su['accuracy'],'Classifier_macro_F1':su['macro_f1'],'Stage1_conf':'N/A','Stage2_threshold':'N/A','Mask_P':'N/A','Mask_R':'N/A','Mask_F1':'N/A','Mask_AP50':'N/A','Mask_AP50_95':'N/A','FP':'N/A','FN':'N/A','WrongClass':'N/A','Latency_ms_per_image':'N/A','Conclusion':'SUPCON_POSITIVE'},
 {'Method':'Exp07 Stage2 Mode A','Type':'Segmentation fixed-point','Classifier_accuracy':su['accuracy'],'Classifier_macro_F1':su['macro_f1'],'Stage1_conf':a['stage1_conf'],'Stage2_threshold':a['stage2_threshold'],'Mask_P':a['Precision'],'Mask_R':a['Recall'],'Mask_F1':a['F1'],'Mask_AP50':'N/A','Mask_AP50_95':'N/A','FP':a['FP'],'FN':a['FN'],'WrongClass':a['WrongClass'],'Latency_ms_per_image':lat['end_to_end_mean_ms_image'],'Conclusion':'NEGATIVE'},
 {'Method':'Exp07 Stage2 Mode B','Type':'Segmentation fixed-point','Classifier_accuracy':su['accuracy'],'Classifier_macro_F1':su['macro_f1'],'Stage1_conf':m['stage1_conf'],'Stage2_threshold':m['stage2_threshold'],'Mask_P':m['Precision'],'Mask_R':m['Recall'],'Mask_F1':m['F1'],'Mask_AP50':'N/A','Mask_AP50_95':'N/A','FP':m['FP'],'FN':m['FN'],'WrongClass':m['WrongClass'],'Latency_ms_per_image':lat['end_to_end_mean_ms_image'],'Conclusion':'NEGATIVE'}]
writecsv(F/'fasttrack_b_summary.csv',summary)
master=[
 {'Experiment':'Exp02 Baseline','Metric_domain':'Segmentation','Primary_metric':'Mask mAP50-95','Value':0.29898123412927496,'Conclusion':'PASS_WITH_NUMERICAL_WAIVER'},
 {'Experiment':'Exp03 Low-conf','Metric_domain':'Segmentation fixed-point','Primary_metric':'Mask F1 @ conf .25','Value':0.4749034749034749,'Conclusion':'POSITIVE'},
 {'Experiment':'Exp04 Crack-only','Metric_domain':'Segmentation','Primary_metric':'Crack AP50-95','Value':0.055484,'Conclusion':'NO_CLEAR_GAIN'},
 {'Experiment':'Exp05 Hard Mining','Metric_domain':'Segmentation','Primary_metric':'Mask mAP50-95','Value':0.311318,'Conclusion':'POSITIVE_CANDIDATE'},
 {'Experiment':'Exp06 CE ROI','Metric_domain':'Classification','Primary_metric':'Macro F1','Value':ce['macro_f1'],'Conclusion':'COMPLETE'},
 {'Experiment':'Exp06 SupCon ROI','Metric_domain':'Classification','Primary_metric':'Macro F1','Value':su['macro_f1'],'Conclusion':'SUPCON_POSITIVE'},
 {'Experiment':'Exp07 Stage2 Mode A','Metric_domain':'Segmentation fixed-point','Primary_metric':'Mask F1','Value':a['F1'],'Conclusion':'NEGATIVE'},
 {'Experiment':'Exp07 Stage2 Mode B','Metric_domain':'Segmentation fixed-point','Primary_metric':'Mask F1','Value':m['F1'],'Conclusion':'NEGATIVE'}]
writecsv(F/'fast_repro_master_summary.csv',master)
# Artifact append and validation
mp=F/'artifact_manifest.csv';old=list(csv.DictReader(mp.open(encoding='utf-8-sig')));known={x['file_path'] for x in old};new=[]
for p in sorted((F/'figures').glob('exp06_roi_ce/**/*.png'))+sorted((F/'figures').glob('exp06_supcon/**/*.png'))+sorted((F/'figures').glob('exp07_stage2/**/*')):
 if not p.is_file() or p.suffix.lower() not in ('.png','.jpg','.jpeg'):continue
 rel=str(p.relative_to(R));im=cv2.imread(str(p));ok=im is not None and im.size>0
 if rel not in known:new.append({'experiment':'fasttrack_b','artifact_type':p.suffix[1:],'file_path':rel,'file_size':p.stat().st_size,'decode_pass':str(ok)})
 if not ok:raise RuntimeError('decode fail '+rel)
writecsv(mp,old+new)
# Project metadata
idx=R/'docs/experiment_index.md';s=idx.read_text();s+='\n| Exp06.0/1 | ROI patch + ResNet18 CE | COMPLETE / VAL ONLY | `docs/exp06_roi_resnet_fast_repro.md` |\n| Exp06.2 | ResNet18 CE + SupCon | COMPLETE / SUPCON_POSITIVE | `docs/exp06_supcon_fast_repro.md` |\n| Exp07 | YOLO + ResNet Stage2 | COMPLETE / NEGATIVE / VAL ONLY | `docs/exp07_stage2_fast_repro.md` |\n';idx.write_text(s)
road=R/'ROADMAP.md';s=road.read_text().replace('- [ ] Exp03+ 其他后续方法（未授权）','- [x] Exp06 ROI ResNet18 CE（COMPLETE）\n- [x] Exp06 SupCon（SUPCON_POSITIVE）\n- [x] Exp07 Stage2（NEGATIVE；组合 probe 未过 Gate）\n- [ ] FastTrack-C KD / SimSiam（未授权）');road.write_text(s)
reg=R/'results/experiment_registry.csv';lines=reg.read_text().rstrip()+f"\nExp06.0-1,ROI patch dataset and ResNet18 CE,COMPLETE,2026-08-13T08:30:00Z,2026-08-13T08:42:00Z,4ac5ec6,/root/autodl-tmp/borescope-new-seg-data/v1,{EXPECTED if False else '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'},scripts/fasttrack_b.sh,results/fast_repro/exp06_roi,2532 train and 592 val patches; CE macro F1 {ce['macro_f1']:.6f}; test untouched\nExp06.2,ResNet18 CE plus SupCon,SUPCON_POSITIVE,2026-08-13T08:43:00Z,2026-08-13T08:55:00Z,4ac5ec6,/root/autodl-tmp/borescope-new-seg-data/v1,35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d,scripts/fasttrack_b.sh,results/fast_repro/exp06_roi/supcon,Macro F1 delta {cmp['macro_f1_delta']:+.6f}; no collapse; test untouched\nExp07,YOLO plus ResNet Stage2,NEGATIVE,2026-08-13T08:56:00Z,2026-08-13T09:10:00Z,4ac5ec6,/root/autodl-tmp/borescope-new-seg-data/v1,35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d,scripts/fasttrack_b.sh,results/fast_repro/exp07_stage2,Mode A and B F1 below YOLO .25; probe not run; test untouched\n";reg.write_text(lines)
state=R/'docs/PROJECT_STATE.md';s=state.read_text();s=s.replace('FastTrack-A complete; stopped for review','FastTrack-B complete; stopped for review').replace('Exp05 fair hard-mining comparison (POSITIVE_CANDIDATE)','Exp07 Stage2 (NEGATIVE); Exp06 SupCon positive as ROI classifier').replace('Exp02.1 provisional baseline `best.pt`; SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`; Gate not passed','Baseline: Exp02.1 `best.pt` SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`; positive candidate: Exp05 hard-mining Treatment; no final best declared').replace('None automatically; FastTrack-B requires explicit authorization','None automatically; FastTrack-C requires explicit authorization').replace('Wait for user review of `docs/handoffs/FASTTRACK_A_REVIEW.md`. Do not start FastTrack-B or access test automatically.','Wait for user review of `docs/handoffs/FASTTRACK_B_REVIEW.md`. Do not start FastTrack-C or access test automatically.');state.write_text(s)
reviewa=R/'docs/handoffs/FASTTRACK_A_REVIEW.md';s=reviewa.read_text().replace('151 个图像 artifact','152 个图像 artifact');reviewa.write_text(s)
change=R/'CHANGELOG.md';s=change.read_text().replace('共 151 个关键图像 artifact','共 152 个 FastTrack-A 关键图像 artifact');s=s.replace('## Unreleased','## Unreleased\n\n- FastTrack-B 完成：Exp06 CE COMPLETE，Exp06 SupCon SUPCON_POSITIVE，Exp07 Stage2 NEGATIVE；HardMining+FrozenStage2 probe 未过 Gate。\n- ROI patch train/val=2532/592，source-image leakage=0；全程 test_accessed=false。\n- 新增 FastTrack-B 总表、跨阶段 master summary、规定图表与定性案例；artifact 全量 decode PASS。');change.write_text(s)
total=len(old)+len(new)
(R/'docs/handoffs/FASTTRACK_B_REVIEW.md').write_text(f'''# FastTrack-B review

FastTrack-B 已完成：Exp06 CE `COMPLETE`，SupCon `SUPCON_POSITIVE`，Exp07 Stage2 `NEGATIVE`。无 Hard Gate，`test_accessed=false`。Stage2 非 positive，HardMining+FrozenStage2=`NOT_RUN_BY_GATE`；未进入 KD/SimSiam。

ROI patch train/val={ds['train_count']}/{ds['val_count']}，8 类与来源详见 Exp06 文档，source-image leakage=0。CE accuracy/macro-F1={ce['accuracy']:.4f}/{ce['macro_f1']:.4f}；SupCon={su['accuracy']:.4f}/{su['macro_f1']:.4f}，macro-F1 Δ={cmp['macro_f1_delta']:+.4f}。

SupCon Stage2 最佳 Mode A（.15/.3）P/R/F1={a['Precision']:.4f}/{a['Recall']:.4f}/{a['F1']:.4f}，FP/FN={a['FP']}/{a['FN']}；Mode B（.15/.7）={m['Precision']:.4f}/{m['Recall']:.4f}/{m['F1']:.4f}，FP/FN={m['FP']}/{m['FN']}。两者均低于 YOLO .25 F1={b['F1']:.4f}。当前值得保留：Exp05 hard mining 与 Exp06 SupCon ROI representation；不保留当前 Stage2 decision rule 为候选。

关键图：`results/fast_repro/figures/exp06_roi_ce/`、`exp06_supcon/`、`exp07_stage2/`。本轮新增 {len(new)} 个 artifact，manifest 总计 {total}，全部 decode PASS。总表：`fasttrack_b_summary.csv`、`fast_repro_master_summary.csv`。

是否进入 FastTrack-C（KD+SimSiam）：可在明确授权后快速覆盖，但证据优先级低于 Exp05 最终验证；本轮已 STOP。

需要重新上传 ChatGPT Source：`docs/PROJECT_STATE.md`、`ROADMAP.md`、`CHANGELOG.md`、`docs/experiment_index.md`、本轮 4 份 Markdown、`results/experiment_registry.csv`、`fasttrack_b_summary.csv`、`fast_repro_master_summary.csv`。
''',encoding='utf-8')
print(json.dumps({'old_artifacts':len(old),'new_artifacts':len(new),'total_artifacts':total,'all_decode_pass':True},indent=2))
