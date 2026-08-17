const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = process.cwd();
const OUT = path.join(ROOT, 'results', 'report_visualization_retro');
fs.mkdirSync(OUT, { recursive: true });
const manifest = [];

const COLORS = { blue: '#4472C4', orange: '#ED7D31', green: '#70AD47', red: '#C00000', gray: '#7F7F7F', purple: '#8064A2', gold: '#FFC000', bg: '#F7F9FC', ink: '#1F2937' };
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const readJson = p => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/); const head = lines.shift().split(',');
  return lines.map(line => { const v = line.split(','); return Object.fromEntries(head.map((h, i) => [h, v[i]])); });
}
const readCsv = p => parseCsv(fs.readFileSync(path.join(ROOT, p), 'utf8'));
function csvEscape(v) { const s = String(v ?? ''); return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; }

function baseSvg(w, h, title, subtitle = '') {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <rect width="100%" height="100%" fill="${COLORS.bg}"/>
  <style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:${COLORS.ink}} .title{font-size:27px;font-weight:700}.sub{font-size:15px;fill:#5B6573}.axis{font-size:13px}.small{font-size:12px}.label{font-size:14px;font-weight:600}</style>
  <text x="${w / 2}" y="38" text-anchor="middle" class="title">${esc(title)}</text>
  ${subtitle ? `<text x="${w / 2}" y="62" text-anchor="middle" class="sub">${esc(subtitle)}</text>` : ''}`;
}
async function saveSvg(rel, svg, meta) {
  const target = path.join(ROOT, rel); fs.mkdirSync(path.dirname(target), { recursive: true });
  await sharp(Buffer.from(svg + '</svg>')).png().toFile(target);
  manifest.push({ figure_path: rel.replaceAll('\\', '/'), ...meta });
}
function axes(w, h, margin, yMax, yLabel = '') {
  const pw = w - margin.l - margin.r, ph = h - margin.t - margin.b;
  let s = `<line x1="${margin.l}" y1="${margin.t}" x2="${margin.l}" y2="${margin.t + ph}" stroke="#697386"/><line x1="${margin.l}" y1="${margin.t + ph}" x2="${margin.l + pw}" y2="${margin.t + ph}" stroke="#697386"/>`;
  for (let i = 0; i <= 5; i++) { const y = margin.t + ph - ph * i / 5; const v = yMax * i / 5; s += `<line x1="${margin.l}" y1="${y}" x2="${margin.l + pw}" y2="${y}" stroke="#D9DEE7"/><text x="${margin.l - 10}" y="${y + 5}" text-anchor="end" class="axis">${v >= 10 ? v.toFixed(0) : v.toFixed(2)}</text>`; }
  if (yLabel) s += `<text transform="translate(18 ${margin.t + ph / 2}) rotate(-90)" text-anchor="middle" class="axis">${esc(yLabel)}</text>`;
  return { s, pw, ph };
}
async function groupedBars(rel, title, subtitle, labels, series, yMax, meta, yLabel = 'Metric') {
  const w = 1200, h = 650, m = { l: 80, r: 35, t: 90, b: 145 }; const { s: ax, pw, ph } = axes(w, h, m, yMax, yLabel);
  const groupW = pw / labels.length, bw = Math.min(34, groupW * .72 / series.length);
  let svg = baseSvg(w, h, title, subtitle) + ax;
  series.forEach((ser, si) => labels.forEach((lab, i) => { const val = Number(ser.values[i]); const bh = ph * val / yMax; const x = m.l + i * groupW + groupW / 2 + (si - (series.length - 1) / 2) * bw - bw / 2; const y = m.t + ph - bh; svg += `<rect x="${x}" y="${y}" width="${bw - 2}" height="${bh}" fill="${ser.color}" rx="2"/><text x="${x + (bw - 2) / 2}" y="${y - 5}" text-anchor="middle" class="small">${val.toFixed(val < 2 ? 3 : 0)}</text>`; }));
  labels.forEach((lab, i) => { const x = m.l + i * groupW + groupW / 2; svg += `<text transform="translate(${x} ${m.t + ph + 18}) rotate(-28)" text-anchor="end" class="axis">${esc(lab)}</text>`; });
  series.forEach((ser, i) => { const x = m.l + i * 190; svg += `<rect x="${x}" y="${h - 32}" width="16" height="16" fill="${ser.color}"/><text x="${x + 22}" y="${h - 19}" class="axis">${esc(ser.name)}</text>`; });
  await saveSvg(rel, svg, meta);
}
async function deltaBars(rel, title, labels, values, meta, yLimit = .35) {
  const w = 1100, h = 610, m = { l: 80, r: 30, t: 90, b: 135 }; const pw = w - m.l - m.r, ph = h - m.t - m.b, zeroY = m.t + ph / 2;
  let svg = baseSvg(w, h, title, '正值表示 SupCon 高于 CE；仅为冻结 VAL ROI 分类结果') + `<line x1="${m.l}" y1="${zeroY}" x2="${m.l + pw}" y2="${zeroY}" stroke="#333" stroke-width="2"/>`;
  const gw = pw / labels.length;
  labels.forEach((lab, i) => { const v = values[i], bh = Math.abs(v) / yLimit * ph / 2, x = m.l + i * gw + gw * .2, y = v >= 0 ? zeroY - bh : zeroY; svg += `<rect x="${x}" y="${y}" width="${gw * .6}" height="${bh}" fill="${v >= 0 ? COLORS.green : COLORS.red}"/><text x="${x + gw * .3}" y="${v >= 0 ? y - 7 : y + bh + 18}" text-anchor="middle" class="label">${v >= 0 ? '+' : ''}${v.toFixed(3)}</text><text transform="translate(${x + gw * .3} ${m.t + ph + 22}) rotate(-25)" text-anchor="end" class="axis">${esc(lab)}</text>`; });
  svg += `<text x="25" y="${zeroY - ph / 4}" transform="rotate(-90 25 ${zeroY - ph / 4})" class="axis">SupCon − CE</text>`;
  await saveSvg(rel, svg, meta);
}
function heatColor(v, max) { const t = Math.max(0, Math.min(1, v / max)); const r = Math.round(245 - 175 * t), g = Math.round(249 - 135 * t), b = Math.round(255 - 30 * t); return `rgb(${r},${g},${b})`; }
async function confusionComparison(rel, ce, su, meta) {
  const names = ce.per_class.map(x => x.class_name); const w = 1500, h = 760, top = 105, cell = 62;
  let svg = baseSvg(w, h, 'Exp06 CE vs SupCon 归一化混淆矩阵', '同一冻结 VAL ROI；行=GT，列=Prediction；数值为行归一化比例');
  for (const [panel, data, x0] of [['CE', ce, 120], ['SupCon', su, 820]]) {
    svg += `<text x="${x0 + cell * 4}" y="88" text-anchor="middle" class="label">${panel}</text>`;
    names.forEach((n, i) => { svg += `<text x="${x0 - 8}" y="${top + i * cell + 38}" text-anchor="end" class="small">${esc(n)}</text><text transform="translate(${x0 + i * cell + 35} ${top + cell * 8 + 10}) rotate(-45)" text-anchor="end" class="small">${esc(n)}</text>`; });
    data.confusion_matrix.forEach((row, i) => { const sum = row.reduce((a, b) => a + b, 0); row.forEach((v, j) => { const q = sum ? v / sum : 0; const x = x0 + j * cell, y = top + i * cell; svg += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${heatColor(q, 1)}" stroke="#fff"/><text x="${x + cell / 2}" y="${y + 37}" text-anchor="middle" class="small">${q.toFixed(2)}</text>`; }); });
  }
  await saveSvg(rel, svg, meta);
}
async function flow(rel, title, nodes, meta, width = 1200) {
  const boxH = 68, gap = 28, h = 100 + nodes.length * (boxH + gap); let svg = baseSvg(width, h, title, '颜色表示证据状态；箭头表示研究逻辑，不代表指标单调提升');
  nodes.forEach((n, i) => { const y = 82 + i * (boxH + gap), x = 170, bw = width - 340; svg += `<rect x="${x}" y="${y}" width="${bw}" height="${boxH}" rx="12" fill="${n.color}" opacity=".94"/><text x="${width / 2}" y="${y + 29}" text-anchor="middle" style="font-size:17px;font-weight:700;fill:white">${esc(n.title)}</text><text x="${width / 2}" y="${y + 52}" text-anchor="middle" style="font-size:13px;fill:white">${esc(n.note || '')}</text>`; if (i < nodes.length - 1) svg += `<line x1="${width / 2}" y1="${y + boxH}" x2="${width / 2}" y2="${y + boxH + gap - 7}" stroke="#556070" stroke-width="3"/><polygon points="${width / 2 - 7},${y + boxH + gap - 12} ${width / 2 + 7},${y + boxH + gap - 12} ${width / 2},${y + boxH + gap}" fill="#556070"/>`; });
  await saveSvg(rel, svg, meta);
}
async function linePlot(rel, title, subtitle, rows, cols, meta) {
  const w = 1100, h = 620, m = { l: 90, r: 40, t: 100, b: 80 }; const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const vals = cols.flatMap(c => rows.map(r => Number(r[c.key]))); const min = Math.min(...vals), max = Math.max(...vals), pad = (max - min || 1) * .08;
  let svg = baseSvg(w, h, title, subtitle) + `<line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t + ph}" stroke="#697386"/><line x1="${m.l}" y1="${m.t + ph}" x2="${m.l + pw}" y2="${m.t + ph}" stroke="#697386"/>`;
  for (let i = 0; i <= 5; i++) { const y = m.t + ph * i / 5, v = max + pad - (max - min + 2 * pad) * i / 5; svg += `<line x1="${m.l}" y1="${y}" x2="${m.l + pw}" y2="${y}" stroke="#D9DEE7"/><text x="${m.l - 8}" y="${y + 5}" text-anchor="end" class="axis">${v.toFixed(3)}</text>`; }
  cols.forEach((c, ci) => { const pts = rows.map((r, i) => { const x = m.l + pw * i / (rows.length - 1), y = m.t + ph * (max + pad - Number(r[c.key])) / (max - min + 2 * pad); return `${x},${y}`; }).join(' '); svg += `<polyline points="${pts}" fill="none" stroke="${c.color}" stroke-width="3"/>`; const lx = m.l + ci * 240; svg += `<line x1="${lx}" y1="${h - 25}" x2="${lx + 28}" y2="${h - 25}" stroke="${c.color}" stroke-width="4"/><text x="${lx + 35}" y="${h - 20}" class="axis">${esc(c.name)}</text>`; });
  svg += `<text x="${w / 2}" y="${h - 50}" text-anchor="middle" class="axis">Epoch (1–100)</text>`;
  await saveSvg(rel, svg, meta);
}
async function contactSheet(rel, files, labels, meta) {
  const tileW = 620, tileH = 390, cols = 2, rows = Math.ceil(files.length / cols); const canvas = sharp({ create: { width: tileW * cols, height: tileH * rows, channels: 3, background: '#F7F9FC' } }); const comps = [];
  for (let i = 0; i < files.length; i++) { const buf = await sharp(path.join(ROOT, files[i])).resize(tileW - 20, tileH - 60, { fit: 'contain', background: '#111' }).jpeg().toBuffer(); comps.push({ input: buf, left: (i % cols) * tileW + 10, top: Math.floor(i / cols) * tileH + 45 }); const labelSvg = `<svg width="${tileW}" height="45"><rect width="100%" height="100%" fill="#25364D"/><text x="12" y="29" font-family="Arial" font-size="18" fill="white">${esc(labels[i])}</text></svg>`; comps.push({ input: Buffer.from(labelSvg), left: (i % cols) * tileW, top: Math.floor(i / cols) * tileH }); }
  const target = path.join(ROOT, rel); fs.mkdirSync(path.dirname(target), { recursive: true }); await canvas.composite(comps).png().toFile(target); manifest.push({ figure_path: rel.replaceAll('\\', '/'), ...meta });
}

function meta(id, experiment, type, source, split, caption, sourceCheckpoint = 'N/A', testAccessed = false) { return { figure_id: id, experiment, figure_type: type, source_data: source, source_checkpoint: sourceCheckpoint, split, sample_ids: 'N/A', new_inference_required: false, visualization_only_reinference: false, metric_selection_performed: false, test_accessed: testAccessed, caption_cn: caption, status: 'PASS' }; }

(async () => {
  const ce = readJson('results/fast_repro/exp06_roi/ce/metrics.json'); const su = readJson('results/fast_repro/exp06_roi/supcon/metrics.json');
  const cls = ce.per_class.map(x => x.class_name);
  await groupedBars('results/report_visualization_retro/exp06_roi_ce/per_class_precision_recall_f1_support.png', 'Exp06 ROI CE：逐类 Precision / Recall / F1', '柱顶为指标；类别标签括号内为 support', cls.map((x, i) => `${x} (n=${ce.per_class[i].support})`), [
    { name: 'Precision', values: ce.per_class.map(x => x.precision), color: COLORS.blue }, { name: 'Recall', values: ce.per_class.map(x => x.recall), color: COLORS.orange }, { name: 'F1', values: ce.per_class.map(x => x.f1), color: COLORS.green }
  ], 1, meta('R06-CE-01', 'Exp06 CE', 'grouped_bar', 'results/fast_repro/exp06_roi/ce/metrics.json', 'VAL ROI', '多数 ROI 类别可分；corrosion 与 background/defect 混淆明显。', ce.checkpoint_sha256), 'Metric');
  await groupedBars('results/report_visualization_retro/exp06_supcon/ce_vs_supcon_overall.png', 'Exp06：CE vs SupCon 总指标', '冻结 VAL ROI；不同于 segmentation mAP', ['Accuracy', 'Macro F1', 'Weighted F1'], [
    { name: 'CE', values: [ce.accuracy, ce.macro_f1, ce.weighted_f1], color: COLORS.blue }, { name: 'SupCon', values: [su.accuracy, su.macro_f1, su.weighted_f1], color: COLORS.orange }
  ], 1, meta('R06-SC-01', 'Exp06 SupCon', 'grouped_bar', 'CE/SupCon metrics.json', 'VAL ROI', 'SupCon 的 ROI macro F1 比 CE 高 0.015894；不外推为 segmentation 提升。', su.checkpoint_sha256));
  await deltaBars('results/report_visualization_retro/exp06_supcon/per_class_recall_delta.png', 'Exp06：逐类 Recall 差值', cls, cls.map((_, i) => su.per_class[i].recall - ce.per_class[i].recall), meta('R06-SC-02', 'Exp06 SupCon', 'delta_bar', 'CE/SupCon metrics.json', 'VAL ROI', 'SupCon−CE Recall；Tip curl 仅 3 个样本，负差值需谨慎解释。', su.checkpoint_sha256));
  await deltaBars('results/report_visualization_retro/exp06_supcon/per_class_f1_delta.png', 'Exp06：逐类 F1 差值', cls, cls.map((_, i) => su.per_class[i].f1 - ce.per_class[i].f1), meta('R06-SC-03', 'Exp06 SupCon', 'delta_bar', 'CE/SupCon metrics.json', 'VAL ROI', 'SupCon−CE F1；总体为正但并非所有类别均改善。', su.checkpoint_sha256));
  await confusionComparison('results/report_visualization_retro/exp06_supcon/ce_vs_supcon_confusion_matrix.png', ce, su, meta('R06-SC-04', 'Exp06 SupCon', 'confusion_matrix_comparison', 'CE/SupCon metrics.json', 'VAL ROI', '相同 VAL ROI 上的 CE 与 SupCon 行归一化混淆矩阵。', su.checkpoint_sha256));

  const stage = readJson('results/fast_repro/exp07_stage2/summary.json'); const grid = readCsv('results/fast_repro/exp07_stage2/grid.csv'); const low = grid.find(x => x.mode === 'stage1_only' && Number(x.stage1_conf) === .15);
  await groupedBars('results/report_visualization_retro/exp07_stage2/fixed_point_quantitative_comparison.png', 'Exp07 Stage2：冻结 operating point 对比', 'AP=N/A；只比较原 fixed-point evaluator 支持的指标', ['YOLO .25', 'Stage1 .15', 'Mode A', 'Mode B'], [
    { name: 'Precision', values: [stage.baseline.Precision, Number(low.Precision), stage.best.A.Precision, stage.best.B.Precision], color: COLORS.blue }, { name: 'Recall', values: [stage.baseline.Recall, Number(low.Recall), stage.best.A.Recall, stage.best.B.Recall], color: COLORS.orange }, { name: 'F1', values: [stage.baseline.F1, Number(low.F1), stage.best.A.F1, stage.best.B.F1], color: COLORS.green }
  ], .65, meta('R07-01', 'Exp07', 'grouped_bar', 'results/fast_repro/exp07_stage2/{grid.csv,summary.json}', 'VAL', 'Mode A/B 均未超过 YOLO .25 的 F1；不得虚构 AP。', su.checkpoint_sha256));
  await groupedBars('results/report_visualization_retro/exp07_stage2/fp_fn_comparison.png', 'Exp07 Stage2：FP / FN 代价', 'Mode B 降低 FP，但增加 FN；体现 precision–recall trade-off', ['YOLO .25', 'Stage1 .15', 'Mode A', 'Mode B'], [
    { name: 'FP', values: [stage.baseline.FP, Number(low.FP), stage.best.A.FP, stage.best.B.FP], color: COLORS.red }, { name: 'FN', values: [stage.baseline.FN, Number(low.FN), stage.best.A.FN, stage.best.B.FN], color: COLORS.gray }
  ], 220, meta('R07-02', 'Exp07', 'grouped_bar', 'results/fast_repro/exp07_stage2/{grid.csv,summary.json}', 'VAL', 'Stage2 的 FP 收益不足以抵消 TP 删除和 FN 增加。', su.checkpoint_sha256), 'Count');
  const qroot = 'results/fast_repro/figures/exp07_stage2/qualitative';
  const qfiles = [
    `${qroot}/success_filter_fp__45.jpg`, `${qroot}/error_filter_tp__160.jpg`,
    `${qroot}/success_reclassify__648.jpg`, `${qroot}/error_reclassify__187.jpg`,
    `${qroot}/success_keep_low_conf_tp__42.jpg`, `${qroot}/error_filter_tp__10.jpg`,
  ];
  await contactSheet('results/report_visualization_retro/exp07_stage2/selected_success_failure_cases.png', qfiles, ['A 正确删除 FP', 'B 错误删除 TP', 'C Mode B 正确重分类', 'D Mode B 有害重分类', 'E 低置信目标被保留', 'F 低置信 TP 被错误删除'], meta('R07-03', 'Exp07', 'contact_sheet_reuse', qfiles.join(';'), 'VAL', '复用历史定性案例；没有重新推理，也没有修改 Stage1 mask。', su.checkpoint_sha256));

  const hist = readCsv('results/fast_repro/exp09_simsiam/ssl/history.csv');
  await linePlot('results/report_visualization_retro/exp09_simsiam/ssl_training_diagnostics.png', 'Exp09 SimSiam：100 epoch 工程诊断曲线', 'loss finite；feature/embedding std 非零。曲线正常不等于 trainable backbone 已学习。', hist, [{ key: 'loss', name: 'Loss', color: COLORS.blue }, { key: 'feature_std', name: 'Feature std', color: COLORS.orange }, { key: 'embedding_std', name: 'Embedding std', color: COLORS.green }], meta('R09-01', 'Exp09', 'line_chart', 'results/fast_repro/exp09_simsiam/ssl/history.csv', 'TRAIN only', '训练表面有限且无明显 collapse，但后续参数审计发现 0/120 trainable 参数改变。'));
  await groupedBars('results/report_visualization_retro/exp09_simsiam/backbone_parameter_audit.png', 'Loss appeared normal, but trainable backbone parameters did not update', '训练曲线表面正常不能证明模型真正完成了参数学习', ['Trainable parameters', 'BatchNorm buffers'], [
    { name: 'Changed', values: [0, 120], color: COLORS.red }, { name: 'Unchanged', values: [120, 0], color: COLORS.gray }
  ], 130, meta('R09-02', 'Exp09', 'parameter_audit', 'results/fast_repro/exp09_transfer_repair/transfer_gate_report.json', 'TRAIN audit', 'trainable changed=0/120；BN buffers changed=120/120，因此 reconstruction 无效。'), 'Tensor count');
  await groupedBars('results/report_visualization_retro/exp09_simsiam/transfer_verification_summary.png', 'Exp09.2a：Transfer verification summary', 'Transfer mechanism PASS_REVISED；问题位于 SSL 参数更新而非 checkpoint transfer', ['Key/shape', 'Immediate trainable', 'FP32 round-trip', 'Native round-trip'], [{ name: 'Verified', values: [240, 120, 240, 240], color: COLORS.green }], 260, meta('R09-03', 'Exp09.2a', 'verification_bar', 'results/fast_repro/exp09_transfer_repair/transfer_gate_report.json', 'Engineering audit', 'key/shape 240/240、immediate trainable 120/120、两种 round-trip 240/240。'));
  await flow('results/report_visualization_retro/exp09_simsiam/exp09_failure_chain.png', 'Exp09 SimSiam 证据链', [
    { title: 'SimSiam 100 epochs', note: 'TRAIN-only；无 TEST', color: COLORS.blue }, { title: 'Loss finite / feature not collapsed', note: '必要但不充分', color: COLORS.green }, { title: 'Parameter delta audit', note: '0/120 trainable changed；120/120 BN buffers changed', color: COLORS.red }, { title: '检查是否为 checkpoint transfer 问题', note: 'Exp09.2a 独立核验', color: COLORS.purple }, { title: 'Transfer PASS_REVISED', note: 'key/shape、in-memory load、round-trip 均通过', color: COLORS.green }, { title: 'INVALID_BY_BACKBONE_NO_UPDATE', note: 'SSL implementation/optimization invalid；不是 SimSiam 方法负结论', color: COLORS.red }, { title: 'Downstream NOT_RUN_BY_GATE', note: 'Performance NOT_EVALUATED', color: COLORS.gray }
  ], meta('R09-04', 'Exp09', 'flow_diagram', 'Exp09/Exp09.2a Markdown + JSON', 'TRAIN/engineering audit', 'SimSiam reconstruction 的完整工程有效性诊断链。'));

  await flow('results/report_visualization_retro/project_overview/full_project_pipeline.png', '新孔探实例分割项目总流程', [
    { title: 'DATA — Exp00 数据/环境审计', note: '24 unpaired 排除；88 near groups；Gate', color: COLORS.blue }, { title: 'Exp01 数据冻结', note: 'JSON→YOLO-seg；group-aware split；0 leakage', color: COLORS.green }, { title: 'BASELINE — Exp02', note: 'YOLO11n-seg；PASS_WITH_NUMERICAL_WAIVER', color: COLORS.blue }, { title: 'ERROR ANALYSIS — Exp03 / Exp04', note: 'Low-conf positive diagnostic；one-class no clear gain', color: COLORS.orange }, { title: 'METHOD BRANCH — Exp05 / Exp06 / Exp07', note: 'Hard Mining preliminary；ROI/SupCon；Stage2 NEGATIVE', color: COLORS.purple }, { title: 'ENGINEERING / VALIDITY — Exp08 / Exp09', note: 'KD SKIPPED；SimSiam INVALID / NOT_EVALUATED', color: COLORS.gray }, { title: 'ROBUSTNESS — Exp10', note: 'Three-seed：HARD_MINING_NOT_CONFIRMED', color: COLORS.red }, { title: 'CANDIDATE FREEZE', note: 'Baseline seed44；freeze before TEST', color: COLORS.gold }, { title: 'Exp11 FINAL TEST', note: 'One frozen evaluation；no post-TEST selection', color: COLORS.green }, { title: 'PROJECT_COMPLETE', note: 'Final method remains Baseline', color: COLORS.ink }
  ], meta('R-PROJ-01', 'Project', 'flow_diagram', 'PROJECT_STATE + Exp00–Exp11 final evidence', 'N/A', '项目不是连续找到更好模型，而是通过 Gate 与多 seed 排除不稳健方案。'));
  await flow('results/report_visualization_retro/project_overview/method_relationships.png', '方法之间的研究逻辑', [
    { title: 'Baseline FN', note: '发现低置信目标与定位/分类错误', color: COLORS.blue }, { title: 'Low-confidence recoverable', note: '91/173 FN；直接降阈值会 FP explosion', color: COLORS.orange }, { title: '需要过滤 FP → ROI classifier', note: 'CE 证明 ROI 分类具有部分可分性', color: COLORS.purple }, { title: 'SupCon representation', note: 'ROI macro F1 +0.015894；不是 segmentation 提升', color: COLORS.green }, { title: 'Stage2', note: '过滤/重分类仍未超过 baseline：NEGATIVE', color: COLORS.red }, { title: 'Hard samples → Hard Mining → 3 seeds', note: 'single-seed gain → mean -0.005543 → NOT_CONFIRMED', color: COLORS.red }, { title: 'Classifier knowledge → KD', note: 'engineering-cost Gate；NOT_EVALUATED', color: COLORS.gray }, { title: 'COCO→borescope domain gap → SimSiam', note: '0/120 trainable update → INVALID / NOT_EVALUATED', color: COLORS.gray }
  ], meta('R-PROJ-02', 'Project', 'logic_diagram', 'Exp03–Exp10 final evidence', 'N/A', '展示实验由前序问题自然导出，而不是无关联地堆方法。'));

  const ds = readCsv('results/project_review/dataset_summary_for_report.csv'); const globalClasses = ds.filter(x => x.Category === 'Class' && x.Scope === 'Global');
  await groupedBars('results/report_visualization_retro/exp00_dataset/dataset_audit_summary.png', 'Exp00 数据审计：7 类实例分布', '1847 instances；最大/最小 20.83:1；24 unpaired 不作 background', globalClasses.map(x => x.Class_or_bin), [{ name: 'Instances', values: globalClasses.map(x => Number(x.Value)), color: COLORS.blue }], 800, meta('R00-01', 'Exp00', 'bar_chart', 'results/project_review/dataset_summary_for_report.csv', 'ALL manifest evidence', '类别严重不均衡，但少数类并不必然是最困难类别。'), 'Instances');
  await groupedBars('results/report_visualization_retro/exp01_dataset/split_summary.png', 'Exp01 Group-aware split', '88/88 near-duplicate groups 未跨 split；split SHA256 已冻结', ['TRAIN', 'VAL', 'TEST'], [{ name: 'Images', values: [668, 154, 147], color: COLORS.blue }, { name: 'Instances', values: [1266, 296, 285], color: COLORS.orange }], 1400, meta('R01-01', 'Exp01', 'grouped_bar', 'frozen split manifest + dataset summary', 'TRAIN/VAL/TEST manifest', '按 near-duplicate 连通组整体划分，避免相邻帧泄漏。'), 'Count');
  await flow('results/report_visualization_retro/exp08_kd/kd_engineering_gate.png', 'Exp08 KD 工程 Gate', [
    { title: 'Frozen SupCon classifier teacher', note: 'eval mode；no gradients', color: COLORS.blue }, { title: 'YOLO P3 ROIAlign auxiliary path', note: 'batch 32 forward/loss finite', color: COLORS.green }, { title: 'Online teacher ROI extraction', note: 'batch4 OOM；chunking 后单步 >70s', color: COLORS.orange }, { title: 'SKIPPED_BY_ENGINEERING_GATE', note: 'formal AUX_CE/KD training NOT_EVALUATED', color: COLORS.gray }
  ], meta('R08-01', 'Exp08', 'flow_diagram', 'docs/exp08_kd_fast_repro.md', 'TRAIN smoke only', 'KD 因工程成本 Gate 停止，不能写成 KD failed。'));
  const test = readJson('results/final_test/exp11_retry1/summary.json');
  await groupedBars('results/report_visualization_retro/exp11_final/val_vs_test_generalization.png', 'Final seed44 Baseline：VAL vs TEST', 'TEST−VAL Mask mAP50-95 = -0.053536；仅作泛化观察，不用于调参', ['Mask mAP50-95'], [{ name: 'Frozen VAL', values: [test.selected_val_mask_map50_95], color: COLORS.blue }, { name: 'One-time TEST', values: [test.Mask.mAP50_95], color: COLORS.orange }], .4, meta('R11-01', 'Exp11', 'grouped_bar', 'results/final_test/exp11_retry1/summary.json', 'TEST existing metrics only', '冻结候选在 TEST 上低于 VAL；TEST 后未训练、未调阈值、未重新选择。', test.checkpoint_sha256, false));

  const fields = ['figure_id','experiment','figure_path','figure_type','source_data','source_checkpoint','split','sample_ids','new_inference_required','visualization_only_reinference','metric_selection_performed','test_accessed','caption_cn','status'];
  const lines = [fields.join(','), ...manifest.map(r => fields.map(k => csvEscape(r[k])).join(','))]; fs.writeFileSync(path.join(OUT, 'figure_manifest.csv'), lines.join('\n') + '\n', 'utf8');
  const decoded = [];
  for (const row of manifest) {
    const figure = path.join(ROOT, row.figure_path);
    const info = await sharp(figure).metadata();
    if (!info.width || !info.height || info.format !== 'png') throw new Error(`PNG decode audit failed: ${row.figure_path}`);
    decoded.push({ figure_path: row.figure_path, width: info.width, height: info.height, format: info.format });
  }
  fs.writeFileSync(path.join(OUT, 'artifact_verification.json'), JSON.stringify({ status: 'PASS', figure_count: manifest.length, decoded_png_count: decoded.length, decoded, no_training: true, no_inference: true, test_accessed: false, metric_selection_performed: false }, null, 2) + '\n');
  console.log(`generated ${manifest.length} report figures`);
})().catch(e => { console.error(e); process.exit(1); });
