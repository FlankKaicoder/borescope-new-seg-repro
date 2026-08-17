import fs from 'fs';
import path from 'path';

const root = process.cwd();
const outDir = path.join(root, 'results', 'report_consolidation');
fs.mkdirSync(outDir, { recursive: true });

function walk(dir, predicate = () => true) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walk(p, predicate));
    else if (predicate(p)) out.push(p);
  }
  return out;
}

function rel(p) { return path.relative(root, p).replaceAll('\\', '/'); }
function q(v) {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}
function csv(file, fields, rows) {
  fs.writeFileSync(file, [fields.join(','), ...rows.map(r => fields.map(k => q(r[k])).join(','))].join('\n') + '\n', 'utf8');
}

const mdFiles = [
  ...walk(path.join(root, 'docs'), p => p.toLowerCase().endsWith('.md')),
  ...fs.readdirSync(root, { withFileTypes: true })
    .filter(e => e.isFile() && e.name.toLowerCase().endsWith('.md'))
    .map(e => path.join(root, e.name)),
].sort();

const docs = mdFiles.map(p => {
  const file = rel(p);
  const text = fs.readFileSync(p, 'utf8');
  // Treat one Chinese character and one English lexical token as comparable
  // language units; counting Latin letters would misclassify technical Chinese
  // documents merely because paths, hashes and metric names are preserved.
  const zh = (text.match(/[\u3400-\u9fff]/g) || []).length;
  const en = (text.match(/[A-Za-z][A-Za-z0-9_.+-]*/g) || []).length;
  const denom = Math.max(1, zh + en);
  const chineseRatio = zh / denom;
  const englishRatio = en / denom;
  let language = 'mixed';
  if (chineseRatio >= 0.70) language = 'zh-dominant';
  else if (englishRatio >= 0.70) language = 'en-dominant';
  const lower = file.toLowerCase();
  const authoritative = /docs\/(exp\d|handoffs\/|project_state|decision_log|historical_methods)|changelog/.test(lower);
  const summary = /review|timeline|takeaways|index|final_|paper|readme|roadmap|project_scope/.test(lower);
  const documentType = authoritative ? 'authoritative_experiment_or_gate' : summary ? 'project_summary' : 'supporting_document';
  const important = authoritative || summary;
  const needsTranslation = important && englishRatio > 0.45;
  const needsRewrite = summary && (englishRatio > 0.30 || file === 'docs/new_dataset_full_experiment_review.md');
  let action = 'keep_original_and_index';
  if (needsTranslation && authoritative) action = 'keep_original_create_zh_companion';
  else if (needsRewrite) action = 'keep_original_create_full_zh_reorganization';
  else if (important) action = 'keep_original_reference_from_zh_hub';
  return {
    file,
    language,
    chinese_ratio: chineseRatio.toFixed(4),
    english_ratio: englishRatio.toFixed(4),
    document_type: documentType,
    authoritative_or_summary: authoritative ? 'authoritative' : summary ? 'summary' : 'supporting',
    needs_translation: needsTranslation,
    needs_rewrite: needsRewrite,
    recommended_action: action,
  };
});
csv(path.join(outDir, 'document_language_audit.csv'), Object.keys(docs[0]), docs);

const figureRoots = [
  'results/fast_repro/figures',
  'results/final_verify/figures',
  'results/final_test',
  'results/project_review',
];
const imageExt = new Set(['.png', '.jpg', '.jpeg']);
const figureFiles = figureRoots.flatMap(r => walk(path.join(root, r), p => imageExt.has(path.extname(p).toLowerCase()))).sort();
const figures = figureFiles.map(p => {
  const file = rel(p);
  const lower = file.toLowerCase();
  const m = lower.match(/exp\d+[a-z0-9_.-]*/);
  const experiment = m ? m[0].replaceAll('_', '-') : lower.includes('project_review') ? 'project_review' : 'unknown';
  let explains = '训练或评估过程的原始可视化证据';
  if (lower.includes('confusion_matrix')) explains = '类别混淆结构';
  else if (lower.includes('qualitative') || lower.includes('success_') || lower.includes('error_')) explains = '成功与失败的定性案例';
  else if (lower.includes('curve') || lower.includes('results.png')) explains = '训练或 P/R/F1/PR 曲线';
  else if (lower.includes('three_seed') || lower.includes('paired')) explains = '三随机种子稳健性或配对效应';
  else if (lower.includes('threshold') || lower.includes('grid')) explains = '冻结阈值下的权衡或网格结果';
  else if (lower.includes('distribution')) explains = '数据分布';
  const rawBatch = /train_batch|val_batch/.test(lower);
  const reportQuality = rawBatch ? 'supporting' : lower.includes('qualitative') ? 'good' : 'report_ready';
  return {
    experiment,
    figure: file,
    exists: true,
    what_it_explains: explains,
    report_quality: reportQuality,
    missing_information: rawBatch ? '需要中文图注与上下文' : '由中文图索引补充讲解口径',
    keep: true,
    regenerate: false,
    new_figure_needed: false,
  };
});
csv(path.join(outDir, 'figure_audit.csv'), Object.keys(figures[0]), figures);

const gaps = [
  ['Exp00', '已有 dataset audit/contact sheets；最终 class/size distribution 可复用', '项目级数据审计摘要图', '新补', '仅基于现有 CSV'],
  ['Exp01', '已有 50 张 round-trip overlays 与 split 证据', 'split/leakage 汇报摘要', '新补', '仅基于 manifest 审计事实'],
  ['Exp02', '已有训练曲线、PR/P/R/F1、confusion matrix、size/error 表', '无需重新推理', '已有', '复用原图'],
  ['Exp03', '已有 threshold、FN recovery、定性错误图', '无需新增', '已有', '复用原图'],
  ['Exp04', '已有 one-class comparison 和标准曲线', '无需新增', '已有', '复用原图'],
  ['Exp05', '已有 Control/Treatment 对比；但单 seed 结论已被 Exp10 覆盖', '汇报时需显著标注 preliminary', '已有', '不得包装为最终提升'],
  ['Exp06 CE', '已有 raw/normalized confusion matrix、F1、training curves', 'P/R/F1+support 汇报合并图；样本级预测无缓存', '新补+无法补', '不做重新推理'],
  ['Exp06 SupCon', '已有 confusion matrix、CE vs SupCon 总指标、loss/std', 'Recall/F1 delta；t-SNE/迁移案例缺 embedding/逐样本预测', '新补+无法补', 't-SNE 不作为定量证据，不重新推理'],
  ['Exp07', '已有 grid、主对比、5 类定性案例', '数值热图和统一 P/R/F1/FP/FN 图；复用案例 contact sheet', '新补', 'AP 保持 N/A'],
  ['Exp08', '已有工程 Gate 文档', '工程 Gate 流程摘要', '新补', '不得写 KD negative'],
  ['Exp09', '已有 loss/std/transfer 图', '参数 changed/unchanged 图和 failure-chain 图', '新补', '只做工程诊断；不得做下游分割对比'],
  ['Exp10', '已有 5 张三 seed 图', '无需新增模型结果', '已有', '复用原图'],
  ['Exp11', '已有 TEST confusion/curves/64 cases/grid', 'VAL vs TEST 汇报摘要', '新补', '只读既有 TEST CSV；不重新访问 TEST'],
];
const gapRows = gaps.map(([experiment, existing, missing, disposition, rationale]) => ({ experiment, existing_visualization: existing, missing_or_needed: missing, disposition, rationale }));
csv(path.join(outDir, 'visualization_gap_matrix.csv'), Object.keys(gapRows[0]), gapRows);

const summary = {
  markdown_count: docs.length,
  english_dominant_markdown: docs.filter(x => x.language === 'en-dominant').length,
  mixed_markdown: docs.filter(x => x.language === 'mixed').length,
  zh_dominant_markdown: docs.filter(x => x.language === 'zh-dominant').length,
  audited_existing_visualizations: figures.length,
  audit_scope: figureRoots,
  no_training: true,
  no_inference: true,
  test_accessed: false,
};
fs.writeFileSync(path.join(outDir, 'audit_summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));
