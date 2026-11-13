/* Offline Exp14 review. CSV quoting and delayed Blob release follow the G3-C0 recovery UI. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const bundle = JSON.parse($('bundle').textContent);
  const byId = new Map(bundle.records.map(record => [record.initial.review_id, record]));
  const assisted = Boolean(bundle.ai_review);
  const storageKey = bundle.phase + (assisted ? ':ai-assisted:v1' : ':review:v1');
  if (bundle.title) {
    document.title = bundle.title;
    document.querySelector('h1').textContent = bundle.title;
  }
  const labels = {
    UNREVIEWED: '待审核', PASS: 'PASS · 修复成功', RETRY_REPAIR: 'RETRY_REPAIR · 重修',
    RETRY_ALIGNMENT: 'RETRY_ALIGNMENT · 重新配准', HOLD: 'HOLD · 暂缓',
  };
  const decisions = new Map(bundle.records.map(record => [record.initial.review_id, {...record.initial}]));
  const drafts = new Map();
  const confirmations = new Map();
  const priorities = new Set();
  let filtered = [];
  let index = 0;
  let current = null;

  function option(select, value, label = value) {
    select.add(new Option(label, value));
  }
  function notice(message, error = false) {
    $('notice').textContent = message;
    $('notice').dataset.error = String(error);
  }
  function snapshot() {
    const payload = {phase: bundle.phase, records: [...decisions.values()]};
    if (assisted) payload.ai_workspace = {
      drafts: [...drafts.values()], confirmations: [...confirmations], priorities: [...priorities],
    };
    return payload;
  }

  function validate(rows) {
    if (!Array.isArray(rows) || !rows.length) throw new Error('审核文件没有记录。');
    const staged = new Map();
    for (const row of rows) {
      if (!row || Object.keys(row).length !== bundle.fields.length ||
          !bundle.fields.every(key => typeof row[key] === 'string')) throw new Error('字段与当前审核格式不一致。');
      const record = byId.get(row.review_id);
      if (!record || staged.has(row.review_id) ||
          !bundle.identity.every(key => row[key] === record.initial[key])) {
        throw new Error('存在过期、重复或不属于本页面的图像身份。');
      }
      if (!bundle.states.includes(row.decision) || !bundle.reasons.includes(row.reason) ||
          !['', 'YES'].includes(row.whole_image_normal) ||
          !['', 'CONFIRMED'].includes(row.b_region_review)) throw new Error('审核决定或确认字段无效。');
      const polygon = JSON.parse(row.polygon_decisions);
      if (!polygon || typeof polygon !== 'object' || Array.isArray(polygon) ||
          !Object.entries(polygon).every(([id, state]) =>
            record.polygons.some(p => p.polygon_id === id) && state !== 'UNREVIEWED' && bundle.states.includes(state))) {
        throw new Error('Polygon ID 或决定无效。');
      }
      const states = Object.values(polygon);
      if (row.whole_image_normal && (row.decision !== 'PASS' || states.some(s => s !== 'PASS'))) {
        throw new Error('整图确认与目标决定冲突。');
      }
      if (row.b_region_review && (!record.aligned || (row.decision !== 'PASS' && !states.includes('PASS')))) {
        throw new Error('区域可用确认需要 Aligned 和至少一个通过的目标。');
      }
      staged.set(row.review_id, {...row});
    }
    return staged;
  }

  function restore(payload) {
    if (payload.phase !== bundle.phase) throw new Error('审核文件属于其他阶段。');
    const staged = validate(payload.records);
    const workspace = assisted ? payload.ai_workspace : null;
    const stagedDrafts = workspace?.drafts?.length ? validate(workspace.drafts) : new Map();
    const stagedConfirmations = new Map(workspace?.confirmations || []);
    const stagedPriorities = new Set(workspace?.priorities || []);
    for (const [id, state] of stagedConfirmations) {
      if (!staged.has(id) || !['HUMAN_CONFIRMED', 'HUMAN_OVERRIDDEN', 'IMPORTED'].includes(state)) {
        throw new Error('人工确认记录无效。');
      }
    }
    if ([...stagedPriorities, ...stagedDrafts.keys()].some(id => !byId.has(id))) throw new Error('草稿身份无效。');
    const same = (a, b) => bundle.fields.every(key => a[key] === b[key]);
    if (assisted) for (const [id, row] of staged) {
      const existing = decisions.get(id);
      const baseline = byId.get(id).initial;
      if (!same(existing, baseline) && !same(row, baseline) && !same(existing, row)) {
        throw new Error('导入决定与当前人工决定冲突。请保留原文件并逐条处理。');
      }
    }
    for (const [id, row] of staged) {
      const baseline = byId.get(id).initial;
      if (assisted && same(row, baseline) && !same(decisions.get(id), baseline)) continue;
      decisions.set(id, row);
      if (assisted && (stagedConfirmations.has(id) || !same(row, baseline))) {
        confirmations.set(id, stagedConfirmations.get(id) || 'IMPORTED');
      }
    }
    for (const [id, row] of stagedDrafts) drafts.set(id, row);
    for (const id of stagedPriorities) priorities.add(id);
    return staged.size;
  }

  function saveLocal() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(snapshot()));
      return true;
    } catch {
      notice('当前浏览器无法本地暂存，请导出 CSV 或 JSON。', true);
      return false;
    }
  }

  function csvCell(value) { return '"' + value.replaceAll('"', '""') + '"'; }
  function csvText() {
    return '\ufeff' + bundle.fields.join(',') + '\r\n' +
      [...decisions.values()].map(row => bundle.fields.map(key => csvCell(row[key])).join(',')).join('\r\n') + '\r\n';
  }

  function parseCsv(text) {
    text = text.replace(/^\ufeff/, '');
    const records = [];
    let row = [], cell = '', quoted = false, closedQuote = false;
    function appendRow() {
      row.push(cell);
      if (row.some(value => value !== '')) records.push(row);
      row = [];
      cell = '';
      closedQuote = false;
    }
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (ch === '"') { quoted = false; closedQuote = true; }
        else cell += ch;
      } else if (ch === ',' || ch === '\n' || ch === '\r') {
        if (ch === ',') { row.push(cell); cell = ''; closedQuote = false; }
        else {
          if (ch === '\r' && text[i + 1] === '\n') i++;
          appendRow();
        }
      } else if (closedQuote) throw new Error('CSV 引号后有无效内容。');
      else if (ch === '"' && cell === '') quoted = true;
      else if (ch === '"') throw new Error('CSV 字段含有未转义引号。');
      else cell += ch;
    }
    if (quoted) throw new Error('CSV 存在未闭合的引号。');
    if (cell || row.length || closedQuote) appendRow();
    const header = records.shift();
    if (!header || header.join(',') !== bundle.fields.join(',')) throw new Error('CSV 表头与当前审核格式不一致。');
    return records.map(values => {
      if (values.length !== header.length) throw new Error('CSV 字段数量错误。');
      return Object.fromEntries(header.map((key, i) => [key, values[i]]));
    });
  }

  function download(text, extension, type) {
    $('copy').value = text;
    $('fallback').hidden = false;
    try {
      const url = URL.createObjectURL(new Blob([text], {type}));
      const link = document.createElement('a');
      link.href = url;
      link.download = (assisted ? 'Exp14_review_AI_assisted_' : 'Exp14_review_') +
        new Date().toISOString().replace(/[:.]/g, '-') + '.' + extension;
      link.hidden = true;
      document.body.appendChild(link);
      link.click();
      setTimeout(() => { link.remove(); URL.revokeObjectURL(url); }, 1500);
      notice('已请求下载。请保存导出文件以便恢复或交接。');
    } catch {
      $('fallback').open = true;
      notice('下载失败，请复制导出内容后保存为文件。', true);
    }
  }

  function picture(container, filename, caption, assets = current.assets) {
    if (!filename) return;
    const figure = document.createElement('figure');
    const link = document.createElement('a');
    link.href = assets + filename;
    link.target = '_blank';
    link.rel = 'noopener';
    const image = document.createElement('img');
    image.src = link.href;
    image.alt = caption;
    const label = document.createElement('figcaption');
    label.textContent = caption;
    link.append(image);
    figure.append(link, label);
    container.append(figure);
  }

  function updateProgress() {
    const complete = [...decisions.values()].filter(row => row.decision !== 'UNREVIEWED').length;
    $('progress').max = decisions.size;
    $('progress').value = complete;
    $('count').textContent = complete + ' / ' + decisions.size + ' 已有状态（含历史决定与 HOLD）';
    if (assisted) $('count').textContent = 'AI 已初审 ' + bundle.ai_review.completed + ' / ' + decisions.size +
      ' · 本页人工确认或导入 ' + confirmations.size;
  }

  function activeRow() {
    const id = current.initial.review_id;
    return assisted ? drafts.get(id) || {...decisions.get(id)} : decisions.get(id);
  }

  function updateConfirmation() {
    const row = activeRow();
    const targetStates = Object.values(JSON.parse(row.polygon_decisions));
    $('whole').disabled = row.decision !== 'PASS' || targetStates.some(state => state !== 'PASS');
    $('region').disabled = !current.aligned || (row.decision !== 'PASS' && !targetStates.includes('PASS'));
    if ($('whole').disabled) row.whole_image_normal = '';
    if ($('region').disabled) row.b_region_review = '';
    $('whole').checked = row.whole_image_normal === 'YES';
    $('region').checked = row.b_region_review === 'CONFIRMED';
  }

  function saveChange() {
    updateConfirmation();
    updateProgress();
    renderAI();
    if (saveLocal()) notice(assisted ? '人工草稿已暂存。点击「确认人工决定」后才计入正式导出。' :
      '已在浏览器暂存。请定期导出。');
  }

  function editableRow() {
    const row = activeRow();
    if (assisted) drafts.set(row.review_id, row);
    return row;
  }

  function renderAI() {
    if (!assisted) return;
    $('ai-panel').hidden = !current;
    if (!current) return;
    const id = current.initial.review_id;
    const ai = current.ai;
    $('ai-state').textContent = ai ? 'AI_DRAFT · ' + labels[ai.ai_decision] : 'AI 尚未查看';
    $('ai-use').textContent = ai ? [ai.ai_reason, '把握程度 ' + ai.ai_confidence,
      'A 候选 ' + ai.ai_a_candidate, 'B 候选 ' + ai.ai_b_candidate,
      ai.needs_human_attention ? '重点复核' : ''].filter(Boolean).join(' · ') : '';
    $('ai-notes').textContent = ai?.ai_notes || '';
    $('ai-polygons').textContent = ai ? Object.entries(ai.ai_polygon_decisions)
      .map(([polygon, state]) => polygon + ' ' + state).join(' · ') : '';
    $('ai-evidence').hidden = !ai;
    if (ai) $('ai-evidence').href = ai.sheet_url;
    const existing = decisions.get(id);
    $('adopt-ai').disabled = !ai || existing.decision !== 'UNREVIEWED' || confirmations.has(id) || drafts.has(id);
    $('edit-ai').disabled = !ai || drafts.has(id);
    $('discard-draft').disabled = !drafts.has(id);
    $('manual-priority').checked = priorities.has(id);
    $('human-state').textContent = [
      confirmations.get(id) || (existing.decision !== 'UNREVIEWED' ? '原审核状态已保留' : '人工待确认'),
      labels[existing.decision], drafts.has(id) ? '有未确认草稿' : '',
    ].filter(Boolean).join(' · ');
  }

  function adoptAI(confirm) {
    const id = current.initial.review_id;
    const ai = current.ai;
    const row = {...decisions.get(id), decision: ai.ai_decision, reason: ai.ai_reason, notes: ai.ai_notes,
      whole_image_normal: '', b_region_review: '', polygon_decisions: JSON.stringify(ai.ai_polygon_decisions)};
    validate([row]);
    if (confirm) {
      decisions.set(id, row);
      confirmations.set(id, 'HUMAN_CONFIRMED');
      drafts.delete(id);
    } else drafts.set(id, row);
    render();
    if (saveLocal()) notice(confirm ? '已记录用户采纳。整图和区域确认可另行填写。' : '已载入人工草稿，可修改后确认。');
  }

  function renderPolygon() {
    $('crops').replaceChildren();
    const selected = current.polygons[Number($('polygon').value)];
    picture($('crops'), selected.before, '原图局部 · ' + selected.class);
    if (current.prior) picture($('crops'), current.prior.crops[selected.polygon_id], '旧 Aligned 同位置局部', current.prior.assets);
    picture($('crops'), selected.after, 'Aligned 同位置局部');
    const missing = selected.unsupported_target_pixels;
    $('target-support').textContent = missing === null ? '配准未验证，暂无同源局部' : '无效目标像素 ' + missing;
  }

  function renderDiagnostics() {
    const gallery = $('diagnostic-images');
    gallery.replaceChildren();
    if (!$('diagnostics').open || !current) return;
    picture(gallery, current.raw, 'Raw · 网关返回');
    if (current.aligned) {
      picture(gallery, 'hard.png', 'Hard · 诊断');
      picture(gallery, 'feather.png', 'Feather · 诊断');
      picture(gallery, 'valid.png', 'Valid Mask · 白色有效');
      picture(gallery, 'valid_target.png', '有效目标区域');
      if (current.bounded) picture(gallery, 'bounded.png', '限于待修 Mask 的源坐标输出');
    }
  }

  function render() {
    current = filtered[index] || null;
    $('empty').hidden = Boolean(current);
    for (const id of ['item', 'form', 'diagnostics']) $(id).hidden = !current;
    $('position').textContent = filtered.length ? (index + 1) + ' / ' + filtered.length : '0 / 0';
    $('prev').disabled = index === 0;
    $('next').disabled = index + 1 >= filtered.length;
    updateProgress();
    renderAI();
    if (!current) {
      for (const id of ['images', 'crops', 'diagnostic-images']) $(id).replaceChildren();
      return;
    }
    $('title').textContent = current.source_id + ' · ' + current.classes.join(' / ') + ' · ' + current.variant_id;
    const historical = [current.historical_visual, current.a2_status].filter(Boolean).join(' / ');
    $('geometry').textContent = [current.geometry_status, current.mapping_method,
      historical && '历史 ' + historical, current.geometry_reason, current.source_hold && 'HOLD · ' + current.source_hold]
      .filter(Boolean).join(' · ');
    if (current.aligned) $('geometry').textContent +=
      ' · 有效覆盖 ' + (current.valid_coverage * 100).toFixed(1) + '%';
    $('images').replaceChildren();
    const prior = current.prior;
    if (!prior) picture($('images'), current.original, 'Original 原图');
    picture($('images'), 'polygon_overlay.png', 'Original + Polygon');
    if (prior) {
      const priorRaw = current.rescue || !prior.aligned;
      picture($('images'), priorRaw ? prior.raw : prior.image,
        prior.variant_id + (priorRaw ? ' · 旧 Raw' : ' · 旧 Aligned'), prior.assets);
    }
    picture($('images'), current.aligned ? 'aligned.png' : current.raw,
      current.aligned ? 'Aligned 修复图' : 'Raw · 未配准');
    if (current.bounded) picture($('images'), 'bounded.png', 'V2 · 限定编辑');
    $('original-link-row').hidden = !prior;
    if (prior) $('original-link').href = current.assets + current.original;
    $('prior-notes').hidden = !prior;
    $('prior-notes').textContent = prior ? prior.notes.map(row => [
      row.variant_id, '原人工 ' + labels[row.decision], row.reason, row.notes,
      Object.entries(row.targets).map(([id, state]) => id + ' ' + labels[state]).join(' · '),
    ].filter(Boolean).join(' · ')).join('\n') : '';
    $('risk').textContent = [
      '相机 UI ' + current.ui,
      current.unsupported_bright_original_pixels > 0 ? '无效区覆盖原图可见像素，请检查机械体和边缘' : '',
      current.model_metadata === 'DIFFERENT' ? '网关返回模型字段 ' + current.reported_model : '',
      current.historical_support, current.source_hold,
    ].filter(Boolean).join(' · ');
    $('geometry-file').href = current.assets + 'geometry.json';
    $('diagnostics').open = false;
    renderDiagnostics();
    const row = activeRow();
    $('decision').value = row.decision;
    $('reason').value = row.reason;
    $('notes').value = row.notes;
    $('polygon').replaceChildren();
    $('polygon-decisions').replaceChildren();
    $('polygon-notes').open = false;
    const existing = JSON.parse(row.polygon_decisions);
    current.polygons.forEach((target, i) => {
      option($('polygon'), String(i), target.polygon_id + ' · ' + target.class);
      const label = document.createElement('label');
      label.className = 'control';
      label.textContent = target.polygon_id;
      const select = document.createElement('select');
      option(select, '', '沿用整体决定');
      for (const [value, name] of Object.entries(labels)) if (value !== 'UNREVIEWED') option(select, value, name);
      select.value = existing[target.polygon_id] || '';
      select.addEventListener('change', () => {
        const row = editableRow();
        const states = JSON.parse(row.polygon_decisions);
        if (select.value) states[target.polygon_id] = select.value;
        else delete states[target.polygon_id];
        row.polygon_decisions = JSON.stringify(states);
        saveChange();
      });
      label.append(select);
      $('polygon-decisions').append(label);
    });
    updateConfirmation();
    renderPolygon();
  }

  function matchingRecords() {
    const category = $('class').value;
    const source = $('source').value.trim();
    const status = $('status-filter').value;
    const ai = assisted ? $('ai-filter').value : '';
    const confidence = assisted ? $('confidence-filter').value : '';
    const confirmation = assisted ? $('confirmation-filter').value : '';
    return bundle.records.filter(record => (!category || record.classes.includes(category)) &&
      (!source || record.source_id.includes(source)) &&
      (!status || decisions.get(record.initial.review_id).decision === status) &&
      (!ai || (record.ai?.ai_decision || 'NOT_REVIEWED') === ai) &&
      (!confidence || record.ai?.ai_confidence === confidence) &&
      (!confirmation || (confirmation === 'confirmed') === (confirmations.has(record.initial.review_id) ||
        decisions.get(record.initial.review_id).decision !== 'UNREVIEWED')));
  }

  function applyFilter(reset = true) {
    const id = current?.initial.review_id;
    filtered = matchingRecords();
    const existing = reset ? -1 : filtered.findIndex(record => record.initial.review_id === id);
    index = existing < 0 ? 0 : existing;
    render();
  }

  for (const category of bundle.class_names) option($('class'), category);
  for (const [value, label] of Object.entries(labels)) {
    option($('decision'), value, label);
    option($('status-filter'), value, label);
  }
  for (const reason of bundle.reasons) option($('reason'), reason, reason || '未选择');
  for (const id of ['class', 'source', 'status-filter']) $(id).addEventListener('input', () => applyFilter());
  if (assisted) {
    for (const id of ['ai-filters', 'ai-actions', 'human-state']) $(id).hidden = false;
    for (const value of ['PASS', 'RETRY_REPAIR', 'RETRY_ALIGNMENT', 'HOLD', 'NOT_REVIEWED']) {
      option($('ai-filter'), value, labels[value] || 'AI 尚未查看');
    }
    for (const value of ['HIGH', 'MEDIUM', 'LOW']) option($('confidence-filter'), value);
    for (const id of ['ai-filter', 'confidence-filter', 'confirmation-filter']) {
      $(id).addEventListener('input', () => applyFilter());
    }
    $('next-priority').addEventListener('click', () => {
      const other = filtered.slice(index + 1).concat(filtered.slice(0, index));
      const match = other.find(r => (r.ai?.needs_human_attention || priorities.has(r.initial.review_id)) &&
        !confirmations.has(r.initial.review_id));
      if (match) { index = filtered.indexOf(match); render(); }
      else notice('当前筛选下没有其他待处理的重点记录。');
    });
    $('adopt-ai').addEventListener('click', () => adoptAI(true));
    $('edit-ai').addEventListener('click', () => adoptAI(false));
    $('confirm-human').addEventListener('click', () => {
      const row = activeRow();
      try {
        if (row.decision === 'UNREVIEWED') throw new Error('请先选择人工决定。');
        validate([row]);
        decisions.set(row.review_id, {...row});
        confirmations.set(row.review_id, 'HUMAN_OVERRIDDEN');
        drafts.delete(row.review_id);
        render();
        if (saveLocal()) notice('已确认人工决定。');
      } catch (error) { notice(error.message, true); }
    });
    $('discard-draft').addEventListener('click', () => {
      drafts.delete(current.initial.review_id);
      render();
      saveLocal();
    });
    $('manual-priority').addEventListener('change', () => {
      const id = current.initial.review_id;
      if ($('manual-priority').checked) priorities.add(id);
      else priorities.delete(id);
      saveLocal();
    });
  }
  for (const [id, delta] of [['prev', -1], ['next', 1]]) $(id).addEventListener('click', () => {
    const visible = current?.initial.review_id;
    filtered = matchingRecords();
    const oldIndex = filtered.findIndex(record => record.initial.review_id === visible);
    index = oldIndex >= 0 ? oldIndex + delta : index + Math.min(0, delta);
    index = Math.min(Math.max(index, 0), Math.max(filtered.length - 1, 0));
    render();
  });
  $('polygon').addEventListener('change', renderPolygon);
  $('diagnostics').addEventListener('toggle', renderDiagnostics);
  for (const id of ['decision', 'reason', 'notes']) $(id).addEventListener('input', () => {
    editableRow()[id] = $(id).value;
    saveChange();
  });
  $('whole').addEventListener('change', () => {
    editableRow().whole_image_normal = $('whole').checked ? 'YES' : '';
    saveChange();
  });
  $('region').addEventListener('change', () => {
    editableRow().b_region_review = $('region').checked ? 'CONFIRMED' : '';
    saveChange();
  });
  $('csv').addEventListener('click', () => download(csvText(), 'csv', 'text/csv;charset=utf-8'));
  $('json').addEventListener('click', () => download(JSON.stringify(snapshot(), null, 2) + '\n', 'json', 'application/json'));
  $('import').addEventListener('click', () => $('file').click());
  $('file').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const payload = file.name.toLowerCase().endsWith('.json') ? JSON.parse(text.replace(/^\ufeff/, '')) :
        {phase: bundle.phase, records: parseCsv(text)};
      const imported = restore(payload);
      applyFilter(false);
      if (saveLocal()) notice('已恢复 ' + imported + ' 条审核决定。');
    } catch (error) {
      notice('导入失败。' + error.message, true);
    } finally {
      event.target.value = '';
    }
  });

  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const payload = JSON.parse(saved);
      if (payload.phase === bundle.phase && Array.isArray(payload.records)) {
        const currentRows = payload.records.filter(row => byId.has(row.review_id));
        if (currentRows.length) notice('已从本地缓存恢复 ' + restore({...payload, records: currentRows}) + ' 条记录。');
      }
    }
  } catch {
    notice('本地缓存不可用，请导入已保存的 CSV 或 JSON。', true);
  }
  applyFilter();
})();
