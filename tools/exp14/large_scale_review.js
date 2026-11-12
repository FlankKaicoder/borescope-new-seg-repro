/* Offline Exp14 review. CSV quoting and delayed Blob release follow the G3-C0 recovery UI. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const bundle = JSON.parse($('bundle').textContent);
  const byId = new Map(bundle.records.map(record => [record.initial.review_id, record]));
  const storageKey = bundle.phase + ':review:v1';
  const labels = {
    UNREVIEWED: '待审核', PASS: 'PASS · 修复成功', RETRY_REPAIR: 'RETRY_REPAIR · 重修',
    RETRY_ALIGNMENT: 'RETRY_ALIGNMENT · 重新配准', HOLD: 'HOLD · 暂缓',
  };
  const decisions = new Map(bundle.records.map(record => [record.initial.review_id, {...record.initial}]));
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
    return {phase: bundle.phase, records: [...decisions.values()]};
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
    for (const [id, row] of staged) decisions.set(id, row);
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
      link.download = 'Exp14_review_' + new Date().toISOString().replace(/[:.]/g, '-') + '.' + extension;
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

  function picture(container, filename, caption) {
    if (!filename) return;
    const figure = document.createElement('figure');
    const link = document.createElement('a');
    link.href = current.assets + filename;
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
  }

  function activeRow() { return decisions.get(current.initial.review_id); }

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
    if (saveLocal()) notice('已在浏览器暂存。请定期导出。');
  }

  function renderPolygon() {
    $('crops').replaceChildren();
    const selected = current.polygons[Number($('polygon').value)];
    picture($('crops'), selected.before, '原图局部 · ' + selected.class);
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
    picture($('images'), current.original, 'Original 原图');
    picture($('images'), 'polygon_overlay.png', 'Original + Polygon');
    picture($('images'), current.aligned ? 'aligned.png' : current.raw, current.aligned ? 'Aligned 修复图' : 'Raw · 未配准');
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
        const states = JSON.parse(activeRow().polygon_decisions);
        if (select.value) states[target.polygon_id] = select.value;
        else delete states[target.polygon_id];
        activeRow().polygon_decisions = JSON.stringify(states);
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
    return bundle.records.filter(record => (!category || record.classes.includes(category)) &&
      (!source || record.source_id.includes(source)) &&
      (!status || decisions.get(record.initial.review_id).decision === status));
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
    activeRow()[id] = $(id).value;
    saveChange();
  });
  $('whole').addEventListener('change', () => {
    activeRow().whole_image_normal = $('whole').checked ? 'YES' : '';
    saveChange();
  });
  $('region').addEventListener('change', () => {
    activeRow().b_region_review = $('region').checked ? 'CONFIRMED' : '';
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
