/* Injected into the private offline G3-C0 review page. */
(() => {
  const cards = [...document.querySelectorAll('article[data-source]')];
  const sourceIds = cards.map(card => card.dataset.source);
  const fields = ['source_id', 'visual_review_status', 'geometry_status', 'review_notes'];
  const validVisual = new Set(['UNREVIEWED', 'VISUAL_PASS', 'VISUAL_FAIL', 'NEEDS_RETRY']);
  const initial = __INITIAL_REVIEW__;
  const storageKey = 'exp13-g3c0-review-v2';
  const status = document.querySelector('#save-status');
  const output = document.querySelector('#csv-copy');

  function geometryOf(card) {
    const label = [...card.querySelectorAll('label')].find(x => x.textContent.startsWith('Geometry'));
    return label ? label.querySelector('select')?.value || label.querySelector('[data-field="geometry"]')?.textContent : '';
  }

  function rowOf(card) {
    return {
      source_id: card.dataset.source,
      visual_review_status: card.querySelector('[data-field="visual"]').value,
      geometry_status: card.dataset.geometry,
      review_notes: card.querySelector('[data-field="notes"]').value
    };
  }

  function setRow(card, row) {
    card.querySelector('[data-field="visual"]').value = row.visual_review_status;
    card.querySelector('[data-field="notes"]').value = row.review_notes;
  }

  function snapshot() { return cards.map(rowOf); }
  function updateCount() {
    const done = cards.filter(card => card.querySelector('[data-field="visual"]').value !== 'UNREVIEWED').length;
    document.querySelector('#count').textContent = `${done} / ${cards.length} reviewed`;
  }

  function saveLocal() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(snapshot()));
      status.textContent = 'Saved locally';
      status.dataset.kind = 'ok';
      return true;
    } catch (error) {
      status.textContent = 'Auto-save unavailable; export CSV';
      status.dataset.kind = 'error';
      return false;
    }
  }

  function csvCell(value) { return `"${String(value ?? '').replaceAll('"', '""')}"`; }
  function makeCsv(rows) {
    return '\ufeff' + fields.join(',') + '\r\n' + rows.map(row => fields.map(field => csvCell(row[field])).join(',')).join('\r\n') + '\r\n';
  }

  function parseCsv(text) {
    text = text.replace(/^\ufeff/, '');
    const records = [];
    let row = [], cell = '', quoted = false, closedQuote = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (ch === '"') { quoted = false; closedQuote = true; }
        else cell += ch;
      } else if (closedQuote && ch === ',') { row.push(cell); cell = ''; closedQuote = false; }
      else if (closedQuote && (ch === '\n' || ch === '\r')) {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(cell); cell = ''; closedQuote = false;
        if (row.some(value => value !== '')) records.push(row);
        row = [];
      } else if (closedQuote) throw new Error('CSV contains unexpected text after a quoted field.');
      else if (ch === '"' && cell === '') quoted = true;
      else if (ch === '"') throw new Error('CSV contains an unexpected quote in an unquoted field.');
      else if (ch === ',') { row.push(cell); cell = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(cell); cell = '';
        if (row.some(value => value !== '')) records.push(row);
        row = [];
      } else cell += ch;
    }
    if (quoted) throw new Error('CSV contains an unterminated quoted field.');
    if (cell !== '' || row.length) { row.push(cell); if (row.some(value => value !== '')) records.push(row); }
    if (!records.length) throw new Error('CSV is empty.');
    const header = records.shift();
    const index = Object.fromEntries(header.map((name, i) => [name.trim(), i]));
    for (const field of fields) if (!(field in index)) throw new Error(`CSV is missing required field: ${field}`);
    return records.map(values => Object.fromEntries(fields.map(field => [field, values[index[field]] ?? ''])));
  }

  function validate(rows) {
    if (rows.length !== sourceIds.length) throw new Error(`Expected ${sourceIds.length} records; found ${rows.length}.`);
    const byId = new Map();
    for (const row of rows) {
      if (byId.has(row.source_id)) throw new Error(`Duplicate source_id: ${row.source_id}`);
      byId.set(row.source_id, row);
    }
    const missing = sourceIds.filter(id => !byId.has(id));
    const extra = [...byId.keys()].filter(id => !sourceIds.includes(id));
    if (missing.length || extra.length) throw new Error('CSV source_id set does not match this 36-source page.');
    for (const card of cards) {
      const row = byId.get(card.dataset.source);
      if (!validVisual.has(row.visual_review_status)) throw new Error(`Invalid Visual status for ${row.source_id}.`);
      if (row.geometry_status !== card.dataset.geometry) throw new Error(`Geometry differs from frozen diagnostics for ${row.source_id}.`);
      if (typeof row.review_notes !== 'string') throw new Error(`Invalid Notes value for ${row.source_id}.`);
    }
    return sourceIds.map(id => byId.get(id));
  }

  function apply(rows) {
    rows.forEach((row, index) => setRow(cards[index], row));
    updateCount();
    saveLocal();
  }

  function report(message, error = false) {
    status.textContent = message;
    status.dataset.kind = error ? 'error' : 'ok';
  }

  function exportCsv() {
    const text = makeCsv(snapshot());
    output.value = text;
    try {
      const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'g3c0_review.csv';
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      setTimeout(() => { link.remove(); URL.revokeObjectURL(url); }, 1500);
      report('CSV export requested. If no file appears, use Copy CSV content below.');
    } catch (error) {
      report('Download unavailable. Use Copy CSV content below.', true);
    }
  }

  cards.forEach(card => {
    const geometryLabel = [...card.querySelectorAll('label')].find(x => x.textContent.startsWith('Geometry'));
    const geometrySelect = geometryLabel?.querySelector('select');
    if (geometrySelect) {
      const geometry = geometrySelect.value;
      const frozenGeometry = document.createElement('output');
      frozenGeometry.textContent = geometry;
      frozenGeometry.dataset.field = 'geometry';
      geometrySelect.replaceWith(frozenGeometry);
      card.dataset.geometry = geometry;
    }
    card.querySelectorAll('[data-field="visual"], [data-field="notes"]').forEach(control => {
      control.addEventListener('input', () => { updateCount(); saveLocal(); });
      control.addEventListener('change', () => { updateCount(); saveLocal(); });
    });
  });

  updateCount();
  let restored = false, storageFailed = false;
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) { apply(validate(JSON.parse(saved))); report('Restored the last locally saved review state.'); restored = true; }
  } catch (error) { report('Auto-save unavailable; export CSV', true); storageFailed = true; }
  if (!restored && initial && initial.length) {
    try { apply(validate(initial)); report(`Restored ${initial.length} records from the provided review CSV.`); restored = true; }
    catch (error) { report(`Embedded recovery data rejected: ${error.message}`, true); }
  }
  if (!restored && !storageFailed) report('No saved review state. Changes will be saved locally when available.');

  document.querySelector('#download').addEventListener('click', exportCsv);
  document.querySelector('#import').addEventListener('click', () => document.querySelector('#file').click());
  document.querySelector('#file').addEventListener('change', async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const rows = validate(parseCsv(await file.text()));
      apply(rows);
      report(`Imported all ${rows.length} review records.`);
    } catch (error) { report(`Import rejected; no records changed. ${error.message}`, true); }
    event.target.value = '';
  });
  document.querySelector('#copy').addEventListener('click', async () => {
    output.value = makeCsv(snapshot());
    try {
      await navigator.clipboard.writeText(output.value);
      report('CSV copied. Paste it into a text editor and save as g3c0_review.csv.');
    } catch (error) {
      output.focus(); output.select();
      report('Clipboard access blocked. CSV is selected below; copy it manually and save as g3c0_review.csv.', true);
    }
  });
  document.querySelector('#filter').addEventListener('change', filter);
  document.querySelector('#search').addEventListener('input', filter);
  function filter() {
    const category = document.querySelector('#filter').value;
    const search = document.querySelector('#search').value.toLowerCase();
    for (const card of cards) card.hidden = (category && card.dataset.class !== category) || !card.dataset.source.toLowerCase().includes(search);
  }
})();
