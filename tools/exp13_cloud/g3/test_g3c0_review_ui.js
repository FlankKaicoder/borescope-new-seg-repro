const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const uiSource = fs.readFileSync(new URL('./review_ui.js', `file://${__dirname.replaceAll('\\', '/')}/`), 'utf8');

class Element {
  constructor(dataset = {}) { this.dataset = dataset; this.style = {}; this.value = ''; this.textContent = ''; this.listeners = {}; this.children = []; this.hidden = false; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  async dispatch(name, event = {}) { return this.listeners[name]?.({ target: this, ...event }); }
  appendChild(child) { this.children.push(child); child.parent = this; return child; }
  append(child) { return this.appendChild(child); }
  remove() { this.removed = true; }
  click() { this.clicked = true; }
  focus() {}
  select() { this.selected = true; }
  replaceWith(child) { this.replaced = child; }
  querySelector(selector) {
    if (selector.includes('data-field="visual"')) return this.visual;
    if (selector.includes('data-field="notes"')) return this.notes;
    if (selector.includes('data-field="geometry"')) return this.geometry;
    if (selector === 'select') return this.geometrySelect;
    return null;
  }
  querySelectorAll(selector) {
    if (selector === 'label') return [this.geometryLabel];
    if (selector.includes('data-field')) return [this.visual, this.notes, this.geometrySelect];
    return [];
  }
}

const ids = Array.from({ length: 36 }, (_, i) => String(i + 1));
const geometries = Object.fromEntries(ids.map((id, i) => [id, i < 28 ? 'GEOMETRY_CANDIDATE' : 'ALIGNMENT_UNVERIFIED']));
const store = new Map();
let blockedStorage = false;
let blockedClipboard = false;
let timers = [];
let lastBlob;
let links = [];
let copied = '';

function createPage(initialRows = []) {
  const cards = ids.map((id, i) => {
    const card = new Element({ source: id, class: 'Burn' });
    card.visual = new Element({ field: 'visual' }); card.visual.value = 'UNREVIEWED';
    card.notes = new Element({ field: 'notes' });
    card.geometrySelect = new Element({ field: 'geometry' }); card.geometrySelect.value = geometries[id];
    card.geometryLabel = new Element(); card.geometryLabel.textContent = 'Geometry';
    card.geometryLabel.querySelector = selector => selector === 'select' ? card.geometrySelect : null;
    return card;
  });
  const elements = Object.fromEntries(['#save-status', '#csv-copy', '#count', '#download', '#import', '#file', '#copy', '#filter', '#search'].map(x => [x, new Element()]));
  elements['#file'].click = () => { elements['#file'].clicked = true; };
  elements['#csv-copy'].focus = () => {}; elements['#csv-copy'].select = () => { elements['#csv-copy'].selected = true; };
  const document = {
    body: new Element(),
    querySelectorAll: selector => selector === 'article[data-source]' ? cards : [],
    querySelector: selector => elements[selector],
    createElement: tag => {
      const element = new Element(); element.tagName = tag;
      if (tag === 'a') links.push(element);
      return element;
    }
  };
  const context = {
    document, Blob: class { constructor(parts, options) { this.parts = parts; this.options = options; lastBlob = this; } },
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL: value => { context.revoked = value; } },
    setTimeout: fn => { timers.push(fn); },
    localStorage: { setItem(k, v) { if (blockedStorage) throw Error('blocked'); store.set(k, v); }, getItem(k) { if (blockedStorage) throw Error('blocked'); return store.get(k) ?? null; } },
    navigator: { clipboard: { writeText: async value => { if (blockedClipboard) throw Error('blocked'); copied = value; } } },
    console, Map, Set, Object, String, Error, JSON
  };
  vm.runInNewContext(uiSource.replace('__INITIAL_REVIEW__', JSON.stringify(initialRows)), context);
  return { cards, elements, context };
}

function csv(rows) {
  const fields = ['source_id', 'visual_review_status', 'geometry_status', 'review_notes'];
  const cell = value => `"${String(value).replaceAll('"', '""')}"`;
  return '\ufeff' + fields.join(',') + '\r\n' + rows.map(r => fields.map(f => cell(r[f])).join(',')).join('\r\n') + '\r\n';
}

(async () => {
  const page = createPage();
  assert.equal(page.cards.length, 36, 'all source cards rendered');
  const recovered = ids.map((id, i) => ({ source_id: id, visual_review_status: i < 14 ? 'VISUAL_PASS' : 'UNREVIEWED', geometry_status: geometries[id], review_notes: i === 0 ? '中文备注, 含逗号，和"引号"\n第二行' : '' }));
  page.elements['#file'].files = [{ text: async () => csv(recovered) }];
  await page.elements['#file'].dispatch('change');
  assert.equal(page.cards[0].notes.value, recovered[0].review_notes, 'Chinese multiline notes restored exactly');
  assert.equal(page.cards[0].visual.value, 'VISUAL_PASS');
  assert.match(page.elements['#count'].textContent, /^14 \/ 36/);
  assert.match(page.elements['#save-status'].textContent, /Imported all 36/);

  store.clear();
  const seeded = createPage(recovered);
  assert.equal(seeded.cards[0].visual.value, 'VISUAL_PASS', 'embedded recovery appears on first open');
  assert.equal(seeded.cards[0].notes.value, recovered[0].review_notes);
  assert.match(seeded.elements['#save-status'].textContent, /Restored 36 records/);
  seeded.cards[0].notes.value = 'newer saved change';
  await seeded.cards[0].notes.dispatch('input');
  const seededRefresh = createPage(recovered);
  assert.equal(seededRefresh.cards[0].notes.value, 'newer saved change', 'new local changes take precedence over embedded recovery on refresh');

  page.cards[1].notes.value = 'edited, "quoted"\nline';
  await page.cards[1].notes.dispatch('input');
  assert.ok(store.size, 'autosave wrote localStorage');
  const refreshed = createPage();
  assert.equal(refreshed.cards[0].visual.value, 'VISUAL_PASS', 'refresh restored saved state');
  assert.equal(refreshed.cards[1].notes.value, 'edited, "quoted"\nline');

  await page.elements['#download'].dispatch('click');
  assert.equal(links.at(-1)?.clicked, true, `download link clicked: ${page.elements['#save-status'].textContent}; links=${links.length}`);
  assert.ok(links.at(-1).parent, 'link was attached to document body');
  assert.equal(page.context.revoked, undefined, 'Blob URL was not immediately revoked');
  assert.equal(timers.length, 1, 'Blob release was deferred');
  timers.shift()(); assert.equal(page.context.revoked, 'blob:test');
  assert.ok(lastBlob.parts[0].startsWith('\ufeff'), 'CSV has UTF-8 BOM');

  const exported = recovered.map((row, i) => i === 1 ? { ...row, review_notes: 'edited, "quoted"\nline' } : row);
  const exportedText = csv(exported);
  page.elements['#file'].files = [{ text: async () => exportedText }];
  await page.elements['#file'].dispatch('change');
  assert.equal(page.cards[1].notes.value, 'edited, "quoted"\nline', 'CSV round trip retained escaped multiline notes');

  const before = page.cards[0].visual.value;
  page.elements['#file'].files = [{ text: async () => csv([...recovered.slice(0, 35), recovered[0]]) }];
  await page.elements['#file'].dispatch('change');
  assert.equal(page.cards[0].visual.value, before, 'duplicate source was rejected transactionally');
  assert.match(page.elements['#save-status'].textContent, /Import rejected/);
  page.elements['#file'].files = [{ text: async () => csv(recovered.slice(1)) }];
  await page.elements['#file'].dispatch('change');
  assert.equal(page.cards[0].visual.value, before, 'missing source was rejected transactionally');
  const geometryMismatch = recovered.map((row, i) => i === 0 ? { ...row, geometry_status: 'ALIGNMENT_UNVERIFIED' } : row);
  page.elements['#file'].files = [{ text: async () => csv(geometryMismatch) }];
  await page.elements['#file'].dispatch('change');
  assert.equal(page.cards[0].visual.value, before, 'frozen geometry mismatch was rejected transactionally');

  blockedClipboard = true;
  await page.elements['#copy'].dispatch('click');
  assert.equal(page.elements['#csv-copy'].selected, true, 'copy fallback selects complete CSV');
  assert.match(page.elements['#save-status'].textContent, /selected below/);
  blockedStorage = true;
  const noStorage = createPage();
  noStorage.cards[0].visual.value = 'VISUAL_FAIL';
  await noStorage.cards[0].visual.dispatch('change');
  assert.match(noStorage.elements['#save-status'].textContent, /Auto-save unavailable/);
  await noStorage.elements['#download'].dispatch('click');
  assert.ok(lastBlob.parts[0].includes('VISUAL_FAIL'), 'CSV export works when localStorage is blocked');
  console.log('G3C0 review UI tests: 10/10 PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
