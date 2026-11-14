/* Exercise the A9 file:// audit page against the private candidate package. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {fileURLToPath, pathToFileURL} = require('node:url');
const {chromium} = require(process.argv[3]);

async function test() {
  const [, , directory, , receipt] = process.argv;
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  const network = [], errors = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (/^https?:/.test(request.url())) network.push(request.url());
    });
    await page.goto(pathToFileURL(path.join(directory, 'index.html')).href);
    const bundle = await page.evaluate(() => JSON.parse(document.querySelector('#bundle').textContent));
    assert.equal(bundle.records.length, 523);
    assert.equal(new Set(bundle.records.map(r => r.id)).size, 523);
    assert.equal(bundle.records.reduce((n, row) => n + row.variants.length, 0), 956);
    const href = page.url();
    const files = new Set(bundle.records.flatMap(r => [r.original_url, ...r.variants.flatMap(
      v => [...Object.values(v.files), ...Object.values(v.polygon_files).flatMap(Object.values)].map(f => f.url))]).filter(Boolean));
    for (const file of files) await fs.access(fileURLToPath(new URL(file, href)));
    const counts = {
      a: bundle.records.filter(r => r.a).length,
      b: bundle.records.filter(r => r.b.length).length,
      both: bundle.counts.A_AND_B,
      excluded: bundle.counts.NEITHER,
      choice: bundle.counts.version_selection_queue,
      conflict: bundle.counts.version_conflict_sources,
      label: bundle.counts.a_label_hold_sources,
      geometry: bundle.records.filter(r => r.source.b_geometry_risk).length,
    };
    for (const [filter, count] of Object.entries(counts)) {
      await page.locator('#filter').selectOption(filter);
      assert.match(await page.locator('#position').innerText(), new RegExp(' / ' + count + '$'));
    }
    await page.locator('#filter').selectOption('all');
    const samples = ['1', '18', '37', '206', '238', '348', '914',
      bundle.records.find(r => r.a?.version === 'V1').id,
      bundle.records.find(r => r.a?.version === 'V4').id];
    for (const id of new Set(samples)) {
      const row = bundle.records.find(r => r.id === id);
      await page.locator('#source').fill(id);
      assert.match(await page.locator('#heading').innerText(), new RegExp('^' + id + ' '));
      for (const [i, variant] of row.variants.entries()) {
        await page.locator('#polygon').selectOption('');
        await page.locator('#version').selectOption(String(i));
        await page.waitForFunction(count => {
          const files = [...document.querySelectorAll('#images img')];
          return files.length === count && files.every(img => img.complete && img.naturalWidth > 0);
        }, Number(Boolean(row.original_url)) + Object.keys(variant.files).length);
        await page.locator('#polygon').selectOption(row.source.selected_polygon_ids[0]);
        const crop = variant.polygon_files[row.source.selected_polygon_ids[0]];
        if (crop) await page.waitForFunction(count => {
          const images = [...document.querySelectorAll('#images img')];
          return images.length === count && images.every(img => img.complete && img.naturalWidth > 0);
        }, Object.keys(crop).length);
      }
    }
    await page.locator('#source').fill('348');
    assert.match(await page.locator('#risk').innerText(), /MECHANICAL_STRUCTURE_ERROR/);
    await page.locator('#source').fill('206');
    assert.match(await page.locator('#risk').innerText(), /HUMAN_UNREVIEWED/);
    assert.equal(await page.locator('textarea, input[type=file]').count(), 0);
    assert.equal(await page.evaluate(() => localStorage.length), 0);
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    const result = {browser: 'Chrome', sources: bundle.records.length, review_rows: 956,
      all_image_paths_accessible: files.size, samples: new Set(samples).size,
      filters: counts, offline: true, read_only: true};
    await fs.writeFile(receipt, JSON.stringify(result, null, 2) + '\n', {flag:'wx'});
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}
test().catch(error => { console.error(error); process.exitCode = 1; });
