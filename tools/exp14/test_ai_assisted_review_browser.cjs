/* Chrome integration against the private assisted page. QA exports are synthetic. */
const {chromium} = require(process.argv[3]);
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

async function test() {
  const [, , directory, , qa, mode, previousCsv, previousDirectory] = process.argv;
  const supplemental = mode === 'a7b';
  const finalRound = mode === 'a8';
  const targeted = mode === 'a6' || mode === 'a7' || supplemental || finalRound;
  await fs.mkdir(qa);
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  const errors = [];
  const network = [];
  try {
    const open = async () => {
      const context = await browser.newContext({acceptDownloads: true});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      page.on('request', req => { if (/^https?:/.test(req.url())) network.push(req.url()); });
      await page.goto(pathToFileURL(path.join(directory, 'index.html')).href);
      return {page, context};
    };
    const {page, context} = await open();
    const selection = await page.evaluate(() => {
      const records = JSON.parse(document.querySelector('#bundle').textContent).records;
      return {
        pass: records.find(r => r.initial.decision === 'UNREVIEWED' && r.ai?.ai_decision === 'PASS'),
        historical: records.find(r => r.initial.decision === 'PASS' && r.ai),
        partial: records.find(r => r.initial.decision === 'UNREVIEWED' && r.ai?.ai_decision === 'RETRY_REPAIR' && r.polygons.length > 1),
      };
    });
    assert(selection.pass && selection.partial && (targeted || selection.historical));
    const show = async (tab, item) => {
      await tab.locator('#source').fill(item.source_id);
      while (!(await tab.locator('#title').innerText()).endsWith(' · ' + item.variant_id)) {
        if (!(await tab.locator('#next').isEnabled())) assert.fail('Variant was filtered out');
        await tab.locator('#next').click();
      }
    };
    if (mode === 'a7' || supplemental || finalRound) {
      const records = await page.evaluate(() => JSON.parse(document.querySelector('#bundle').textContent).records);
      const loadedImages = async () => page.waitForFunction(() => {
        const images = [...document.querySelectorAll('#images img, #crops img, #diagnostic-images img')];
        return images.length > 0 && images.every(img => img.complete && img.naturalWidth > 0);
      });
      for (const record of records) {
        await show(page, record);
        await page.locator('#diagnostics summary').click();
        for (const index of record.polygons.keys()) {
          await page.locator('#polygon').selectOption(String(index));
          await loadedImages();
        }
        const evidence = await page.locator('#ai-evidence').getAttribute('href');
        assert.ok(evidence);
        await fs.access(path.resolve(directory, evidence));
      }
      console.log('Chrome loaded the comparison and diagnostic images for', records.length, mode, 'records');
    }
    const save = async (id, name) => {
      const pending = page.waitForEvent('download');
      await page.locator(id).click();
      const receipt = await pending;
      if (supplemental) assert.match(receipt.suggestedFilename(), /^Exp14_A7B_CSV-B_/);
      if (finalRound) assert.match(receipt.suggestedFilename(), /^Exp14_A8_FINAL_V4_/);
      const file = path.join(qa, name);
      await receipt.saveAs(file);
      return file;
    };
    if (!targeted) {
      await show(page, selection.historical);
      assert.equal(await page.locator('#adopt-ai').isDisabled(), true);
      assert.equal(await page.locator('#decision').inputValue(), 'PASS');
    }
    await show(page, selection.pass);
    const pass = selection.pass;
    const firstPolygon = pass.polygons[0];
    const previous = pass.comparison?.length ? pass.comparison.filter(item => item.assets) : [pass.prior];
    const earlierImages = previous.reduce((count, item) => count +
      (pass.rescue && pass.comparison?.length && item.aligned ? 2 : 1), 0);
    const expected = targeted ? 3 + Number(pass.bounded) + earlierImages +
      Number(!!firstPolygon.after) + previous.filter(item => item.crops[firstPolygon.polygon_id]).length +
      Number(!!(pass.rescue && pass.comparison?.length && pass.aligned)) : 5;
    await page.waitForFunction(count => {
      const images = [...document.querySelectorAll('#images img, #crops img')];
      return images.length === count && images.every(img => img.complete && img.naturalWidth > 0);
    }, expected);
    if (targeted) {
      assert.match(await page.locator('#prior-notes').innerText(), /原人工/);
      if (supplemental) {
        assert.match(await page.locator('#review-id').innerText(), new RegExp(pass.initial.review_id + '$'));
      }
      assert.equal(await page.locator('#whole').isChecked(), false);
      if (previousCsv) {
        await page.locator('#file').setInputFiles(previousCsv);
        await page.waitForFunction(() => document.querySelector('#notice').dataset.error === 'true');
        assert.equal(await page.locator('#decision').inputValue(), 'UNREVIEWED');
      }
    }
    assert.match(await page.locator('#ai-state').innerText(), /AI_DRAFT/);
    assert.equal(await page.locator('#decision').inputValue(), 'UNREVIEWED');
    await page.locator('#adopt-ai').click();
    assert.match(await page.locator('#human-state').innerText(), /HUMAN_CONFIRMED/);
    assert.equal(await page.locator('#whole').isChecked(), false);
    assert.equal(await page.locator('#region').isChecked(), false);
    await show(page, selection.partial);
    await page.locator('#edit-ai').click();
    const note = '仅浏览器测试，破边 "复核"\n第二行，局部多目标。';
    await page.locator('#notes').fill(note);
    await page.locator('#polygon-notes summary').click();
    await page.locator('#polygon-decisions select').first().selectOption('HOLD');
    const olderChoice = finalRound ? selection.partial.final_versions.find(v => v !== selection.partial.variant_id) : '';
    if (finalRound) {
      assert.ok(olderChoice);
      assert.match(await page.locator('#final-suggestion').innerText(), /最终版本建议/);
      await page.locator('#final-version').selectOption(olderChoice);
    }
    const before = JSON.parse(await fs.readFile(await save('#json', 'QA_ONLY_before_confirmation.json'), 'utf8'));
    const id = selection.partial.initial.review_id;
    assert.deepEqual(before.records.find(row => row.review_id === id), selection.partial.initial);
    assert.equal(before.ai_workspace.drafts.find(row => row.review_id === id).notes, note);
    await page.reload();
    await show(page, selection.partial);
    assert.equal(await page.locator('#notes').inputValue(), note);
    await page.locator('#confirm-human').click();
    assert.match(await page.locator('#human-state').innerText(), /HUMAN_OVERRIDDEN/);
    const csv = await save('#csv', 'QA_ONLY_review.csv');
    const json = await save('#json', 'QA_ONLY_review.json');
    const final = JSON.parse(await fs.readFile(json, 'utf8'));
    if (finalRound) assert.equal(final.records.find(row => row.review_id === id).final_version_choice, olderChoice);
    assert.equal(final.records.find(row => row.review_id === id).notes, note);
    assert.equal(JSON.parse(final.records.find(row => row.review_id === id).polygon_decisions)
      [selection.partial.polygons[0].polygon_id], 'HOLD');
    if (targeted) {
      const phases = {a6: 'exp14_a6_targeted_v2', a7: 'exp14_a7_targeted_v3', a7b: 'exp14_a7b_v3_completion', a8: 'exp14_a8_final_v4'};
      assert.equal(final.phase, phases[mode]);
      assert.ok(!final.records.some(row => row.review_id === pass.prior.review_id));
    } else {
      assert.deepEqual(final.records.find(row => row.review_id === selection.historical.initial.review_id),
        selection.historical.initial);
    }
    if (supplemental) {
      const old = await context.newPage();
      old.on('pageerror', error => errors.push(error.message));
      await old.goto(pathToFileURL(path.join(previousDirectory, 'index.html')).href);
      const historical = await old.evaluate(() => JSON.parse(document.querySelector('#bundle').textContent));
      assert.equal(historical.records.length, 23);
      const ids = new Set(historical.records.map(record => record.initial.review_id));
      assert.ok(final.records.every(row => !ids.has(row.review_id)));
      await old.locator('#file').setInputFiles(previousCsv);
      await old.waitForFunction(() => document.querySelector('#notice').textContent.includes('已恢复 23 条'));
      const storage = (tab, phase) => tab.evaluate(name => localStorage.getItem(name + ':ai-assisted:v1'), phase);
      const before = await storage(old, historical.phase);
      for (const file of [csv, json]) {
        await old.locator('#file').setInputFiles(file);
        await old.waitForFunction(() => document.querySelector('#notice').dataset.error === 'true');
        assert.equal(await storage(old, historical.phase), before);
      }
      await old.reload();
      await old.waitForFunction(() => document.querySelector('#notice').textContent.includes('已从本地缓存恢复 23 条'));
      assert.equal(await storage(old, historical.phase), before);
      const preserved = JSON.parse(before);
      assert.equal(preserved.phase, historical.phase);
      assert.equal(preserved.records.length, 23);
      assert.equal(JSON.parse(await storage(page, final.phase)).phase, final.phase);
    }
    await context.close();

    for (const file of [csv, json]) {
      const fresh = await open();
      await fresh.page.locator('#file').setInputFiles(file);
      await fresh.page.waitForFunction(() => document.querySelector('#notice').textContent.includes('已恢复'));
      await show(fresh.page, selection.partial);
      assert.equal(await fresh.page.locator('#notes').inputValue(), note);
      if (finalRound) assert.equal(await fresh.page.locator('#final-version').inputValue(), olderChoice);
      await fresh.page.locator('#ai-filter').selectOption('RETRY_REPAIR');
      await fresh.page.locator('#confidence-filter').selectOption(selection.partial.ai.ai_confidence);
      assert.equal(await fresh.page.locator('#item').isVisible(), true);
      if (supplemental || finalRound) {
        await fresh.page.locator('#class').selectOption(selection.partial.classes[0]);
        await fresh.page.locator('#status-filter').selectOption(selection.partial.ai.ai_decision);
        assert.equal(await fresh.page.locator('#item').isVisible(), true);
      }
      await fresh.context.close();
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    console.log('Chrome assisted review: file images, historical protection, AI isolation, human confirmation, draft reload, filters, Unicode CSV/JSON PASS');
  } finally {
    await browser.close();
  }
}
test().catch(error => { console.error(error); process.exitCode = 1; });
