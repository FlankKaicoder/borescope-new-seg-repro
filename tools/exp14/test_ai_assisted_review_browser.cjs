/* Chrome integration against the private assisted page. QA exports are synthetic. */
const {chromium} = require(process.argv[3]);
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

async function test() {
  const [, , directory, , qa, mode, previousCsv] = process.argv;
  const a6 = mode === 'a6';
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
    assert(selection.pass && selection.partial && (a6 || selection.historical));
    const show = async (tab, item) => {
      await tab.locator('#source').fill(item.source_id);
      while (!(await tab.locator('#title').innerText()).endsWith(' · ' + item.variant_id)) {
        if (!(await tab.locator('#next').isEnabled())) assert.fail('Variant was filtered out');
        await tab.locator('#next').click();
      }
    };
    const save = async (id, name) => {
      const pending = page.waitForEvent('download');
      await page.locator(id).click();
      const receipt = await pending;
      const file = path.join(qa, name);
      await receipt.saveAs(file);
      return file;
    };
    if (!a6) {
      await show(page, selection.historical);
      assert.equal(await page.locator('#adopt-ai').isDisabled(), true);
      assert.equal(await page.locator('#decision').inputValue(), 'PASS');
    }
    await show(page, selection.pass);
    const pass = selection.pass;
    const firstPolygon = pass.polygons[0];
    const expected = a6 ? 2 + Number(pass.bounded) + 1 + 1 + Number(!!firstPolygon.after) +
      Number(!!pass.prior.crops[firstPolygon.polygon_id]) : 5;
    await page.waitForFunction(count => {
      const images = [...document.querySelectorAll('#images img, #crops img')];
      return images.length === count && images.every(img => img.complete && img.naturalWidth > 0);
    }, expected);
    if (a6) {
      assert.match(await page.locator('#prior-notes').innerText(), /原人工/);
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
    assert.equal(final.records.find(row => row.review_id === id).notes, note);
    assert.equal(JSON.parse(final.records.find(row => row.review_id === id).polygon_decisions)
      [selection.partial.polygons[0].polygon_id], 'HOLD');
    if (a6) {
      assert.equal(final.phase, 'exp14_a6_targeted_v2');
      assert.ok(!final.records.some(row => row.review_id === pass.prior.review_id));
    } else {
      assert.deepEqual(final.records.find(row => row.review_id === selection.historical.initial.review_id),
        selection.historical.initial);
    }
    await context.close();

    for (const file of [csv, json]) {
      const fresh = await open();
      await fresh.page.locator('#file').setInputFiles(file);
      await fresh.page.waitForFunction(() => document.querySelector('#notice').textContent.includes('已恢复'));
      await show(fresh.page, selection.partial);
      assert.equal(await fresh.page.locator('#notes').inputValue(), note);
      await fresh.page.locator('#ai-filter').selectOption('RETRY_REPAIR');
      await fresh.page.locator('#confidence-filter').selectOption(selection.partial.ai.ai_confidence);
      assert.equal(await fresh.page.locator('#item').isVisible(), true);
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
