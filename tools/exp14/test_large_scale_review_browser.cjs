/* Chrome QA using private assets. The exports are synthetic and kept outside the review folder. */
const {chromium} = require(process.argv[3]);
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');

async function test() {
  const [, , review, , qa] = process.argv;
  await fs.mkdir(qa);
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    const url = pathToFileURL(path.join(review, 'index.html')).href;
    const errors = [];
    const network = [];
    const open = async () => {
      const context = await browser.newContext({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      page.on('request', request => { if (/^https?:/i.test(request.url())) network.push(request.url()); });
      await page.goto(url);
      return {page, context};
    };
    const {page, context} = await open();
    const item = await page.evaluate(() => {
      const {records} = JSON.parse(document.querySelector('#bundle').textContent);
      return records.find(row => row.aligned && row.initial.decision === 'UNREVIEWED');
    });
    assert(item, 'Find an aligned pending case');
    const show = async (tab, search) => {
      await tab.locator('#source').fill(search);
      if ((await tab.locator('#title').innerText()).endsWith(item.variant_id)) return;
      for (let i = 0; i < 20 && await tab.locator('#next').isEnabled(); i++) {
        await tab.locator('#next').click();
        if ((await tab.locator('#title').innerText()).endsWith(item.variant_id)) return;
      }
      assert.fail('Requested variant unavailable after filtering');
    };
    await show(page, item.source_id);
    await page.waitForFunction(() => {
      const images = [...document.querySelectorAll('#images img, #crops img')];
      return images.length === 5 && images.every(img => img.complete && img.naturalWidth > 0);
    });
    assert.equal(await page.locator('img').count(), 5, 'Only current variant images loaded');
    assert.equal(await page.locator('#whole').isDisabled(), true);
    await page.locator('#decision').selectOption('PASS');
    assert.equal(await page.locator('#whole').isChecked(), false, 'Local repair PASS does not imply whole-image normal');
    await page.locator('#decision').selectOption('RETRY_REPAIR');
    await page.locator('#reason').selectOption('缺材未补全');
    const notes = '仅 QA，检查边缘, "重修"\n第二行：缺陷未消失。\n尾行';
    await page.locator('#notes').fill(notes);
    await page.locator('#polygon-notes summary').click();
    await page.locator('#polygon-decisions select').first().selectOption('RETRY_REPAIR');
    await page.locator('#class').selectOption(item.classes[0]);
    await page.locator('#source').fill('');
    if (await page.locator('#next').isEnabled()) await page.locator('#next').click();
    await show(page, item.source_id);
    assert.equal(await page.locator('#notes').inputValue(), notes);
    await page.locator('#diagnostics summary').click();
    await page.waitForFunction(() => {
      const images = [...document.querySelectorAll('#diagnostic-images img')];
      return images.length === 5 && images.every(img => img.complete && img.naturalWidth > 0);
    });
    assert.equal(await page.locator('#diagnostic-images img').count(), 5);
    const save = async (selector, name) => {
      const pending = page.waitForEvent('download');
      await page.locator(selector).click();
      const download = await pending;
      const destination = path.join(qa, name);
      await download.saveAs(destination);
      return destination;
    };
    const csv = await save('#csv', 'QA_ONLY_review.csv');
    const json = await save('#json', 'QA_ONLY_review.json');
    const decoded = JSON.parse(await fs.readFile(json, 'utf8'));
    const saved = decoded.records.find(row => row.review_id === item.initial.review_id);
    assert.equal(saved.notes, notes);
    assert.equal(JSON.parse(saved.polygon_decisions)[item.polygons[0].polygon_id], 'RETRY_REPAIR');
    assert.deepEqual((await fs.readFile(csv)).subarray(0, 3), Buffer.from([0xef, 0xbb, 0xbf]));
    await context.close();

    for (const file of [csv, json]) {
      const fresh = await open();
      await show(fresh.page, item.source_id);
      assert.equal(await fresh.page.locator('#decision').inputValue(), 'UNREVIEWED');
      await fresh.page.locator('#file').setInputFiles(file);
      await fresh.page.waitForFunction(() => document.querySelector('#notice').textContent.includes('已恢复'));
      await show(fresh.page, item.source_id);
      assert.equal(await fresh.page.locator('#decision').inputValue(), 'RETRY_REPAIR');
      assert.equal(await fresh.page.locator('#notes').inputValue(), notes);
      assert.equal(await fresh.page.locator('#reason').inputValue(), '缺材未补全');
      await fresh.page.locator('#polygon-notes summary').click();
      assert.equal(await fresh.page.locator('#polygon-decisions select').first().inputValue(), 'RETRY_REPAIR');
      saved.raw_sha256 = 'stale';
      await fresh.page.locator('#file').setInputFiles({
        name: 'QA_ONLY_wrong.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(decoded)),
      });
      await fresh.page.waitForFunction(() => document.querySelector('#notice').textContent.includes('导入失败'));
      assert.equal(await fresh.page.locator('#notes').inputValue(), notes);
      await fresh.context.close();
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    await fs.writeFile(path.join(qa, 'QA_ONLY_result.json'), JSON.stringify({
      status: 'PASS', browser: 'Chrome', file_protocol: true, image_load: true, filters_and_navigation: true,
      csv_and_json_download_restore_in_fresh_contexts: true, chinese_multiline_exact: true,
      polygon_decision_exact: true, stale_identity_rejected: true, remote_requests: 0,
    }, null, 2) + '\n', {flag: 'wx'});
    console.log('Chrome offline review round-trip PASS');
  } finally {
    await browser.close();
  }
}

test().catch(error => {console.error(error); process.exitCode = 1;});
