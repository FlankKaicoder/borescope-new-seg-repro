const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.argv[3]);

async function run() {
  const url = pathToFileURL(process.argv[2]).href;
  const browser = await chromium.launch({channel:'chrome', headless:true});
  const errors = [];
  const network = [];
  const open = async () => {
    const context = await browser.newContext({acceptDownloads:true});
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
    await page.goto(url);
    return {context,page};
  };
  try {
    const {page} = await open();
    const bundle = await page.locator('#bundle').evaluate(node => JSON.parse(node.textContent));
    assert.equal(bundle.rows.length, 392);
    assert.equal(new Set(bundle.rows.map(row => row.source_id)).size, 392);
    const historical = 'HUMAN_REVIEWED_ASSET_REUSABLE';
    const proposed = 'RECOVERY_PROPOSED_PENDING_HUMAN';
    const candidates = bundle.rows.filter(row => row.disposition === proposed && row.recommended_asset_id);
    const main = candidates.find(row => row.assets.length > 1);
    assert.ok(main, 'A proposal with alternative pixels is needed for QA');
    const batch = candidates.filter(row => row.source_id !== main.source_id).slice(0,2);
    assert.equal(batch.length, 2);
    const pick = async (viewer, row) => {
      await viewer.locator('#filter').selectOption('all');
      await viewer.locator('#search').fill(row.source_id);
      assert.equal(await viewer.locator('#source').textContent(), row.source_id);
      await viewer.waitForFunction(() => {
        const images = [...document.querySelectorAll('#whole img')];
        return images.length >= 2 && images.every(img => img.complete && img.naturalWidth > 0);
      });
    };
    const exportFile = async (viewer, button) => {
      const event = viewer.waitForEvent('download');
      await viewer.locator(button).click();
      const download = await event;
      return download.path();
    };
    const importFile = async (viewer, path, name) => {
      await viewer.locator('#import-file').setInputFiles({
        name, mimeType:name.endsWith('.json') ? 'application/json' : 'text/csv', buffer:await fs.readFile(path)
      });
      await viewer.waitForFunction(() => document.querySelector('#notice').textContent.startsWith('已导入'));
    };
    assert.equal(await page.locator('#filter').inputValue(), 'waiting');
    const originalPending = bundle.rows.filter(row => row.disposition !== historical).length;
    assert.equal(await page.locator('#position').textContent(), `1 / ${originalPending}`);
    await pick(page, main);
    const recommended = main.assets.find(asset => asset.asset_id === main.recommended_asset_id);
    assert.equal(await page.locator('#asset').inputValue(), recommended.asset_id);
    await page.locator('#accept').click();
    assert.match(await page.locator('#recorded').textContent(), new RegExp(recommended.sha256));
    const alternate = main.assets.find(asset => asset.asset_id !== recommended.asset_id);
    assert.notEqual(alternate.asset_id, recommended.asset_id);
    const reason = "'+复核原因";
    const notes = '=第一行，带逗号 "引号"\n第二行机械结构确认';
    await page.locator('#asset').selectOption(alternate.asset_id);
    await page.locator('#reason').fill(reason);
    await page.locator('#notes').fill(notes);
    await page.locator('#alternate').click();
    assert.match(await page.locator('#recorded').textContent(), new RegExp(alternate.sha256));
    for (const row of batch) {
      await pick(page, row);
      await page.locator('#batch-check').check();
    }
    await page.locator('#batch-open').click();
    assert.equal(await page.locator('#batch-scope li').count(), 2);
    await page.locator('#batch-confirm').click();
    const csv = await exportFile(page, '#export-csv');
    const json = await exportFile(page, '#backup-json');
    const records = JSON.parse(await fs.readFile(json,'utf8')).records;
    assert.equal(records.length, 3);
    const stored = records.find(record => record.source_id === main.source_id);
    assert.equal(stored.action, 'CHOOSE_ALTERNATE');
    assert.equal(stored.asset_id, alternate.asset_id);
    assert.equal(stored.asset_type, alternate.asset_type);
    assert.equal(stored.asset_sha256, alternate.sha256);
    assert.equal(stored.reason, reason);
    assert.equal(stored.notes, notes);
    for (const row of batch) {
      const record = records.find(item => item.source_id === row.source_id);
      assert.equal(record.action, 'ACCEPT_RECOMMENDED');
      assert.equal(record.batch_scope, batch.map(item => item.source_id).sort().join('|'));
      assert.ok(record.operation_id);
    }
    await page.reload();
    await pick(page, main);
    assert.equal(await page.locator('#asset').inputValue(), alternate.asset_id);
    assert.equal(await page.locator('#notes').inputValue(), notes);
    for (const [path,name] of [[csv,'review.csv'],[json,'review.json']]) {
      const {page:fresh,context} = await open();
      await importFile(fresh,path,name);
      await pick(fresh, main);
      assert.equal(await fresh.locator('#asset').inputValue(), alternate.asset_id);
      assert.equal(await fresh.locator('#reason').inputValue(), reason);
      assert.equal(await fresh.locator('#notes').inputValue(), notes);
      const tampered = JSON.parse(await fs.readFile(json,'utf8'));
      tampered.records.find(item => item.source_id === main.source_id).asset_sha256 = '0'.repeat(64);
      await fresh.locator('#import-file').setInputFiles({
        name:'tampered.json', mimeType:'application/json', buffer:Buffer.from(JSON.stringify(tampered))
      });
      await fresh.waitForFunction(() => document.querySelector('#notice').textContent.startsWith('导入失败'));
      assert.match(await fresh.locator('#recorded').textContent(), new RegExp(alternate.sha256));
      await context.close();
    }
    const loaded = await page.evaluate(async () => {
      const bundle = JSON.parse(document.getElementById('bundle').textContent);
      const required = new Set();
      for (const row of bundle.rows) {
        const original = row.polygon || row.original;
        const historical = row.assets.find(item => item.asset_id === row.historical_asset_id);
        const recommended = row.assets.find(item => item.asset_id === row.recommended_asset_id);
        const assets = [original, historical, recommended, ...row.crops.map(crop => crop.before)];
        for (const asset of assets) if (asset?.url) required.add(asset.url);
      }
      const urls = [...required];
      for (let start = 0; start < urls.length; start += 24) {
        await Promise.all(urls.slice(start, start + 24).map(async url => {
          const img = new Image();
          img.src = new URL(url, document.baseURI).href;
          await img.decode();
          if (!img.naturalWidth) throw Error('Empty visual evidence');
        }));
      }
      return urls.length;
    });
    assert.ok(loaded >= 392);
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    const result = {browser:'Chrome', sources:392, decoded_primary_or_reference_images:loaded, image_panels:true, refresh:true,
      csv_roundtrip:true, json_roundtrip:true, tampered_hash_rejected:true, exact_batch_scope:true, offline:true};
    if (process.argv[4]) await fs.writeFile(process.argv[4],JSON.stringify(result,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
