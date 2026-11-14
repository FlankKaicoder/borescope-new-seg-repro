const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.argv[3]);

async function main() {
  const browser = await chromium.launch({channel:'chrome', headless:true});
  try {
    const page = await browser.newPage();
    const errors = [], network = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', req => { if (/^https?:/.test(req.url())) network.push(req.url()); });
    await page.goto(pathToFileURL(process.argv[2]).href);
    const bundle = await page.locator('#bundle').evaluate(node => JSON.parse(node.textContent));
    let choices = 0;
    async function images(id, count) {
      if (count === 0) return assert.equal(await page.locator('#' + id + ' img').count(), 0);
      await page.waitForFunction(([id, count]) => {
        const imgs = Array.from(document.querySelectorAll('#' + id + ' img'));
        return imgs.length === count && imgs.every(img => img.complete && img.naturalWidth > 0);
      }, [id, count]);
    }
    for (const route of ['a', 'b']) {
      assert.equal(bundle[route].length, 36);
      await page.locator('#route').selectOption(route);
      assert.equal(await page.locator('#position').textContent(), '1 / 36');
      const group = bundle[route][0].group;
      await page.locator('#group').selectOption(group);
      assert.match(await page.locator('#title').textContent(), new RegExp(group + '$'));
      await page.locator('#group').selectOption('');
      for (const row of bundle[route]) {
        await page.locator('#search').fill(row.id);
        assert.match(await page.locator('#title').textContent(), new RegExp('^' + row.id + ' '));
        assert.equal(await page.locator('#review').textContent(), row.review);
        await images('whole', Object.keys(row.whole).length);
        for (const [pid, files] of Object.entries(row.polygons)) {
          await page.locator('#polygon').selectOption(pid);
          await images('crops', Object.keys(files).length);
          choices += 1;
        }
      }
      await page.locator('#search').fill('00000000');
      assert.ok(await page.locator('#empty').isVisible());
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    const result = {browser:'Chrome', a_cases:36, b_cases:36, polygon_selections:choices, offline:true};
    await fs.writeFile(process.argv[4], JSON.stringify(result, null, 2) + '\n', {flag:'wx'});
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
