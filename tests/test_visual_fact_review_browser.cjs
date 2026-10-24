/* Actual review images and UI, with temporary scripted drafts only; no human GT. */
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || '/tmp/pwcli/node_modules/playwright');

const root = path.resolve(__dirname, '..');
const review = process.env.FACT_REVIEW_ROOT || path.join(root, 'outputs/acceptance/core_upgrade/u5_dual_fact_review_v2_20261024');
const output = process.env.FACT_REVIEW_CHECK_OUTPUT || path.join(root, 'outputs/acceptance/core_upgrade/completion_followup_20261023/fact_browser_final');
fs.mkdirSync(output, {recursive: true});
const template = JSON.parse(fs.readFileSync(path.join(review, 'reviewer_A_template.json'), 'utf8'));

(async () => {
  const browser = await chromium.launch({executablePath: '/snap/chromium/current/usr/lib/chromium-browser/chrome',
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
  const page = await context.newPage();
  const errors = [], checks = {};
  page.on('pageerror', e => errors.push(e.message));
  const pageUrl = pathToFileURL(path.join(review, 'index.html')).href;
  const imageReady = async () => page.waitForFunction(() => {
    const image = document.querySelector('.photo img'); return image?.complete && image.naturalWidth > 0;
  });
  const download = async filename => {
    const pending = page.waitForEvent('download'); await page.locator('#export').click();
    const file = await pending; await file.saveAs(path.join(output, filename));
    return JSON.parse(fs.readFileSync(path.join(output, filename), 'utf8'));
  };
  try {
    await page.goto(pageUrl); await imageReady();
    assert.equal(await page.locator('article').count(), 1); checks.bounded_image_page = true;
    assert.match(await page.locator('#message').textContent(), /已完成 0/); checks.initial_pending = true;
    checks.real_preview_loaded = true;
    await page.locator('#reviewer').fill('scripted browser A');
    await page.locator('input[data-review="source_ref"]').fill('scripted temporary test only');
    await page.locator('#complete').click();
    assert.match(await page.locator('#message').textContent(), /完成每条事实/); checks.partial_review_rejected = true;
    const selects = page.locator('[data-fact] select');
    for (let i = 0; i < await selects.count(); i++) await selects.nth(i).selectOption('supported');
    await selects.first().selectOption('error');
    await page.locator('#complete').click();
    assert.match(await page.locator('#message').textContent(), /错误项需填写说明/);
    const first = page.locator('[data-fact]').first();
    await first.locator('input[data-field="correction"]').fill('scripted correction');
    await first.locator('input[data-field="duplicate"]').check();
    await page.locator('textarea[data-omissions="clothing"]').fill('scripted omission');
    await page.locator('#complete').click();
    assert.match(await page.locator('#message').textContent(), /已完成 1/); checks.complete_review_validation = true;
    await page.locator('#next').click(); await page.locator('#prev').click();
    assert.equal(await page.locator('[data-fact] input[data-field="correction"]').first().inputValue(), 'scripted correction');
    checks.draft_survives_page = true;
    await page.reload(); await imageReady();
    assert.equal(await page.locator('#reviewer').inputValue(), 'scripted browser A');
    assert.equal(await page.locator('[data-fact] select').first().inputValue(), 'error'); checks.draft_survives_reload = true;
    const draftA = await download('scripted_review_A.json');
    assert.equal(draftA.protocol_sha256, template.protocol_sha256);
    assert.equal(draftA.human_decisions_applied, false); assert.equal(draftA.final_test_locked, false);
    checks.export_protocol_bound = true;
    await page.locator('#slot').selectOption('B');
    assert.equal(await page.locator('#reviewer').inputValue(), '');
    assert.equal(await page.locator('[data-fact] select').first().inputValue(), 'pending'); checks.reviewers_keep_separate_drafts = true;
    await page.locator('#reviewer').fill('scripted browser B');
    await page.locator('#kind').selectOption('ocr'); await imageReady();
    const predicted = template.rows.find(row => row.kind === 'ocr').predicted_text;
    assert.equal(await page.locator('body').evaluate((body, text) => body.textContent.includes(text), predicted), false);
    assert.equal(await page.locator('#reference-text').inputValue(), ''); checks.OCR_reference_not_prefilled_or_prediction_displayed = true;
    await page.locator('#reference-text').fill('脚本临时转录');
    await page.locator('input[data-review="source_ref"]').fill('scripted temporary test only');
    await page.locator('#complete').click();
    const draftB = await download('scripted_review_B.json');
    assert.equal(draftB.slot, 'B');
    checks.OCR_transcription_exported = true;
    const bad = structuredClone(draftB); bad.rows[0].source_sha256 = '0'.repeat(64);
    fs.writeFileSync(path.join(output, 'scripted_changed_binding.json'), JSON.stringify(bad));
    await page.locator('#import').setInputFiles(path.join(output, 'scripted_changed_binding.json'));
    assert.match(await page.locator('#message').textContent(), /恢复失败/); checks.changed_binding_rejected = true;
    await page.locator('#import').setInputFiles(path.join(output, 'scripted_review_B.json'));
    assert.equal(await page.locator('#reference-text').inputValue(), '脚本临时转录'); checks.import_restores = true;
    await page.reload(); await imageReady();
    assert.equal(await page.locator('#slot').inputValue(), 'B', 'reload keeps the active independent reviewer');
    assert.equal(await page.locator('#kind').inputValue(), 'ocr', 'reload keeps the selected review task');
    assert.equal(await page.locator('#reference-text').inputValue(), '脚本临时转录'); checks.reviewer_and_task_survive_reload = true;
    for (const width of [1440, 1024, 768, 540, 390]) {
      await page.setViewportSize({width, height: 1000});
      const sizes = await page.evaluate(() => [innerWidth, document.documentElement.scrollWidth]);
      assert.ok(sizes[1] <= sizes[0], 'horizontal overflow at width ' + width);
      assert.equal(await page.locator('article').count(), 1);
    }
    checks.five_widths_no_overflow = true;
    await page.screenshot({path: path.join(output, 'OCR_mobile.png'), fullPage: true});
    await page.setViewportSize({width: 1440, height: 1000});
    await page.locator('#slot').selectOption('A'); await page.locator('#kind').selectOption('visual');
    await imageReady(); await page.screenshot({path: path.join(output, 'facts_desktop.png'), fullPage: true});
    const blocked = await browser.newContext({viewport: {width: 390, height: 900}});
    await blocked.addInitScript(() => { Storage.prototype.setItem = () => { throw new Error('scripted storage failure'); }; });
    const blockedPage = await blocked.newPage(); await blockedPage.goto(pageUrl);
    await blockedPage.locator('#reviewer').fill('scripted storage check');
    await blockedPage.locator('input[data-review="source_ref"]').fill('scripted temporary test only');
    assert.match(await blockedPage.locator('#message').textContent(), /保存失败/);
    assert.match(await blockedPage.locator('#message').textContent(), /导出/);
    checks.storage_failure_stays_visible = true;
    await blocked.close();
    assert.equal(errors.length, 0, errors.join('\n')); checks.no_script_errors = true;
    const result = {status: 'passed', checks, errors, human_labels_added: 0,
      limitations: ['Scripted temporary drafts only; original templates remain pending and no human GT is created.']};
    fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } catch (error) {
    fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify({status: 'failed', checks, errors,
      failure: error.message, human_labels_added: 0}, null, 2));
    await page.screenshot({path: path.join(output, 'failure.png'), fullPage: true}).catch(() => {});
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
