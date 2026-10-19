/* Run with WORKSPACE_URL and PLAYWRIGHT_PATH if they differ from local defaults.
 * Uses real browsing/knowledge/review APIs. GPU-only actions are intercepted and
 * checked for request/response behavior; this does not validate model quality.
 */
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || '/tmp/pwcli/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const base = process.env.WORKSPACE_URL || 'http://127.0.0.1:18299';
const shots = path.join(root, 'outputs/workspace_qa');
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'theatre-browser-'));
const play = path.join(root, 'data/ncpa_test/京剧《天下归心》');
const photos = path.join(play, '【整理完成】京剧《天下归心》彩排');
fs.mkdirSync(shots, {recursive: true});

(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || '/snap/chromium/current/usr/lib/chromium-browser/chrome', headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const idle = () => page.locator('body:not([aria-busy="true"])').waitFor({timeout: 45000});
  const clickView = async view => { await page.locator(`.nav-item[data-view="${view}"]`).click(); };
  const noOverflow = async () => assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'page overflows horizontally');
  const post = async (url, form) => {
    const response = await context.request.post(base + url, {form});
    assert.equal(response.status(), 200, await response.text());
    return (await response.json()).data;
  };
  try {
    await page.goto(base + '/organize');
    await idle();
    assert.equal(await page.locator('#gallery-empty').isVisible(), true);
    await page.screenshot({path: path.join(shots, 'empty-desktop.png'), fullPage: true});
    await noOverflow();

    // Both legacy pages remain linked together and can return to the workspace.
    await page.getByRole('link', {name: '旧版界面', exact: true}).click();
    await page.waitForURL(base + '/legacy/ui');
    await page.getByRole('link', {name: '📦 智能整理（旧版）', exact: true}).click();
    await page.waitForURL(base + '/legacy/organize');
    await page.getByRole('link', {name: '🎭 照片处理（旧版）', exact: true}).click();
    await page.waitForURL(base + '/legacy/ui');
    await page.getByRole('link', {name: '返回新版工作台', exact: true}).click();
    await page.waitForURL(base + '/organize');
    await idle();

    await page.locator('#import-button').click();
    await page.locator('#input-dir').fill('/no/such/theatre');
    await page.locator('#project-form button[type=submit]').click();
    await idle();
    assert.equal(await page.locator('#project-error').isVisible(), true);
    assert.equal(await page.locator('#project-dialog').isVisible(), true);
    await page.locator('#project-name').fill('京剧《天下归心》· 彩排');
    await page.locator('#input-dir').fill(photos);
    await page.locator('#output-dir').fill(fixture);
    await page.locator('#project-form button[type=submit]').click();
    await idle();
    const total = Number(await page.locator('#metric-total').textContent());
    assert.ok(total >= 9, 'real theatrical photos are loaded');
    await page.locator('.photo-card').first().click();
    await page.locator('#preview-photo').click();
    assert.equal(await page.locator('#image-dialog').isVisible(), true);
    await page.keyboard.press('Escape');
    await page.locator('#photo-search').fill('不可能命中的搜索词');
    assert.equal(await page.locator('#no-matches').isVisible(), true);
    await page.locator('#clear-filters').click();
    assert.equal(await page.locator('.photo-card').count(), total);

    await clickView('knowledge');
    await page.locator('#info-path').fill(path.join(play, '剧目信息.txt'));
    await page.locator('#play-root').fill(play);
    await page.locator('#knowledge-form button[type=submit]').click();
    await idle();
    assert.equal(await page.locator('.knowledge-scene').count(), 5);
    await page.locator('#use-embedder').uncheck();
    await page.locator('#knowledge-analyze').click();
    await idle();
    assert.equal(await page.locator('#breadcrumb-view').textContent(), '场景复核');
    assert.ok(Number(await page.locator('#metric-pending').textContent()) > 0);
    assert.ok(await page.locator('.evidence-item').count() > 0);
    await page.locator('.photo-card img').first().waitFor();
    await page.locator('.photo-card img').evaluateAll(images => images.forEach(img => { img.loading = 'eager'; }));
    await page.waitForFunction(() => [...document.querySelectorAll('.photo-card img')].every(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({path: path.join(shots, 'review-desktop.png'), fullPage: true});
    await noOverflow();

    const pending = Number(await page.locator('#metric-pending').textContent());
    if (await page.locator('#accept-review').isEnabled()) await page.locator('#accept-review').click();
    else { await page.locator('#review-label').fill('第1幕克段'); await page.locator('#review-decision-form button[type=submit]').click(); }
    await idle();
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 1);
    assert.equal(Number(await page.locator('#metric-pending').textContent()), pending - 1);
    await page.locator('#review-label').fill('第2幕庆功');
    await page.locator('#review-note').fill('浏览器验收：对照剧情修订');
    await page.locator('#review-decision-form button[type=submit]').click();
    await idle();
    await page.locator('#reject-review').click();
    await idle();
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 2);
    await page.locator('[data-filter=approved]').click();
    assert.equal(await page.locator('.photo-card').count(), 2);
    await page.locator('#photo-search').fill('第2幕庆功');
    assert.ok(await page.locator('.photo-card').count() >= 1);
    await page.locator('#photo-search').fill('');

    await clickView('export');
    await page.locator('#export-labels').click();
    await idle();
    assert.match(await page.locator('#scene-evidence-file').inputValue(), /scene_labels\.json$/);
    const download = await context.request.get(base + await page.locator('#labels-download a').getAttribute('href'));
    assert.equal(download.status(), 200);
    assert.equal(Object.keys(await download.json()).length, 2);

    // Check an unchecked optional boolean reaches the actual endpoint as false.
    let organizePayload;
    await page.route('**/api/organize/run', async route => {
      organizePayload = route.request().postData();
      await route.fulfill({json: {code: 200, data: {total_input: total, kept: 6, discarded: total - 6, n_scenes: 3, report_path: path.join(fixture, 'organize_report.json')}}});
    });
    await page.locator('#organize-form button[type=submit]').click();
    await idle();
    assert.match(organizePayload, /name="classify_role"\r\n\r\nfalse/);
    assert.match(organizePayload, /scene_labels\.json/);
    assert.match(await page.locator('#organize-result').textContent(), /归档完成/);
    await page.unroute('**/api/organize/run');

    // Exercise the actual cluster response schema and reversible ignore/restore UI.
    await clickView('actors');
    let clusterStatus = 'pending', assignedTo = null, scanFails = false;
    await page.route('**/api/organize/face_cluster/scan', route => route.fulfill({json: {code: 200, data: {task_id: 'browser-cluster'}}}));
    await page.route('**/api/progress/browser-cluster', route => route.fulfill({json: scanFails ? {progress: 20, status: 'failed', error: '扫描失败测试', message: '扫描失败测试'} : {progress: 100}}));
    await page.route('**/api/organize/face_cluster/result?*', route => route.fulfill({json: {code: 200, data: {clusters: [{id: 'cluster-01', size: 5, cohesion: .8, status: clusterStatus, assigned_to: assignedTo, suggest_manual: false, preview: path.join(root, 'data/sample_images/2.jpg')}]}}}));
    await page.route('**/api/organize/face_cluster/ignore', async route => { clusterStatus = clusterStatus === 'ignored' ? 'pending' : 'ignored'; await route.fulfill({json: {code: 200, data: {}}}); });
    await page.route('**/api/organize/face_cluster/assign', async route => { clusterStatus = 'assigned'; assignedTo = '陈少云（饰颍考叔）'; await route.fulfill({json: {code: 200, data: {}}}); });
    await page.locator('#cluster-button').click(); await idle();
    assert.equal(await page.locator('.cluster-card').count(), 1);
    await page.locator('[data-ignore]').click(); await idle();
    assert.match(await page.locator('[data-ignore]').textContent(), /恢复/);
    await page.locator('[data-ignore]').click(); await idle();
    assert.match(await page.locator('[data-ignore]').textContent(), /忽略/);
    await page.locator('[data-cluster] input').fill('陈少云（饰颍考叔）');
    await page.locator('[data-cluster] button[type=submit]').click(); await idle();
    assert.match(await page.locator('.cluster-card').textContent(), /已入库：陈少云/);
    scanFails = true;
    await page.locator('#cluster-button').click(); await idle();
    assert.match(await page.locator('#notice').textContent(), /扫描失败测试/);
    assert.equal(await page.locator('#cluster-button').isEnabled(), true);

    await page.reload(); await idle();
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 2, 'human review persists after reload');
    assert.equal(await page.locator('#use-embedder').isChecked(), false, 'model choice persists after reload');
    await clickView('photos');
    await page.locator('.photo-card').first().click();
    await page.locator('.photo-card img').evaluateAll(images => images.forEach(img => { img.loading = 'eager'; }));
    await page.waitForFunction(() => [...document.querySelectorAll('.photo-card img, #preview-photo img')].every(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({path: path.join(shots, 'archive-desktop.png'), fullPage: false});
    for (const width of [1024, 768, 540, 390]) {
      await page.setViewportSize({width, height: 900});
      assert.equal(await page.getByRole('link', {name: '旧版界面', exact: true}).isVisible(), true);
      await noOverflow();
      for (const view of ['knowledge', 'actors', 'export', 'review', 'photos']) { await clickView(view); await noOverflow(); }
      if (width === 390) {
        await page.waitForFunction(() => [...document.querySelectorAll('.photo-card img')].filter(img => img.getBoundingClientRect().top < innerHeight).every(img => img.complete && img.naturalWidth > 0));
        await page.screenshot({path: path.join(shots, 'archive-mobile.png'), fullPage: false});
      }
    }
    await page.locator('#project-settings').click();
    await noOverflow();
    await page.keyboard.press('Escape');

    // User/record text must never become markup or executable attributes.
    const malicious = '<img src=x onerror="window.workspaceXss=1">';
    const reviewFile = path.join(fixture, 'review/review_queue.jsonl');
    const rows = fs.readFileSync(reviewFile, 'utf8').trim().split('\n').map(JSON.parse);
    await post('/api/organize/scene/review', {review_dir: path.join(fixture, 'review'), image: rows[0].image, action: 'modify', label: malicious});
    await page.reload(); await idle(); await clickView('photos'); await page.locator('.photo-card').first().click();
    assert.equal(await page.evaluate(() => window.workspaceXss), undefined);
    assert.equal(await page.locator('#inspector img').count(), 1);

    // A queue larger than the API's page size must not lose off-screen records.
    const longPhotos = Array.from({length: 205}, (_, index) => ({name: `photo-${String(index).padStart(3, '0')}.jpg`, path: `${photos}/photo-${String(index).padStart(3, '0')}.jpg`, size: 100, metadata: {people: []}}));
    const longRecords = longPhotos.map(photo => ({image: photo.path, label: '第1幕克段', scene_id: 'act-01', review: {status: 'pending'}}));
    const offsets = [];
    await page.route('**/api/workspace/photos?*', route => route.fulfill({json: {code: 200, data: {input_dir: photos, photos: longPhotos, total: 205, truncated: false}}}));
    await page.route('**/api/organize/scene/review?*', route => {
      const offset = Number(new URL(route.request().url()).searchParams.get('offset'));
      offsets.push(offset);
      return route.fulfill({json: {code: 200, data: {records: longRecords.slice(offset, offset + 200), summary: {pending: 205, approved: 0, rejected: 0}}}});
    });
    await page.reload(); await idle(); await clickView('photos');
    assert.equal(Number(await page.locator('#metric-pending').textContent()), 205);
    assert.deepEqual(offsets, [0, 200]);
    assert.equal(await page.locator('.photo-card').count(), 36);
    await page.locator('#next-page').click();
    assert.equal(await page.locator('#page-number').textContent(), '2 / 6');
    await page.locator('#photo-search').fill('photo-204');
    assert.equal(await page.locator('.photo-card').count(), 1);
    await page.locator('#import-button').click();
    assert.equal(await page.locator('#input-dir').inputValue(), '', 'switching projects starts with fresh context');
    assert.notEqual(await page.locator('#output-dir').inputValue(), fixture);
    await page.keyboard.press('Escape');
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log(JSON.stringify({passed: true, realPhotos: total, checks: ['empty/error states', 'real photo previews', 'knowledge ingest', 'offline scene analysis', 'review accept/modify/reject', 'search/filter', 'label download', 'organize payload', 'cluster assign/ignore/restore/error', 'refresh persistence', 'responsive 1440/1024/768/540/390', 'escaped evidence', '205 record pagination', 'independent project contexts'], screenshots: shots}, null, 2));
  } finally {
    await browser.close();
    fs.rmSync(fixture, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
