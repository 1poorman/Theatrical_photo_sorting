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
  const noOverflow = async () => {
    const overflow = await page.evaluate(() => ({width: innerWidth, scroll: document.documentElement.scrollWidth,
      elements: [...document.querySelectorAll('body *')].filter(element => element.getBoundingClientRect().right > innerWidth)
        .map(element => element.id || element.className).slice(0, 8)}));
    assert.ok(overflow.scroll <= overflow.width, `page overflows horizontally: ${JSON.stringify(overflow)}`);
  };
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
    await page.waitForFunction(() => document.querySelectorAll('.knowledge-scene').length === 5, null, {timeout: 45000});
    await idle();
    assert.equal(await page.locator('.knowledge-scene').count(), 5);
    await page.locator('#use-embedder').uncheck();
    let releaseTaskView = false, taskCreates = 0;
    page.on('request', request => { if (request.url() === base + '/api/organize/tasks' && request.method() === 'POST') taskCreates++; });
    await page.route('**/api/organize/tasks/*', async route => {
      const response = await route.fetch();
      const body = await response.json();
      if (!releaseTaskView && body.data) {
        body.data = {...body.data, status: 'running', result: null, succeeded: 0, failed: 0, pending: total};
      }
      await route.fulfill({response, json: body});
    });
    await page.locator('#knowledge-analyze').click();
    await idle();
    await page.locator('#analysis-task').waitFor();
    const analysisId = await page.evaluate(() => JSON.parse(localStorage.getItem('theatre-workspace-v1')).analysisTask);
    assert.ok(analysisId);
    await page.reload(); await idle();
    await page.locator('#analysis-cancel').waitFor();
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('theatre-workspace-v1')).analysisTask), analysisId);
    assert.equal(taskCreates, 1, 'refresh must query the existing task without submitting again');
    releaseTaskView = true;
    await page.waitForFunction(() => Number(document.querySelector('#metric-pending').textContent) > 0, null, {timeout: 45000});
    await idle();
    await page.unroute('**/api/organize/tasks/*');
    assert.equal(await page.locator('#breadcrumb-view').textContent(), '场景复核');
    assert.ok(Number(await page.locator('#metric-pending').textContent()) > 0);
    assert.ok(await page.locator('.evidence-item').count() > 0);
    await page.locator('.photo-card img').first().waitFor();
    await page.locator('.photo-card img').evaluateAll(images => images.forEach(img => { img.loading = 'eager'; }));
    await page.waitForFunction(() => [...document.querySelectorAll('.photo-card img')].every(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({path: path.join(shots, 'review-desktop.png'), fullPage: true});
    await noOverflow();

    const pending = Number(await page.locator('#metric-pending').textContent());
    const firstPhoto = await page.locator('.photo-card.selected').getAttribute('data-photo');
    await page.locator('#review-label').fill('草稿：待核对幕次');
    await page.locator('#review-note').fill('切图和刷新后保留的备注');
    await page.locator('.photo-card').nth(1).click();
    await page.locator(`.photo-card[data-photo="${firstPhoto}"]`).click();
    assert.equal(await page.locator('#review-label').inputValue(), '草稿：待核对幕次');
    assert.equal(await page.locator('#review-note').inputValue(), '切图和刷新后保留的备注');
    await page.locator('#photo-search').fill('不可能命中的搜索词');
    await page.locator('#photo-search').fill('');
    assert.equal(await page.locator('#review-label').inputValue(), '草稿：待核对幕次');
    await page.reload();
    await idle();
    assert.equal(await page.locator('#review-label').inputValue(), '草稿：待核对幕次');
    assert.equal(await page.locator('#review-note').inputValue(), '切图和刷新后保留的备注');
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 0);
    assert.equal(await page.locator('#review-draft-status').isVisible(), true);
    await page.route('**/api/organize/scene/review', route => route.fulfill({status: 500, json: {detail: '保存失败测试'}}));
    await page.locator('#review-decision-form button[type=submit]').click(); await idle();
    assert.equal(await page.locator('#review-label').inputValue(), '草稿：待核对幕次');
    assert.equal(await page.locator('#review-note').inputValue(), '切图和刷新后保留的备注');
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 0);
    await page.unroute('**/api/organize/scene/review');
    const concurrentRecord = await (await context.request.get(base + '/api/organize/scene/review?' +
      new URLSearchParams({review_dir: path.join(fixture, 'review'), image: firstPhoto}))).json();
    await post('/api/organize/scene/review', {review_dir: path.join(fixture, 'review'), image: firstPhoto,
      action: 'modify', label: '另一位复核者的标签', expected_revision: concurrentRecord.data.record.review_revision});
    await page.locator('#review-decision-form button[type=submit]').click(); await idle();
    assert.match(await page.locator('#notice').textContent(), /已被更新/);
    assert.equal(await page.locator('#review-label').inputValue(), '草稿：待核对幕次');
    assert.equal(await page.locator('#review-note').inputValue(), '切图和刷新后保留的备注');
    assert.match(await page.locator('.scene-choice').textContent(), /另一位复核者的标签/);
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 1);
    if (await page.locator('#accept-review').isEnabled()) await page.locator('#accept-review').click();
    else { await page.locator('#review-label').fill('第1幕克段'); await page.locator('#review-decision-form button[type=submit]').click(); }
    await idle();
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 1);
    assert.equal(Number(await page.locator('#metric-pending').textContent()), pending - 1);
    assert.equal(await page.evaluate(photo => Object.keys(localStorage).some(key =>
      key.startsWith('theatre-review-draft-v1:') && JSON.parse(key.slice('theatre-review-draft-v1:'.length))[2] === photo), firstPhoto), false);
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
    await page.route('**/api/organize/tasks', async route => {
      organizePayload = route.request().postDataJSON();
      await route.fulfill({status: 202, json: {code: 200, data: {task_id: 'browser-organize', status: 'queued'}}});
    });
    let organizeState = 'running', organizeResume = 0;
    await page.route('**/api/organize/tasks/browser-organize/cancel', async route => {
      organizeState = 'cancelled';
      await route.fulfill({json: {code: 200, data: {status: 'cancel_requested'}}});
    });
    await page.route('**/api/organize/tasks/browser-organize/resume', async route => {
      organizeState = 'completed_with_errors'; organizeResume++;
      await route.fulfill({status: 202, json: {code: 200, data: {status: 'queued'}}});
    });
    await page.route('**/api/organize/tasks/browser-organize', route => route.fulfill({json: {code: 200, data: {
      status: organizeState, succeeded: 6, failed: organizeState === 'completed_with_errors' ? 1 : 0,
      pending: organizeState === 'completed_with_errors' ? 0 : total - 6,
      result: organizeState === 'completed_with_errors' ? {total_input: total, kept: 6, discarded: total - 7, failed: 1, n_scenes: 3,
        report_path: path.join(fixture, 'organize_report.json')} : null}}}));
    await page.locator('#identity-registry-file').fill(path.join(fixture, 'identity_registry.json'));
    await page.locator('#organize-form button[type=submit]').click();
    await idle();
    await page.locator('#analysis-cancel').waitFor();
    await page.reload(); await idle();
    await page.locator('#analysis-cancel').waitFor();
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('theatre-workspace-v1')).analysisKind), 'organize');
    await page.locator('#analysis-cancel').click();
    await page.locator('#analysis-resume').waitFor();
    await page.locator('#analysis-resume').click();
    await page.locator('#organize-result').waitFor();
    await idle();
    assert.equal(organizePayload.kind, 'organize');
    assert.equal(organizePayload.params.classify_role, false);
    assert.equal(organizePayload.params.identity_registry_file, path.join(fixture, 'identity_registry.json'));
    assert.equal(await page.locator('#identity-registry-file').inputValue(), path.join(fixture, 'identity_registry.json'));
    assert.match(organizePayload.params.scene_evidence_file, /scene_labels\.json/);
    assert.match(await page.locator('#organize-result').textContent(), /归档完成/);
    assert.match(await page.locator('#organize-result').textContent(), /部分图片失败/);
    assert.equal(organizeResume, 1);
    await page.unroute('**/api/organize/tasks');

    // Exercise the actual cluster response schema and reversible ignore/restore UI.
    await clickView('actors');
    let clusterStatus = 'pending', assignedTo = null, scanFails = false, scanState = 'running', scanSubmissions = 0;
    await page.route('**/api/organize/tasks', async route => {
      assert.equal(route.request().postDataJSON().kind, 'face_cluster'); scanSubmissions++;
      await route.fulfill({json: {code: 200, data: {task_id: 'browser-cluster'}}});
    });
    await page.route('**/api/organize/tasks/browser-cluster', route => route.fulfill({json: {code: 200, data: {
      status: scanFails ? 'failed' : scanState, error: scanFails ? '扫描失败测试' : null,
      succeeded: 2, failed: 0, pending: scanState === 'completed' ? 0 : 1,
      result: scanState === 'completed' ? {n_images: total, n_faces: 5, n_clusters: 1, failed: 0} : null}}}));
    await page.route('**/api/organize/tasks/browser-cluster/cancel', async route => { scanState = 'cancelled'; await route.fulfill({json: {code: 200, data: {status: scanState}}}); });
    await page.route('**/api/organize/tasks/browser-cluster/resume', async route => { scanState = 'completed'; await route.fulfill({json: {code: 200, data: {status: scanState}}}); });
    await page.route('**/api/organize/face_cluster/result?*', route => route.fulfill({json: {code: 200, data: scanState === 'completed' ? {clusters: [{id: 'cluster-01', size: 5, cohesion: .8, status: clusterStatus, assigned_to: assignedTo, suggest_manual: false, preview: path.join(root, 'data/sample_images/2.jpg')}]} : {progress: 20}}}));
    await page.route('**/api/organize/face_cluster/ignore', async route => { clusterStatus = clusterStatus === 'ignored' ? 'pending' : 'ignored'; await route.fulfill({json: {code: 200, data: {}}}); });
    await page.route('**/api/organize/face_cluster/assign', async route => { clusterStatus = 'assigned'; assignedTo = '陈少云（饰颍考叔）'; await route.fulfill({json: {code: 200, data: {}}}); });
    await page.locator('#cluster-button').click(); await idle();
    await page.locator('#analysis-cancel').waitFor();
    await page.reload(); await idle(); await clickView('actors');
    await page.locator('#analysis-cancel').waitFor();
    assert.equal(scanSubmissions, 1, 'cluster refresh reuses task without resubmission');
    await page.locator('#analysis-cancel').click(); await page.locator('#analysis-resume').waitFor();
    await page.locator('#analysis-resume').click(); await page.locator('.cluster-card').waitFor(); await idle();
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
    await page.waitForFunction(() => document.querySelector('#analysis-task-message').textContent.includes('扫描失败测试'));
    assert.match(await page.locator('#analysis-task-message').textContent(), /扫描失败测试/);
    assert.equal(await page.locator('#cluster-button').isEnabled(), true);

    await page.reload(); await idle();
    assert.equal(await page.locator('.cluster-card').count(), 1, 'completed cluster decisions survive refresh');
    await page.locator('#analysis-new').click(); await idle();
    await page.unroute('**/api/organize/tasks');
    let databaseState = 'running', databaseCreates = 0, databaseDry = true;
    await page.route('**/api/organize/tasks', async route => {
      const body = route.request().postDataJSON();
      assert.equal(body.kind, 'face_database');
      assert.equal(body.params.organized_dirs, photos);
      assert.equal(body.params.output_dir, path.join(fixture, 'face_database'));
      databaseDry = body.params.dry_run; databaseCreates++;
      await route.fulfill({status: 202, json: {code: 200, data: {task_id: `browser-database-${databaseCreates}`}}});
    });
    await page.route('**/api/organize/tasks/browser-database-*', route => route.fulfill({json: {code: 200, data: {
      status: databaseState, succeeded: 2, failed: databaseState === 'completed_with_errors' ? 1 : 0,
      pending: databaseState === 'running' ? 3 : 0,
      result: databaseState.startsWith('completed') ? {scanned: total, added: 2, failed: databaseState === 'completed_with_errors' ? 1 : 0,
        dry_run: databaseDry, added_persons: ['陈少云（饰颍考叔）'], report_path: path.join(fixture, 'face_database/face_database_report.json')} : null}}}));
    await page.route('**/api/organize/tasks/browser-database-*/cancel', async route => {
      databaseState = 'cancelled'; await route.fulfill({json: {code: 200, data: {status: databaseState}}});
    });
    await page.route('**/api/organize/tasks/browser-database-*/resume', async route => {
      databaseState = 'completed_with_errors'; await route.fulfill({json: {code: 200, data: {status: 'queued'}}});
    });
    await clickView('actors');
    await page.locator('#actor-source').fill(photos); await page.locator('#actor-dry-run').check();
    await page.locator('#actor-build-form button[type=submit]').click(); await idle();
    await page.locator('#analysis-cancel').waitFor();
    await page.reload(); await idle(); await clickView('actors');
    await page.locator('#analysis-cancel').waitFor();
    assert.equal(databaseCreates, 1, 'database refresh does not resubmit');
    await page.locator('#analysis-cancel').click(); await page.locator('#analysis-resume').waitFor();
    await page.locator('#analysis-resume').click();
    await page.waitForFunction(() => document.querySelector('#actor-result').textContent.includes('照片检查完成'));
    await idle();
    assert.match(await page.locator('#actor-result').textContent(), /可用锚定照片/);
    assert.match(await page.locator('#actor-result').textContent(), /部分图片处理失败/);
    assert.equal(await page.getByRole('link', {name: '下载人脸建库报告 ↗'}).count(), 1);
    await page.reload(); await idle(); await clickView('actors');
    await page.locator('#actor-result').waitFor();
    assert.match(await page.locator('#actor-result').textContent(), /照片检查完成/);
    assert.equal(databaseCreates, 1, 'completed database report restores without resubmission');
    databaseState = 'completed';
    await page.locator('#actor-source').fill(photos); await page.locator('#actor-dry-run').uncheck();
    await page.locator('#actor-build-form button[type=submit]').click();
    await page.waitForFunction(() => document.querySelector('#actor-result').textContent.includes('演员库已更新'));
    await idle();
    assert.equal(databaseCreates, 2);
    assert.match(await page.locator('#actor-result').textContent(), /新增锚定照片/);
    await page.unroute('**/api/organize/tasks');
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
    const rows = (await (await context.request.get(base + '/api/organize/scene/review?' +
      new URLSearchParams({review_dir: path.join(fixture, 'review'), limit: 100}))).json()).data.records;
    await post('/api/organize/scene/review', {review_dir: path.join(fixture, 'review'), image: rows[0].image, action: 'modify', label: malicious});
    await page.reload(); await idle(); await clickView('photos'); await page.locator('.photo-card').first().click();
    assert.equal(await page.evaluate(() => window.workspaceXss), undefined);
    assert.equal(await page.locator('#inspector img').count(), 1);

    // A queue larger than the API's page size must not lose off-screen records.
    const longPhotos = Array.from({length: 10000}, (_, index) => ({name: `photo-${String(index).padStart(3, '0')}.jpg`, path: `${photos}/photo-${String(index).padStart(3, '0')}.jpg`, size: 100, metadata: {people: []}}));
    const longRecords = longPhotos.map(photo => ({image: photo.path, label: '第1幕克段', scene_id: 'act-01', review: {status: 'pending'}}));
    const offsets = [];
    await page.route('**/api/workspace/photos?*', route => route.fulfill({json: {code: 200, data: {input_dir: photos, photos: longPhotos, total: 10000, truncated: false}}}));
    await page.route('**/api/organize/scene/review?*', route => {
      const offset = Number(new URL(route.request().url()).searchParams.get('offset'));
      offsets.push(offset);
      return route.fulfill({json: {code: 200, data: {records: longRecords.slice(offset, offset + 200), summary: {pending: 10000, approved: 0, rejected: 0}}}});
    });
    await page.reload(); await idle(); await clickView('photos');
    assert.equal(Number(await page.locator('#metric-pending').textContent()), 10000);
    assert.deepEqual(offsets, Array.from({length: 51}, (_, index) => index * 200));
    assert.equal(await page.locator('.photo-card').count(), 36);
    await page.locator('#next-page').click();
    assert.equal(await page.locator('#page-number').textContent(), '2 / 278');
    await page.locator('#photo-search').fill('photo-9999');
    assert.equal(await page.locator('.photo-card').count(), 1);
    await page.locator('.photo-card').click();
    const queriesBeforeDecision = offsets.length;
    let decisionPosts = 0;
    await page.route('**/api/organize/scene/review', route => {
      decisionPosts++;
      return route.fulfill({json: {code: 200, data: {
        record: {...longRecords[9999], review: {status: 'approved'}},
        summary: {pending: 9999, approved: 1, rejected: 0}}}});
    });
    await page.locator('#accept-review').click(); await idle();
    assert.equal(decisionPosts, 1);
    assert.equal(offsets.length, queriesBeforeDecision, 'a review decision must not download the queue again');
    assert.equal(Number(await page.locator('#metric-pending').textContent()), 9999);
    assert.equal(Number(await page.locator('#metric-approved').textContent()), 1);
    assert.equal(await page.locator('.photo-card').count(), 1);
    await page.locator('#import-button').click();
    assert.equal(await page.locator('#input-dir').inputValue(), '', 'switching projects starts with fresh context');
    assert.notEqual(await page.locator('#output-dir').inputValue(), fixture);
    await page.keyboard.press('Escape');
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log(JSON.stringify({passed: true, realPhotos: total, checks: ['empty/error states', 'real photo previews', 'knowledge ingest', 'offline scene analysis', 'background analysis refresh reuses task without resubmission', 'review accept/modify/reject', 'draft survives selection/search/reload/failure', 'revision conflict refreshes record and preserves draft', 'successful decision clears draft', 'search/filter', 'label download', 'organize payload', 'background organize refresh/cancel/resume/partial failures', 'cluster refresh/cancel/resume and persistent assign/ignore/restore/error', 'face database refresh/cancel/resume/partial report and dry-run/build payload', 'refresh persistence', 'responsive 1440/1024/768/540/390', 'escaped evidence', '10000 record pagination and bounded DOM', 'decision updates one record without queue reload', 'independent project contexts'], screenshots: shots}, null, 2));
  } finally {
    await browser.close();
    fs.rmSync(fixture, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
