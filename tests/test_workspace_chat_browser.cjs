const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || '/tmp/pwcli/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || '/snap/chromium/current/usr/lib/chromium-browser/chrome',
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport: {width: 390, height: 850}});
    const errors = [], sent = [], pages = [];
    page.on('pageerror', error => errors.push(error.message));
    const project = {input: '/photos/A', output: '/results/A', knowledge: '/kb', db: '/faces', review: '/review/A', name: '项目 A'};
    await page.addInitScript(value => {
      if (!localStorage.getItem('theatre-workspace-v1')) localStorage.setItem('theatre-workspace-v1', JSON.stringify(value));
    }, project);
    const hostile = '<img src=x onerror="window.bad=1">';
    const facts = {tasks: {job: {tool: 'scene_reason', outcome: {status: 'submitted'}}}, artifacts: {
      report_path: {kind: 'report_path', path: '/results/A/organize_report.json'}}};
    let release, seeds = 0;
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url()), apiPath = url.pathname;
      const json = data => route.fulfill({json: {code: 200, data}});
      if (apiPath === '/organize' || apiPath.startsWith('/static/')) return route.fulfill({path: path.join(__dirname,
        '../app', apiPath === '/organize' ? 'templates/workspace.html' : apiPath.slice(1))});
      if (apiPath === '/api/health') return json({});
      if (apiPath === '/api/workspace/photos') return json({input_dir: url.searchParams.get('input_dir'), total: 0, photos: []});
      if (apiPath.endsWith('/play/knowledge')) return route.fulfill({status: 404, json: {message: 'no knowledge'}});
      if (apiPath.endsWith('/review/metadata')) return json({records: []});
      if (apiPath.endsWith('/review/seed')) {
        assert.match(request.postData(), /name="candidate_input_dir"\r\n\r\n\/photos\/A/); seeds++; return json({});
      }
      if (apiPath.endsWith('/tasks/job')) return json({status: 'completed', succeeded: 2, failed: 0, total: 2, result: {output_file: '/results/A/scenes.jsonl'}});
      if (apiPath.endsWith('/history')) {
        pages.push(apiPath);
        const older = url.searchParams.has('before');
        return json({...facts, input_dir: project.input, items: [{role: 'assistant', content: older ? '更早的一轮' : hostile}], next_before: older ? null : 5});
      }
      if (apiPath.endsWith('/chat')) {
        const form = await new Response(request.postDataBuffer(), {headers: {'content-type': request.headers()['content-type']}}).formData();
        sent.push(Object.fromEntries(form));
        if (form.get('message') === '延迟回答') await new Promise(resolve => { release = resolve; });
        if (form.get('message') === '故障') return route.fulfill({status: 503, json: {message: '模型不可用'}});
        return json({session_id: form.get('session_id'), workflow: {...facts, memory: {input_dir: project.input}},
          answer: hostile, tool_calls: [{tool: 'scene_reason', outcome: {status: 'submitted'}}]});
      }
      throw new Error('Unexpected request: ' + apiPath);
    });
    const click = id => page.locator('#' + id).click();
    const send = async message => { await page.locator('#chat-message').fill(message); await click('chat-send'); };
    await page.goto('http://workspace.test/organize#chat');
    await page.waitForFunction(() => !document.querySelector('#chat-send').disabled);
    await send('检查目录');
    await page.getByText(hostile, {exact: true}).waitFor();
    assert.equal(sent.length, 1);
    assert.equal(JSON.parse(sent[0].context_json).input_dir, project.input);
    assert.ok(sent[0].session_id);
    assert.equal(await page.locator('#chat-log img').count(), 0);
    await page.getByText('完成 2 / 2，失败 0').waitFor();
    assert.match(await page.getByText('下载报告').getAttribute('href'), /^\/api\/workspace\/artifact\?/);
    assert.equal(seeds, 0);
    await page.getByText('将场景证据载入复核').click();
    await page.waitForURL('**/organize#review');
    assert.equal(seeds, 1);
    await page.locator('.nav-item[data-view="chat"]').click();
    await click('chat-older'); await page.getByText('更早的一轮').waitFor();
    assert.match(pages.at(-1), new RegExp(sent[0].session_id));
    await page.reload(); await page.getByText(hostile, {exact: true}).waitFor();
    await page.locator('#chat-mode').selectOption('offline'); await send('故障');
    await page.getByText(/请求失败：模型不可用/).waitFor();
    assert.equal(sent[1].session_id, sent[0].session_id); assert.equal(sent[1].use_llm, 'false');
    assert.equal(await page.locator('#chat-message').inputValue(), '故障');
    await send('延迟回答'); await click('import-button');
    await page.locator('#input-dir').fill('/photos/B'); await page.locator('#output-dir').fill('/results/B');
    await page.locator('#project-form button[type="submit"]').click();
    await page.getByText('0 张剧照 · /photos/B', {exact: true}).waitFor();
    release(); await page.locator('.nav-item[data-view="chat"]').click();
    await page.getByText('暂无对话记录。').waitFor();
    assert.equal(await page.locator('#chat-log article').count(), 0);
    await send('B 的第一轮'); await page.getByText(hostile, {exact: true}).waitFor();
    assert.notEqual(sent[3].session_id, sent[0].session_id);
    assert.equal(await page.locator('#chat-workflow button').count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.deepEqual(errors, []);
    console.log('Workspace chat: context, steps, live status, artifacts, review, paging, restore, errors and isolation passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
