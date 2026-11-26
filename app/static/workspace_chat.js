'use strict';

function createWorkspaceChat({api, fileURL, notify, state, openReview}) {
  const element = id => document.getElementById(`chat-${id}`);
  const node = (tag, text) => { const result = document.createElement(tag); result.textContent = text; return result; };
  const sessions = new Map();
  const labels = {submitted: '已提交，尚未完成', queued: '排队中', running: '执行中', cancel_requested: '正在取消',
    succeeded: '执行成功', completed: '已完成', completed_with_errors: '部分失败', partial: '部分失败',
    interrupted: '已中断', cancelled: '已取消', failed: '失败'};
  const reports = new Set(['scene_labels.json', 'organize_report.json', 'actor_view_report.json', 'face_database_report.json']);
  let active = null, timer, pollVersion = 0;

  function controls() {
    const disabled = !active || active.busy || state.busy;
    for (const id of ['message', 'mode', 'send', 'refresh', 'older']) element(id).disabled = disabled;
    element('older').hidden = !active?.before;
  }

  function activate(project) {
    const key = 'theatre-chat-v1:' + JSON.stringify([project.input, project.output]);
    let id = sessions.get(key);
    try { id = id || localStorage.getItem(key); } catch (_) {}
    active = {project, id, key, tasks: {}, artifacts: {}, live: {}, before: null, busy: false};
    element('context').textContent = `当前项目：${project.name} · ${project.input}`;
    element('message').value = ''; element('log').replaceChildren(); element('workflow').replaceChildren();
    element('status').textContent = '';
    viewChanged();
  }

  function messageRow(row) {
    const item = document.createElement('article');
    const title = {user: '你', assistant: '项目助手', tool: '工具步骤'}[row.role] || '记录';
    item.append(node('strong', title + (row.fallback ? ' · 降级回答' : '')));
    if (row.role !== 'tool') item.append(node('p', row.content));
    else {
      let call;
      try { call = JSON.parse(row.content); } catch (_) {}
      const status = call?.outcome?.status || (call?.ok === false ? 'failed' : '');
      const details = document.createElement('details');
      details.append(node('summary', `${row.tool || call?.tool || '工具'} · ${labels[status] || '查看记录'}`),
        node('pre', row.content));
      item.append(details);
    }
    return item;
  }

  function renderWorkflow(chat) {
    const list = element('workflow'); list.replaceChildren();
    const artifacts = new Map();
    function artifact(kind, path) { if (typeof path === 'string' && path) artifacts.set(path, kind); }
    for (const item of Object.values(chat.artifacts)) artifact(item.kind, item.path);
    for (const [id, record] of Object.entries(chat.tasks)) {
      const live = chat.live[id], status = live?.status || record.outcome?.status;
      const item = node('div', `${record.tool} · ${labels[status] || status} · ${id}`);
      if (live?.total != null) item.append(node('p', `完成 ${live.succeeded} / ${live.total}，失败 ${live.failed}`));
      if (live?.error) item.append(node('p', live.error));
      if (live?.pollError) item.append(node('p', '查询失败：' + live.pollError));
      for (const ref of record.outcome?.artifacts || []) artifact(ref.kind, ref.path);
      for (const kind of ['output_file', 'report_path', 'output_dir', 'review_dir']) artifact(kind, live?.result?.[kind]);
      const evidence = live?.result?.output_file || record.outcome?.artifacts?.find(ref => ref.kind === 'output_file')?.path;
      if (record.tool === 'scene_reason' && evidence && ['completed', 'completed_with_errors', 'succeeded', 'partial'].includes(status)) {
        const button = node('button', '将场景证据载入复核'); button.className = 'button secondary';
        button.onclick = () => openReview(chat.project.review, evidence); item.append(button);
      }
      list.append(item);
    }
    for (const [path, kind] of artifacts) {
      const item = node('div', path);
      if (kind === 'review_dir') {
        const button = node('button', '打开场景复核'); button.className = 'button secondary';
        button.onclick = () => openReview(path); item.append(button);
      } else if (reports.has(path.split('/').pop())) {
        const link = node('a', '下载报告'); link.href = fileURL(path); item.append(link);
      }
      list.append(item);
    }
    if (!list.childNodes.length) list.append(node('p', '暂无关联任务或产物。'));
  }

  async function poll(chat, version) {
    if (chat !== active || version !== pollVersion || state.view !== 'chat') return;
    const pending = Object.keys(chat.tasks).filter(id => ['submitted', 'queued', 'running', 'cancel_requested'].includes(
      chat.live[id]?.status || chat.tasks[id].outcome?.status));
    if (!pending.length) return;
    const updates = await Promise.all(pending.map(async id => {
      try { return [id, await api(`/api/organize/tasks/${encodeURIComponent(id)}`)]; }
      catch (error) { return [id, {...chat.live[id], pollError: error.message}]; }
    }));
    if (chat !== active || version !== pollVersion || state.view !== 'chat') return;
    Object.assign(chat.live, Object.fromEntries(updates));
    renderWorkflow(chat); timer = setTimeout(() => poll(chat, version), 3000);
  }

  function workflow(chat, source) {
    if ((source.input_dir || source.memory?.input_dir) !== chat.project.input) source = {};
    chat.tasks = source.tasks || {}; chat.artifacts = source.artifacts || {}; chat.live = {};
    renderWorkflow(chat); clearTimeout(timer); poll(chat, ++pollVersion);
  }

  async function historyPage(older = false) {
    const chat = active;
    if (!chat || chat.busy) return;
    if (!chat.id) { element('status').textContent = '暂无对话记录。'; return; }
    chat.busy = true; controls(); element('status').textContent = '正在读取记录…';
    try {
      const query = new URLSearchParams({limit: '24'});
      if (older) query.set('before', chat.before);
      const page = await api(`/api/organize/agent/session/${encodeURIComponent(chat.id)}/history?${query}`);
      if (chat !== active) return;
      const rows = page.items.map(messageRow);
      if (older) element('log').prepend(...rows); else element('log').replaceChildren(...rows);
      chat.before = page.next_before;
      if (!older) workflow(chat, page);
      element('status').textContent = page.items.length ? '' : '暂无对话记录。';
    } catch (error) {
      if (chat === active) element('status').textContent = error.status === 404 ? '暂无对话记录。' : `读取失败：${error.message}`;
    } finally { chat.busy = false; if (chat === active) controls(); }
  }

  function viewChanged() {
    clearTimeout(timer); pollVersion++; controls();
    if (state.view === 'chat') historyPage();
  }

  element('older').onclick = () => historyPage(true);
  element('refresh').onclick = () => historyPage();
  element('form').onsubmit = async event => {
    event.preventDefault();
    const chat = active, message = element('message').value.trim();
    if (!chat || chat.busy || state.busy || !message) return;
    if (!chat.id) {
      chat.id = Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, '0')).join('');
      sessions.set(chat.key, chat.id);
      try { localStorage.setItem(chat.key, chat.id); } catch (_) { notify('浏览器无法保存会话编号；刷新后无法自动接续。'); }
    }
    const project = chat.project, offline = element('mode').value === 'offline';
    const context = {input_dir: project.input, workspace_output_dir: project.output, play: project.name,
      knowledge_dir: project.knowledge, review_dir: project.review, db_root: project.db};
    for (const [key, value] of Object.entries({info_path: project.info, play_root: project.playRoot,
      scene_evidence_file: project.labels, identity_registry_file: project.identityRegistry, image_path: state.selected})) {
      if (value) context[key] = value;
    }
    chat.busy = true; controls(); element('status').textContent = '正在处理本轮请求…';
    try {
      const result = await api(`/api/organize/agent/${offline ? '' : 'advanced/'}chat`, {message, session_id: chat.id,
        context_json: JSON.stringify(context), ...(offline ? {use_llm: false} : {})});
      if (chat !== active) return;
      element('message').value = '';
      element('log').append(messageRow({role: 'user', content: message}), ...(result.tool_calls || []).map(call =>
        messageRow({role: 'tool', tool: call.tool, content: JSON.stringify(call)})),
        messageRow({role: 'assistant', content: result.answer, fallback: result.fallback}));
      element('log').lastChild.scrollIntoView({block: 'nearest'});
      workflow(chat, result.workflow); element('status').textContent = '';
    } catch (error) {
      if (chat === active) element('status').textContent = `请求失败：${error.message}。请刷新记录核对执行状态，不会自动重发。`;
    } finally { chat.busy = false; if (chat === active) controls(); }
  };
  return {activate, controls, viewChanged};
}
