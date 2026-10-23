/* The workspace uses the existing organize APIs; project settings stay in this browser. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const icons = {
    images: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m3 17 5-5 4 4 3-3 6 6"/>',
    book: '<path d="M12 5c-3-2-7-2-10-1v15c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 1Zm0 0v15"/>',
    people: '<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 5"/>',
    check: '<rect x="3" y="3" width="18" height="18" rx="4"/><path d="m7 12 3 3 7-7"/>',
    export: '<path d="M12 3v12m-4-4 4 4 4-4M4 14v6h16v-6"/>',
    folder: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7h18"/>',
    sliders: '<path d="M4 7h5m4 0h7M4 17h9m4 0h3"/><circle cx="11" cy="7" r="2"/><circle cx="15" cy="17" r="2"/>',
    'arrow-up-right': '<path d="M6 18 18 6M6 6h12v12"/>',
    'chevron-right': '<path d="m9 5 7 7-7 7"/>',
    drive: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 13h18M16 16h2"/>',
    plus: '<path d="M12 4v16M4 12h16"/>',
    refresh: '<path d="M20 7a9 9 0 0 0-15-2L2 8m0-5v5h5m-3 9a9 9 0 0 0 15 2l3-3m0 5v-5h-5"/>',
    sparkles: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Zm7-2v4m-2-2h4"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    layers: '<path d="m12 2 10 6-10 6L2 8l10-6ZM2 12l10 6 10-6M2 16l10 6 10-6"/>',
    focus: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><circle cx="12" cy="12" r="4"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>'
  };
  const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.images}</svg>`;
  document.querySelectorAll('[data-icon]').forEach(el => { el.outerHTML = icon(el.dataset.icon); });
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const join = (root, name) => `${root.replace(/\/+$/, '')}/${name}`;
  const basename = name => String(name || '').split('/').pop();
  const projectKey = 'theatre-workspace-v1';
  const drafts = new Map();
  let draftStorageWarning = false;
  function draftKey(photo) {
    return 'theatre-review-draft-v1:' + JSON.stringify([state.project.input, state.project.review, photo.path]);
  }
  function readDraft(photo) {
    const key = draftKey(photo);
    if (drafts.has(key)) return drafts.get(key);
    try {
      const draft = JSON.parse(localStorage.getItem(key));
      if (draft && typeof draft.label === 'string' && typeof draft.note === 'string') {
        drafts.set(key, draft); return draft;
      }
    } catch (_) { /* Keep the form available if browser storage is unavailable. */ }
    return null;
  }
  function writeDraft(photo) {
    const draft = {label: $('review-label').value, note: $('review-note').value};
    const key = draftKey(photo);
    drafts.set(key, draft);
    $('review-draft-status').hidden = false;
    try { localStorage.setItem(key, JSON.stringify(draft)); }
    catch (_) {
      if (!draftStorageWarning) notice('浏览器无法持久保存草稿；本页切图会保留输入，刷新前请保存修订。', true);
      draftStorageWarning = true;
    }
  }
  function clearDraft(photo) {
    const key = draftKey(photo);
    drafts.delete(key);
    try { localStorage.removeItem(key); } catch (_) { /* Current-page state is still cleared. */ }
  }
  function mergeReview(previous, record) {
    const updated = {...record, ...(previous.source_image ? {source_image: previous.source_image} : {})};
    state.records.set(previous.image, updated);
    const before = previous.review?.status || 'pending', after = updated.review?.status || 'pending';
    if (before !== after) {
      state.summary[before] = Math.max(0, (state.summary[before] || 0) - 1);
      state.summary[after] = (state.summary[after] || 0) + 1;
    }
  }
  const state = {project: null, photos: [], records: new Map(), knowledge: null, summary: null,
    selected: null, view: 'photos', filter: 'all', query: '', page: 0, busy: false, taskId: null, generation: 0,
    reviewPending: new Map()};
  const pageSize = 36;
  const titles = {photos: '影像资料', knowledge: '剧目与剧情', actors: '演员与角色', review: '场景复核', export: '整理与导出'};
  const statuses = {pending: '待复核', approved: '已确认', rejected: '已拒绝', unprocessed: '待分析'};
  const evidenceNames = {filename: '文件名线索', knowledge: '剧情依据', rule: '文件名规则', person: '人物线索', people: '人物线索', visual: '图文相似度', observation: '画面观察', ocr: '可见文字识别', language: '语言模型', time: '时间线索'};

  async function api(url, data) {
    const options = {};
    if (data) {
      options.method = 'POST';
      options.body = new FormData();
      Object.entries(data).forEach(([key, value]) => { if (value !== undefined && value !== null) options.body.set(key, String(value)); });
    }
    const response = await fetch(url, options);
    let result;
    try { result = await response.json(); } catch (_) { throw new Error('服务未返回有效结果，请检查服务状态后重试。'); }
    if (!response.ok || (result.code !== undefined && result.code !== 200)) {
      let message = result.message || result.detail || `请求失败（${response.status}）`;
      if (Array.isArray(message)) message = message.map(item => `${item.loc?.slice(-1)[0] || ''}：${item.msg}`).join('；');
      if (result.available?.length) message += `；可用演员：${result.available.join('、')}`;
      const error = new Error(message); error.status = response.status; throw error;
    }
    return result.data ?? result;
  }
  function notice(message, error = false) {
    $('notice').className = `notice${error ? ' error' : ''}`;
    $('notice').innerHTML = `<button aria-label="关闭提示">×</button><span>${esc(message)}</span>`;
    $('notice').hidden = false;
    $('notice').querySelector('button').onclick = () => { $('notice').hidden = true; };
  }
  function progress(message) { $('task-status').textContent = message; }
  async function task(message, operation) {
    if (state.busy) return;
    state.busy = true;
    $('notice').hidden = true;
    $('task-status').hidden = false;
    progress(message);
    document.body.setAttribute('aria-busy', 'true');
    const controls = [...document.querySelectorAll('button:not([data-view]):not([data-filter]):not([data-photo]):not([data-close]):not([data-task-control]), input, select')];
    const previous = controls.map(el => el.disabled);
    controls.forEach(el => { el.disabled = true; });
    try { return await operation(); }
    catch (error) { notice(error.message || '操作失败，请稍后重试。', true); }
    finally {
      controls.forEach((el, i) => { if (el.isConnected) el.disabled = previous[i]; });
      state.busy = false;
      $('task-status').hidden = true;
      document.body.removeAttribute('aria-busy');
      updateControls();
      renderInspector();
    }
  }
  function needProject() {
    if (state.project) return true;
    openProject();
    return false;
  }
  function saveProject() {
    try { localStorage.setItem(projectKey, JSON.stringify(state.project)); }
    catch (_) { notice('浏览器无法保存项目设置；当前工作仍可继续。'); }
  }
  function thumb(photo, size = 640) { return `/api/workspace/thumbnail?${new URLSearchParams({input_dir: state.project.input, name: photo.name, size})}`; }
  const fileURL = path => {
    const name = basename(path), root = path.slice(0, path.lastIndexOf('/')) || '.';
    return /\.(jpg|jpeg|png|bmp|tiff)$/i.test(name)
      ? `/api/workspace/thumbnail?${new URLSearchParams({input_dir: root, name})}`
      : `/api/workspace/artifact?${new URLSearchParams({output_dir: root, name})}`;
  };
  function recordFor(photo) { return state.records.get(photo.path) || state.records.get(photo.name); }
  function statusFor(photo) { return recordFor(photo)?.review?.status || (recordFor(photo) ? 'pending' : 'unprocessed'); }
  function filteredPhotos() {
    return state.photos.filter(photo => {
      const record = recordFor(photo);
      return (state.filter === 'all' || statusFor(photo) === state.filter) &&
        (!state.query || [photo.name, record?.label, ...((photo.metadata.people || []).flat())].join(' ').toLowerCase().includes(state.query));
    });
  }
  function updateControls() {
    if (state.busy) return;
    ['analyze-button', 'knowledge-analyze', 'cluster-button'].forEach(id => { $(id).disabled = !state.photos.length; });
    $('export-labels').disabled = !state.summary || !state.summary.approved;
    $('reload-button').disabled = !state.project;
    const totalPages = Math.ceil(filteredPhotos().length / pageSize);
    $('prev-page').disabled = state.page <= 0;
    $('next-page').disabled = state.page >= totalPages - 1;
  }
  function renderHeader() {
    const project = state.project;
    $('project-title').textContent = project?.name || '开启一场影像整理';
    $('project-subtitle').textContent = project ? `${state.photos.length} 张剧照 · ${project.input}` : '连接剧照、人物与剧情，建立清晰可追溯的戏剧影像档案。';
    $('import-button').innerHTML = icon(project ? 'folder' : 'plus') + (project ? '切换剧照目录' : '导入剧照目录');
    const pending = state.summary?.pending ?? 0;
    const approved = state.summary?.approved ?? 0;
    $('nav-total').textContent = project ? state.photos.length : '—';
    $('nav-pending').textContent = state.summary ? pending : '—';
    $('metric-total').textContent = project ? state.photos.length : '—';
    $('metric-pending').textContent = state.summary ? pending : '—';
    $('metric-approved').textContent = state.summary ? approved : '—';
    $('metric-scenes').textContent = state.knowledge ? state.knowledge.scenes.length : '—';
    $('metric-scenes-note').textContent = state.knowledge ? '已关联剧情' : '等待关联剧情';
    $('filter-pending').textContent = pending;
    $('step-photos').textContent = project ? `已汇集 ${state.photos.length} 张剧照` : '选择本次整理的剧照';
    $('step-knowledge').textContent = state.knowledge ? `已关联 ${state.knowledge.scenes.length} 个幕次` : '用剧目资料辅助场景分析';
    $('step-review').textContent = state.summary ? `已确认 ${approved} · 待复核 ${pending}` : '确认幕次，生成规范档案';
    $('export-summary').innerHTML = state.summary ? `${approved} <small>已确认 · ${pending} 待复核 · ${state.summary.rejected || 0} 已拒绝</small>` : '尚未载入复核队列';
    $('output-location').textContent = project ? `保存位置：${join(project.output, 'organized')}` : '请先选择剧照目录。';
    document.querySelectorAll('.workflow button').forEach((el, index) => el.classList.toggle('done', [!!project, !!state.knowledge, approved > 0][index]));
    updateControls();
  }
  function setView(view) {
    if (!titles[view]) view = 'photos';
    const changed = state.view !== view;
    state.view = view;
    document.querySelectorAll('.view').forEach(el => { el.hidden = true; });
    $(view === 'photos' || view === 'review' ? 'gallery-view' : `${view}-view`).hidden = false;
    document.querySelectorAll('.nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.view === view);
      if (el.dataset.view === view) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
    });
    $('breadcrumb-view').textContent = titles[view];
    $('gallery-title').textContent = view === 'review' ? '场景标签人工复核' : '影像资料';
    $('gallery-caption').textContent = view === 'review' ? '对照剧照与依据，确认它属于哪一幕' : '浏览本次剧照，关联人物与故事';
    if (changed && (view === 'review' || view === 'photos')) { state.filter = view === 'review' ? 'pending' : 'all'; state.page = 0; }
    history.replaceState(null, '', `#${view}`);
    renderGallery();
  }
  function renderGallery(hydrate = true) {
    const filtered = filteredPhotos();
    state.page = Math.min(state.page, Math.max(0, Math.ceil(filtered.length / pageSize) - 1));
    document.querySelectorAll('[data-filter]').forEach(el => {
      el.classList.toggle('active', el.dataset.filter === state.filter);
      el.setAttribute('aria-pressed', String(el.dataset.filter === state.filter));
    });
    $('gallery-empty').hidden = state.photos.length > 0;
    $('no-matches').hidden = !state.photos.length || filtered.length > 0;
    $('gallery-pagination').hidden = !filtered.length;
    $('gallery-count').textContent = `显示 ${state.page * pageSize + 1}–${Math.min((state.page + 1) * pageSize, filtered.length)} / ${filtered.length} 张`;
    $('page-number').textContent = `${state.page + 1} / ${Math.max(1, Math.ceil(filtered.length / pageSize))}`;
    $('photo-grid').innerHTML = filtered.slice(state.page * pageSize, (state.page + 1) * pageSize).map(photo => {
      const record = recordFor(photo), status = statusFor(photo), meta = photo.metadata;
      const label = record?.label && record.label !== 'unknown' ? record.label : (meta.scene || '场景待确认');
      return `<button class="photo-card ${state.selected === photo.path ? 'selected' : ''}" data-photo="${esc(photo.path)}" aria-label="查看 ${esc(photo.name)}" aria-pressed="${state.selected === photo.path}"><div class="photo-image"><img src="${esc(thumb(photo))}" alt="${esc(photo.name)}" loading="lazy"><span class="photo-status ${status}">${statuses[status] || '待复核'}</span>${state.selected === photo.path ? `<span class="photo-selected">${icon('check')}</span>` : ''}</div><div class="photo-caption"><strong title="${esc(photo.name)}">${esc(photo.name)}</strong><span>${esc(label)}</span><footer><span>${esc((meta.people || []).map(person => person[0]).join('、') || '人物待识别')}</span><span>${esc(photo.name.split('.').pop().toUpperCase())}</span></footer></div></button>`;
    }).join('');
    $('photo-grid').querySelectorAll('[data-photo]').forEach(button => { button.onclick = () => selectPhoto(button.dataset.photo, true); });
    updateControls();
    renderInspector();
    if (hydrate) hydrateVisibleReviews();
  }
  const inspectorPlaceholder = $('inspector-body').innerHTML;
  function selectPhoto(path, scroll = false) {
    state.selected = path;
    state.project.selected = path; saveProject();
    renderGallery();
    if (scroll && window.innerWidth <= 720) $('inspector').scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  function renderInspector() {
    const photo = state.photos.find(item => item.path === state.selected);
    const body = $('inspector-body');
    if (!photo) { body.className = 'inspector-placeholder'; body.innerHTML = inspectorPlaceholder; return; }
    const record = recordFor(photo), meta = photo.metadata, status = statusFor(photo);
    body.className = 'inspector-detail';
    body.innerHTML = `<button class="detail-preview" id="preview-photo" aria-label="放大剧照"><img src="${esc(thumb(photo, 960))}" alt="${esc(photo.name)}"><span>查看大图 ↗</span></button><h3 class="detail-name">${esc(photo.name)}</h3><span class="tag">${statuses[status]}</span><dl class="detail-meta"><dt>人物线索</dt><dd>${esc((meta.people || []).map(person => `${person[0]} 饰 ${person[1]}`).join('、') || '暂无文件名标注')}</dd><dt>拍摄日期</dt><dd>${esc(meta.date || '未记录')}</dd><dt>剧场</dt><dd>${esc(meta.venue || '未记录')}</dd><dt>摄影</dt><dd>${esc(meta.photographer || '未记录')}</dd></dl>`;
    $('preview-photo').onclick = () => {
      $('lightbox-name').textContent = photo.name;
      $('lightbox-image').src = thumb(photo, 1920);
      $('lightbox-image').alt = photo.name;
      $('image-dialog').showModal();
    };
    if (record && !record._metadata_only) {
      const knownLabel = record.label && record.label !== 'unknown';
      const options = state.knowledge?.scene_options || [];
      const confidence = Number.isFinite(record.confidence) ? `${Math.round(record.confidence * 100)}% 证据分` : '证据分未提供';
      body.insertAdjacentHTML('beforeend', `<div class="detail-section"><h4>场景判断 <span class="confidence">${esc(confidence)}</span></h4><div class="scene-choice">${esc(knownLabel ? record.label : '尚不能确定幕次')}</div><p>${record.review?.status === 'approved' ? '此标签已经过人工确认。' : '候选结果需对照画面确认，证据分不是正确概率。'}</p><div>${(record.evidence_types || []).map(type => `<span class="tag">${esc(type === 'visual' && record.retrieval_mode !== 'visual' ? '候选文本匹配' : evidenceNames[type] || type)}</span>`).join('')}</div>${(record.alternatives || []).length ? '<h4>其他幕次候选</h4><div class="alternatives">' + record.alternatives.map((item, index) => `<button class="alternative-button" data-alternative="${index}" ${state.busy ? 'disabled' : ''}>${esc(item.label)}</button>`).join('') + '</div>' : ''}</div><details class="detail-section"><summary>判断依据 · ${(record.evidence || []).length} 条</summary>${(record.evidence || []).length ? record.evidence.map(item => `<div class="evidence-item"><strong>${esc(evidenceNames[item.type] || item.type || '参考依据')}</strong><p>${esc(item.quote || '未提供文本摘录')}</p><small>${esc(item.source_ref || item.ref || item.id || '')}</small></div>`).join('') : '<p>没有可引用的依据，请结合原图与剧情手动判断。</p>'}</details><form id="review-decision-form" class="detail-section"><h4>确认与修订</h4><label for="review-label">正确的场景标签</label><input id="review-label" list="scene-options" value="${esc(knownLabel ? record.label : '')}" placeholder="选择或输入幕次名称" ${state.busy ? 'disabled' : ''}><datalist id="scene-options">${options.map(item => `<option value="${esc(item.label)}"></option>`).join('')}</datalist><label for="review-note">复核备注（可选）</label><input id="review-note" placeholder="记录修改原因" ${state.busy ? 'disabled' : ''}><div class="review-actions"><button type="button" class="button primary" id="accept-review" ${!knownLabel || state.busy ? 'disabled' : ''}>确认当前标签</button><button type="submit" class="button secondary" ${state.busy ? 'disabled' : ''}>保存修订</button><button type="button" class="button subtle" id="reject-review" ${state.busy ? 'disabled' : ''}>拒绝此判断</button></div></form>`);
      $('review-decision-form').insertAdjacentHTML('afterbegin', '<p id="review-draft-status" hidden>未保存的修订草稿</p>');
      const draft = readDraft(photo);
      if (draft) {
        $('review-label').value = draft.label; $('review-note').value = draft.note;
        $('review-draft-status').hidden = false;
      }
      $('review-label').oninput = $('review-note').oninput = () => writeDraft(photo);
      body.querySelectorAll('[data-alternative]').forEach(button => { button.onclick = () => {
        $('review-label').value = record.alternatives[Number(button.dataset.alternative)].label;
        writeDraft(photo);
      }; });
      $('accept-review').onclick = () => decide(record, 'accept');
      $('reject-review').onclick = () => decide(record, 'reject');
      $('review-decision-form').onsubmit = event => { event.preventDefault(); decide(record, 'modify'); };
    } else if (record?._metadata_only) {
      body.insertAdjacentHTML('beforeend', '<div class="detail-section"><h4>正在读取这张照片的复核依据…</h4></div>');
    } else {
      body.insertAdjacentHTML('beforeend', '<div class="detail-section"><h4>场景尚未分析</h4><p>关联剧目知识库后，点击「分析场景」生成幕次候选和依据。</p><button class="button secondary" id="detail-knowledge">关联剧目资料 →</button></div>');
      $('detail-knowledge').onclick = () => setView('knowledge');
    }
    const filtered = filteredPhotos(), index = filtered.findIndex(item => item.path === photo.path);
    body.insertAdjacentHTML('beforeend', `<div class="detail-pagination"><button class="button subtle" id="detail-prev" ${index <= 0 ? 'disabled' : ''}>← 上一张</button><button class="button subtle" id="detail-next" ${index < 0 || index >= filtered.length - 1 ? 'disabled' : ''}>下一张 →</button></div>`);
    $('detail-prev').onclick = () => moveSelection(-1);
    $('detail-next').onclick = () => moveSelection(1);
  }
  function moveSelection(delta) {
    const filtered = filteredPhotos(), index = filtered.findIndex(photo => photo.path === state.selected);
    const target = filtered[index + delta];
    if (target) { state.page = Math.floor((index + delta) / pageSize); selectPhoto(target.path); }
  }

  async function loadReviews() {
    const project = state.project, generation = state.generation;
    const result = await api(`/api/organize/scene/review/metadata?${new URLSearchParams({review_dir: project.review})}`);
    const rows = result.records;
    if (generation !== state.generation) return;
    // Match full paths first. Basenames only support legacy evidence that stored no path.
    const currentPaths = new Set(state.photos.map(photo => photo.path));
    const names = new Set(state.photos.map(photo => photo.name));
    const matching = rows.filter(record => currentPaths.has(record.source_image || record.image) || (!record.source_image && record.image === basename(record.image) && names.has(record.image)));
    state.records = new Map(matching.map(record => {
      const previous = state.records.get(record.image);
      return [record.image, previous && previous.review_revision >= record.review_revision ? previous : record];
    }));
    state.summary = {pending: 0, approved: 0, rejected: 0};
    state.records.forEach(record => { const status = record.review?.status || 'pending'; state.summary[status] = (state.summary[status] || 0) + 1; });
    state.queueMismatch = rows.length !== matching.length;
    if (state.queueMismatch) notice('该复核目录包含不属于当前照片目录的记录。请在项目设置中选择此剧目专用的复核目录；已暂停标签导出。', true);
    renderHeader(); renderGallery();
    await hydrateVisibleReviews();
  }
  function hydrateVisibleReviews() {
    if (!state.project) return Promise.resolve();
    const project = state.project, generation = state.generation;
    const photos = filteredPhotos().slice(state.page * pageSize, (state.page + 1) * pageSize);
    const selected = state.photos.find(photo => photo.path === state.selected);
    if (selected && !photos.includes(selected)) photos.push(selected);
    const needed = photos.filter(photo => recordFor(photo)?._metadata_only && !state.reviewPending.has(photo.path));
    if (needed.length) {
      const request = (async () => {
        try {
          const response = await fetch('/api/organize/scene/review/records', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({review_dir: project.review, images: needed.map(photo => photo.path)})});
          const body = await response.json();
          if (!response.ok || !Array.isArray(body.data?.records)) throw new Error(body.detail || body.message || '读取复核依据失败');
          if (generation !== state.generation) return;
          body.data.records.forEach(record => {
            const previous = state.records.get(record.image);
            if (previous && record.review_revision >= previous.review_revision) mergeReview(previous, record);
          });
          renderHeader(); renderGallery(false);
        } catch (error) {
          if (generation === state.generation) notice(error.message, true);
        } finally {
          if (generation === state.generation) needed.forEach(photo => state.reviewPending.delete(photo.path));
        }
      })();
      needed.forEach(photo => state.reviewPending.set(photo.path, request));
    }
    // A repaint may already have started these requests. Opening/reloading a
    // project must await the same promises before its controls become available.
    return Promise.all([...new Set(photos.map(photo => state.reviewPending.get(photo.path)).filter(Boolean))]);
  }
  async function loadKnowledge(silent = false) {
    try {
      const result = await api(`/api/organize/play/knowledge?${new URLSearchParams({output_dir: state.project.knowledge})}`);
      state.knowledge = result;
      renderKnowledge(); renderHeader(); renderInspector();
      return true;
    } catch (error) {
      state.knowledge = null; renderKnowledge(); renderHeader();
      if (!silent || error.status !== 404) throw error;
      return false;
    }
  }
  function renderKnowledge() {
    const data = state.knowledge;
    $('knowledge-content').className = data ? '' : 'compact-empty';
    $('knowledge-content').innerHTML = data ? (data.scenes.length ? data.scenes.map((label, index) => `<div class="knowledge-scene"><span>${String(index + 1).padStart(2, '0')}</span><strong>${esc(label)}</strong></div>`).join('') : '<p>资料已载入，但未识别到幕次。请补充「第 N 幕」等明确的剧情分段后重新建库。</p>') + `<div class="role-tags">${(data.roles || []).map(role => `<span class="tag">${esc(typeof role === 'string' ? role : `${role.actor || ''}${role.actor ? ' 饰 ' : ''}${role.role || ''}`)}</span>`).join('')}</div>` : '<p>建立或载入知识库后，在这里查看幕次与角色。</p>';
  }
  function populateProject() {
    const project = state.project;
    $('knowledge-dir').value = project.knowledge;
    $('info-path').value = project.info || '';
    $('play-root').value = project.playRoot || project.input;
    $('evidence-file').value = project.evidence || join(project.output, 'scene_evidence.jsonl');
    $('actor-source').value = project.input;
    $('scene-evidence-file').value = project.labels || '';
    $('identity-registry-file').value = project.identityRegistry || '';
    $('use-embedder').checked = project.useEmbedder !== false;
    $('use-llm').checked = project.useLLM === true;
  }
  async function activateProject(project, result) {
    clearTimeout(analysisTimer);
    analysisRunning = false;
    $('analysis-task').hidden = true;
    state.generation++;
    state.project = {...project, input: result.input_dir};
    state.photos = result.photos; state.records = new Map(); state.knowledge = null; state.summary = null;
    state.reviewPending = new Map();
    state.selected = result.photos.some(photo => photo.path === project.selected) ? project.selected : null;
    state.query = ''; state.page = 0; state.taskId = project.clusterTask || null; state.queueMismatch = false;
    $('photo-search').value = ''; $('actor-result').hidden = true; $('organize-result').hidden = true;
    $('labels-download').innerHTML = ''; $('cluster-grid').innerHTML = '';
    populateProject(); saveProject(); renderKnowledge(); renderHeader(); renderGallery();
    const restored = await Promise.allSettled([loadKnowledge(true), loadReviews()]);
    restored.forEach(result => { if (result.status === 'rejected') notice(`工作区已打开，部分资料暂未载入：${result.reason.message}`, true); });
    if (state.taskId) { try { await loadClusters(); } catch (error) { notice(`演员分组暂未载入：${error.message}`, true); } }
    if (result.truncated) notice(`目录有 ${result.total} 张图片，当前显示前 ${result.photos.length} 张。请分成较小的目录整理。`, true);
    else if (!result.total) notice('这个目录中没有支持的照片；如果照片存放在子目录，请直接选择那个子目录。');
    if (state.project.analysisTask) {
      const activeProject = state.project;
      analysisTimer = setTimeout(() => pollAnalysis(activeProject), 0);
    }
  }
  function openProject(newProject = false) {
    if (state.busy) return;
    const project = newProject ? null : state.project;
    $('project-name').value = project?.name || '';
    $('input-dir').value = project?.input || '';
    $('output-dir').value = project?.output || `outputs/workspace/${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}`;
    $('db-root').value = project?.db || 'data/face_database';
    $('project-knowledge').value = project?.knowledge || '';
    $('project-review').value = project?.review || '';
    $('project-error').hidden = true;
    $('project-dialog').showModal();
  }
  $('project-form').onsubmit = async event => {
    event.preventDefault(); if (state.busy) return;
    const input = $('input-dir').value.trim(), output = $('output-dir').value.trim();
    if (!input || !output) return;
    if (input.replace(/\/+$/, '') === output.replace(/\/+$/, '')) { $('project-error').textContent = '请选择与原始照片不同的结果目录。'; $('project-error').hidden = false; return; }
    const previous = state.project;
    const project = {name: $('project-name').value.trim() || basename(input.replace(/\/+$/, '')), input, output,
      db: $('db-root').value.trim() || 'data/face_database',
      knowledge: $('project-knowledge').value.trim() || join(output, 'knowledge'),
      review: $('project-review').value.trim() || join(output, 'review')};
    if (previous?.input === input && previous?.output === output) Object.assign(project, {info: previous.info, playRoot: previous.playRoot, evidence: previous.evidence, labels: previous.labels, useEmbedder: previous.useEmbedder, useLLM: previous.useLLM, selected: previous.selected, identityRegistry: previous.identityRegistry});
    await task('正在读取剧照目录…', async () => {
      try {
        const result = await api(`/api/workspace/photos?${new URLSearchParams({input_dir: input})}`);
        $('project-dialog').close(); setView('photos');
        await activateProject(project, result);
      } catch (error) { $('project-error').textContent = error.message; $('project-error').hidden = false; throw error; }
    });
  };
  let analysisTimer = null;
  let analysisRunning = false;
  async function applyAnalysis(project, result) {
    if (project.analysisKind === 'face_database') {
      await task('正在载入人脸库构建结果…', async () => {
        showFaceDatabaseResult(result); project.analysisApplied = project.analysisTask; saveProject(); setView('actors');
        notice(result.dry_run ? '检查完成。取消「仅统计」后可将锚定照片写入演员库。' : '演员库已更新，可用于整理及生成演员专辑。');
      });
      return;
    }
    if (project.analysisKind === 'face_cluster') {
      await task('正在载入演员分组…', async () => {
        state.taskId = project.analysisTask; project.clusterTask = state.taskId;
        await loadClusters(); project.analysisApplied = project.analysisTask; saveProject(); setView('actors');
        notice(`扫描完成，请核对分组并填写演员及角色。${result.failed ? `${result.failed} 张图片失败，原图保留。` : ''}`);
      });
      return;
    }
    if (project.analysisKind === 'knowledge') {
      await task('知识库构建完成，正在载入幕次…', async () => {
        await loadKnowledge(); project.analysisApplied = project.analysisTask; saveProject();
        notice(`剧目知识库已建立，识别到 ${state.knowledge.scenes.length} 个幕次。${result.failed ? `${result.failed} 个媒体条目失败，请查阅构建报告。` : ''}`);
      });
      return;
    }
    if (project.analysisKind === 'organize') {
      await task('正在载入整理报告…', async () => {
        showOrganizeResult(project, result, project.labels);
        project.analysisApplied = project.analysisTask; saveProject();
        notice(`整理完成：保留 ${result.kept} 张，划分 ${result.n_scenes} 个场景。${result.failed ? `${result.failed} 张处理失败，原图均保留。` : ''}`);
      });
      return;
    }
    await task('分析完成，正在载入场景复核队列…', async () => {
      project.evidence = result.output_file; saveProject(); $('evidence-file').value = result.output_file;
      await api('/api/organize/scene/review/seed', {review_dir: project.review, evidence_file: result.output_file});
      await loadReviews(); setView('review');
      const first = filteredPhotos()[0]; if (first) selectPhoto(first.path);
      project.analysisApplied = project.analysisTask; saveProject();
      if (!state.queueMismatch) notice(`已分析 ${result.n_images} 张剧照，${result.n_unknown} 张幕次未确定。${result.n_failed ? `${result.n_failed} 张处理失败，已保留错误记录供复核。` : ''}${result.n_visual ? `${result.n_visual} 张使用了图文匹配。` : '本次使用文件名与剧情线索；图文匹配未参与或不可用。'}${result.llm ? '已连接语言模型；实际调用及回退以证据记录为准。' : ''}已有复核记录保留原结果。`);
    });
  }
  async function pollAnalysis(project) {
    clearTimeout(analysisTimer);
    if (state.project !== project || !project.analysisTask) return;
    try {
      const status = await api(`/api/organize/tasks/${encodeURIComponent(project.analysisTask)}`);
      if (state.project !== project) return;
      analysisRunning = ['queued', 'running', 'cancel_requested'].includes(status.status);
      const name = {knowledge: '知识库构建', organize: '整理', scene_reason: '场景分析', face_cluster: '演员扫描', face_database: '人脸建库'}[project.analysisKind || 'scene_reason'];
      const labels = {queued: '排队中', running: '执行中', cancel_requested: '等待当前操作结束后取消', cancelled: '已取消', interrupted: '已中断', failed: '执行失败', completed: '完成', completed_with_errors: '完成，部分条目失败'};
      $('analysis-task').hidden = false;
      $('analysis-task-message').textContent = `${name}${labels[status.status] || status.status}：${status.preparing ? `${status.preparing} 张图片待校验；` : ''}${status.succeeded} 项完成，${status.failed} 项失败，${status.pending} 项待处理。${status.error || ''}`;
      $('analysis-cancel').textContent = `取消${name}`;
      $('analysis-cancel').hidden = !analysisRunning || status.status === 'cancel_requested';
      $('analysis-resume').hidden = !['cancelled', 'interrupted', 'failed'].includes(status.status);
      $('analysis-new').hidden = !['cancelled', 'interrupted', 'failed', 'completed_with_errors'].includes(status.status);
      $('analysis-new').textContent = project.analysisKind && project.analysisKind !== 'scene_reason' ? '结束此任务，保留已有文件' : '按当前资料重新分析';
      if (project.analysisKind === 'face_database' && status.result && project.analysisApplied === project.analysisTask && $('actor-result').hidden) {
        showFaceDatabaseResult(status.result);
      }
      if (['completed', 'completed_with_errors'].includes(status.status) && project.analysisApplied !== project.analysisTask && !state.busy) {
        await applyAnalysis(project, status.result);
      }
      if (analysisRunning || (status.result && project.analysisApplied !== project.analysisTask)) {
        analysisTimer = setTimeout(() => pollAnalysis(project), 1000);
      }
    } catch (error) {
      if (state.project !== project) return;
      $('analysis-task').hidden = false;
      $('analysis-task-message').textContent = `暂时无法查询分析任务：${error.message}。将自动重试。`;
      analysisTimer = setTimeout(() => pollAnalysis(project), 3000);
    }
  }
  $('analysis-cancel').onclick = async () => {
    const project = state.project;
    try { await api(`/api/organize/tasks/${encodeURIComponent(project.analysisTask)}/cancel`, {}); await pollAnalysis(project); }
    catch (error) { notice(error.message, true); }
  };
  $('analysis-resume').onclick = async () => {
    const project = state.project;
    try { await api(`/api/organize/tasks/${encodeURIComponent(project.analysisTask)}/resume`, {}); await pollAnalysis(project); }
    catch (error) { notice(error.message, true); }
  };
  $('analysis-new').onclick = async () => {
    if (state.busy) return;
    const project = state.project;
    try {
      await api(`/api/organize/tasks/${encodeURIComponent(project.analysisTask)}/cancel`, {});
      clearTimeout(analysisTimer); analysisRunning = false;
      project.analysisTask = ''; project.analysisApplied = ''; project.analysisKey = ''; saveProject();
      if (!project.analysisKind || project.analysisKind === 'scene_reason') await analyze();
      else { $('analysis-task').hidden = true; notice('已结束此任务，已生成文件保留。可以重新提交任务。'); }
    } catch (error) { notice(error.message, true); }
  };
  async function analyze() {
    if (!needProject() || state.busy) return;
    if (analysisRunning) { notice('场景分析正在后台运行；可以继续看图复核或取消当前分析。'); return; }
    if (state.project.analysisTask && state.project.analysisApplied !== state.project.analysisTask) { await pollAnalysis(state.project); notice('请先继续未完成图片或完成结果载入。'); return; }
    if (!state.knowledge?.scenes.length) { setView('knowledge'); notice('请先建立或载入包含幕次的剧目知识库。'); return; }
    await task('正在提交场景分析任务…', async () => {
      const project = state.project;
      if (!project.analysisKey) {
        project.analysisKey = typeof crypto.randomUUID === 'function' ? crypto.randomUUID()
          : Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, '0')).join('');
        saveProject();
      }
      const response = await fetch('/api/organize/tasks', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({kind: 'scene_reason', idempotency_key: project.analysisKey, params: {
          input_dir: project.input, knowledge_dir: project.knowledge, output_file: join(project.output, 'scene_evidence.jsonl'),
          use_embedder: $('use-embedder').checked, use_llm: $('use-llm').checked}})});
      const body = await response.json();
      if (!response.ok) throw new Error(body.message || '无法创建分析任务');
      project.analysisTask = body.data.task_id; project.analysisKey = ''; saveProject();
      project.analysisKind = 'scene_reason'; saveProject();
      analysisRunning = true;
    });
    if (state.project.analysisTask) await pollAnalysis(state.project);
  }
  async function submitBackground(project, kind, params) {
    if (analysisRunning || (project.analysisTask && project.analysisApplied !== project.analysisTask)) {
      throw new Error('请先完成、取消或恢复当前后台任务。');
    }
    const key = `${kind}:${Date.now()}:${Array.from(crypto.getRandomValues(new Uint8Array(8))).join('-')}`;
    // Keep the submission key across connection errors so retry is idempotent.
    if (!project.backgroundSubmission || project.backgroundSubmission.kind !== kind || JSON.stringify(project.backgroundSubmission.params) !== JSON.stringify(params)) {
      project.backgroundSubmission = {kind, params, idempotency_key: key}; saveProject();
    }
    const response = await fetch('/api/organize/tasks', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(project.backgroundSubmission)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || '无法创建后台任务');
    project.analysisTask = body.data.task_id; project.analysisKind = kind; project.backgroundSubmission = null;
    if (kind === 'face_cluster') state.taskId = body.data.task_id;
    analysisRunning = true; saveProject();
    analysisTimer = setTimeout(() => pollAnalysis(project), 0);
  }
  async function decide(record, action) {
    if (state.busy) return;
    const label = $('review-label').value.trim(), note = $('review-note').value.trim();
    if (action === 'modify' && (!label || label === 'unknown')) { notice('请选择或填写一个明确的幕次名称。', true); $('review-label').focus(); return; }
    const option = [...(state.knowledge?.scene_options || []), ...(record.alternatives || [])].find(item => item.label === label);
    const filtered = filteredPhotos(), index = filtered.findIndex(photo => photo.path === state.selected);
    const photo = state.photos.find(photo => photo.path === state.selected);
    const next = filtered[index + 1] || filtered[index - 1];
    await task('正在保存复核决定…', async () => {
      let result;
      try {
        result = await api('/api/organize/scene/review', {review_dir: state.project.review, image: record.image, action,
          label: action === 'modify' ? label : '', scene_id: action === 'modify' ? (option?.scene_id || '') : '', reviewer: 'workspace', note,
          expected_revision: record.review_revision});
      } catch (error) {
        if (error.status === 409) {
          const latest = await api(`/api/organize/scene/review?${new URLSearchParams({review_dir: state.project.review, image: record.image})}`);
          mergeReview(record, latest.record); renderHeader(); renderGallery();
        }
        throw error;
      }
      if (photo) clearDraft(photo);
      mergeReview(record, result.record);
      // Existing label exports are now stale and will be regenerated before organizing.
      state.project.labels = ''; $('scene-evidence-file').value = ''; $('labels-download').innerHTML = ''; saveProject();
      if (state.filter === 'pending') {
        state.selected = next?.path || filteredPhotos()[0]?.path || null;
        state.project.selected = state.selected; saveProject();
        const nextIndex = filteredPhotos().findIndex(photo => photo.path === state.selected);
        state.page = Math.max(0, Math.floor(nextIndex / pageSize));
      }
      renderHeader(); renderGallery();
      notice(action === 'reject' ? '已拒绝此判断，归档时保留场景编号。' : '复核已保存，将使用最新标签归档。');
    });
  }
  async function exportLabels() {
    if (state.queueMismatch) throw new Error('复核目录混有其他照片的记录，请先在项目设置中选择独立的复核目录。');
    const result = await api('/api/organize/scene/review/export', {review_dir: state.project.review});
    state.project.labels = result.path; saveProject(); $('scene-evidence-file').value = result.path;
    $('labels-download').innerHTML = `<a class="download-link" href="${esc(fileURL(result.path))}" download>下载已确认标签（${result.n_labels} 条） ↗</a>`;
    return result;
  }

  $('knowledge-form').onsubmit = event => {
    event.preventDefault(); if (!needProject()) return;
    const project = state.project;
    const info = $('info-path').value.trim(), playRoot = $('play-root').value.trim(), knowledge = $('knowledge-dir').value.trim();
    if (!info || !playRoot || !knowledge) return;
    task('正在读取剧情、幕次与角色资料…', async () => {
      Object.assign(project, {info, playRoot, knowledge}); saveProject();
      await submitBackground(project, 'knowledge', {info_path: info, play_root: playRoot, output_dir: knowledge});
    });
  };
  $('load-knowledge').onclick = () => {
    if (!needProject()) return;
    const knowledge = $('knowledge-dir').value.trim();
    if (!knowledge) { $('knowledge-dir').focus(); return; }
    task('正在载入剧目知识库…', async () => {
      state.project.knowledge = knowledge;
      await loadKnowledge(); saveProject(); notice('已关联剧目知识库。');
    });
  };
  $('review-import-form').onsubmit = event => {
    event.preventDefault(); if (!needProject()) return;
    const evidence = $('evidence-file').value.trim();
    task('正在载入场景分析结果…', async () => {
      await api('/api/organize/scene/review/seed', {review_dir: state.project.review, evidence_file: evidence});
      state.project.evidence = evidence; saveProject(); await loadReviews(); setView('review');
    });
  };
  $('export-labels').onclick = () => { if (needProject()) task('正在导出已确认场景标签…', async () => { const result = await exportLabels(); notice(`已导出 ${result.n_labels} 条场景标签，整理时将自动引用。`); }); };
  $('identity-registry-file').oninput = () => {
    if (state.project) { state.project.identityRegistry = $('identity-registry-file').value.trim(); saveProject(); }
  };
  $('organize-form').onsubmit = event => {
    event.preventDefault(); if (!needProject()) return;
    const project = state.project;
    task('正在准备最新的人工复核标签…', async () => {
      await loadReviews();
      const labels = await exportLabels();
      await submitBackground(project, 'organize', {input_dir: project.input, output_dir: project.output, db_root: project.db,
        keep_per_bucket: Number($('keep-per-bucket').value), gap_seconds: Number($('gap-seconds').value),
        classify_role: $('classify-role').checked, scene_evidence_file: labels.path,
        identity_registry_file: project.identityRegistry || ''});
    });
  };
  function showOrganizeResult(project, result, labelsPath) {
      $('organize-result').hidden = false;
      $('organize-result').innerHTML = `<h3>${result.failed ? '归档完成，部分图片失败' : '归档完成'}</h3><div class="result-stats"><div><strong>${esc(result.total_input)}</strong><span>输入剧照</span></div><div><strong>${esc(result.kept)}</strong><span>精选保留</span></div><div><strong>${esc(result.discarded)}</strong><span>未选用</span></div><div><strong>${esc(result.n_scenes)}</strong><span>场景分组</span></div><div><strong>${esc(result.failed || 0)}</strong><span>处理失败</span></div></div><p>归档目录：${esc(join(project.output, 'organized'))}</p><div class="result-links"><a href="${esc(fileURL(result.report_path))}" download>下载整理报告 ↗</a>${labelsPath ? `<a href="${esc(fileURL(labelsPath))}" download>下载场景标签 ↗</a>` : ''}<button class="text-button" id="browse-organized">浏览归档剧照 →</button></div>`;
      $('browse-organized').onclick = () => task('正在载入归档剧照…', async () => {
        const input = join(project.output, 'organized');
        const result = await api(`/api/workspace/photos?${new URLSearchParams({input_dir: input})}`);
        // Browse results without reusing the source queue for a differently named collection.
        await activateProject({...project, name: `${project.name} · 已归档`, input, output: join(project.output, 'next'), review: join(project.output, 'organized_review'), labels: '', analysisTask: '', analysisApplied: '', analysisKey: '', analysisKind: '', backgroundSubmission: null}, result);
        setView('photos');
      });
  }

  function showFaceDatabaseResult(result) {
    $('actor-result').hidden = false;
    $('actor-result').innerHTML = `<h3>${result.dry_run ? '照片检查完成' : '演员库已更新'}</h3><div class="result-stats"><div><strong>${esc(result.scanned ?? 0)}</strong><span>扫描照片</span></div><div><strong>${esc(result.added ?? 0)}</strong><span>${result.dry_run ? '可用锚定照片' : '新增锚定照片'}</span></div><div><strong>${esc(result.failed ?? 0)}</strong><span>处理失败</span></div></div><div class="role-tags">${(result.added_persons || []).map(person => `<span class="tag">${esc(person)}</span>`).join('')}</div>${result.failed ? '<p>部分图片处理失败，原图保留，请查看构建报告。</p>' : ''}${result.report_path ? `<a href="${esc(fileURL(result.report_path))}" download>下载人脸建库报告 ↗</a>` : ''}`;
  }
  $('actor-build-form').onsubmit = event => {
    event.preventDefault(); if (!needProject()) return;
    const source = $('actor-source').value.trim(), dryRun = $('actor-dry-run').checked;
    task('正在提交人脸库构建任务…', () => submitBackground(state.project, 'face_database', {
      organized_dirs: source, db_root: state.project.db, dry_run: dryRun,
      output_dir: join(state.project.output, 'face_database')}));
  };
  $('actor-view-form').onsubmit = event => {
    event.preventDefault(); if (!needProject()) return;
    const person = $('actor-filter').value.trim().replace(/[、，]/g, ',');
    task('正在识别演员并生成照片专辑…', async () => {
      const result = await api('/api/organize/actor_view', {input_dir: state.project.input, output_dir: join(state.project.output, 'actors'), db_root: state.project.db, person,
        identity_registry_file: state.project.identityRegistry || ''});
      $('actor-result').hidden = false;
      $('actor-result').innerHTML = `<h3>演员专辑已生成</h3><p>保存位置：${esc(join(state.project.output, 'actors/actor_views'))}</p><div class="role-tags">${Object.entries(result.persons || {}).map(([name, count]) => `<span class="tag">${esc(name)} · ${esc(count)} 张</span>`).join('') || '<p>未识别到库内演员，请补充清晰锚定照片后重试。</p>'}</div>`;
      notice('演员专辑生成完成。');
    });
  };
  async function loadClusters() {
    const result = await api(`/api/organize/face_cluster/result?${new URLSearchParams({task_id: state.taskId})}`);
    if (!result.clusters) return false;
    const clusters = result.clusters;
    $('cluster-grid').innerHTML = clusters.map((cluster, index) => `<article class="cluster-card ${cluster.status === 'assigned' ? 'assigned' : ''} ${cluster.status === 'ignored' ? 'ignored' : ''}"><img src="${esc(fileURL(cluster.preview))}" alt="演员分组 ${index + 1}" loading="lazy"><h3>演员分组 ${index + 1} · ${esc(cluster.size ?? cluster.n_faces ?? 0)} 张人脸</h3><p>${cluster.status === 'assigned' ? `已入库：${esc(cluster.assigned_to)}` : cluster.status === 'ignored' ? '已忽略' : !cluster.suggest_manual ? '可命名入库' : '请核对是否为同一演员，必要时补充清晰照片'}</p><form data-cluster="${index}"><input aria-label="分组 ${index + 1} 的演员及角色" placeholder="演员（饰角色）" required><div class="inline-actions"><button class="button primary" type="submit">命名入库</button><button class="button secondary" type="button" data-ignore="${index}">${cluster.status === 'ignored' ? '恢复分组' : '忽略'}</button></div></form></article>`).join('') || '<div class="panel"><p>没有检出可用的人脸分组。可补充近景照片，或使用已标注照片建库。</p></div>';
    $('cluster-grid').querySelectorAll('[data-cluster]').forEach(form => { form.onsubmit = event => {
      event.preventDefault(); const cluster = clusters[Number(form.dataset.cluster)], person = form.querySelector('input').value.trim();
      if (!person) return;
      task('正在保存演员及角色…', async () => { await api('/api/organize/face_cluster/assign', {task_id: state.taskId, cluster_id: cluster.id, person, db_root: state.project.db}); await loadClusters(); notice('演员已加入人脸库。'); });
    }; });
    $('cluster-grid').querySelectorAll('[data-ignore]').forEach(button => { button.onclick = () => {
      const cluster = clusters[Number(button.dataset.ignore)];
      task('正在更新分组…', async () => { await api('/api/organize/face_cluster/ignore', {task_id: state.taskId, cluster_id: cluster.id, ignored: cluster.status !== 'ignored'}); await loadClusters(); });
    }; });
    return true;
  }
  $('cluster-button').onclick = () => {
    if (!needProject()) return;
    task('正在提交演员扫描任务…', () => submitBackground(state.project, 'face_cluster', {
      input_dir: state.project.input, output_dir: join(state.project.output, 'clusters')}));
  };

  document.querySelectorAll('[data-view]').forEach(button => { button.onclick = () => setView(button.dataset.view); });
  document.querySelectorAll('[data-filter]').forEach(button => { button.onclick = () => { state.filter = button.dataset.filter; state.page = 0; renderGallery(); }; });
  document.querySelectorAll('[data-close]').forEach(button => { button.onclick = () => $(button.dataset.close).close(); });
  ['import-button', 'empty-import'].forEach(id => { $(id).onclick = () => openProject(true); });
  $('project-settings').onclick = () => openProject();
  ['analyze-button', 'knowledge-analyze'].forEach(id => { $(id).onclick = analyze; });
  ['use-embedder', 'use-llm'].forEach(id => { $(id).onchange = () => { if (state.project) { state.project.useEmbedder = $('use-embedder').checked; state.project.useLLM = $('use-llm').checked; saveProject(); } }; });
  $('photo-search').oninput = event => { state.query = event.target.value.trim().toLowerCase(); state.page = 0; renderGallery(); };
  $('clear-filters').onclick = () => { state.filter = 'all'; state.query = ''; state.page = 0; $('photo-search').value = ''; renderGallery(); };
  $('prev-page').onclick = () => { state.page--; renderGallery(); };
  $('next-page').onclick = () => { state.page++; renderGallery(); };
  $('reload-button').onclick = () => task('正在刷新影像与复核状态…', async () => {
    const result = await api(`/api/workspace/photos?${new URLSearchParams({input_dir: state.project.input})}`);
    state.photos = result.photos; await loadReviews();
    notice('影像与复核状态已更新。');
  });
  $('sample-button').onclick = () => task('正在读取项目中的真实样例剧照…', async () => {
    const result = await api('/api/workspace/photos?input_dir=data%2Fsample_images');
    const output = `outputs/workspace/sample-${Date.now()}`;
    await activateProject({name: '项目样例 · 影像浏览', input: result.input_dir, output, db: 'data/face_database', knowledge: join(output, 'knowledge'), review: join(output, 'review')}, result);
    notice('已载入项目自带样例。场景分析需要先关联与这些照片对应的剧目资料。');
  });
  document.addEventListener('error', event => {
    if (event.target.tagName !== 'IMG') return;
    const container = event.target.closest('.photo-image, .detail-preview');
    if (container) container.classList.add('image-error');
    else event.target.alt = '图片暂不可用';
  }, true);
  document.addEventListener('keydown', event => {
    if (document.querySelector('dialog[open]') || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(document.activeElement?.tagName) || !['photos', 'review'].includes(state.view)) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); moveSelection(event.key === 'ArrowLeft' ? -1 : 1); }
  });
  async function start() {
    renderHeader(); setView(location.hash.slice(1) || 'photos');
    api('/api/health').then(() => { $('service-status').className = 'service-status online'; $('service-status').innerHTML = '<span></span>服务已连接'; }).catch(() => { $('service-status').className = 'service-status offline'; $('service-status').innerHTML = '<span></span>服务暂不可用'; });
    let project;
    try { project = JSON.parse(localStorage.getItem(projectKey)); } catch (_) { /* Private browsing or invalid old settings. */ }
    if (!project || !['input', 'output', 'knowledge', 'review', 'db'].every(key => typeof project[key] === 'string' && project[key])) return;
    await task('正在恢复上次的剧目工作区…', async () => {
      const result = await api(`/api/workspace/photos?${new URLSearchParams({input_dir: project.input})}`);
      await activateProject(project, result);
    });
  }
  start();
})();
