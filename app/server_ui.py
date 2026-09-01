# -*- coding: utf-8 -*-
"""
server_ui.py - 可视化界面服务（FastAPI）

职责：
- 复用 main.py 中定义的全部 API 路由（人脸识别 / 人像检测分割 / 修复 / 检索聚类）
- 额外提供 /ui 可视化操作界面（浏览器直接操作，无需 curl）
- 独立启动入口，默认端口 8199，与纯 API 服务 (server.py, 8198) 互不冲突

启动方式：python app/server_ui.py
访问界面：http://localhost:8199/ui
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.responses import HTMLResponse, RedirectResponse

from main import app  # 复用 main.py 中定义的全部 API 路由与模型

ORGANIZE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能整理 - 戏剧照片整理系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
               background: #f0f2f5; color: #333; min-height: 100vh; }
        .topbar { background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
                  color: #fff; padding: 18px 40px; display: flex;
                  align-items: center; justify-content: space-between;
                  box-shadow: 0 2px 12px rgba(0,0,0,0.15); position: sticky; top: 0; z-index: 100; }
        .topbar h1 { font-size: 20px; font-weight: 600; }
        .topbar .sub { font-size: 12px; opacity: 0.85; margin-top: 2px; }
        .nav { display: flex; gap: 10px; }
        .nav a { color: #fff; text-decoration: none; font-size: 13px; padding: 8px 16px;
                 border-radius: 8px; background: rgba(255,255,255,0.12);
                 border: 1px solid rgba(255,255,255,0.25); transition: background 0.2s; }
        .nav a:hover { background: rgba(255,255,255,0.25); }
        .nav a.active { background: rgba(255,255,255,0.32); font-weight: 600; }
        .container { max-width: 1280px; margin: 24px auto; padding: 0 20px;
                     display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }
        .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); padding: 22px; }
        .card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.1); }
        .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px;
                   padding-bottom: 10px; border-bottom: 2px solid #eef2f7; color: #14532d; }
        .card .desc { font-size: 12px; color: #64748b; margin-bottom: 14px; line-height: 1.6; }
        .form-group { margin-bottom: 12px; }
        .form-group label { display: block; font-size: 12px; color: #666; margin-bottom: 5px; font-weight: 500; }
        .form-group .hint { font-size: 11px; color: #94a3b8; margin-top: 3px; }
        input[type="text"], input[type="number"], select { width: 100%; padding: 8px 12px;
                     border: 1px solid #d1d5db; border-radius: 6px; font-size: 13px; }
        input:focus, select:focus { outline: none; border-color: #16a34a;
                     box-shadow: 0 0 0 3px rgba(22,163,74,0.1); }
        .row { display: flex; gap: 10px; }
        .row .form-group { flex: 1; }
        .btn { display: inline-block; width: 100%; padding: 10px; border: none; border-radius: 6px;
               background: #16a34a; color: #fff; font-size: 14px; font-weight: 500;
               cursor: pointer; transition: background 0.2s, transform 0.1s; }
        .btn:hover { background: #15803d; }
        .btn:active { transform: scale(0.98); }
        .btn.btn-blue { background: #2563eb; }
        .btn.btn-blue:hover { background: #1d4ed8; }
        .result { margin-top: 14px; padding: 12px; border-radius: 8px; font-size: 13px;
                  display: none; background: #f8fafc; border: 1px solid #e2e8f0; }
        .result.success { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
        .result.error { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
        .result.loading { background: #f0fdf4; border-color: #bbf7d0; color: #14532d;
                  text-align: center; padding: 16px; }
        .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid #bbf7d0;
                   border-top-color: #16a34a; border-radius: 50%;
                   animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result pre { background: #fff; padding: 8px; border-radius: 4px; font-size: 11px;
                      overflow-x: auto; margin-top: 8px; max-height: 260px; overflow-y: auto; }
        .stat-row { display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
        .stat { flex: 1; min-width: 90px; background: #f0fdf4; border: 1px solid #bbf7d0;
                border-radius: 8px; padding: 10px; text-align: center; }
        .stat .num { font-size: 20px; font-weight: 700; color: #14532d; }
        .stat .lbl { font-size: 11px; color: #64748b; margin-top: 2px; }
        .file-list { margin-top: 10px; max-height: 300px; overflow-y: auto;
                     border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; }
        .file-list .item { padding: 7px 10px; font-size: 12px;
                     border-bottom: 1px solid #f1f5f9; }
        .file-list .item:hover { background: #f8fafc; }
        .file-list .name { word-break: break-all; color: #334155; }
        .actor-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                      gap: 10px; margin-top: 10px; }
        .actor-item { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                      padding: 10px; text-align: center; }
        .actor-item .name { font-size: 12px; font-weight: 600; color: #14532d; margin-top: 2px; }
        .actor-item .cnt { font-size: 11px; color: #64748b; }
        .actor-item .bar { height: 5px; background: #e2e8f0; border-radius: 3px;
                      margin-top: 6px; overflow: hidden; }
        .actor-item .bar i { display: block; height: 100%; border-radius: 3px;
                      background: linear-gradient(90deg, #16a34a, #4ade80); }
        .cluster-card { position: relative; }
        .cluster-card.assigned { border-color: #86efac; background: #f0fdf4; }
        .cluster-card.ignored { opacity: 0.45; }
        .cluster-card .badge {
            position: absolute; top: 8px; right: 8px; font-size: 10px;
            padding: 2px 8px; border-radius: 10px; font-weight: 600;
        }
        .badge.ok { background: #dcfce7; color: #166534; }
        .badge.warn { background: #fef9c3; color: #854d0e; }
        .cluster-card img { width: 100%; height: auto; border-radius: 6px;
            border: 1px solid #e2e8f0; background: #f1f5f9; }
        .cluster-card input[type="text"] { margin-top: 6px; }
        .cluster-card .btn-row { display: flex; gap: 6px; margin-top: 6px; }
        .cluster-card .btn-row button { flex: 1; padding: 6px; border: none;
            border-radius: 6px; font-size: 12px; cursor: pointer; font-weight: 500; }
        .btn-confirm { background: #16a34a; color: #fff; }
        .btn-confirm:hover { background: #15803d; }
        .btn-ignore { background: #f1f5f9; color: #334155; }
        .btn-ignore:hover { background: #e2e8f0; }
        .progressbar { height: 8px; background: #e2e8f0; border-radius: 4px;
            overflow: hidden; margin-top: 8px; }
        .progressbar i { display: block; height: 100%; width: 0;
            background: linear-gradient(90deg, #16a34a, #4ade80); transition: width 0.5s; }
        .steps { display: flex; gap: 4px; margin-top: 10px; }
        .step { flex: 1; text-align: center; font-size: 10px; color: #94a3b8;
                padding: 6px 2px; border-radius: 4px; background: #f8fafc; }
        .step.done { background: #dcfce7; color: #166534; font-weight: 600; }
        .step.active { background: #fef9c3; color: #854d0e; font-weight: 600; }
        .nav { display: flex; gap: 10px; }
        .nav a {
            color: #fff; text-decoration: none; font-size: 13px;
            padding: 8px 16px; border-radius: 8px;
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
            transition: background 0.2s; white-space: nowrap;
        }
        .nav a:hover { background: rgba(255,255,255,0.25); }
        .nav a.active { background: rgba(255,255,255,0.32); font-weight: 600; }
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
            .topbar { padding: 14px 20px; flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="topbar">
        <div>
            <h1>📦 智能整理（Smart Organizer）</h1>
            <div class="sub">连拍去重 → 人脸识别 → 行当分类 → 场景划分 → 规范命名</div>
        </div>
        <div class="nav">
            <a href="/ui">🎭 照片处理</a>
            <a href="/organize" class="active">📦 智能整理</a>
            <a href="/docs" target="_blank">API 文档</a>
        </div>
    </div>

    <div class="container">
        <!-- ① 半自动人脸库构建 -->
        <div class="card">
            <h2>① 半自动人脸库构建</h2>
            <div class="desc">
                从「整理完成」或「原始」目录构建演员锚定库。<br>
                · 单人照片直接归属（文件名仅一位「X饰Y」演员）<br>
                · 多人同框照片用已建库演员匹配消去后传播归属<br>
                · 谢幕/合影/归属歧义自动跳过，同人一致性校验拦截异人
            </div>
            <form id="build-db-form">
                <div class="form-group">
                    <label>源目录列表（逗号分隔多个目录）</label>
                    <input type="text" id="db-dirs" name="organized_dirs"
                           placeholder="/path/【整理完成】剧目彩排,/path/【原始】...-摄影XX" required>
                    <div class="hint">目录名需含日期+剧目，文件名含「演员饰角色」标注</div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>人脸库根目录（留空默认 data/face_database）</label>
                        <input type="text" id="db-root" name="db_root">
                    </div>
                    <div class="form-group">
                        <label>每人最多锚定张数</label>
                        <input type="number" id="db-max-anchors" name="max_anchors" value="8" min="1" max="50">
                    </div>
                </div>
                <div class="form-group">
                    <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
                        <input type="checkbox" id="db-dry-run" name="dry_run" value="true" style="width:auto">
                        试运行（仅统计，不写入）
                    </label>
                </div>
                <button type="submit" class="btn">开始建库</button>
            </form>
            <div id="build-db-result" class="result"></div>
        </div>

        <!-- ② 按演员整理视图 -->
        <div class="card">
            <h2>② 按演员整理视图</h2>
            <div class="desc">
                对目标目录逐图识别库内演员，按演员生成照片专辑
                （actor_views/演员名/ + actor_view_report.json）。
            </div>
            <form id="actor-view-form">
                <div class="form-group">
                    <label>目标照片目录</label>
                    <input type="text" id="av-input" name="input_dir" placeholder="/path/to/photos" required>
                </div>
                <div class="form-group">
                    <label>输出目录</label>
                    <input type="text" id="av-output" name="output_dir" placeholder="/path/to/output" required>
                </div>
                <div class="form-group">
                    <label>限定演员（可选，留空为库内全部演员）</label>
                    <input type="text" id="av-person" name="person"
                           placeholder="如：陈少云 或 陈少云（饰颍考叔），多人用逗号分隔：陈少云,史依弘">
                    <div class="hint">支持人名模糊匹配；也可直接粘贴库内子目录路径（自动取目录名）</div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>人脸库根目录（留空默认）</label>
                        <input type="text" id="av-db-root" name="db_root">
                    </div>
                    <div class="form-group">
                        <label>命中阈值（余弦）</label>
                        <input type="number" id="av-threshold" name="threshold" value="0.55" step="0.05" min="0.3" max="0.9">
                    </div>
                </div>
                <button type="submit" class="btn btn-blue">生成演员视图</button>
            </form>
            <div id="actor-view-result" class="result"></div>
        </div>

        <!-- ③ 智能整理流水线 -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>③ 智能整理流水线（完整流程）</h2>
            <div class="desc">
                输入原始摄影目录，自动完成：连拍去重选优（景别感知，每景别保留 top-K）→
                人脸识别生成人物列表 → 戏曲行当分类（可选）→ EXIF 时间场景划分 →
                规范命名输出（organized/）。完整报告见输出目录 organize_report.json。
            </div>
            <div class="steps" id="pipeline-steps">
                <div class="step" data-step="1">1 解析</div>
                <div class="step" data-step="2">2 连拍去重</div>
                <div class="step" data-step="3">3 人脸识别</div>
                <div class="step" data-step="4">4 行当分类</div>
                <div class="step" data-step="5">5 场景划分</div>
                <div class="step" data-step="6">6 规范命名</div>
            </div>
            <form id="organize-form" style="margin-top: 12px;">
                <div class="row">
                    <div class="form-group">
                        <label>原始照片目录</label>
                        <input type="text" id="org-input" name="input_dir"
                               placeholder="/path/【原始N】...-摄影XX" required>
                    </div>
                    <div class="form-group">
                        <label>输出目录</label>
                        <input type="text" id="org-output" name="output_dir" placeholder="/path/to/output" required>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>每景别桶保留张数</label>
                        <input type="number" id="org-keep" name="keep_per_bucket" value="2" min="1" max="10">
                    </div>
                    <div class="form-group">
                        <label>场景时间间隔阈值（秒）</label>
                        <input type="number" id="org-gap" name="gap_seconds" value="300" min="30" max="3600" step="30">
                    </div>
                    <div class="form-group">
                        <label style="display:flex;align-items:center;gap:6px;cursor:pointer;margin-top:22px">
                            <input type="checkbox" id="org-role" name="classify_role" value="true" checked style="width:auto">
                            启用行当分类（戏曲类）
                        </label>
                    </div>
                </div>
                <div class="form-group">
                    <label>场景人工标注映射文件（可选，JSON：{"scene-01": "第1幕克段", ...}）</label>
                    <input type="text" id="org-labels" name="scene_labels_file" placeholder="/path/to/scene_labels.json">
                </div>
                <div class="form-group">
                    <label>本地人脸库目录（留空默认 data/face_database）</label>
                    <input type="text" id="org-db-root" name="db_root"
                           placeholder="data/face_database">
                    <div class="hint">识别只用该目录下的锚定子目录（不查 ES）；新演员请先经卡片①/④建库</div>
                </div>
                <button type="submit" class="btn" id="org-submit">执行智能整理</button>
            </form>
            <div id="organize-result" class="result"></div>
        </div>

        <!-- ④ 人脸聚类建库（无标注目录自举） -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>④ 人脸聚类建库（无标注目录自举）</h2>
            <div class="desc">
                文件名没有「演员饰角色」标注时用本功能：扫描目录检出全部人脸并按身份聚类，
                每簇生成拼图预览，人工给簇命名后即写入人脸库。<br>
                · 内聚度高、人脸数足的簇为<b>高置信</b>，命名即可入库；<br>
                · 内聚度低或人脸过少的簇标记<b>建议人工补图</b>——请为该演员手动收集 3~5 张
                人脸图放入人脸库子目录（路径2兜底），或忽略。<br>
                · 扫描为后台任务，完成后可逐簇命名入库。
            </div>
            <form id="cluster-scan-form">
                <div class="row">
                    <div class="form-group">
                        <label>照片目录（可无任何人物标注）</label>
                        <input type="text" id="cl-input" name="input_dir" placeholder="/path/to/photos" required>
                    </div>
                    <div class="form-group">
                        <label>结果目录（cluster_state.json + 预览拼图）</label>
                        <input type="text" id="cl-output" name="output_dir" placeholder="/path/to/output" required>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>聚类距离阈值（余弦，越大簇越少）</label>
                        <input type="number" id="cl-dist" name="dist_threshold" value="0.5" step="0.05" min="0.3" max="0.8">
                    </div>
                    <div class="form-group">
                        <label>最小人脸面积 px²（过滤远景龙套）</label>
                        <input type="number" id="cl-minface" name="min_face_area" value="1600" step="400" min="400">
                    </div>
                </div>
                <button type="submit" class="btn">开始扫描聚类</button>
            </form>
            <div id="cluster-scan-result" class="result"></div>
            <div id="cluster-list" style="display:none; margin-top:14px;">
                <div class="stat-row" id="cluster-stats"></div>
                <div id="cluster-grid" class="actor-grid"></div>
            </div>
        </div>
    </div>

    <script>
        function showLoading(el, msg) {
            el.className = 'result loading'; el.style.display = 'block';
            el.innerHTML = '<div class="spinner"></div>' + msg;
        }
        function showError(el, data) {
            el.className = 'result error'; el.style.display = 'block';
            el.innerHTML = '<strong>❌ 错误：</strong>' + ((data && data.message) || data);
        }
        function esc(s) {
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
        function submitForm(formId, url, resultId, loadingMsg, onDone) {
            const form = document.getElementById(formId);
            const el = document.getElementById(resultId);
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                showLoading(el, loadingMsg);
                const fd = new FormData(this);
                if (!fd.get('db_root')) fd.delete('db_root');   // 空则用服务端默认
                if (!fd.get('person')) fd.delete('person');     // 空则为全部演员
                fetch(url, { method: 'POST', body: fd })
                    .then(r => r.json())
                    .then(d => {
                        if (d.code !== 200) {
                            // 人名未命中：附可用人物列表
                            if (d.available || (d.data && d.data.available)) {
                                const avail = d.available || d.data.available;
                                el.className = 'result error'; el.style.display = 'block';
                                el.innerHTML = '<strong>❌ 人名未命中人脸库</strong><div class="file-list">'
                                    + avail.map(p => '<div class="item"><span class="name">' + esc(p) + '</span></div>').join('')
                                    + '</div><p style="font-size:11px;color:#64748b;margin-top:6px">可点击上方人名关键词重试，或复制完整目录名</p>';
                                return;
                            }
                            showError(el, d); return;
                        }
                        el.className = 'result success'; el.style.display = 'block';
                        el.innerHTML = onDone(d.data);
                    })
                    .catch(err => showError(el, { message: err.message }));
            });
        }

        // ① 建库
        submitForm('build-db-form', '/api/organize/build_face_database', 'build-db-result',
            '正在构建人脸库（单人归属 → 多人锚定传播）...', function(d) {
                let html = '<strong>✅ 建库完成</strong><div class="stat-row">';
                html += stat(d.scanned, '扫描') + stat(d.added, '新增锚定')
                      + stat((d.added_persons || []).length, '新增人物')
                      + stat(d.skipped_inconsistent || 0, '一致性拦截') + '</div>';
                if ((d.added_persons || []).length) {
                    html += '<div class="file-list">';
                    d.added_persons.forEach(p => {
                        html += '<div class="item"><span class="name">' + esc(p) + '</span></div>';
                    });
                    html += '</div>';
                }
                return html + '<pre>' + JSON.stringify(d, null, 2) + '</pre>';
            });

        function stat(num, lbl) {
            return '<div class="stat"><div class="num">' + num + '</div><div class="lbl">' + lbl + '</div></div>';
        }

        // ② 演员视图
        submitForm('actor-view-form', '/api/organize/actor_view', 'actor-view-result',
            '正在逐图识别人脸并生成演员专辑...', function(d) {
                let html = '<strong>✅ 视图生成完成</strong>';
                if (d.person_filter) html += '<p style="font-size:12px">限定演员：' + esc(d.person_filter) + '</p>';
                html += '<div class="stat-row">';
                const persons = Object.entries(d.persons || {}).sort((a, b) => b[1] - a[1]);
                html += stat(d.total_images, '总图片') + stat(persons.length, '命中演员') + '</div>';
                if (persons.length) {
                    const maxCnt = Math.max.apply(null, persons.map(x => x[1]).concat([1]));
                    html += '<div class="actor-grid">';
                    persons.forEach(x => {
                        html += '<div class="actor-item"><div class="name">' + esc(x[0])
                              + '</div><div class="cnt">' + x[1] + ' 张</div>'
                              + '<div class="bar"><i style="width:' + Math.round(x[1] / maxCnt * 100) + '%"></i></div></div>';
                    });
                    html += '</div>';
                }
                return html;
            });

        // ③ 流水线（步骤动画）
        function setStep(n, done) {
            document.querySelectorAll('#pipeline-steps .step').forEach(s => {
                s.classList.remove('active', 'done');
                const i = parseInt(s.dataset.step);
                if (i < n || (i === n && done)) s.classList.add('done');
                else if (i === n) s.classList.add('active');
            });
        }
        submitForm('organize-form', '/api/organize/run', 'organize-result',
            '正在执行智能整理流水线（耗时与图片数量正相关）...', function(d) {
                setStep(6, true);
                let html = '<strong>✅ 整理完成</strong><div class="stat-row">';
                html += stat(d.total_input, '输入') + stat(d.kept, '保留')
                      + stat(d.discarded, '剪除') + stat(d.n_scenes, '场景') + '</div>';
                html += '<p style="margin-top:8px;font-size:12px">报告：<code>'
                      + esc(d.report_path) + '</code></p>';
                return html;
            });
        document.getElementById('organize-form').addEventListener('submit', function() {
            let step = 1; setStep(1);
            const timer = setInterval(() => {
                step += 1;
                if (step > 5) { clearInterval(timer); return; }
                setStep(step);
            }, 4000);
        });

        // ---------- ④ 人脸聚类建库 ----------
        let clusterTaskId = null;
        let clusterTimer = null;

        document.getElementById('cluster-scan-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const el = document.getElementById('cluster-scan-result');
            showLoading(el, '正在扫描聚类（后台任务，自动轮询进度）...');
            const fd = new FormData(this);
            fetch('/api/organize/face_cluster/scan', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(d => {
                    if (d.code !== 200) { showError(el, d); return; }
                    clusterTaskId = d.data.task_id;
                    el.innerHTML = '<div class="progressbar"><i id="cl-bar"></i></div>'
                                 + '<p id="cl-msg" style="font-size:12px;margin-top:6px">排队中...</p>';
                    el.className = 'result loading'; el.style.display = 'block';
                    clusterTimer = setInterval(pollCluster, 3000);
                })
                .catch(err => showError(el, { message: err.message }));
        });

        function pollCluster() {
            fetch('/api/organize/face_cluster/result?task_id=' + clusterTaskId)
                .then(r => r.json())
                .then(d => {
                    const data = d.data || {};
                    if (d.code !== 200 && data.error) {
                        clearInterval(clusterTimer);
                        showError(document.getElementById('cluster-scan-result'), d);
                        return;
                    }
                    if (data.progress !== undefined && data.progress < 100) {
                        const bar = document.getElementById('cl-bar');
                        const msg = document.getElementById('cl-msg');
                        if (bar) bar.style.width = data.progress + '%';
                        if (msg) msg.textContent = data.message || ('进度 ' + data.progress + '%');
                        return;
                    }
                    clearInterval(clusterTimer);
                    renderClusters(data);
                })
                .catch(err => {
                    clearInterval(clusterTimer);
                    showError(document.getElementById('cluster-scan-result'), { message: err.message });
                });
        }

        function renderClusters(data) {
            const el = document.getElementById('cluster-scan-result');
            el.className = 'result success'; el.style.display = 'block';
            const nOk = data.clusters.filter(c => !c.suggest_manual && c.status === 'pending').length;
            el.innerHTML = '<strong>✅ 聚类完成</strong><div class="stat-row">'
                + stat(data.n_images, '图片') + stat(data.n_faces, '人脸')
                + stat(data.clusters.length, '簇')
                + stat(data.n_suggest_manual, '建议人工补图')
                + stat(nOk, '待命名') + '</div>'
                + '<p style="font-size:12px;margin-top:6px">状态文件: <code>' + esc(data.state_path) + '</code></p>';

            const list = document.getElementById('cluster-list');
            list.style.display = 'block';
            const grid = document.getElementById('cluster-grid');
            grid.innerHTML = '';
            // 大簇在前，忽略簇垫底
            const order = { assigned: 0, pending: 1, ignored: 2 };
            const sorted = data.clusters.slice().sort((a, b) =>
                (order[a.status] - order[b.status]) || (b.size - a.size));
            sorted.forEach(c => { grid.appendChild(clusterCard(c)); });
        }

        function clusterCard(c) {
            const div = document.createElement('div');
            div.className = 'actor-item cluster-card' + (c.status === 'assigned' ? ' assigned' : c.status === 'ignored' ? ' ignored' : '');
            let badge = '';
            if (c.status === 'assigned') badge = '<span class="badge ok">已入库 ' + esc(c.assigned_to || '') + '</span>';
            else if (c.suggest_manual) badge = '<span class="badge warn">建议人工补图</span>';
            else if (c.status === 'pending') badge = '<span class="badge ok">待命名</span>';
            else badge = '<span class="badge warn">已忽略</span>';

            div.innerHTML = badge
                + '<img src="/api/file?path=' + encodeURIComponent(c.preview) + '" alt="' + c.id + '">'
                + '<div class="name">' + c.id + '</div>'
                + '<div class="cnt">' + c.size + ' 脸 · 内聚度 ' + c.cohesion + '</div>';

            if (c.status === 'assigned') return div;

            const input = document.createElement('input');
            input.type = 'text';
            input.placeholder = '演员（饰角色）';
            input.value = c.status === 'assigned' ? (c.assigned_to || '') : '';
            input.className = 'cl-name';
            div.appendChild(input);

            const row = document.createElement('div');
            row.className = 'btn-row';
            const btnOk = document.createElement('button');
            btnOk.className = 'btn-confirm';
            btnOk.textContent = '命名入库';
            btnOk.onclick = () => assignCluster(clusterTaskId, c.id, input.value, div);
            const btnIg = document.createElement('button');
            btnIg.className = 'btn-ignore';
            btnIg.textContent = c.status === 'ignored' ? '恢复' : '忽略';
            btnIg.onclick = () => ignoreCluster(clusterTaskId, c.id, c.status !== 'ignored', div);
            row.appendChild(btnOk); row.appendChild(btnIg);
            div.appendChild(row);
            return div;
        }

        function assignCluster(taskId, clusterId, person, card) {
            if (!person || !person.trim()) { alert('请输入人名（格式建议：演员（饰角色））'); return; }
            const fd = new FormData();
            fd.append('task_id', taskId); fd.append('cluster_id', clusterId);
            fd.append('person', person.trim());
            fetch('/api/organize/face_cluster/assign', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(d => {
                    if (d.code !== 200) { alert('入库失败: ' + (d.message || d)); return; }
                    card.classList.add('assigned');
                    card.querySelector('.btn-row').remove();
                    card.querySelector('.cl-name').remove();
                    card.querySelector('.badge')?.remove();
                    const b = document.createElement('span');
                    b.className = 'badge ok';
                    b.textContent = '已入库 ' + person.trim();
                    card.appendChild(b);
                })
                .catch(err => alert('入库失败: ' + err.message));
        }

        function ignoreCluster(taskId, clusterId, ignored, card) {
            const fd = new FormData();
            fd.append('task_id', taskId); fd.append('cluster_id', clusterId);
            fd.append('ignored', ignored ? 'true' : 'false');
            fetch('/api/organize/face_cluster/ignore', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(d => {
                    if (d.code !== 200) { alert('操作失败: ' + (d.message || d)); return; }
                    card.classList.toggle('ignored', ignored);
                    const badge = card.querySelector('.badge');
                    if (badge) badge.textContent = ignored ? '已忽略' : '待命名';
                })
                .catch(err => alert('操作失败: ' + err.message));
        }
    </script>
</body>
</html>
"""


# ---------- 原有可视化界面 HTML（照片处理） ----------

UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>戏剧照片处理工具 - 可视化界面</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
            background: #f0f2f5;
            color: #333;
            min-height: 100vh;
        }
        /* 顶部导航 */
        .topbar {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            color: #fff;
            padding: 18px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .topbar h1 { font-size: 20px; font-weight: 600; }
        .topbar .sub { font-size: 12px; opacity: 0.85; margin-top: 2px; }
        .topbar .status-pill {
            background: rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 13px;
            cursor: pointer;
            border: 1px solid rgba(255,255,255,0.3);
            transition: background 0.2s;
        }
        .topbar .status-pill:hover { background: rgba(255,255,255,0.25); }
        .topbar .status-pill .dot {
            display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; background: #4ade80; margin-right: 6px;
        }
        /* 主容器 */
        .container {
            max-width: 1200px;
            margin: 24px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 20px;
        }
        .card {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            padding: 22px;
            transition: box-shadow 0.2s;
        }
        .card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.1); }
        .card h2 {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eef2f7;
            color: #1e3a8a;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .form-group { margin-bottom: 12px; }
        .form-group label {
            display: block;
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
            font-weight: 500;
        }
        input[type="text"], input[type="number"], select, textarea {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 13px;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
        }
        input[type="file"] {
            width: 100%;
            padding: 8px;
            border: 1px dashed #cbd5e1;
            border-radius: 6px;
            font-size: 12px;
            background: #f8fafc;
            cursor: pointer;
        }
        .btn {
            display: inline-block;
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 6px;
            background: #2563eb;
            color: #fff;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }
        .btn:hover { background: #1d4ed8; }
        .btn:active { transform: scale(0.98); }
        .btn:disabled { background: #93c5fd; cursor: not-allowed; }
        .btn.btn-green { background: #16a34a; }
        .btn.btn-green:hover { background: #15803d; }
        .btn.btn-ghost { background: #f1f5f9; color: #334155; }
        .btn.btn-ghost:hover { background: #e2e8f0; }
        /* 结果区 */
        .result {
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            display: none;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .result.success { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
        .result.error { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
        .result.loading {
            background: #eff6ff; border-color: #bfdbfe; color: #1e40af;
            text-align: center; padding: 16px;
        }
        .spinner {
            display: inline-block; width: 18px; height: 18px;
            border: 3px solid #bfdbfe; border-top-color: #2563eb;
            border-radius: 50%; animation: spin 0.8s linear infinite;
            vertical-align: middle; margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result pre {
            background: #fff; padding: 8px; border-radius: 4px;
            font-size: 11px; overflow-x: auto; margin-top: 8px;
            max-height: 200px; overflow-y: auto;
        }
        /* 图片展示 */
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .image-item { text-align: center; }
        .image-item img {
            width: 100%; height: 140px; object-fit: cover;
            border-radius: 6px; border: 1px solid #e2e8f0;
            background: #f1f5f9;
        }
        .image-item p { font-size: 11px; color: #64748b; margin-top: 4px; }
        /* 人脸结果 */
        .face-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 10px; margin-top: 10px;
        }
        .face-item {
            background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 8px; text-align: center;
        }
        .face-item img {
            width: 100%; height: 130px; object-fit: cover;
            border-radius: 6px; background: #f1f5f9;
        }
        .face-item .name { font-size: 12px; font-weight: 600; color: #1e3a8a; margin-top: 6px; }
        .face-item .conf { font-size: 11px; color: #64748b; }
        /* 链接列表 */
        .links { margin-top: 8px; font-size: 12px; }
        .links a { color: #2563eb; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
            .topbar { padding: 14px 20px; }
        }
    </style>
</head>
<body>
    <div class="topbar">
        <div>
            <h1>🎭 戏剧照片处理工具</h1>
            <div class="sub">人脸识别 / 人像检测分割 / 背景修复 / 图像检索</div>
        </div>
        <div style="display:flex;align-items:center;gap:14px;">
            <div class="status-pill" onclick="checkHealth()" title="点击检测服务状态">
                <span class="dot"></span><span id="status-text">检测服务状态</span>
            </div>
            <nav class="nav">
                <a href="/ui" class="active">🎭 照片处理</a>
                <a href="/organize">📦 智能整理</a>
            </nav>
        </div>
    </div>

    <div class="container">
        <!-- 人脸识别 -->
        <div class="card">
            <h2>👤 人脸识别</h2>
            <form id="recognize-form">
                <div class="form-group">
                    <label>上传待识别图片</label>
                    <input type="file" id="recognize-image" name="image" accept="image/*" required>
                </div>
                <button type="submit" class="btn">识别图片中的人脸</button>
            </form>
            <div id="recognize-result" class="result"></div>
        </div>

        <!-- 构建人脸库 -->
        <div class="card">
            <h2>📂 构建人脸库</h2>
            <form id="build-db-form">
                <div class="form-group">
                    <label>人脸库文件夹路径（每个子目录对应一个人物）</label>
                    <input type="text" id="face-db-folder" name="face_db_folder"
                           placeholder="/path/to/face/database/folder" required>
                </div>
                <button type="submit" class="btn btn-green">构建人脸库</button>
            </form>
            <div id="build-db-result" class="result"></div>
        </div>

        <!-- 组合处理 -->
        <div class="card">
            <h2>🎬 组合处理（检测+分割+镜头+修复）</h2>
            <form id="process-form">
                <div class="form-group">
                    <label>上传待处理图片</label>
                    <input type="file" id="process-image" name="image" accept="image/*" required>
                </div>
                <button type="submit" class="btn">执行组合处理</button>
            </form>
            <div id="process-result" class="result"></div>
        </div>

        <!-- 构建图像索引 -->
        <div class="card">
            <h2>📚 构建图像索引</h2>
            <form id="index-form">
                <div class="form-group">
                    <label>图片文件夹路径</label>
                    <input type="text" id="image-folder" name="image_folder"
                           placeholder="/path/to/image/folder" required>
                </div>
                <div class="form-group">
                    <label>索引保存路径</label>
                    <input type="text" id="index-save-path" name="index_save_path"
                           placeholder="/path/to/save/index" required>
                </div>
                <div class="form-group">
                    <label>嵌入模型</label>
                    <select id="index-model-name" name="model_name">
                        <option value="resnet50">ResNet50</option>
                        <option value="resnet101">ResNet101</option>
                        <option value="dinov2_small">DINOv2 Small</option>
                        <option value="dinov2_base">DINOv2 Base</option>
                        <option value="dinov2_large">DINOv2 Large</option>
                        <option value="siglip2_base">SigLIP 2 Base</option>
                        <option value="siglip2_large">SigLIP 2 Large</option>
                        <option value="siglip2_so400m">SigLIP 2 So400m</option>
                    </select>
                </div>
                <button type="submit" class="btn btn-green">构建索引</button>
            </form>
            <div id="index-result" class="result"></div>
        </div>

        <!-- 相似图片检索 -->
        <div class="card">
            <h2>🔍 相似图片检索</h2>
            <form id="search-form">
                <div class="form-group">
                    <label>查询图片</label>
                    <input type="file" id="search-query-image" name="query_image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label>索引路径（留空使用最近构建的索引）</label>
                    <input type="text" id="search-index-path" name="index_path"
                           placeholder="/path/to/index">
                </div>
                <div class="form-group">
                    <label>返回结果数量</label>
                    <input type="number" id="search-top-k" name="top_k" value="5" min="1" max="20">
                </div>
                <button type="submit" class="btn">检索相似图片</button>
            </form>
            <div id="search-result" class="result"></div>
        </div>

        <!-- 相似图片聚类 -->
        <div class="card">
            <h2>🗂️ 相似图片聚类分组</h2>
            <form id="group-form">
                <div class="form-group">
                    <label>图片文件夹路径</label>
                    <input type="text" id="group-image-folder" name="image_folder"
                           placeholder="/path/to/image/folder" required>
                </div>
                <div class="form-group">
                    <label>嵌入模型</label>
                    <select id="group-model-name" name="model_name">
                        <option value="resnet50">ResNet50</option>
                        <option value="resnet101">ResNet101</option>
                        <option value="dinov2_small">DINOv2 Small</option>
                        <option value="dinov2_base">DINOv2 Base</option>
                        <option value="dinov2_large">DINOv2 Large</option>
                        <option value="siglip2_base">SigLIP 2 Base</option>
                        <option value="siglip2_large">SigLIP 2 Large</option>
                        <option value="siglip2_so400m">SigLIP 2 So400m</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>保存目录</label>
                    <input type="text" id="group-save-dir" name="save_dir" placeholder="similar_groups">
                </div>
                <div class="form-group">
                    <label>聚类数量（可选，留空自动优化）</label>
                    <input type="number" id="group-n-clusters" name="n_clusters" min="2" max="50"
                           placeholder="留空则按轮廓系数自动选择">
                </div>
                <div class="form-group">
                    <label>每聚类最少图片数（小于该值的聚类将被过滤）</label>
                    <input type="number" id="group-min-cluster-size" name="min_cluster_size" value="2" min="1" max="100">
                </div>
                <div class="form-group">
                    <label>每组拼图展示图片数</label>
                    <input type="number" id="group-images-per-group" name="images_per_group" value="4" min="1" max="12">
                </div>
                <button type="submit" class="btn btn-green">聚类分组</button>
            </form>
            <div id="group-result" class="result"></div>
        </div>
    </div>

    <script>
        // ---------- 工具函数 ----------
        function showLoading(el, msg) {
            el.className = 'result loading';
            el.style.display = 'block';
            el.innerHTML = `<div class="spinner"></div>${msg}`;
        }
        function showResult(el, data, isError) {
            el.style.display = 'block';
            if (isError || (data && data.code && data.code !== 200)) {
                el.className = 'result error';
                el.innerHTML = `<strong>❌ 错误：</strong>${(data && data.message) || data}`;
                return;
            }
            el.className = 'result success';
            if (typeof data === 'object' && data.data) {
                renderData(el, data.data);
            } else {
                el.innerHTML = `<strong>✅ 成功</strong><pre>${JSON.stringify(data, null, 2)}</pre>`;
            }
        }
        function renderData(el, data) {
            let html = '<strong>✅ 成功</strong>';
            // 人脸识别
            if (data.recognized_faces) {
                html += `<p>识别到 ${data.recognized_faces.length} 张人脸</p><div class="face-grid">`;
                data.recognized_faces.forEach(f => {
                    html += `<div class="face-item">
                        <img src="/api/file?path=${encodeURIComponent(f.face_image_path)}" alt="face">
                        <div class="name">${f.identified_as}</div>
                        <div class="conf">置信度: ${f.identification_confidence.toFixed(3)}</div>
                    </div>`;
                });
                html += '</div>';
                if (data.annotated_image_path) {
                    html += `<div class="image-grid">
                        <div class="image-item"><img src="/api/file?path=${encodeURIComponent(data.annotated_image_path)}" alt="标注图"><p>标注图</p></div>
                    </div>`;
                }
                el.innerHTML = html;
                return;
            }
            // 组合处理
            if (data.detect_path) {
                const items = [
                    ['original_path', '原图'], ['detect_path', '人像检测'],
                    ['extracted_path', '服装分割'], ['pose_path', '镜头姿态'],
                    ['inpainted_path', '背景修复']
                ];
                html += `<p>镜头类型：<b>${data.shot_type}</b>（面积比 ${data.area_ratio}）</p><div class="image-grid">`;
                items.forEach(([k, label]) => {
                    if (data[k]) {
                        html += `<div class="image-item"><img src="/api/file?path=${encodeURIComponent(data[k])}" alt="${label}"><p>${label}</p></div>`;
                    }
                });
                html += '</div>';
                el.innerHTML = html;
                return;
            }
            // 检索结果
            if (data.results) {
                if (data.results.length === 0) {
                    el.innerHTML = html + '<p>未找到相似图片</p>';
                    return;
                }
                html += `<p>找到 ${data.results.length} 张相似图片</p><div class="image-grid">`;
                data.results.forEach((r, i) => {
                    const sim = (r.similarity !== undefined) ? `相似度 ${r.similarity.toFixed(3)}` : '';
                    html += `<div class="image-item">
                        <img src="/api/file?path=${encodeURIComponent(r.image_path)}" alt="result ${i+1}">
                        <p>#${i+1} ${sim}</p>
                    </div>`;
                });
                html += '</div>';
                el.innerHTML = html;
                return;
            }
            // 聚类分组
            if (data.groups) {
                if (data.groups.length === 0) {
                    el.innerHTML = html + '<p>未找到满足条件的聚类分组</p>';
                    return;
                }
                html += `<p>共 ${data.group_count} 个分组</p>`;
                data.groups.forEach(g => {
                    html += `<p><b>分组 ${g.group_id}</b>：<span style="color:#2563eb">${g.image_count} 张</span></p>`;
                    const collages = g.collage_paths || [];
                    if (collages.length > 0) {
                        html += '<div class="image-grid">';
                        collages.forEach((cp, pi) => {
                            const label = collages.length > 1 ? `分组拼图 第${pi+1}/${collages.length}页` : '分组拼图';
                            html += `<div class="image-item">
                                <img src="/api/file?path=${encodeURIComponent(cp)}" alt="分组 ${g.group_id} 拼图${pi+1}" style="height:auto">
                                <p>${label}</p>
                            </div>`;
                        });
                        html += '</div>';
                    }
                    const shown = g.image_names.slice(0, 8);
                    const more = g.image_count > shown.length ? ` … 等共 ${g.image_count} 张` : '';
                    html += `<p style="font-size:11px;color:#64748b;word-break:break-all">${shown.join('、')}${more}</p>`;
                });
                el.innerHTML = html;
                return;
            }
            // 其他（索引构建、人脸库等）
            html += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            el.innerHTML = html;
        }
        function submitForm(formId, url, resultId, loadingMsg) {
            const form = document.getElementById(formId);
            const el = document.getElementById(resultId);
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                showLoading(el, loadingMsg);
                const fd = new FormData(this);
                fetch(url, { method: 'POST', body: fd })
                    .then(r => r.json())
                    .then(d => showResult(el, d))
                    .catch(err => showResult(el, { message: err.message }, true));
            });
        }

        // ---------- 健康检查 ----------
        function checkHealth() {
            const pill = document.querySelector('.status-pill .dot');
            const txt = document.getElementById('status-text');
            txt.textContent = '检测中...';
            fetch('/api/health')
                .then(r => r.json())
                .then(d => {
                    pill.style.background = '#4ade80';
                    txt.textContent = `服务正常 (${d.device})`;
                })
                .catch(() => {
                    pill.style.background = '#f87171';
                    txt.textContent = '服务不可用';
                });
        }

        // ---------- 表单绑定 ----------
        submitForm('recognize-form', '/api/face/recognize', 'recognize-result', '正在识别人脸...');
        submitForm('build-db-form', '/api/face/build_database', 'build-db-result', '正在构建人脸库...');
        submitForm('process-form', '/api/image/process', 'process-result', '正在组合处理...');
        submitForm('index-form', '/api/embedding/build_index', 'index-result', '正在构建索引...');
        submitForm('search-form', '/api/embedding/search', 'search-result', '正在检索...');
        submitForm('group-form', '/api/embedding/group_similar', 'group-result', '正在聚类分组...');

        // 页面加载时自动检测一次服务状态
        checkHealth();
    </script>
</body>
</html>
"""


# ---------- 界面路由（附加到 main.app，不影响原有 API） ----------

@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui_page():
    """可视化操作界面"""
    return HTMLResponse(content=UI_HTML)


@app.get("/organize", response_class=HTMLResponse, include_in_schema=False)
async def organize_page():
    """智能整理可视化界面"""
    return HTMLResponse(content=ORGANIZE_HTML)


# 移除 main.py 中根路径指向 /docs 的路由，改为指向可视化界面 /ui
_main_index_route = None
for _i, _r in enumerate(app.routes):
    if getattr(_r, 'path', None) == '/' and getattr(_r, 'name', None) == 'index':
        _main_index_route = _r
        break
if _main_index_route is not None:
    app.routes.remove(_main_index_route)

@app.get("/", include_in_schema=False)
async def index_redirect():
    """根路径重定向到可视化界面"""
    return RedirectResponse(url="/ui")


if __name__ == "__main__":
    import uvicorn

    # 统一日志：业务日志 + uvicorn 日志 + print 全部落盘 logs/server.log
    from core_modules.tools.logger import setup_logging
    _logger, _log_file = setup_logging()
    _logger.info("Starting UI server on http://0.0.0.0:8199 (ui: http://0.0.0.0:8199/ui)")

    print("=" * 60)
    print("可视化界面服务启动")
    print("  界面地址: http://localhost:8199/ui")
    print("  API 文档: http://localhost:8199/docs")
    print("=" * 60)
    # log_config=None：交给我们自己的日志配置，uvicorn 日志统一落盘
    uvicorn.run(app, host="0.0.0.0", port=8199, log_config=None, log_level="info")
