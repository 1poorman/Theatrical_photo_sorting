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

# ---------- 可视化界面 HTML ----------

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
        <div class="status-pill" onclick="checkHealth()" title="点击检测服务状态">
            <span class="dot"></span><span id="status-text">检测服务状态</span>
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
