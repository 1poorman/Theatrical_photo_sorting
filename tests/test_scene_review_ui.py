# -*- coding: utf-8 -*-
"""M15 UI 冒烟：/organize 页面包含场景复核卡片与关键端点。"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(PROJECT_ROOT, 'app'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'app'))

from fastapi.testclient import TestClient
from server_ui import ORGANIZE_HTML
from main import app

assert 'data-view="review"' in ORGANIZE_HTML
assert 'name="scene_evidence_file"' in ORGANIZE_HTML
with open(os.path.join(PROJECT_ROOT, 'app', 'static', 'workspace.js'), encoding='utf-8') as f:
    script = f.read()
assert '/api/organize/scene/review' in script
assert '/review/seed' in script and '/review/export' in script

resp = TestClient(app).get('/organize')
assert resp.status_code == 200
assert '场景复核' in resp.text and 'scene_evidence_file' in resp.text
print('✅ M15 场景复核 UI 冒烟通过（工作台 + scene_evidence_file 入口）')
