# syntax=docker/dockerfile:1
# Theatrical Photo Sorting 服务镜像
# CUDA 12.8 运行时由 PyTorch 轮子自带，与 ONNX Runtime / tensorrt-cu12 保持一致。
# GPU 依赖宿主机 NVIDIA 驱动 + nvidia-container-toolkit（见 docker-compose.yml）
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 系统依赖（OpenCV 运行库 + 健康检查 curl）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 基础镜像的旧 pip 无法正确规范化部分新 wheel 的项目名称。
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install --upgrade 'pip>=25,<27'

# 先装 PyTorch（cu128 索引，层缓存友好），再装其余依赖。
# ModelScope 的 pipeline 还需要 framework 扩展依赖。
COPY requirements/ ./requirements/
ARG DEPENDENCY_INDEX_URL=https://pypi.org/simple
# 通用依赖先走可配置镜像源，CUDA/PyTorch wheel 仍从官方索引获取。
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install --index-url "$DEPENDENCY_INDEX_URL" \
    filelock 'typing-extensions>=4.10' 'sympy>=1.13.3' 'networkx>=2.5.1' \
    jinja2 'fsspec>=0.8.5' numpy==2.2.6 pillow==12.3.0
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR pip install --index-url "$DEPENDENCY_INDEX_URL" \
    -r requirements/vision.txt "modelscope[framework]==1.39.1"

# 工作台的高级智能体与视觉服务使用同一 Python 环境。
RUN --mount=type=cache,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR pip install --index-url "$DEPENDENCY_INDEX_URL" -r requirements/agent.txt

# pip 的 CUDA/TRT 动态库不在系统默认搜索路径，注册后 ORT 可直接加载。
RUN python -c "import site; from pathlib import Path; root = Path(site.getsitepackages()[0]); dirs = sorted({str(p) for pattern in ('nvidia/*/lib', 'tensorrt*_libs', 'torch/lib') for p in root.glob(pattern) if p.is_dir()}); Path('/etc/ld.so.conf.d/python-cuda.conf').write_text('\n'.join(dirs) + '\n')" \
    && ldconfig

# 复制项目代码（weights/ data/ 等通过 .dockerignore 排除，运行时挂载）
COPY app/ ./app/
COPY config/ ./config/
COPY core_modules/ ./core_modules/
COPY tools/ ./tools/

# 运行产物目录
RUN mkdir -p /app/outputs /app/logs /app/weights

EXPOSE 8198 8199

# 默认启动可视化工作台，全部 API 同样由 8199 端口提供。
CMD ["python", "app/server_ui.py"]
