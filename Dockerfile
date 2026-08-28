# Theatrical Photo Sorting 服务镜像
# 基础镜像用 python:3.10-slim：CUDA 运行时由 pip 轮子（torch cu130 / onnxruntime / tensorrt）自带，
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

# 先装 PyTorch（cu130 索引，层缓存友好），再装其余依赖
COPY requirements_scrfd_arcface.txt .
RUN pip install torch==2.9.0 torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cu130 \
    && pip install -r requirements_scrfd_arcface.txt

# 复制项目代码（weights/ data/ 等通过 .dockerignore 排除，运行时挂载）
COPY app/ ./app/
COPY config/ ./config/
COPY core_modules/ ./core_modules/
COPY tests/ ./tests/

# 运行产物目录
RUN mkdir -p /app/outputs /app/logs /app/weights

EXPOSE 8198 8199

# 默认启动纯 API 服务（可视化界面：python app/server_ui.py）
CMD ["python", "app/server.py"]
