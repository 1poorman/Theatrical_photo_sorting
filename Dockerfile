# 基于NVIDIA CUDA镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 设置conda环境变量
ENV PATH=/opt/conda/bin:$PATH

# 创建conda环境 inpaint
RUN conda create -n inpaint python=3.9 -y

# 激活环境并安装依赖
SHELL ["conda", "run", "-n", "inpaint", "/bin/bash", "-c"]

# 安装PyTorch CUDA版本
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 安装xformers（本地whl文件）
RUN pip install /app/weights/xformers-0.0.23+cu118-cp39-cp39-manylinux2014_x86_64.whl

# 安装其他核心依赖
RUN pip install \
    opencv-python==4.10.0.82 \
    numpy==1.26.4 \
    transformers==4.30.0 \
    modelscope==1.10.0 \
    ultralytics \
    faiss-gpu \
    flask \
    scikit-learn \
    Pillow \
    tqdm \
    matplotlib \
    scikit-image \
    timm \
    einops \
    scipy \
    pyyaml \
    werkzeug

# 创建工作目录
WORKDIR /app

# 复制项目代码（排除dinov2目录通过.dockerignore）
COPY . /app/

# 创建必要的目录
RUN mkdir -p /app/outputs /app/weights /app/data

# 设置权限
RUN chmod -R 755 /app

# 暴露端口
EXPOSE 8089

# 设置入口点 - 使用conda环境运行
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "inpaint", "python"]

# 默认启动Server.py
CMD ["Server.py"]
