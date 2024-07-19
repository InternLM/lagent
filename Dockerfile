# 使用官方的 Python 镜像作为基础镜像
FROM python:3.10-slim

# 安装必要的工具
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ARG LLM_URL
ENV LLM_URL=${LLM_URL}

# 设置环境变量
ENV PATH=/opt/conda/bin:$PATH

# 创建并激活 fastapi 环境
RUN conda create --name fastapi python=3.10 -y

# 激活环境并安装依赖包
RUN /bin/bash -c "source activate fastapi && \
    pip install sse-starlette janus pyvis -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install lmdeploy==0.2.3 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install git+https://github.com/liujiangning30/lagent@ljn/fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple"

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到工作目录
COPY . /app

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["/bin/bash", "-c", "source activate fastapi && uvicorn app:app --host 0.0.0.0 --port 8000"]
