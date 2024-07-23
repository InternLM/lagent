# 使用Dockerfile-base制作的镜像作为基础镜像
FROM lagent-fastapi-puyu:latest

ARG PUYU_API_KEY
ENV PUYU_API_KEY=${PUYU_API_KEY}

ARG BING_API_KEY
ENV BING_API_KEY=${BING_API_KEY}

# 克隆git仓库
RUN git clone --branch ljn/fastapi https://github.com/liujiangning30/lagent.git /app
WORKDIR /app

# 激活环境并安装依赖包
RUN /bin/bash -c "source activate fastapi && \
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple"

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["/bin/bash", "-c", "source activate fastapi && uvicorn app:app --host 0.0.0.0 --port 8000"]
