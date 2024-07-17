FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# 复制本地文件到容器中
COPY . /app

# 复制环境配置文件
COPY environment.yml /app/environment.yml

# 创建并激活 conda 环境
RUN conda env create -f /app/environment.yml && \
    echo "source activate internlm" > ~/.bashrc

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
