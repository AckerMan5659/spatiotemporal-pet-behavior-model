FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements_api.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements_api.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
