# 使用轻量级的 Python 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（main.py 必须和 openai.py 在一起）
COPY main.py .
COPY openai.py .

# 暴露代码中定义的端口
EXPOSE 30016

# 启动命令
CMD ["python", "openai.py"]
