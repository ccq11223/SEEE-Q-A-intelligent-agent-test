FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 只装在线服务需要的依赖，镜像更小、构建更快
COPY requirements-app.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-app.txt

COPY app ./app
COPY pipeline ./pipeline
COPY config ./config
COPY data/work ./data/work

# 在线服务只需要课件原图，其余中间产物不进镜像
RUN find data/work -type f ! -name "*.jpg" ! -name "*.png" -delete

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
