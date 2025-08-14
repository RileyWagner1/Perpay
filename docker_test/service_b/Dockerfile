FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn[standard]==0.30.0 httpx==0.27.0
COPY app.py /app/app.py
EXPOSE 8001
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8001"]
