FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models/translation-en-ru ./models/translation-en-ru
ENV PYTHONUNBUFFERED=1 RUNS_DIR=/data
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8765"]
