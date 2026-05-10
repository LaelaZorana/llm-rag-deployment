FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn numpy transformers torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
