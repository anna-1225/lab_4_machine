
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py /app/src/app.py

COPY exported_model /app/exported_model

ENV MODEL_PATH=/app/exported_model

RUN useradd --create-home appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
