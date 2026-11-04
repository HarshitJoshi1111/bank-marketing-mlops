# Prediction API for the champion model (built after `dvc repro`, which exports it to models/serving).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Exactly the library versions the model was trained with (a pickled scikit-learn model needs them)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/__init__.py src/params.py src/prepare.py src/
COPY serve/ serve/
COPY models/serving/ models/serving/

RUN useradd --create-home api
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
