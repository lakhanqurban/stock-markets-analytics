# Hourly crypto algo-trading pipeline (Stock Market Analytics Zoomcamp 2026)
FROM python:3.11-slim

WORKDIR /app

# system deps for pyarrow/matplotlib wheels are self-contained; keep image slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY notebooks/ notebooks/
# data/, models/, outputs/ are mounted at runtime (git-ignored, not baked in)
RUN mkdir -p data/raw models outputs

# default: full pipeline from downloaded (incremental) data
# docker run --rm -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs hourly-crypto \
#   python -m src.pipeline --source download
CMD ["python", "-m", "src.pipeline", "--source", "file"]
