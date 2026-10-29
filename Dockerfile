FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY finsight ./finsight
# Set RERANK=1 to bake the CPU cross-encoder into the image (adds ~1GB).
ARG RERANK=0
RUN pip install --upgrade pip && \
    if [ "$RERANK" = "1" ]; then pip install ".[rerank,langfuse]"; else pip install ".[langfuse]"; fi && \
    pip install pandas

COPY ui ./ui
COPY eval ./eval

# data/ (filings, XBRL cache, index, traces) is mounted as a volume - see docker-compose.yml
EXPOSE 8000 8501
CMD ["uvicorn", "finsight.api:app", "--host", "0.0.0.0", "--port", "8000"]
