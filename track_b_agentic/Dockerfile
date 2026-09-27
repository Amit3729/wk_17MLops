FROM python:3.12-slim
# uv installs exactly what uv.lock pins - the same resolution as a dev machine.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

COPY pyproject.toml uv.lock .python-version ./
# --frozen: fail rather than re-resolve; --no-dev: no pytest in the image.
RUN uv sync --frozen --no-dev --no-install-project

# Bake the ONNX embedding weights into the image so a cold container does not
# download ~130MB on its first request (and can run with no network at all).
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBEDDING_MODEL}', cache_dir='/app/.cache/fastembed')"

COPY app ./app
COPY data ./data
COPY streamlit_app.py .

# Non-root: the container writes only to the Chroma directory and the model cache.
RUN useradd --create-home --uid 10001 assistant && chown -R assistant:assistant /app
USER assistant

EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
