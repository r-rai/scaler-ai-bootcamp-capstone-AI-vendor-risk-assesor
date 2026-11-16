# syntax=docker/dockerfile:1
# One image for the API, the UI and the sandboxed fetcher (docker-compose.yml chooses the command).
# Two stages: build wheels with a compiler, run without one.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt requirements-prod.txt ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements-prod.txt

FROM python:3.12-slim AS runtime
ARG GIT_SHA=unknown
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    VRA_GIT_SHA=$GIT_SHA \
    VRA_RUNS_DIR=/data/runs \
    HF_HOME=/data/models-cache/hf FASTEMBED_CACHE_PATH=/data/models-cache/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/tmp
# A fixed, unprivileged user: the container never runs as root.
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
 && mkdir -p /data/runs /data/models-cache && chown -R app:app /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app assessor ./assessor
COPY --chown=app:app app ./app
COPY --chown=app:app fetcher ./fetcher
COPY --chown=app:app data ./data
COPY --chown=app:app eval ./eval
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app run_assessor.py ./
USER app
EXPOSE 8000 8501 8600
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
