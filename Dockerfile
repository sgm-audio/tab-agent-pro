# Stage 1: Build deps — compilers, git, Python wheels
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip

# CPU-only wheels. The PyPI torch wheel also pulls 12 NVIDIA CUDA packages
# (~2.5 GB) that a CPU Space cannot use.
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# basic-pitch declares a TensorFlow dependency on Linux, but uses ONNX here.
# Install it with --no-deps and add back the runtime deps it actually imports
# (mir_eval + resampy are imported by basic_pitch.note_creation and are not
# pulled in by anything else in requirements.txt). The version must be quoted,
# otherwise the shell treats ">=0.4.0" as a redirection.
RUN pip install --no-deps "basic-pitch>=0.4.0,<0.5" "mir_eval>=0.6" "resampy>=0.2.2,<0.4.3" && \
    grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt && \
    pip install -r /tmp/req-nobp.txt

# requirements.txt is range-based, so the resolve above can drift from what was
# tested. Re-apply the lock (--no-deps: satisfied packages are skipped and
# TensorFlow cannot be pulled back in), then fail the build if the installed
# set and the lock disagree.
COPY requirements.lock pyproject.toml ./
COPY tools ./tools
RUN pip install --no-deps -r requirements.lock && python tools/check_lockfile.py

# Stage 2: Runtime — lean, no build toolchain
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces routes to port 7860 and requires the app to listen on
# 0.0.0.0 (not 127.0.0.1) so the Space proxy can reach it.
ENV HOST=0.0.0.0 \
    PORT=7860 \
    # Writable cache location for the owner of /app; models land here.
    CACHE_DIR=/app/.cache/tab_agent \
    XDG_CACHE_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN groupadd -r tabagent && useradd -r -g tabagent -d /app -s /sbin/nologin tabagent

# Application code is read-only to the runtime user; only the cache dirs are
# writable, so a request cannot modify code or the image.
COPY --chown=root:root . .
RUN mkdir -p /app/.cache/tab_agent /app/user_memory /tmp/tab_agent_outputs \
    && chown -R tabagent:tabagent /app/.cache /app/user_memory /tmp/tab_agent_outputs \
    && chmod -R a-w /app/*.py /app/reaper 2>/dev/null || true

USER tabagent

EXPOSE 7860

# Verify a transcription backend actually imported before declaring the image
# good. Runs as the runtime user, so it also proves the venv is readable.
RUN python -c "import agents; assert agents.BASIC_PITCH_AVAILABLE, 'Basic Pitch failed to import'"

# Probe readiness, not liveness: /health is 200 even when every transcription
# backend failed to import, /ready is 503 in that case. This is what makes the
# healthcheck able to catch the "image ships no working backend" failure mode.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/ready', timeout=3)"

# exec form so SIGTERM reaches uvicorn directly (graceful shutdown, no shell).
CMD ["python", "app.py"]
