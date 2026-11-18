# Stage 1: Build deps — compilers, git, Python wheels
FROM python:3.12-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

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

# Torch wheels. Defaults to the CPU-only index. For a GPU image, override at
# build time:
#
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#                -t tab-agent-cuda .
#
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.0 \
#                -t tab-agent-rocm .
#
# Match the ONNX Runtime package to your backend too (see requirements.txt),
# e.g. onnxruntime-gpu for CUDA or onnxruntime-vulkan for AMD Vulkan.
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url "$TORCH_INDEX_URL"

COPY requirements.txt requirements.lock pyproject.toml ./
COPY tools/check_lockfile.py ./tools/

# Install the ONNX-backed Basic Pitch without its default TensorFlow dependency.
# Constrain resolution to the runtime lock; mir_eval and resampy are explicitly
# declared in requirements.txt because Basic Pitch imports them at runtime.
RUN pip install --no-deps -c requirements.lock "basic-pitch>=0.4.0,<0.5" "mir_eval>=0.6" "resampy>=0.2.2,<0.4.3" && \
    grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt && \
    pip install -c requirements.lock -r /tmp/req-nobp.txt

# Re-apply the lock without resolving deps (including those installed with
# Torch), and fail the build if the installed set or its closure disagrees.
RUN python tools/check_lockfile.py --interpreter && \
    pip install --no-deps -r requirements.lock && \
    python tools/check_lockfile.py

# Stage 2: Runtime — lean, no build toolchain
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The container listens on 0.0.0.0 so a local Docker client (or any
    # reverse proxy in front of it) can reach the web UI.
    HOST=0.0.0.0 \
    PORT=7860 \
    # Writable cache location for the owner of /app; models land here.
    XDG_CACHE_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN groupadd -r tabagent && useradd -r -g tabagent -d /app -s /sbin/nologin tabagent

# Application code is read-only to the runtime user; make only the required
# cache, preferences and output paths writable.
COPY --chown=root:root . .
RUN mkdir -p /app/.cache /app/user_memory /tmp/tab_agent_outputs \
    && chown -R tabagent:tabagent /app/.cache /app/user_memory /tmp/tab_agent_outputs \
    && chmod a-w /app/*.py

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
