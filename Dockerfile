# Stage 1: Build deps — compilers, git, Python wheels
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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

# Stage 2: Runtime — lean, no build toolchain
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Hugging Face Spaces routes to port 7860 and requires the app to listen on
# 0.0.0.0 (not 127.0.0.1) so the Space proxy can reach it.
ENV HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN groupadd -r tabagent && useradd -r -g tabagent -d /app -s /sbin/nologin tabagent \
    && chown -R tabagent:tabagent /app

COPY . .

USER tabagent

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Verify a transcription backend actually imported before declaring the image good
RUN python -c "import agents; assert agents.BASIC_PITCH_AVAILABLE, 'Basic Pitch failed to import'"

CMD ["python", "app.py"]
