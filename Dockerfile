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
RUN pip install torchcodec --index-url https://download.pytorch.org/whl/cpu || true

COPY requirements.txt .
# ponytail: basic-pitch declares tensorflow dep but uses ONNX on Linux
RUN pip install --no-deps basic-pitch>=0.4.0 && \
    grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt && \
    pip install -r /tmp/req-nobp.txt

# Stage 2: Runtime — lean, no build toolchain
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN groupadd -r tabagent && useradd -r -g tabagent -d /app -s /sbin/nologin tabagent \
    && chown -R tabagent:tabagent /app

COPY . .

USER tabagent

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "app.py"]
