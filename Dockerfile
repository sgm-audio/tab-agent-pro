# 1. Base Image: Python 3.10 Slim (Lightweight, No GPU drivers)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 2. System Dependencies
# ffmpeg: Required for Demucs & Librosa audio processing
# libsndfile1: Required for soundfile library
# build-essential: Required for compiling some Python extensions
# git: Required for installing YourMT3 from GitHub
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Python Setup
RUN python -m pip install --upgrade pip

# 4. Install Deep Learning Frameworks (CPU Versions)
# We do this BEFORE requirements.txt to prevent pip from grabbing heavy GPU versions
# PyTorch is used by Demucs, TensorFlow is used by Basic Pitch
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# torchcodec needed for torchaudio.save on newer torchaudio versions
RUN pip install torchcodec --index-url https://download.pytorch.org/whl/cpu || true

# 5. Install Project Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 7. Copy Codebase
COPY . .

# 8. Default Command
CMD ["python", "app.py"]
