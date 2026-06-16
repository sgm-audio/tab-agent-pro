#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Auto-install if missing
if ! python3 -c "import gradio, basic_pitch, demucs, librosa" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

if [ "${1:-}" = "--web" ] || [ "${1:-}" = "" ]; then
  exec python3 app.py
else
  exec python3 main.py "$@"
fi
