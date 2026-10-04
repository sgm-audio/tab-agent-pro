#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Auto-install if missing
if ! python3 -c "import gradio, basic_pitch, demucs, librosa" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Load .env if present (app.py also reads it; main.py imports nothing that does,
# so honour it here as well). Real environment variables take priority.
if [ -f "$DIR/.env" ]; then
  while IFS='=' read -r key value; do
    case "$key" in
      ''|\#*) continue ;;
    esac
    key="$(echo "$key" | tr -d '[:space:]')"
    value="${value%\"}"; value="${value#\"}"
    if [ -n "$key" ] && [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < "$DIR/.env"
fi

if [ "${1:-}" = "--web" ] || [ "${1:-}" = "" ]; then
  exec python3 app.py
else
  exec python3 main.py "$@"
fi
