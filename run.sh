#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Auto-install if missing
if ! python3 -c "import gradio, basic_pitch, demucs, librosa" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install --no-deps "basic-pitch>=0.4.0,<0.5"
  pip install -r requirements.txt
fi

# Load .env if present (app.py and main.py also read it; honour it in the shell
# environment as well). Real environment variables take priority.
if [ -f "$DIR/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    # Strip leading whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in
      ''|\#*) continue ;;
    esac
    case "$line" in
      *=*)
        key="${line%%=*}"
        value="${line#*=}"
        key="$(printf '%s' "$key" | tr -d '[:space:]')"
        # Trim surrounding whitespace from value
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        # Strip matching surrounding quotes
        case "$value" in
          \"*\") value="${value#\"}"; value="${value%\"}" ;;
          \'*\') value="${value#\'}"; value="${value%\'}" ;;
        esac
        if [ -n "$key" ] && [ -z "${!key:-}" ]; then
          export "$key=$value"
        fi
        ;;
    esac
  done < "$DIR/.env"
fi

if [ "${1:-}" = "--web" ] || [ "${1:-}" = "" ]; then
  exec python3 app.py
else
  exec python3 main.py "$@"
fi
