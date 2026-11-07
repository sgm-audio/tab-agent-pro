#!/usr/bin/env bash
#
# Upload this checkout to a Hugging Face Docker Space.
#
# The Space runtime only needs the files used inside the image — keep this list
# in sync with .dockerignore-less requirements (see docs-audit P2-3: the live
# demo had drifted from the repository because no deployment step existed).
#
# Usage:
#   HF_TOKEN=hf_xxx ./tools/deploy_space.sh <owner>/<space> [revision]
#
# Example:
#   HF_TOKEN=hf_xxx ./tools/deploy_space.sh scottymills/tab-agent-pro

set -euo pipefail

SPACE="${1:?usage: deploy_space.sh <owner>/<space> [revision]}"
REVISION="${2:-main}"

# The `hf` CLI comes from huggingface_hub[cli]; fail with a clear message
# instead of a bare "command not found" (docs-audit H8).
if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' CLI is required: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens" >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

FILES=(
  Dockerfile
  requirements.txt
  requirements.lock
  agents.py
  app.py
  main.py
  monitoring.py
  init_memory.py
  suno_postprocessor.py
  README.md
)

INCLUDE_ARGS=()
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  INCLUDE_ARGS+=("--include" "$f")
done

echo "Uploading ${#FILES[@]} files + examples/ to $SPACE ($REVISION) ..."
hf upload "$SPACE" . . \
  --repo-type space \
  --revision "$REVISION" \
  "${INCLUDE_ARGS[@]}" \
  --include "examples/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc"

host="$(printf '%s' "$SPACE" | tr '/_' '--' | tr '[:upper:]' '[:lower:]')"
echo
echo "Uploaded. Check the build at https://huggingface.co/spaces/$SPACE"
echo "Health:   https://${host}.hf.space/health"
