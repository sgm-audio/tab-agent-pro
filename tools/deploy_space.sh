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

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens" >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

FILES=(
  Dockerfile
  requirements.txt
  agents.py
  app.py
  main.py
  monitoring.py
  init_memory.py
  suno_postprocessor.py
  README.md
)

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

echo "Uploading ${#FILES[@]} files + examples/ to $SPACE ($REVISION) ..."
hf upload "$SPACE" . "$REVISION" \
  --repo-type space \
  --exclude ".git/*" \
  --exclude "tests/*" \
  --exclude "tools/*" \
  --exclude "reaper/*" \
  --exclude "*.md" \
  --include "README.md" \
  --include "${FILES[@]}" \
  --include "examples/*"

host="$(echo "$SPACE" | tr '/_' '--')"
echo
echo "Uploaded. Check the build at https://huggingface.co/spaces/$SPACE"
echo "Health:   https://${host}.hf.space/health"
