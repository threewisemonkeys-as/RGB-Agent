#!/usr/bin/env bash
# Reproducible launcher for RGB-Agent under rootless static podman (no sudo, no subuid).
# Runs the bundled local ls20 environment in OFFLINE mode (no ARC_API_KEY needed),
# with Gemini 2.5 Flash via OpenRouter as the analyzer model.
#
# Usage: ./run_rootless.sh [extra rgb-swarm args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Rootless podman (installed to ~/.local) + docker->podman shim are on PATH here.
export PATH="$HOME/.local/bin:$PATH"
export CONTAINERS_CONF="$HOME/.config/containers/containers.conf"

exec ./.venv/bin/rgb-swarm \
  --suite ls20 \
  --max-actions 50 \
  --operation-mode offline \
  --model openrouter/google/gemini-2.5-flash \
  --no-resume \
  --no-tool-restrictions \
  "$@"
