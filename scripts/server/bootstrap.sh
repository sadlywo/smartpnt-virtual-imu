#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$BUNDLE_PROJECT_ROOT"
BUNDLE_PREFLIGHT_FLAGS=()
if [[ "${1:-}" == "--require-gpu" && $# == 1 ]]; then
  BUNDLE_PREFLIGHT_FLAGS+=(--require-gpu)
elif [[ $# != 0 ]]; then
  printf 'Usage: bash scripts/server/bootstrap.sh [--require-gpu]\n' >&2
  exit 2
fi

printf 'READS: bundle_manifest.json, SHA256SUMS, configs/server_environment.yaml, pyproject.toml, uv.lock and curated data\n'
printf 'WRITES: .venv, uv dependency cache, managed Python 3.12 if needed, runs/server_bootstrap_<time>/preflight.json\n'
printf 'SKIPS: training, raw datasets, prior runs, remote operations, system driver installation\n'
command -v uv >/dev/null || { printf 'FAILED: install uv from https://docs.astral.sh/uv/getting-started/installation/ and retry\n' >&2; exit 1; }
printf 'TARGET: Ubuntu 22.04 / Python 3.12 / PyTorch 2.4.0 / CUDA 12.1 / NumPy 1.26.4; isolated .venv\n'
if ! uv python find 3.12 >/dev/null 2>&1; then
  uv python install 3.12
fi
BUNDLE_BOOTSTRAP_PYTHON="$(uv python find 3.12)"
"$BUNDLE_BOOTSTRAP_PYTHON" -B -S scripts/server/preflight.py --checksums-only
uv lock --check --offline --python 3.12
uv sync --frozen --python 3.12 --no-dev
BUNDLE_BOOTSTRAP_RUN="runs/server_bootstrap_$(date -u +%Y%m%d_%H%M%S_%N)"
mkdir -p -- "$BUNDLE_BOOTSTRAP_RUN"
.venv/bin/python -B scripts/server/preflight.py "${BUNDLE_PREFLIGHT_FLAGS[@]}" --report "$BUNDLE_BOOTSTRAP_RUN/preflight.json"
printf 'VERIFIED: frozen environment installed; preflight report: %s/preflight.json\n' "$BUNDLE_BOOTSTRAP_RUN"
