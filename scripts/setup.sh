#!/usr/bin/env bash
# One-time setup on Linux / WSL2 / macOS: creates .venv, installs torch and the package.
#   bash scripts/setup.sh          # install
#   bash scripts/setup.sh --data   # install + download data
# (RunPod: use scripts/runpod_setup.sh, torch is preinstalled there.)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
CUDA=${CUDA:-cu124}

[ -d .venv ] || "$PYTHON" -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install -q --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found -> installing torch ($CUDA)"
    "$PY" -m pip install -q torch --index-url "https://download.pytorch.org/whl/$CUDA"
else
    "$PY" -m pip install -q torch
fi
"$PY" -m pip install -q -e ".[dev]"
[ -f .env ] || cp .env.example .env

"$PY" -m nyload doctor
if [ "${1:-}" = "--data" ]; then "$PY" -m nyload data --years 2013-2023; fi
echo; echo "ready: ./run.sh train"
