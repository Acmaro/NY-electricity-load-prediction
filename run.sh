#!/usr/bin/env bash
# Linux / WSL2 / macOS / RunPod entry point: ./run.sh <doctor|data|train|eval|runs|test> [args...]
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif python -c "import nyload" 2>/dev/null; then PY=python   # e.g. RunPod system install
else echo "no .venv yet -> run: bash scripts/setup.sh"; exit 1; fi
exec "$PY" -m nyload "$@"
