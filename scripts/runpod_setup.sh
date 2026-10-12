#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (use a PyTorch template; torch+CUDA are preinstalled).
# Data and outputs live on the network volume (/workspace) so they survive pod restarts.
#
#   cd /workspace && git clone https://github.com/Acmaro/NY-electricity-load-prediction.git
#   cd NY-electricity-load-prediction && bash scripts/runpod_setup.sh
#   source ~/.nyload_env && ./run.sh train
set -euo pipefail

VOLUME=${VOLUME:-/workspace}
YEARS=${YEARS:-2013-2023}

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda)"

pip install -q -e ".[dev]"

cat > ~/.nyload_env <<ENV
export NYLOAD_DATA_DIR=$VOLUME/nyload/data
export NYLOAD_OUTPUT_DIR=$VOLUME/nyload/outputs
ENV
grep -q nyload_env ~/.bashrc || echo "source ~/.nyload_env" >> ~/.bashrc
source ~/.nyload_env

# Downloads are cached on the volume, so re-running only builds what is missing.
python -m nyload data --years "$YEARS"
python -m nyload test
echo "ready: ./run.sh train"
