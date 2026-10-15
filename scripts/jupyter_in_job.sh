#!/usr/bin/env bash
# Run JupyterLab (envs/train) on the GPU of a job you already have, e.g. one started with srun/salloc.
#
#   srun --jobid=<id> --overlap bash scripts/jupyter_in_job.sh | tee logs/jupyter-<id>.log
#
# Run it from a tmux window on the login node so it survives disconnects. The URL to paste into VS Code
# ("Existing Jupyter Server") appears in the output as http://gnodeXX:PORT/lab?token=...
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# GPU nodes are shared, so let the OS pick a free port.
PORT="$(envs/train/bin/python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
echo "node $(hostname), port $PORT, GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

exec envs/train/bin/jupyter lab --no-browser --ip="$(hostname)" --port="$PORT" --notebook-dir="$PWD"
