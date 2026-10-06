#!/usr/bin/env bash
# Hold one RTX 4090 for 24 h and run JupyterLab from envs/train on it.
# No terminal is attached to the job, so a frozen shell can never take it down:
# shells on the node are separate `srun --jobid=<id> --overlap --pty bash` steps.
#
#   sbatch scripts/jupyter_gpu.sbatch        # from the repo root
#   grep -o 'http://gnode[^ ]*' logs/jupyter-<jobid>.log | head -1   # URL for VS Code
#
#SBATCH --job-name=jupyter-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jupyter-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# GPU nodes are shared, so let the OS pick a free port.
PORT="$(envs/train/bin/python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
echo "node $(hostname), port $PORT, GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

exec envs/train/bin/jupyter lab --no-browser --ip="$(hostname)" --port="$PORT" \
    --notebook-dir="$SLURM_SUBMIT_DIR"
