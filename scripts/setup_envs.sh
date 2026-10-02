#!/usr/bin/env bash
# Build both uv environments. Run on a GPU node: --torch-backend=auto picks the
# PyTorch/CUDA build from the driver nvidia-smi reports, so on a login node
# (no GPU) it would silently select CPU wheels.
#
#   bash scripts/setup_envs.sh          # both envs
#   bash scripts/setup_envs.sh serve    # only one
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

PY_VERSION=3.12
KERNEL_NAME="train-4090"
KERNEL_DISPLAY="train (4090)"
WHAT="${1:-all}"

command -v uv >/dev/null || { echo "uv not found: install it first (see SETUP.md)" >&2; exit 1; }
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "No GPU visible on $(hostname). Run this inside a GPU job, e.g.:" >&2
    echo "  srun --partition=gpu --gpus-per-node=rtx4090:1 --cpus-per-task=8 --mem=40G --time=02:00:00 --pty bash" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

make_venv() {
    [ -x "envs/$1/bin/python" ] || uv venv --python "$PY_VERSION" --seed --managed-python "envs/$1"
    # uv writes a .gitignore with "*" into the venv, which would hide the lock file from git.
    printf '*\n!requirements.lock\n' > "envs/$1/.gitignore"
}

torch_summary() {
    "envs/$1/bin/python" -c 'import torch; print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={torch.cuda.is_available()}"); assert torch.cuda.is_available(), "torch cannot see the GPU"'
}

setup_serve() {
    echo "=== envs/serve (vLLM) ==="
    make_venv serve
    # vLLM pins its own torch; nothing else that depends on torch goes in here.
    uv pip install --python envs/serve/bin/python --torch-backend=auto vllm openai
    torch_summary serve
    envs/serve/bin/python -c 'import vllm; print("vllm", vllm.__version__)'
    uv pip freeze --python envs/serve/bin/python > envs/serve/requirements.lock
}

# FlashAttention 2 only from a prebuilt wheel that matches torch + CUDA + Python
# exactly. Never compiled from source here; without a match we stay on SDPA.
install_flash_attn() {
    local py=envs/train/bin/python tag url
    tag="$($py - <<'EOF'
import sys, torch
major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cuda = (torch.version.cuda or "").split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"cu{cuda}torch{major_minor}cxx11abi{abi}-{cp}-{cp}-linux_x86_64.whl")
EOF
)"
    url="$(curl -fsSL -m 30 'https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=10' \
        | grep -o '"browser_download_url": *"[^"]*"' | cut -d'"' -f4 | grep -F "+$tag" | head -1 || true)"
    if [ -z "$url" ]; then
        echo "flash-attn: no prebuilt wheel for $tag -> using PyTorch SDPA"
        return 0
    fi
    echo "flash-attn: installing $url"
    if uv pip install --python "$py" --no-deps "$url" && $py -c 'import flash_attn; print("flash_attn", flash_attn.__version__)'; then
        return 0
    fi
    echo "flash-attn: wheel did not import cleanly, removing -> using PyTorch SDPA"
    uv pip uninstall --python "$py" flash-attn || true
}

setup_train() {
    echo "=== envs/train ==="
    make_venv train
    uv pip install --python envs/train/bin/python --torch-backend=auto torch
    local before after
    before="$(envs/train/bin/python -c 'import torch; print(torch.__version__)')"
    # Same backend flag again so the resolver keeps the CUDA build of torch.
    uv pip install --python envs/train/bin/python --torch-backend=auto \
        transformers accelerate datasets peft trl bitsandbytes tokenizers \
        sacrebleu matplotlib pandas tqdm jupyterlab ipykernel
    after="$(envs/train/bin/python -c 'import torch; print(torch.__version__)')"
    [ "$before" = "$after" ] || { echo "torch changed during install: $before -> $after" >&2; exit 1; }
    torch_summary train
    install_flash_attn
    # The repo itself (editable) plus pytest, ruff and pre-commit; none of these touch torch.
    uv pip install --python envs/train/bin/python --torch-backend=auto -e ".[dev]"
    envs/train/bin/python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"
    uv pip freeze --python envs/train/bin/python > envs/train/requirements.lock
}

case "$WHAT" in
    serve) setup_serve ;;
    train) setup_train ;;
    all)   setup_serve; setup_train ;;
    *)     echo "usage: $0 [serve|train|all]" >&2; exit 2 ;;
esac
echo "Done. Next: make serve-check && make train-check"
