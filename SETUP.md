# Setup

Two isolated uv environments on an RTX 4090 node, plus checks that prove each one works.

> **Status (2026-10-02): partly done.** uv, the cache variables, both empty Python 3.12
> environments, the scripts and the Makefile are in place. The packages are **not installed
> yet** and no check has run: that needs a GPU node (see [Finish the install](#finish-the-install)).
> Rows marked *pending* below get filled in from `make versions` after that.

## Machine

This is not a single workstation but the UBELIX cluster, which changes a few things:

| | |
|---|---|
| OS | Rocky Linux 9.8, native Linux (kernel 5.14) |
| Login node `submit01` | no GPU, no `nvidia-smi`; 125 GB RAM, 128 cores (shared) |
| GPU nodes `gnode22..39` | 8x RTX 4090 each, 740 GB RAM; reached through Slurm (`--partition=gpu --gpus-per-node=rtx4090:1`) |
| Home (`/storage/homefs`, GPFS) | 1 TB quota, about 488 GB free; 1M file quota, about 620k free |
| Scratch (`/rs_scratch`) | 125 TB free, no quota, but purged: `users/kr23w045` no longer exists |
| System Python | Anaconda 3.11.7 (`base` auto-activated); not used by this project |
| sudo | not available, and nothing here needs it; drivers and system CUDA are untouched |

## What is installed and why

| Piece | Where | Why |
|---|---|---|
| uv 0.12.22 | `~/.local/bin` (official installer) | creates the envs, resolves the right PyTorch/CUDA build |
| CPython 3.12.15 (uv-managed) | `~/.local/share/uv/python` | both envs use it; independent of Anaconda |
| `envs/serve` | this repo | vLLM + `openai` client only. vLLM pins its own PyTorch/CUDA build, so nothing else that depends on torch goes in here |
| `envs/train` | this repo | CUDA PyTorch, transformers, accelerate, datasets, peft, trl, bitsandbytes, tokenizers, sacrebleu, matplotlib, pandas, tqdm, jupyterlab, ipykernel; plus this repo installed editable (`pip install -e .[dev]`) with pytest, ruff and pre-commit. The editable install and the dev tools need no GPU and were installed from the login node on 2026-10-03 |
| Jupyter kernel `train (4090)` | `~/.local/share/jupyter/kernels/train-4090` | *pending*, registered by `scripts/setup_envs.sh` |
| FlashAttention 2 | `envs/train` | *pending*, optional: installed only if a prebuilt wheel matches torch + CUDA + Python exactly, otherwise PyTorch SDPA. Never compiled from source |

Caches, exported in `~/.bashrc`:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export UV_CACHE_DIR="$HOME/.cache/uv"
```

Both are on home GPFS, not on scratch. Scratch has far more space, but it is purged, and a
half-purged uv cache can install incomplete packages without an error. Home is persistent, has
488 GB free, and is the same filesystem as `envs/`, so uv hardlinks packages instead of copying
them. `~/.cache/huggingface` already holds about 452 GB of older models and datasets; clearing
the stale ones would free most of the quota.

## Versions

| Component | Version |
|---|---|
| NVIDIA driver | *pending* |
| Max CUDA (driver) | *pending* |
| uv | 0.12.22 |
| Python (both envs) | 3.12.15 |
| torch, `envs/serve` | *pending* |
| vllm | *pending* |
| torch, `envs/train` | *pending* |
| transformers | *pending* |
| trl | *pending* |
| peft | *pending* |
| bitsandbytes | *pending* |

Exact pins land in `envs/serve/requirements.lock` and `envs/train/requirements.lock`
(`uv pip freeze`), which are tracked in git; the environments themselves are ignored.

## Finish the install

`--torch-backend=auto` reads the driver version from `nvidia-smi`, so the install has to run on
a GPU node. On the login node it would quietly pick CPU wheels, which is why the script refuses
to run there.

```bash
srun --partition=gpu --gpus-per-node=rtx4090:1 --cpus-per-task=8 --mem=40G --time=04:00:00 --pty bash
cd ~/vdoc-agent
make setup          # installs both envs, kernel, lock files
make serve-check
make train-check
```

## Use the environments

```bash
source ~/vdoc-agent/envs/serve/bin/activate    # vLLM
source ~/vdoc-agent/envs/train/bin/activate    # training stack
deactivate
```

The Makefile calls each env's binaries by path, so no activation is needed for it, and the
auto-activated Anaconda `base` does not interfere.

## Run the checks

| Command | What it does |
|---|---|
| `make serve-check` | starts `vllm serve Qwen/Qwen3-0.6B --max-model-len 4096 --gpu-memory-utilization 0.85`, waits for `/health`, sends one streamed chat completion and one JSON-schema request, prints time to first token and tokens/s, stops the server |
| `make train-check` | torch (device, capability, bf16, TF32, 8192x8192 bf16 matmul in TFLOPS), transformers (config + forward pass), peft (LoRA r=16 on q/k/v/o), bitsandbytes (4-bit NF4 footprint), trl (SFTTrainer, 10 steps on 50 examples, loss and peak memory) |
| `make versions` | driver, CUDA and package versions per env |
| `make serve MODEL=<hf-id>` | `vllm serve` with defaults; override `PORT`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` |
| `make lock` | rewrite both lock files |

GPU targets stop with a message when no GPU is visible. Run them inside a GPU job, or add
`SLURM=1` on the login node to wrap the target in a one-off `srun` (flags in `SRUN_FLAGS`):

```bash
make serve-check SLURM=1
```

Only one GPU job at a time: `serve-check` reserves 85% of the card, so do not run it alongside
`train-check` in the same allocation.

## Problems hit and fixes

| Problem | Fix |
|---|---|
| The interactive GPU job (17037754, gnode22) was killed after 10 minutes, so there was no GPU to install or test on | Package install and both checks are deferred to `make setup` on a GPU node |
| Login node has no `nvidia-smi`, so `--torch-backend=auto` would select CPU wheels | `scripts/setup_envs.sh` and the GPU make targets refuse to run without a visible GPU |
| GPU nodes are shared by several users, so port 8000 may be taken | `check_serve.sh` asks the OS for a free port and binds to 127.0.0.1; `make serve` keeps 8000 and takes `PORT=` |
| Scratch is the largest disk but is purged | Caches kept on home GPFS (see above) |
| Repo was cloned to `~/vdoc-agent`, the plan said `~/projects/vdoc-agent` | Everything lives in `~/vdoc-agent`. The envs are not relocatable: after moving the repo, delete `envs/serve` and `envs/train` (keep the lock files) and run `make setup` again |
