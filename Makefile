SHELL := /bin/bash
.DEFAULT_GOAL := help

SERVE := envs/serve
TRAIN := envs/train

MODEL         ?= Qwen/Qwen3-0.6B
HOST          ?= 127.0.0.1
PORT          ?= 8000
MAX_MODEL_LEN ?= 4096
GPU_MEM_UTIL  ?= 0.85

export HF_HOME      ?= $(HOME)/.cache/huggingface
export UV_CACHE_DIR ?= $(HOME)/.cache/uv
export PATH         := $(HOME)/.local/bin:$(PATH)

# GPU targets must run on a GPU node. From the login node either open a job
# first, or pass SLURM=1 to wrap the target in a one-off srun allocation.
SRUN_FLAGS ?= --partition=gpu --gpus-per-node=rtx4090:1 --cpus-per-task=8 --mem=40G --time=01:00:00
ifeq ($(SLURM),1)
GPU_RUN := srun $(SRUN_FLAGS)
else
GPU_RUN :=
endif

.PHONY: help need-gpu setup lock serve-check train-check versions serve test lint corpus eval-retrieval eval-smoke eval train-sft train-dpo train-grpo demo

help:
	@echo "make setup          build envs/serve and envs/train (GPU node)"
	@echo "make serve-check    vLLM smoke test: chat + JSON-schema request"
	@echo "make train-check    torch / transformers / peft / bitsandbytes / trl checks"
	@echo "make versions       driver, CUDA and package versions per env"
	@echo "make serve MODEL=<hf-id> [PORT=8000 MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.85]"
	@echo "make lock           refresh envs/*/requirements.lock"
	@echo "make test           pytest (envs/train)"
	@echo "make lint           ruff check + ruff format --check (envs/train)"
	@echo "make corpus | eval-retrieval | eval-smoke | eval | train-sft | train-dpo | train-grpo | demo"
	@echo "                    not written yet: each prints the day it is due"
	@echo "Add SLURM=1 to run a GPU target through srun from the login node."

need-gpu:
ifneq ($(SLURM),1)
	@command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || { \
		echo "No GPU visible on $$(hostname). Run inside a GPU job, or use: make $(MAKECMDGOALS) SLURM=1"; exit 1; }
endif

setup: need-gpu
	$(GPU_RUN) bash scripts/setup_envs.sh

lock:
	uv pip freeze --python $(SERVE)/bin/python > $(SERVE)/requirements.lock
	uv pip freeze --python $(TRAIN)/bin/python > $(TRAIN)/requirements.lock

serve-check: need-gpu
	$(GPU_RUN) bash scripts/check_serve.sh

train-check: need-gpu
	$(GPU_RUN) $(TRAIN)/bin/python scripts/check_train.py

versions:
	@if command -v nvidia-smi >/dev/null; then \
		nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; \
		nvidia-smi | grep -o "CUDA Version: [0-9.]*"; \
	else echo "driver: n/a (no GPU on $$(hostname))"; fi
	@echo "uv: $$(uv --version)"
	@$(SERVE)/bin/python scripts/versions.py serve torch vllm openai transformers
	@$(TRAIN)/bin/python scripts/versions.py train torch transformers trl peft bitsandbytes accelerate datasets tokenizers flash-attn

serve: need-gpu
	$(GPU_RUN) $(SERVE)/bin/vllm serve $(MODEL) --host $(HOST) --port $(PORT) \
		--max-model-len $(MAX_MODEL_LEN) --gpu-memory-utilization $(GPU_MEM_UTIL)

test:
	$(TRAIN)/bin/python -m pytest

lint:
	$(TRAIN)/bin/ruff check .
	$(TRAIN)/bin/ruff format --check .

# Placeholders until the matching day of the plan.
corpus:
	@echo "coming on day 5"

eval-retrieval:
	@echo "coming on day 6"

eval-smoke:
	@echo "coming on day 13"

eval:
	@echo "coming on day 13"

train-sft:
	@echo "coming on day 21"

train-dpo:
	@echo "coming on day 24"

train-grpo:
	@echo "coming on day 27"

demo:
	@echo "coming on day 29"
