#!/usr/bin/env bash
# Toy overnight run inside the GPU job, one step after the other on the same GPU:
#   1. LoRA SFT of Qwen3-VL-4B on ViDoRe V3 computer science       (scripts/toy/sft_train.py)
#   2. answer every pharma dev question with the base model and the adapter (scripts/toy/answer_dev.py)
#   3. grade both answer files with the 8B judge                        (scripts/toy/judge.py)
#
#   srun --jobid=<id> --overlap bash scripts/toy/overnight.sh             # full run
#   SMOKE=1 srun --jobid=<id> --overlap bash scripts/toy/overnight.sh     # 2 training steps, 5 questions
#
# Everything lands in data/runs/toy-<time>/; summary.txt holds the side-by-side table.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"

PY=envs/train/bin/python
VLLM=envs/serve/bin/vllm
BASE=Qwen/Qwen3-VL-4B-Instruct
JUDGE=Qwen/Qwen3-VL-8B-Instruct-FP8
OUT="data/runs/toy-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"

if [ "${SMOKE:-0}" = 1 ]; then
    ADAPTER=checkpoints/toy-sft-smoke
    TRAIN_ARGS=(--max-steps 2 --out "$ADAPTER")
    LIMIT=(--limit 5)
else
    ADAPTER=checkpoints/toy-sft-cs
    TRAIN_ARGS=(--out "$ADAPTER")
    LIMIT=()
fi

PORT="$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
URL="http://127.0.0.1:$PORT/v1"
SERVER=""

start_server() {  # start_server <log name> <vllm serve args...>; returns once /health answers
    local name=$1; shift
    setsid "$VLLM" serve "$@" --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization 0.85 >"$OUT/$name.log" 2>&1 &
    SERVER=$!
    for _ in $(seq 1 600); do
        curl -sf --noproxy '*' -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
        kill -0 "$SERVER" 2>/dev/null || { echo "server $name died, see $OUT/$name.log" >&2; tail -n 30 "$OUT/$name.log" >&2; return 1; }
        sleep 2
    done
    echo "server $name not ready after 20 min" >&2
    return 1
}

stop_server() {  # stop the whole process group, then wait so the GPU memory is free again
    [ -n "$SERVER" ] && kill -0 "$SERVER" 2>/dev/null || return 0
    kill -INT -- "-$SERVER" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER" 2>/dev/null || true
    SERVER=""
    sleep 5
}
trap stop_server EXIT

echo "[$(date +%T)] 1/3 train -> $ADAPTER"
$PY scripts/toy/sft_train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$OUT/train.log"

echo "[$(date +%T)] 2/3 answer dev: base and adapter"
start_server serve-4b "$BASE" --max-model-len 16384 --enable-lora --lora-modules "toy=$ADAPTER" --max-lora-rank 16
$PY scripts/toy/answer_dev.py --model "$BASE" --out "$OUT/answers-base.jsonl" --base-url "$URL" "${LIMIT[@]}"
$PY scripts/toy/answer_dev.py --model toy --out "$OUT/answers-toy.jsonl" --base-url "$URL" "${LIMIT[@]}"
stop_server

echo "[$(date +%T)] 3/3 judge both with $JUDGE"
start_server serve-judge "$JUDGE" --max-model-len 4096
$PY scripts/toy/judge.py "$OUT/answers-base.jsonl" "$OUT/answers-toy.jsonl" --judge "$JUDGE" --base-url "$URL" \
    | tee "$OUT/summary.txt"
stop_server

echo "[$(date +%T)] done: $OUT/summary.txt"
