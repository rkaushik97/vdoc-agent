#!/usr/bin/env bash
# Smoke test for envs/serve: start vLLM, wait for /health, run check_serve.py
# against it, then shut the server down cleanly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/envs/serve"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HOST=127.0.0.1
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"   # first run also downloads the model
LOG="${LOG:-$ROOT/logs/check_serve.log}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# FlashInfer's sampler needs nvcc at startup, and no CUDA toolkit here matches vLLM's build.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Keep localhost traffic off any site proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

[ -x "$VENV/bin/vllm" ] || { echo "envs/serve is not installed: run scripts/setup_envs.sh serve" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "No GPU visible on $(hostname): run inside a GPU job" >&2; exit 1; }

# GPU nodes are shared, so 8000 may belong to someone else: take a free port.
PORT="${PORT:-$("$VENV/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')}"
mkdir -p "$(dirname "$LOG")"

echo "[serve-check] starting vllm serve $MODEL on $HOST:$PORT (log: $LOG)"
# setsid gives the server its own process group, so cleanup reaches its workers.
setsid "$VENV/bin/vllm" serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    >"$LOG" 2>&1 &
SERVER_PID=$!

stop_server() {
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$SERVER_PID" 2>/dev/null || return 0
        sleep 0.5
    done
    echo "[serve-check] server ignored SIGINT for 30s, killing its process group" >&2
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    return 1
}
trap 'stop_server || true' EXIT

t0=$SECONDS
until curl -sf --noproxy '*' -o /dev/null "http://$HOST:$PORT/health"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[serve-check] FAIL: server exited during startup. Last log lines:" >&2
        tail -n 40 "$LOG" >&2
        exit 1
    fi
    if [ $((SECONDS - t0)) -ge "$STARTUP_TIMEOUT" ]; then
        echo "[serve-check] FAIL: /health not ready after ${STARTUP_TIMEOUT}s. Last log lines:" >&2
        tail -n 40 "$LOG" >&2
        exit 1
    fi
    sleep 2
done
echo "[serve-check] /health OK after $((SECONDS - t0))s"

"$VENV/bin/python" "$ROOT/scripts/check_serve.py" --base-url "http://$HOST:$PORT/v1" --model "$MODEL"

trap - EXIT
if stop_server; then
    wait "$SERVER_PID" 2>/dev/null || true
    echo "[serve-check] server shut down cleanly"
else
    echo "[serve-check] FAIL: server needed SIGKILL" >&2
    exit 1
fi
echo "[serve-check] PASS"
