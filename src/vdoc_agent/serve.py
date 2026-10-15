"""Start and stop a vLLM server (from envs/serve) from Python, for notebooks and scripts.

    server = VLLMServer("Qwen/Qwen3-VL-4B-Instruct", max_model_len=16384).start()
    client = OpenAI(base_url=server.base_url, api_key="none")
    ...
    server.stop()

or as a context manager: `with VLLMServer(model) as server: ...`. Only one model fits on the GPU
at a time, so stop one server before starting the next. The log goes to logs/vllm-<model>.log.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM = REPO_ROOT / "envs" / "serve" / "bin" / "vllm"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VLLMServer:
    def __init__(
        self,
        model: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        extra_args: tuple[str, ...] = (),
        startup_timeout: int = 1200,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = extra_args
        self.startup_timeout = startup_timeout
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.log_path = REPO_ROOT / "logs" / f"vllm-{model.split('/')[-1]}.log"
        self.proc: subprocess.Popen | None = None

    def start(self) -> VLLMServer:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            # FlashInfer's sampler JIT-compiles with nvcc, which the GPU nodes lack
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        cmd = [
            str(VLLM), "serve", self.model,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            *self.extra_args,
        ]  # fmt: skip
        with open(self.log_path, "w") as log:
            # own process group, so stop() reaches vLLM's worker processes too
            self.proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.time() + self.startup_timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup, see {self.log_path}")
            try:
                no_proxy.open(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return self
            except OSError:
                pass
            if time.time() > deadline:
                self.stop()
                raise TimeoutError(f"vLLM not ready in time, see {self.log_path}")
            time.sleep(2)

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGINT)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        time.sleep(5)  # let the GPU memory be released before the next server starts

    def __enter__(self) -> VLLMServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
