"""Run logging: one JSON record per run under reports/runs/.

``log_run(name, config, metrics)`` writes ``reports/runs/<UTC timestamp>_<name>.json`` with the
config and metrics plus what is needed to trace the number back: git commit and dirty flag,
hostname, GPU name, and the Python / torch / transformers versions.
"""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "reports" / "runs"


def log_run(
    name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    runs_dir: Path | None = None,
) -> Path:
    """Write one run record and return its path.

    The file is ``<runs_dir>/<YYYYMMDDTHHMMSSZ>_<name>.json``; ``runs_dir`` defaults to
    ``reports/runs`` in the repo root. ``name`` is reduced to ``[A-Za-z0-9._-]`` for the file
    name and kept verbatim inside the record. Values that are not JSON types are stored as
    ``str(value)``.
    """
    now = datetime.now(UTC)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = _unique(runs_dir / f"{now:%Y%m%dT%H%M%SZ}_{slug}.json")

    record = {
        "name": name,
        "timestamp": now.isoformat(timespec="seconds"),
        "git": {"commit": _git_commit(), "dirty": _git_dirty()},
        "host": socket.gethostname(),
        "gpu": _gpu_name(),
        "versions": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "config": config,
        "metrics": metrics,
    }
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def _unique(path: Path) -> Path:
    """Avoid clobbering a record written in the same second with the same name."""
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip()


def _git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    """True if anything is modified, staged or untracked; None outside a git checkout."""
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # torch missing, or installed without a working CUDA runtime
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _package_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None
