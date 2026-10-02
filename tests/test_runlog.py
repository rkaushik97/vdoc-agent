"""Tests for vdoc_agent.runlog.log_run."""

import json
import re
from pathlib import Path

from vdoc_agent import runlog

RECORD_KEYS = {"name", "timestamp", "git", "host", "gpu", "versions", "config", "metrics"}


def test_writes_record_with_config_and_metrics(tmp_path):
    path = runlog.log_run("smoke test", {"lr": 1e-4}, {"anls": 0.5}, runs_dir=tmp_path)

    assert path.parent == tmp_path
    assert re.fullmatch(r"\d{8}T\d{6}Z_smoke-test\.json", path.name)

    record = json.loads(path.read_text())
    assert set(record) == RECORD_KEYS
    assert record["name"] == "smoke test"
    assert record["config"] == {"lr": 1e-4}
    assert record["metrics"] == {"anls": 0.5}
    assert record["timestamp"].endswith("+00:00")
    assert record["host"]
    assert record["versions"]["python"]
    assert "torch" in record["versions"] and "transformers" in record["versions"]


def test_records_git_commit_and_dirty_flag(tmp_path):
    record = json.loads(runlog.log_run("git", {}, {}, runs_dir=tmp_path).read_text())
    commit, dirty = record["git"]["commit"], record["git"]["dirty"]

    assert (commit is None) == (dirty is None)
    if commit is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
        assert isinstance(dirty, bool)


def test_creates_runs_dir_and_never_overwrites(tmp_path):
    runs_dir = tmp_path / "nested" / "runs"
    first = runlog.log_run("same", {}, {}, runs_dir=runs_dir)
    second = runlog.log_run("same", {}, {}, runs_dir=runs_dir)

    assert first.exists() and second.exists()
    assert first != second


def test_non_json_values_are_stringified(tmp_path):
    path = runlog.log_run("paths", {"out": Path("checkpoints/x")}, {}, runs_dir=tmp_path)
    assert json.loads(path.read_text())["config"] == {"out": "checkpoints/x"}
