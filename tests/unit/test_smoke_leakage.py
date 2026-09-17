from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from schemaguard.experiments.execution import (
    _reported_failure_category,
    _training_targets,
    _worker_error_text,
)
from schemaguard.models.adapters.contracts import AdapterFailure, FailureCategory


def test_model_fit_target_read_is_exactly_the_training_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = tmp_path / "data/processed/openml/1464/targets.parquet"
    target_path.parent.mkdir(parents=True)
    targets = pd.DataFrame(
        {
            "__sg_row_id": ["train-a", "train-b", "cal-a", "test-a"],
            "target_code": [0, 1, 0, 1],
        }
    )
    targets.to_parquet(target_path, index=False)
    assignment = pd.DataFrame(
        {
            "__sg_row_id": ["train-a", "train-b", "cal-a", "test-a"],
            "partition": ["train", "train", "calibration", "test"],
        }
    )

    original_read = pd.read_parquet
    observed_filters: list[Any] = []

    def guarded_read(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if Path(path) == target_path:
            observed_filters.append(kwargs.get("filters"))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    target_codes, training_targets = _training_targets(tmp_path, ["train-a", "train-b"], assignment)
    assert target_codes.tolist() == [0, 1]
    assert training_targets["__sg_row_id"].tolist() == ["train-a", "train-b"]
    assert observed_filters == [[("__sg_row_id", "in", ["train-a", "train-b"])]]

    observed_filters.clear()
    for forbidden_ids in (["cal-a"], ["test-a"], ["train-a", "test-a"]):
        with pytest.raises(ValueError, match="training rows only"):
            _training_targets(tmp_path, list(forbidden_ids), assignment)
    assert observed_filters == []


def test_assignment_order_mutation_cannot_expand_training_label_access(tmp_path: Path) -> None:
    target_path = tmp_path / "data/processed/openml/1464/targets.parquet"
    target_path.parent.mkdir(parents=True)
    pd.DataFrame({"__sg_row_id": ["train-a", "train-b"], "target_code": [0, 1]}).to_parquet(
        target_path, index=False
    )
    assignment = pd.DataFrame(
        {
            "__sg_row_id": ["train-b", "train-a"],
            "partition": ["train", "train"],
        }
    )
    with pytest.raises(ValueError, match="training rows only"):
        _training_targets(tmp_path, ["train-a", "train-b"], assignment)


def test_offline_network_guard_blocks_and_counts_before_socket_activity() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = """
from schemaguard.experiments.execution import _install_offline_network_guard
import socket
attempts = _install_offline_network_guard()
try:
    socket.create_connection(("127.0.0.1", 1), timeout=0.01)
except PermissionError:
    pass
else:
    raise AssertionError("offline guard allowed a socket connection")
assert attempts[0] >= 1
print(attempts[0])
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    probe = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert int(probe.stdout.strip()) >= 1


def test_worker_error_keeps_blocked_adapter_category() -> None:
    failure = AdapterFailure(FailureCategory.BLOCKED_CHECKPOINT, "checkpoint unavailable")
    error = _worker_error_text(failure)
    assert error.startswith("BLOCKED_CHECKPOINT: AdapterFailure:")
    assert _reported_failure_category("FAIL_MODEL_RUNTIME", error) == "BLOCKED_CHECKPOINT"
    assert _reported_failure_category("FAIL_TIMEOUT", "worker timed out") == "FAIL_TIMEOUT"
