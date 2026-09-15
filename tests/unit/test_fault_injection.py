from __future__ import annotations

import pandas as pd
import pytest

from schemaguard.compatibility.checkpoint_cache import (
    CacheError,
    atomic_json_write,
    read_checkpoint_metadata,
    resolve_offline_checkpoint,
)
from schemaguard.models.registry import ModelSpec, RegistryError, validate_constructor_parameters
from schemaguard.utils.process_lock import ProcessLock, Timeout


def test_truncated_json_and_parquet_results_are_rejected(tmp_path) -> None:
    manifest = tmp_path / "truncated.json"
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(CacheError):
        read_checkpoint_metadata(manifest)
    parquet = tmp_path / "truncated.parquet"
    parquet.write_bytes(b"PAR1")
    with pytest.raises(Exception):
        pd.read_parquet(parquet)


def test_interrupted_atomic_write_preserves_previous_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "result.json"
    destination.write_text("old", encoding="utf-8")

    def interrupt(*args, **kwargs):
        raise OSError("injected interruption")

    monkeypatch.setattr("schemaguard.compatibility.checkpoint_cache.os.replace", interrupt)
    with pytest.raises(OSError):
        atomic_json_write(destination, {"new": True})
    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_stale_lock_is_recovered_and_active_lock_times_out(tmp_path) -> None:
    lock_path = tmp_path / "artifact.lock"
    lock_path.write_text("stale", encoding="utf-8")
    with ProcessLock(lock_path):
        assert lock_path.exists()
        with pytest.raises(Timeout):
            with ProcessLock(lock_path, timeout=0.05):
                pass


def test_unsupported_constructor_parameter_is_blocked() -> None:
    model = ModelSpec(
        id="LR-1.9",
        class_path="sklearn.linear_model.LogisticRegression",
        package="scikit-learn",
        expected_version="1.9.1",
        checkpoint=None,
        parameters={},
    )
    with pytest.raises(RegistryError):
        validate_constructor_parameters(model, {"unsupported_parameter": True})


def test_offline_cache_miss_is_rejected_without_network(tmp_path) -> None:
    with pytest.raises(CacheError):
        resolve_offline_checkpoint(tmp_path, "missing")
