"""Unit tests for the bounded GPU profiler without loading foundation models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from schemaguard.compatibility.gpu_profiles import (
    _fixture,
    _is_oom,
    _profiles,
)

ROOT = Path(__file__).resolve().parents[2]


def test_profile_selection_is_deterministic_and_uses_metadata(tmp_path: Path) -> None:
    inventory = {
        "datasets": [
            {"openml_data_id": 1, "observed_rows": 500, "observed_predictors": 4, "class_count": 2},
            {
                "openml_data_id": 2,
                "observed_rows": 2000,
                "observed_predictors": 20,
                "class_count": 3,
            },
            {
                "openml_data_id": 3,
                "observed_rows": 4000,
                "observed_predictors": 40,
                "class_count": 5,
            },
        ]
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    first = _profiles(path)
    second = _profiles(path)
    assert first == second
    assert {item["source_dataset_id"] for item in first} <= {1, 2, 3}
    assert all(item["train_rows"] >= 20 and item["test_rows"] >= 10 for item in first)


def test_synthetic_profile_fixture_is_stable() -> None:
    profile = {
        "profile_id": "test",
        "source_dataset_id": 2,
        "rows": 100,
        "predictors": 4,
        "classes": 3,
        "train_rows": 60,
        "test_rows": 20,
    }
    first = _fixture(profile, 1729)
    second = _fixture(profile, 1729)
    assert first[3] == second[3]
    assert first[0].equals(second[0])
    assert np.array_equal(first[2], second[2])


def test_oom_classifier_distinguishes_memory_failures() -> None:
    assert _is_oom(MemoryError("out of memory"))
    assert _is_oom(RuntimeError("CUDA out of memory"))
    assert not _is_oom(RuntimeError("constructor failed"))
