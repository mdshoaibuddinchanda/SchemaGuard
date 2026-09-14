"""Read-only evidence checks for the dataset registry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.integration, pytest.mark.evidence]


def test_all_fourteen_dataset_validation_records_pass() -> None:
    inventory = json.loads(
        (ROOT / "results/validation/dataset_registry_report.json").read_text(encoding="utf-8")
    )
    table = pd.read_parquet(ROOT / "results/validation/dataset_registry_validation.parquet")
    assert inventory["dataset_count"] == 14
    assert len(inventory["datasets"]) == 14
    assert len(table) == 14
    assert set(table["status"]) == {"PASS"}
    splice = next(item for item in inventory["datasets"] if item["openml_data_id"] == 46)
    assert splice["ignored_attributes"] == ["Instance_name"]
    assert splice["observed_predictors"] == 60


def test_data_foundation_hash_comparison_is_unchanged() -> None:
    comparison = json.loads(
        (ROOT / "artifacts/data_foundation/review/data_foundation_hash_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["unchanged"] is True
    before = json.loads(
        (ROOT / "artifacts/data_foundation/review/data_foundation_hashes_before.json").read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        (ROOT / "artifacts/data_foundation/review/data_foundation_hashes_after.json").read_text(
            encoding="utf-8"
        )
    )
    assert before == after
