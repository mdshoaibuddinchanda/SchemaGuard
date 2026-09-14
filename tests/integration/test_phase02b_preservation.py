"""Read-only integration checks for Phase 02B generated evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from schemaguard.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


def test_all_fourteen_dataset_validation_records_pass() -> None:
    inventory = json.loads(
        (ROOT / "results/validation/schemaorbit14_inventory.json").read_text(encoding="utf-8")
    )
    table = pd.read_parquet(ROOT / "results/validation/dataset_validation.parquet")
    assert inventory["dataset_count"] == 14
    assert len(inventory["datasets"]) == 14
    assert len(table) == 14
    assert set(table["status"]) == {"PASS"}
    splice = next(item for item in inventory["datasets"] if item["openml_data_id"] == 46)
    assert splice["ignored_attributes"] == ["Instance_name"]
    assert splice["observed_predictors"] == 60


def test_phase01_hash_comparison_is_unchanged() -> None:
    comparison = json.loads(
        (ROOT / "artifacts/phase_02b_dataset_gpu/review/phase01_hash_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["unchanged"] is True
    before = json.loads(
        (ROOT / "artifacts/phase_02b_dataset_gpu/review/phase01_hashes_before.json").read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        (ROOT / "artifacts/phase_02b_dataset_gpu/review/phase01_hashes_after.json").read_text(
            encoding="utf-8"
        )
    )
    assert before == after


def test_gpu_report_is_serial_and_within_soft_limit() -> None:
    report = json.loads(
        (ROOT / "results/validation/gpu_capacity_report.json").read_text(encoding="utf-8")
    )
    assert report["foundation_models_serialized"] is True
    assert report["status"] == "PASS"
    assert {item["model_id"] for item in report["rows"]} == {"TPFN3-8.5", "TICL2-2.2"}
    successful = [
        item
        for item in report["rows"]
        if item.get("status") == "PASS" and item.get("device") == "cuda"
    ]
    assert successful
    assert max(item["peak_vram_reserved_mib"] for item in successful) < report["gpu_soft_limit_mib"]
    assert all(
        item["repeat_max_abs_diff"] <= 1.0e-5
        for item in successful
        if item["repeat_max_abs_diff"] is not None
    )
    assert all(item["post_cleanup_gpu_allocated_mib"] is not None for item in successful)


def test_frozen_checkpoint_hashes_are_recorded() -> None:
    report = json.loads(
        (ROOT / "results/validation/gpu_capacity_report.json").read_text(encoding="utf-8")
    )
    for checkpoint in (
        "tabpfn-v3-classifier-v3_default.ckpt",
        "tabicl-classifier-v2-20260212.ckpt",
    ):
        assert sha256_file(ROOT / checkpoint) in report["checkpoint_sha256"].values()
