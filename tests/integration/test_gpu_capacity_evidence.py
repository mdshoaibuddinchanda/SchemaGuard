"""Read-only evidence checks for GPU capacity profiling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemaguard.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.integration, pytest.mark.evidence]


def test_gpu_report_is_serial_and_within_soft_limit() -> None:
    report = json.loads(
        (ROOT / "results/validation/gpu_capacity_report.json").read_text(encoding="utf-8")
    )
    assert report["foundation_models_serialized"] is True
    assert report["stage"] == "gpu_capacity"
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
