from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from schemaguard.runner.contracts import ResourceRecord
from schemaguard.runner.resources import classify_resource_limits


def test_resource_limit_classifier_fails_closed_at_measured_caps() -> None:
    assert (
        classify_resource_limits(
            process_tree_rss_mib=28673,
            gpu_reserved_mib=None,
            ram_hard_limit_mib=28672,
            vram_soft_limit_mib=3600,
        )
        == "FAIL_RESOURCE_LIMIT"
    )
    assert (
        classify_resource_limits(
            process_tree_rss_mib=None,
            gpu_reserved_mib=3601,
            ram_hard_limit_mib=28672,
            vram_soft_limit_mib=3600,
        )
        == "FAIL_RESOURCE_LIMIT"
    )
    assert (
        classify_resource_limits(
            process_tree_rss_mib=None,
            gpu_reserved_mib=None,
            ram_hard_limit_mib=28672,
            vram_soft_limit_mib=3600,
        )
        is None
    )


def test_missing_gpu_telemetry_remains_null_and_incomplete() -> None:
    now = datetime.now(UTC)
    record = ResourceRecord(
        started_at=now,
        ended_at=now,
        wall_seconds=0,
        cpu_seconds=None,
        worker_pid=None,
        exit_code=None,
        peak_worker_rss_mib=None,
        peak_process_tree_rss_mib=None,
        gpu_allocated_mib=None,
        gpu_reserved_mib=None,
        gpu_free_before_mib=None,
        timed_out=False,
        soft_limit_warning=False,
        hard_limit_passed=True,
        cleanup_passed=True,
        telemetry_complete=False,
    )
    assert record.gpu_reserved_mib is None


def test_complete_telemetry_cannot_fabricate_missing_peak_ram() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="peak RSS"):
        ResourceRecord(
            started_at=now,
            ended_at=now,
            wall_seconds=0,
            cpu_seconds=0,
            worker_pid=123,
            exit_code=0,
            peak_worker_rss_mib=None,
            peak_process_tree_rss_mib=None,
            gpu_allocated_mib=None,
            gpu_reserved_mib=None,
            gpu_free_before_mib=None,
            timed_out=False,
            soft_limit_warning=False,
            hard_limit_passed=True,
            cleanup_passed=True,
            telemetry_complete=True,
        )
