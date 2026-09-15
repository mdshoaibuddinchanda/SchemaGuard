from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.artifact_contracts import GpuCapacityReportContract

HASH = "a" * 64


def _row(
    model_id: str,
    *,
    device: str = "cuda",
    status: str = "PASS",
    strategy: str = "repeated_inference_single_worker",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": 2,
        "model_id": model_id,
        "profile_id": "minimum_rows",
        "device": device,
        "strategy": strategy,
        "seed": 1729,
        "cycles": 2,
        "status": status,
        "process_exit": "success",
        "failure_category": "PASS" if status == "PASS" else status,
        "start_time": "2026-09-15T00:00:00Z",
        "end_time": "2026-09-15T00:01:00Z",
        "runtime_seconds": 1.0,
        "prediction_shape": [4, 2],
        "class_order": [0, 1],
        "monitoring_complete": True,
        "peak_ram_mib": 100.0,
        "peak_vram_reserved_mib": 100.0 if device == "cuda" else None,
    }
    row.update(overrides)
    return row


def _report(*rows: dict[str, object], requested: int | None = None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "stage": "gpu_capacity",
        "status": "PASS",
        "gpu_state": {
            "available": True,
            "name": "synthetic",
            "total_mib": 4096.0,
            "free_mib": 3000.0,
        },
        "gpu_headroom_mib": 512.0,
        "gpu_soft_limit_mib": 3600.0,
        "foundation_models_serialized": True,
        "profiles": [
            {
                "profile_id": "minimum_rows",
                "source_dataset_id": 1,
                "rows": 100,
                "predictors": 4,
                "classes": 2,
                "train_rows": 60,
                "test_rows": 20,
            }
        ],
        "checkpoint_sha256": {"TPFN3-8.5": HASH, "TICL2-2.2": HASH},
        "registry_sha256": HASH,
        "runtime_config_sha256": HASH,
        "profile_count": 1,
        "rows": list(rows),
        "optimization": [],
        "fallback_policy": "controlled synthetic policy",
        "requested_profiles": len(rows) if requested is None else requested,
        "accounted_profiles": len(rows),
        "monitoring_complete": True,
    }


def test_fully_valid_gpu_report_passes() -> None:
    report = GpuCapacityReportContract.model_validate(_report(_row("TPFN3-8.5"), _row("TICL2-2.2")))
    assert report.status == "PASS"


@pytest.mark.parametrize(
    "report",
    [
        _report(_row("TPFN3-8.5")),
        _report(_row("TPFN3-8.5"), _row("TICL2-2.2"), requested=3),
        _report(_row("TPFN3-8.5"), _row("TICL2-2.2", monitoring_complete=False)),
        _report(_row("TPFN3-8.5", peak_vram_reserved_mib=3601.0), _row("TICL2-2.2")),
        _report(
            _row("TPFN3-8.5", status="FAIL", process_exit="exception"),
            _row("TICL2-2.2"),
        ),
        _report(
            _row("TPFN3-8.5", status="FAIL", process_exit="timeout", timed_out=True),
            _row("TICL2-2.2"),
        ),
        _report(
            _row("TPFN3-8.5", status="FAIL"),
            _row("TICL2-2.2"),
        ),
    ],
)
def test_inconsistent_passing_reports_are_rejected(report: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GpuCapacityReportContract.model_validate(report)


def test_zero_requested_profiles_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GpuCapacityReportContract.model_validate(
            _report(_row("TPFN3-8.5"), _row("TICL2-2.2"), requested=0)
        )


def test_cpu_fallback_is_explicitly_classified() -> None:
    report = _report(
        _row(
            "TPFN3-8.5",
            device="cpu",
            status="PASS_WITH_CPU_FALLBACK",
            strategy="cpu_fallback",
            fallback_status="CUDA unavailable; CPU passed",
            failure_category="PASS_WITH_CPU_FALLBACK",
        ),
        _row("TICL2-2.2"),
    )
    assert GpuCapacityReportContract.model_validate(report).rows[0].fallback_status


def test_missing_fallback_classification_is_rejected() -> None:
    report = _report(
        _row(
            "TPFN3-8.5",
            device="cpu",
            status="PASS_WITH_CPU_FALLBACK",
            strategy="cpu_fallback",
            fallback_status=None,
        ),
        _row("TICL2-2.2"),
    )
    with pytest.raises(ValidationError):
        GpuCapacityReportContract.model_validate(report)
