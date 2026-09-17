from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.cache.fault_policy import CANONICAL_FAULT_NAMES
from schemaguard.runner.faults import run_fault_injection_suite


def test_required_fault_injection_matrix_is_executed_and_sanitized() -> None:
    evidence = run_fault_injection_suite(source_commit="a" * 40)
    assert evidence.expected_fault_count == 30
    assert tuple(record.fault_name for record in evidence.records) == CANONICAL_FAULT_NAMES
    assert all(record.status == "PASS" for record in evidence.records)
    assert all(record.lock_released is True for record in evidence.records)
    assert all(
        record.expected_artifact_accepted == record.artifact_accepted
        and record.expected_resume_behavior == record.resume_behavior
        and record.expected_failure_category == record.observed_failure_category
        for record in evidence.records
    )
    assert evidence.offline_network_attempt_count == 1
    assert {record.evidence_sha256 for record in evidence.records}
    with pytest.raises(ValidationError):
        type(evidence).model_validate(
            {**evidence.model_dump(mode="json"), "records": evidence.records[:-1]}
        )
