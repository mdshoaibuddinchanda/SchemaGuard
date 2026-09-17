"""Single frozen policy for cache/scheduler fault-injection evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ..utils.hashing import sha256_canonical_json


@dataclass(frozen=True, slots=True)
class FaultExpectation:
    injection_point: str
    expected_failure_category: str
    expected_artifact_accepted: bool
    expected_resume_behavior: str


FAULT_POLICY: Final[Mapping[str, FaultExpectation]] = MappingProxyType(
    {
        "payload_write_interrupted": FaultExpectation(
            "process stopped during payload write",
            "CACHE_MISS",
            False,
            "RECOMPUTED_AFTER_INTERRUPTED_TRANSACTION",
        ),
        "completion_interrupted": FaultExpectation(
            "payload durable before completion marker",
            "CACHE_MISS",
            False,
            "RECOMPUTED_AFTER_INTERRUPTED_TRANSACTION",
        ),
        "manifest_truncated": FaultExpectation(
            "manifest parser", "FAIL_CACHE_INTEGRITY", False, "RECOMPUTED_AFTER_QUARANTINE"
        ),
        "payload_truncated": FaultExpectation(
            "payload checksum validation",
            "FAIL_CACHE_INTEGRITY",
            False,
            "RECOMPUTED_AFTER_QUARANTINE",
        ),
        "completion_marker_missing": FaultExpectation(
            "completion marker validation",
            "FAIL_CACHE_INTEGRITY",
            False,
            "RECOMPUTED_AFTER_QUARANTINE",
        ),
        "payload_checksum_wrong": FaultExpectation(
            "payload checksum validation",
            "FAIL_CACHE_INTEGRITY",
            False,
            "RECOMPUTED_AFTER_QUARANTINE",
        ),
        "manifest_identity_wrong": FaultExpectation(
            "canonical identity validation",
            "FAIL_CACHE_INTEGRITY",
            False,
            "RECOMPUTED_AFTER_QUARANTINE",
        ),
        "cache_key_wrong": FaultExpectation(
            "key recomputation", "FAIL_CACHE_INTEGRITY", False, "RECOMPUTED_AFTER_QUARANTINE"
        ),
        "artifact_schema_wrong": FaultExpectation(
            "strict manifest schema validation",
            "FAIL_CACHE_INTEGRITY",
            False,
            "RECOMPUTED_AFTER_QUARANTINE",
        ),
        "failed_artifact_candidate": FaultExpectation(
            "failed-state cache candidate",
            "FAIL_CACHE_INTEGRITY",
            False,
            "RECOMPUTED_AFTER_QUARANTINE",
        ),
        "model_configuration_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "implementation_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "dependency_lock_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "dataset_checksum_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "split_checksum_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "view_certificate_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "checkpoint_checksum_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "device_policy_change": FaultExpectation(
            "change one scientific or implementation identity field",
            "CACHE_MISS",
            False,
            "NEW_IDENTITY_REQUIRES_COMPUTE",
        ),
        "two_process_same_identity_publication": FaultExpectation(
            "two spawned writers publish identical bytes simultaneously",
            "ONE_VALIDATED_ARTIFACT",
            True,
            "WINNING_ARTIFACT_REUSED",
        ),
        "same_key_different_payload": FaultExpectation(
            "publish different bytes for an already validated identity",
            "FAIL_CACHE_INTEGRITY",
            True,
            "ORIGINAL_IMMUTABLE_ENTRY_RETAINED",
        ),
        "lock_holder_process_crash": FaultExpectation(
            "owner exits without executing lock-release cleanup",
            "PASS",
            False,
            "LOCK_REACQUIRED_AFTER_OWNER_EXIT",
        ),
        "stale_lock_file_without_owner": FaultExpectation(
            "leave old lock-file bytes without an active OS lock",
            "PASS",
            False,
            "LOCK_ACQUIRED_WITHOUT_DELETING_STALE_FILE",
        ),
        "worker_exceeds_hard_ram": FaultExpectation(
            "inject a measured process-tree RSS sample above configured hard cap",
            "FAIL_RESOURCE_LIMIT",
            False,
            "NO_ARTIFACT_PUBLISHED",
        ),
        "worker_timeout": FaultExpectation(
            "sleep worker exceeds its strict wall-time deadline",
            "FAIL_TIMEOUT",
            False,
            "NO_ARTIFACT_PUBLISHED",
        ),
        "worker_exit_without_result": FaultExpectation(
            "worker exits abruptly before writing a result envelope",
            "FAIL_MODEL_RUNTIME",
            False,
            "NO_ARTIFACT_PUBLISHED",
        ),
        "restart_with_running_task": FaultExpectation(
            "persist an expired lease whose worker and owner processes are gone",
            "PASS",
            False,
            "PENDING_FOR_SAFE_RETRY",
        ),
        "missing_cache_index": FaultExpectation(
            "remove the derivative index", "INDEX_REBUILT", True, "MANIFEST_REDISCOVERED"
        ),
        "corrupt_cache_index": FaultExpectation(
            "truncate the SQLite index", "INDEX_REBUILT", True, "MANIFEST_REDISCOVERED"
        ),
        "complete_state_without_valid_payload": FaultExpectation(
            "persist COMPLETE state without the content-addressed payload",
            "RECOMPUTE_INVALID_COMPLETE",
            True,
            "TASK_REEXECUTED_AND_VALIDATED",
        ),
        "network_attempt_during_offline_execution": FaultExpectation(
            "worker attempts loopback socket connection under Python audit deny hook",
            "FAIL_MODEL_RUNTIME",
            False,
            "NETWORK_DENIED_NO_ARTIFACT",
        ),
    }
)

CANONICAL_FAULT_NAMES: Final[tuple[str, ...]] = tuple(FAULT_POLICY)
FAULT_EVIDENCE_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    "fault_name",
    "injection_point",
    "expected_failure_category",
    "observed_failure_category",
    "expected_artifact_accepted",
    "artifact_accepted",
    "expected_resume_behavior",
    "resume_behavior",
    "lock_released",
)


def canonical_fault_evidence_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact payload bound by a per-record fault evidence digest."""

    return {field: record[field] for field in FAULT_EVIDENCE_PAYLOAD_FIELDS}


def fault_evidence_sha256(record: Mapping[str, Any]) -> str:
    """Hash all expected and observed behavior, excluding only hash and status."""

    return sha256_canonical_json(canonical_fault_evidence_payload(record))
