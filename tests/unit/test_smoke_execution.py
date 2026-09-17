from __future__ import annotations

from types import SimpleNamespace

from scripts.run_smoke_experiment import _condition_status


def _condition(status: str, category: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(status=status, failure_category=category)


def test_unapproved_network_attempt_invalidates_smoke_run() -> None:
    assert (
        _condition_status([_condition("PASS")], True, "PASS", 1)
        == "REPAIR_REQUIRED"
    )


def test_checkpoint_block_is_reported_as_blocked() -> None:
    assert (
        _condition_status([_condition("BLOCKED", "BLOCKED_CHECKPOINT")], True, "PASS", 0)
        == "BLOCKED"
    )


def test_capability_failure_is_partial_but_integrity_failure_requires_repair() -> None:
    assert (
        _condition_status([_condition("FAIL", "FAIL_CUDA_OOM")], True, "PASS", 0)
        == "PARTIAL_VALID"
    )
    assert (
        _condition_status([_condition("FAIL", "FAIL_PROBABILITY")], True, "PASS", 0)
        == "REPAIR_REQUIRED"
    )
