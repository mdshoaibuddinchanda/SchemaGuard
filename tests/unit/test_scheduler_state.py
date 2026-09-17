from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemaguard.runner.state import InvalidStateTransition, TaskStateStore
from schemaguard.utils.io import atomic_write_json
from tests.unit.cache_scheduler_helpers import plan


def test_task_state_transitions_and_explicit_retry_policy(tmp_path: Path) -> None:
    run_plan = plan()
    state = TaskStateStore(tmp_path, run_plan)
    state.initialize()
    task_id = run_plan.tasks[0].task_id
    with pytest.raises(InvalidStateTransition):
        state.transition(task_id, "COMPLETE")
    state.transition(task_id, "RUNNING")
    with pytest.raises(ValidationError, match="complete task state requires"):
        state.transition(task_id, "COMPLETE")
    state.transition(task_id, "FAILED", reason="injected", failure_category="FAIL_MODEL_RUNTIME")
    with pytest.raises(InvalidStateTransition):
        state.transition(task_id, "PENDING")
    state.transition(task_id, "PENDING", reason="EXPLICIT_RETRY", retry_failed=True)
    assert state.read().tasks[task_id].state == "PENDING"


def test_abandoned_running_task_requires_expired_lease_and_dead_owner(tmp_path: Path) -> None:
    run_plan = plan()
    state = TaskStateStore(tmp_path, run_plan)
    state.initialize()
    task_id = run_plan.tasks[0].task_id
    state.transition(task_id, "RUNNING")
    snapshot = state.read()
    record = snapshot.tasks[task_id]
    old = datetime.now(UTC) - timedelta(seconds=120)
    attempts = list(record.attempts)
    attempts[-1] = attempts[-1].model_copy(
        update={"heartbeat_at": old, "owner_pid": 2_000_000_000, "worker_pid": None}
    )
    snapshot.tasks[task_id] = record.model_copy(update={"attempts": attempts})
    atomic_write_json(state.path, snapshot.model_dump(mode="json"))
    assert state.recover_abandoned(60) == 1
    recovered = state.read().tasks[task_id]
    assert recovered.state == "PENDING"
    assert recovered.attempts[-1].state == "FAILED"
    assert recovered.transitions[-1].reason == "ABANDONED_LEASE_RECOVERY"


def test_live_owner_is_not_recovered_and_truncated_state_fails_closed(tmp_path: Path) -> None:
    run_plan = plan()
    state = TaskStateStore(tmp_path, run_plan)
    state.initialize()
    task_id = run_plan.tasks[0].task_id
    state.transition(task_id, "RUNNING")
    assert state.recover_abandoned(0) == 0
    state.path.write_text("{truncated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="truncated"):
        state.read()
