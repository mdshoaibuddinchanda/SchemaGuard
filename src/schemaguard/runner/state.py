"""Atomic task-state persistence and lease-aware restart recovery."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..utils.io import atomic_write_json
from ..utils.process_lock import ProcessLock
from .contracts import (
    ResourceRecord,
    RunPlan,
    SchedulerState,
    TaskAttempt,
    TaskStateRecord,
    TaskStatus,
    TaskTransition,
    utc_now,
)

_ALLOWED: dict[str, set[str]] = {
    "PENDING": {"RUNNING", "CANCELLED"},
    "RUNNING": {"COMPLETE", "FAILED", "BLOCKED", "CANCELLED"},
    "COMPLETE": set(),
    "FAILED": set(),
    "BLOCKED": set(),
    "CANCELLED": set(),
}


class StateIntegrityError(RuntimeError):
    """Persisted scheduler state is corrupt or incompatible with the run plan."""


class InvalidStateTransition(RuntimeError):
    """The requested task-state transition is not authorized."""


class TaskStateStore:
    def __init__(self, root: str | Path, plan: RunPlan, *, lock_timeout: float = 120.0) -> None:
        self.root = Path(root)
        self.path = self.root / "runs" / f"{plan.plan_hash}.json"
        self.lock_path = self.root / "locks" / f"{plan.plan_hash}.state.lock"
        self.plan = plan
        self.lock_timeout = lock_timeout

    def initialize(self) -> SchedulerState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with ProcessLock(self.lock_path, timeout=self.lock_timeout):
            if self.path.is_file():
                state = self._read()
                expected = {task.task_id: task for task in self.plan.tasks}
                if state.plan_hash != self.plan.plan_hash or set(state.tasks) != set(expected):
                    raise StateIntegrityError("persisted task state does not match the run plan")
                if any(state.tasks[key].task != task for key, task in expected.items()):
                    raise StateIntegrityError("persisted task specifications differ from the plan")
                return state
            state = SchedulerState(
                schema_version=1,
                plan_hash=self.plan.plan_hash,
                tasks={task.task_id: TaskStateRecord(task=task) for task in self.plan.tasks},
            )
            self._write(state)
            return state

    def read(self) -> SchedulerState:
        return self._read()

    def transition(
        self,
        task_id: str,
        to_state: TaskStatus,
        *,
        reason: str | None = None,
        failure_category: str | None = None,
        resource: ResourceRecord | None = None,
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
        cache_hit: bool = False,
        retry_failed: bool = False,
    ) -> TaskStateRecord:
        with ProcessLock(self.lock_path, timeout=self.lock_timeout):
            state = self._read()
            if task_id not in state.tasks:
                raise StateIntegrityError("task ID is not part of the persisted plan")
            current = state.tasks[task_id]
            from_state = current.state
            retry_invalid_complete = (
                from_state == "COMPLETE"
                and to_state == "PENDING"
                and reason == "INVALID_CACHE_ARTIFACT"
            )
            retry_abandoned = (
                from_state == "RUNNING"
                and to_state == "PENDING"
                and reason == "ABANDONED_LEASE_RECOVERY"
            )
            retry_failed_state = from_state == "FAILED" and to_state == "PENDING" and retry_failed
            if to_state not in _ALLOWED[from_state] and not (
                retry_invalid_complete or retry_abandoned or retry_failed_state
            ):
                raise InvalidStateTransition(f"illegal task transition {from_state} -> {to_state}")

            now = utc_now()
            attempts = list(current.attempts)
            if to_state == "RUNNING":
                attempts.append(
                    TaskAttempt(
                        attempt_number=len(attempts) + 1,
                        state="RUNNING",
                        started_at=now,
                        heartbeat_at=now,
                        owner_pid=os.getpid(),
                    )
                )
            elif retry_abandoned and attempts:
                # The lost worker attempt is closed before an explicit fresh attempt.
                attempts[-1] = attempts[-1].model_copy(
                    update={
                        "state": "FAILED",
                        "ended_at": now,
                        "heartbeat_at": now,
                        "failure_category": "FAIL_MODEL_RUNTIME",
                        "reason": reason,
                    }
                )
            elif attempts and from_state == "RUNNING":
                attempts[-1] = attempts[-1].model_copy(
                    update={
                        "state": to_state,
                        "ended_at": now,
                        "heartbeat_at": now,
                        "resource": resource,
                        "failure_category": failure_category,
                        "reason": reason,
                    }
                )

            transition = TaskTransition(
                from_state=from_state,
                to_state=to_state,
                timestamp=now,
                reason=reason,
            )
            updated = current.model_copy(
                update={
                    "state": to_state,
                    "attempts": attempts,
                    "transitions": [*current.transitions, transition],
                    "artifact_path": artifact_path if to_state == "COMPLETE" else None,
                    "artifact_sha256": artifact_sha256 if to_state == "COMPLETE" else None,
                    "cache_hit": cache_hit if to_state == "COMPLETE" else False,
                    "failure_category": failure_category
                    if to_state in {"FAILED", "BLOCKED"}
                    else None,
                    "failure_reason": reason if to_state in {"FAILED", "BLOCKED"} else None,
                    "updated_at": now,
                }
            )
            state.tasks[task_id] = updated
            self._write(state)
            return updated

    def heartbeat(self, task_id: str) -> None:
        with ProcessLock(self.lock_path, timeout=self.lock_timeout):
            state = self._read()
            record = state.tasks[task_id]
            if record.state != "RUNNING" or not record.attempts:
                raise InvalidStateTransition("only running tasks can heartbeat")
            now = utc_now()
            attempts = list(record.attempts)
            attempts[-1] = attempts[-1].model_copy(update={"heartbeat_at": now})
            state.tasks[task_id] = record.model_copy(
                update={"attempts": attempts, "updated_at": now}
            )
            self._write(state)

    def set_worker_pid(self, task_id: str, worker_pid: int) -> None:
        with ProcessLock(self.lock_path, timeout=self.lock_timeout):
            state = self._read()
            record = state.tasks[task_id]
            if record.state != "RUNNING" or not record.attempts:
                raise InvalidStateTransition("worker PID can only be recorded for a running task")
            attempts = list(record.attempts)
            attempts[-1] = attempts[-1].model_copy(update={"worker_pid": worker_pid})
            state.tasks[task_id] = record.model_copy(update={"attempts": attempts})
            self._write(state)

    def recover_abandoned(self, lease_seconds: float) -> int:
        """Recover expired RUNNING leases only after their owner process has exited."""

        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("psutil is required for safe abandoned-task recovery") from exc
        with ProcessLock(self.lock_path, timeout=self.lock_timeout):
            state = self._read()
            now = datetime.now(UTC)
            recovered = 0
            for task_id, record in state.tasks.items():
                if record.state != "RUNNING" or not record.attempts:
                    continue
                attempt = record.attempts[-1]
                if (now - attempt.heartbeat_at).total_seconds() <= lease_seconds:
                    continue
                if attempt.owner_pid is not None and psutil.pid_exists(attempt.owner_pid):
                    continue
                if attempt.worker_pid is not None and psutil.pid_exists(attempt.worker_pid):
                    continue
                transition = TaskTransition(
                    from_state="RUNNING",
                    to_state="PENDING",
                    timestamp=now,
                    reason="ABANDONED_LEASE_RECOVERY",
                )
                closed_attempts = list(record.attempts)
                closed_attempts[-1] = attempt.model_copy(
                    update={
                        "state": "FAILED",
                        "ended_at": now,
                        "heartbeat_at": now,
                        "failure_category": "FAIL_MODEL_RUNTIME",
                        "reason": "ABANDONED_LEASE_RECOVERY",
                    }
                )
                state.tasks[task_id] = record.model_copy(
                    update={
                        "state": "PENDING",
                        "attempts": closed_attempts,
                        "transitions": [*record.transitions, transition],
                        "failure_category": None,
                        "failure_reason": None,
                        "updated_at": now,
                    }
                )
                recovered += 1
            if recovered:
                self._write(state)
            return recovered

    def _read(self) -> SchedulerState:
        try:
            return SchedulerState.model_validate(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise StateIntegrityError("scheduler state is missing, truncated, or invalid") from exc

    def _write(self, state: SchedulerState) -> None:
        validated = SchedulerState.model_validate(state.model_dump(mode="json"))
        atomic_write_json(self.path, validated.model_dump(mode="json"))


__all__ = ["InvalidStateTransition", "StateIntegrityError", "TaskStateStore"]
