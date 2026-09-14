"""Structured JSON Lines event logging."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


class EventLogger:
    """Append one structured event at a time to a JSONL log."""

    def __init__(self, path: str | Path, phase: str, dataset_id: str) -> None:
        self.path = Path(path)
        self.phase = phase
        self.dataset_id = dataset_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        stage: str,
        level: Literal["INFO", "WARNING", "ERROR"] = "INFO",
        event: str = "stage",
        *,
        cache_status: str | None = None,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "phase": self.phase,
            "stage": stage,
            "level": level,
            "event": event,
            "dataset_id": self.dataset_id,
        }
        if cache_status is not None:
            record["cache_status"] = cache_status
        if duration_ms is not None:
            record["duration_ms"] = round(duration_ms, 3)
        if details is not None:
            record["details"] = details
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
