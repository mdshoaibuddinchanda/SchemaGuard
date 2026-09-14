"""Cross-process lock used to serialize foundation-model/checkpoint work."""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock, Timeout


class ProcessLock:
    def __init__(self, path: str | Path, timeout: float = 30.0) -> None:
        self.path = str(path)
        self.timeout = timeout
        self._lock = FileLock(self.path)

    def __enter__(self) -> ProcessLock:
        self._lock.acquire(timeout=self.timeout)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._lock.release()


__all__ = ["ProcessLock", "Timeout"]
