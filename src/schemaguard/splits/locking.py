"""Split-specific locking built on the repository's cross-process lock."""

from __future__ import annotations

from pathlib import Path

from schemaguard.utils.process_lock import ProcessLock


def split_lock(root: str | Path, dataset_id: int, seed: int, timeout: float = 300.0) -> ProcessLock:
    path = Path(root) / "data" / "cache" / "locks" / f"split-{dataset_id}-{seed}.lock"
    return ProcessLock(path, timeout=timeout)


__all__ = ["ProcessLock", "split_lock"]
