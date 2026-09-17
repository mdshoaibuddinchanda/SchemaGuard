"""Per-identity cross-process locking for Windows and POSIX runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from filelock import Timeout

from ..utils.process_lock import ProcessLock


class CacheLockTimeout(TimeoutError):
    """Raised when another process retains an identity lock past the deadline."""


class CacheKeyLock:
    def __init__(self, root: str | Path, cache_key: str, timeout: float = 120.0) -> None:
        if len(cache_key) != 64 or any(char not in "0123456789abcdef" for char in cache_key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        if timeout <= 0:
            raise ValueError("lock timeout must be positive")
        self.path = Path(root) / ".locks" / f"{cache_key}.lock"
        self.timeout = timeout
        self._lock = ProcessLock(self.path, timeout=timeout)
        self._acquired = False

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.__enter__()
        except Timeout as exc:
            raise CacheLockTimeout(f"timed out acquiring cache lock {self.path.name}") from exc
        self._acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._acquired:
            self._lock.__exit__(exc_type, exc, traceback)
            self._acquired = False
