"""Rebuildable SQLite lookup index; cache manifests remain authoritative."""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from ..utils.process_lock import ProcessLock
from .store import CacheArtifact, CacheStore


@dataclass(frozen=True)
class CacheIndexRecord:
    cache_key: str
    artifact_kind: str
    task_identity: str
    status: str
    payload_path: str
    manifest_path: str
    payload_sha256: str
    payload_size_bytes: int
    created_at: str
    validated_at: str
    source_implementation_sha256: str


class CacheIndex:
    """Maintain one transactional writer and portable cache-root-relative paths."""

    def __init__(self, path: str | Path, cache_root: str | Path) -> None:
        self.path = Path(path)
        self.cache_root = Path(cache_root).resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS artifacts (
                    cache_key TEXT PRIMARY KEY,
                    artifact_kind TEXT NOT NULL,
                    task_identity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_size_bytes INTEGER NOT NULL CHECK(payload_size_bytes >= 0),
                    created_at TEXT NOT NULL,
                    validated_at TEXT NOT NULL,
                    source_implementation_sha256 TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifact_kind ON artifacts(artifact_kind)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_identity ON artifacts(task_identity)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_status ON artifacts(status)")
        except Exception:
            connection.close()
            raise
        return connection

    def _connect_readonly(self) -> sqlite3.Connection:
        """Open the derivative index without creating tables or taking a writer lock."""

        if not self.path.is_file():
            raise FileNotFoundError(self.path.name)
        uri_path = quote(self.path.resolve().as_posix(), safe="/:\\")
        connection = sqlite3.connect(f"file:{uri_path}?mode=ro", timeout=30.0, uri=True)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def add(self, artifact: CacheArtifact) -> None:
        """Upsert one already validated manifest reference under the sole writer lock."""

        with ProcessLock(self.lock_path, timeout=30.0):
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(self._upsert_sql(), self._values(artifact))

    def lookup(
        self,
        *,
        cache_key: str | None = None,
        artifact_kind: str | None = None,
        task_identity: str | None = None,
        status: str | None = None,
    ) -> list[CacheIndexRecord]:
        """Return derivative index rows; callers must validate the manifest before reuse."""

        filters = {
            "cache_key": cache_key,
            "artifact_kind": artifact_kind,
            "task_identity": task_identity,
            "status": status,
        }
        where = [(key, value) for key, value in filters.items() if value is not None]
        query = "SELECT * FROM artifacts"
        parameters: list[str] = []
        if where:
            query += " WHERE " + " AND ".join(f"{key} = ?" for key, _ in where)
            parameters = [value for _, value in where]
        query += " ORDER BY cache_key"
        try:
            with closing(self._connect_readonly()) as connection:
                rows = connection.execute(query, parameters).fetchall()
        except (sqlite3.DatabaseError, FileNotFoundError):
            return []
        return [CacheIndexRecord(*row) for row in rows]

    def rebuild(self, store: CacheStore) -> int:
        """Reconstruct entries from self-validating manifests, never from the old index."""

        with ProcessLock(self.lock_path, timeout=30.0):
            entries = store.iter_validated()
            try:
                with closing(self._connect()) as connection, connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("DELETE FROM artifacts")
                    connection.executemany(self._upsert_sql(), [self._values(x) for x in entries])
            except sqlite3.DatabaseError:
                self._quarantine_corrupt_index()
                with closing(self._connect()) as connection, connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("DELETE FROM artifacts")
                    connection.executemany(self._upsert_sql(), [self._values(x) for x in entries])
        return len(entries)

    def _quarantine_corrupt_index(self) -> None:
        if self.path.is_file():
            backup = self.path.with_name(f"{self.path.name}.corrupt-{uuid.uuid4().hex}")
            os.replace(self.path, backup)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            sidecar.unlink(missing_ok=True)

    def _values(self, artifact: CacheArtifact) -> tuple[object, ...]:
        payload = artifact.payload_path.resolve().relative_to(self.cache_root).as_posix()
        manifest = artifact.manifest_path.resolve().relative_to(self.cache_root).as_posix()
        return (
            artifact.cache_key,
            artifact.artifact_kind,
            artifact.producing_task_identity,
            "COMPLETE",
            payload,
            manifest,
            artifact.payload_sha256,
            artifact.payload_size_bytes,
            artifact.created_at,
            datetime.now(UTC).isoformat(),
            artifact.source_implementation_sha256,
        )

    @staticmethod
    def _upsert_sql() -> str:
        return """INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
            artifact_kind=excluded.artifact_kind,
            task_identity=excluded.task_identity,
            status=excluded.status,
            payload_path=excluded.payload_path,
            manifest_path=excluded.manifest_path,
            payload_sha256=excluded.payload_sha256,
            payload_size_bytes=excluded.payload_size_bytes,
            created_at=excluded.created_at,
            validated_at=excluded.validated_at,
            source_implementation_sha256=excluded.source_implementation_sha256"""


__all__ = ["CacheIndex", "CacheIndexRecord"]
