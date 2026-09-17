"""Small spawn-safe worker entry point; model packages are never imported here."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def configure_worker_threads() -> None:
    """Apply the frozen native thread ceiling inside each spawned worker."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "2"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=2)
    except ImportError:
        # Environment limits remain authoritative when threadpoolctl is absent.
        pass


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def worker_entry(task: dict[str, Any], output_path: str, result_path: str) -> None:
    """Perform a deterministic synthetic task and return only file metadata."""

    configure_worker_threads()
    network_attempts = [0]

    def deny_network(event: str, _: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            network_attempts[0] += 1
            raise PermissionError("offline scheduler worker forbids network access")

    sys.addaudithook(deny_network)
    output = Path(output_path)
    result = Path(result_path)
    started = time.perf_counter()
    cpu_start = time.process_time()
    try:
        operation = task["operation"]
        if operation == "exit_without_result":
            os._exit(17)
        if operation == "partial_exit":
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"uncommitted-partial-payload")
            os._exit(18)
        if operation == "network_attempt":
            import socket

            socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        if operation == "sleep":
            time.sleep(float(task["delay_seconds"]))
        if operation == "raise_error":
            raise RuntimeError("injected worker exception")
        payload = json.dumps(
            {
                "schema_version": 1,
                "task_id": task["task_id"],
                "cache_key": hashlib.sha256(
                    json.dumps(
                        task["cache_identity"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "payload_token": task["payload_token"],
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(f".{output.name}.{os.getpid()}.part")
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, output)
        digest = hashlib.sha256(payload).hexdigest()
        worker_rss_mib: float | None = None
        try:
            import psutil

            worker_rss_mib = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except Exception:
            pass
        _atomic_json(
            result,
            {
                "schema_version": 1,
                "ok": True,
                "task_id": task["task_id"],
                "payload_sha256": digest,
                "payload_size_bytes": len(payload),
                "worker_pid": os.getpid(),
                "worker_rss_mib": worker_rss_mib,
                "cpu_seconds": max(0.0, time.process_time() - cpu_start),
                "worker_seconds": max(0.0, time.perf_counter() - started),
                "network_attempt_count": network_attempts[0],
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
        )
    except BaseException as exc:
        _atomic_json(
            result,
            {
                "schema_version": 1,
                "ok": False,
                "task_id": task.get("task_id", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "worker_pid": os.getpid(),
                "cpu_seconds": max(0.0, time.process_time() - cpu_start),
                "worker_seconds": max(0.0, time.perf_counter() - started),
                "network_attempt_count": network_attempts[0],
            },
        )


__all__ = ["configure_worker_threads", "worker_entry"]
