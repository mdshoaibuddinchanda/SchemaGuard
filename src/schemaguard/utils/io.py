"""Atomic filesystem operations used by the data foundation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
from pydantic import BaseModel


def _atomic_replace(destination: Path, writer: Any, suffix: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=suffix,
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes via a destination-local temporary file and atomic replace."""
    destination = Path(path)
    _atomic_replace(destination, lambda handle: handle.write(data), ".part")


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write canonical JSON with sorted keys, two-space indentation, and a final newline."""
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, rendered)


def atomic_write_parquet(
    path: str | Path,
    frame: pd.DataFrame,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    """Write a pandas frame to Parquet atomically without serializing its index."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part.parquet",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(
            temporary,
            index=False,
            compression=compression,
            compression_level=compression_level,
        )
        with temporary.open("rb+") as written:
            written.seek(0, os.SEEK_END)
            written.flush()
            os.fsync(written.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json_validated[ModelT: BaseModel](
    path: str | Path, model: type[ModelT] | None = None
) -> Any:
    """Read JSON and optionally validate it against a Pydantic model."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return model.model_validate(value) if model is not None else value
