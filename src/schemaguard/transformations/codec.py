"""Structured JSON-safe encoding for categorical values and parameters."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pandas as pd


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def encode_typed_value(value: Any) -> dict[str, Any]:
    """Encode one scalar without relying on repr parsing or evaluation."""

    if _is_missing(value):
        return {"type": "missing", "value": None}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, np.integer):
        return {"type": type(value).__name__, "value": int(value)}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, np.floating):
        numeric = float(value)
        if np.isfinite(numeric):
            return {"type": type(value).__name__, "value": numeric}
        return {"type": type(value).__name__, "value": str(numeric)}
    if isinstance(value, float):
        if np.isfinite(value):
            return {"type": "float", "value": value}
        return {"type": "float", "value": str(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, pd.Timestamp):
        return {"type": "pandas_timestamp", "value": value.isoformat()}
    raise TypeError(f"unsupported categorical scalar type: {type(value)!r}")


def decode_typed_value(encoded: dict[str, Any]) -> Any:
    """Decode a value encoded by :func:`encode_typed_value`."""

    value_type = encoded.get("type")
    value = cast(Any, encoded.get("value"))
    if value_type == "missing":
        return np.nan
    if value_type == "str":
        if not isinstance(value, str):
            raise ValueError("encoded string category must contain a string value")
        return value
    if value_type == "bool":
        if not isinstance(value, bool):
            raise ValueError("encoded boolean category must contain a boolean value")
        return value
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if isinstance(value_type, str) and value_type.startswith("int"):
        return np.dtype(value_type).type(int(value))
    if isinstance(value_type, str) and value_type.startswith("uint"):
        return np.dtype(value_type).type(int(value))
    if isinstance(value_type, str) and value_type.startswith("float"):
        return np.dtype(value_type).type(float(value))
    if value_type == "pandas_timestamp":
        return pd.Timestamp(value)
    raise ValueError(f"unsupported encoded scalar type: {value_type!r}")


def typed_value_key(value: Any) -> str:
    """Return a collision-free canonical key for a typed scalar."""

    return json.dumps(
        encode_typed_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def decode_typed_key(key: str) -> Any:
    try:
        encoded = json.loads(key)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid typed category key") from exc
    if not isinstance(encoded, dict):
        raise ValueError("typed category key must decode to an object")
    return decode_typed_value(encoded)


__all__ = ["decode_typed_key", "decode_typed_value", "encode_typed_value", "typed_value_key"]
