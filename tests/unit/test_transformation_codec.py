from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schemaguard.transformations.base import restore_series_dtype
from schemaguard.transformations.codec import (
    decode_typed_key,
    decode_typed_value,
    encode_typed_value,
    typed_value_key,
)


@pytest.mark.parametrize(
    "value",
    [
        "a:b",
        "quoted'\"",
        r"slash\\value",
        "",
        "Δ-unicode",
        True,
        7,
        np.int16(-3),
        1.25,
        np.float32(2.5),
        pd.NA,
    ],
)
def test_typed_codec_roundtrips_json_safe_scalars(value) -> None:
    encoded = encode_typed_value(value)
    decoded = decode_typed_value(encoded)
    assert typed_value_key(value) == typed_value_key(decoded)
    assert decode_typed_key(typed_value_key(value)) is not None


def test_typed_codec_does_not_collide_bool_and_integer() -> None:
    assert typed_value_key(True) != typed_value_key(1)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "Int64"])
def test_integer_dtype_restoration_is_exact(dtype: str) -> None:
    source = (
        pd.Series([1, None, -2], dtype=dtype)
        if dtype == "Int64"
        else pd.Series([1, -2], dtype=dtype)
    )
    restored = restore_series_dtype(source.astype("Float64"), dtype)
    assert str(restored.dtype) == dtype
    assert restored.isna().tolist() == (
        [False, True, False] if dtype == "Int64" else [False, False]
    )


def test_integer_dtype_restoration_rejects_fractional_and_overflow_values() -> None:
    with pytest.raises(ValueError, match="safely restored"):
        restore_series_dtype(pd.Series([1.25]), "int64")
    with pytest.raises(ValueError, match="overflow"):
        restore_series_dtype(pd.Series([128.0]), "int8")
