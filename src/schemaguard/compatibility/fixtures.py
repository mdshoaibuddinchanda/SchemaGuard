"""Small deterministic fixtures that are independent of Phase 01 data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureFixture:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    target: np.ndarray
    classes: tuple[int, ...]
    hash: str
    capabilities: dict[str, Any]


def _hash_fixture(train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray) -> str:
    payload = {
        "train": train.to_dict(orient="split"),
        "test": test.to_dict(orient="split"),
        "target": target.tolist(),
        "dtypes": {name: str(dtype) for name, dtype in train.dtypes.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric_fixture(seed: int, train_rows: int, test_rows: int, classes: int) -> FeatureFixture:
    rng = np.random.default_rng(seed)
    total = train_rows + test_rows
    x = rng.normal(size=(total, 5))
    y = np.arange(total) % classes
    rng.shuffle(y)
    columns = [f"numeric_{i}" for i in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    frame.insert(0, "row_id", [f"fixture-{i:04d}" for i in range(total)])
    train = frame.iloc[:train_rows].reset_index(drop=True)
    test = frame.iloc[train_rows:].reset_index(drop=True)
    target = y[:train_rows].astype(int)
    return FeatureFixture(
        name="binary_numerical" if classes == 2 else "multiclass_numerical",
        train=train,
        test=test,
        target=target,
        classes=tuple(range(classes)),
        hash=_hash_fixture(train, test, target),
        capabilities={"numeric": True, "categorical": False, "missing": False},
    )


def binary_numerical(seed: int = 1729, train_rows: int = 48, test_rows: int = 16) -> FeatureFixture:
    return _numeric_fixture(seed, train_rows, test_rows, 2)


def multiclass_numerical(
    seed: int = 1729, train_rows: int = 60, test_rows: int = 18
) -> FeatureFixture:
    return _numeric_fixture(seed + 1, train_rows, test_rows, 3)


def mixed_categorical(seed: int = 1729) -> FeatureFixture:
    rng = np.random.default_rng(seed + 2)
    total = 64
    target = np.arange(total) % 2
    rng.shuffle(target)
    frame = pd.DataFrame(
        {
            "row_id": [f"mixed-{i:04d}" for i in range(total)],
            "numeric_a": rng.normal(size=total),
            "numeric_b": rng.normal(size=total),
            "category": np.array(["red", "green", "blue", "red"] * 16, dtype=object),
        }
    )
    train, test = frame.iloc[:48].copy(), frame.iloc[48:].copy()
    train["category"] = train["category"].astype(str)
    test["category"] = test["category"].astype(str)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    target = target[:48].astype(int)
    return FeatureFixture(
        name="mixed_categorical",
        train=train,
        test=test,
        target=target,
        classes=(0, 1),
        hash=_hash_fixture(train, test, target),
        capabilities={"numeric": True, "categorical": True, "missing": False},
    )


def missing_values(seed: int = 1729) -> FeatureFixture:
    fixture = binary_numerical(seed)
    train, test = fixture.train.copy(), fixture.test.copy()
    train.loc[[3, 17], "numeric_1"] = np.nan
    test.loc[[2], "numeric_3"] = np.nan
    return FeatureFixture(
        name="missing_values",
        train=train,
        test=test,
        target=fixture.target.copy(),
        classes=fixture.classes,
        hash=_hash_fixture(train, test, fixture.target),
        capabilities={"numeric": True, "categorical": False, "missing": True},
    )


def constant_column_invalid(seed: int = 1729) -> FeatureFixture:
    fixture = binary_numerical(seed)
    train, test = fixture.train.copy(), fixture.test.copy()
    train["constant"] = 1.0
    test["constant"] = 1.0
    return FeatureFixture(
        name="constant_column_invalid_input",
        train=train,
        test=test,
        target=fixture.target.copy(),
        classes=fixture.classes,
        hash=_hash_fixture(train, test, fixture.target),
        capabilities={"numeric": True, "categorical": False, "invalid": "constant_column"},
    )


def nonfinite_invalid(seed: int = 1729) -> FeatureFixture:
    fixture = binary_numerical(seed)
    train, test = fixture.train.copy(), fixture.test.copy()
    train.loc[4, "numeric_0"] = np.inf
    test.loc[4, "numeric_0"] = -np.inf
    return FeatureFixture(
        name="nonfinite_invalid_input",
        train=train,
        test=test,
        target=fixture.target.copy(),
        classes=fixture.classes,
        hash=_hash_fixture(train, test, fixture.target),
        capabilities={"numeric": True, "categorical": False, "invalid": "nonfinite"},
    )


def unseen_category_inference(seed: int = 1729) -> FeatureFixture:
    fixture = mixed_categorical(seed)
    train, test = fixture.train.copy(), fixture.test.copy()
    test.loc[0, "category"] = "unseen-category"
    return FeatureFixture(
        name="unseen_category_inference",
        train=train,
        test=test,
        target=fixture.target.copy(),
        classes=fixture.classes,
        hash=_hash_fixture(train, test, fixture.target),
        capabilities={
            "numeric": True,
            "categorical": True,
            "missing": False,
            "unseen_category": True,
        },
    )


def fixture_catalog(
    seed: int = 1729,
    *,
    binary_train_rows: int = 48,
    binary_test_rows: int = 16,
    multiclass_train_rows: int = 60,
    multiclass_test_rows: int = 18,
) -> dict[str, FeatureFixture]:
    return {
        "binary_numerical": binary_numerical(seed, binary_train_rows, binary_test_rows),
        "multiclass_numerical": multiclass_numerical(
            seed, multiclass_train_rows, multiclass_test_rows
        ),
        "mixed_categorical": mixed_categorical(seed),
        "missing_values": missing_values(seed),
        "constant_column_invalid_input": constant_column_invalid(seed),
        "nonfinite_invalid_input": nonfinite_invalid(seed),
        "unseen_category_inference": unseen_category_inference(seed),
    }
