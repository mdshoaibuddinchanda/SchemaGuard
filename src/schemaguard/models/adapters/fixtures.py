"""Deterministic adapter fixtures with predictor-only inference partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...compatibility.fixtures import fixture_catalog
from ...constants import ROW_ID_COLUMN
from ...utils.hashing import hash_dataframe_logically, sha256_canonical_json


@dataclass(frozen=True)
class AdapterFixture:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    target: np.ndarray
    classes: tuple[int, ...]
    sha256: str
    capabilities: tuple[str, ...]


def _fixture(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: np.ndarray,
    capabilities: tuple[str, ...],
) -> AdapterFixture:
    train = train.rename(columns={"row_id": ROW_ID_COLUMN}).reset_index(drop=True)
    test = test.rename(columns={"row_id": ROW_ID_COLUMN}).reset_index(drop=True)
    y = np.asarray(target, dtype=np.int64).copy()
    if len(train) != len(y) or set(np.unique(y)) != set(range(int(y.max()) + 1)):
        raise ValueError("fixture target codes must align and be contiguous from zero")
    if min(np.bincount(y).tolist()) < 5:
        raise ValueError("every fixture class must contain at least five training examples")
    digest = sha256_canonical_json(
        {
            "fixture_id": name,
            "train_features": hash_dataframe_logically(train),
            "inference_features": hash_dataframe_logically(test),
            "training_targets": y.tolist(),
            "class_order": sorted(int(value) for value in np.unique(y)),
            "capabilities": capabilities,
        }
    )
    return AdapterFixture(
        name=name,
        train=train,
        test=test,
        target=y,
        classes=tuple(sorted(int(value) for value in np.unique(y))),
        sha256=digest,
        capabilities=capabilities,
    )


def adapter_fixture_catalog(seed: int = 1729) -> dict[str, AdapterFixture]:
    """Build six deterministic train/inference fixtures; no inference labels exist."""

    base = fixture_catalog(seed)
    binary = base["binary_numerical"]
    multi = base["multiclass_numerical"]
    missing = base["missing_values"]
    mixed = base["mixed_categorical"]
    unseen = base["unseen_category_inference"]

    categorical_rng = np.random.default_rng(seed + 31)
    categorical_target = np.arange(64, dtype=np.int64) % 2
    categorical_rng.shuffle(categorical_target)
    categorical = pd.DataFrame(
        {
            "row_id": [f"categorical-{index:04d}" for index in range(64)],
            "region": np.asarray(["north", "south", "west", "east"] * 16, dtype=object),
            "segment": np.asarray(["retail", "business"] * 32, dtype=object),
        }
    )
    missing_category_train = mixed.train.copy(deep=True)
    missing_category_test = mixed.test.copy(deep=True)
    missing_category_train.loc[3, "category"] = None
    missing_category_test.loc[2, "category"] = None

    definitions = {
        "binary_numerical": (
            binary.train,
            binary.test,
            binary.target,
            ("binary", "numerical"),
        ),
        "multiclass_numerical": (
            multi.train,
            multi.test,
            multi.target,
            ("multiclass", "numerical"),
        ),
        "missing_numerical": (
            missing.train,
            missing.test,
            missing.target,
            ("binary", "numerical", "missing"),
        ),
        "categorical_only": (
            categorical.iloc[:48].copy(),
            categorical.iloc[48:].copy(),
            categorical_target[:48],
            ("binary", "categorical"),
        ),
        "mixed_categorical": (
            mixed.train,
            mixed.test,
            mixed.target,
            ("binary", "numerical", "categorical"),
        ),
        "unseen_category": (
            unseen.train,
            unseen.test,
            unseen.target,
            ("binary", "numerical", "categorical", "unseen_category"),
        ),
        "missing_categorical": (
            missing_category_train,
            missing_category_test,
            mixed.target,
            ("binary", "numerical", "categorical", "missing"),
        ),
    }
    return {
        name: _fixture(name, train, test, target, capabilities)
        for name, (train, test, target, capabilities) in definitions.items()
    }
