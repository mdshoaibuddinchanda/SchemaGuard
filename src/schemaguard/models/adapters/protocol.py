"""Typed public lifecycle implemented consistently by each frozen estimator adapter."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

from .contracts import AdapterResourceRecord, PredictionResult


@runtime_checkable
class ModelAdapter(Protocol):
    model_id: str
    seed: int
    device: str
    fitted: bool

    def fit(
        self,
        training_features: pd.DataFrame,
        training_targets: np.ndarray,
        *,
        fixture_id: str,
        fixture_sha256: str,
        split_identity: str,
        transformation_identity: str | None = None,
    ) -> Self: ...

    def predict_proba(
        self,
        features: pd.DataFrame,
        *,
        partition: str,
        fixture_id: str,
        fixture_sha256: str,
        split_identity: str,
        transformation_identity: str | None = None,
    ) -> PredictionResult: ...

    def capability(self) -> dict[str, object]: ...

    def resource_record(self) -> AdapterResourceRecord: ...

    def release(self) -> None: ...
