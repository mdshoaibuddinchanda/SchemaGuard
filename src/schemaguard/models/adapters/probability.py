"""Adapter output validation and canonical class-order normalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..probability import ProbabilityValidationError, normalize_probability_matrix
from .contracts import AdapterFailure, FailureCategory


def align_adapter_probabilities(
    probabilities: Any,
    observed_classes: Any,
    canonical_classes: Any,
    *,
    row_count: int,
    sum_atol: float = 1.0e-6,
) -> tuple[np.ndarray, list[Any], float]:
    """Validate and reorder a model-native matrix without concealing invalid outputs."""

    try:
        matrix, classes, maximum_sum_error = normalize_probability_matrix(
            probabilities,
            observed_classes,
            canonical_classes,
            row_count=row_count,
            sum_atol=sum_atol,
        )
    except (ProbabilityValidationError, TypeError, ValueError) as exc:
        message = str(exc)
        category = (
            FailureCategory.FAIL_CLASS_ORDER
            if any(word in message.lower() for word in ("class", "label", "column"))
            else FailureCategory.FAIL_PROBABILITY
        )
        raise AdapterFailure(category, message) from exc
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (row_count, len(canonical_classes)):
        raise AdapterFailure(
            FailureCategory.FAIL_PROBABILITY,
            "probability matrix shape does not match requested rows/classes",
        )
    return matrix, list(classes), maximum_sum_error
